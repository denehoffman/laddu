use std::mem::size_of;

use laddu_expr::{BinaryOp, UnaryOp};
use nalgebra::DMatrix;
use num::{
    complex::{Complex, Complex32, Complex64},
    traits::Float,
};

use super::{RuntimeError, RuntimeResult};

/// Contiguous row-major storage with a fixed, checked row width.
///
/// Cache layouts use this type instead of open-coded offset arithmetic.  The
/// backing vector remains private so every row/range access goes through the
/// overflow-checked helpers below.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct FlatRows<T> {
    width: usize,
    values: Vec<T>,
}

impl<T> FlatRows<T> {
    pub(crate) fn try_with_capacity(width: usize, rows: usize) -> RuntimeResult<Self> {
        if width == 0 {
            return Err(RuntimeError::InvalidShape {
                index: 0,
                message: "flat row width must be nonzero".into(),
            });
        }
        let capacity = rows
            .checked_mul(width)
            .ok_or_else(|| RuntimeError::InvalidShape {
                index: rows,
                message: format!("flat row allocation overflowed for width {width}"),
            })?;
        let element_size = size_of::<T>();
        if element_size != 0 {
            let bytes =
                capacity
                    .checked_mul(element_size)
                    .ok_or_else(|| RuntimeError::InvalidShape {
                        index: rows,
                        message: format!("flat row byte allocation overflowed for width {width}"),
                    })?;
            if bytes > isize::MAX as usize {
                return Err(RuntimeError::InvalidShape {
                    index: rows,
                    message: format!(
                        "flat row allocation exceeds addressable capacity: {bytes} bytes"
                    ),
                });
            }
        }
        Ok(Self {
            width,
            values: Vec::with_capacity(capacity),
        })
    }

    #[cfg_attr(not(feature = "jit"), allow(dead_code))]
    pub(crate) fn width(&self) -> usize {
        self.width
    }

    #[allow(dead_code)]
    pub(crate) fn len(&self) -> usize {
        self.values.len()
    }

    pub(crate) fn capacity(&self) -> usize {
        self.values.capacity()
    }

    #[cfg_attr(not(feature = "jit"), allow(dead_code))]
    pub(crate) fn as_ptr(&self) -> *const T {
        self.values.as_ptr()
    }

    pub(crate) fn push_row(&mut self, row: impl IntoIterator<Item = T>) -> RuntimeResult<()> {
        let start = self.values.len();
        self.values.extend(row);
        let actual = self.values.len() - start;
        if actual != self.width {
            self.values.truncate(start);
            return Err(RuntimeError::InvalidShape {
                index: start / self.width,
                message: format!("flat row has len {actual}, expected {}", self.width),
            });
        }
        Ok(())
    }

    pub(crate) fn row(&self, row: usize) -> RuntimeResult<&[T]> {
        let start = row
            .checked_mul(self.width)
            .ok_or_else(|| RuntimeError::InvalidShape {
                index: row,
                message: "flat row offset overflowed".into(),
            })?;
        let end = start
            .checked_add(self.width)
            .ok_or_else(|| RuntimeError::InvalidShape {
                index: row,
                message: "flat row end overflowed".into(),
            })?;
        self.values
            .get(start..end)
            .ok_or_else(|| RuntimeError::InvalidShape {
                index: row,
                message: format!(
                    "flat row {row} out of bounds for len {}",
                    self.values.len() / self.width
                ),
            })
    }

    #[allow(dead_code)]
    pub(crate) fn range(&self, start: usize, end: usize) -> RuntimeResult<&[T]> {
        if start > end {
            return Err(RuntimeError::InvalidShape {
                index: start,
                message: format!("flat row range {start}..{end} is reversed"),
            });
        }
        let start_value =
            start
                .checked_mul(self.width)
                .ok_or_else(|| RuntimeError::InvalidShape {
                    index: start,
                    message: "flat row range offset overflowed".into(),
                })?;
        let end_value = end
            .checked_mul(self.width)
            .ok_or_else(|| RuntimeError::InvalidShape {
                index: end,
                message: "flat row range end overflowed".into(),
            })?;
        self.values
            .get(start_value..end_value)
            .ok_or_else(|| RuntimeError::InvalidShape {
                index: start,
                message: format!("flat row range {start}..{end} out of bounds"),
            })
    }
}

/// Borrowed, typed view over one evaluated value.
pub(super) enum ValueRef<'a> {
    Scalar(&'a Complex64),
    Vector(&'a [Complex64]),
    Matrix {
        rows: usize,
        cols: usize,
        values: &'a [Complex64],
    },
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum Value {
    Scalar(Complex64),
    Vector(Vec<Complex64>),
    Matrix {
        rows: usize,
        cols: usize,
        values: Vec<Complex64>,
    },
}

impl Value {
    pub(super) fn as_ref(&self) -> ValueRef<'_> {
        match self {
            Self::Scalar(value) => ValueRef::Scalar(value),
            Self::Vector(values) => ValueRef::Vector(values),
            Self::Matrix { rows, cols, values } => ValueRef::Matrix {
                rows: *rows,
                cols: *cols,
                values,
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum F32Value {
    Scalar(Complex32),
    Vector(Vec<Complex32>),
    Matrix {
        rows: usize,
        cols: usize,
        values: Vec<Complex32>,
    },
}

impl F32Value {
    pub(super) fn from_value(value: Value) -> Self {
        match value {
            Value::Scalar(value) => Self::Scalar(Complex32::new(value.re as f32, value.im as f32)),
            Value::Vector(values) => Self::Vector(
                values
                    .into_iter()
                    .map(|value| Complex32::new(value.re as f32, value.im as f32))
                    .collect(),
            ),
            Value::Matrix { rows, cols, values } => Self::Matrix {
                rows,
                cols,
                values: values
                    .into_iter()
                    .map(|value| Complex32::new(value.re as f32, value.im as f32))
                    .collect(),
            },
        }
    }

    pub(super) fn kind(&self) -> &'static str {
        match self {
            Self::Scalar(_) => "scalar",
            Self::Vector(_) => "vector",
            Self::Matrix { .. } => "matrix",
        }
    }
}

pub(super) fn f32_scalar_at(
    values: &[F32Value],
    id: laddu_kernel::ir::KernelValueId,
) -> RuntimeResult<Complex32> {
    match &values[id.index()] {
        F32Value::Scalar(value) => Ok(*value),
        value => Err(RuntimeError::TypeMismatch {
            index: id.index(),
            expected: "scalar",
            actual: value.kind(),
        }),
    }
}

pub(super) fn f32_vector_at(
    values: &[F32Value],
    id: laddu_kernel::ir::KernelValueId,
) -> RuntimeResult<&[Complex32]> {
    match &values[id.index()] {
        F32Value::Vector(values) => Ok(values),
        value => Err(RuntimeError::TypeMismatch {
            index: id.index(),
            expected: "vector",
            actual: value.kind(),
        }),
    }
}

pub(super) fn f32_matrix_at(
    values: &[F32Value],
    id: laddu_kernel::ir::KernelValueId,
) -> RuntimeResult<(usize, usize, &[Complex32])> {
    match &values[id.index()] {
        F32Value::Matrix { rows, cols, values } => Ok((*rows, *cols, values)),
        value => Err(RuntimeError::TypeMismatch {
            index: id.index(),
            expected: "matrix",
            actual: value.kind(),
        }),
    }
}

pub(super) fn scalar_at(values: &[Value], index: usize) -> RuntimeResult<Complex64> {
    match values[index].as_ref() {
        ValueRef::Scalar(value) => Ok(*value),
        ValueRef::Vector(_) => Err(RuntimeError::TypeMismatch {
            index,
            expected: "scalar",
            actual: "vector",
        }),
        ValueRef::Matrix { .. } => Err(RuntimeError::TypeMismatch {
            index,
            expected: "scalar",
            actual: "matrix",
        }),
    }
}

pub(super) fn vector_at(values: &[Value], index: usize) -> RuntimeResult<&[Complex64]> {
    match values[index].as_ref() {
        ValueRef::Vector(value) => Ok(value),
        ValueRef::Scalar(_) => Err(RuntimeError::TypeMismatch {
            index,
            expected: "vector",
            actual: "scalar",
        }),
        ValueRef::Matrix { .. } => Err(RuntimeError::TypeMismatch {
            index,
            expected: "vector",
            actual: "matrix",
        }),
    }
}

pub(super) fn matrix_at(
    values: &[Value],
    index: usize,
) -> RuntimeResult<(usize, usize, &[Complex64])> {
    match values[index].as_ref() {
        ValueRef::Matrix { rows, cols, values } => Ok((rows, cols, values)),
        ValueRef::Scalar(_) => Err(RuntimeError::TypeMismatch {
            index,
            expected: "matrix",
            actual: "scalar",
        }),
        ValueRef::Vector(_) => Err(RuntimeError::TypeMismatch {
            index,
            expected: "matrix",
            actual: "vector",
        }),
    }
}

pub(super) fn scalar_at_optional(
    values: &[Option<Value>],
    index: usize,
) -> RuntimeResult<Complex64> {
    values.get(index).and_then(Option::as_ref).map_or_else(
        || {
            Err(RuntimeError::InvalidShape {
                index,
                message: "required cache prerequisite was not evaluated".into(),
            })
        },
        |value| match value.as_ref() {
            ValueRef::Scalar(value) => Ok(*value),
            ValueRef::Vector(_) => Err(RuntimeError::TypeMismatch {
                index,
                expected: "scalar",
                actual: "vector",
            }),
            ValueRef::Matrix { .. } => Err(RuntimeError::TypeMismatch {
                index,
                expected: "scalar",
                actual: "matrix",
            }),
        },
    )
}

pub(super) fn vector_at_optional(
    values: &[Option<Value>],
    index: usize,
) -> RuntimeResult<&[Complex64]> {
    values.get(index).and_then(Option::as_ref).map_or_else(
        || {
            Err(RuntimeError::InvalidShape {
                index,
                message: "required cache prerequisite was not evaluated".into(),
            })
        },
        |value| match value.as_ref() {
            ValueRef::Vector(value) => Ok(value),
            ValueRef::Scalar(_) => Err(RuntimeError::TypeMismatch {
                index,
                expected: "vector",
                actual: "scalar",
            }),
            ValueRef::Matrix { .. } => Err(RuntimeError::TypeMismatch {
                index,
                expected: "vector",
                actual: "matrix",
            }),
        },
    )
}

pub(super) fn matrix_at_optional(
    values: &[Option<Value>],
    index: usize,
) -> RuntimeResult<(usize, usize, &[Complex64])> {
    values.get(index).and_then(Option::as_ref).map_or_else(
        || {
            Err(RuntimeError::InvalidShape {
                index,
                message: "required cache prerequisite was not evaluated".into(),
            })
        },
        |value| match value.as_ref() {
            ValueRef::Matrix { rows, cols, values } => Ok((rows, cols, values)),
            ValueRef::Scalar(_) => Err(RuntimeError::TypeMismatch {
                index,
                expected: "matrix",
                actual: "scalar",
            }),
            ValueRef::Vector(_) => Err(RuntimeError::TypeMismatch {
                index,
                expected: "matrix",
                actual: "vector",
            }),
        },
    )
}

pub(super) fn matrix_values_row_major(matrix: &DMatrix<Complex64>) -> Vec<Complex64> {
    let mut values = Vec::with_capacity(matrix.nrows() * matrix.ncols());
    for row in 0..matrix.nrows() {
        for col in 0..matrix.ncols() {
            values.push(matrix[(row, col)]);
        }
    }
    values
}

pub(super) fn matrix_values_row_major_f32(matrix: &DMatrix<Complex32>) -> Vec<Complex32> {
    let mut values = Vec::with_capacity(matrix.nrows() * matrix.ncols());
    for row in 0..matrix.nrows() {
        for col in 0..matrix.ncols() {
            values.push(matrix[(row, col)]);
        }
    }
    values
}

pub(super) fn eval_unary<T: Float>(op: UnaryOp, input: Complex<T>) -> Complex<T> {
    match op {
        UnaryOp::Neg => -input,
        UnaryOp::Real => Complex::from(input.re),
        UnaryOp::Imag => Complex::from(input.im),
        UnaryOp::Conj => input.conj(),
        UnaryOp::NormSqr => Complex::from(input.norm_sqr()),
        UnaryOp::Sqrt => input.sqrt(),
        UnaryOp::Exp => input.exp(),
        UnaryOp::Sin => input.sin(),
        UnaryOp::Cos => input.cos(),
        UnaryOp::Log => input.ln(),
        UnaryOp::PowI(power) => input.powi(power),
    }
}

pub(super) fn eval_binary<T: Float>(op: BinaryOp, lhs: Complex<T>, rhs: Complex<T>) -> Complex<T> {
    match op {
        BinaryOp::Add => lhs + rhs,
        BinaryOp::Sub => lhs - rhs,
        BinaryOp::Mul => lhs * rhs,
        BinaryOp::Div => lhs / rhs,
        BinaryOp::Atan2 => Complex::from(lhs.re.atan2(rhs.re)),
    }
}

impl Value {
    pub(super) fn kind(&self) -> &'static str {
        match self {
            Self::Scalar(_) => "scalar",
            Self::Vector(_) => "vector",
            Self::Matrix { .. } => "matrix",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::FlatRows;

    #[test]
    fn flat_rows_reject_zero_width_and_capacity_overflow() {
        assert!(FlatRows::<u8>::try_with_capacity(0, 1).is_err());
        assert!(FlatRows::<u8>::try_with_capacity(usize::MAX, 2).is_err());
        assert!(FlatRows::<[u8; 2]>::try_with_capacity(usize::MAX / 2 + 1, 1).is_err());
    }

    #[test]
    fn flat_rows_allow_large_zero_sized_capacity() {
        assert!(FlatRows::<()>::try_with_capacity(1, usize::MAX).is_ok());
    }

    #[test]
    fn flat_rows_check_last_row_and_ranges() {
        let mut rows = FlatRows::try_with_capacity(2, 2).unwrap();
        rows.push_row([1, 2]).unwrap();
        rows.push_row([3, 4]).unwrap();
        assert_eq!(rows.row(1).unwrap(), &[3, 4]);
        assert_eq!(rows.range(0, 2).unwrap(), &[1, 2, 3, 4]);
        assert!(rows.row(2).is_err());
        assert!(rows.range(2, 1).is_err());
    }
}
