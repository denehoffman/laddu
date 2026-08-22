use laddu_expr::{BinaryOp, UnaryOp};
use nalgebra::DMatrix;
use num::{
    complex::{Complex, Complex32, Complex64},
    traits::Float,
};

use super::{RuntimeError, RuntimeResult};

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
    match &values[index] {
        Value::Scalar(value) => Ok(*value),
        value => Err(RuntimeError::TypeMismatch {
            index,
            expected: "scalar",
            actual: value.kind(),
        }),
    }
}

pub(super) fn vector_at(values: &[Value], index: usize) -> RuntimeResult<&[Complex64]> {
    match &values[index] {
        Value::Vector(value) => Ok(value),
        value => Err(RuntimeError::TypeMismatch {
            index,
            expected: "vector",
            actual: value.kind(),
        }),
    }
}

pub(super) fn matrix_at(
    values: &[Value],
    index: usize,
) -> RuntimeResult<(usize, usize, &[Complex64])> {
    match &values[index] {
        Value::Matrix { rows, cols, values } => Ok((*rows, *cols, values)),
        value => Err(RuntimeError::TypeMismatch {
            index,
            expected: "matrix",
            actual: value.kind(),
        }),
    }
}

pub(super) fn scalar_at_optional(
    values: &[Option<Value>],
    index: usize,
) -> RuntimeResult<Complex64> {
    match values.get(index).and_then(Option::as_ref) {
        Some(Value::Scalar(value)) => Ok(*value),
        Some(value) => Err(RuntimeError::TypeMismatch {
            index,
            expected: "scalar",
            actual: value.kind(),
        }),
        None => Err(RuntimeError::InvalidShape {
            index,
            message: "required cache prerequisite was not evaluated".into(),
        }),
    }
}

pub(super) fn vector_at_optional(
    values: &[Option<Value>],
    index: usize,
) -> RuntimeResult<&[Complex64]> {
    match values.get(index).and_then(Option::as_ref) {
        Some(Value::Vector(value)) => Ok(value),
        Some(value) => Err(RuntimeError::TypeMismatch {
            index,
            expected: "vector",
            actual: value.kind(),
        }),
        None => Err(RuntimeError::InvalidShape {
            index,
            message: "required cache prerequisite was not evaluated".into(),
        }),
    }
}

pub(super) fn matrix_at_optional(
    values: &[Option<Value>],
    index: usize,
) -> RuntimeResult<(usize, usize, &[Complex64])> {
    match values.get(index).and_then(Option::as_ref) {
        Some(Value::Matrix { rows, cols, values }) => Ok((*rows, *cols, values)),
        Some(value) => Err(RuntimeError::TypeMismatch {
            index,
            expected: "matrix",
            actual: value.kind(),
        }),
        None => Err(RuntimeError::InvalidShape {
            index,
            message: "required cache prerequisite was not evaluated".into(),
        }),
    }
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
