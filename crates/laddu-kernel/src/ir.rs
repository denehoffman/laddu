pub use crate::KernelIrError;
use laddu_expr::{BinaryOp, UnaryOp, parameters::ParamId};
use num::complex::Complex64;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct KernelValueId(usize);

impl KernelValueId {
    pub fn from_index(index: usize) -> Self {
        Self(index)
    }

    pub fn index(self) -> usize {
        self.0
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum KernelValueKind {
    Real,
    Complex,
    Vector { len: usize },
    Matrix { rows: usize, cols: usize },
}

impl KernelValueKind {
    pub fn width(self) -> usize {
        match self {
            Self::Real | Self::Complex => 1,
            Self::Vector { len } => len,
            Self::Matrix { rows, cols } => rows * cols,
        }
    }

    fn scalar_combine(self, rhs: Self) -> Option<Self> {
        match (self, rhs) {
            (Self::Real, Self::Real) => Some(Self::Real),
            (Self::Real | Self::Complex, Self::Real | Self::Complex) => Some(Self::Complex),
            _ => None,
        }
    }

    fn is_scalar(self) -> bool {
        matches!(self, Self::Real | Self::Complex)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum KernelValueClass {
    Invariant,
    Event,
}

#[derive(Clone, Debug)]
pub enum KernelInstruction {
    Cached(usize),
    RealConstant(f64),
    ComplexConstant(Complex64),
    Parameter(ParamId),
    Unary {
        op: UnaryOp,
        input: KernelValueId,
    },
    Binary {
        op: BinaryOp,
        lhs: KernelValueId,
        rhs: KernelValueId,
    },
    Add(Vec<KernelValueId>),
    Mul(Vec<KernelValueId>),
    Complex {
        re: KernelValueId,
        im: KernelValueId,
    },
    Vector(Vec<KernelValueId>),
    Matrix {
        rows: usize,
        cols: usize,
        elements: Vec<KernelValueId>,
    },
    Component {
        input: KernelValueId,
        index: usize,
    },
    MatrixElement {
        input: KernelValueId,
        row: usize,
        col: usize,
    },
    MatMul {
        lhs: KernelValueId,
        rhs: KernelValueId,
    },
    MatVec {
        matrix: KernelValueId,
        vector: KernelValueId,
    },
    Dot {
        lhs: KernelValueId,
        rhs: KernelValueId,
    },
    Solve {
        matrix: KernelValueId,
        rhs: KernelValueId,
    },
    SolveRow {
        row_slot: usize,
        rhs: Vec<KernelValueId>,
    },
}

impl KernelInstruction {
    pub fn operands(&self) -> Vec<KernelValueId> {
        match self {
            Self::Cached(_)
            | Self::RealConstant(_)
            | Self::ComplexConstant(_)
            | Self::Parameter(_) => Vec::new(),
            Self::Unary { input, .. }
            | Self::Component { input, .. }
            | Self::MatrixElement { input, .. } => vec![*input],
            Self::Binary { lhs, rhs, .. } | Self::MatMul { lhs, rhs } | Self::Dot { lhs, rhs } => {
                vec![*lhs, *rhs]
            }
            Self::MatVec { matrix, vector }
            | Self::Solve {
                matrix,
                rhs: vector,
            } => vec![*matrix, *vector],
            Self::Add(values)
            | Self::Mul(values)
            | Self::Vector(values)
            | Self::SolveRow { rhs: values, .. } => values.clone(),
            Self::Complex { re, im } => vec![*re, *im],
            Self::Matrix { elements, .. } => elements.clone(),
        }
    }

    fn shape_error(
        value: usize,
        operation: &'static str,
        message: impl Into<String>,
    ) -> KernelIrError {
        KernelIrError::InvalidShape {
            value,
            operation,
            message: message.into(),
        }
    }

    fn expected_kind(
        &self,
        values: &[KernelValue],
        value: usize,
    ) -> Result<Option<KernelValueKind>, KernelIrError> {
        let kind = |id: KernelValueId| values[id.index()].kind;
        Ok(Some(match self {
            Self::Cached(_) => return Ok(None),
            Self::RealConstant(_) | Self::Parameter(_) => KernelValueKind::Real,
            Self::ComplexConstant(value) if value.im == 0.0 => KernelValueKind::Real,
            Self::ComplexConstant(_) => KernelValueKind::Complex,
            Self::Unary { op, input } => {
                if !kind(*input).is_scalar() {
                    return Err(Self::shape_error(
                        value,
                        "unary operation",
                        "input is not scalar",
                    ));
                }
                match op {
                    UnaryOp::Real | UnaryOp::Imag | UnaryOp::NormSqr => KernelValueKind::Real,
                    _ => kind(*input),
                }
            }
            Self::Binary { op, lhs, rhs } => {
                if *op == BinaryOp::Atan2 {
                    if kind(*lhs) != KernelValueKind::Real || kind(*rhs) != KernelValueKind::Real {
                        return Err(Self::shape_error(
                            value,
                            "atan2",
                            "both inputs must be real",
                        ));
                    }
                    KernelValueKind::Real
                } else {
                    kind(*lhs).scalar_combine(kind(*rhs)).ok_or_else(|| {
                        Self::shape_error(value, "binary operation", "both inputs must be scalar")
                    })?
                }
            }
            Self::Add(terms) | Self::Mul(terms) => {
                let operation = if matches!(self, Self::Add(_)) {
                    "addition"
                } else {
                    "multiplication"
                };
                let mut terms = terms.iter();
                let first = terms
                    .next()
                    .ok_or(KernelIrError::EmptyOperands { value, operation })?;
                terms.try_fold(kind(*first), |acc, term| {
                    acc.scalar_combine(kind(*term)).ok_or_else(|| {
                        Self::shape_error(value, operation, "all inputs must be scalar")
                    })
                })?
            }
            Self::Complex { re, im } => {
                if kind(*re) != KernelValueKind::Real || kind(*im) != KernelValueKind::Real {
                    return Err(Self::shape_error(
                        value,
                        "complex construction",
                        "components must be real",
                    ));
                }
                KernelValueKind::Complex
            }
            Self::Vector(elements) => {
                if elements.iter().any(|element| !kind(*element).is_scalar()) {
                    return Err(Self::shape_error(
                        value,
                        "vector construction",
                        "elements must be scalar",
                    ));
                }
                KernelValueKind::Vector {
                    len: elements.len(),
                }
            }
            Self::Matrix {
                rows,
                cols,
                elements,
            } => {
                if elements.len() != rows * cols
                    || elements.iter().any(|element| !kind(*element).is_scalar())
                {
                    return Err(Self::shape_error(
                        value,
                        "matrix construction",
                        format!("expected {} scalar elements", rows * cols),
                    ));
                }
                KernelValueKind::Matrix {
                    rows: *rows,
                    cols: *cols,
                }
            }
            Self::Component { input, index } => match kind(*input) {
                KernelValueKind::Vector { len } if *index < len => KernelValueKind::Complex,
                actual => {
                    return Err(Self::shape_error(
                        value,
                        "component",
                        format!("index {index} is invalid for {actual:?}"),
                    ));
                }
            },
            Self::MatrixElement { input, row, col } => match kind(*input) {
                KernelValueKind::Matrix { rows, cols } if *row < rows && *col < cols => {
                    KernelValueKind::Complex
                }
                actual => {
                    return Err(Self::shape_error(
                        value,
                        "matrix element",
                        format!("index ({row}, {col}) is invalid for {actual:?}"),
                    ));
                }
            },
            Self::MatMul { lhs, rhs } => match (kind(*lhs), kind(*rhs)) {
                (
                    KernelValueKind::Matrix { rows, cols: inner },
                    KernelValueKind::Matrix {
                        rows: rhs_rows,
                        cols,
                    },
                ) if inner == rhs_rows => KernelValueKind::Matrix { rows, cols },
                shapes => {
                    return Err(Self::shape_error(
                        value,
                        "matrix multiplication",
                        format!("incompatible operands {shapes:?}"),
                    ));
                }
            },
            Self::MatVec { matrix, vector } => match (kind(*matrix), kind(*vector)) {
                (KernelValueKind::Matrix { rows, cols }, KernelValueKind::Vector { len })
                    if cols == len =>
                {
                    KernelValueKind::Vector { len: rows }
                }
                shapes => {
                    return Err(Self::shape_error(
                        value,
                        "matrix-vector multiplication",
                        format!("incompatible operands {shapes:?}"),
                    ));
                }
            },
            Self::Dot { lhs, rhs } => match (kind(*lhs), kind(*rhs)) {
                (KernelValueKind::Vector { len }, KernelValueKind::Vector { len: rhs_len })
                    if len == rhs_len =>
                {
                    KernelValueKind::Complex
                }
                shapes => {
                    return Err(Self::shape_error(
                        value,
                        "dot product",
                        format!("incompatible operands {shapes:?}"),
                    ));
                }
            },
            Self::Solve { matrix, rhs } => match (kind(*matrix), kind(*rhs)) {
                (KernelValueKind::Matrix { rows, cols }, KernelValueKind::Vector { len })
                    if rows == cols && rows == len =>
                {
                    KernelValueKind::Vector { len }
                }
                shapes => {
                    return Err(Self::shape_error(
                        value,
                        "linear solve",
                        format!("incompatible operands {shapes:?}"),
                    ));
                }
            },
            Self::SolveRow { rhs, .. } => {
                if rhs.is_empty() || rhs.iter().any(|entry| !kind(*entry).is_scalar()) {
                    return Err(Self::shape_error(
                        value,
                        "specialized solve row",
                        "right-hand side must contain scalars",
                    ));
                }
                KernelValueKind::Complex
            }
        }))
    }

    fn expected_class(&self, values: &[KernelValue]) -> KernelValueClass {
        match self {
            Self::Cached(_) | Self::SolveRow { .. } => KernelValueClass::Event,
            Self::RealConstant(_) | Self::ComplexConstant(_) | Self::Parameter(_) => {
                KernelValueClass::Invariant
            }
            _ if self
                .operands()
                .iter()
                .any(|operand| values[operand.index()].class == KernelValueClass::Event) =>
            {
                KernelValueClass::Event
            }
            _ => KernelValueClass::Invariant,
        }
    }
}

#[derive(Clone, Debug)]
pub struct KernelValue {
    pub kind: KernelValueKind,
    pub class: KernelValueClass,
    pub instruction: KernelInstruction,
}

#[derive(Clone, Debug)]
pub struct ScalarKernelIr {
    values: Vec<KernelValue>,
    root: KernelValueId,
}

impl ScalarKernelIr {
    pub fn new(values: Vec<KernelValue>, root: KernelValueId) -> Result<Self, KernelIrError> {
        let ir = Self { values, root };
        ir.validate()?;
        if !ir.values[ir.root.index()].kind.is_scalar() {
            return Err(KernelIrError::InvalidShape {
                value: ir.root.index(),
                operation: "kernel root",
                message: "root must be scalar".into(),
            });
        }
        Ok(ir)
    }

    pub fn validate(&self) -> Result<(), KernelIrError> {
        if self.values.is_empty() {
            return Err(KernelIrError::Empty);
        }
        if self.root.index() >= self.values.len() {
            return Err(KernelIrError::RootOutOfBounds {
                root: self.root.index(),
                len: self.values.len(),
            });
        }
        for (index, value) in self.values.iter().enumerate() {
            for operand in value.instruction.operands() {
                if operand.index() >= index {
                    return Err(KernelIrError::InvalidOperand {
                        value: index,
                        operand: operand.index(),
                    });
                }
            }
            if let Some(expected) = value.instruction.expected_kind(&self.values, index)?
                && value.kind != expected
            {
                return Err(KernelIrError::KindMismatch {
                    value: index,
                    expected,
                    actual: value.kind,
                });
            }
            let expected = value.instruction.expected_class(&self.values);
            if value.class != expected {
                return Err(KernelIrError::ClassMismatch {
                    value: index,
                    expected,
                    actual: value.class,
                });
            }
        }
        Ok(())
    }

    pub fn values(&self) -> &[KernelValue] {
        &self.values
    }
    pub fn root(&self) -> KernelValueId {
        self.root
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_aggregate_operations() {
        let values = vec![
            KernelValue {
                kind: KernelValueKind::Complex,
                class: KernelValueClass::Invariant,
                instruction: KernelInstruction::ComplexConstant(Complex64::new(1.0, 2.0)),
            },
            KernelValue {
                kind: KernelValueKind::Vector { len: 1 },
                class: KernelValueClass::Invariant,
                instruction: KernelInstruction::Vector(vec![KernelValueId::from_index(0)]),
            },
            KernelValue {
                kind: KernelValueKind::Complex,
                class: KernelValueClass::Invariant,
                instruction: KernelInstruction::Dot {
                    lhs: KernelValueId::from_index(1),
                    rhs: KernelValueId::from_index(1),
                },
            },
        ];
        ScalarKernelIr::new(values, KernelValueId::from_index(2)).unwrap();
    }

    #[test]
    fn rejects_forward_references() {
        let error = ScalarKernelIr::new(
            vec![
                KernelValue {
                    kind: KernelValueKind::Real,
                    class: KernelValueClass::Invariant,
                    instruction: KernelInstruction::Unary {
                        op: UnaryOp::Neg,
                        input: KernelValueId::from_index(1),
                    },
                },
                KernelValue {
                    kind: KernelValueKind::Real,
                    class: KernelValueClass::Invariant,
                    instruction: KernelInstruction::RealConstant(1.0),
                },
            ],
            KernelValueId::from_index(0),
        )
        .unwrap_err();
        assert_eq!(
            error,
            KernelIrError::InvalidOperand {
                value: 0,
                operand: 1
            }
        );
    }

    #[test]
    fn rejects_invalid_matrix_shapes() {
        let error = ScalarKernelIr::new(
            vec![
                KernelValue {
                    kind: KernelValueKind::Real,
                    class: KernelValueClass::Invariant,
                    instruction: KernelInstruction::RealConstant(1.0),
                },
                KernelValue {
                    kind: KernelValueKind::Matrix { rows: 2, cols: 2 },
                    class: KernelValueClass::Invariant,
                    instruction: KernelInstruction::Matrix {
                        rows: 2,
                        cols: 2,
                        elements: vec![KernelValueId::from_index(0)],
                    },
                },
            ],
            KernelValueId::from_index(0),
        )
        .unwrap_err();
        assert!(matches!(error, KernelIrError::InvalidShape { .. }));
    }
}
