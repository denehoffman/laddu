use laddu_expr::{BinaryOp, UnaryOp, parameters::ParamId};
use num::complex::Complex64;
use thiserror::Error;

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum KernelIrError {
    #[error("kernel IR contains no values")]
    Empty,
    #[error("kernel root value {root} is out of bounds for {len} values")]
    RootOutOfBounds { root: usize, len: usize },
    #[error("kernel value {value} references non-prior value {operand}")]
    InvalidOperand { value: usize, operand: usize },
    #[error("kernel value {value} has no operands for {operation}")]
    EmptyOperands {
        value: usize,
        operation: &'static str,
    },
    #[error("kernel value {value} has kind {actual:?}, but its instruction produces {expected:?}")]
    KindMismatch {
        value: usize,
        expected: KernelScalarKind,
        actual: KernelScalarKind,
    },
    #[error("kernel value {value} has class {actual:?}, but its dependencies require {expected:?}")]
    ClassMismatch {
        value: usize,
        expected: KernelValueClass,
        actual: KernelValueClass,
    },
    #[error("kernel value {value} constructs a complex value from a non-real component")]
    InvalidComplexComponent { value: usize },
}

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
pub enum KernelScalarKind {
    Real,
    Complex,
}

impl KernelScalarKind {
    fn combine(self, rhs: Self) -> Self {
        if self == Self::Complex || rhs == Self::Complex {
            Self::Complex
        } else {
            Self::Real
        }
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
            Self::Unary { input, .. } => vec![*input],
            Self::Binary { lhs, rhs, .. } => vec![*lhs, *rhs],
            Self::Add(terms) | Self::Mul(terms) => terms.clone(),
            Self::Complex { re, im } => vec![*re, *im],
            Self::SolveRow { rhs, .. } => rhs.clone(),
        }
    }

    fn expected_kind(
        &self,
        values: &[KernelValue],
        value: usize,
    ) -> Result<Option<KernelScalarKind>, KernelIrError> {
        Ok(Some(match self {
            Self::Cached(_) => return Ok(None),
            Self::RealConstant(_) | Self::Parameter(_) => KernelScalarKind::Real,
            Self::ComplexConstant(value) => {
                if value.im == 0.0 {
                    KernelScalarKind::Real
                } else {
                    KernelScalarKind::Complex
                }
            }
            Self::SolveRow { .. } => KernelScalarKind::Complex,
            Self::Unary { op, input } => match op {
                UnaryOp::Real | UnaryOp::Imag | UnaryOp::NormSqr => KernelScalarKind::Real,
                UnaryOp::Neg
                | UnaryOp::Conj
                | UnaryOp::Sqrt
                | UnaryOp::Exp
                | UnaryOp::Sin
                | UnaryOp::Cos
                | UnaryOp::Log
                | UnaryOp::PowI(_) => values[input.index()].kind,
            },
            Self::Binary { op, lhs, rhs } => match op {
                BinaryOp::Atan2 => KernelScalarKind::Real,
                BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div => {
                    values[lhs.index()].kind.combine(values[rhs.index()].kind)
                }
            },
            Self::Add(terms) | Self::Mul(terms) => {
                let mut terms = terms.iter();
                let first = terms.next().ok_or(KernelIrError::EmptyOperands {
                    value,
                    operation: match self {
                        Self::Add(_) => "addition",
                        Self::Mul(_) => "multiplication",
                        _ => unreachable!(),
                    },
                })?;
                terms.fold(values[first.index()].kind, |kind, term| {
                    kind.combine(values[term.index()].kind)
                })
            }
            Self::Complex { re, im } => {
                if values[re.index()].kind != KernelScalarKind::Real
                    || values[im.index()].kind != KernelScalarKind::Real
                {
                    return Err(KernelIrError::InvalidComplexComponent { value });
                }
                KernelScalarKind::Complex
            }
        }))
    }

    fn expected_class(&self, values: &[KernelValue]) -> KernelValueClass {
        match self {
            Self::Cached(_) | Self::SolveRow { .. } => KernelValueClass::Event,
            Self::RealConstant(_) | Self::ComplexConstant(_) | Self::Parameter(_) => {
                KernelValueClass::Invariant
            }
            _ => {
                if self
                    .operands()
                    .iter()
                    .any(|operand| values[operand.index()].class == KernelValueClass::Event)
                {
                    KernelValueClass::Event
                } else {
                    KernelValueClass::Invariant
                }
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct KernelValue {
    pub kind: KernelScalarKind,
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
    fn validates_typed_topological_scalar_ir() {
        let ir = ScalarKernelIr::new(
            vec![
                KernelValue {
                    kind: KernelScalarKind::Real,
                    class: KernelValueClass::Invariant,
                    instruction: KernelInstruction::RealConstant(2.0),
                },
                KernelValue {
                    kind: KernelScalarKind::Real,
                    class: KernelValueClass::Event,
                    instruction: KernelInstruction::Cached(0),
                },
                KernelValue {
                    kind: KernelScalarKind::Real,
                    class: KernelValueClass::Event,
                    instruction: KernelInstruction::Binary {
                        op: BinaryOp::Mul,
                        lhs: KernelValueId::from_index(0),
                        rhs: KernelValueId::from_index(1),
                    },
                },
            ],
            KernelValueId::from_index(2),
        )
        .unwrap();

        assert_eq!(ir.root().index(), 2);
    }

    #[test]
    fn rejects_forward_references() {
        let error = ScalarKernelIr::new(
            vec![
                KernelValue {
                    kind: KernelScalarKind::Real,
                    class: KernelValueClass::Invariant,
                    instruction: KernelInstruction::Unary {
                        op: UnaryOp::Neg,
                        input: KernelValueId::from_index(1),
                    },
                },
                KernelValue {
                    kind: KernelScalarKind::Real,
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
    fn rejects_incorrect_kind_and_dependency_class() {
        let kind_error = ScalarKernelIr::new(
            vec![KernelValue {
                kind: KernelScalarKind::Complex,
                class: KernelValueClass::Invariant,
                instruction: KernelInstruction::RealConstant(1.0),
            }],
            KernelValueId::from_index(0),
        )
        .unwrap_err();
        assert!(matches!(kind_error, KernelIrError::KindMismatch { .. }));

        let class_error = ScalarKernelIr::new(
            vec![KernelValue {
                kind: KernelScalarKind::Real,
                class: KernelValueClass::Event,
                instruction: KernelInstruction::RealConstant(1.0),
            }],
            KernelValueId::from_index(0),
        )
        .unwrap_err();
        assert!(matches!(class_error, KernelIrError::ClassMismatch { .. }));
    }
}
