use super::*;

impl KernelInstruction {
    /// Returns the stable instruction name used in diagnostics and support reporting.
    pub fn diagnostic_name(&self) -> &'static str {
        match self {
            Self::Cached(_) => "Cached",
            Self::RealConstant(_) => "RealConstant",
            Self::ComplexConstant(_) => "ComplexConstant",
            Self::Parameter(_) => "Parameter",
            Self::Unary { .. } => "Unary",
            Self::Binary { .. } => "Binary",
            Self::Add(_) => "Add",
            Self::Mul(_) => "Mul",
            Self::Complex { .. } => "Complex",
            Self::Vector(_) => "Vector",
            Self::Matrix { .. } => "Matrix",
            Self::Component { .. } => "Component",
            Self::MatrixElement { .. } => "MatrixElement",
            Self::MatMul { .. } => "MatMul",
            Self::MatVec { .. } => "MatVec",
            Self::Dot { .. } => "Dot",
            Self::Solve { .. } => "Solve",
            Self::SolveRow { .. } => "SolveRow",
            Self::SolveRowAdjointElement { .. } => "SolveRowAdjointElement",
        }
    }

    /// Returns the backend-independent event-dependence rule for this instruction.
    pub fn event_dependence(&self) -> KernelEventDependence {
        match self {
            Self::Cached(_) | Self::SolveRow { .. } | Self::SolveRowAdjointElement { .. } => {
                KernelEventDependence::Event
            }
            Self::RealConstant(_) | Self::ComplexConstant(_) | Self::Parameter(_) => {
                KernelEventDependence::Invariant
            }
            _ => KernelEventDependence::Operands,
        }
    }

    /// Returns the direct input value identifiers.
    pub fn operands(&self) -> Vec<KernelValueId> {
        let mut operands = Vec::new();
        self.visit_operands(|operand| operands.push(operand));
        operands
    }

    pub(super) fn visit_operands(&self, mut visit: impl FnMut(KernelValueId)) {
        match self {
            Self::Cached(_)
            | Self::RealConstant(_)
            | Self::ComplexConstant(_)
            | Self::Parameter(_) => {}
            Self::Unary { input, .. }
            | Self::Component { input, .. }
            | Self::MatrixElement { input, .. }
            | Self::SolveRowAdjointElement { adjoint: input, .. } => visit(*input),
            Self::Binary { lhs, rhs, .. } | Self::MatMul { lhs, rhs } | Self::Dot { lhs, rhs } => {
                visit(*lhs);
                visit(*rhs);
            }
            Self::MatVec { matrix, vector }
            | Self::Solve {
                matrix,
                rhs: vector,
            } => {
                visit(*matrix);
                visit(*vector);
            }
            Self::Add(values)
            | Self::Mul(values)
            | Self::Vector(values)
            | Self::SolveRow { rhs: values, .. } => values.iter().copied().for_each(visit),
            Self::Complex { re, im } => {
                visit(*re);
                visit(*im);
            }
            Self::Matrix { elements, .. } => elements.iter().copied().for_each(visit),
        }
    }

    pub(super) fn validate_operand_order(&self, value: usize) -> Result<(), KernelIrError> {
        let mut invalid_operand = None;
        self.visit_operands(|operand| {
            if invalid_operand.is_none() && operand.index() >= value {
                invalid_operand = Some(operand.index());
            }
        });
        invalid_operand.map_or(Ok(()), |operand| {
            Err(KernelIrError::InvalidOperand { value, operand })
        })
    }
}
