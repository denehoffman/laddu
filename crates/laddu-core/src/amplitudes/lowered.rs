use num::complex::Complex64;

/// Execution-only program kinds derived from optimized IR.
///
/// These variants classify lowered runtimes by output contract rather than by planning logic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LoweredProgramKind {
    Value,
    Gradient,
    ValueGradient,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LoweredUnaryOp {
    Neg,
    Real,
    Imag,
    Conj,
    NormSqr,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LoweredBinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum LoweredInstruction {
    Constant {
        dst: usize,
        value: Complex64,
    },
    LoadAmplitude {
        dst: usize,
        amplitude_index: usize,
    },
    Unary {
        dst: usize,
        input: usize,
        op: LoweredUnaryOp,
    },
    Binary {
        dst: usize,
        left: usize,
        right: usize,
        op: LoweredBinaryOp,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LoweredRuntimeLayout {
    scratch_slots: usize,
    root_slot: usize,
}

impl LoweredRuntimeLayout {
    pub(crate) fn new(scratch_slots: usize, root_slot: usize) -> Self {
        debug_assert!(scratch_slots == 0 || root_slot < scratch_slots);
        Self {
            scratch_slots,
            root_slot,
        }
    }

    pub(crate) fn scratch_slots(&self) -> usize {
        self.scratch_slots
    }

    pub(crate) fn root_slot(&self) -> usize {
        self.root_slot
    }
}

/// A compact execution-only program derived from optimized IR.
///
/// Invariants:
/// - Instructions are ordered for direct forward execution.
/// - `layout.root_slot()` identifies the final result slot for the program kind.
/// - The program carries no semantic/planning metadata beyond what execution requires.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LoweredProgram {
    kind: LoweredProgramKind,
    instructions: Vec<LoweredInstruction>,
    layout: LoweredRuntimeLayout,
}

impl LoweredProgram {
    pub(crate) fn new(
        kind: LoweredProgramKind,
        instructions: Vec<LoweredInstruction>,
        layout: LoweredRuntimeLayout,
    ) -> Self {
        Self {
            kind,
            instructions,
            layout,
        }
    }

    pub(crate) fn kind(&self) -> LoweredProgramKind {
        self.kind
    }

    pub(crate) fn instructions(&self) -> &[LoweredInstruction] {
        &self.instructions
    }

    pub(crate) fn layout(&self) -> &LoweredRuntimeLayout {
        &self.layout
    }

    pub(crate) fn scratch_slots(&self) -> usize {
        self.layout.scratch_slots()
    }

    pub(crate) fn root_slot(&self) -> usize {
        self.layout.root_slot()
    }
}

/// Collection of lowered execution programs derived from the same specialized IR instance.
///
/// The value/gradient/value+gradient programs are siblings which must all correspond to the same
/// expression tree, active-mask specialization, and lowering assumptions.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct LoweredExpressionRuntime {
    value_program: Option<LoweredProgram>,
    gradient_program: Option<LoweredProgram>,
    value_gradient_program: Option<LoweredProgram>,
}

impl LoweredExpressionRuntime {
    pub(crate) fn new(
        value_program: Option<LoweredProgram>,
        gradient_program: Option<LoweredProgram>,
        value_gradient_program: Option<LoweredProgram>,
    ) -> Self {
        Self {
            value_program,
            gradient_program,
            value_gradient_program,
        }
    }

    pub(crate) fn value_program(&self) -> Option<&LoweredProgram> {
        self.value_program.as_ref()
    }

    pub(crate) fn gradient_program(&self) -> Option<&LoweredProgram> {
        self.gradient_program.as_ref()
    }

    pub(crate) fn value_gradient_program(&self) -> Option<&LoweredProgram> {
        self.value_gradient_program.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        LoweredBinaryOp, LoweredExpressionRuntime, LoweredInstruction, LoweredProgram,
        LoweredProgramKind, LoweredRuntimeLayout,
    };
    use num::complex::Complex64;

    #[test]
    fn lowered_program_exposes_layout_and_kind() {
        let program = LoweredProgram::new(
            LoweredProgramKind::Value,
            vec![
                LoweredInstruction::LoadAmplitude {
                    dst: 0,
                    amplitude_index: 1,
                },
                LoweredInstruction::Constant {
                    dst: 1,
                    value: Complex64::new(2.0, 0.0),
                },
                LoweredInstruction::Binary {
                    dst: 2,
                    left: 0,
                    right: 1,
                    op: LoweredBinaryOp::Mul,
                },
            ],
            LoweredRuntimeLayout::new(3, 2),
        );

        assert_eq!(program.kind(), LoweredProgramKind::Value);
        assert_eq!(program.scratch_slots(), 3);
        assert_eq!(program.root_slot(), 2);
        assert_eq!(program.instructions().len(), 3);
    }

    #[test]
    fn lowered_runtime_can_hold_independent_program_kinds() {
        let value_program = LoweredProgram::new(
            LoweredProgramKind::Value,
            vec![LoweredInstruction::Constant {
                dst: 0,
                value: Complex64::new(1.0, 0.0),
            }],
            LoweredRuntimeLayout::new(1, 0),
        );
        let gradient_program = LoweredProgram::new(
            LoweredProgramKind::Gradient,
            vec![LoweredInstruction::Constant {
                dst: 0,
                value: Complex64::new(0.0, 0.0),
            }],
            LoweredRuntimeLayout::new(1, 0),
        );

        let runtime = LoweredExpressionRuntime::new(
            Some(value_program.clone()),
            Some(gradient_program.clone()),
            None,
        );

        assert_eq!(runtime.value_program(), Some(&value_program));
        assert_eq!(runtime.gradient_program(), Some(&gradient_program));
        assert!(runtime.value_gradient_program().is_none());
    }
}
