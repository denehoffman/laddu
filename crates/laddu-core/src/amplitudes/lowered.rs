use super::ir::{ExpressionIR, IrBinaryOp, IrNode, IrUnaryOp};
use nalgebra::DVector;
use num::complex::Complex64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LoweringError {
    EmptyIr,
}

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

    pub(crate) fn evaluate_into(
        &self,
        amplitude_values: &[Complex64],
        scratch: &mut [Complex64],
    ) -> Complex64 {
        debug_assert!(scratch.len() >= self.scratch_slots());
        for instruction in &self.instructions {
            match *instruction {
                LoweredInstruction::Constant { dst, value } => {
                    scratch[dst] = value;
                }
                LoweredInstruction::LoadAmplitude {
                    dst,
                    amplitude_index,
                } => {
                    scratch[dst] = amplitude_values
                        .get(amplitude_index)
                        .copied()
                        .unwrap_or_default();
                }
                LoweredInstruction::Unary { dst, input, op } => {
                    let value = scratch[input];
                    scratch[dst] = match op {
                        LoweredUnaryOp::Neg => -value,
                        LoweredUnaryOp::Real => Complex64::new(value.re, 0.0),
                        LoweredUnaryOp::Imag => Complex64::new(value.im, 0.0),
                        LoweredUnaryOp::Conj => value.conj(),
                        LoweredUnaryOp::NormSqr => Complex64::new(value.norm_sqr(), 0.0),
                    };
                }
                LoweredInstruction::Binary {
                    dst,
                    left,
                    right,
                    op,
                } => {
                    let left_value = scratch[left];
                    let right_value = scratch[right];
                    scratch[dst] = match op {
                        LoweredBinaryOp::Add => left_value + right_value,
                        LoweredBinaryOp::Sub => left_value - right_value,
                        LoweredBinaryOp::Mul => left_value * right_value,
                        LoweredBinaryOp::Div => left_value / right_value,
                    };
                }
            }
        }
        scratch[self.root_slot()]
    }

    pub(crate) fn evaluate_gradient_into(
        &self,
        amplitude_values: &[Complex64],
        amplitude_gradients: &[DVector<Complex64>],
        value_scratch: &mut [Complex64],
        gradient_scratch: &mut [DVector<Complex64>],
    ) -> DVector<Complex64> {
        debug_assert_eq!(self.kind, LoweredProgramKind::Gradient);
        debug_assert!(value_scratch.len() >= self.scratch_slots());
        debug_assert!(gradient_scratch.len() >= self.scratch_slots());

        for instruction in &self.instructions {
            match *instruction {
                LoweredInstruction::Constant { dst, value } => {
                    value_scratch[dst] = value;
                    gradient_scratch[dst].fill(Complex64::ZERO);
                }
                LoweredInstruction::LoadAmplitude {
                    dst,
                    amplitude_index,
                } => {
                    value_scratch[dst] = amplitude_values
                        .get(amplitude_index)
                        .copied()
                        .unwrap_or_default();
                    if let Some(source) = amplitude_gradients.get(amplitude_index) {
                        gradient_scratch[dst].clone_from(source);
                    } else {
                        gradient_scratch[dst].fill(Complex64::ZERO);
                    }
                }
                LoweredInstruction::Unary { dst, input, op } => {
                    let value = value_scratch[input];
                    value_scratch[dst] = match op {
                        LoweredUnaryOp::Neg => -value,
                        LoweredUnaryOp::Real => Complex64::new(value.re, 0.0),
                        LoweredUnaryOp::Imag => Complex64::new(value.im, 0.0),
                        LoweredUnaryOp::Conj => value.conj(),
                        LoweredUnaryOp::NormSqr => Complex64::new(value.norm_sqr(), 0.0),
                    };
                    let input_grad = gradient_scratch[input].clone();
                    let dst_grad = &mut gradient_scratch[dst];
                    match op {
                        LoweredUnaryOp::Neg => {
                            for (dst_item, input_item) in dst_grad.iter_mut().zip(input_grad.iter())
                            {
                                *dst_item = -*input_item;
                            }
                        }
                        LoweredUnaryOp::Real => {
                            for (dst_item, input_item) in dst_grad.iter_mut().zip(input_grad.iter())
                            {
                                *dst_item = Complex64::new(input_item.re, 0.0);
                            }
                        }
                        LoweredUnaryOp::Imag => {
                            for (dst_item, input_item) in dst_grad.iter_mut().zip(input_grad.iter())
                            {
                                *dst_item = Complex64::new(input_item.im, 0.0);
                            }
                        }
                        LoweredUnaryOp::Conj => {
                            for (dst_item, input_item) in dst_grad.iter_mut().zip(input_grad.iter())
                            {
                                *dst_item = input_item.conj();
                            }
                        }
                        LoweredUnaryOp::NormSqr => {
                            let conj_input = value.conj();
                            for (dst_item, input_item) in dst_grad.iter_mut().zip(input_grad.iter())
                            {
                                *dst_item =
                                    Complex64::new(2.0 * (*input_item * conj_input).re, 0.0);
                            }
                        }
                    }
                }
                LoweredInstruction::Binary {
                    dst,
                    left,
                    right,
                    op,
                } => {
                    let left_value = value_scratch[left];
                    let right_value = value_scratch[right];
                    value_scratch[dst] = match op {
                        LoweredBinaryOp::Add => left_value + right_value,
                        LoweredBinaryOp::Sub => left_value - right_value,
                        LoweredBinaryOp::Mul => left_value * right_value,
                        LoweredBinaryOp::Div => left_value / right_value,
                    };
                    let left_grad = gradient_scratch[left].clone();
                    let right_grad = gradient_scratch[right].clone();
                    let dst_grad = &mut gradient_scratch[dst];
                    match op {
                        LoweredBinaryOp::Add => {
                            for ((dst_item, left_item), right_item) in dst_grad
                                .iter_mut()
                                .zip(left_grad.iter())
                                .zip(right_grad.iter())
                            {
                                *dst_item = *left_item + *right_item;
                            }
                        }
                        LoweredBinaryOp::Sub => {
                            for ((dst_item, left_item), right_item) in dst_grad
                                .iter_mut()
                                .zip(left_grad.iter())
                                .zip(right_grad.iter())
                            {
                                *dst_item = *left_item - *right_item;
                            }
                        }
                        LoweredBinaryOp::Mul => {
                            for ((dst_item, left_item), right_item) in dst_grad
                                .iter_mut()
                                .zip(left_grad.iter())
                                .zip(right_grad.iter())
                            {
                                *dst_item = *left_item * right_value + *right_item * left_value;
                            }
                        }
                        LoweredBinaryOp::Div => {
                            let denom = right_value * right_value;
                            for ((dst_item, left_item), right_item) in dst_grad
                                .iter_mut()
                                .zip(left_grad.iter())
                                .zip(right_grad.iter())
                            {
                                *dst_item =
                                    (*left_item * right_value - *right_item * left_value) / denom;
                            }
                        }
                    }
                }
            }
        }

        gradient_scratch[self.root_slot()].clone()
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

    pub(crate) fn from_ir_value_only(ir: &ExpressionIR) -> Result<Self, LoweringError> {
        let value_program = Some(LoweredProgram::from_ir_value_only(ir)?);
        Ok(Self::new(value_program, None, None))
    }

    pub(crate) fn from_ir_value_gradient(ir: &ExpressionIR) -> Result<Self, LoweringError> {
        let value_program = Some(LoweredProgram::from_ir_value_only(ir)?);
        let gradient_program = Some(LoweredProgram::from_ir_gradient_only(ir)?);
        Ok(Self::new(value_program, gradient_program, None))
    }
}

impl LoweredProgram {
    pub(crate) fn from_ir_value_only(ir: &ExpressionIR) -> Result<Self, LoweringError> {
        if ir.node_count() == 0 {
            return Err(LoweringError::EmptyIr);
        }

        let instructions = ir
            .nodes()
            .iter()
            .enumerate()
            .map(|(dst, node)| match *node {
                IrNode::Constant(value) => LoweredInstruction::Constant { dst, value },
                IrNode::Amp(amplitude_index) => LoweredInstruction::LoadAmplitude {
                    dst,
                    amplitude_index,
                },
                IrNode::Unary { op, input } => LoweredInstruction::Unary {
                    dst,
                    input,
                    op: match op {
                        IrUnaryOp::Neg => LoweredUnaryOp::Neg,
                        IrUnaryOp::Real => LoweredUnaryOp::Real,
                        IrUnaryOp::Imag => LoweredUnaryOp::Imag,
                        IrUnaryOp::Conj => LoweredUnaryOp::Conj,
                        IrUnaryOp::NormSqr => LoweredUnaryOp::NormSqr,
                    },
                },
                IrNode::Binary { op, left, right } => LoweredInstruction::Binary {
                    dst,
                    left,
                    right,
                    op: match op {
                        IrBinaryOp::Add => LoweredBinaryOp::Add,
                        IrBinaryOp::Sub => LoweredBinaryOp::Sub,
                        IrBinaryOp::Mul => LoweredBinaryOp::Mul,
                        IrBinaryOp::Div => LoweredBinaryOp::Div,
                    },
                },
            })
            .collect();

        Ok(Self::new(
            LoweredProgramKind::Value,
            instructions,
            LoweredRuntimeLayout::new(ir.node_count(), ir.root()),
        ))
    }

    pub(crate) fn from_ir_gradient_only(ir: &ExpressionIR) -> Result<Self, LoweringError> {
        if ir.node_count() == 0 {
            return Err(LoweringError::EmptyIr);
        }

        let instructions = ir
            .nodes()
            .iter()
            .enumerate()
            .map(|(dst, node)| match *node {
                IrNode::Constant(value) => LoweredInstruction::Constant { dst, value },
                IrNode::Amp(amplitude_index) => LoweredInstruction::LoadAmplitude {
                    dst,
                    amplitude_index,
                },
                IrNode::Unary { op, input } => LoweredInstruction::Unary {
                    dst,
                    input,
                    op: match op {
                        IrUnaryOp::Neg => LoweredUnaryOp::Neg,
                        IrUnaryOp::Real => LoweredUnaryOp::Real,
                        IrUnaryOp::Imag => LoweredUnaryOp::Imag,
                        IrUnaryOp::Conj => LoweredUnaryOp::Conj,
                        IrUnaryOp::NormSqr => LoweredUnaryOp::NormSqr,
                    },
                },
                IrNode::Binary { op, left, right } => LoweredInstruction::Binary {
                    dst,
                    left,
                    right,
                    op: match op {
                        IrBinaryOp::Add => LoweredBinaryOp::Add,
                        IrBinaryOp::Sub => LoweredBinaryOp::Sub,
                        IrBinaryOp::Mul => LoweredBinaryOp::Mul,
                        IrBinaryOp::Div => LoweredBinaryOp::Div,
                    },
                },
            })
            .collect();

        Ok(Self::new(
            LoweredProgramKind::Gradient,
            instructions,
            LoweredRuntimeLayout::new(ir.node_count(), ir.root()),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        LoweredBinaryOp, LoweredExpressionRuntime, LoweredInstruction, LoweredProgram,
        LoweredProgramKind, LoweredRuntimeLayout,
    };
    use crate::amplitudes::ir::{compile_expression_ir, DependenceClass};
    use crate::amplitudes::ExpressionNode;
    use nalgebra::DVector;
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

    #[test]
    fn lowering_from_ir_value_only_preserves_program_shape() {
        let ir = compile_expression_ir(
            &ExpressionNode::NormSqr(Box::new(ExpressionNode::Mul(
                Box::new(ExpressionNode::Amp(0)),
                Box::new(ExpressionNode::Conj(Box::new(ExpressionNode::Amp(1)))),
            ))),
            &[true, true],
            &[DependenceClass::Mixed, DependenceClass::Mixed],
        );

        let program = LoweredProgram::from_ir_value_only(&ir).unwrap();

        assert_eq!(program.kind(), LoweredProgramKind::Value);
        assert_eq!(program.scratch_slots(), ir.node_count());
        assert_eq!(program.root_slot(), ir.root());
        assert_eq!(program.instructions().len(), ir.node_count());
    }

    #[test]
    fn lowered_value_program_matches_ir_evaluation() {
        let ir = compile_expression_ir(
            &ExpressionNode::Add(
                Box::new(ExpressionNode::NormSqr(Box::new(ExpressionNode::Amp(0)))),
                Box::new(ExpressionNode::Div(
                    Box::new(ExpressionNode::Conj(Box::new(ExpressionNode::Amp(1)))),
                    Box::new(ExpressionNode::One),
                )),
            ),
            &[true, true],
            &[DependenceClass::Mixed, DependenceClass::Mixed],
        );
        let program = LoweredProgram::from_ir_value_only(&ir).unwrap();
        let amplitude_values = [Complex64::new(1.5, -0.25), Complex64::new(-2.0, 0.5)];
        let mut ir_scratch = vec![Complex64::ZERO; ir.node_count()];
        let mut lowered_scratch = vec![Complex64::ZERO; program.scratch_slots()];

        let ir_value = ir.evaluate_into(&amplitude_values, &mut ir_scratch);
        let lowered_value = program.evaluate_into(&amplitude_values, &mut lowered_scratch);

        assert_eq!(lowered_value, ir_value);
    }

    #[test]
    fn lowered_runtime_from_ir_populates_value_program_only() {
        let ir = compile_expression_ir(&ExpressionNode::Amp(0), &[true], &[DependenceClass::Mixed]);

        let runtime = LoweredExpressionRuntime::from_ir_value_only(&ir).unwrap();

        assert!(runtime.value_program().is_some());
        assert!(runtime.gradient_program().is_none());
        assert!(runtime.value_gradient_program().is_none());
    }

    #[test]
    fn lowered_gradient_program_matches_ir_evaluation() {
        let ir = compile_expression_ir(
            &ExpressionNode::NormSqr(Box::new(ExpressionNode::Mul(
                Box::new(ExpressionNode::Amp(0)),
                Box::new(ExpressionNode::Conj(Box::new(ExpressionNode::Amp(1)))),
            ))),
            &[true, true],
            &[DependenceClass::Mixed, DependenceClass::Mixed],
        );
        let program = LoweredProgram::from_ir_gradient_only(&ir).unwrap();
        let amplitude_values = [Complex64::new(1.5, -0.25), Complex64::new(-2.0, 0.5)];
        let amplitude_gradients = vec![
            DVector::from_vec(vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)]),
            DVector::from_vec(vec![Complex64::new(-0.5, 0.25), Complex64::new(0.2, -0.1)]),
        ];
        let mut ir_value_scratch = vec![Complex64::ZERO; ir.node_count()];
        let mut ir_gradient_scratch = (0..ir.node_count())
            .map(|_| DVector::zeros(2))
            .collect::<Vec<_>>();
        let mut lowered_value_scratch = vec![Complex64::ZERO; program.scratch_slots()];
        let mut lowered_gradient_scratch = (0..program.scratch_slots())
            .map(|_| DVector::zeros(2))
            .collect::<Vec<_>>();

        let ir_gradient = ir.evaluate_gradient_into(
            &amplitude_values,
            &amplitude_gradients,
            &mut ir_value_scratch,
            &mut ir_gradient_scratch,
        );
        let lowered_gradient = program.evaluate_gradient_into(
            &amplitude_values,
            &amplitude_gradients,
            &mut lowered_value_scratch,
            &mut lowered_gradient_scratch,
        );

        assert_eq!(lowered_gradient, ir_gradient);
    }

    #[test]
    fn lowered_runtime_from_ir_populates_gradient_program() {
        let ir = compile_expression_ir(&ExpressionNode::Amp(0), &[true], &[DependenceClass::Mixed]);

        let runtime = LoweredExpressionRuntime::from_ir_value_gradient(&ir).unwrap();

        assert!(runtime.value_program().is_some());
        assert!(runtime.gradient_program().is_some());
        assert!(runtime.value_gradient_program().is_none());
    }
}
