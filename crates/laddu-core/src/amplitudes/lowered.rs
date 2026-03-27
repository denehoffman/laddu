use super::ir::{ExpressionIR, IrBinaryOp, IrNode, IrUnaryOp};
use nalgebra::DVector;
use num::complex::Complex64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LoweringError {
    EmptyIr,
}

#[derive(Clone, Debug, PartialEq)]
struct LoweredProgramTemplate {
    instructions: Vec<LoweredInstruction>,
    layout: LoweredRuntimeLayout,
}

fn instruction_inputs(instruction: &LoweredInstruction) -> [Option<usize>; 2] {
    match *instruction {
        LoweredInstruction::Constant { .. } | LoweredInstruction::LoadAmplitude { .. } => {
            [None, None]
        }
        LoweredInstruction::Unary { input, .. } => [Some(input), None],
        LoweredInstruction::Binary { left, right, .. } => [Some(left), Some(right)],
    }
}

fn instruction_destination(instruction: &LoweredInstruction) -> usize {
    match *instruction {
        LoweredInstruction::Constant { dst, .. }
        | LoweredInstruction::LoadAmplitude { dst, .. }
        | LoweredInstruction::Unary { dst, .. }
        | LoweredInstruction::Binary { dst, .. } => dst,
    }
}

fn remap_instruction_slots(
    instruction: &LoweredInstruction,
    dst: usize,
    inputs: [Option<usize>; 2],
) -> LoweredInstruction {
    match *instruction {
        LoweredInstruction::Constant { value, .. } => LoweredInstruction::Constant { dst, value },
        LoweredInstruction::LoadAmplitude {
            amplitude_index, ..
        } => LoweredInstruction::LoadAmplitude {
            dst,
            amplitude_index,
        },
        LoweredInstruction::Unary { op, .. } => LoweredInstruction::Unary {
            dst,
            input: inputs[0].expect("unary instruction should have one input"),
            op,
        },
        LoweredInstruction::Binary { op, .. } => LoweredInstruction::Binary {
            dst,
            left: inputs[0].expect("binary instruction should have left input"),
            right: inputs[1].expect("binary instruction should have right input"),
            op,
        },
    }
}

fn allocate_reused_slots(
    instructions: &[LoweredInstruction],
    root_dst: usize,
) -> (Vec<LoweredInstruction>, LoweredRuntimeLayout) {
    let mut last_use = vec![0usize; instructions.len()];
    for (index, instruction) in instructions.iter().enumerate() {
        for input in instruction_inputs(instruction).into_iter().flatten() {
            last_use[input] = index;
        }
    }

    let mut value_slots = vec![usize::MAX; instructions.len()];
    let mut free_slots = Vec::new();
    let mut next_slot = 0usize;
    let mut remapped = Vec::with_capacity(instructions.len());

    for (index, instruction) in instructions.iter().enumerate() {
        let inputs = instruction_inputs(instruction).map(|slot| slot.map(|src| value_slots[src]));
        let dst_slot = free_slots.pop().unwrap_or_else(|| {
            let slot = next_slot;
            next_slot += 1;
            slot
        });
        let original_dst = instruction_destination(instruction);
        value_slots[original_dst] = dst_slot;
        remapped.push(remap_instruction_slots(instruction, dst_slot, inputs));

        for input in instruction_inputs(instruction).into_iter().flatten() {
            if last_use[input] == index {
                free_slots.push(value_slots[input]);
            }
        }
    }

    (
        remapped,
        LoweredRuntimeLayout::new(next_slot, value_slots[root_dst]),
    )
}

fn collect_live_ir_nodes(ir: &ExpressionIR) -> Vec<usize> {
    fn visit(node_index: usize, nodes: &[IrNode], live: &mut [bool]) {
        if live[node_index] {
            return;
        }
        live[node_index] = true;
        match nodes[node_index] {
            IrNode::Constant(_) | IrNode::Amp(_) => {}
            IrNode::Unary { input, .. } => visit(input, nodes, live),
            IrNode::Binary { left, right, .. } => {
                visit(left, nodes, live);
                visit(right, nodes, live);
            }
        }
    }

    let mut live = vec![false; ir.node_count()];
    visit(ir.root(), ir.nodes(), &mut live);
    live.into_iter()
        .enumerate()
        .filter_map(|(index, is_live)| is_live.then_some(index))
        .collect()
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

fn apply_unary_op(op: LoweredUnaryOp, value: Complex64) -> Complex64 {
    match op {
        LoweredUnaryOp::Neg => -value,
        LoweredUnaryOp::Real => Complex64::new(value.re, 0.0),
        LoweredUnaryOp::Imag => Complex64::new(value.im, 0.0),
        LoweredUnaryOp::Conj => value.conj(),
        LoweredUnaryOp::NormSqr => Complex64::new(value.norm_sqr(), 0.0),
    }
}

fn apply_binary_op(
    op: LoweredBinaryOp,
    left_value: Complex64,
    right_value: Complex64,
) -> Complex64 {
    match op {
        LoweredBinaryOp::Add => left_value + right_value,
        LoweredBinaryOp::Sub => left_value - right_value,
        LoweredBinaryOp::Mul => left_value * right_value,
        LoweredBinaryOp::Div => left_value / right_value,
    }
}

fn apply_unary_gradient_op(
    op: LoweredUnaryOp,
    value: Complex64,
    input_grad: &DVector<Complex64>,
    dst_grad: &mut DVector<Complex64>,
) {
    match op {
        LoweredUnaryOp::Neg => {
            for (dst_item, input_item) in dst_grad.iter_mut().zip(input_grad.iter()) {
                *dst_item = -*input_item;
            }
        }
        LoweredUnaryOp::Real => {
            for (dst_item, input_item) in dst_grad.iter_mut().zip(input_grad.iter()) {
                *dst_item = Complex64::new(input_item.re, 0.0);
            }
        }
        LoweredUnaryOp::Imag => {
            for (dst_item, input_item) in dst_grad.iter_mut().zip(input_grad.iter()) {
                *dst_item = Complex64::new(input_item.im, 0.0);
            }
        }
        LoweredUnaryOp::Conj => {
            for (dst_item, input_item) in dst_grad.iter_mut().zip(input_grad.iter()) {
                *dst_item = input_item.conj();
            }
        }
        LoweredUnaryOp::NormSqr => {
            let conj_input = value.conj();
            for (dst_item, input_item) in dst_grad.iter_mut().zip(input_grad.iter()) {
                *dst_item = Complex64::new(2.0 * (*input_item * conj_input).re, 0.0);
            }
        }
    }
}

fn apply_binary_gradient_op(
    op: LoweredBinaryOp,
    left_value: Complex64,
    right_value: Complex64,
    left_grad: &DVector<Complex64>,
    right_grad: &DVector<Complex64>,
    dst_grad: &mut DVector<Complex64>,
) {
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
                *dst_item = (*left_item * right_value - *right_item * left_value) / denom;
            }
        }
    }
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
                    scratch[dst] = apply_unary_op(op, scratch[input]);
                }
                LoweredInstruction::Binary {
                    dst,
                    left,
                    right,
                    op,
                } => {
                    scratch[dst] = apply_binary_op(op, scratch[left], scratch[right]);
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
        self.evaluate_gradient_like(
            amplitude_values,
            amplitude_gradients,
            value_scratch,
            gradient_scratch,
        )
    }

    pub(crate) fn evaluate_value_gradient_into(
        &self,
        amplitude_values: &[Complex64],
        amplitude_gradients: &[DVector<Complex64>],
        value_scratch: &mut [Complex64],
        gradient_scratch: &mut [DVector<Complex64>],
    ) -> (Complex64, DVector<Complex64>) {
        debug_assert_eq!(self.kind, LoweredProgramKind::ValueGradient);
        let gradient = self.evaluate_gradient_like(
            amplitude_values,
            amplitude_gradients,
            value_scratch,
            gradient_scratch,
        );
        (value_scratch[self.root_slot()], gradient)
    }

    fn evaluate_gradient_like(
        &self,
        amplitude_values: &[Complex64],
        amplitude_gradients: &[DVector<Complex64>],
        value_scratch: &mut [Complex64],
        gradient_scratch: &mut [DVector<Complex64>],
    ) -> DVector<Complex64> {
        debug_assert!(matches!(
            self.kind,
            LoweredProgramKind::Gradient | LoweredProgramKind::ValueGradient
        ));
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
                    value_scratch[dst] = apply_unary_op(op, value);
                    let input_grad = gradient_scratch[input].clone();
                    let dst_grad = &mut gradient_scratch[dst];
                    apply_unary_gradient_op(op, value, &input_grad, dst_grad);
                }
                LoweredInstruction::Binary {
                    dst,
                    left,
                    right,
                    op,
                } => {
                    let left_value = value_scratch[left];
                    let right_value = value_scratch[right];
                    value_scratch[dst] = apply_binary_op(op, left_value, right_value);
                    let left_grad = gradient_scratch[left].clone();
                    let right_grad = gradient_scratch[right].clone();
                    let dst_grad = &mut gradient_scratch[dst];
                    apply_binary_gradient_op(
                        op,
                        left_value,
                        right_value,
                        &left_grad,
                        &right_grad,
                        dst_grad,
                    );
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
        let value_gradient_program = Some(LoweredProgram::from_ir_value_gradient(ir)?);
        Ok(Self::new(
            value_program,
            gradient_program,
            value_gradient_program,
        ))
    }
}

impl LoweredProgram {
    fn lower_ir_template(ir: &ExpressionIR) -> Result<LoweredProgramTemplate, LoweringError> {
        if ir.node_count() == 0 {
            return Err(LoweringError::EmptyIr);
        }

        let live_nodes = collect_live_ir_nodes(ir);
        let mut remap = vec![usize::MAX; ir.node_count()];
        for (new_index, old_index) in live_nodes.iter().copied().enumerate() {
            remap[old_index] = new_index;
        }
        let instructions = live_nodes
            .into_iter()
            .map(|old_index| match ir.nodes()[old_index] {
                IrNode::Constant(value) => LoweredInstruction::Constant {
                    dst: remap[old_index],
                    value,
                },
                IrNode::Amp(amplitude_index) => LoweredInstruction::LoadAmplitude {
                    dst: remap[old_index],
                    amplitude_index,
                },
                IrNode::Unary { op, input } => LoweredInstruction::Unary {
                    dst: remap[old_index],
                    input: remap[input],
                    op: match op {
                        IrUnaryOp::Neg => LoweredUnaryOp::Neg,
                        IrUnaryOp::Real => LoweredUnaryOp::Real,
                        IrUnaryOp::Imag => LoweredUnaryOp::Imag,
                        IrUnaryOp::Conj => LoweredUnaryOp::Conj,
                        IrUnaryOp::NormSqr => LoweredUnaryOp::NormSqr,
                    },
                },
                IrNode::Binary { op, left, right } => LoweredInstruction::Binary {
                    dst: remap[old_index],
                    left: remap[left],
                    right: remap[right],
                    op: match op {
                        IrBinaryOp::Add => LoweredBinaryOp::Add,
                        IrBinaryOp::Sub => LoweredBinaryOp::Sub,
                        IrBinaryOp::Mul => LoweredBinaryOp::Mul,
                        IrBinaryOp::Div => LoweredBinaryOp::Div,
                    },
                },
            })
            .collect::<Vec<_>>();

        let root_slot = remap[ir.root()];
        let (instructions, layout) = allocate_reused_slots(&instructions, root_slot);

        Ok(LoweredProgramTemplate {
            instructions,
            layout,
        })
    }

    fn from_template(kind: LoweredProgramKind, template: &LoweredProgramTemplate) -> Self {
        Self::new(kind, template.instructions.clone(), template.layout.clone())
    }

    pub(crate) fn from_ir_value_only(ir: &ExpressionIR) -> Result<Self, LoweringError> {
        let template = Self::lower_ir_template(ir)?;
        Ok(Self::from_template(LoweredProgramKind::Value, &template))
    }

    pub(crate) fn from_ir_gradient_only(ir: &ExpressionIR) -> Result<Self, LoweringError> {
        let template = Self::lower_ir_template(ir)?;
        Ok(Self::from_template(LoweredProgramKind::Gradient, &template))
    }

    pub(crate) fn from_ir_value_gradient(ir: &ExpressionIR) -> Result<Self, LoweringError> {
        let template = Self::lower_ir_template(ir)?;
        Ok(Self::from_template(
            LoweredProgramKind::ValueGradient,
            &template,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        apply_binary_op, apply_unary_op, LoweredBinaryOp, LoweredExpressionRuntime,
        LoweredInstruction, LoweredProgram, LoweredProgramKind, LoweredRuntimeLayout,
        LoweredUnaryOp,
    };
    use crate::amplitudes::ir::{
        compile_expression_ir, DependenceClass, ExpressionIR, IrBinaryOp, IrNode, IrUnaryOp,
    };
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
    fn lowered_op_helpers_match_expected_complex_semantics() {
        let value = Complex64::new(2.0, -3.0);
        let other = Complex64::new(-1.0, 0.5);

        assert_eq!(apply_unary_op(LoweredUnaryOp::Neg, value), -value);
        assert_eq!(
            apply_unary_op(LoweredUnaryOp::Real, value),
            Complex64::new(value.re, 0.0)
        );
        assert_eq!(
            apply_unary_op(LoweredUnaryOp::Imag, value),
            Complex64::new(value.im, 0.0)
        );
        assert_eq!(apply_unary_op(LoweredUnaryOp::Conj, value), value.conj());
        assert_eq!(
            apply_unary_op(LoweredUnaryOp::NormSqr, value),
            Complex64::new(value.norm_sqr(), 0.0)
        );

        assert_eq!(
            apply_binary_op(LoweredBinaryOp::Add, value, other),
            value + other
        );
        assert_eq!(
            apply_binary_op(LoweredBinaryOp::Sub, value, other),
            value - other
        );
        assert_eq!(
            apply_binary_op(LoweredBinaryOp::Mul, value, other),
            value * other
        );
        assert_eq!(
            apply_binary_op(LoweredBinaryOp::Div, value, other),
            value / other
        );
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
        assert_eq!(program.instructions().len(), ir.node_count());
        assert!(program.scratch_slots() <= ir.node_count());
        assert!(program.root_slot() < program.scratch_slots());
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
        assert!(runtime.value_gradient_program().is_some());
    }

    #[test]
    fn shared_lowering_template_drives_all_program_kinds() {
        let ir = compile_expression_ir(
            &ExpressionNode::Mul(
                Box::new(ExpressionNode::Amp(0)),
                Box::new(ExpressionNode::Conj(Box::new(ExpressionNode::Amp(1)))),
            ),
            &[true, true],
            &[DependenceClass::Mixed, DependenceClass::Mixed],
        );

        let template = LoweredProgram::lower_ir_template(&ir).unwrap();
        let value = LoweredProgram::from_template(LoweredProgramKind::Value, &template);
        let gradient = LoweredProgram::from_template(LoweredProgramKind::Gradient, &template);
        let fused = LoweredProgram::from_template(LoweredProgramKind::ValueGradient, &template);

        assert_eq!(value.instructions(), gradient.instructions());
        assert_eq!(value.instructions(), fused.instructions());
        assert_eq!(value.layout(), gradient.layout());
        assert_eq!(value.layout(), fused.layout());
    }

    #[test]
    fn lowering_prunes_dead_nodes_after_specialization() {
        let ir = ExpressionIR::from_test_nodes(
            vec![
                IrNode::Amp(0),
                IrNode::Unary {
                    op: IrUnaryOp::Conj,
                    input: 0,
                },
                IrNode::Amp(1),
                IrNode::Binary {
                    op: IrBinaryOp::Mul,
                    left: 2,
                    right: 2,
                },
            ],
            1,
        );
        let program = LoweredProgram::from_ir_value_only(&ir).unwrap();

        assert!(program.scratch_slots() < ir.node_count());
        assert!(program.root_slot() < program.scratch_slots());
    }

    #[test]
    fn lowering_reuses_slots_when_values_die_early() {
        let ir = compile_expression_ir(
            &ExpressionNode::Mul(
                Box::new(ExpressionNode::Add(
                    Box::new(ExpressionNode::Amp(0)),
                    Box::new(ExpressionNode::Amp(1)),
                )),
                Box::new(ExpressionNode::Add(
                    Box::new(ExpressionNode::Amp(2)),
                    Box::new(ExpressionNode::Amp(3)),
                )),
            ),
            &[true, true, true, true],
            &[
                DependenceClass::Mixed,
                DependenceClass::Mixed,
                DependenceClass::Mixed,
                DependenceClass::Mixed,
            ],
        );
        let program = LoweredProgram::from_ir_value_only(&ir).unwrap();

        assert!(program.instructions().len() > program.scratch_slots());
    }

    #[test]
    fn lowered_value_gradient_program_matches_ir_evaluation() {
        let ir = compile_expression_ir(
            &ExpressionNode::NormSqr(Box::new(ExpressionNode::Mul(
                Box::new(ExpressionNode::Amp(0)),
                Box::new(ExpressionNode::Conj(Box::new(ExpressionNode::Amp(1)))),
            ))),
            &[true, true],
            &[DependenceClass::Mixed, DependenceClass::Mixed],
        );
        let program = LoweredProgram::from_ir_value_gradient(&ir).unwrap();
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

        let ir_value_gradient = ir.evaluate_value_gradient_into(
            &amplitude_values,
            &amplitude_gradients,
            &mut ir_value_scratch,
            &mut ir_gradient_scratch,
        );
        let lowered_value_gradient = program.evaluate_value_gradient_into(
            &amplitude_values,
            &amplitude_gradients,
            &mut lowered_value_scratch,
            &mut lowered_gradient_scratch,
        );

        assert_eq!(lowered_value_gradient.0, ir_value_gradient.0);
        assert_eq!(lowered_value_gradient.1, ir_value_gradient.1);
    }
}
