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

fn detect_norm_sqr_operand(
    instructions: &[Option<LoweredInstruction>],
    left: usize,
    right: usize,
) -> Option<usize> {
    match (&instructions[left], &instructions[right]) {
        (
            Some(LoweredInstruction::Unary {
                op: LoweredUnaryOp::Conj,
                input,
                ..
            }),
            _,
        ) if *input == right => Some(right),
        (
            _,
            Some(LoweredInstruction::Unary {
                op: LoweredUnaryOp::Conj,
                input,
                ..
            }),
        ) if *input == left => Some(left),
        _ => None,
    }
}

fn optimize_instruction_sequence(
    instructions: &[LoweredInstruction],
    root_dst: usize,
) -> (Vec<LoweredInstruction>, usize) {
    let mut kept = vec![None; instructions.len()];
    let mut constants = vec![None; instructions.len()];

    for instruction in instructions {
        match *instruction {
            LoweredInstruction::Constant { dst, value } => {
                kept[dst] = Some(LoweredInstruction::Constant { dst, value });
                constants[dst] = Some(value);
            }
            LoweredInstruction::LoadAmplitude {
                dst,
                amplitude_index,
            } => {
                kept[dst] = Some(LoweredInstruction::LoadAmplitude {
                    dst,
                    amplitude_index,
                });
                constants[dst] = None;
            }
            LoweredInstruction::Unary { dst, input, op } => {
                if let Some(value) = constants[input] {
                    let folded = apply_unary_op(op, value);
                    kept[dst] = Some(LoweredInstruction::Constant { dst, value: folded });
                    constants[dst] = Some(folded);
                    continue;
                }

                let rewritten = match (op, kept[input]) {
                    (
                        LoweredUnaryOp::Conj,
                        Some(LoweredInstruction::Unary {
                            op:
                                unary_op @ (LoweredUnaryOp::Real
                                | LoweredUnaryOp::Imag
                                | LoweredUnaryOp::NormSqr),
                            input: unary_input,
                            ..
                        }),
                    ) => Some(LoweredInstruction::Unary {
                        dst,
                        input: unary_input,
                        op: unary_op,
                    }),
                    (
                        LoweredUnaryOp::Real,
                        Some(LoweredInstruction::Unary {
                            op:
                                unary_op @ (LoweredUnaryOp::Real
                                | LoweredUnaryOp::Imag
                                | LoweredUnaryOp::NormSqr),
                            input: unary_input,
                            ..
                        }),
                    ) => Some(LoweredInstruction::Unary {
                        dst,
                        input: unary_input,
                        op: unary_op,
                    }),
                    (
                        LoweredUnaryOp::Imag,
                        Some(LoweredInstruction::Unary {
                            op: LoweredUnaryOp::Real | LoweredUnaryOp::NormSqr,
                            ..
                        }),
                    ) => {
                        kept[dst] = Some(LoweredInstruction::Constant {
                            dst,
                            value: Complex64::ZERO,
                        });
                        constants[dst] = Some(Complex64::ZERO);
                        continue;
                    }
                    (
                        LoweredUnaryOp::Real,
                        Some(LoweredInstruction::Binary {
                            op: LoweredBinaryOp::Mul,
                            left,
                            right,
                            ..
                        }),
                    ) => detect_norm_sqr_operand(&kept, left, right)
                        .map(|input| LoweredInstruction::Unary {
                            dst,
                            input,
                            op: LoweredUnaryOp::NormSqr,
                        })
                        .or(Some(LoweredInstruction::Unary { dst, input, op })),
                    (
                        LoweredUnaryOp::NormSqr,
                        Some(LoweredInstruction::Unary {
                            op: LoweredUnaryOp::Conj | LoweredUnaryOp::Neg,
                            input,
                            ..
                        }),
                    ) => Some(LoweredInstruction::Unary {
                        dst,
                        input,
                        op: LoweredUnaryOp::NormSqr,
                    }),
                    _ => Some(LoweredInstruction::Unary { dst, input, op }),
                };

                if let Some(instruction) = rewritten {
                    kept[dst] = Some(instruction);
                    constants[dst] = None;
                }
            }
            LoweredInstruction::Binary {
                dst,
                left,
                right,
                op,
            } => {
                if let (Some(left_value), Some(right_value)) = (constants[left], constants[right]) {
                    let folded = apply_binary_op(op, left_value, right_value);
                    kept[dst] = Some(LoweredInstruction::Constant { dst, value: folded });
                    constants[dst] = Some(folded);
                    continue;
                }

                let rewritten = match op {
                    LoweredBinaryOp::Add if constants[left] == Some(Complex64::ZERO) => {
                        Some(LoweredInstruction::Unary {
                            dst,
                            input: right,
                            op: LoweredUnaryOp::Identity,
                        })
                    }
                    LoweredBinaryOp::Add if constants[right] == Some(Complex64::ZERO) => {
                        Some(LoweredInstruction::Unary {
                            dst,
                            input: left,
                            op: LoweredUnaryOp::Identity,
                        })
                    }
                    LoweredBinaryOp::Sub if constants[right] == Some(Complex64::ZERO) => {
                        Some(LoweredInstruction::Unary {
                            dst,
                            input: left,
                            op: LoweredUnaryOp::Identity,
                        })
                    }
                    LoweredBinaryOp::Sub if left == right => {
                        kept[dst] = Some(LoweredInstruction::Constant {
                            dst,
                            value: Complex64::ZERO,
                        });
                        constants[dst] = Some(Complex64::ZERO);
                        continue;
                    }
                    LoweredBinaryOp::Mul
                        if constants[left] == Some(Complex64::ZERO)
                            || constants[right] == Some(Complex64::ZERO) =>
                    {
                        kept[dst] = Some(LoweredInstruction::Constant {
                            dst,
                            value: Complex64::ZERO,
                        });
                        constants[dst] = Some(Complex64::ZERO);
                        None
                    }
                    LoweredBinaryOp::Mul if constants[left] == Some(Complex64::new(1.0, 0.0)) => {
                        Some(LoweredInstruction::Unary {
                            dst,
                            input: right,
                            op: LoweredUnaryOp::Identity,
                        })
                    }
                    LoweredBinaryOp::Mul if constants[right] == Some(Complex64::new(1.0, 0.0)) => {
                        Some(LoweredInstruction::Unary {
                            dst,
                            input: left,
                            op: LoweredUnaryOp::Identity,
                        })
                    }
                    LoweredBinaryOp::Div if constants[left] == Some(Complex64::ZERO) => {
                        kept[dst] = Some(LoweredInstruction::Constant {
                            dst,
                            value: Complex64::ZERO,
                        });
                        constants[dst] = Some(Complex64::ZERO);
                        None
                    }
                    LoweredBinaryOp::Div if constants[right] == Some(Complex64::new(1.0, 0.0)) => {
                        Some(LoweredInstruction::Unary {
                            dst,
                            input: left,
                            op: LoweredUnaryOp::Identity,
                        })
                    }
                    _ => Some(LoweredInstruction::Binary {
                        dst,
                        left,
                        right,
                        op,
                    }),
                };

                if let Some(instruction) = rewritten {
                    kept[dst] = Some(instruction);
                    constants[dst] = None;
                }
            }
        }
    }

    let mut remap = vec![usize::MAX; kept.len()];
    let mut compact_instructions = Vec::new();
    for (old_dst, instruction) in kept.into_iter().enumerate() {
        if let Some(instruction) = instruction {
            remap[old_dst] = compact_instructions.len();
            compact_instructions.push(instruction);
        }
    }
    let compact_root = remap[root_dst];
    let compact_instructions = compact_instructions
        .into_iter()
        .map(|instruction| {
            let dst = remap[instruction_destination(&instruction)];
            let inputs =
                instruction_inputs(&instruction).map(|slot| slot.map(|index| remap[index]));
            remap_instruction_slots(&instruction, dst, inputs)
        })
        .collect::<Vec<_>>();

    (compact_instructions, compact_root)
}

fn allocate_reused_slots(
    instructions: &[LoweredInstruction],
    root_dst: usize,
) -> (Vec<LoweredInstruction>, LoweredRuntimeLayout) {
    debug_assert!(!instructions.is_empty());
    debug_assert!(root_dst < instructions.len());
    let mut last_use = vec![0usize; instructions.len()];
    for (index, instruction) in instructions.iter().enumerate() {
        for input in instruction_inputs(instruction).into_iter().flatten() {
            debug_assert!(input < instructions.len());
            last_use[input] = index;
        }
    }

    let mut value_slots = vec![usize::MAX; instructions.len()];
    let mut free_slots = Vec::new();
    let mut next_slot = 0usize;
    let mut remapped = Vec::with_capacity(instructions.len());

    for (index, instruction) in instructions.iter().enumerate() {
        debug_assert_eq!(instruction_destination(instruction), index);
        let inputs = instruction_inputs(instruction).map(|slot| slot.map(|src| value_slots[src]));
        debug_assert!(inputs.into_iter().flatten().all(|slot| slot != usize::MAX));
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

fn collect_live_ir_nodes_from_root(ir: &ExpressionIR, root: usize) -> Vec<usize> {
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
    visit(root, ir.nodes(), &mut live);
    live.into_iter()
        .enumerate()
        .filter_map(|(index, is_live)| is_live.then_some(index))
        .collect()
}

fn collect_live_ir_nodes(ir: &ExpressionIR) -> Vec<usize> {
    collect_live_ir_nodes_from_root(ir, ir.root())
}

fn collect_live_ir_nodes_from_roots(ir: &ExpressionIR, roots: &[usize]) -> Vec<usize> {
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
    for &root in roots {
        visit(root, ir.nodes(), &mut live);
    }
    live.into_iter()
        .enumerate()
        .filter_map(|(index, is_live)| is_live.then_some(index))
        .collect()
}

fn collect_live_ir_nodes_from_root_with_zeroed(
    ir: &ExpressionIR,
    root: usize,
    zeroed_nodes: &[bool],
) -> Vec<usize> {
    fn visit(node_index: usize, nodes: &[IrNode], live: &mut [bool], zeroed_nodes: &[bool]) {
        if live[node_index] {
            return;
        }
        live[node_index] = true;
        if zeroed_nodes.get(node_index).copied().unwrap_or(false) {
            return;
        }
        match nodes[node_index] {
            IrNode::Constant(_) | IrNode::Amp(_) => {}
            IrNode::Unary { input, .. } => visit(input, nodes, live, zeroed_nodes),
            IrNode::Binary { left, right, .. } => {
                visit(left, nodes, live, zeroed_nodes);
                visit(right, nodes, live, zeroed_nodes);
            }
        }
    }

    let mut live = vec![false; ir.node_count()];
    visit(root, ir.nodes(), &mut live, zeroed_nodes);
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
    Identity,
    Neg,
    Real,
    Imag,
    Conj,
    NormSqr,
}

impl LoweredUnaryOp {
    fn is_real_output(self) -> bool {
        matches!(
            self,
            LoweredUnaryOp::Real
                | LoweredUnaryOp::Imag
                | LoweredUnaryOp::NormSqr
                | LoweredUnaryOp::Identity
        )
    }
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
        LoweredUnaryOp::Identity => value,
        LoweredUnaryOp::Neg => -value,
        LoweredUnaryOp::Real => Complex64::new(value.re, 0.0),
        LoweredUnaryOp::Imag => Complex64::new(value.im, 0.0),
        LoweredUnaryOp::Conj => value.conj(),
        LoweredUnaryOp::NormSqr => Complex64::new(value.norm_sqr(), 0.0),
    }
}

fn apply_unary_op_real(op: LoweredUnaryOp, value: Complex64) -> Option<f64> {
    match op {
        LoweredUnaryOp::Identity => {
            if value.im == 0.0 {
                Some(value.re)
            } else {
                None
            }
        }
        LoweredUnaryOp::Real => Some(value.re),
        LoweredUnaryOp::Imag => Some(value.im),
        LoweredUnaryOp::NormSqr => Some(value.norm_sqr()),
        LoweredUnaryOp::Neg => None,
        LoweredUnaryOp::Conj => None,
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
    input_grad: &[Complex64],
    dst_grad: &mut [Complex64],
) {
    match op {
        LoweredUnaryOp::Identity => {
            for (dst_item, input_item) in dst_grad.iter_mut().zip(input_grad.iter()) {
                *dst_item = *input_item;
            }
        }
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
    left_grad: &[Complex64],
    right_grad: &[Complex64],
    dst_grad: &mut [Complex64],
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

fn gradient_slot_pair_mut(
    gradient_scratch: &mut [DVector<Complex64>],
    src: usize,
    dst: usize,
) -> (&DVector<Complex64>, &mut DVector<Complex64>) {
    debug_assert_ne!(src, dst);
    if src < dst {
        let (left, right) = gradient_scratch.split_at_mut(dst);
        (&left[src], &mut right[0])
    } else {
        let (left, right) = gradient_scratch.split_at_mut(src);
        (&right[0], &mut left[dst])
    }
}

fn gradient_slot_triple_mut(
    gradient_scratch: &mut [DVector<Complex64>],
    left: usize,
    right: usize,
    dst: usize,
) -> (
    &DVector<Complex64>,
    &DVector<Complex64>,
    &mut DVector<Complex64>,
) {
    debug_assert_ne!(left, dst);
    debug_assert_ne!(right, dst);
    debug_assert!(left < gradient_scratch.len());
    debug_assert!(right < gradient_scratch.len());
    debug_assert!(dst < gradient_scratch.len());
    let ptr = gradient_scratch.as_mut_ptr();
    // SAFETY: `dst` is assigned before current inputs are released, so the lowered slot
    // allocator guarantees the destination slot is distinct from the live source slots for
    // this step. `left` and `right` may alias each other because they are returned as shared
    // references, but neither may alias `dst`, which is returned as the sole mutable reference.
    unsafe { (&*ptr.add(left), &*ptr.add(right), &mut *ptr.add(dst)) }
}

fn flat_gradient_slot_pair_mut(
    gradient_scratch: &mut [Complex64],
    grad_dim: usize,
    src: usize,
    dst: usize,
) -> (&[Complex64], &mut [Complex64]) {
    debug_assert_ne!(src, dst);
    let src_start = src * grad_dim;
    let dst_start = dst * grad_dim;
    if src < dst {
        let (left, right) = gradient_scratch.split_at_mut(dst_start);
        (
            &left[src_start..src_start + grad_dim],
            &mut right[..grad_dim],
        )
    } else {
        let (left, right) = gradient_scratch.split_at_mut(src_start);
        (
            &right[..grad_dim],
            &mut left[dst_start..dst_start + grad_dim],
        )
    }
}

fn flat_gradient_slot_triple_mut(
    gradient_scratch: &mut [Complex64],
    grad_dim: usize,
    left: usize,
    right: usize,
    dst: usize,
) -> (&[Complex64], &[Complex64], &mut [Complex64]) {
    debug_assert_ne!(left, dst);
    debug_assert_ne!(right, dst);
    debug_assert!(grad_dim > 0);
    let slot_count = gradient_scratch.len() / grad_dim;
    debug_assert_eq!(slot_count * grad_dim, gradient_scratch.len());
    debug_assert!(left < slot_count);
    debug_assert!(right < slot_count);
    debug_assert!(dst < slot_count);
    let ptr = gradient_scratch.as_mut_ptr();
    let left_start = left * grad_dim;
    let right_start = right * grad_dim;
    let dst_start = dst * grad_dim;
    // SAFETY: each slot occupies a disjoint `grad_dim` range in the flat scratch buffer. The
    // lowered slot allocator guarantees `dst` is distinct from the live source slots for this
    // instruction, so the destination row does not overlap the source rows. `left` and `right`
    // may alias each other, but both are returned as shared slices.
    unsafe {
        (
            std::slice::from_raw_parts(ptr.add(left_start), grad_dim),
            std::slice::from_raw_parts(ptr.add(right_start), grad_dim),
            std::slice::from_raw_parts_mut(ptr.add(dst_start), grad_dim),
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LoweredRuntimeLayout {
    scratch_slots: usize,
    root_slot: usize,
}

impl LoweredRuntimeLayout {
    pub(crate) fn new(scratch_slots: usize, root_slot: usize) -> Self {
        debug_assert!(scratch_slots > 0);
        debug_assert!(root_slot < scratch_slots);
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
                    scratch[dst] = if let Some(real_value) = apply_unary_op_real(op, scratch[input])
                    {
                        Complex64::new(real_value, 0.0)
                    } else {
                        apply_unary_op(op, scratch[input])
                    };
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
                    value_scratch[dst] = if let Some(real_value) = apply_unary_op_real(op, value) {
                        Complex64::new(real_value, 0.0)
                    } else {
                        apply_unary_op(op, value)
                    };
                    let (input_grad, dst_grad) =
                        gradient_slot_pair_mut(gradient_scratch, input, dst);
                    apply_unary_gradient_op(
                        op,
                        value,
                        input_grad.as_slice(),
                        dst_grad.as_mut_slice(),
                    );
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
                    let (left_grad, right_grad, dst_grad) =
                        gradient_slot_triple_mut(gradient_scratch, left, right, dst);
                    apply_binary_gradient_op(
                        op,
                        left_value,
                        right_value,
                        left_grad.as_slice(),
                        right_grad.as_slice(),
                        dst_grad.as_mut_slice(),
                    );
                }
            }
        }

        gradient_scratch[self.root_slot()].clone()
    }

    pub(crate) fn evaluate_gradient_into_flat(
        &self,
        amplitude_values: &[Complex64],
        amplitude_gradients: &[DVector<Complex64>],
        value_scratch: &mut [Complex64],
        gradient_scratch: &mut [Complex64],
        grad_dim: usize,
    ) -> DVector<Complex64> {
        debug_assert_eq!(self.kind, LoweredProgramKind::Gradient);
        self.evaluate_gradient_like_flat(
            amplitude_values,
            amplitude_gradients,
            value_scratch,
            gradient_scratch,
            grad_dim,
        )
    }

    pub(crate) fn evaluate_value_gradient_into_flat(
        &self,
        amplitude_values: &[Complex64],
        amplitude_gradients: &[DVector<Complex64>],
        value_scratch: &mut [Complex64],
        gradient_scratch: &mut [Complex64],
        grad_dim: usize,
    ) -> (Complex64, DVector<Complex64>) {
        debug_assert_eq!(self.kind, LoweredProgramKind::ValueGradient);
        let gradient = self.evaluate_gradient_like_flat(
            amplitude_values,
            amplitude_gradients,
            value_scratch,
            gradient_scratch,
            grad_dim,
        );
        (value_scratch[self.root_slot()], gradient)
    }

    fn evaluate_gradient_like_flat(
        &self,
        amplitude_values: &[Complex64],
        amplitude_gradients: &[DVector<Complex64>],
        value_scratch: &mut [Complex64],
        gradient_scratch: &mut [Complex64],
        grad_dim: usize,
    ) -> DVector<Complex64> {
        debug_assert!(matches!(
            self.kind,
            LoweredProgramKind::Gradient | LoweredProgramKind::ValueGradient
        ));
        debug_assert!(value_scratch.len() >= self.scratch_slots());
        debug_assert!(gradient_scratch.len() >= self.scratch_slots() * grad_dim);

        if grad_dim == 0 {
            for instruction in &self.instructions {
                match *instruction {
                    LoweredInstruction::Constant { dst, value } => {
                        value_scratch[dst] = value;
                    }
                    LoweredInstruction::LoadAmplitude {
                        dst,
                        amplitude_index,
                    } => {
                        value_scratch[dst] = amplitude_values
                            .get(amplitude_index)
                            .copied()
                            .unwrap_or_default();
                    }
                    LoweredInstruction::Unary { dst, input, op } => {
                        let value = value_scratch[input];
                        value_scratch[dst] =
                            if let Some(real_value) = apply_unary_op_real(op, value) {
                                Complex64::new(real_value, 0.0)
                            } else {
                                apply_unary_op(op, value)
                            };
                    }
                    LoweredInstruction::Binary {
                        dst,
                        left,
                        right,
                        op,
                    } => {
                        value_scratch[dst] =
                            apply_binary_op(op, value_scratch[left], value_scratch[right]);
                    }
                }
            }
            return DVector::zeros(0);
        }

        for instruction in &self.instructions {
            match *instruction {
                LoweredInstruction::Constant { dst, value } => {
                    value_scratch[dst] = value;
                    gradient_scratch[dst * grad_dim..(dst + 1) * grad_dim].fill(Complex64::ZERO);
                }
                LoweredInstruction::LoadAmplitude {
                    dst,
                    amplitude_index,
                } => {
                    value_scratch[dst] = amplitude_values
                        .get(amplitude_index)
                        .copied()
                        .unwrap_or_default();
                    let dst_grad = &mut gradient_scratch[dst * grad_dim..(dst + 1) * grad_dim];
                    if let Some(source) = amplitude_gradients.get(amplitude_index) {
                        dst_grad.copy_from_slice(source.as_slice());
                    } else {
                        dst_grad.fill(Complex64::ZERO);
                    }
                }
                LoweredInstruction::Unary { dst, input, op } => {
                    let value = value_scratch[input];
                    value_scratch[dst] = if let Some(real_value) = apply_unary_op_real(op, value) {
                        Complex64::new(real_value, 0.0)
                    } else {
                        apply_unary_op(op, value)
                    };
                    let (input_grad, dst_grad) =
                        flat_gradient_slot_pair_mut(gradient_scratch, grad_dim, input, dst);
                    apply_unary_gradient_op(op, value, input_grad, dst_grad);
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
                    let (left_grad, right_grad, dst_grad) =
                        flat_gradient_slot_triple_mut(gradient_scratch, grad_dim, left, right, dst);
                    apply_binary_gradient_op(
                        op,
                        left_value,
                        right_value,
                        left_grad,
                        right_grad,
                        dst_grad,
                    );
                }
            }
        }

        DVector::from_column_slice(
            &gradient_scratch[self.root_slot() * grad_dim..(self.root_slot() + 1) * grad_dim],
        )
    }
}

/// Collection of lowered execution programs derived from the same specialized IR instance.
///
/// The value/gradient/value+gradient programs are siblings which must all correspond to the same
/// expression tree, active-mask specialization, and lowering assumptions.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LoweredExpressionRuntime {
    value_program: LoweredProgram,
    gradient_program: LoweredProgram,
    value_gradient_program: LoweredProgram,
}

/// Compact lowered runtime for cached normalization factors.
///
/// Cached parameter-factor paths only need standalone value and gradient execution; they do not
/// need the fused value+gradient program carried by the main lowered runtime family.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct LoweredFactorRuntime {
    value_program: Option<LoweredProgram>,
    gradient_program: Option<LoweredProgram>,
}

impl LoweredFactorRuntime {
    pub(crate) fn new(
        value_program: Option<LoweredProgram>,
        gradient_program: Option<LoweredProgram>,
    ) -> Self {
        Self {
            value_program,
            gradient_program,
        }
    }

    pub(crate) fn value_program(&self) -> Option<&LoweredProgram> {
        self.value_program.as_ref()
    }

    pub(crate) fn gradient_program(&self) -> Option<&LoweredProgram> {
        self.gradient_program.as_ref()
    }

    pub(crate) fn from_ir_root_value_gradient(
        ir: &ExpressionIR,
        root: usize,
    ) -> Result<Self, LoweringError> {
        let value_program = Some(LoweredProgram::from_ir_root_value_only(ir, root)?);
        let gradient_program = Some(LoweredProgram::from_ir_root_gradient_only(ir, root)?);
        Ok(Self::new(value_program, gradient_program))
    }
}

impl LoweredExpressionRuntime {
    pub(crate) fn new(
        value_program: LoweredProgram,
        gradient_program: LoweredProgram,
        value_gradient_program: LoweredProgram,
    ) -> Self {
        Self {
            value_program,
            gradient_program,
            value_gradient_program,
        }
    }

    pub(crate) fn value_program(&self) -> &LoweredProgram {
        &self.value_program
    }

    pub(crate) fn gradient_program(&self) -> &LoweredProgram {
        &self.gradient_program
    }

    pub(crate) fn value_gradient_program(&self) -> &LoweredProgram {
        &self.value_gradient_program
    }

    pub(crate) fn from_ir_value_gradient(ir: &ExpressionIR) -> Result<Self, LoweringError> {
        let value_program = LoweredProgram::from_ir_value_only(ir)?;
        let gradient_program = LoweredProgram::from_ir_gradient_only(ir)?;
        let value_gradient_program = LoweredProgram::from_ir_value_gradient(ir)?;
        Ok(Self::new(
            value_program,
            gradient_program,
            value_gradient_program,
        ))
    }

    pub(crate) fn from_ir_root_value_gradient(
        ir: &ExpressionIR,
        root: usize,
    ) -> Result<Self, LoweringError> {
        let value_program = LoweredProgram::from_ir_root_value_only(ir, root)?;
        let gradient_program = LoweredProgram::from_ir_root_gradient_only(ir, root)?;
        let value_gradient_program = LoweredProgram::from_ir_root_value_gradient(ir, root)?;
        Ok(Self::new(
            value_program,
            gradient_program,
            value_gradient_program,
        ))
    }

    pub(crate) fn from_ir_residual_terms(
        ir: &ExpressionIR,
        residual_terms: &[usize],
    ) -> Result<Self, LoweringError> {
        let value_program = LoweredProgram::from_ir_residual_terms_value_only(ir, residual_terms)?;
        let gradient_program =
            LoweredProgram::from_ir_residual_terms_gradient_only(ir, residual_terms)?;
        let value_gradient_program =
            LoweredProgram::from_ir_residual_terms_value_gradient(ir, residual_terms)?;
        Ok(Self::new(
            value_program,
            gradient_program,
            value_gradient_program,
        ))
    }

    pub(crate) fn from_ir_zeroed_value_gradient(
        ir: &ExpressionIR,
        zeroed_nodes: &[bool],
    ) -> Result<Self, LoweringError> {
        let value_program = LoweredProgram::from_ir_zeroed_value_only(ir, zeroed_nodes)?;
        let gradient_program = LoweredProgram::from_ir_zeroed_gradient_only(ir, zeroed_nodes)?;
        let value_gradient_program =
            LoweredProgram::from_ir_zeroed_value_gradient(ir, zeroed_nodes)?;
        Ok(Self::new(
            value_program,
            gradient_program,
            value_gradient_program,
        ))
    }
}

impl LoweredProgram {
    fn lower_ir_template_from_root(
        ir: &ExpressionIR,
        root: usize,
    ) -> Result<LoweredProgramTemplate, LoweringError> {
        if ir.node_count() == 0 {
            return Err(LoweringError::EmptyIr);
        }

        let live_nodes = collect_live_ir_nodes_from_root(ir, root);
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
                        // IR never emits identity directly; lowering adds it during peephole specialization.
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

        let root_slot = remap[root];
        let (instructions, root_slot) = optimize_instruction_sequence(&instructions, root_slot);
        let (instructions, layout) = allocate_reused_slots(&instructions, root_slot);

        Ok(LoweredProgramTemplate {
            instructions,
            layout,
        })
    }

    fn lower_ir_template(ir: &ExpressionIR) -> Result<LoweredProgramTemplate, LoweringError> {
        Self::lower_ir_template_from_root(ir, ir.root())
    }

    fn lower_ir_template_from_root_with_zeroed(
        ir: &ExpressionIR,
        root: usize,
        zeroed_nodes: &[bool],
    ) -> Result<LoweredProgramTemplate, LoweringError> {
        if ir.node_count() == 0 {
            return Err(LoweringError::EmptyIr);
        }

        let live_nodes = collect_live_ir_nodes_from_root_with_zeroed(ir, root, zeroed_nodes);
        let mut remap = vec![usize::MAX; ir.node_count()];
        for (new_index, old_index) in live_nodes.iter().copied().enumerate() {
            remap[old_index] = new_index;
        }
        let instructions = live_nodes
            .into_iter()
            .map(|old_index| {
                if zeroed_nodes.get(old_index).copied().unwrap_or(false) {
                    LoweredInstruction::Constant {
                        dst: remap[old_index],
                        value: Complex64::ZERO,
                    }
                } else {
                    match ir.nodes()[old_index] {
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
                                // IR never emits identity directly; lowering adds it during peephole specialization.
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
                    }
                }
            })
            .collect::<Vec<_>>();

        let root_slot = remap[root];
        let (instructions, root_slot) = optimize_instruction_sequence(&instructions, root_slot);
        let (instructions, layout) = allocate_reused_slots(&instructions, root_slot);
        Ok(LoweredProgramTemplate {
            instructions,
            layout,
        })
    }

    fn lower_ir_template_from_roots(
        ir: &ExpressionIR,
        roots: &[usize],
    ) -> Result<LoweredProgramTemplate, LoweringError> {
        if ir.node_count() == 0 || roots.is_empty() {
            return Err(LoweringError::EmptyIr);
        }

        let live_nodes = collect_live_ir_nodes_from_roots(ir, roots);
        let mut remap = vec![usize::MAX; ir.node_count()];
        for (new_index, old_index) in live_nodes.iter().copied().enumerate() {
            remap[old_index] = new_index;
        }
        let mut instructions = live_nodes
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
                        // IR never emits identity directly; lowering adds it during peephole specialization.
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

        let mut remapped_roots = roots.iter().map(|&root| remap[root]).collect::<Vec<_>>();
        let combined_root = if remapped_roots.len() == 1 {
            remapped_roots[0]
        } else {
            let mut accumulator = remapped_roots.remove(0);
            let mut next_dst = instructions.len();
            for root in remapped_roots {
                instructions.push(LoweredInstruction::Binary {
                    dst: next_dst,
                    left: accumulator,
                    right: root,
                    op: LoweredBinaryOp::Add,
                });
                accumulator = next_dst;
                next_dst += 1;
            }
            accumulator
        };

        let (instructions, root_slot) = optimize_instruction_sequence(&instructions, combined_root);
        let (instructions, layout) = allocate_reused_slots(&instructions, root_slot);
        Ok(LoweredProgramTemplate {
            instructions,
            layout,
        })
    }

    fn from_template(kind: LoweredProgramKind, template: &LoweredProgramTemplate) -> Self {
        debug_assert!(!template.instructions.is_empty());
        debug_assert_eq!(
            template.layout.scratch_slots() > 0,
            !template.instructions.is_empty()
        );
        debug_assert!(template.layout.root_slot() < template.layout.scratch_slots());
        debug_assert!(template.instructions.iter().all(|instruction| {
            let dst = instruction_destination(instruction);
            dst < template.layout.scratch_slots()
                && instruction_inputs(instruction)
                    .into_iter()
                    .flatten()
                    .all(|input| input < template.layout.scratch_slots())
        }));
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

    pub(crate) fn from_ir_root_value_only(
        ir: &ExpressionIR,
        root: usize,
    ) -> Result<Self, LoweringError> {
        let template = Self::lower_ir_template_from_root(ir, root)?;
        Ok(Self::from_template(LoweredProgramKind::Value, &template))
    }

    pub(crate) fn from_ir_root_gradient_only(
        ir: &ExpressionIR,
        root: usize,
    ) -> Result<Self, LoweringError> {
        let template = Self::lower_ir_template_from_root(ir, root)?;
        Ok(Self::from_template(LoweredProgramKind::Gradient, &template))
    }

    pub(crate) fn from_ir_root_value_gradient(
        ir: &ExpressionIR,
        root: usize,
    ) -> Result<Self, LoweringError> {
        let template = Self::lower_ir_template_from_root(ir, root)?;
        Ok(Self::from_template(
            LoweredProgramKind::ValueGradient,
            &template,
        ))
    }

    pub(crate) fn from_ir_residual_terms_value_only(
        ir: &ExpressionIR,
        residual_terms: &[usize],
    ) -> Result<Self, LoweringError> {
        let template = Self::lower_ir_template_from_roots(ir, residual_terms)?;
        Ok(Self::from_template(LoweredProgramKind::Value, &template))
    }

    pub(crate) fn from_ir_residual_terms_gradient_only(
        ir: &ExpressionIR,
        residual_terms: &[usize],
    ) -> Result<Self, LoweringError> {
        let template = Self::lower_ir_template_from_roots(ir, residual_terms)?;
        Ok(Self::from_template(LoweredProgramKind::Gradient, &template))
    }

    pub(crate) fn from_ir_residual_terms_value_gradient(
        ir: &ExpressionIR,
        residual_terms: &[usize],
    ) -> Result<Self, LoweringError> {
        let template = Self::lower_ir_template_from_roots(ir, residual_terms)?;
        Ok(Self::from_template(
            LoweredProgramKind::ValueGradient,
            &template,
        ))
    }

    pub(crate) fn from_ir_zeroed_value_only(
        ir: &ExpressionIR,
        zeroed_nodes: &[bool],
    ) -> Result<Self, LoweringError> {
        let template = Self::lower_ir_template_from_root_with_zeroed(ir, ir.root(), zeroed_nodes)?;
        Ok(Self::from_template(LoweredProgramKind::Value, &template))
    }

    pub(crate) fn from_ir_zeroed_gradient_only(
        ir: &ExpressionIR,
        zeroed_nodes: &[bool],
    ) -> Result<Self, LoweringError> {
        let template = Self::lower_ir_template_from_root_with_zeroed(ir, ir.root(), zeroed_nodes)?;
        Ok(Self::from_template(LoweredProgramKind::Gradient, &template))
    }

    pub(crate) fn from_ir_zeroed_value_gradient(
        ir: &ExpressionIR,
        zeroed_nodes: &[bool],
    ) -> Result<Self, LoweringError> {
        let template = Self::lower_ir_template_from_root_with_zeroed(ir, ir.root(), zeroed_nodes)?;
        Ok(Self::from_template(
            LoweredProgramKind::ValueGradient,
            &template,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        apply_binary_op, apply_unary_op, apply_unary_op_real, LoweredBinaryOp,
        LoweredExpressionRuntime, LoweredFactorRuntime, LoweredInstruction, LoweredProgram,
        LoweredProgramKind, LoweredRuntimeLayout, LoweredUnaryOp,
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
            apply_unary_op_real(LoweredUnaryOp::Real, value),
            Some(value.re)
        );
        assert_eq!(
            apply_unary_op_real(LoweredUnaryOp::Imag, value),
            Some(value.im)
        );
        assert_eq!(
            apply_unary_op_real(LoweredUnaryOp::NormSqr, value),
            Some(value.norm_sqr())
        );
        assert_eq!(apply_unary_op_real(LoweredUnaryOp::Conj, value), None);
        assert!(LoweredUnaryOp::Identity.is_real_output());
        assert!(LoweredUnaryOp::NormSqr.is_real_output());
        assert!(!LoweredUnaryOp::Conj.is_real_output());

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
    fn lowered_value_execution_keeps_real_unary_outputs_real() {
        let program = LoweredProgram::new(
            LoweredProgramKind::Value,
            vec![
                LoweredInstruction::LoadAmplitude {
                    dst: 0,
                    amplitude_index: 0,
                },
                LoweredInstruction::Unary {
                    dst: 1,
                    input: 0,
                    op: LoweredUnaryOp::Real,
                },
            ],
            LoweredRuntimeLayout::new(2, 1),
        );
        let mut scratch = vec![Complex64::ZERO; program.scratch_slots()];

        let result = program.evaluate_into(&[Complex64::new(2.0, -3.0)], &mut scratch);

        assert_eq!(result, Complex64::new(2.0, 0.0));
        assert_eq!(scratch[program.root_slot()].im, 0.0);
    }

    #[test]
    fn lowered_runtime_can_hold_full_program_family() {
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
        let value_gradient_program = LoweredProgram::new(
            LoweredProgramKind::ValueGradient,
            vec![LoweredInstruction::Constant {
                dst: 0,
                value: Complex64::new(2.0, 0.0),
            }],
            LoweredRuntimeLayout::new(1, 0),
        );

        let runtime = LoweredExpressionRuntime::new(
            value_program.clone(),
            gradient_program.clone(),
            value_gradient_program.clone(),
        );

        assert_eq!(runtime.value_program(), &value_program);
        assert_eq!(runtime.gradient_program(), &gradient_program);
        assert_eq!(runtime.value_gradient_program(), &value_gradient_program);
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
    fn lowered_gradient_program_flat_scratch_matches_dvector_scratch() {
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
        let mut dvector_value_scratch = vec![Complex64::ZERO; program.scratch_slots()];
        let mut dvector_gradient_scratch = (0..program.scratch_slots())
            .map(|_| DVector::zeros(2))
            .collect::<Vec<_>>();
        let mut flat_value_scratch = vec![Complex64::ZERO; program.scratch_slots()];
        let mut flat_gradient_scratch = vec![Complex64::ZERO; program.scratch_slots() * 2];

        let dvector_gradient = program.evaluate_gradient_into(
            &amplitude_values,
            &amplitude_gradients,
            &mut dvector_value_scratch,
            &mut dvector_gradient_scratch,
        );
        let flat_gradient = program.evaluate_gradient_into_flat(
            &amplitude_values,
            &amplitude_gradients,
            &mut flat_value_scratch,
            &mut flat_gradient_scratch,
            2,
        );

        assert_eq!(flat_gradient, dvector_gradient);
    }

    #[test]
    fn lowered_runtime_from_ir_populates_gradient_program() {
        let ir = compile_expression_ir(&ExpressionNode::Amp(0), &[true], &[DependenceClass::Mixed]);

        let runtime = LoweredExpressionRuntime::from_ir_value_gradient(&ir).unwrap();

        assert_eq!(runtime.value_program().kind(), LoweredProgramKind::Value);
        assert_eq!(
            runtime.gradient_program().kind(),
            LoweredProgramKind::Gradient
        );
        assert_eq!(
            runtime.value_gradient_program().kind(),
            LoweredProgramKind::ValueGradient
        );
    }

    #[test]
    fn lowered_factor_runtime_omits_fused_program() {
        let ir = compile_expression_ir(
            &ExpressionNode::Mul(
                Box::new(ExpressionNode::Amp(0)),
                Box::new(ExpressionNode::Conj(Box::new(ExpressionNode::Amp(1)))),
            ),
            &[true, true],
            &[DependenceClass::Mixed, DependenceClass::Mixed],
        );

        let runtime = LoweredFactorRuntime::from_ir_root_value_gradient(&ir, ir.root()).unwrap();

        assert!(runtime.value_program().is_some());
        assert!(runtime.gradient_program().is_some());
    }

    #[test]
    fn root_specific_lowering_matches_ir_subgraph_evaluation() {
        let ir = compile_expression_ir(
            &ExpressionNode::NormSqr(Box::new(ExpressionNode::Mul(
                Box::new(ExpressionNode::Amp(0)),
                Box::new(ExpressionNode::Conj(Box::new(ExpressionNode::Amp(1)))),
            ))),
            &[true, true],
            &[DependenceClass::Mixed, DependenceClass::Mixed],
        );
        let root = 2;
        let value_program = LoweredProgram::from_ir_root_value_only(&ir, root).unwrap();
        let gradient_program = LoweredProgram::from_ir_root_gradient_only(&ir, root).unwrap();
        let amplitude_values = [Complex64::new(1.5, -0.25), Complex64::new(-2.0, 0.5)];
        let amplitude_gradients = vec![
            DVector::from_vec(vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)]),
            DVector::from_vec(vec![Complex64::new(-0.5, 0.25), Complex64::new(0.2, -0.1)]),
        ];
        let mut ir_value_scratch = vec![Complex64::ZERO; ir.node_count()];
        let mut ir_gradient_scratch = (0..ir.node_count())
            .map(|_| DVector::zeros(2))
            .collect::<Vec<_>>();
        let _ = ir.evaluate_gradient_into(
            &amplitude_values,
            &amplitude_gradients,
            &mut ir_value_scratch,
            &mut ir_gradient_scratch,
        );
        let mut lowered_value_scratch = vec![Complex64::ZERO; value_program.scratch_slots()];
        let mut lowered_gradient_value_scratch =
            vec![Complex64::ZERO; gradient_program.scratch_slots()];
        let mut lowered_gradient_scratch = (0..gradient_program.scratch_slots())
            .map(|_| DVector::zeros(2))
            .collect::<Vec<_>>();

        let lowered_value =
            value_program.evaluate_into(&amplitude_values, &mut lowered_value_scratch);
        let lowered_gradient = gradient_program.evaluate_gradient_into(
            &amplitude_values,
            &amplitude_gradients,
            &mut lowered_gradient_value_scratch,
            &mut lowered_gradient_scratch,
        );

        assert_eq!(lowered_value, ir_value_scratch[root]);
        assert_eq!(lowered_gradient, ir_gradient_scratch[root]);
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
    fn peephole_folds_constant_only_subgraphs() {
        let ir = compile_expression_ir(
            &ExpressionNode::Add(
                Box::new(ExpressionNode::Neg(Box::new(ExpressionNode::One))),
                Box::new(ExpressionNode::One),
            ),
            &[],
            &[],
        );

        let program = LoweredProgram::from_ir_value_only(&ir).unwrap();

        assert_eq!(
            *program.instructions().last().unwrap(),
            LoweredInstruction::Constant {
                dst: program.root_slot(),
                value: Complex64::ZERO,
            }
        );
    }

    #[test]
    fn peephole_folds_zero_imaginary_projection() {
        let ir = compile_expression_ir(
            &ExpressionNode::Imag(Box::new(ExpressionNode::Real(Box::new(
                ExpressionNode::Amp(0),
            )))),
            &[true],
            &[DependenceClass::Mixed],
        );

        let program = LoweredProgram::from_ir_value_only(&ir).unwrap();
        let mut scratch = vec![Complex64::ZERO; program.scratch_slots()];

        assert_eq!(
            program.evaluate_into(&[Complex64::new(2.0, -3.0)], &mut scratch),
            Complex64::ZERO
        );
    }

    #[test]
    fn peephole_rewrites_real_conj_mul_to_norm_sqr() {
        let ir = compile_expression_ir(
            &ExpressionNode::Real(Box::new(ExpressionNode::Mul(
                Box::new(ExpressionNode::Conj(Box::new(ExpressionNode::Amp(0)))),
                Box::new(ExpressionNode::Amp(0)),
            ))),
            &[true],
            &[DependenceClass::Mixed],
        );

        let program = LoweredProgram::from_ir_value_only(&ir).unwrap();

        assert!(matches!(
            program.instructions().last(),
            Some(LoweredInstruction::Unary {
                op: LoweredUnaryOp::NormSqr,
                ..
            })
        ));
    }

    #[test]
    fn peephole_rewrites_add_zero_to_identity() {
        let ir = compile_expression_ir(
            &ExpressionNode::Add(
                Box::new(ExpressionNode::Amp(0)),
                Box::new(ExpressionNode::Zero),
            ),
            &[true],
            &[DependenceClass::Mixed],
        );

        let program = LoweredProgram::from_ir_value_only(&ir).unwrap();

        assert!(matches!(
            program.instructions().last(),
            Some(LoweredInstruction::Unary {
                op: LoweredUnaryOp::Identity,
                ..
            })
        ));
    }

    #[test]
    fn peephole_rewrites_conj_of_real_chain() {
        let ir = compile_expression_ir(
            &ExpressionNode::Conj(Box::new(ExpressionNode::Real(Box::new(
                ExpressionNode::Amp(0),
            )))),
            &[true],
            &[DependenceClass::Mixed],
        );

        let program = LoweredProgram::from_ir_value_only(&ir).unwrap();

        assert!(matches!(
            program.instructions().last(),
            Some(LoweredInstruction::Unary {
                op: LoweredUnaryOp::Real,
                ..
            })
        ));
    }

    #[test]
    fn peephole_rewrites_norm_sqr_of_conj_chain() {
        let ir = compile_expression_ir(
            &ExpressionNode::NormSqr(Box::new(ExpressionNode::Conj(Box::new(
                ExpressionNode::Amp(0),
            )))),
            &[true],
            &[DependenceClass::Mixed],
        );

        let program = LoweredProgram::from_ir_value_only(&ir).unwrap();
        let mut scratch = vec![Complex64::ZERO; program.scratch_slots()];

        assert_eq!(
            program.evaluate_into(&[Complex64::new(2.0, -3.0)], &mut scratch),
            Complex64::new(13.0, 0.0)
        );
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

    #[test]
    fn lowered_value_gradient_program_flat_scratch_matches_dvector_scratch() {
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
        let mut dvector_value_scratch = vec![Complex64::ZERO; program.scratch_slots()];
        let mut dvector_gradient_scratch = (0..program.scratch_slots())
            .map(|_| DVector::zeros(2))
            .collect::<Vec<_>>();
        let mut flat_value_scratch = vec![Complex64::ZERO; program.scratch_slots()];
        let mut flat_gradient_scratch = vec![Complex64::ZERO; program.scratch_slots() * 2];

        let dvector_value_gradient = program.evaluate_value_gradient_into(
            &amplitude_values,
            &amplitude_gradients,
            &mut dvector_value_scratch,
            &mut dvector_gradient_scratch,
        );
        let flat_value_gradient = program.evaluate_value_gradient_into_flat(
            &amplitude_values,
            &amplitude_gradients,
            &mut flat_value_scratch,
            &mut flat_gradient_scratch,
            2,
        );

        assert_eq!(flat_value_gradient, dvector_value_gradient);
    }
}
