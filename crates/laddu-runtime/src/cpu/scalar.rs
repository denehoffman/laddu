use laddu_expr::{
    BinaryOp, UnaryOp,
    parameters::{ParamId, ParamValues},
};
use laddu_kernel::ir::{
    KernelInstruction, KernelValue, KernelValueClass, KernelValueId, KernelValueKind,
    ScalarKernelIr,
};
use num::complex::Complex64;

use super::layout::{eval_binary, eval_unary};
use super::{CpuBatchCache, CpuExecutionMode, Precision, RuntimeError, RuntimeResult};

#[cfg(feature = "jit")]
use crate::jit::{JitPrecision, JitScalarKernel};

pub(super) const SCALAR_BLOCK_SIZE: usize = 32;

#[derive(Clone, Debug)]
pub(super) enum ScalarExecutor {
    Interpreter(ScalarEvaluationPlan),
    #[cfg(feature = "jit")]
    Jit(JitScalarKernel),
}

impl ScalarExecutor {
    pub(super) fn prepare(
        plan: &ScalarKernelIr,
        mode: CpuExecutionMode,
        precision: Precision,
    ) -> Option<Self> {
        #[cfg(not(feature = "jit"))]
        let _ = precision;
        match mode {
            CpuExecutionMode::Auto => {
                #[cfg(feature = "jit")]
                {
                    let jit_precision = match precision {
                        Precision::F32 => JitPrecision::F32,
                        Precision::Auto | Precision::F64 => JitPrecision::F64,
                    };
                    if let Ok(Some(kernel)) =
                        JitScalarKernel::compile_with_precision(plan, jit_precision)
                    {
                        return Some(Self::Jit(kernel));
                    }
                }
                ScalarEvaluationPlan::from_kernel_ir(plan).map(Self::Interpreter)
            }
            CpuExecutionMode::Interpreter => {
                ScalarEvaluationPlan::from_kernel_ir(plan).map(Self::Interpreter)
            }
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct ScalarEvaluationPlan {
    pub(super) invariant_instructions: Vec<ScalarInvariantInstruction>,
    pub(super) invariant_real_slot_count: usize,
    pub(super) invariant_complex_slot_count: usize,
    pub(super) event_instructions: Vec<ScalarEventInstruction>,
    pub(super) event_real_slot_count: usize,
    pub(super) event_complex_slot_count: usize,
    pub(super) outputs: Vec<ScalarOperand>,
}

impl ScalarEvaluationPlan {
    pub(super) fn from_kernel_ir(ir: &ScalarKernelIr) -> Option<Self> {
        Self::from_kernel_values(ir.values(), &[ir.root()])
    }

    pub(super) fn from_kernel_values(
        values: &[KernelValue],
        outputs: &[KernelValueId],
    ) -> Option<Self> {
        let mut required = vec![false; values.len()];
        let mut pending = outputs.to_vec();
        while let Some(id) = pending.pop() {
            if required[id.index()] {
                continue;
            }
            required[id.index()] = true;
            pending.extend(values[id.index()].instruction.operands());
        }
        let mut operands = Vec::with_capacity(values.len());
        let mut invariant_instructions = Vec::new();
        let mut invariant_real_slots = 0;
        let mut invariant_complex_slots = 0;
        let mut event_instructions = Vec::new();

        for (index, value) in values.iter().enumerate() {
            if !required[index] {
                operands.push(None);
                continue;
            }
            if !matches!(value.kind, KernelValueKind::Real | KernelValueKind::Complex) {
                return None;
            }
            let instruction = ScalarInstruction::from_kernel(&value.instruction, &operands);
            let operand = match value.class {
                KernelValueClass::Invariant => match value.kind {
                    KernelValueKind::Real => {
                        let slot = invariant_real_slots;
                        invariant_real_slots += 1;
                        invariant_instructions.push((ScalarSlot::Real(slot), instruction));
                        ScalarOperand::InvariantReal(slot)
                    }
                    KernelValueKind::Complex => {
                        let slot = invariant_complex_slots;
                        invariant_complex_slots += 1;
                        invariant_instructions.push((ScalarSlot::Complex(slot), instruction));
                        ScalarOperand::InvariantComplex(slot)
                    }
                    KernelValueKind::Vector { .. } | KernelValueKind::Matrix { .. } => {
                        unreachable!("aggregate values were rejected before scalar lowering")
                    }
                },
                KernelValueClass::Event => {
                    let slot = event_instructions.len();
                    let output = match value.kind {
                        KernelValueKind::Real => ScalarSlot::Real(slot),
                        KernelValueKind::Complex => ScalarSlot::Complex(slot),
                        KernelValueKind::Vector { .. } | KernelValueKind::Matrix { .. } => {
                            unreachable!("aggregate values were rejected before scalar lowering")
                        }
                    };
                    event_instructions.push((output, instruction));
                    match value.kind {
                        KernelValueKind::Real => ScalarOperand::EventReal(slot),
                        KernelValueKind::Complex => ScalarOperand::EventComplex(slot),
                        KernelValueKind::Vector { .. } | KernelValueKind::Matrix { .. } => {
                            unreachable!("aggregate values were rejected before scalar lowering")
                        }
                    }
                }
            };
            operands.push(Some(operand));
        }

        Some(Self::new(
            invariant_instructions,
            event_instructions,
            outputs
                .iter()
                .map(|output| operands[output.index()].expect("kernel output is required"))
                .collect(),
            invariant_real_slots,
            invariant_complex_slots,
        ))
    }

    fn new(
        invariant_instructions: Vec<(ScalarSlot, ScalarInstruction)>,
        event_instructions: Vec<(ScalarSlot, ScalarInstruction)>,
        outputs: Vec<ScalarOperand>,
        invariant_real_slot_count: usize,
        invariant_complex_slot_count: usize,
    ) -> Self {
        let mut last_use = vec![0; event_instructions.len()];
        for (index, (_, instruction)) in event_instructions.iter().enumerate() {
            instruction.record_event_uses(&mut last_use, index);
        }
        for output in &outputs {
            output.record_event_use(&mut last_use, event_instructions.len());
        }

        let mut logical_to_physical = vec![usize::MAX; event_instructions.len()];
        let mut free_real_slots = Vec::new();
        let mut free_complex_slots = Vec::new();
        let mut next_real_slot = 0;
        let mut next_complex_slot = 0;
        let mut slotted_instructions = Vec::with_capacity(event_instructions.len());

        for (index, (output, instruction)) in event_instructions.into_iter().enumerate() {
            let output_slot = match output {
                ScalarSlot::Real(_) => {
                    let slot = if let Some(slot) = free_real_slots.pop() {
                        slot
                    } else {
                        let slot = next_real_slot;
                        next_real_slot += 1;
                        slot
                    };
                    ScalarSlot::Real(slot)
                }
                ScalarSlot::Complex(_) => {
                    let slot = if let Some(slot) = free_complex_slots.pop() {
                        slot
                    } else {
                        let slot = next_complex_slot;
                        next_complex_slot += 1;
                        slot
                    };
                    ScalarSlot::Complex(slot)
                }
            };
            logical_to_physical[index] = output_slot.index();

            let mut event_inputs = Vec::new();
            instruction.collect_event_slots(&mut event_inputs);
            event_inputs.sort_unstable();
            event_inputs.dedup();

            slotted_instructions.push(ScalarEventInstruction {
                output_slot,
                instruction: instruction.remap_event_operands(&logical_to_physical),
            });

            for input in event_inputs {
                if last_use[input] == index {
                    match output_slot_for_event(&slotted_instructions, input) {
                        ScalarSlot::Real(slot) => free_real_slots.push(slot),
                        ScalarSlot::Complex(slot) => free_complex_slots.push(slot),
                    }
                }
            }
        }

        Self {
            invariant_instructions: invariant_instructions
                .into_iter()
                .map(|(output_slot, instruction)| ScalarInvariantInstruction {
                    output_slot,
                    instruction,
                })
                .collect(),
            invariant_real_slot_count,
            invariant_complex_slot_count,
            event_instructions: slotted_instructions,
            event_real_slot_count: next_real_slot,
            event_complex_slot_count: next_complex_slot,
            outputs: outputs
                .into_iter()
                .map(|output| output.remap_event(&logical_to_physical))
                .collect(),
        }
    }

    pub(super) fn root(&self) -> ScalarOperand {
        self.outputs[0]
    }
}

fn output_slot_for_event(
    instructions: &[ScalarEventInstruction],
    logical_slot: usize,
) -> ScalarSlot {
    instructions
        .get(logical_slot)
        .map(|instruction| instruction.output_slot)
        .expect("event input is produced by an earlier event instruction")
}

#[derive(Copy, Clone, Debug)]
pub(super) enum ScalarSlot {
    Real(usize),
    Complex(usize),
}

impl ScalarSlot {
    fn index(self) -> usize {
        match self {
            Self::Real(slot) | Self::Complex(slot) => slot,
        }
    }
}

#[derive(Clone, Default)]
pub(super) struct ScalarInvariantValues {
    pub(super) real: Vec<f64>,
    pub(super) complex: Vec<Complex64>,
}

#[derive(Clone, Default)]
pub(super) struct ScalarEventWorkspace {
    pub(super) real: Vec<[f64; SCALAR_BLOCK_SIZE]>,
    pub(super) complex: Vec<[Complex64; SCALAR_BLOCK_SIZE]>,
}

#[derive(Copy, Clone, Debug)]
pub(super) enum ScalarOperand {
    InvariantReal(usize),
    InvariantComplex(usize),
    EventReal(usize),
    EventComplex(usize),
}

impl ScalarOperand {
    pub(super) fn complex_value(
        self,
        invariant: &ScalarInvariantValues,
        event: &ScalarEventWorkspace,
    ) -> Complex64 {
        match self {
            Self::InvariantReal(slot) => Complex64::from(invariant.real[slot]),
            Self::InvariantComplex(slot) => invariant.complex[slot],
            Self::EventReal(slot) => Complex64::from(event.real[slot][0]),
            Self::EventComplex(slot) => event.complex[slot][0],
        }
    }

    fn real_value(self, invariant: &ScalarInvariantValues, event: &ScalarEventWorkspace) -> f64 {
        match self {
            Self::InvariantReal(slot) => invariant.real[slot],
            Self::InvariantComplex(slot) => invariant.complex[slot].re,
            Self::EventReal(slot) => event.real[slot][0],
            Self::EventComplex(slot) => event.complex[slot][0].re,
        }
    }

    pub(super) fn block_complex_value(
        self,
        invariant: &ScalarInvariantValues,
        event: &ScalarEventWorkspace,
        lane: usize,
    ) -> Complex64 {
        match self {
            Self::InvariantReal(slot) => Complex64::from(invariant.real[slot]),
            Self::InvariantComplex(slot) => invariant.complex[slot],
            Self::EventReal(slot) => Complex64::from(event.real[slot][lane]),
            Self::EventComplex(slot) => event.complex[slot][lane],
        }
    }

    pub(super) fn block_real_value(
        self,
        invariant: &ScalarInvariantValues,
        event: &ScalarEventWorkspace,
        lane: usize,
    ) -> f64 {
        match self {
            Self::InvariantReal(slot) => invariant.real[slot],
            Self::InvariantComplex(slot) => invariant.complex[slot].re,
            Self::EventReal(slot) => event.real[slot][lane],
            Self::EventComplex(slot) => event.complex[slot][lane].re,
        }
    }

    fn collect_event_slot(self, slots: &mut Vec<usize>) {
        if let Self::EventReal(slot) | Self::EventComplex(slot) = self {
            slots.push(slot);
        }
    }

    fn record_event_use(self, last_use: &mut [usize], instruction_index: usize) {
        if let Self::EventReal(slot) | Self::EventComplex(slot) = self {
            last_use[slot] = instruction_index;
        }
    }

    fn remap_event(self, logical_to_physical: &[usize]) -> Self {
        match self {
            Self::InvariantReal(slot) => Self::InvariantReal(slot),
            Self::InvariantComplex(slot) => Self::InvariantComplex(slot),
            Self::EventReal(slot) => Self::EventReal(logical_to_physical[slot]),
            Self::EventComplex(slot) => Self::EventComplex(logical_to_physical[slot]),
        }
    }
}

#[derive(Clone, Debug)]
pub(super) enum OperandRun {
    InvariantReal(Vec<usize>),
    InvariantComplex(Vec<usize>),
    EventReal(Vec<usize>),
    EventComplex(Vec<usize>),
}

impl OperandRun {
    fn from_operands(operands: impl IntoIterator<Item = ScalarOperand>) -> Vec<Self> {
        let mut runs = Vec::new();
        for operand in operands {
            match (runs.last_mut(), operand) {
                (Some(Self::InvariantReal(slots)), ScalarOperand::InvariantReal(slot))
                | (Some(Self::InvariantComplex(slots)), ScalarOperand::InvariantComplex(slot))
                | (Some(Self::EventReal(slots)), ScalarOperand::EventReal(slot))
                | (Some(Self::EventComplex(slots)), ScalarOperand::EventComplex(slot)) => {
                    slots.push(slot)
                }
                (_, ScalarOperand::InvariantReal(slot)) => {
                    runs.push(Self::InvariantReal(vec![slot]))
                }
                (_, ScalarOperand::InvariantComplex(slot)) => {
                    runs.push(Self::InvariantComplex(vec![slot]))
                }
                (_, ScalarOperand::EventReal(slot)) => runs.push(Self::EventReal(vec![slot])),
                (_, ScalarOperand::EventComplex(slot)) => runs.push(Self::EventComplex(vec![slot])),
            }
        }
        runs
    }

    fn add_to_complex(
        &self,
        value: &mut Complex64,
        invariant: &ScalarInvariantValues,
        event: &ScalarEventWorkspace,
    ) {
        match self {
            Self::InvariantReal(slots) => {
                for slot in slots {
                    *value += invariant.real[*slot];
                }
            }
            Self::InvariantComplex(slots) => {
                for slot in slots {
                    *value += invariant.complex[*slot];
                }
            }
            Self::EventReal(slots) => {
                for slot in slots {
                    *value += event.real[*slot][0];
                }
            }
            Self::EventComplex(slots) => {
                for slot in slots {
                    *value += event.complex[*slot][0];
                }
            }
        }
    }

    fn add_to_real(
        &self,
        value: &mut f64,
        invariant: &ScalarInvariantValues,
        event: &ScalarEventWorkspace,
    ) {
        match self {
            Self::InvariantReal(slots) => {
                for slot in slots {
                    *value += invariant.real[*slot];
                }
            }
            Self::EventReal(slots) => {
                for slot in slots {
                    *value += event.real[*slot][0];
                }
            }
            Self::InvariantComplex(_) | Self::EventComplex(_) => {
                unreachable!("complex operand appeared in real add instruction")
            }
        }
    }

    fn multiply_into_complex(
        &self,
        value: &mut Complex64,
        invariant: &ScalarInvariantValues,
        event: &ScalarEventWorkspace,
    ) {
        match self {
            Self::InvariantReal(slots) => {
                for slot in slots {
                    *value *= invariant.real[*slot];
                }
            }
            Self::InvariantComplex(slots) => {
                for slot in slots {
                    *value *= invariant.complex[*slot];
                }
            }
            Self::EventReal(slots) => {
                for slot in slots {
                    *value *= event.real[*slot][0];
                }
            }
            Self::EventComplex(slots) => {
                for slot in slots {
                    *value *= event.complex[*slot][0];
                }
            }
        }
    }

    fn multiply_into_real(
        &self,
        value: &mut f64,
        invariant: &ScalarInvariantValues,
        event: &ScalarEventWorkspace,
    ) {
        match self {
            Self::InvariantReal(slots) => {
                for slot in slots {
                    *value *= invariant.real[*slot];
                }
            }
            Self::EventReal(slots) => {
                for slot in slots {
                    *value *= event.real[*slot][0];
                }
            }
            Self::InvariantComplex(_) | Self::EventComplex(_) => {
                unreachable!("complex operand appeared in real multiply instruction")
            }
        }
    }

    fn collect_event_slots(&self, slots: &mut Vec<usize>) {
        if let Self::EventReal(event_slots) | Self::EventComplex(event_slots) = self {
            slots.extend(event_slots);
        }
    }

    fn record_event_uses(&self, last_use: &mut [usize], instruction_index: usize) {
        if let Self::EventReal(event_slots) | Self::EventComplex(event_slots) = self {
            for slot in event_slots {
                last_use[*slot] = instruction_index;
            }
        }
    }

    fn remap_events(&self, logical_to_physical: &[usize]) -> Self {
        match self {
            Self::InvariantReal(slots) => Self::InvariantReal(slots.clone()),
            Self::InvariantComplex(slots) => Self::InvariantComplex(slots.clone()),
            Self::EventReal(slots) => Self::EventReal(
                slots
                    .iter()
                    .map(|slot| logical_to_physical[*slot])
                    .collect(),
            ),
            Self::EventComplex(slots) => Self::EventComplex(
                slots
                    .iter()
                    .map(|slot| logical_to_physical[*slot])
                    .collect(),
            ),
        }
    }
}

#[derive(Clone, Debug)]
pub(super) enum ScalarInstruction {
    Cached(usize),
    Constant(Complex64),
    Parameter(ParamId),
    Unary {
        op: UnaryOp,
        input: ScalarOperand,
    },
    Binary {
        op: BinaryOp,
        lhs: ScalarOperand,
        rhs: ScalarOperand,
    },
    Add(Vec<OperandRun>),
    Mul(Vec<OperandRun>),
    Complex {
        re: ScalarOperand,
        im: ScalarOperand,
    },
    SolveRow {
        row_slot: usize,
        rhs: Vec<ScalarOperand>,
    },
    SolveRowAdjointElement {
        row_slot: usize,
        index: usize,
        len: usize,
        adjoint: ScalarOperand,
    },
}

impl ScalarInstruction {
    fn from_kernel(instruction: &KernelInstruction, operands: &[Option<ScalarOperand>]) -> Self {
        let operand = |id: KernelValueId| {
            operands[id.index()].expect("required instruction operand was lowered")
        };
        match instruction {
            KernelInstruction::Cached(slot) => Self::Cached(*slot),
            KernelInstruction::RealConstant(value) => Self::Constant(Complex64::from(*value)),
            KernelInstruction::ComplexConstant(value) => Self::Constant(*value),
            KernelInstruction::Parameter(id) => Self::Parameter(*id),
            KernelInstruction::Unary { op, input } => Self::Unary {
                op: *op,
                input: operand(*input),
            },
            KernelInstruction::Binary { op, lhs, rhs } => Self::Binary {
                op: *op,
                lhs: operand(*lhs),
                rhs: operand(*rhs),
            },
            KernelInstruction::Add(terms) => Self::Add(OperandRun::from_operands(
                terms.iter().map(|id| operand(*id)),
            )),
            KernelInstruction::Mul(factors) => Self::Mul(OperandRun::from_operands(
                factors.iter().map(|id| operand(*id)),
            )),
            KernelInstruction::Complex { re, im } => Self::Complex {
                re: operand(*re),
                im: operand(*im),
            },
            KernelInstruction::SolveRow { row_slot, rhs } => Self::SolveRow {
                row_slot: *row_slot,
                rhs: rhs.iter().map(|id| operand(*id)).collect(),
            },
            KernelInstruction::SolveRowAdjointElement {
                row_slot,
                index,
                len,
                adjoint,
            } => Self::SolveRowAdjointElement {
                row_slot: *row_slot,
                index: *index,
                len: *len,
                adjoint: operand(*adjoint),
            },
            KernelInstruction::Vector(_)
            | KernelInstruction::Matrix { .. }
            | KernelInstruction::Component { .. }
            | KernelInstruction::MatrixElement { .. }
            | KernelInstruction::MatMul { .. }
            | KernelInstruction::MatVec { .. }
            | KernelInstruction::Dot { .. }
            | KernelInstruction::Solve { .. } => {
                unreachable!("aggregate instruction cannot enter the scalar interpreter")
            }
        }
    }

    fn collect_event_slots(&self, slots: &mut Vec<usize>) {
        match self {
            Self::Cached(_) | Self::Constant(_) | Self::Parameter(_) => {}
            Self::Unary { input, .. } => input.collect_event_slot(slots),
            Self::Binary { lhs, rhs, .. } => {
                lhs.collect_event_slot(slots);
                rhs.collect_event_slot(slots);
            }
            Self::Add(runs) | Self::Mul(runs) => {
                for run in runs {
                    run.collect_event_slots(slots);
                }
            }
            Self::Complex { re, im } => {
                re.collect_event_slot(slots);
                im.collect_event_slot(slots);
            }
            Self::SolveRow { rhs, .. } => {
                for operand in rhs {
                    operand.collect_event_slot(slots);
                }
            }
            Self::SolveRowAdjointElement { adjoint, .. } => {
                adjoint.collect_event_slot(slots);
            }
        }
    }

    fn record_event_uses(&self, last_use: &mut [usize], instruction_index: usize) {
        match self {
            Self::Cached(_) | Self::Constant(_) | Self::Parameter(_) => {}
            Self::Unary { input, .. } => input.record_event_use(last_use, instruction_index),
            Self::Binary { lhs, rhs, .. } => {
                lhs.record_event_use(last_use, instruction_index);
                rhs.record_event_use(last_use, instruction_index);
            }
            Self::Add(runs) | Self::Mul(runs) => {
                for run in runs {
                    run.record_event_uses(last_use, instruction_index);
                }
            }
            Self::Complex { re, im } => {
                re.record_event_use(last_use, instruction_index);
                im.record_event_use(last_use, instruction_index);
            }
            Self::SolveRow { rhs, .. } => {
                for operand in rhs {
                    operand.record_event_use(last_use, instruction_index);
                }
            }
            Self::SolveRowAdjointElement { adjoint, .. } => {
                adjoint.record_event_use(last_use, instruction_index);
            }
        }
    }

    fn remap_event_operands(self, logical_to_physical: &[usize]) -> Self {
        match self {
            Self::Cached(slot) => Self::Cached(slot),
            Self::Constant(value) => Self::Constant(value),
            Self::Parameter(id) => Self::Parameter(id),
            Self::Unary { op, input } => Self::Unary {
                op,
                input: input.remap_event(logical_to_physical),
            },
            Self::Binary { op, lhs, rhs } => Self::Binary {
                op,
                lhs: lhs.remap_event(logical_to_physical),
                rhs: rhs.remap_event(logical_to_physical),
            },
            Self::Add(runs) => Self::Add(
                runs.iter()
                    .map(|run| run.remap_events(logical_to_physical))
                    .collect(),
            ),
            Self::Mul(runs) => Self::Mul(
                runs.iter()
                    .map(|run| run.remap_events(logical_to_physical))
                    .collect(),
            ),
            Self::Complex { re, im } => Self::Complex {
                re: re.remap_event(logical_to_physical),
                im: im.remap_event(logical_to_physical),
            },
            Self::SolveRow { row_slot, rhs } => Self::SolveRow {
                row_slot,
                rhs: rhs
                    .iter()
                    .map(|operand| operand.remap_event(logical_to_physical))
                    .collect(),
            },
            Self::SolveRowAdjointElement {
                row_slot,
                index,
                len,
                adjoint,
            } => Self::SolveRowAdjointElement {
                row_slot,
                index,
                len,
                adjoint: adjoint.remap_event(logical_to_physical),
            },
        }
    }

    pub(super) fn evaluate_real(
        &self,
        params: Option<&ParamValues>,
        cache: Option<(&CpuBatchCache, usize)>,
        invariant: &ScalarInvariantValues,
        event: &ScalarEventWorkspace,
    ) -> RuntimeResult<f64> {
        Ok(match self {
            Self::Cached(slot) => {
                cache
                    .expect("cached instruction requires an event cache")
                    .0
                    .scalar(
                        *slot,
                        cache.expect("cached instruction requires an event cache").1,
                    )?
                    .re
            }
            Self::Constant(value) => value.re,
            Self::Parameter(id) => params
                .expect("parameter instruction requires parameter values")
                .get(*id)
                .map_err(|err| RuntimeError::Parameter(err.to_string()))?,
            Self::Unary { op, input } => match op {
                UnaryOp::Neg => -input.real_value(invariant, event),
                UnaryOp::Real | UnaryOp::Conj => input.complex_value(invariant, event).re,
                UnaryOp::Imag => input.complex_value(invariant, event).im,
                UnaryOp::NormSqr => input.complex_value(invariant, event).norm_sqr(),
                UnaryOp::Sqrt => input.real_value(invariant, event).sqrt(),
                UnaryOp::Exp => input.real_value(invariant, event).exp(),
                UnaryOp::Sin => input.real_value(invariant, event).sin(),
                UnaryOp::Cos => input.real_value(invariant, event).cos(),
                UnaryOp::Log => input.real_value(invariant, event).ln(),
                UnaryOp::PowI(power) => input.real_value(invariant, event).powi(*power),
            },
            Self::Binary { op, lhs, rhs } => {
                let lhs = lhs.real_value(invariant, event);
                let rhs = rhs.real_value(invariant, event);
                match op {
                    BinaryOp::Add => lhs + rhs,
                    BinaryOp::Sub => lhs - rhs,
                    BinaryOp::Mul => lhs * rhs,
                    BinaryOp::Div => lhs / rhs,
                    BinaryOp::Atan2 => lhs.atan2(rhs),
                }
            }
            Self::Add(runs) => {
                let mut value = 0.0;
                for run in runs {
                    run.add_to_real(&mut value, invariant, event);
                }
                value
            }
            Self::Mul(runs) => {
                let mut value = 1.0;
                for run in runs {
                    run.multiply_into_real(&mut value, invariant, event);
                }
                value
            }
            Self::Complex { .. } | Self::SolveRow { .. } | Self::SolveRowAdjointElement { .. } => {
                unreachable!("complex-only instruction appeared in real scalar slot")
            }
        })
    }

    pub(super) fn evaluate_complex(
        &self,
        params: Option<&ParamValues>,
        cache: Option<(&CpuBatchCache, usize)>,
        invariant: &ScalarInvariantValues,
        event: &ScalarEventWorkspace,
    ) -> RuntimeResult<Complex64> {
        Ok(match self {
            Self::Cached(slot) => cache
                .expect("cached instruction requires an event cache")
                .0
                .scalar(
                    *slot,
                    cache.expect("cached instruction requires an event cache").1,
                )?,
            Self::Constant(value) => *value,
            Self::Parameter(id) => Complex64::from(
                params
                    .expect("parameter instruction requires parameter values")
                    .get(*id)
                    .map_err(|err| RuntimeError::Parameter(err.to_string()))?,
            ),
            Self::Unary { op, input } => eval_unary(*op, input.complex_value(invariant, event)),
            Self::Binary { op, lhs, rhs } => eval_binary(
                *op,
                lhs.complex_value(invariant, event),
                rhs.complex_value(invariant, event),
            ),
            Self::Add(runs) => {
                let mut value = Complex64::ZERO;
                for run in runs {
                    run.add_to_complex(&mut value, invariant, event);
                }
                value
            }
            Self::Mul(runs) => {
                let mut value = Complex64::ONE;
                for run in runs {
                    run.multiply_into_complex(&mut value, invariant, event);
                }
                value
            }
            Self::Complex { re, im } => Complex64::new(
                re.real_value(invariant, event),
                im.real_value(invariant, event),
            ),
            Self::SolveRow { row_slot, rhs } => {
                let (cache, row) = cache.expect("solve row instruction requires an event cache");
                let inverse_row = cache.solve_row(*row_slot, row)?;
                if inverse_row.len() != rhs.len() {
                    return Err(RuntimeError::InvalidShape {
                        index: row,
                        message: format!(
                            "specialized solve row has len {}, expected {}",
                            inverse_row.len(),
                            rhs.len()
                        ),
                    });
                }
                inverse_row
                    .iter()
                    .zip(rhs)
                    .map(|(lhs, operand)| lhs * operand.complex_value(invariant, event))
                    .sum()
            }
            Self::SolveRowAdjointElement {
                row_slot,
                index,
                len,
                adjoint,
            } => {
                let (cache, row) =
                    cache.expect("solve-row adjoint instruction requires an event cache");
                let inverse_row = cache.solve_row(*row_slot, row)?;
                if inverse_row.len() != *len {
                    return Err(RuntimeError::InvalidShape {
                        index: row,
                        message: format!(
                            "specialized solve row has len {}, expected {len}",
                            inverse_row.len()
                        ),
                    });
                }
                adjoint.complex_value(invariant, event) * inverse_row[*index].conj()
            }
        })
    }
}

#[derive(Clone, Debug)]
pub(super) struct ScalarEventInstruction {
    pub(super) output_slot: ScalarSlot,
    pub(super) instruction: ScalarInstruction,
}

#[derive(Clone, Debug)]
pub(super) struct ScalarInvariantInstruction {
    pub(super) output_slot: ScalarSlot,
    pub(super) instruction: ScalarInstruction,
}
