use std::{
    collections::HashMap,
    mem::size_of,
    sync::{Arc, OnceLock},
};

use laddu_autodiff::{AutodiffMode, AutodiffPlan, AutodiffResult};
use laddu_compile::{CachePlan, CompiledModel, ReductionPlan};
#[cfg(test)]
use laddu_data::data::accurate::AccurateComplex64;
use laddu_data::{
    data::accurate::AccurateF64,
    data::{CacheStorage, Dataset, EventBatch},
    schema::Schema,
};
use laddu_expr::{
    BinaryOp, ExprGraph, ExprId, ExprNode, P4Component, UnaryOp, ValueKind,
    parameters::{ParamId, ParamLayout, ParamValues},
};
use laddu_kernel::ir::{
    KernelInstruction, KernelValue, KernelValueClass, KernelValueId, KernelValueKind,
    ScalarKernelIr,
};
use nalgebra::{DMatrix, DVector, Dyn, LU};
use num::complex::Complex64;
use rayon::prelude::*;

use crate::{RuntimeError, RuntimeResult, execution::CpuExecution};

#[cfg(feature = "jit")]
use crate::jit::{JitCacheView, JitGradientKernel, JitScalarKernel};

const SCALAR_BLOCK_SIZE: usize = 32;

pub trait EventLookup {
    fn scalar(&self, name: &str) -> Option<Complex64>;

    fn p4_component(&self, name: &str, component: P4Component) -> Option<Complex64> {
        let key = format!("{}.{}", name, component.label());
        self.scalar(&key)
    }
}

impl<F> EventLookup for F
where
    F: for<'a> Fn(&'a str) -> Option<Complex64>,
{
    fn scalar(&self, name: &str) -> Option<Complex64> {
        self(name)
    }
}

impl EventLookup for HashMap<String, Complex64> {
    fn scalar(&self, name: &str) -> Option<Complex64> {
        self.get(name).copied()
    }
}

impl EventLookup for HashMap<String, f64> {
    fn scalar(&self, name: &str) -> Option<Complex64> {
        self.get(name).copied().map(Complex64::from)
    }
}

#[derive(Clone, Debug, Default)]
pub struct CpuBackend;

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum CpuExecutionMode {
    #[default]
    Auto,
    Interpreter,
}

#[derive(Clone, Debug)]
pub struct CpuPlan {
    graph: ExprGraph,
    params: ParamLayout,
    parameter_slots: Vec<Option<ParamId>>,
    autodiff: AutodiffPlan,
    cache_plan: CachePlan,
    cache_slots: Vec<Option<usize>>,
    cached_evaluation_nodes: Vec<ExprId>,
    cached_value_slots: Vec<Option<usize>>,
    scalar_kernel: Option<ScalarKernelIr>,
    scalar_executor: Option<ScalarExecutor>,
    #[cfg_attr(not(feature = "jit"), allow(dead_code))]
    gradient_executor: GradientExecutor,
    cache_materialization_nodes: Vec<ExprId>,
    solve_components: Vec<Option<SolveComponentPlan>>,
    solve_rhs_elements: Vec<Option<Vec<ExprId>>>,
    solve_row_matrices: Vec<SolveRowMatrixPlan>,
    solve_row_keys: Vec<(ExprId, usize, usize)>,
    factor_matrix_slots: Vec<Option<usize>>,
    factor_matrices: Vec<(ExprId, usize)>,
    constant_factor_slots: Vec<Option<usize>>,
    constant_factors: Vec<Arc<OnceLock<DynamicLu>>>,
}

#[derive(Clone, Debug)]
enum ScalarExecutor {
    Interpreter(ScalarEvaluationPlan),
    #[cfg(feature = "jit")]
    Jit(JitScalarKernel),
}

impl ScalarExecutor {
    fn prepare(plan: &ScalarKernelIr, mode: CpuExecutionMode) -> Option<Self> {
        match mode {
            CpuExecutionMode::Auto => {
                #[cfg(feature = "jit")]
                if let Ok(Some(kernel)) = JitScalarKernel::compile(plan) {
                    return Some(Self::Jit(kernel));
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
enum GradientExecutor {
    Interpreter,
    #[cfg(feature = "jit")]
    Jit(JitGradientKernel),
}

impl GradientExecutor {
    fn prepare(
        plan: Option<&ScalarKernelIr>,
        params: &ParamLayout,
        mode: CpuExecutionMode,
    ) -> Self {
        #[cfg(not(feature = "jit"))]
        let _ = (plan, params, mode);
        #[cfg(feature = "jit")]
        if mode == CpuExecutionMode::Auto
            && let Some(plan) = plan
            && let Ok(Some(kernel)) = JitGradientKernel::compile(plan, params.free_params())
        {
            return Self::Jit(kernel);
        }
        Self::Interpreter
    }
}

#[derive(Copy, Clone, Debug)]
struct SolveComponentPlan {
    rhs: ExprId,
    row_slot: usize,
    dimension: usize,
}

#[derive(Clone, Debug)]
struct SolveRowMatrixPlan {
    matrix: ExprId,
    dimension: usize,
    rows: Vec<(usize, usize)>,
}

#[derive(Clone, Debug)]
struct ScalarEvaluationPlan {
    invariant_instructions: Vec<ScalarInvariantInstruction>,
    invariant_real_slot_count: usize,
    invariant_complex_slot_count: usize,
    event_instructions: Vec<ScalarEventInstruction>,
    event_real_slot_count: usize,
    event_complex_slot_count: usize,
    root: ScalarOperand,
}

impl ScalarEvaluationPlan {
    fn from_kernel_ir(ir: &ScalarKernelIr) -> Option<Self> {
        let mut operands = Vec::with_capacity(ir.values().len());
        let mut invariant_instructions = Vec::new();
        let mut invariant_real_slots = 0;
        let mut invariant_complex_slots = 0;
        let mut event_instructions = Vec::new();

        for value in ir.values() {
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
            operands.push(operand);
        }

        Some(Self::new(
            invariant_instructions,
            event_instructions,
            operands[ir.root().index()],
            invariant_real_slots,
            invariant_complex_slots,
        ))
    }

    fn new(
        invariant_instructions: Vec<(ScalarSlot, ScalarInstruction)>,
        event_instructions: Vec<(ScalarSlot, ScalarInstruction)>,
        root: ScalarOperand,
        invariant_real_slot_count: usize,
        invariant_complex_slot_count: usize,
    ) -> Self {
        let mut last_use = vec![0; event_instructions.len()];
        for (index, (_, instruction)) in event_instructions.iter().enumerate() {
            instruction.record_event_uses(&mut last_use, index);
        }
        root.record_event_use(&mut last_use, event_instructions.len());

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
            root: root.remap_event(&logical_to_physical),
        }
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
enum ScalarSlot {
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

#[derive(Default)]
struct ScalarInvariantValues {
    real: Vec<f64>,
    complex: Vec<Complex64>,
}

#[derive(Default)]
struct ScalarEventWorkspace {
    real: Vec<[f64; SCALAR_BLOCK_SIZE]>,
    complex: Vec<[Complex64; SCALAR_BLOCK_SIZE]>,
}

#[derive(Copy, Clone, Debug)]
enum ScalarOperand {
    InvariantReal(usize),
    InvariantComplex(usize),
    EventReal(usize),
    EventComplex(usize),
}

impl ScalarOperand {
    fn complex_value(
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

    fn block_complex_value(
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

    fn block_real_value(
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
enum OperandRun {
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
enum ScalarInstruction {
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
}

impl ScalarInstruction {
    fn from_kernel(instruction: &KernelInstruction, operands: &[ScalarOperand]) -> Self {
        let operand = |id: KernelValueId| operands[id.index()];
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
        }
    }

    fn evaluate_real(
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
            Self::Unary { op, input } => {
                let input = input.real_value(invariant, event);
                match op {
                    UnaryOp::Neg => -input,
                    UnaryOp::Real | UnaryOp::Conj => input,
                    UnaryOp::Imag => 0.0,
                    UnaryOp::NormSqr => input * input,
                    UnaryOp::Sqrt => input.sqrt(),
                    UnaryOp::Exp => input.exp(),
                    UnaryOp::Sin => input.sin(),
                    UnaryOp::Cos => input.cos(),
                    UnaryOp::Log => input.ln(),
                    UnaryOp::PowI(power) => input.powi(*power),
                }
            }
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
            Self::Complex { .. } | Self::SolveRow { .. } => {
                unreachable!("complex-only instruction appeared in real scalar slot")
            }
        })
    }

    fn evaluate_complex(
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
        })
    }
}

#[derive(Clone, Debug)]
struct ScalarEventInstruction {
    output_slot: ScalarSlot,
    instruction: ScalarInstruction,
}

#[derive(Clone, Debug)]
struct ScalarInvariantInstruction {
    output_slot: ScalarSlot,
    instruction: ScalarInstruction,
}

impl CpuBackend {
    pub fn prepare(&self, model: &CompiledModel) -> CpuPlan {
        self.prepare_with_modes(model, AutodiffMode::Forward, CpuExecutionMode::Auto)
            .expect("forward autodiff supports every compiled expression node")
    }

    pub fn prepare_with_execution_mode(
        &self,
        model: &CompiledModel,
        execution_mode: CpuExecutionMode,
    ) -> CpuPlan {
        self.prepare_with_modes(model, AutodiffMode::Forward, execution_mode)
            .expect("forward autodiff supports every compiled expression node")
    }

    pub fn prepare_with_autodiff_mode(
        &self,
        model: &CompiledModel,
        mode: AutodiffMode,
    ) -> AutodiffResult<CpuPlan> {
        self.prepare_with_modes(model, mode, CpuExecutionMode::Auto)
    }

    pub fn prepare_with_modes(
        &self,
        model: &CompiledModel,
        autodiff_mode: AutodiffMode,
        execution_mode: CpuExecutionMode,
    ) -> AutodiffResult<CpuPlan> {
        let cache_plan = model.cache_plan().clone();
        let parameter_slots: Vec<Option<ParamId>> = model
            .graph()
            .nodes()
            .iter()
            .map(|node| match node {
                ExprNode::ScalarParam(parameter) => Some(
                    model
                        .params()
                        .id(parameter.name())
                        .expect("compiled parameter is present in the parameter layout"),
                ),
                _ => None,
            })
            .collect();
        let mut cache_slots = vec![None; model.graph().nodes().len()];
        for (slot, entry) in cache_plan.entries().iter().enumerate() {
            cache_slots[entry.node().index()] = Some(slot);
        }
        let (solve_components, solve_rhs_elements, solve_row_matrices, solve_row_keys) =
            self.solve_component_plans(model);
        let (cached_evaluation_nodes, cached_value_slots) = cached_evaluation_schedule(
            model.graph(),
            &cache_slots,
            &solve_components,
            &solve_rhs_elements,
        );
        let scalar_kernel = self.scalar_kernel_ir(
            model,
            &cached_evaluation_nodes,
            &cache_slots,
            &parameter_slots,
            &solve_components,
            &solve_rhs_elements,
        );
        let scalar_executor = scalar_kernel
            .as_ref()
            .and_then(|kernel| ScalarExecutor::prepare(kernel, execution_mode));
        let gradient_executor =
            GradientExecutor::prepare(scalar_kernel.as_ref(), model.params(), execution_mode);
        let mut cache_required_nodes = vec![false; model.graph().nodes().len()];
        for node in cache_plan.materialization_nodes() {
            cache_required_nodes[node.index()] = true;
        }
        for plan in &solve_row_matrices {
            mark_required(model.graph(), plan.matrix, &mut cache_required_nodes);
        }
        let cache_materialization_nodes = cache_required_nodes
            .into_iter()
            .enumerate()
            .filter_map(|(index, required)| {
                required.then(|| ExprId::from_index(index).expect("graph too large"))
            })
            .collect();
        let mut factor_matrix_slots = vec![None; model.graph().nodes().len()];
        let mut factor_matrices = Vec::new();
        let mut constant_factor_slots = vec![None; model.graph().nodes().len()];
        let mut constant_factors = Vec::new();
        for (index, node) in model.graph().nodes().iter().enumerate() {
            let ExprNode::Solve { matrix, .. } = node else {
                continue;
            };
            if cached_value_slots[index].is_none() {
                continue;
            }
            let facts = model
                .node_facts(*matrix)
                .expect("compiled model facts cover every graph node");
            let dependency = facts.dependency;
            if dependency.depends_on_free_params || dependency.depends_on_fixed_params {
                continue;
            }
            let ValueKind::Matrix { rows, cols } = facts.value_kind else {
                continue;
            };
            if rows != cols {
                continue;
            }
            if dependency.depends_on_event {
                if factor_matrix_slots[matrix.index()].is_none() {
                    let slot = factor_matrices.len();
                    factor_matrix_slots[matrix.index()] = Some(slot);
                    factor_matrices.push((*matrix, rows));
                }
            } else if constant_factor_slots[matrix.index()].is_none() {
                let slot = constant_factors.len();
                constant_factor_slots[matrix.index()] = Some(slot);
                constant_factors.push(Arc::new(OnceLock::new()));
            }
        }
        Ok(CpuPlan {
            graph: model.graph().clone(),
            params: model.params().clone(),
            parameter_slots,
            autodiff: AutodiffPlan::from_model(model, autodiff_mode)?,
            cache_plan,
            cache_slots,
            cached_evaluation_nodes,
            cached_value_slots,
            scalar_kernel,
            scalar_executor,
            gradient_executor,
            cache_materialization_nodes,
            solve_components,
            solve_rhs_elements,
            solve_row_matrices,
            solve_row_keys,
            factor_matrix_slots,
            factor_matrices,
            constant_factor_slots,
            constant_factors,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ValueGradient {
    value: Complex64,
    gradient: Vec<Complex64>,
}

/// The scalar value and free-parameter gradient produced by a reduction.
#[derive(Clone, Debug, PartialEq)]
pub struct ReductionEvaluation {
    value: f64,
    gradient: Vec<f64>,
}

impl ReductionEvaluation {
    pub fn value(&self) -> f64 {
        self.value
    }

    pub fn gradient(&self) -> &[f64] {
        &self.gradient
    }

    pub fn into_parts(self) -> (f64, Vec<f64>) {
        (self.value, self.gradient)
    }
}

impl ValueGradient {
    pub fn value(&self) -> Complex64 {
        self.value
    }

    pub fn gradient(&self) -> &[Complex64] {
        &self.gradient
    }

    pub fn into_parts(self) -> (Complex64, Vec<Complex64>) {
        (self.value, self.gradient)
    }
}

struct RealGradientAccumulator {
    value: AccurateF64,
    gradient: Vec<AccurateF64>,
}

impl RealGradientAccumulator {
    fn zero(parameter_count: usize) -> Self {
        Self {
            value: AccurateF64::zero(),
            gradient: (0..parameter_count).map(|_| AccurateF64::zero()).collect(),
        }
    }

    fn push(&mut self, weight: f64, value: f64, derivative: f64, model_gradient: &[Complex64]) {
        self.value.push(weight * value);
        for (sum, model_derivative) in self.gradient.iter_mut().zip(model_gradient) {
            sum.push(weight * derivative * model_derivative.re);
        }
    }

    fn merge(&mut self, other: Self) {
        self.value.merge(other.value);
        for (target, source) in self.gradient.iter_mut().zip(other.gradient) {
            target.merge(source);
        }
    }

    fn finish(self) -> (f64, Vec<f64>) {
        (
            self.value.finish(),
            self.gradient.into_iter().map(AccurateF64::finish).collect(),
        )
    }
}

impl CpuPlan {
    fn scalar_interpreter_plan(&self) -> Option<&ScalarEvaluationPlan> {
        match (&self.scalar_kernel, &self.scalar_executor) {
            (Some(_), Some(ScalarExecutor::Interpreter(plan))) => Some(plan),
            #[cfg(feature = "jit")]
            (Some(_), Some(ScalarExecutor::Jit(_))) => None,
            (Some(_), None) | (None, None) => None,
            (None, Some(_)) => unreachable!("executor requires kernel IR"),
        }
    }

    #[cfg(feature = "jit")]
    fn scalar_jit_kernel(&self) -> Option<&JitScalarKernel> {
        match (&self.scalar_kernel, &self.scalar_executor) {
            (Some(_), Some(ScalarExecutor::Jit(kernel))) => Some(kernel),
            (Some(_), Some(ScalarExecutor::Interpreter(_))) | (Some(_), None) | (None, None) => {
                None
            }
            (None, Some(_)) => unreachable!("executor requires kernel IR"),
        }
    }

    #[cfg(feature = "jit")]
    fn gradient_jit_kernel(&self) -> Option<&JitGradientKernel> {
        match &self.gradient_executor {
            GradientExecutor::Jit(kernel) => Some(kernel),
            GradientExecutor::Interpreter => None,
        }
    }

    fn parameter_value(&self, params: &ParamValues, node: usize) -> RuntimeResult<f64> {
        let id = self.parameter_slots[node].ok_or_else(|| RuntimeError::InvalidShape {
            index: node,
            message: "node is not a parameter".into(),
        })?;
        params
            .get(id)
            .map_err(|err| RuntimeError::Parameter(err.to_string()))
    }

    pub fn parameter_count(&self) -> usize {
        self.params.len()
    }

    pub fn free_parameter_count(&self) -> usize {
        self.params.n_free()
    }

    pub fn cache_plan(&self) -> &CachePlan {
        &self.cache_plan
    }

    pub fn evaluate(&self, params: &ParamValues) -> RuntimeResult<Complex64> {
        self.evaluate_inner(params, None)
    }

    pub fn evaluate_with_gradient(&self, params: &ParamValues) -> RuntimeResult<ValueGradient> {
        #[cfg(feature = "jit")]
        if let (Some(value_kernel), Some(gradient_kernel)) =
            (self.scalar_jit_kernel(), self.gradient_jit_kernel())
        {
            let value = value_kernel.evaluate_invariant(params)?;
            let mut real = Vec::new();
            let mut imag = Vec::new();
            gradient_kernel.evaluate_invariant_component(params, 0, &mut real)?;
            gradient_kernel.evaluate_invariant_component(params, 1, &mut imag)?;
            let gradient = real
                .into_iter()
                .zip(imag)
                .map(|(re, im)| Complex64::new(re, im))
                .collect();
            return Ok(ValueGradient { value, gradient });
        }
        let values = self.evaluate_values(params, None)?;
        self.value_gradient(values, None)
    }

    pub fn evaluate_with_event(
        &self,
        params: &ParamValues,
        event: &impl EventLookup,
    ) -> RuntimeResult<Complex64> {
        self.evaluate_inner(params, Some(event))
    }

    pub fn evaluate_with_event_and_gradient(
        &self,
        params: &ParamValues,
        event: &impl EventLookup,
    ) -> RuntimeResult<ValueGradient> {
        let values = self.evaluate_values(params, Some(event))?;
        self.value_gradient(values, None)
    }

    pub fn cache_event_batch(&self, batch: &EventBatch) -> RuntimeResult<CpuBatchCache> {
        let event_columns = self.event_columns(batch.schema())?;
        let mut cache = CpuBatchCache::new(
            &self.cache_plan,
            &self.factor_matrices,
            &self.solve_row_keys,
            batch.len(),
        );
        for row in 0..batch.len() {
            let values = self.evaluate_cache_values_for_row(batch, row, &event_columns)?;
            for (slot, entry) in self.cache_plan.entries().iter().enumerate() {
                let value = values[entry.node().index()]
                    .as_ref()
                    .expect("cacheable node should have been evaluated")
                    .clone();
                cache.push(slot, value)?;
            }
            for plan in &self.solve_row_matrices {
                let (rows, cols, values) = matrix_at_optional(&values, plan.matrix.index())?;
                if rows != plan.dimension || cols != plan.dimension {
                    return Err(RuntimeError::InvalidShape {
                        index: plan.matrix.index(),
                        message: format!(
                            "specialized solve expected a {}x{} matrix, got {rows}x{cols}",
                            plan.dimension, plan.dimension
                        ),
                    });
                }
                let transpose_factor = DMatrix::from_row_slice(rows, cols, values).transpose().lu();
                for (slot, index) in &plan.rows {
                    let mut basis = DVector::zeros(plan.dimension);
                    basis[*index] = Complex64::ONE;
                    let inverse_row = transpose_factor
                        .solve(&basis)
                        .ok_or(RuntimeError::SingularMatrix(plan.matrix.index()))?;
                    cache.push_solve_row(*slot, inverse_row.iter().copied())?;
                }
            }
            for (slot, (matrix, _)) in self.factor_matrices.iter().enumerate() {
                let (rows, cols, values) = matrix_at_optional(&values, matrix.index())?;
                cache.push_factor(slot, DMatrix::from_row_slice(rows, cols, values).lu())?;
            }
        }
        cache.set_weights((0..batch.len()).map(|row| batch.weights_at(row)).collect());
        Ok(cache)
    }

    pub fn evaluate_cache(
        &self,
        params: &ParamValues,
        cache: &CpuBatchCache,
    ) -> RuntimeResult<Vec<Complex64>> {
        self.check_batch_cache(cache)?;
        #[cfg(feature = "jit")]
        if let Some(kernel) = self.scalar_jit_kernel() {
            let mut output = Vec::with_capacity(cache.len());
            kernel.evaluate(params, cache, 0, cache.len(), &mut output)?;
            return Ok(output);
        }
        let invariant = self.scalar_invariant_values(params)?;
        let mut out = Vec::with_capacity(cache.len());
        let mut workspace = ScalarEventWorkspace::default();
        for row in 0..cache.len() {
            out.push(self.evaluate_cache_row_prepared(
                params,
                cache,
                row,
                invariant.as_ref(),
                &mut workspace,
            )?);
        }
        Ok(out)
    }

    pub fn evaluate_cache_row(
        &self,
        params: &ParamValues,
        cache: &CpuBatchCache,
        row: usize,
    ) -> RuntimeResult<Complex64> {
        self.check_batch_cache(cache)?;
        self.evaluate_cache_row_unchecked(params, cache, row)
    }

    fn evaluate_cache_row_unchecked(
        &self,
        params: &ParamValues,
        cache: &CpuBatchCache,
        row: usize,
    ) -> RuntimeResult<Complex64> {
        let invariant = self.scalar_invariant_values(params)?;
        self.evaluate_cache_row_prepared(
            params,
            cache,
            row,
            invariant.as_ref(),
            &mut ScalarEventWorkspace::default(),
        )
    }

    fn evaluate_cache_row_prepared(
        &self,
        params: &ParamValues,
        cache: &CpuBatchCache,
        row: usize,
        invariant: Option<&ScalarInvariantValues>,
        workspace: &mut ScalarEventWorkspace,
    ) -> RuntimeResult<Complex64> {
        #[cfg(feature = "jit")]
        if let Some(kernel) = self.scalar_jit_kernel() {
            let mut output = Vec::with_capacity(1);
            kernel.evaluate(params, cache, row, row + 1, &mut output)?;
            return Ok(output[0]);
        }
        if let (Some(plan), Some(invariant)) = (self.scalar_interpreter_plan(), invariant) {
            return self.evaluate_scalar_cache_row(cache, row, plan, invariant, workspace);
        }
        let values = self.evaluate_values_from_cache(params, cache, row)?;
        self.cached_scalar_at(&values, self.graph.root())
    }

    fn scalar_invariant_values(
        &self,
        params: &ParamValues,
    ) -> RuntimeResult<Option<ScalarInvariantValues>> {
        let Some(plan) = self.scalar_interpreter_plan() else {
            return Ok(None);
        };
        let mut values = ScalarInvariantValues {
            real: vec![0.0; plan.invariant_real_slot_count],
            complex: vec![Complex64::ZERO; plan.invariant_complex_slot_count],
        };
        let event = ScalarEventWorkspace::default();
        for instruction in &plan.invariant_instructions {
            match instruction.output_slot {
                ScalarSlot::Real(slot) => {
                    values.real[slot] = instruction.instruction.evaluate_real(
                        Some(params),
                        None,
                        &values,
                        &event,
                    )?;
                }
                ScalarSlot::Complex(slot) => {
                    values.complex[slot] = instruction.instruction.evaluate_complex(
                        Some(params),
                        None,
                        &values,
                        &event,
                    )?;
                }
            }
        }
        Ok(Some(values))
    }

    fn evaluate_scalar_cache_row(
        &self,
        cache: &CpuBatchCache,
        row: usize,
        plan: &ScalarEvaluationPlan,
        invariant: &ScalarInvariantValues,
        values: &mut ScalarEventWorkspace,
    ) -> RuntimeResult<Complex64> {
        values.real.clear();
        values
            .real
            .resize(plan.event_real_slot_count, [0.0; SCALAR_BLOCK_SIZE]);
        values.complex.clear();
        values.complex.resize(
            plan.event_complex_slot_count,
            [Complex64::ZERO; SCALAR_BLOCK_SIZE],
        );
        for event_instruction in &plan.event_instructions {
            match event_instruction.output_slot {
                ScalarSlot::Real(slot) => {
                    values.real[slot][0] = event_instruction.instruction.evaluate_real(
                        None,
                        Some((cache, row)),
                        invariant,
                        values,
                    )?;
                }
                ScalarSlot::Complex(slot) => {
                    values.complex[slot][0] = event_instruction.instruction.evaluate_complex(
                        None,
                        Some((cache, row)),
                        invariant,
                        values,
                    )?;
                }
            }
        }
        Ok(plan.root.complex_value(invariant, values))
    }

    fn evaluate_cache_block_prepared(
        &self,
        params: &ParamValues,
        cache: &CpuBatchCache,
        start: usize,
        end: usize,
        invariant: Option<&ScalarInvariantValues>,
        workspace: &mut ScalarEventWorkspace,
        output: &mut Vec<Complex64>,
        #[cfg(feature = "jit")] jit_cache: Option<&JitCacheView>,
    ) -> RuntimeResult<()> {
        #[cfg(feature = "jit")]
        if let Some(kernel) = self.scalar_jit_kernel() {
            let owned;
            let jit_cache = if let Some(jit_cache) = jit_cache {
                jit_cache
            } else {
                owned = JitScalarKernel::prepare_cache(cache);
                &owned
            };
            return kernel.evaluate_prepared(params, jit_cache, start, end, output);
        }
        if let (Some(plan), Some(invariant)) = (self.scalar_interpreter_plan(), invariant) {
            return self.evaluate_scalar_cache_block(
                cache, start, end, plan, invariant, workspace, output,
            );
        }
        output.clear();
        for row in start..end {
            output
                .push(self.evaluate_cache_row_prepared(params, cache, row, invariant, workspace)?);
        }
        Ok(())
    }

    fn evaluate_scalar_cache_block(
        &self,
        cache: &CpuBatchCache,
        start: usize,
        end: usize,
        plan: &ScalarEvaluationPlan,
        invariant: &ScalarInvariantValues,
        workspace: &mut ScalarEventWorkspace,
        output: &mut Vec<Complex64>,
    ) -> RuntimeResult<()> {
        let block_len = end - start;
        workspace
            .real
            .resize(plan.event_real_slot_count, [0.0; SCALAR_BLOCK_SIZE]);
        workspace.complex.resize(
            plan.event_complex_slot_count,
            [Complex64::ZERO; SCALAR_BLOCK_SIZE],
        );

        for event_instruction in &plan.event_instructions {
            match event_instruction.output_slot {
                ScalarSlot::Real(slot) => {
                    let output_slot = slot;
                    match &event_instruction.instruction {
                        ScalarInstruction::Cached(slot) => {
                            workspace.real[output_slot][..block_len]
                                .copy_from_slice(cache.real_range(*slot, start, end)?);
                        }
                        ScalarInstruction::Unary { op, input } => {
                            for lane in 0..block_len {
                                workspace.real[output_slot][lane] = match op {
                                    UnaryOp::Neg => {
                                        -input.block_real_value(invariant, workspace, lane)
                                    }
                                    UnaryOp::Real | UnaryOp::Conj => {
                                        input.block_complex_value(invariant, workspace, lane).re
                                    }
                                    UnaryOp::Imag => {
                                        input.block_complex_value(invariant, workspace, lane).im
                                    }
                                    UnaryOp::NormSqr => input
                                        .block_complex_value(invariant, workspace, lane)
                                        .norm_sqr(),
                                    UnaryOp::Sqrt => {
                                        input.block_real_value(invariant, workspace, lane).sqrt()
                                    }
                                    UnaryOp::Exp => {
                                        input.block_real_value(invariant, workspace, lane).exp()
                                    }
                                    UnaryOp::Sin => {
                                        input.block_real_value(invariant, workspace, lane).sin()
                                    }
                                    UnaryOp::Cos => {
                                        input.block_real_value(invariant, workspace, lane).cos()
                                    }
                                    UnaryOp::Log => {
                                        input.block_real_value(invariant, workspace, lane).ln()
                                    }
                                    UnaryOp::PowI(power) => input
                                        .block_real_value(invariant, workspace, lane)
                                        .powi(*power),
                                };
                            }
                        }
                        ScalarInstruction::Binary { op, lhs, rhs } => {
                            for lane in 0..block_len {
                                let lhs = lhs.block_real_value(invariant, workspace, lane);
                                let rhs = rhs.block_real_value(invariant, workspace, lane);
                                workspace.real[output_slot][lane] = match op {
                                    BinaryOp::Add => lhs + rhs,
                                    BinaryOp::Sub => lhs - rhs,
                                    BinaryOp::Mul => lhs * rhs,
                                    BinaryOp::Div => lhs / rhs,
                                    BinaryOp::Atan2 => lhs.atan2(rhs),
                                };
                            }
                        }
                        ScalarInstruction::Add(runs) => {
                            workspace.real[output_slot][..block_len].fill(0.0);
                            for run in runs {
                                match run {
                                    OperandRun::InvariantReal(slots) => {
                                        for slot in slots {
                                            let operand = invariant.real[*slot];
                                            for lane in 0..block_len {
                                                workspace.real[output_slot][lane] += operand;
                                            }
                                        }
                                    }
                                    OperandRun::EventReal(slots) => {
                                        for slot in slots {
                                            for lane in 0..block_len {
                                                workspace.real[output_slot][lane] +=
                                                    workspace.real[*slot][lane];
                                            }
                                        }
                                    }
                                    OperandRun::InvariantComplex(_)
                                    | OperandRun::EventComplex(_) => {
                                        unreachable!("complex operand appeared in real add")
                                    }
                                }
                            }
                        }
                        ScalarInstruction::Mul(runs) => {
                            workspace.real[output_slot][..block_len].fill(1.0);
                            for run in runs {
                                match run {
                                    OperandRun::InvariantReal(slots) => {
                                        for slot in slots {
                                            let operand = invariant.real[*slot];
                                            for lane in 0..block_len {
                                                workspace.real[output_slot][lane] *= operand;
                                            }
                                        }
                                    }
                                    OperandRun::EventReal(slots) => {
                                        for slot in slots {
                                            for lane in 0..block_len {
                                                workspace.real[output_slot][lane] *=
                                                    workspace.real[*slot][lane];
                                            }
                                        }
                                    }
                                    OperandRun::InvariantComplex(_)
                                    | OperandRun::EventComplex(_) => {
                                        unreachable!("complex operand appeared in real multiply")
                                    }
                                }
                            }
                        }
                        ScalarInstruction::Constant(_)
                        | ScalarInstruction::Parameter(_)
                        | ScalarInstruction::Complex { .. }
                        | ScalarInstruction::SolveRow { .. } => {
                            unreachable!("non-real event instruction appeared in a real slot")
                        }
                    }
                }
                ScalarSlot::Complex(slot) => {
                    let output_slot = slot;
                    match &event_instruction.instruction {
                        ScalarInstruction::Cached(slot) => {
                            workspace.complex[output_slot][..block_len]
                                .copy_from_slice(cache.complex_range(*slot, start, end)?);
                        }
                        ScalarInstruction::Unary { op, input } => {
                            for lane in 0..block_len {
                                let input = input.block_complex_value(invariant, workspace, lane);
                                workspace.complex[output_slot][lane] = eval_unary(*op, input);
                            }
                        }
                        ScalarInstruction::Binary { op, lhs, rhs } => {
                            for lane in 0..block_len {
                                let lhs = lhs.block_complex_value(invariant, workspace, lane);
                                let rhs = rhs.block_complex_value(invariant, workspace, lane);
                                workspace.complex[output_slot][lane] = eval_binary(*op, lhs, rhs);
                            }
                        }
                        ScalarInstruction::Add(runs) => {
                            workspace.complex[output_slot][..block_len].fill(Complex64::ZERO);
                            for run in runs {
                                match run {
                                    OperandRun::InvariantReal(slots) => {
                                        for slot in slots {
                                            let operand = invariant.real[*slot];
                                            for lane in 0..block_len {
                                                workspace.complex[output_slot][lane] += operand;
                                            }
                                        }
                                    }
                                    OperandRun::InvariantComplex(slots) => {
                                        for slot in slots {
                                            let operand = invariant.complex[*slot];
                                            for lane in 0..block_len {
                                                workspace.complex[output_slot][lane] += operand;
                                            }
                                        }
                                    }
                                    OperandRun::EventReal(slots) => {
                                        for slot in slots {
                                            for lane in 0..block_len {
                                                workspace.complex[output_slot][lane] +=
                                                    workspace.real[*slot][lane];
                                            }
                                        }
                                    }
                                    OperandRun::EventComplex(slots) => {
                                        for slot in slots {
                                            for lane in 0..block_len {
                                                let operand = workspace.complex[*slot][lane];
                                                workspace.complex[output_slot][lane] += operand;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        ScalarInstruction::Mul(runs) => {
                            workspace.complex[output_slot][..block_len].fill(Complex64::ONE);
                            for run in runs {
                                match run {
                                    OperandRun::InvariantReal(slots) => {
                                        for slot in slots {
                                            let operand = invariant.real[*slot];
                                            for lane in 0..block_len {
                                                workspace.complex[output_slot][lane] *= operand;
                                            }
                                        }
                                    }
                                    OperandRun::InvariantComplex(slots) => {
                                        for slot in slots {
                                            let operand = invariant.complex[*slot];
                                            for lane in 0..block_len {
                                                workspace.complex[output_slot][lane] *= operand;
                                            }
                                        }
                                    }
                                    OperandRun::EventReal(slots) => {
                                        for slot in slots {
                                            for lane in 0..block_len {
                                                workspace.complex[output_slot][lane] *=
                                                    workspace.real[*slot][lane];
                                            }
                                        }
                                    }
                                    OperandRun::EventComplex(slots) => {
                                        for slot in slots {
                                            for lane in 0..block_len {
                                                let operand = workspace.complex[*slot][lane];
                                                workspace.complex[output_slot][lane] *= operand;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        ScalarInstruction::Complex { re, im } => {
                            for lane in 0..block_len {
                                workspace.complex[output_slot][lane] = Complex64::new(
                                    re.block_real_value(invariant, workspace, lane),
                                    im.block_real_value(invariant, workspace, lane),
                                );
                            }
                        }
                        ScalarInstruction::SolveRow { row_slot, rhs } => {
                            for lane in 0..block_len {
                                let inverse_row = cache.solve_row(*row_slot, start + lane)?;
                                if inverse_row.len() != rhs.len() {
                                    return Err(RuntimeError::InvalidShape {
                                        index: start + lane,
                                        message: format!(
                                            "specialized solve row has len {}, expected {}",
                                            inverse_row.len(),
                                            rhs.len()
                                        ),
                                    });
                                }
                                workspace.complex[output_slot][lane] = inverse_row
                                    .iter()
                                    .zip(rhs)
                                    .map(|(lhs, operand)| {
                                        lhs * operand
                                            .block_complex_value(invariant, workspace, lane)
                                    })
                                    .sum();
                            }
                        }
                        ScalarInstruction::Constant(_) | ScalarInstruction::Parameter(_) => {
                            unreachable!("invariant instruction appeared in the event tape")
                        }
                    }
                }
            }
        }

        output.clear();
        output.reserve(block_len);
        match plan.root {
            ScalarOperand::InvariantReal(slot) => {
                output.resize(block_len, Complex64::from(invariant.real[slot]))
            }
            ScalarOperand::InvariantComplex(slot) => {
                output.resize(block_len, invariant.complex[slot])
            }
            ScalarOperand::EventReal(slot) => {
                output.extend(
                    workspace.real[slot][..block_len]
                        .iter()
                        .copied()
                        .map(Complex64::from),
                );
            }
            ScalarOperand::EventComplex(slot) => {
                output.extend_from_slice(&workspace.complex[slot][..block_len]);
            }
        }
        Ok(())
    }

    pub fn evaluate_cache_row_with_gradient(
        &self,
        params: &ParamValues,
        cache: &CpuBatchCache,
        row: usize,
    ) -> RuntimeResult<ValueGradient> {
        self.check_batch_cache(cache)?;
        self.evaluate_cache_row_with_gradient_unchecked(params, cache, row)
    }

    fn evaluate_cache_row_with_gradient_unchecked(
        &self,
        params: &ParamValues,
        cache: &CpuBatchCache,
        row: usize,
    ) -> RuntimeResult<ValueGradient> {
        #[cfg(feature = "jit")]
        if self.gradient_jit_kernel().is_some() {
            return self
                .evaluate_cache_gradient_jit(params, cache, row, row + 1)?
                .pop()
                .ok_or_else(|| RuntimeError::InvalidShape {
                    index: row,
                    message: "single-row JIT gradient produced no value".into(),
                });
        }
        let values = self.evaluate_values_from_cache(params, cache, row)?;
        self.value_gradient(values, Some((cache, row)))
    }

    pub fn evaluate_cache_with_gradient(
        &self,
        params: &ParamValues,
        cache: &CpuBatchCache,
    ) -> RuntimeResult<Vec<ValueGradient>> {
        self.check_batch_cache(cache)?;
        #[cfg(feature = "jit")]
        if self.gradient_jit_kernel().is_some() {
            return self.evaluate_cache_gradient_jit(params, cache, 0, cache.len());
        }
        (0..cache.len())
            .map(|row| self.evaluate_cache_row_with_gradient_unchecked(params, cache, row))
            .collect()
    }

    #[cfg(feature = "jit")]
    fn evaluate_cache_gradient_jit(
        &self,
        params: &ParamValues,
        cache: &CpuBatchCache,
        start: usize,
        end: usize,
    ) -> RuntimeResult<Vec<ValueGradient>> {
        let (Some(value_kernel), Some(gradient_kernel)) =
            (self.scalar_jit_kernel(), self.gradient_jit_kernel())
        else {
            return Err(RuntimeError::InvalidShape {
                index: self.graph.root().index(),
                message: "JIT gradient evaluation requires both scalar and gradient kernels".into(),
            });
        };
        let view = JitScalarKernel::prepare_cache(cache);
        let mut values = Vec::new();
        let mut real = Vec::new();
        let mut imag = Vec::new();
        value_kernel.evaluate_prepared(params, &view, start, end, &mut values)?;
        gradient_kernel.evaluate_prepared(params, &view, start, end, 0, &mut real)?;
        gradient_kernel.evaluate_prepared(params, &view, start, end, 1, &mut imag)?;
        let parameter_count = self.free_parameter_count();
        Ok(values
            .into_iter()
            .enumerate()
            .map(|(row, value)| ValueGradient {
                value,
                gradient: (0..parameter_count)
                    .map(|parameter| {
                        let index = row * parameter_count + parameter;
                        Complex64::new(real[index], imag[index])
                    })
                    .collect(),
            })
            .collect())
    }

    pub fn evaluate_batch(
        &self,
        params: &ParamValues,
        batch: &EventBatch,
    ) -> RuntimeResult<Vec<Complex64>> {
        let cache = self.cache_event_batch(batch)?;
        self.evaluate_cache(params, &cache)
    }

    pub fn evaluate_batch_with_gradient(
        &self,
        params: &ParamValues,
        batch: &EventBatch,
    ) -> RuntimeResult<Vec<ValueGradient>> {
        let cache = self.cache_event_batch(batch)?;
        self.evaluate_cache_with_gradient(params, &cache)
    }

    pub fn cache_dataset(&self, dataset: &Dataset) -> RuntimeResult<CpuCachedDataset> {
        self.cache_dataset_with_plan(dataset, dataset.read_plan())
    }

    fn cache_dataset_with_plan(
        &self,
        dataset: &Dataset,
        read_plan: laddu_data::io::ReadPlan,
    ) -> RuntimeResult<CpuCachedDataset> {
        let mut batches = Vec::new();
        let mut sum_weights = 0.0;
        for batch in dataset
            .batches_with_plan(read_plan)
            .map_err(|err| RuntimeError::Data(err.to_string()))?
        {
            let batch = batch.map_err(|err| RuntimeError::Data(err.to_string()))?;
            let cached = CpuCachedBatch {
                cache: self.cache_event_batch(&batch)?,
            };
            sum_weights += cached.sum_weights();
            batches.push(cached);
        }
        Ok(CpuCachedDataset {
            batches,
            sum_weights,
        })
    }

    pub fn prepare_dataset(
        &self,
        execution: &CpuExecution,
        dataset: &Dataset,
    ) -> RuntimeResult<CpuPreparedDataset> {
        let read_plan = execution.read_plan(dataset.read_plan());
        match dataset.cache_storage() {
            CacheStorage::Resident => {
                let local = self.cache_dataset_with_plan(dataset, read_plan);
                if !execution.all_succeeded(local.is_ok()) {
                    return local.and(Err(RuntimeError::DistributedPeerFailure));
                }
                let dataset = local?;
                let stats = PreparedDatasetStats {
                    local_events: dataset.len(),
                    global_events: execution.sum_usize(dataset.len()),
                    local_batches: dataset.batches().len(),
                    sum_weights: execution.sum_f64(dataset.sum_weights()),
                    resident_bytes: dataset.resident_bytes(),
                    storage: CacheStorage::Resident,
                };
                Ok(CpuPreparedDataset::Resident { dataset, stats })
            }
            CacheStorage::Streaming => {
                let local = (|| {
                    let mut local_events = 0;
                    let mut local_batches = 0;
                    let mut sum_weights = AccurateF64::zero();
                    for batch in dataset
                        .batches_with_plan(read_plan)
                        .map_err(|error| RuntimeError::Data(error.to_string()))?
                    {
                        let batch = batch.map_err(|error| RuntimeError::Data(error.to_string()))?;
                        local_events += batch.len();
                        local_batches += 1;
                        for row in 0..batch.len() {
                            sum_weights.push(batch.weights_at(row));
                        }
                    }
                    Ok::<_, RuntimeError>((local_events, local_batches, sum_weights.finish()))
                })();
                if !execution.all_succeeded(local.is_ok()) {
                    return local.and(Err(RuntimeError::DistributedPeerFailure));
                }
                let (local_events, local_batches, sum_weights) = local?;
                Ok(CpuPreparedDataset::Streaming {
                    dataset: dataset.clone(),
                    stats: PreparedDatasetStats {
                        local_events,
                        global_events: execution.sum_usize(local_events),
                        local_batches,
                        sum_weights: execution.sum_f64(sum_weights),
                        resident_bytes: 0,
                        storage: CacheStorage::Streaming,
                    },
                    read_plan,
                })
            }
        }
    }

    /// Execute a weighted reduction over a prepared dataset.
    pub fn reduce(
        &self,
        execution: &CpuExecution,
        params: &ParamValues,
        dataset: &CpuPreparedDataset,
        reduction: ReductionPlan,
    ) -> RuntimeResult<f64> {
        let local = match dataset {
            CpuPreparedDataset::Resident { dataset, .. } => {
                self.reduce_cached(execution, params, dataset, reduction)
            }
            CpuPreparedDataset::Streaming {
                dataset, read_plan, ..
            } => (|| {
                let mut total = AccurateF64::zero();
                for batch in dataset
                    .batches_with_plan(*read_plan)
                    .map_err(|error| RuntimeError::Data(error.to_string()))?
                {
                    let batch = batch.map_err(|error| RuntimeError::Data(error.to_string()))?;
                    let cached = CpuCachedDataset {
                        sum_weights: (0..batch.len()).map(|row| batch.weights_at(row)).sum(),
                        batches: vec![CpuCachedBatch {
                            cache: self.cache_event_batch(&batch)?,
                        }],
                    };
                    total.push(self.reduce_cached(execution, params, &cached, reduction)?);
                }
                Ok(total.finish())
            })(),
        };
        if !execution.all_succeeded(local.is_ok()) {
            return local.and(Err(RuntimeError::DistributedPeerFailure));
        }
        Ok(execution.sum_f64(local?))
    }

    /// Execute a weighted reduction and its free-parameter gradient.
    pub fn reduce_with_gradient(
        &self,
        execution: &CpuExecution,
        params: &ParamValues,
        dataset: &CpuPreparedDataset,
        reduction: ReductionPlan,
    ) -> RuntimeResult<ReductionEvaluation> {
        let (value, gradient) =
            self.try_reduce_weighted_with_gradient(execution, params, dataset, |value| {
                reduction
                    .apply(value)
                    .map(|output| output.into_parts())
                    .map_err(RuntimeError::from)
            })?;
        Ok(ReductionEvaluation { value, gradient })
    }

    fn try_reduce_weighted_with_gradient<E, F>(
        &self,
        execution: &CpuExecution,
        params: &ParamValues,
        dataset: &CpuPreparedDataset,
        transform: F,
    ) -> Result<(f64, Vec<f64>), E>
    where
        E: From<RuntimeError> + Send,
        F: Fn(Complex64) -> Result<(f64, f64), E> + Send + Sync,
    {
        let local = match dataset {
            CpuPreparedDataset::Resident { dataset, .. } => {
                self.try_reduce_weighted_with_gradient_cached(execution, params, dataset, transform)
            }
            CpuPreparedDataset::Streaming {
                dataset, read_plan, ..
            } => (|| {
                let mut value = AccurateF64::zero();
                let mut gradient = (0..self.free_parameter_count())
                    .map(|_| AccurateF64::zero())
                    .collect::<Vec<_>>();
                for batch in dataset
                    .batches_with_plan(*read_plan)
                    .map_err(|error| E::from(RuntimeError::Data(error.to_string())))?
                {
                    let batch =
                        batch.map_err(|error| E::from(RuntimeError::Data(error.to_string())))?;
                    let cached = CpuCachedDataset {
                        sum_weights: (0..batch.len()).map(|row| batch.weights_at(row)).sum(),
                        batches: vec![CpuCachedBatch {
                            cache: self.cache_event_batch(&batch)?,
                        }],
                    };
                    let (partial_value, partial_gradient) = self
                        .try_reduce_weighted_with_gradient_cached(
                            execution, params, &cached, &transform,
                        )?;
                    value.push(partial_value);
                    for (sum, partial) in gradient.iter_mut().zip(partial_gradient) {
                        sum.push(partial);
                    }
                }
                Ok::<_, E>((
                    value.finish(),
                    gradient.into_iter().map(AccurateF64::finish).collect(),
                ))
            })(),
        };
        if !execution.all_succeeded(local.is_ok()) {
            return local.and(Err(E::from(RuntimeError::DistributedPeerFailure)));
        }
        let (local_value, local_gradient) = local?;
        Ok((
            execution.sum_f64(local_value),
            execution.sum_slice(&local_gradient),
        ))
    }

    pub fn evaluate_cached_dataset(
        &self,
        params: &ParamValues,
        dataset: &CpuCachedDataset,
    ) -> RuntimeResult<Vec<Complex64>> {
        let total_len = dataset.batches.iter().map(CpuCachedBatch::len).sum();
        let mut out = Vec::with_capacity(total_len);
        let invariant = self.scalar_invariant_values(params)?;
        let mut workspace = ScalarEventWorkspace::default();
        for batch in &dataset.batches {
            self.check_batch_cache(batch.cache())?;
            for row in 0..batch.len() {
                out.push(self.evaluate_cache_row_prepared(
                    params,
                    batch.cache(),
                    row,
                    invariant.as_ref(),
                    &mut workspace,
                )?);
            }
        }
        Ok(out)
    }

    pub fn evaluate_cached_dataset_with_gradient(
        &self,
        params: &ParamValues,
        dataset: &CpuCachedDataset,
    ) -> RuntimeResult<Vec<ValueGradient>> {
        let total_len = dataset.batches.iter().map(CpuCachedBatch::len).sum();
        let mut out = Vec::with_capacity(total_len);
        for batch in &dataset.batches {
            out.extend(self.evaluate_cache_with_gradient(params, batch.cache())?);
        }
        Ok(out)
    }

    fn try_weighted_sum_cached<E, F>(
        &self,
        params: &ParamValues,
        dataset: &CpuCachedDataset,
        mut f: F,
    ) -> Result<f64, E>
    where
        E: From<RuntimeError>,
        F: FnMut(Complex64) -> Result<f64, E>,
    {
        let mut sum = 0.0;
        let invariant = self.scalar_invariant_values(params)?;
        let mut workspace = ScalarEventWorkspace::default();
        for batch in dataset.batches() {
            self.check_batch_cache(batch.cache())?;
            for row in 0..batch.len() {
                let value = self.evaluate_cache_row_prepared(
                    params,
                    batch.cache(),
                    row,
                    invariant.as_ref(),
                    &mut workspace,
                )?;
                sum += batch.weights()[row] * f(value)?;
            }
        }
        Ok(sum)
    }

    #[cfg(test)]
    fn weighted_sum_cached<F>(
        &self,
        params: &ParamValues,
        dataset: &CpuCachedDataset,
        mut f: F,
    ) -> RuntimeResult<f64>
    where
        F: FnMut(Complex64) -> f64,
    {
        self.try_weighted_sum_cached(params, dataset, |value| Ok(f(value)))
    }

    fn try_weighted_real_sum_with_gradient_cached<E, F>(
        &self,
        params: &ParamValues,
        dataset: &CpuCachedDataset,
        mut transform: F,
    ) -> Result<(f64, Vec<f64>), E>
    where
        E: From<RuntimeError>,
        F: FnMut(Complex64) -> Result<(f64, f64), E>,
    {
        #[cfg(feature = "jit")]
        if let (Some(value_kernel), Some(gradient_kernel)) =
            (self.scalar_jit_kernel(), self.gradient_jit_kernel())
        {
            return self.try_weighted_real_sum_with_jit_gradient_cached(
                params,
                dataset,
                transform,
                value_kernel,
                gradient_kernel,
            );
        }
        let mut total = RealGradientAccumulator::zero(self.free_parameter_count());
        for batch in dataset.batches() {
            self.check_batch_cache(batch.cache())?;
            for row in 0..batch.len() {
                let evaluation =
                    self.evaluate_cache_row_with_gradient_unchecked(params, batch.cache(), row)?;
                let (value, derivative) = transform(evaluation.value())?;
                total.push(
                    batch.weights()[row],
                    value,
                    derivative,
                    evaluation.gradient(),
                );
            }
        }
        Ok(total.finish())
    }

    #[cfg(feature = "jit")]
    fn try_weighted_real_sum_with_jit_gradient_cached<E, F>(
        &self,
        params: &ParamValues,
        dataset: &CpuCachedDataset,
        mut transform: F,
        value_kernel: &JitScalarKernel,
        gradient_kernel: &JitGradientKernel,
    ) -> Result<(f64, Vec<f64>), E>
    where
        E: From<RuntimeError>,
        F: FnMut(Complex64) -> Result<(f64, f64), E>,
    {
        let mut total = RealGradientAccumulator::zero(self.free_parameter_count());
        let mut values = Vec::new();
        let mut tangents = Vec::new();
        let mut derivatives = Vec::new();
        for batch in dataset.batches() {
            self.check_batch_cache(batch.cache())?;
            let cache = JitScalarKernel::prepare_cache(batch.cache());
            for block in 0..batch.len().div_ceil(SCALAR_BLOCK_SIZE) {
                let start = block * SCALAR_BLOCK_SIZE;
                let end = (start + SCALAR_BLOCK_SIZE).min(batch.len());
                value_kernel.evaluate_prepared(params, &cache, start, end, &mut values)?;
                derivatives.clear();
                derivatives.reserve(values.len());
                for (lane, value) in values.iter().copied().enumerate() {
                    let (value, derivative) = transform(value)?;
                    let weight = batch.weights()[start + lane];
                    total.value.push(weight * value);
                    derivatives.push(weight * derivative);
                }
                gradient_kernel.evaluate_prepared(params, &cache, start, end, 0, &mut tangents)?;
                for (lane, factor) in derivatives.iter().enumerate() {
                    for free_index in 0..self.free_parameter_count() {
                        total.gradient[free_index].push(
                            factor * tangents[lane * self.free_parameter_count() + free_index],
                        );
                    }
                }
            }
        }
        Ok(total.finish())
    }

    #[cfg(test)]
    fn try_weighted_complex_sum_cached<E, F>(
        &self,
        params: &ParamValues,
        dataset: &CpuCachedDataset,
        mut f: F,
    ) -> Result<Complex64, E>
    where
        E: From<RuntimeError>,
        F: FnMut(Complex64) -> Result<Complex64, E>,
    {
        let mut sum = Complex64::default();
        let invariant = self.scalar_invariant_values(params)?;
        let mut workspace = ScalarEventWorkspace::default();
        for batch in dataset.batches() {
            self.check_batch_cache(batch.cache())?;
            for row in 0..batch.len() {
                let value = self.evaluate_cache_row_prepared(
                    params,
                    batch.cache(),
                    row,
                    invariant.as_ref(),
                    &mut workspace,
                )?;
                sum += f(value)? * batch.weights()[row];
            }
        }
        Ok(sum)
    }

    #[cfg(test)]
    fn weighted_complex_sum_cached<F>(
        &self,
        params: &ParamValues,
        dataset: &CpuCachedDataset,
        mut f: F,
    ) -> RuntimeResult<Complex64>
    where
        F: FnMut(Complex64) -> Complex64,
    {
        self.try_weighted_complex_sum_cached(params, dataset, |value| Ok(f(value)))
    }

    fn reduce_cached(
        &self,
        execution: &CpuExecution,
        params: &ParamValues,
        dataset: &CpuCachedDataset,
        reduction: ReductionPlan,
    ) -> RuntimeResult<f64> {
        if execution.is_parallel() {
            execution.install(|| {
                self.par_try_weighted_sum_cached(params, dataset, |value| {
                    reduction
                        .apply(value)
                        .map(|output| output.value())
                        .map_err(RuntimeError::from)
                })
            })
        } else {
            self.try_weighted_sum_cached(params, dataset, |value| {
                reduction
                    .apply(value)
                    .map(|output| output.value())
                    .map_err(RuntimeError::from)
            })
        }
    }

    fn try_reduce_weighted_with_gradient_cached<E, F>(
        &self,
        execution: &CpuExecution,
        params: &ParamValues,
        dataset: &CpuCachedDataset,
        transform: F,
    ) -> Result<(f64, Vec<f64>), E>
    where
        E: From<RuntimeError> + Send,
        F: Fn(Complex64) -> Result<(f64, f64), E> + Send + Sync,
    {
        if execution.is_parallel() {
            execution.install(|| {
                self.par_try_weighted_real_sum_with_gradient_cached(params, dataset, transform)
            })
        } else {
            self.try_weighted_real_sum_with_gradient_cached(params, dataset, transform)
        }
    }

    pub(crate) fn par_try_weighted_sum_cached<E, F>(
        &self,
        params: &ParamValues,
        dataset: &CpuCachedDataset,
        f: F,
    ) -> Result<f64, E>
    where
        E: From<RuntimeError> + Send,
        F: Fn(Complex64) -> Result<f64, E> + Send + Sync,
    {
        let mut total = AccurateF64::zero();
        let invariant = self.scalar_invariant_values(params)?;
        for batch in dataset.batches() {
            self.check_batch_cache(batch.cache())?;
            #[cfg(feature = "jit")]
            let jit_cache = self
                .scalar_jit_kernel()
                .map(|_| JitScalarKernel::prepare_cache(batch.cache()));
            let n_blocks = batch.len().div_ceil(SCALAR_BLOCK_SIZE);
            let partial = (0..n_blocks)
                .into_par_iter()
                .try_fold(
                    || {
                        (
                            AccurateF64::zero(),
                            ScalarEventWorkspace::default(),
                            Vec::new(),
                        )
                    },
                    |(mut acc, mut workspace, mut output), block| {
                        let start = block * SCALAR_BLOCK_SIZE;
                        let end = (start + SCALAR_BLOCK_SIZE).min(batch.len());
                        self.evaluate_cache_block_prepared(
                            params,
                            batch.cache(),
                            start,
                            end,
                            invariant.as_ref(),
                            &mut workspace,
                            &mut output,
                            #[cfg(feature = "jit")]
                            jit_cache.as_ref(),
                        )?;
                        for (lane, value) in output.iter().copied().enumerate() {
                            acc.push(batch.weights()[start + lane] * f(value)?);
                        }
                        Ok::<_, E>((acc, workspace, output))
                    },
                )
                .try_reduce(
                    || {
                        (
                            AccurateF64::zero(),
                            ScalarEventWorkspace::default(),
                            Vec::new(),
                        )
                    },
                    |(mut lhs, workspace, output), (rhs, _, _)| {
                        lhs.merge(rhs);
                        Ok::<_, E>((lhs, workspace, output))
                    },
                )?;
            total.merge(partial.0);
        }
        Ok(total.finish())
    }

    #[cfg(test)]
    pub(crate) fn par_weighted_sum_cached<F>(
        &self,
        params: &ParamValues,
        dataset: &CpuCachedDataset,
        f: F,
    ) -> RuntimeResult<f64>
    where
        F: Fn(Complex64) -> f64 + Send + Sync,
    {
        self.par_try_weighted_sum_cached(params, dataset, |value| Ok(f(value)))
    }

    pub(crate) fn par_try_weighted_real_sum_with_gradient_cached<E, F>(
        &self,
        params: &ParamValues,
        dataset: &CpuCachedDataset,
        transform: F,
    ) -> Result<(f64, Vec<f64>), E>
    where
        E: From<RuntimeError> + Send,
        F: Fn(Complex64) -> Result<(f64, f64), E> + Send + Sync,
    {
        #[cfg(feature = "jit")]
        if let (Some(value_kernel), Some(gradient_kernel)) =
            (self.scalar_jit_kernel(), self.gradient_jit_kernel())
        {
            return self.par_try_weighted_real_sum_with_jit_gradient_cached(
                params,
                dataset,
                transform,
                value_kernel,
                gradient_kernel,
            );
        }
        let mut total = RealGradientAccumulator::zero(self.free_parameter_count());
        for batch in dataset.batches() {
            self.check_batch_cache(batch.cache())?;
            let partial = (0..batch.len())
                .into_par_iter()
                .try_fold(
                    || RealGradientAccumulator::zero(self.free_parameter_count()),
                    |mut accumulator, row| {
                        let evaluation = self.evaluate_cache_row_with_gradient_unchecked(
                            params,
                            batch.cache(),
                            row,
                        )?;
                        let (value, derivative) = transform(evaluation.value())?;
                        accumulator.push(
                            batch.weights()[row],
                            value,
                            derivative,
                            evaluation.gradient(),
                        );
                        Ok::<_, E>(accumulator)
                    },
                )
                .try_reduce(
                    || RealGradientAccumulator::zero(self.free_parameter_count()),
                    |mut lhs, rhs| {
                        lhs.merge(rhs);
                        Ok::<_, E>(lhs)
                    },
                )?;
            total.merge(partial);
        }
        Ok(total.finish())
    }

    #[cfg(feature = "jit")]
    fn par_try_weighted_real_sum_with_jit_gradient_cached<E, F>(
        &self,
        params: &ParamValues,
        dataset: &CpuCachedDataset,
        transform: F,
        value_kernel: &JitScalarKernel,
        gradient_kernel: &JitGradientKernel,
    ) -> Result<(f64, Vec<f64>), E>
    where
        E: From<RuntimeError> + Send,
        F: Fn(Complex64) -> Result<(f64, f64), E> + Send + Sync,
    {
        let mut total = RealGradientAccumulator::zero(self.free_parameter_count());
        for batch in dataset.batches() {
            self.check_batch_cache(batch.cache())?;
            let cache = JitScalarKernel::prepare_cache(batch.cache());
            let n_blocks = batch.len().div_ceil(SCALAR_BLOCK_SIZE);
            let partial = (0..n_blocks)
                .into_par_iter()
                .try_fold(
                    || {
                        (
                            RealGradientAccumulator::zero(self.free_parameter_count()),
                            Vec::new(),
                            Vec::new(),
                            Vec::new(),
                        )
                    },
                    |(mut accumulator, mut values, mut tangents, mut derivatives), block| {
                        let start = block * SCALAR_BLOCK_SIZE;
                        let end = (start + SCALAR_BLOCK_SIZE).min(batch.len());
                        value_kernel.evaluate_prepared(params, &cache, start, end, &mut values)?;
                        derivatives.clear();
                        derivatives.reserve(values.len());
                        for (lane, value) in values.iter().copied().enumerate() {
                            let (value, derivative) = transform(value)?;
                            let weight = batch.weights()[start + lane];
                            accumulator.value.push(weight * value);
                            derivatives.push(weight * derivative);
                        }
                        gradient_kernel.evaluate_prepared(
                            params,
                            &cache,
                            start,
                            end,
                            0,
                            &mut tangents,
                        )?;
                        for (lane, factor) in derivatives.iter().enumerate() {
                            for free_index in 0..self.free_parameter_count() {
                                accumulator.gradient[free_index].push(
                                    factor
                                        * tangents[lane * self.free_parameter_count() + free_index],
                                );
                            }
                        }
                        Ok::<_, E>((accumulator, values, tangents, derivatives))
                    },
                )
                .try_reduce(
                    || {
                        (
                            RealGradientAccumulator::zero(self.free_parameter_count()),
                            Vec::new(),
                            Vec::new(),
                            Vec::new(),
                        )
                    },
                    |(mut lhs, values, tangents, derivatives), (rhs, _, _, _)| {
                        lhs.merge(rhs);
                        Ok::<_, E>((lhs, values, tangents, derivatives))
                    },
                )?;
            total.merge(partial.0);
        }
        Ok(total.finish())
    }

    #[cfg(test)]
    pub(crate) fn par_try_weighted_complex_sum_cached<E, F>(
        &self,
        params: &ParamValues,
        dataset: &CpuCachedDataset,
        f: F,
    ) -> Result<Complex64, E>
    where
        E: From<RuntimeError> + Send,
        F: Fn(Complex64) -> Result<Complex64, E> + Send + Sync,
    {
        let mut total = AccurateComplex64::zero();
        let invariant = self.scalar_invariant_values(params)?;
        for batch in dataset.batches() {
            self.check_batch_cache(batch.cache())?;
            #[cfg(feature = "jit")]
            let jit_cache = self
                .scalar_jit_kernel()
                .map(|_| JitScalarKernel::prepare_cache(batch.cache()));
            let n_blocks = batch.len().div_ceil(SCALAR_BLOCK_SIZE);
            let partial = (0..n_blocks)
                .into_par_iter()
                .try_fold(
                    || {
                        (
                            AccurateComplex64::zero(),
                            ScalarEventWorkspace::default(),
                            Vec::new(),
                        )
                    },
                    |(mut acc, mut workspace, mut output), block| {
                        let start = block * SCALAR_BLOCK_SIZE;
                        let end = (start + SCALAR_BLOCK_SIZE).min(batch.len());
                        self.evaluate_cache_block_prepared(
                            params,
                            batch.cache(),
                            start,
                            end,
                            invariant.as_ref(),
                            &mut workspace,
                            &mut output,
                            #[cfg(feature = "jit")]
                            jit_cache.as_ref(),
                        )?;
                        for (lane, value) in output.iter().copied().enumerate() {
                            acc.push(f(value)? * batch.weights()[start + lane]);
                        }
                        Ok::<_, E>((acc, workspace, output))
                    },
                )
                .try_reduce(
                    || {
                        (
                            AccurateComplex64::zero(),
                            ScalarEventWorkspace::default(),
                            Vec::new(),
                        )
                    },
                    |(mut lhs, workspace, output), (rhs, _, _)| {
                        lhs.merge(rhs);
                        Ok::<_, E>((lhs, workspace, output))
                    },
                )?;
            total.merge(partial.0);
        }
        Ok(total.finish())
    }

    #[cfg(test)]
    pub(crate) fn par_weighted_complex_sum_cached<F>(
        &self,
        params: &ParamValues,
        dataset: &CpuCachedDataset,
        f: F,
    ) -> RuntimeResult<Complex64>
    where
        F: Fn(Complex64) -> Complex64 + Send + Sync,
    {
        self.par_try_weighted_complex_sum_cached(params, dataset, |value| Ok(f(value)))
    }

    fn evaluate_inner(
        &self,
        params: &ParamValues,
        event: Option<&dyn EventLookup>,
    ) -> RuntimeResult<Complex64> {
        #[cfg(feature = "jit")]
        if event.is_none()
            && let Some(kernel) = self.scalar_jit_kernel()
        {
            if params.as_slice().len() != self.params.len() {
                return Err(RuntimeError::Parameter(format!(
                    "expected {} parameter values, got {}",
                    self.params.len(),
                    params.as_slice().len()
                )));
            }
            return kernel.evaluate_invariant(params);
        }
        let values = self.evaluate_values(params, event)?;
        scalar_at(&values, self.graph.root().index())
    }

    fn value_gradient(
        &self,
        values: Vec<Value>,
        cached_factors: Option<(&CpuBatchCache, usize)>,
    ) -> RuntimeResult<ValueGradient> {
        let value = if cached_factors.is_some() {
            self.cached_scalar_at(&values, self.graph.root())?
        } else {
            scalar_at(&values, self.graph.root().index())?
        };
        let gradient = DerivativeWorkspace::new(self, &values, cached_factors).gradient()?;
        Ok(ValueGradient { value, gradient })
    }

    fn solve_primal(
        &self,
        matrix_id: ExprId,
        dimension: usize,
        matrix: &[Complex64],
        rhs: &DVector<Complex64>,
        node_index: usize,
        cached: Option<(&CpuBatchCache, usize)>,
    ) -> RuntimeResult<DVector<Complex64>> {
        let solution = if let (Some(slot), Some((cache, row))) =
            (self.factor_matrix_slots[matrix_id.index()], cached)
        {
            cache.factor(slot, row)?.solve(rhs)
        } else if let Some(slot) = self.constant_factor_slots[matrix_id.index()] {
            self.constant_factors[slot]
                .get_or_init(|| DMatrix::from_row_slice(dimension, dimension, matrix).lu())
                .solve(rhs)
        } else {
            DMatrix::from_row_slice(dimension, dimension, matrix)
                .lu()
                .solve(rhs)
        };
        solution.ok_or(RuntimeError::SingularMatrix(node_index))
    }

    fn event_columns(&self, schema: &Schema) -> RuntimeResult<Vec<Option<EventColumn>>> {
        self.graph
            .nodes()
            .iter()
            .map(|node| {
                if let ExprNode::EventScalar(name) = node {
                    Ok(Some(EventColumn::Scalar(
                        schema
                            .scalar_index(name)
                            .ok_or_else(|| RuntimeError::MissingEventColumn(name.to_string()))?,
                    )))
                } else if let ExprNode::EventP4Component { name, component } = node {
                    Ok(Some(EventColumn::P4Component {
                        col: schema
                            .p4_index(name)
                            .ok_or_else(|| RuntimeError::MissingEventColumn(name.to_string()))?,
                        component: *component,
                    }))
                } else {
                    Ok(None)
                }
            })
            .collect()
    }

    fn evaluate_cache_values_for_row(
        &self,
        batch: &EventBatch,
        row: usize,
        event_columns: &[Option<EventColumn>],
    ) -> RuntimeResult<Vec<Option<Value>>> {
        let mut values = vec![None; self.graph.nodes().len()];

        for id in &self.cache_materialization_nodes {
            let index = id.index();
            let node = &self.graph.nodes()[index];
            let value = match node {
                ExprNode::RealConst(value) => Value::Scalar(Complex64::from(*value)),
                ExprNode::ComplexConst(value) => Value::Scalar(*value),
                ExprNode::EventScalar(name) => {
                    let col = event_columns[index]
                        .ok_or_else(|| RuntimeError::MissingEventColumn(name.to_string()))?;
                    let EventColumn::Scalar(col) = col else {
                        return Err(RuntimeError::MissingEventColumn(name.to_string()));
                    };
                    Value::Scalar(Complex64::from(batch.scalar_at(col, row)))
                }
                ExprNode::EventP4Component { name, component } => {
                    let col = event_columns[index]
                        .ok_or_else(|| RuntimeError::MissingEventColumn(name.to_string()))?;
                    let EventColumn::P4Component {
                        col,
                        component: actual,
                    } = col
                    else {
                        return Err(RuntimeError::MissingEventColumn(name.to_string()));
                    };
                    debug_assert_eq!(actual, *component);
                    let p4 = batch.p4_at(col, row);
                    let value = match component {
                        P4Component::Px => p4.x,
                        P4Component::Py => p4.y,
                        P4Component::Pz => p4.z,
                        P4Component::E => p4.t,
                    };
                    Value::Scalar(Complex64::from(value))
                }
                ExprNode::Unary { op, input } => {
                    let input = scalar_at_optional(&values, input.index())?;
                    Value::Scalar(eval_unary(*op, input))
                }
                ExprNode::Binary { op, lhs, rhs } => {
                    let lhs = scalar_at_optional(&values, lhs.index())?;
                    let rhs = scalar_at_optional(&values, rhs.index())?;
                    Value::Scalar(eval_binary(*op, lhs, rhs))
                }
                ExprNode::NaryAdd { terms } => {
                    let mut sum = Complex64::ZERO;
                    for term in terms {
                        sum += scalar_at_optional(&values, term.index())?;
                    }
                    Value::Scalar(sum)
                }
                ExprNode::NaryMul { factors } => {
                    let mut product = Complex64::ONE;
                    for factor in factors {
                        product *= scalar_at_optional(&values, factor.index())?;
                    }
                    Value::Scalar(product)
                }
                ExprNode::Complex { re, im } => {
                    let re = scalar_at_optional(&values, re.index())?;
                    let im = scalar_at_optional(&values, im.index())?;
                    Value::Scalar(Complex64::new(re.re, im.re))
                }
                ExprNode::Vector { elements } => Value::Vector(
                    elements
                        .iter()
                        .map(|id| scalar_at_optional(&values, id.index()))
                        .collect::<RuntimeResult<_>>()?,
                ),
                ExprNode::Matrix {
                    rows,
                    cols,
                    elements,
                } => {
                    if elements.len() != rows * cols {
                        return Err(RuntimeError::InvalidShape {
                            index,
                            message: format!(
                                "matrix has {} elements for shape {rows}x{cols}",
                                elements.len()
                            ),
                        });
                    }
                    Value::Matrix {
                        rows: *rows,
                        cols: *cols,
                        values: elements
                            .iter()
                            .map(|id| scalar_at_optional(&values, id.index()))
                            .collect::<RuntimeResult<_>>()?,
                    }
                }
                ExprNode::Component { input, index: i } => {
                    let vector = vector_at_optional(&values, input.index())?;
                    Value::Scalar(*vector.get(*i).ok_or_else(|| RuntimeError::InvalidShape {
                        index,
                        message: format!(
                            "component index {i} out of bounds for len {}",
                            vector.len()
                        ),
                    })?)
                }
                ExprNode::MatrixElement { input, row, col } => {
                    let (rows, cols, matrix) = matrix_at_optional(&values, input.index())?;
                    if *row >= rows || *col >= cols {
                        return Err(RuntimeError::InvalidShape {
                            index,
                            message: format!(
                                "matrix element ({row}, {col}) out of bounds for shape {rows}x{cols}"
                            ),
                        });
                    }
                    Value::Scalar(matrix[row * cols + col])
                }
                ExprNode::MatMul { lhs, rhs } => {
                    let (lhs_rows, lhs_cols, lhs) = matrix_at_optional(&values, lhs.index())?;
                    let (rhs_rows, rhs_cols, rhs) = matrix_at_optional(&values, rhs.index())?;
                    if lhs_cols != rhs_rows {
                        return Err(RuntimeError::InvalidShape {
                            index,
                            message: format!(
                                "cannot multiply {lhs_rows}x{lhs_cols} by {rhs_rows}x{rhs_cols}"
                            ),
                        });
                    }
                    let lhs = DMatrix::from_row_slice(lhs_rows, lhs_cols, lhs);
                    let rhs = DMatrix::from_row_slice(rhs_rows, rhs_cols, rhs);
                    let out = lhs * rhs;
                    Value::Matrix {
                        rows: out.nrows(),
                        cols: out.ncols(),
                        values: matrix_values_row_major(&out),
                    }
                }
                ExprNode::MatVec { matrix, vector } => {
                    let (rows, cols, matrix) = matrix_at_optional(&values, matrix.index())?;
                    let vector = vector_at_optional(&values, vector.index())?;
                    if cols != vector.len() {
                        return Err(RuntimeError::InvalidShape {
                            index,
                            message: format!(
                                "cannot multiply {rows}x{cols} matrix by len {} vector",
                                vector.len()
                            ),
                        });
                    }
                    let matrix = DMatrix::from_row_slice(rows, cols, matrix);
                    let vector = DVector::from_row_slice(vector);
                    Value::Vector((matrix * vector).iter().copied().collect())
                }
                ExprNode::Dot { lhs, rhs } => {
                    let lhs = vector_at_optional(&values, lhs.index())?;
                    let rhs = vector_at_optional(&values, rhs.index())?;
                    if lhs.len() != rhs.len() {
                        return Err(RuntimeError::InvalidShape {
                            index,
                            message: format!(
                                "cannot dot len {} vector with len {} vector",
                                lhs.len(),
                                rhs.len()
                            ),
                        });
                    }
                    Value::Scalar(lhs.iter().zip(rhs).map(|(lhs, rhs)| lhs * rhs).sum())
                }
                ExprNode::Solve { matrix, rhs } => {
                    let matrix_id = *matrix;
                    let (rows, cols, matrix) = matrix_at_optional(&values, matrix_id.index())?;
                    let rhs = vector_at_optional(&values, rhs.index())?;
                    if rows != cols || rows != rhs.len() {
                        return Err(RuntimeError::InvalidShape {
                            index,
                            message: format!(
                                "cannot solve {rows}x{cols} matrix against len {} vector",
                                rhs.len()
                            ),
                        });
                    }
                    let rhs = DVector::from_row_slice(rhs);
                    let solution = self.solve_primal(matrix_id, rows, matrix, &rhs, index, None)?;
                    Value::Vector(solution.iter().copied().collect())
                }
                ExprNode::ScalarParam(_) => {
                    return Err(RuntimeError::InvalidShape {
                        index,
                        message: "parameter-dependent node cannot be part of an event cache".into(),
                    });
                }
            };
            values[index] = Some(value);
        }

        Ok(values)
    }

    fn evaluate_values(
        &self,
        params: &ParamValues,
        event: Option<&dyn EventLookup>,
    ) -> RuntimeResult<Vec<Value>> {
        let mut values = Vec::with_capacity(self.graph.nodes().len());

        for (index, node) in self.graph.nodes().iter().enumerate() {
            let value = match node {
                ExprNode::RealConst(value) => Value::Scalar(Complex64::from(*value)),
                ExprNode::ComplexConst(value) => Value::Scalar(*value),
                ExprNode::ScalarParam(_) => {
                    Value::Scalar(Complex64::from(self.parameter_value(params, index)?))
                }
                ExprNode::EventScalar(name) => {
                    let Some(event) = event else {
                        return Err(RuntimeError::MissingEventScalar(name.to_string()));
                    };
                    Value::Scalar(
                        event
                            .scalar(name)
                            .ok_or_else(|| RuntimeError::MissingEventScalar(name.to_string()))?,
                    )
                }
                ExprNode::EventP4Component { name, component } => {
                    let Some(event) = event else {
                        return Err(RuntimeError::MissingEventScalar(format!(
                            "{name}.{}",
                            component.label()
                        )));
                    };
                    Value::Scalar(event.p4_component(name, *component).ok_or_else(|| {
                        RuntimeError::MissingEventScalar(format!("{name}.{}", component.label()))
                    })?)
                }
                ExprNode::Unary { op, input } => {
                    let input = scalar_at(&values, input.index())?;
                    Value::Scalar(eval_unary(*op, input))
                }
                ExprNode::Binary { op, lhs, rhs } => {
                    let lhs = scalar_at(&values, lhs.index())?;
                    let rhs = scalar_at(&values, rhs.index())?;
                    Value::Scalar(eval_binary(*op, lhs, rhs))
                }
                ExprNode::NaryAdd { terms } => {
                    let mut sum = Complex64::ZERO;
                    for term in terms {
                        sum += scalar_at(&values, term.index())?;
                    }
                    Value::Scalar(sum)
                }
                ExprNode::NaryMul { factors } => {
                    let mut product = Complex64::ONE;
                    for factor in factors {
                        product *= scalar_at(&values, factor.index())?;
                    }
                    Value::Scalar(product)
                }
                ExprNode::Complex { re, im } => {
                    let re = scalar_at(&values, re.index())?;
                    let im = scalar_at(&values, im.index())?;
                    Value::Scalar(Complex64::new(re.re, im.re))
                }
                ExprNode::Vector { elements } => Value::Vector(
                    elements
                        .iter()
                        .map(|id| scalar_at(&values, id.index()))
                        .collect::<RuntimeResult<_>>()?,
                ),
                ExprNode::Matrix {
                    rows,
                    cols,
                    elements,
                } => {
                    if elements.len() != rows * cols {
                        return Err(RuntimeError::InvalidShape {
                            index,
                            message: format!(
                                "matrix has {} elements for shape {rows}x{cols}",
                                elements.len()
                            ),
                        });
                    }
                    Value::Matrix {
                        rows: *rows,
                        cols: *cols,
                        values: elements
                            .iter()
                            .map(|id| scalar_at(&values, id.index()))
                            .collect::<RuntimeResult<_>>()?,
                    }
                }
                ExprNode::Component { input, index: i } => {
                    let vector = vector_at(&values, input.index())?;
                    Value::Scalar(*vector.get(*i).ok_or_else(|| RuntimeError::InvalidShape {
                        index,
                        message: format!(
                            "component index {i} out of bounds for len {}",
                            vector.len()
                        ),
                    })?)
                }
                ExprNode::MatrixElement { input, row, col } => {
                    let (rows, cols, matrix) = matrix_at(&values, input.index())?;
                    if *row >= rows || *col >= cols {
                        return Err(RuntimeError::InvalidShape {
                            index,
                            message: format!(
                                "matrix element ({row}, {col}) out of bounds for shape {rows}x{cols}"
                            ),
                        });
                    }
                    Value::Scalar(matrix[row * cols + col])
                }
                ExprNode::MatMul { lhs, rhs } => {
                    let (lhs_rows, lhs_cols, lhs) = matrix_at(&values, lhs.index())?;
                    let (rhs_rows, rhs_cols, rhs) = matrix_at(&values, rhs.index())?;
                    if lhs_cols != rhs_rows {
                        return Err(RuntimeError::InvalidShape {
                            index,
                            message: format!(
                                "cannot multiply {lhs_rows}x{lhs_cols} by {rhs_rows}x{rhs_cols}"
                            ),
                        });
                    }
                    let lhs = DMatrix::from_row_slice(lhs_rows, lhs_cols, lhs);
                    let rhs = DMatrix::from_row_slice(rhs_rows, rhs_cols, rhs);
                    let out = lhs * rhs;
                    Value::Matrix {
                        rows: out.nrows(),
                        cols: out.ncols(),
                        values: matrix_values_row_major(&out),
                    }
                }
                ExprNode::MatVec { matrix, vector } => {
                    let (rows, cols, matrix) = matrix_at(&values, matrix.index())?;
                    let vector = vector_at(&values, vector.index())?;
                    if cols != vector.len() {
                        return Err(RuntimeError::InvalidShape {
                            index,
                            message: format!(
                                "cannot multiply {rows}x{cols} matrix by len {} vector",
                                vector.len()
                            ),
                        });
                    }
                    let matrix = DMatrix::from_row_slice(rows, cols, matrix);
                    let vector = DVector::from_row_slice(vector);
                    Value::Vector((matrix * vector).iter().copied().collect())
                }
                ExprNode::Dot { lhs, rhs } => {
                    let lhs = vector_at(&values, lhs.index())?;
                    let rhs = vector_at(&values, rhs.index())?;
                    if lhs.len() != rhs.len() {
                        return Err(RuntimeError::InvalidShape {
                            index,
                            message: format!(
                                "cannot dot len {} vector with len {} vector",
                                lhs.len(),
                                rhs.len()
                            ),
                        });
                    }
                    Value::Scalar(lhs.iter().zip(rhs).map(|(lhs, rhs)| lhs * rhs).sum())
                }
                ExprNode::Solve { matrix, rhs } => {
                    let matrix_id = *matrix;
                    let (rows, cols, matrix) = matrix_at(&values, matrix_id.index())?;
                    let rhs = vector_at(&values, rhs.index())?;
                    if rows != cols || rows != rhs.len() {
                        return Err(RuntimeError::InvalidShape {
                            index,
                            message: format!(
                                "cannot solve {rows}x{cols} matrix against len {} vector",
                                rhs.len()
                            ),
                        });
                    }
                    let rhs = DVector::from_row_slice(rhs);
                    let solution = self.solve_primal(matrix_id, rows, matrix, &rhs, index, None)?;
                    Value::Vector(solution.iter().copied().collect())
                }
            };
            values.push(value);
        }

        Ok(values)
    }

    fn evaluate_values_from_cache(
        &self,
        params: &ParamValues,
        cache: &CpuBatchCache,
        row: usize,
    ) -> RuntimeResult<Vec<Value>> {
        let mut values = Vec::with_capacity(self.cached_evaluation_nodes.len());

        for id in &self.cached_evaluation_nodes {
            let index = id.index();
            let node = &self.graph.nodes()[index];
            if let Some(slot) = self.cache_slots[index] {
                values.push(cache.value(slot, row)?);
                continue;
            }
            let value = match node {
                ExprNode::RealConst(value) => Value::Scalar(Complex64::from(*value)),
                ExprNode::ComplexConst(value) => Value::Scalar(*value),
                ExprNode::ScalarParam(_) => {
                    Value::Scalar(Complex64::from(self.parameter_value(params, index)?))
                }
                ExprNode::EventScalar(name) => {
                    return Err(RuntimeError::MissingEventScalar(name.to_string()));
                }
                ExprNode::EventP4Component { name, component } => {
                    return Err(RuntimeError::MissingEventScalar(format!(
                        "{name}.{}",
                        component.label()
                    )));
                }
                ExprNode::Unary { op, input } => {
                    let input = self.cached_scalar_at(&values, *input)?;
                    Value::Scalar(eval_unary(*op, input))
                }
                ExprNode::Binary { op, lhs, rhs } => {
                    let lhs = self.cached_scalar_at(&values, *lhs)?;
                    let rhs = self.cached_scalar_at(&values, *rhs)?;
                    Value::Scalar(eval_binary(*op, lhs, rhs))
                }
                ExprNode::NaryAdd { terms } => {
                    let mut sum = Complex64::ZERO;
                    for term in terms {
                        sum += self.cached_scalar_at(&values, *term)?;
                    }
                    Value::Scalar(sum)
                }
                ExprNode::NaryMul { factors } => {
                    let mut product = Complex64::ONE;
                    for factor in factors {
                        product *= self.cached_scalar_at(&values, *factor)?;
                    }
                    Value::Scalar(product)
                }
                ExprNode::Complex { re, im } => {
                    let re = self.cached_scalar_at(&values, *re)?;
                    let im = self.cached_scalar_at(&values, *im)?;
                    Value::Scalar(Complex64::new(re.re, im.re))
                }
                ExprNode::Vector { elements } => Value::Vector(
                    elements
                        .iter()
                        .map(|id| self.cached_scalar_at(&values, *id))
                        .collect::<RuntimeResult<_>>()?,
                ),
                ExprNode::Matrix {
                    rows,
                    cols,
                    elements,
                } => {
                    if elements.len() != rows * cols {
                        return Err(RuntimeError::InvalidShape {
                            index,
                            message: format!(
                                "matrix has {} elements for shape {rows}x{cols}",
                                elements.len()
                            ),
                        });
                    }
                    Value::Matrix {
                        rows: *rows,
                        cols: *cols,
                        values: elements
                            .iter()
                            .map(|id| self.cached_scalar_at(&values, *id))
                            .collect::<RuntimeResult<_>>()?,
                    }
                }
                ExprNode::Component { input, index: i } => {
                    if let Some(plan) = self.solve_components[index] {
                        let inverse_row = cache.solve_row(plan.row_slot, row)?;
                        if inverse_row.len() != plan.dimension {
                            return Err(RuntimeError::InvalidShape {
                                index,
                                message: format!(
                                    "specialized solve expected row len {}, got {}",
                                    plan.dimension,
                                    inverse_row.len()
                                ),
                            });
                        }
                        if let Some(elements) = &self.solve_rhs_elements[plan.rhs.index()] {
                            if elements.len() != plan.dimension {
                                return Err(RuntimeError::InvalidShape {
                                    index,
                                    message: format!(
                                        "specialized solve expected {} RHS elements, got {}",
                                        plan.dimension,
                                        elements.len()
                                    ),
                                });
                            }
                            Value::Scalar(
                                inverse_row
                                    .iter()
                                    .zip(elements)
                                    .map(|(lhs, rhs)| {
                                        Ok(lhs * self.cached_scalar_at(&values, *rhs)?)
                                    })
                                    .sum::<RuntimeResult<Complex64>>()?,
                            )
                        } else {
                            let rhs = self.cached_vector_at(&values, plan.rhs)?;
                            if rhs.len() != plan.dimension {
                                return Err(RuntimeError::InvalidShape {
                                    index,
                                    message: format!(
                                        "specialized solve expected RHS len {}, got {}",
                                        plan.dimension,
                                        rhs.len()
                                    ),
                                });
                            }
                            Value::Scalar(
                                inverse_row
                                    .iter()
                                    .zip(rhs)
                                    .map(|(lhs, rhs)| lhs * rhs)
                                    .sum(),
                            )
                        }
                    } else {
                        let vector = self.cached_vector_at(&values, *input)?;
                        Value::Scalar(*vector.get(*i).ok_or_else(|| {
                            RuntimeError::InvalidShape {
                                index,
                                message: format!(
                                    "component index {i} out of bounds for len {}",
                                    vector.len()
                                ),
                            }
                        })?)
                    }
                }
                ExprNode::MatrixElement { input, row, col } => {
                    let (rows, cols, matrix) = self.cached_matrix_at(&values, *input)?;
                    if *row >= rows || *col >= cols {
                        return Err(RuntimeError::InvalidShape {
                            index,
                            message: format!(
                                "matrix element ({row}, {col}) out of bounds for shape {rows}x{cols}"
                            ),
                        });
                    }
                    Value::Scalar(matrix[row * cols + col])
                }
                ExprNode::MatMul { lhs, rhs } => {
                    let (lhs_rows, lhs_cols, lhs) = self.cached_matrix_at(&values, *lhs)?;
                    let (rhs_rows, rhs_cols, rhs) = self.cached_matrix_at(&values, *rhs)?;
                    if lhs_cols != rhs_rows {
                        return Err(RuntimeError::InvalidShape {
                            index,
                            message: format!(
                                "cannot multiply {lhs_rows}x{lhs_cols} by {rhs_rows}x{rhs_cols}"
                            ),
                        });
                    }
                    let lhs = DMatrix::from_row_slice(lhs_rows, lhs_cols, lhs);
                    let rhs = DMatrix::from_row_slice(rhs_rows, rhs_cols, rhs);
                    let out = lhs * rhs;
                    Value::Matrix {
                        rows: out.nrows(),
                        cols: out.ncols(),
                        values: matrix_values_row_major(&out),
                    }
                }
                ExprNode::MatVec { matrix, vector } => {
                    let (rows, cols, matrix) = self.cached_matrix_at(&values, *matrix)?;
                    let vector = self.cached_vector_at(&values, *vector)?;
                    if cols != vector.len() {
                        return Err(RuntimeError::InvalidShape {
                            index,
                            message: format!(
                                "cannot multiply {rows}x{cols} matrix by len {} vector",
                                vector.len()
                            ),
                        });
                    }
                    let matrix = DMatrix::from_row_slice(rows, cols, matrix);
                    let vector = DVector::from_row_slice(vector);
                    Value::Vector((matrix * vector).iter().copied().collect())
                }
                ExprNode::Dot { lhs, rhs } => {
                    let lhs = self.cached_vector_at(&values, *lhs)?;
                    let rhs = self.cached_vector_at(&values, *rhs)?;
                    if lhs.len() != rhs.len() {
                        return Err(RuntimeError::InvalidShape {
                            index,
                            message: format!(
                                "cannot dot len {} vector with len {} vector",
                                lhs.len(),
                                rhs.len()
                            ),
                        });
                    }
                    Value::Scalar(lhs.iter().zip(rhs).map(|(lhs, rhs)| lhs * rhs).sum())
                }
                ExprNode::Solve { matrix, rhs } => {
                    let matrix_id = *matrix;
                    let (rows, cols, matrix) = self.cached_matrix_at(&values, matrix_id)?;
                    let rhs = self.cached_vector_at(&values, *rhs)?;
                    if rows != cols || rows != rhs.len() {
                        return Err(RuntimeError::InvalidShape {
                            index,
                            message: format!(
                                "cannot solve {rows}x{cols} matrix against len {} vector",
                                rhs.len()
                            ),
                        });
                    }
                    let rhs = DVector::from_row_slice(rhs);
                    let solution = self.solve_primal(
                        matrix_id,
                        rows,
                        matrix,
                        &rhs,
                        index,
                        Some((cache, row)),
                    )?;
                    Value::Vector(solution.iter().copied().collect())
                }
            };
            values.push(value);
        }

        Ok(values)
    }

    fn cached_value_slot(&self, id: ExprId) -> RuntimeResult<usize> {
        self.cached_value_slots[id.index()].ok_or_else(|| RuntimeError::InvalidShape {
            index: id.index(),
            message: "node is not part of the cached evaluation schedule".into(),
        })
    }

    fn cached_scalar_at(&self, values: &[Value], id: ExprId) -> RuntimeResult<Complex64> {
        scalar_at(values, self.cached_value_slot(id)?)
    }

    fn cached_vector_at<'a>(
        &self,
        values: &'a [Value],
        id: ExprId,
    ) -> RuntimeResult<&'a [Complex64]> {
        vector_at(values, self.cached_value_slot(id)?)
    }

    fn cached_matrix_at<'a>(
        &self,
        values: &'a [Value],
        id: ExprId,
    ) -> RuntimeResult<(usize, usize, &'a [Complex64])> {
        matrix_at(values, self.cached_value_slot(id)?)
    }

    fn check_batch_cache(&self, cache: &CpuBatchCache) -> RuntimeResult<()> {
        if cache.nodes
            == self
                .cache_plan
                .entries()
                .iter()
                .map(|entry| entry.node())
                .collect::<Vec<_>>()
            && cache.factor_nodes
                == self
                    .factor_matrices
                    .iter()
                    .map(|(node, _)| *node)
                    .collect::<Vec<_>>()
            && cache.solve_row_keys == self.solve_row_keys
        {
            Ok(())
        } else {
            Err(RuntimeError::InvalidCacheLayout)
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Value {
    Scalar(Complex64),
    Vector(Vec<Complex64>),
    Matrix {
        rows: usize,
        cols: usize,
        values: Vec<Complex64>,
    },
}

type DynamicLu = LU<Complex64, Dyn, Dyn>;

struct DerivativeWorkspace<'a> {
    plan: &'a CpuPlan,
    primals: &'a [Value],
    tangents: Vec<Option<Value>>,
    factors: HashMap<usize, DynamicLu>,
    cached_factors: Option<(&'a CpuBatchCache, usize)>,
}

impl<'a> DerivativeWorkspace<'a> {
    fn new(
        plan: &'a CpuPlan,
        primals: &'a [Value],
        cached_factors: Option<(&'a CpuBatchCache, usize)>,
    ) -> Self {
        Self {
            plan,
            primals,
            tangents: vec![None; plan.graph.nodes().len()],
            factors: HashMap::new(),
            cached_factors,
        }
    }

    fn gradient(&mut self) -> RuntimeResult<Vec<Complex64>> {
        let mut gradient = Vec::with_capacity(self.plan.autodiff.parameter_count());
        for parameter in 0..self.plan.autodiff.parameter_count() {
            let active = self
                .plan
                .autodiff
                .active_nodes(parameter)
                .expect("free parameter index is valid");
            for id in active {
                self.differentiate_node(*id)?;
            }
            gradient.push(self.scalar_tangent(self.plan.graph.root())?);
            for id in active {
                self.tangents[id.index()] = None;
            }
        }
        Ok(gradient)
    }

    fn differentiate_node(&mut self, id: ExprId) -> RuntimeResult<()> {
        let index = id.index();
        let node = self.plan.graph.nodes()[index].clone();
        let tangent = match node {
            ExprNode::ScalarParam(_) => Value::Scalar(Complex64::ONE),
            ExprNode::Unary { op, input } => {
                let input_value = self.primal_scalar(input)?;
                let output_value = self.primal_scalar(id)?;
                let input_tangent = self.scalar_tangent(input)?;
                let value = match op {
                    UnaryOp::Neg => -input_tangent,
                    UnaryOp::Real => Complex64::from(input_tangent.re),
                    UnaryOp::Imag => Complex64::from(input_tangent.im),
                    UnaryOp::Conj => input_tangent.conj(),
                    UnaryOp::NormSqr => {
                        Complex64::from(2.0 * (input_value.conj() * input_tangent).re)
                    }
                    UnaryOp::Sqrt => input_tangent / (2.0 * output_value),
                    UnaryOp::Exp => output_value * input_tangent,
                    UnaryOp::Sin => input_value.cos() * input_tangent,
                    UnaryOp::Cos => -input_value.sin() * input_tangent,
                    UnaryOp::Log => input_tangent / input_value,
                    UnaryOp::PowI(power) => {
                        if power == 0 {
                            Complex64::ZERO
                        } else if power == i32::MIN {
                            power as f64 * output_value * input_tangent / input_value
                        } else {
                            power as f64 * input_value.powi(power - 1) * input_tangent
                        }
                    }
                };
                Value::Scalar(value)
            }
            ExprNode::Binary { op, lhs, rhs } => {
                let lhs_value = self.primal_scalar(lhs)?;
                let rhs_value = self.primal_scalar(rhs)?;
                let lhs_tangent = self.scalar_tangent(lhs)?;
                let rhs_tangent = self.scalar_tangent(rhs)?;
                let value = match op {
                    BinaryOp::Add => lhs_tangent + rhs_tangent,
                    BinaryOp::Sub => lhs_tangent - rhs_tangent,
                    BinaryOp::Mul => lhs_tangent * rhs_value + lhs_value * rhs_tangent,
                    BinaryOp::Div => {
                        (lhs_tangent * rhs_value - lhs_value * rhs_tangent) / rhs_value.powi(2)
                    }
                    BinaryOp::Atan2 => {
                        let denominator = lhs_value.re.powi(2) + rhs_value.re.powi(2);
                        Complex64::from(
                            (rhs_value.re * lhs_tangent.re - lhs_value.re * rhs_tangent.re)
                                / denominator,
                        )
                    }
                };
                Value::Scalar(value)
            }
            ExprNode::NaryAdd { terms } => {
                Value::Scalar(terms.into_iter().try_fold(Complex64::ZERO, |sum, term| {
                    Ok::<_, RuntimeError>(sum + self.scalar_tangent(term)?)
                })?)
            }
            ExprNode::NaryMul { factors } => {
                let mut product = Complex64::ONE;
                let mut derivative = Complex64::ZERO;
                for factor in factors {
                    let value = self.primal_scalar(factor)?;
                    derivative = derivative * value + product * self.scalar_tangent(factor)?;
                    product *= value;
                }
                Value::Scalar(derivative)
            }
            ExprNode::Complex { re, im } => Value::Scalar(Complex64::new(
                self.scalar_tangent(re)?.re,
                self.scalar_tangent(im)?.re,
            )),
            ExprNode::Vector { .. }
                if self.cached_factors.is_some()
                    && self.plan.cached_value_slots[index].is_none()
                    && self.plan.solve_rhs_elements[index].is_some() =>
            {
                Value::Vector(Vec::new())
            }
            ExprNode::Vector { elements } => Value::Vector(
                elements
                    .into_iter()
                    .map(|element| self.scalar_tangent(element))
                    .collect::<RuntimeResult<_>>()?,
            ),
            ExprNode::Matrix {
                rows,
                cols,
                elements,
            } => {
                if elements.len() != rows * cols {
                    return Err(RuntimeError::InvalidShape {
                        index,
                        message: format!(
                            "matrix has {} elements for shape {rows}x{cols}",
                            elements.len()
                        ),
                    });
                }
                Value::Matrix {
                    rows,
                    cols,
                    values: elements
                        .into_iter()
                        .map(|element| self.scalar_tangent(element))
                        .collect::<RuntimeResult<_>>()?,
                }
            }
            ExprNode::Component { input, index: i } => {
                if let (Some(plan), Some((cache, row))) =
                    (self.plan.solve_components[index], self.cached_factors)
                {
                    let inverse_row = cache.solve_row(plan.row_slot, row)?;
                    if let Some(elements) = &self.plan.solve_rhs_elements[plan.rhs.index()] {
                        Value::Scalar(
                            inverse_row
                                .iter()
                                .zip(elements)
                                .map(|(lhs, rhs)| Ok(lhs * self.scalar_tangent(*rhs)?))
                                .sum::<RuntimeResult<Complex64>>()?,
                        )
                    } else {
                        let rhs_tangent = self.vector_tangent_value(plan.rhs, plan.dimension)?;
                        Value::Scalar(
                            inverse_row
                                .iter()
                                .zip(rhs_tangent)
                                .map(|(lhs, rhs)| lhs * rhs)
                                .sum(),
                        )
                    }
                } else {
                    let vector = self.vector_tangent(input)?;
                    Value::Scalar(*vector.get(i).ok_or_else(|| RuntimeError::InvalidShape {
                        index,
                        message: format!(
                            "component index {i} out of bounds for len {}",
                            vector.len()
                        ),
                    })?)
                }
            }
            ExprNode::MatrixElement { input, row, col } => {
                let (rows, cols, matrix) = self.matrix_tangent(input)?;
                if row >= rows || col >= cols {
                    return Err(RuntimeError::InvalidShape {
                        index,
                        message: format!(
                            "matrix element ({row}, {col}) out of bounds for shape {rows}x{cols}"
                        ),
                    });
                }
                Value::Scalar(matrix[row * cols + col])
            }
            ExprNode::MatMul { lhs, rhs } => {
                let (lhs_rows, lhs_cols, lhs_value) = self.primal_matrix(lhs)?;
                let (rhs_rows, rhs_cols, rhs_value) = self.primal_matrix(rhs)?;
                if lhs_cols != rhs_rows {
                    return Err(RuntimeError::InvalidShape {
                        index,
                        message: format!(
                            "cannot multiply {lhs_rows}x{lhs_cols} by {rhs_rows}x{rhs_cols}"
                        ),
                    });
                }
                let lhs_value = DMatrix::from_row_slice(lhs_rows, lhs_cols, lhs_value);
                let rhs_value = DMatrix::from_row_slice(rhs_rows, rhs_cols, rhs_value);
                let lhs_tangent = self.matrix_tangent_value(lhs, lhs_rows, lhs_cols)?;
                let rhs_tangent = self.matrix_tangent_value(rhs, rhs_rows, rhs_cols)?;
                let output = lhs_tangent * &rhs_value + lhs_value * rhs_tangent;
                Value::Matrix {
                    rows: output.nrows(),
                    cols: output.ncols(),
                    values: matrix_values_row_major(&output),
                }
            }
            ExprNode::MatVec { matrix, vector } => {
                let (rows, cols, matrix_value) = self.primal_matrix(matrix)?;
                let vector_value = self.primal_vector(vector)?;
                if cols != vector_value.len() {
                    return Err(RuntimeError::InvalidShape {
                        index,
                        message: format!(
                            "cannot multiply {rows}x{cols} matrix by len {} vector",
                            vector_value.len()
                        ),
                    });
                }
                let matrix_value = DMatrix::from_row_slice(rows, cols, matrix_value);
                let vector_value = DVector::from_row_slice(vector_value);
                let matrix_tangent = self.matrix_tangent_value(matrix, rows, cols)?;
                let vector_tangent = DVector::from_vec(self.vector_tangent_value(vector, cols)?);
                Value::Vector(
                    (matrix_tangent * vector_value + matrix_value * vector_tangent)
                        .iter()
                        .copied()
                        .collect(),
                )
            }
            ExprNode::Dot { lhs, rhs } => {
                let lhs_value = self.primal_vector(lhs)?;
                let rhs_value = self.primal_vector(rhs)?;
                if lhs_value.len() != rhs_value.len() {
                    return Err(RuntimeError::InvalidShape {
                        index,
                        message: format!(
                            "cannot dot len {} vector with len {} vector",
                            lhs_value.len(),
                            rhs_value.len()
                        ),
                    });
                }
                let lhs_tangent = self.vector_tangent_value(lhs, lhs_value.len())?;
                let rhs_tangent = self.vector_tangent_value(rhs, rhs_value.len())?;
                Value::Scalar(
                    lhs_tangent
                        .iter()
                        .zip(rhs_value)
                        .map(|(lhs, rhs)| lhs * rhs)
                        .sum::<Complex64>()
                        + lhs_value
                            .iter()
                            .zip(rhs_tangent)
                            .map(|(lhs, rhs)| lhs * rhs)
                            .sum::<Complex64>(),
                )
            }
            ExprNode::Solve { matrix, rhs } => {
                if self.cached_factors.is_some() && self.plan.cached_value_slots[index].is_none() {
                    // Specialized components differentiate the RHS directly and never read this.
                    self.tangents[index] = Some(Value::Vector(Vec::new()));
                    return Ok(());
                }
                let (rows, cols, matrix_value) = self.primal_matrix(matrix)?;
                let solution = self.primal_vector(id)?;
                let rhs_value = self.primal_vector(rhs)?;
                if rows != cols || rows != rhs_value.len() {
                    return Err(RuntimeError::InvalidShape {
                        index,
                        message: format!(
                            "cannot solve {rows}x{cols} matrix against len {} vector",
                            rhs_value.len()
                        ),
                    });
                }
                let matrix_tangent = self.matrix_tangent_value(matrix, rows, cols)?;
                let rhs_tangent = DVector::from_vec(self.vector_tangent_value(rhs, rows)?);
                let solution = DVector::from_row_slice(solution);
                let tangent_rhs = rhs_tangent - matrix_tangent * solution;
                let tangent = if let (Some(slot), Some((cache, row))) = (
                    self.plan.factor_matrix_slots[matrix.index()],
                    self.cached_factors,
                ) {
                    cache
                        .factor(slot, row)?
                        .solve(&tangent_rhs)
                        .ok_or(RuntimeError::SingularMatrix(index))?
                } else if let Some(slot) = self.plan.constant_factor_slots[matrix.index()] {
                    self.plan.constant_factors[slot]
                        .get_or_init(|| DMatrix::from_row_slice(rows, cols, matrix_value).lu())
                        .solve(&tangent_rhs)
                        .ok_or(RuntimeError::SingularMatrix(index))?
                } else {
                    let matrix_value = DMatrix::from_row_slice(rows, cols, matrix_value);
                    self.factors
                        .entry(matrix.index())
                        .or_insert_with(|| matrix_value.lu())
                        .solve(&tangent_rhs)
                        .ok_or(RuntimeError::SingularMatrix(index))?
                };
                Value::Vector(tangent.iter().copied().collect())
            }
            ExprNode::RealConst(_)
            | ExprNode::ComplexConst(_)
            | ExprNode::EventScalar(_)
            | ExprNode::EventP4Component { .. } => {
                return Err(RuntimeError::InvalidShape {
                    index,
                    message: "parameter-independent node appeared in a derivative lane".into(),
                });
            }
        };
        self.tangents[index] = Some(tangent);
        Ok(())
    }

    fn primal_scalar(&self, id: ExprId) -> RuntimeResult<Complex64> {
        if self.cached_factors.is_some() {
            self.plan.cached_scalar_at(self.primals, id)
        } else {
            scalar_at(self.primals, id.index())
        }
    }

    fn primal_vector(&self, id: ExprId) -> RuntimeResult<&[Complex64]> {
        if self.cached_factors.is_some() {
            self.plan.cached_vector_at(self.primals, id)
        } else {
            vector_at(self.primals, id.index())
        }
    }

    fn primal_matrix(&self, id: ExprId) -> RuntimeResult<(usize, usize, &[Complex64])> {
        if self.cached_factors.is_some() {
            self.plan.cached_matrix_at(self.primals, id)
        } else {
            matrix_at(self.primals, id.index())
        }
    }

    fn scalar_tangent(&self, id: ExprId) -> RuntimeResult<Complex64> {
        match &self.tangents[id.index()] {
            Some(Value::Scalar(value)) => Ok(*value),
            Some(value) => Err(RuntimeError::TypeMismatch {
                index: id.index(),
                expected: "scalar tangent",
                actual: value.kind(),
            }),
            None => Ok(Complex64::ZERO),
        }
    }

    fn vector_tangent(&self, id: ExprId) -> RuntimeResult<&[Complex64]> {
        match &self.tangents[id.index()] {
            Some(Value::Vector(values)) => Ok(values),
            Some(value) => Err(RuntimeError::TypeMismatch {
                index: id.index(),
                expected: "vector tangent",
                actual: value.kind(),
            }),
            None => Err(RuntimeError::InvalidShape {
                index: id.index(),
                message: "inactive vector tangent requested without a target length".into(),
            }),
        }
    }

    fn vector_tangent_value(&self, id: ExprId, len: usize) -> RuntimeResult<Vec<Complex64>> {
        match &self.tangents[id.index()] {
            Some(Value::Vector(values)) if values.len() == len => Ok(values.clone()),
            Some(Value::Vector(values)) => Err(RuntimeError::InvalidShape {
                index: id.index(),
                message: format!("vector tangent has len {}, expected {len}", values.len()),
            }),
            Some(value) => Err(RuntimeError::TypeMismatch {
                index: id.index(),
                expected: "vector tangent",
                actual: value.kind(),
            }),
            None => Ok(vec![Complex64::ZERO; len]),
        }
    }

    fn matrix_tangent(&self, id: ExprId) -> RuntimeResult<(usize, usize, &[Complex64])> {
        match &self.tangents[id.index()] {
            Some(Value::Matrix { rows, cols, values }) => Ok((*rows, *cols, values)),
            Some(value) => Err(RuntimeError::TypeMismatch {
                index: id.index(),
                expected: "matrix tangent",
                actual: value.kind(),
            }),
            None => Err(RuntimeError::InvalidShape {
                index: id.index(),
                message: "inactive matrix tangent requested without a target shape".into(),
            }),
        }
    }

    fn matrix_tangent_value(
        &self,
        id: ExprId,
        rows: usize,
        cols: usize,
    ) -> RuntimeResult<DMatrix<Complex64>> {
        match &self.tangents[id.index()] {
            Some(Value::Matrix {
                rows: actual_rows,
                cols: actual_cols,
                values,
            }) if *actual_rows == rows && *actual_cols == cols => {
                Ok(DMatrix::from_row_slice(rows, cols, values))
            }
            Some(Value::Matrix {
                rows: actual_rows,
                cols: actual_cols,
                ..
            }) => Err(RuntimeError::InvalidShape {
                index: id.index(),
                message: format!(
                    "matrix tangent has shape {actual_rows}x{actual_cols}, expected {rows}x{cols}"
                ),
            }),
            Some(value) => Err(RuntimeError::TypeMismatch {
                index: id.index(),
                expected: "matrix tangent",
                actual: value.kind(),
            }),
            None => Ok(DMatrix::zeros(rows, cols)),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum EventColumn {
    Scalar(usize),
    P4Component { col: usize, component: P4Component },
}

#[derive(Clone, Debug)]
pub struct CpuBatchCache {
    len: usize,
    weights: Vec<f64>,
    sum_weights: f64,
    nodes: Vec<ExprId>,
    pub(crate) slots: Vec<CachedSlot>,
    factor_nodes: Vec<ExprId>,
    factor_slots: Vec<CachedFactorSlot>,
    solve_row_keys: Vec<(ExprId, usize, usize)>,
    pub(crate) solve_row_slots: Vec<CachedSolveRowSlot>,
}

impl CpuBatchCache {
    fn new(
        cache_plan: &CachePlan,
        factor_matrices: &[(ExprId, usize)],
        solve_row_keys: &[(ExprId, usize, usize)],
        len: usize,
    ) -> Self {
        Self {
            len,
            weights: vec![1.0; len],
            sum_weights: len as f64,
            nodes: cache_plan
                .entries()
                .iter()
                .map(|entry| entry.node())
                .collect(),
            slots: cache_plan
                .entries()
                .iter()
                .map(|entry| CachedSlot::new(entry.value_kind(), len))
                .collect(),
            factor_nodes: factor_matrices.iter().map(|(node, _)| *node).collect(),
            factor_slots: factor_matrices
                .iter()
                .map(|(_, dimension)| CachedFactorSlot::new(*dimension))
                .collect(),
            solve_row_keys: solve_row_keys.to_vec(),
            solve_row_slots: solve_row_keys
                .iter()
                .map(|(_, _, dimension)| CachedSolveRowSlot::new(*dimension))
                .collect(),
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    pub fn sum_weights(&self) -> f64 {
        self.sum_weights
    }

    pub fn resident_bytes(&self) -> usize {
        self.weights.capacity() * size_of::<f64>()
            + self.nodes.capacity() * size_of::<ExprId>()
            + self
                .slots
                .iter()
                .map(CachedSlot::resident_bytes)
                .sum::<usize>()
            + self.factor_nodes.capacity() * size_of::<ExprId>()
            + self
                .factor_slots
                .iter()
                .map(CachedFactorSlot::resident_bytes)
                .sum::<usize>()
            + self.solve_row_keys.capacity() * size_of::<(ExprId, usize, usize)>()
            + self
                .solve_row_slots
                .iter()
                .map(CachedSolveRowSlot::resident_bytes)
                .sum::<usize>()
    }

    fn set_weights(&mut self, weights: Vec<f64>) {
        self.sum_weights = weights.iter().sum();
        self.weights = weights;
    }

    fn push(&mut self, slot: usize, value: Value) -> RuntimeResult<()> {
        let len = self.slots.len();
        self.slots
            .get_mut(slot)
            .ok_or(RuntimeError::InvalidCache {
                expected: len,
                actual: slot + 1,
            })?
            .push(value)
    }

    fn value(&self, slot: usize, row: usize) -> RuntimeResult<Value> {
        if row >= self.len {
            return Err(RuntimeError::InvalidShape {
                index: row,
                message: format!("cache row {row} out of bounds for len {}", self.len),
            });
        }
        self.slots
            .get(slot)
            .ok_or(RuntimeError::InvalidCache {
                expected: self.slots.len(),
                actual: slot + 1,
            })?
            .value(row)
    }

    fn scalar(&self, slot: usize, row: usize) -> RuntimeResult<Complex64> {
        if row >= self.len {
            return Err(RuntimeError::InvalidShape {
                index: row,
                message: format!("cache row {row} out of bounds for len {}", self.len),
            });
        }
        self.slots
            .get(slot)
            .ok_or(RuntimeError::InvalidCache {
                expected: self.slots.len(),
                actual: slot + 1,
            })?
            .scalar(row)
    }

    fn real_range(&self, slot: usize, start: usize, end: usize) -> RuntimeResult<&[f64]> {
        if start > end || end > self.len {
            return Err(RuntimeError::InvalidShape {
                index: start,
                message: format!(
                    "cache range {start}..{end} out of bounds for len {}",
                    self.len
                ),
            });
        }
        self.slots
            .get(slot)
            .ok_or(RuntimeError::InvalidCache {
                expected: self.slots.len(),
                actual: slot + 1,
            })?
            .real_range(start, end)
    }

    fn complex_range(&self, slot: usize, start: usize, end: usize) -> RuntimeResult<&[Complex64]> {
        if start > end || end > self.len {
            return Err(RuntimeError::InvalidShape {
                index: start,
                message: format!(
                    "cache range {start}..{end} out of bounds for len {}",
                    self.len
                ),
            });
        }
        self.slots
            .get(slot)
            .ok_or(RuntimeError::InvalidCache {
                expected: self.slots.len(),
                actual: slot + 1,
            })?
            .complex_range(start, end)
    }

    fn push_factor(&mut self, slot: usize, factor: DynamicLu) -> RuntimeResult<()> {
        let len = self.factor_slots.len();
        self.factor_slots
            .get_mut(slot)
            .ok_or(RuntimeError::InvalidCache {
                expected: len,
                actual: slot + 1,
            })?
            .push(factor)
    }

    fn factor(&self, slot: usize, row: usize) -> RuntimeResult<&DynamicLu> {
        self.factor_slots
            .get(slot)
            .ok_or(RuntimeError::InvalidCache {
                expected: self.factor_slots.len(),
                actual: slot + 1,
            })?
            .factor(row)
    }

    fn push_solve_row(
        &mut self,
        slot: usize,
        values: impl IntoIterator<Item = Complex64>,
    ) -> RuntimeResult<()> {
        let len = self.solve_row_slots.len();
        self.solve_row_slots
            .get_mut(slot)
            .ok_or(RuntimeError::InvalidCache {
                expected: len,
                actual: slot + 1,
            })?
            .push(values)
    }

    fn solve_row(&self, slot: usize, row: usize) -> RuntimeResult<&[Complex64]> {
        self.solve_row_slots
            .get(slot)
            .ok_or(RuntimeError::InvalidCache {
                expected: self.solve_row_slots.len(),
                actual: slot + 1,
            })?
            .row(row)
    }
}

#[derive(Clone, Debug)]
pub struct CpuCachedBatch {
    cache: CpuBatchCache,
}

impl CpuCachedBatch {
    pub fn cache(&self) -> &CpuBatchCache {
        &self.cache
    }

    pub fn len(&self) -> usize {
        self.cache.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    pub fn weights(&self) -> &[f64] {
        self.cache.weights()
    }

    pub fn sum_weights(&self) -> f64 {
        self.cache.sum_weights()
    }

    pub fn resident_bytes(&self) -> usize {
        self.cache.resident_bytes()
    }
}

#[derive(Clone, Debug, Default)]
pub struct CpuCachedDataset {
    batches: Vec<CpuCachedBatch>,
    sum_weights: f64,
}

#[derive(Copy, Clone, Debug, PartialEq)]
/// Fixed statistics collected when a dataset is prepared for repeated evaluation.
pub struct PreparedDatasetStats {
    local_events: usize,
    global_events: usize,
    local_batches: usize,
    sum_weights: f64,
    resident_bytes: usize,
    storage: CacheStorage,
}

impl PreparedDatasetStats {
    pub fn local_events(&self) -> usize {
        self.local_events
    }

    pub fn global_events(&self) -> usize {
        self.global_events
    }

    pub fn local_batches(&self) -> usize {
        self.local_batches
    }

    pub fn sum_weights(&self) -> f64 {
        self.sum_weights
    }

    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }

    pub fn storage(&self) -> CacheStorage {
        self.storage
    }
}

#[derive(Clone)]
/// A dataset prepared according to its [`CacheStorage`] policy.
///
/// Resident datasets own all event-dependent cache values. Streaming datasets retain the source
/// and read plan and rebuild transient batch caches on every reduction.
pub enum CpuPreparedDataset {
    Resident {
        dataset: CpuCachedDataset,
        stats: PreparedDatasetStats,
    },
    Streaming {
        dataset: Dataset,
        read_plan: laddu_data::io::ReadPlan,
        stats: PreparedDatasetStats,
    },
}

impl std::fmt::Debug for CpuPreparedDataset {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CpuPreparedDataset")
            .field("stats", self.stats())
            .finish_non_exhaustive()
    }
}

impl CpuPreparedDataset {
    pub fn stats(&self) -> &PreparedDatasetStats {
        match self {
            Self::Resident { stats, .. } | Self::Streaming { stats, .. } => stats,
        }
    }
}

impl CpuCachedDataset {
    pub fn batches(&self) -> &[CpuCachedBatch] {
        &self.batches
    }

    pub fn len(&self) -> usize {
        self.batches.iter().map(CpuCachedBatch::len).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.batches.iter().all(CpuCachedBatch::is_empty)
    }

    pub fn sum_weights(&self) -> f64 {
        self.sum_weights
    }

    pub fn resident_bytes(&self) -> usize {
        self.batches
            .iter()
            .map(CpuCachedBatch::resident_bytes)
            .sum()
    }
}

#[derive(Clone, Debug)]
struct CachedFactorSlot {
    dimension: usize,
    factors: Vec<DynamicLu>,
}

#[derive(Clone, Debug)]
pub(crate) struct CachedSolveRowSlot {
    pub(crate) dimension: usize,
    pub(crate) values: Vec<Complex64>,
}

impl CachedSolveRowSlot {
    fn new(dimension: usize) -> Self {
        Self {
            dimension,
            values: Vec::new(),
        }
    }

    fn push(&mut self, values: impl IntoIterator<Item = Complex64>) -> RuntimeResult<()> {
        let start = self.values.len();
        self.values.extend(values);
        let actual = self.values.len() - start;
        if actual != self.dimension {
            return Err(RuntimeError::InvalidShape {
                index: start / self.dimension,
                message: format!(
                    "cached solve row has len {actual}, expected {}",
                    self.dimension
                ),
            });
        }
        Ok(())
    }

    fn row(&self, row: usize) -> RuntimeResult<&[Complex64]> {
        let start = row
            .checked_mul(self.dimension)
            .ok_or_else(|| RuntimeError::InvalidShape {
                index: row,
                message: "cached solve row offset overflowed".into(),
            })?;
        self.values
            .get(start..start + self.dimension)
            .ok_or_else(|| RuntimeError::InvalidShape {
                index: row,
                message: format!(
                    "cached solve row {row} out of bounds for len {}",
                    self.values.len() / self.dimension
                ),
            })
    }

    fn resident_bytes(&self) -> usize {
        self.values.capacity() * size_of::<Complex64>()
    }
}

impl CachedFactorSlot {
    fn new(dimension: usize) -> Self {
        Self {
            dimension,
            factors: Vec::new(),
        }
    }

    fn push(&mut self, factor: DynamicLu) -> RuntimeResult<()> {
        self.factors.push(factor);
        Ok(())
    }

    fn factor(&self, row: usize) -> RuntimeResult<&DynamicLu> {
        self.factors
            .get(row)
            .ok_or_else(|| RuntimeError::InvalidShape {
                index: row,
                message: format!(
                    "factor row {row} out of bounds for len {}",
                    self.factors.len()
                ),
            })
    }

    fn resident_bytes(&self) -> usize {
        self.factors.capacity()
            * (self.dimension * self.dimension * size_of::<Complex64>()
                + self.dimension * size_of::<usize>())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum CachedSlot {
    Real(Vec<f64>),
    Complex(Vec<Complex64>),
    Vector {
        len: usize,
        values: Vec<Complex64>,
    },
    Matrix {
        rows: usize,
        cols: usize,
        values: Vec<Complex64>,
    },
}

impl CachedSlot {
    #[cfg(feature = "jit")]
    pub(crate) fn values_ptr(&self) -> *const u8 {
        match self {
            Self::Real(values) => values.as_ptr().cast(),
            Self::Complex(values) | Self::Vector { values, .. } | Self::Matrix { values, .. } => {
                values.as_ptr().cast()
            }
        }
    }

    #[cfg(feature = "jit")]
    pub(crate) fn width(&self) -> usize {
        match self {
            Self::Real(_) | Self::Complex(_) => 1,
            Self::Vector { len, .. } => *len,
            Self::Matrix { rows, cols, .. } => rows * cols,
        }
    }

    fn new(kind: ValueKind, events: usize) -> Self {
        match kind {
            ValueKind::Real => Self::Real(Vec::with_capacity(events)),
            ValueKind::Complex => Self::Complex(Vec::with_capacity(events)),
            ValueKind::Vector { len } => Self::Vector {
                len,
                values: Vec::with_capacity(events.saturating_mul(len)),
            },
            ValueKind::Matrix { rows, cols } => Self::Matrix {
                rows,
                cols,
                values: Vec::with_capacity(events.saturating_mul(rows).saturating_mul(cols)),
            },
        }
    }

    fn resident_bytes(&self) -> usize {
        match self {
            Self::Real(values) => values.capacity() * size_of::<f64>(),
            Self::Complex(values) => values.capacity() * size_of::<Complex64>(),
            Self::Vector { values, .. } | Self::Matrix { values, .. } => {
                values.capacity() * size_of::<Complex64>()
            }
        }
    }

    fn push(&mut self, value: Value) -> RuntimeResult<()> {
        match (self, value) {
            (Self::Real(values), Value::Scalar(value)) => {
                values.push(value.re);
                Ok(())
            }
            (Self::Complex(values), Value::Scalar(value)) => {
                values.push(value);
                Ok(())
            }
            (Self::Vector { len, values }, Value::Vector(value)) if *len == value.len() => {
                values.extend(value);
                Ok(())
            }
            (
                Self::Matrix { rows, cols, values },
                Value::Matrix {
                    rows: value_rows,
                    cols: value_cols,
                    values: value,
                },
            ) if *rows == value_rows && *cols == value_cols => {
                values.extend(value);
                Ok(())
            }
            (_, value) => Err(RuntimeError::InvalidShape {
                index: 0,
                message: format!("cached value kind did not match slot: {}", value.kind()),
            }),
        }
    }

    fn value(&self, row: usize) -> RuntimeResult<Value> {
        match self {
            Self::Real(values) => values
                .get(row)
                .copied()
                .map(Complex64::from)
                .map(Value::Scalar)
                .ok_or_else(|| RuntimeError::InvalidShape {
                    index: row,
                    message: format!("cache row {row} out of bounds"),
                }),
            Self::Complex(values) => values.get(row).copied().map(Value::Scalar).ok_or_else(|| {
                RuntimeError::InvalidShape {
                    index: row,
                    message: format!("cache row {row} out of bounds"),
                }
            }),
            Self::Vector { len, values } => {
                let start = row
                    .checked_mul(*len)
                    .ok_or_else(|| RuntimeError::InvalidShape {
                        index: row,
                        message: "cache vector row offset overflowed".into(),
                    })?;
                let end = start + *len;
                values
                    .get(start..end)
                    .map(|value| Value::Vector(value.to_vec()))
                    .ok_or_else(|| RuntimeError::InvalidShape {
                        index: row,
                        message: format!("cache row {row} out of bounds"),
                    })
            }
            Self::Matrix { rows, cols, values } => {
                let len = rows * cols;
                let start = row
                    .checked_mul(len)
                    .ok_or_else(|| RuntimeError::InvalidShape {
                        index: row,
                        message: "cache matrix row offset overflowed".into(),
                    })?;
                let end = start + len;
                values
                    .get(start..end)
                    .map(|value| Value::Matrix {
                        rows: *rows,
                        cols: *cols,
                        values: value.to_vec(),
                    })
                    .ok_or_else(|| RuntimeError::InvalidShape {
                        index: row,
                        message: format!("cache row {row} out of bounds"),
                    })
            }
        }
    }

    fn scalar(&self, row: usize) -> RuntimeResult<Complex64> {
        match self {
            Self::Real(values) => values
                .get(row)
                .copied()
                .map(Complex64::from)
                .ok_or_else(|| RuntimeError::InvalidShape {
                    index: row,
                    message: format!("cache row {row} out of bounds"),
                }),
            Self::Complex(values) => {
                values
                    .get(row)
                    .copied()
                    .ok_or_else(|| RuntimeError::InvalidShape {
                        index: row,
                        message: format!("cache row {row} out of bounds"),
                    })
            }
            Self::Vector { .. } | Self::Matrix { .. } => Err(RuntimeError::TypeMismatch {
                index: row,
                expected: "scalar",
                actual: match self {
                    Self::Vector { .. } => "vector",
                    Self::Matrix { .. } => "matrix",
                    Self::Real(_) | Self::Complex(_) => unreachable!(),
                },
            }),
        }
    }

    fn real_range(&self, start: usize, end: usize) -> RuntimeResult<&[f64]> {
        match self {
            Self::Real(values) => {
                values
                    .get(start..end)
                    .ok_or_else(|| RuntimeError::InvalidShape {
                        index: start,
                        message: format!("cache range {start}..{end} out of bounds"),
                    })
            }
            Self::Complex(_) | Self::Vector { .. } | Self::Matrix { .. } => {
                Err(RuntimeError::TypeMismatch {
                    index: start,
                    expected: "real scalar",
                    actual: match self {
                        Self::Complex(_) => "complex scalar",
                        Self::Vector { .. } => "vector",
                        Self::Matrix { .. } => "matrix",
                        Self::Real(_) => unreachable!(),
                    },
                })
            }
        }
    }

    fn complex_range(&self, start: usize, end: usize) -> RuntimeResult<&[Complex64]> {
        match self {
            Self::Complex(values) => {
                values
                    .get(start..end)
                    .ok_or_else(|| RuntimeError::InvalidShape {
                        index: start,
                        message: format!("cache range {start}..{end} out of bounds"),
                    })
            }
            Self::Real(_) | Self::Vector { .. } | Self::Matrix { .. } => {
                Err(RuntimeError::TypeMismatch {
                    index: start,
                    expected: "complex scalar",
                    actual: match self {
                        Self::Real(_) => "real scalar",
                        Self::Vector { .. } => "vector",
                        Self::Matrix { .. } => "matrix",
                        Self::Complex(_) => unreachable!(),
                    },
                })
            }
        }
    }
}

impl Value {
    fn kind(&self) -> &'static str {
        match self {
            Self::Scalar(_) => "scalar",
            Self::Vector(_) => "vector",
            Self::Matrix { .. } => "matrix",
        }
    }
}

fn scalar_at(values: &[Value], index: usize) -> RuntimeResult<Complex64> {
    match &values[index] {
        Value::Scalar(value) => Ok(*value),
        value => Err(RuntimeError::TypeMismatch {
            index,
            expected: "scalar",
            actual: value.kind(),
        }),
    }
}

fn vector_at(values: &[Value], index: usize) -> RuntimeResult<&[Complex64]> {
    match &values[index] {
        Value::Vector(value) => Ok(value),
        value => Err(RuntimeError::TypeMismatch {
            index,
            expected: "vector",
            actual: value.kind(),
        }),
    }
}

fn matrix_at(values: &[Value], index: usize) -> RuntimeResult<(usize, usize, &[Complex64])> {
    match &values[index] {
        Value::Matrix { rows, cols, values } => Ok((*rows, *cols, values)),
        value => Err(RuntimeError::TypeMismatch {
            index,
            expected: "matrix",
            actual: value.kind(),
        }),
    }
}

fn scalar_at_optional(values: &[Option<Value>], index: usize) -> RuntimeResult<Complex64> {
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

fn vector_at_optional(values: &[Option<Value>], index: usize) -> RuntimeResult<&[Complex64]> {
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

fn matrix_at_optional(
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

impl CpuBackend {
    fn solve_component_plans(
        &self,
        model: &CompiledModel,
    ) -> (
        Vec<Option<SolveComponentPlan>>,
        Vec<Option<Vec<ExprId>>>,
        Vec<SolveRowMatrixPlan>,
        Vec<(ExprId, usize, usize)>,
    ) {
        let mut components = vec![None; model.graph().nodes().len()];
        let mut rhs_elements = vec![None; model.graph().nodes().len()];
        let mut row_slots = HashMap::<(ExprId, usize), usize>::new();
        let mut row_keys = Vec::new();
        let mut matrix_slots = HashMap::<ExprId, usize>::new();
        let mut matrices = Vec::<SolveRowMatrixPlan>::new();

        for (node_index, node) in model.graph().nodes().iter().enumerate() {
            let ExprNode::Component { input, index } = node else {
                continue;
            };
            let Some(ExprNode::Solve { matrix, rhs }) = model.graph().node(*input) else {
                continue;
            };
            let matrix_facts = model
                .node_facts(*matrix)
                .expect("compiled model facts cover every graph node");
            let matrix_dependency = matrix_facts.dependency;
            if !matrix_dependency.depends_on_event
                || matrix_dependency.depends_on_free_params
                || matrix_dependency.depends_on_fixed_params
            {
                continue;
            }
            let rhs_facts = model
                .node_facts(*rhs)
                .expect("compiled model facts cover every graph node");
            let rhs_dependency = rhs_facts.dependency;
            if !rhs_dependency.depends_on_free_params && !rhs_dependency.depends_on_fixed_params {
                continue;
            }
            let ValueKind::Matrix { rows, cols } = matrix_facts.value_kind else {
                continue;
            };
            let ValueKind::Vector { len } = rhs_facts.value_kind else {
                continue;
            };
            if rows == 0 || rows != cols || rows != len || *index >= rows {
                continue;
            }

            let row_slot = if let Some(slot) = row_slots.get(&(*matrix, *index)) {
                *slot
            } else {
                let slot = row_keys.len();
                row_slots.insert((*matrix, *index), slot);
                row_keys.push((*matrix, *index, rows));
                let matrix_slot = if let Some(slot) = matrix_slots.get(matrix) {
                    *slot
                } else {
                    let slot = matrices.len();
                    matrix_slots.insert(*matrix, slot);
                    matrices.push(SolveRowMatrixPlan {
                        matrix: *matrix,
                        dimension: rows,
                        rows: Vec::new(),
                    });
                    slot
                };
                matrices[matrix_slot].rows.push((slot, *index));
                slot
            };
            components[node_index] = Some(SolveComponentPlan {
                rhs: *rhs,
                row_slot,
                dimension: rows,
            });
            if let Some(ExprNode::Vector { elements }) = model.graph().node(*rhs) {
                rhs_elements[rhs.index()] = Some(elements.clone());
            }
        }

        (components, rhs_elements, matrices, row_keys)
    }

    #[allow(clippy::too_many_arguments)]
    fn scalar_kernel_ir(
        &self,
        model: &CompiledModel,
        evaluation_nodes: &[ExprId],
        cache_slots: &[Option<usize>],
        parameter_slots: &[Option<ParamId>],
        solve_components: &[Option<SolveComponentPlan>],
        solve_rhs_elements: &[Option<Vec<ExprId>>],
    ) -> Option<ScalarKernelIr> {
        let mut value_ids = vec![None; model.graph().nodes().len()];
        let mut values = Vec::with_capacity(evaluation_nodes.len());

        for id in evaluation_nodes {
            let index = id.index();
            let value_id = |id: ExprId| value_ids[id.index()];
            let instruction = if let Some(cache_slot) = cache_slots[index] {
                KernelInstruction::Cached(cache_slot)
            } else {
                match model.graph().node(*id)? {
                    ExprNode::RealConst(value) => KernelInstruction::RealConstant(*value),
                    ExprNode::ComplexConst(value) => KernelInstruction::ComplexConstant(*value),
                    ExprNode::ScalarParam(_) => {
                        KernelInstruction::Parameter(parameter_slots[index]?)
                    }
                    ExprNode::Unary { op, input } => KernelInstruction::Unary {
                        op: *op,
                        input: value_id(*input)?,
                    },
                    ExprNode::Binary { op, lhs, rhs } => KernelInstruction::Binary {
                        op: *op,
                        lhs: value_id(*lhs)?,
                        rhs: value_id(*rhs)?,
                    },
                    ExprNode::NaryAdd { terms } => KernelInstruction::Add(
                        terms
                            .iter()
                            .map(|term| value_id(*term))
                            .collect::<Option<_>>()?,
                    ),
                    ExprNode::NaryMul { factors } => KernelInstruction::Mul(
                        factors
                            .iter()
                            .map(|factor| value_id(*factor))
                            .collect::<Option<_>>()?,
                    ),
                    ExprNode::Complex { re, im } => KernelInstruction::Complex {
                        re: value_id(*re)?,
                        im: value_id(*im)?,
                    },
                    ExprNode::Vector { elements } => KernelInstruction::Vector(
                        elements
                            .iter()
                            .map(|element| value_id(*element))
                            .collect::<Option<_>>()?,
                    ),
                    ExprNode::Matrix {
                        rows,
                        cols,
                        elements,
                    } => KernelInstruction::Matrix {
                        rows: *rows,
                        cols: *cols,
                        elements: elements
                            .iter()
                            .map(|element| value_id(*element))
                            .collect::<Option<_>>()?,
                    },
                    ExprNode::Component {
                        input,
                        index: component,
                    } => {
                        if let Some(solve) = solve_components[index]
                            && let Some(elements) = solve_rhs_elements[solve.rhs.index()].as_ref()
                        {
                            KernelInstruction::SolveRow {
                                row_slot: solve.row_slot,
                                rhs: elements
                                    .iter()
                                    .map(|element| value_id(*element))
                                    .collect::<Option<_>>()?,
                            }
                        } else {
                            KernelInstruction::Component {
                                input: value_id(*input)?,
                                index: *component,
                            }
                        }
                    }
                    ExprNode::MatrixElement { input, row, col } => {
                        KernelInstruction::MatrixElement {
                            input: value_id(*input)?,
                            row: *row,
                            col: *col,
                        }
                    }
                    ExprNode::MatMul { lhs, rhs } => KernelInstruction::MatMul {
                        lhs: value_id(*lhs)?,
                        rhs: value_id(*rhs)?,
                    },
                    ExprNode::MatVec { matrix, vector } => KernelInstruction::MatVec {
                        matrix: value_id(*matrix)?,
                        vector: value_id(*vector)?,
                    },
                    ExprNode::Dot { lhs, rhs } => KernelInstruction::Dot {
                        lhs: value_id(*lhs)?,
                        rhs: value_id(*rhs)?,
                    },
                    ExprNode::Solve { matrix, rhs } => KernelInstruction::Solve {
                        matrix: value_id(*matrix)?,
                        rhs: value_id(*rhs)?,
                    },
                    // Event leaves are always materialized by the cache schedule before
                    // final-evaluation IR is lowered.
                    ExprNode::EventScalar(_) | ExprNode::EventP4Component { .. } => return None,
                }
            };
            let facts = model.node_facts(*id)?;
            let kind = match facts.value_kind {
                ValueKind::Real => KernelValueKind::Real,
                ValueKind::Complex => KernelValueKind::Complex,
                ValueKind::Vector { len } => KernelValueKind::Vector { len },
                ValueKind::Matrix { rows, cols } => KernelValueKind::Matrix { rows, cols },
            };
            let class = if facts.dependency.depends_on_event {
                KernelValueClass::Event
            } else {
                KernelValueClass::Invariant
            };
            let kernel_id = KernelValueId::from_index(values.len());
            values.push(KernelValue {
                kind,
                class,
                instruction,
            });
            value_ids[index] = Some(kernel_id);
        }

        Some(
            ScalarKernelIr::new(values, value_ids[model.graph().root().index()]?)
                .expect("runtime lowering must produce valid scalar kernel IR"),
        )
    }
}

fn cached_evaluation_schedule(
    graph: &ExprGraph,
    cache_slots: &[Option<usize>],
    solve_components: &[Option<SolveComponentPlan>],
    solve_rhs_elements: &[Option<Vec<ExprId>>],
) -> (Vec<ExprId>, Vec<Option<usize>>) {
    let mut required = vec![false; graph.nodes().len()];
    mark_cached_evaluation_node(
        graph,
        graph.root(),
        cache_slots,
        solve_components,
        solve_rhs_elements,
        &mut required,
    );
    let nodes = required
        .into_iter()
        .enumerate()
        .filter_map(|(index, required)| {
            required.then(|| ExprId::from_index(index).expect("graph too large"))
        })
        .collect::<Vec<_>>();
    let mut value_slots = vec![None; graph.nodes().len()];
    for (slot, id) in nodes.iter().enumerate() {
        value_slots[id.index()] = Some(slot);
    }
    (nodes, value_slots)
}

fn mark_cached_evaluation_node(
    graph: &ExprGraph,
    id: ExprId,
    cache_slots: &[Option<usize>],
    solve_components: &[Option<SolveComponentPlan>],
    solve_rhs_elements: &[Option<Vec<ExprId>>],
    required: &mut [bool],
) {
    if required[id.index()] {
        return;
    }
    required[id.index()] = true;
    if cache_slots[id.index()].is_some() {
        return;
    }
    if let Some(plan) = solve_components[id.index()] {
        if let Some(elements) = &solve_rhs_elements[plan.rhs.index()] {
            for element in elements {
                mark_cached_evaluation_node(
                    graph,
                    *element,
                    cache_slots,
                    solve_components,
                    solve_rhs_elements,
                    required,
                );
            }
        } else {
            mark_cached_evaluation_node(
                graph,
                plan.rhs,
                cache_slots,
                solve_components,
                solve_rhs_elements,
                required,
            );
        }
        return;
    }
    if let Some(node) = graph.node(id) {
        for child in node_children(node) {
            mark_cached_evaluation_node(
                graph,
                child,
                cache_slots,
                solve_components,
                solve_rhs_elements,
                required,
            );
        }
    }
}

fn mark_required(graph: &ExprGraph, id: ExprId, required: &mut [bool]) {
    if required[id.index()] {
        return;
    }
    required[id.index()] = true;
    if let Some(node) = graph.node(id) {
        for child in node_children(node) {
            mark_required(graph, child, required);
        }
    }
}

fn node_children(node: &ExprNode) -> Vec<ExprId> {
    node.child_ids()
}

fn matrix_values_row_major(matrix: &DMatrix<Complex64>) -> Vec<Complex64> {
    let mut values = Vec::with_capacity(matrix.nrows() * matrix.ncols());
    for row in 0..matrix.nrows() {
        for col in 0..matrix.ncols() {
            values.push(matrix[(row, col)]);
        }
    }
    values
}

fn eval_unary(op: UnaryOp, input: Complex64) -> Complex64 {
    match op {
        UnaryOp::Neg => -input,
        UnaryOp::Real => Complex64::from(input.re),
        UnaryOp::Imag => Complex64::from(input.im),
        UnaryOp::Conj => input.conj(),
        UnaryOp::NormSqr => Complex64::from(input.norm_sqr()),
        UnaryOp::Sqrt => input.sqrt(),
        UnaryOp::Exp => input.exp(),
        UnaryOp::Sin => input.sin(),
        UnaryOp::Cos => input.cos(),
        UnaryOp::Log => input.ln(),
        UnaryOp::PowI(power) => input.powi(power),
    }
}

fn eval_binary(op: BinaryOp, lhs: Complex64, rhs: Complex64) -> Complex64 {
    match op {
        BinaryOp::Add => lhs + rhs,
        BinaryOp::Sub => lhs - rhs,
        BinaryOp::Mul => lhs * rhs,
        BinaryOp::Div => lhs / rhs,
        BinaryOp::Atan2 => Complex64::from(lhs.re.atan2(rhs.re)),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use laddu_compile::{CompileOptions, CompiledModel};
    use laddu_data::{
        RealVec4,
        data::{Dataset, EventBatch, OwnedEvent},
        schema::Schema,
    };
    use laddu_expr::{
        P4Component, atan2, complex, dot, event_p4_component, event_scalar, matrix, parameter,
        polar_complex, solve, vector,
    };

    use super::*;

    fn evaluate(expr: &laddu_expr::Expr) -> Complex64 {
        let model = CompiledModel::from_expr(expr).unwrap();
        let params = Arc::new(model.params().clone()).default_values();
        CpuBackend.prepare(&model).evaluate(&params).unwrap()
    }

    fn finite_difference(plan: &CpuPlan, params: &ParamValues, parameter: usize) -> Complex64 {
        let h = 1.0e-6;
        let mut plus = params.clone();
        let mut minus = params.clone();
        let id = params.layout().free_params()[parameter];
        let free_id = params.layout().free_id(id).unwrap().unwrap();
        let value = params.get(id).unwrap();
        plus.set_free(free_id, value + h).unwrap();
        minus.set_free(free_id, value - h).unwrap();
        (plan.evaluate(&plus).unwrap() - plan.evaluate(&minus).unwrap()) / (2.0 * h)
    }

    #[test]
    fn evaluates_scalar_expression_with_parameters() {
        let expr = (2.0 * parameter!("x", initial: 3.0)
            + complex(
                parameter!("re", initial: 1.0),
                parameter!("im", initial: 2.0),
            ))
        .norm_sqr();

        assert_eq!(evaluate(&expr), Complex64::from(53.0));
    }

    #[test]
    fn forward_gradients_match_scalar_complex_finite_differences() {
        let x = laddu_expr::Expr::from(parameter!("x", initial: 0.4));
        let y = laddu_expr::Expr::from(parameter!("y", initial: -0.2));
        let expression = complex(x.clone().sin(), y.clone().exp()).norm_sqr() + (x * y).cos();
        let model = CompiledModel::from_expr(&expression).unwrap();
        let params = Arc::new(model.params().clone()).default_values();
        let plan = CpuBackend.prepare(&model);
        let result = plan.evaluate_with_gradient(&params).unwrap();

        for (parameter, derivative) in result.gradient().iter().enumerate() {
            let expected = finite_difference(&plan, &params, parameter);
            assert!((derivative - expected).norm() < 1.0e-8);
        }
    }

    #[cfg(feature = "jit")]
    #[test]
    fn jit_gradient_reduction_matches_interpreter() {
        let x = event_scalar("x").real();
        let scale = laddu_expr::Expr::from(parameter!("scale", initial: 0.4));
        let phase = laddu_expr::Expr::from(parameter!("phase", initial: -0.2));
        let intensity =
            complex((x.clone() * &scale).sin(), (x.clone() + phase).cos()).norm_sqr() + 0.5;
        let expression = complex(intensity, x * scale);
        let model = CompiledModel::from_expr(&expression).unwrap();
        let params = Arc::new(model.params().clone()).default_values();
        let automatic = CpuBackend.prepare(&model);
        let interpreted =
            CpuBackend.prepare_with_execution_mode(&model, CpuExecutionMode::Interpreter);
        assert!(matches!(
            automatic.gradient_executor,
            GradientExecutor::Jit(_)
        ));

        let schema = Arc::new(Schema::new(std::iter::empty::<&str>(), ["x"], true).unwrap());
        let batch = EventBatch::from_events(
            schema,
            [
                OwnedEvent::weighted(vec![], vec![0.25], 0.5),
                OwnedEvent::weighted(vec![], vec![0.75], 1.5),
                OwnedEvent::weighted(vec![], vec![1.25], 2.0),
            ],
        )
        .unwrap();
        let actual = automatic
            .evaluate_cache_with_gradient(&params, &automatic.cache_event_batch(&batch).unwrap())
            .unwrap();
        let expected = interpreted
            .evaluate_cache_with_gradient(&params, &interpreted.cache_event_batch(&batch).unwrap())
            .unwrap();
        for (actual, expected) in actual.iter().zip(&expected) {
            assert!((actual.value() - expected.value()).norm() < 1.0e-12);
            for (actual, expected) in actual.gradient().iter().zip(expected.gradient()) {
                assert!((actual - expected).norm() < 1.0e-12);
            }
        }
        let dataset = Dataset::from_batch(batch);
        let execution = CpuExecution::local(crate::CpuExecutionOptions {
            threads: crate::ThreadPolicy::Serial,
            ..crate::CpuExecutionOptions::default()
        })
        .unwrap();
        let automatic_data = automatic.prepare_dataset(&execution, &dataset).unwrap();
        let interpreted_data = interpreted.prepare_dataset(&execution, &dataset).unwrap();
        let automatic_result = automatic
            .reduce_with_gradient(
                &execution,
                &params,
                &automatic_data,
                ReductionPlan::weighted_log_positive_real(),
            )
            .unwrap();
        let interpreted_result = interpreted
            .reduce_with_gradient(
                &execution,
                &params,
                &interpreted_data,
                ReductionPlan::weighted_log_positive_real(),
            )
            .unwrap();

        assert!((automatic_result.value() - interpreted_result.value()).abs() < 1.0e-12);
        for (actual, expected) in automatic_result
            .gradient()
            .iter()
            .zip(interpreted_result.gradient())
        {
            assert!((actual - expected).abs() < 1.0e-12);
        }
    }

    #[test]
    fn forward_gradients_cover_unary_atan2_and_zero_products() {
        let x = laddu_expr::Expr::from(parameter!("x", initial: 0.8));
        let y = laddu_expr::Expr::from(parameter!("y", initial: 0.0));
        let z = complex(x.clone(), y.clone());
        let expression = x.clone().sqrt()
            + x.clone().log()
            + x.clone().powi(-2)
            + x.clone().sin()
            + x.clone().cos()
            + x.clone().exp()
            + z.clone().conj().real()
            + z.clone().imag()
            + z.norm_sqr()
            + atan2(y.clone(), x.clone())
            + y * x;
        let model = CompiledModel::from_expr(&expression).unwrap();
        let params = Arc::new(model.params().clone()).default_values();
        let plan = CpuBackend.prepare(&model);
        let result = plan.evaluate_with_gradient(&params).unwrap();

        for (parameter, derivative) in result.gradient().iter().enumerate() {
            let expected = finite_difference(&plan, &params, parameter);
            assert!((derivative - expected).norm() < 1.0e-7);
        }
    }

    #[test]
    fn solve_gradients_match_finite_differences_for_matrix_and_rhs_parameters() {
        let a = laddu_expr::Expr::from(parameter!("a", initial: 2.0));
        let b = laddu_expr::Expr::from(parameter!("b", initial: 0.3));
        let r = laddu_expr::Expr::from(parameter!("r", initial: 1.2));
        let solution = solve(
            matrix([[a, b], [0.2.into(), 1.7.into()]]),
            vector([r, complex(0.5, -0.1)]),
        );
        let expression = dot(solution, vector([complex(1.0, 0.2), (-0.4).into()]));
        let model = CompiledModel::from_expr(&expression).unwrap();
        let params = Arc::new(model.params().clone()).default_values();
        let plan = CpuBackend.prepare(&model);
        let result = plan.evaluate_with_gradient(&params).unwrap();

        for (parameter, derivative) in result.gradient().iter().enumerate() {
            let expected = finite_difference(&plan, &params, parameter);
            assert!((derivative - expected).norm() < 1.0e-8);
        }
    }

    #[test]
    fn evaluates_event_scalars() {
        let expr = laddu_expr::event_scalar("x") * 2.0;
        let model = CompiledModel::from_expr(&expr).unwrap();
        let params = Arc::new(model.params().clone()).default_values();
        let plan = CpuBackend.prepare(&model);
        let event = HashMap::from([("x".to_owned(), Complex64::from(3.0))]);

        assert_eq!(
            plan.evaluate_with_event(&params, &event).unwrap(),
            Complex64::from(6.0)
        );
    }

    #[test]
    fn scalar_kernel_ir_preserves_typed_dependency_classes() {
        let coefficient = complex(parameter!("re", initial: 2.0), 1.0);
        let expr = coefficient * event_scalar("x");
        let model = CompiledModel::from_expr(&expr).unwrap();
        let plan = CpuBackend.prepare(&model);
        let kernel = plan.scalar_kernel.as_ref().unwrap();

        assert!(
            kernel
                .values()
                .iter()
                .any(|value| value.class == KernelValueClass::Invariant)
        );
        assert!(
            kernel
                .values()
                .iter()
                .any(|value| value.class == KernelValueClass::Event)
        );
        let root = &kernel.values()[kernel.root().index()];
        assert_eq!(root.kind, KernelValueKind::Complex);
        assert_eq!(root.class, KernelValueClass::Event);
        assert!(matches!(root.instruction, KernelInstruction::Mul(_)));
    }

    #[test]
    fn cpu_execution_mode_selects_retained_interpreter() {
        let expr = laddu_expr::Expr::from(parameter!("x", initial: 2.0)).exp() + 1.0;
        let model = CompiledModel::from_expr(&expr).unwrap();
        let params = Arc::new(model.params().clone()).default_values();
        let automatic = CpuBackend.prepare(&model);
        let interpreted =
            CpuBackend.prepare_with_execution_mode(&model, CpuExecutionMode::Interpreter);

        assert!(matches!(
            interpreted.scalar_executor,
            Some(ScalarExecutor::Interpreter(_))
        ));
        assert_eq!(
            automatic.evaluate(&params).unwrap(),
            interpreted.evaluate(&params).unwrap()
        );

        let empty_model = CompiledModel::from_expr(&laddu_expr::Expr::from(1.0)).unwrap();
        let wrong_params = empty_model.params().default_values();
        assert!(matches!(
            automatic.evaluate(&wrong_params),
            Err(RuntimeError::Parameter(_))
        ));
    }

    #[cfg(feature = "jit")]
    #[test]
    fn auto_jit_matches_interpreter_for_supported_real_arithmetic() {
        let x = laddu_expr::Expr::from(parameter!("x", initial: 2.0));
        let y = laddu_expr::Expr::from(parameter!("y", initial: -0.5));
        let expr = (x * 3.0 + y) / 2.0;
        let model = CompiledModel::from_expr(&expr).unwrap();
        let params = Arc::new(model.params().clone()).default_values();
        let automatic = CpuBackend.prepare(&model);
        let interpreted =
            CpuBackend.prepare_with_execution_mode(&model, CpuExecutionMode::Interpreter);

        assert!(matches!(
            automatic.scalar_executor,
            Some(ScalarExecutor::Jit(_))
        ));
        assert!(matches!(
            automatic.gradient_executor,
            GradientExecutor::Jit(_)
        ));
        assert_eq!(
            automatic.evaluate(&params).unwrap(),
            interpreted.evaluate(&params).unwrap()
        );
        assert_eq!(
            automatic.evaluate_with_gradient(&params).unwrap(),
            interpreted.evaluate_with_gradient(&params).unwrap()
        );
    }

    #[cfg(feature = "jit")]
    #[test]
    fn auto_jit_supports_complex_transcendentals() {
        let expr = laddu_expr::Expr::from(parameter!("x", initial: 2.0)).exp();
        let model = CompiledModel::from_expr(&expr).unwrap();
        let plan = CpuBackend.prepare(&model);

        assert!(matches!(plan.scalar_executor, Some(ScalarExecutor::Jit(_))));
        let params = Arc::new(model.params().clone()).default_values();
        assert_eq!(plan.evaluate(&params).unwrap(), Complex64::from(2.0).exp());
    }

    #[test]
    fn evaluates_p4_schema_components_and_atan2() {
        let expr = event_p4_component("ks1", P4Component::E)
            + event_p4_component("ks1", P4Component::Px)
            + atan2(
                event_p4_component("ks1", P4Component::Py),
                event_p4_component("ks1", P4Component::Px),
            );
        let model = CompiledModel::from_expr(&expr).unwrap();
        let params = Arc::new(model.params().clone()).default_values();
        let plan = CpuBackend.prepare(&model);
        let batch = EventBatch::from_events(
            Arc::new(Schema::new(["ks1"], std::iter::empty::<&str>(), false).unwrap()),
            [OwnedEvent::new(
                vec![RealVec4::new(3.0, 4.0, 5.0, 10.0)],
                vec![],
            )],
        )
        .unwrap();

        assert_eq!(
            plan.evaluate_batch(&params, &batch).unwrap()[0],
            Complex64::from(13.0 + 4.0_f64.atan2(3.0))
        );
    }

    #[test]
    fn batch_cache_evaluates_without_original_event_batch() {
        let expr = event_scalar("x").real().sin() * parameter!("scale", initial: 2.0);
        let model = CompiledModel::from_expr(&expr).unwrap();
        let layout = Arc::new(model.params().clone());
        let mut params = layout.default_values();
        let plan = CpuBackend.prepare_with_execution_mode(&model, CpuExecutionMode::Interpreter);
        let batch = EventBatch::from_events(
            Arc::new(Schema::new(std::iter::empty::<&str>(), ["x"], false).unwrap()),
            [
                OwnedEvent::new(vec![], vec![0.5]),
                OwnedEvent::new(vec![], vec![1.0]),
            ],
        )
        .unwrap();
        let cache = plan.cache_event_batch(&batch).unwrap();

        assert_eq!(cache.weights(), &[1.0, 1.0]);
        assert!(matches!(&cache.slots[0], CachedSlot::Real(values) if values.len() == 2));
        assert_eq!(cache.slots[0].resident_bytes(), 2 * size_of::<f64>());
        assert_eq!(
            plan.evaluate_cache(&params, &cache).unwrap(),
            vec![
                Complex64::from(2.0 * 0.5_f64.sin()),
                Complex64::from(2.0 * 1.0_f64.sin())
            ]
        );

        let scale = layout
            .free_id(layout.id("scale").unwrap())
            .unwrap()
            .unwrap();
        params.set_free(scale, 3.0).unwrap();
        assert_eq!(
            plan.evaluate_cache(&params, &cache).unwrap(),
            vec![
                Complex64::from(3.0 * 0.5_f64.sin()),
                Complex64::from(3.0 * 1.0_f64.sin())
            ]
        );
    }

    #[test]
    fn real_cache_slots_use_half_the_scalar_payload_of_complex_slots() {
        let real_model =
            CompiledModel::from_expr(&(parameter!("scale") * event_scalar("x").real().sin()))
                .unwrap();
        let complex_model = CompiledModel::from_expr(
            &(parameter!("scale") * complex(event_scalar("x").sin(), event_scalar("x").cos())),
        )
        .unwrap();
        let batch = EventBatch::from_events(
            Arc::new(Schema::new(std::iter::empty::<&str>(), ["x"], false).unwrap()),
            [
                OwnedEvent::new(vec![], vec![0.5]),
                OwnedEvent::new(vec![], vec![1.0]),
            ],
        )
        .unwrap();
        let real_cache = CpuBackend
            .prepare(&real_model)
            .cache_event_batch(&batch)
            .unwrap();
        let complex_cache = CpuBackend
            .prepare(&complex_model)
            .cache_event_batch(&batch)
            .unwrap();

        assert!(matches!(&real_cache.slots[0], CachedSlot::Real(_)));
        assert!(matches!(&complex_cache.slots[0], CachedSlot::Complex(_)));
        assert_eq!(
            complex_cache.slots[0].resident_bytes(),
            2 * real_cache.slots[0].resident_bytes()
        );
    }

    #[test]
    fn selected_event_only_solve_components_cache_inverse_rows() {
        let expression = solve(
            matrix([[event_scalar("x") + 2.0]]),
            vector([parameter!("rhs", initial: 3.0)]),
        )
        .component(0);
        let model = CompiledModel::from_expr(&expression).unwrap();
        let params = Arc::new(model.params().clone()).default_values();
        let plan = CpuBackend.prepare_with_execution_mode(&model, CpuExecutionMode::Interpreter);
        let scalar_plan = plan.scalar_interpreter_plan().unwrap();
        assert!(!scalar_plan.invariant_instructions.is_empty());
        assert!(!scalar_plan.event_instructions.is_empty());
        let batch = EventBatch::from_events(
            Arc::new(Schema::new(std::iter::empty::<&str>(), ["x"], false).unwrap()),
            [
                OwnedEvent::new(vec![], vec![0.0]),
                OwnedEvent::new(vec![], vec![1.0]),
            ],
        )
        .unwrap();
        let cache = plan.cache_event_batch(&batch).unwrap();

        assert!(cache.factor_slots.is_empty());
        assert_eq!(cache.solve_row_slots.len(), 1);
        assert_eq!(cache.solve_row_slots[0].values.len(), 2);
        assert!(cache.resident_bytes() > 0);
        let first = plan
            .evaluate_cache_row_with_gradient(&params, &cache, 0)
            .unwrap();
        let second = plan
            .evaluate_cache_row_with_gradient(&params, &cache, 1)
            .unwrap();
        assert_eq!(first.value(), Complex64::from(1.5));
        assert_eq!(first.gradient(), &[Complex64::from(0.5)]);
        assert_eq!(second.value(), Complex64::from(1.0));
        assert_eq!(second.gradient(), &[Complex64::from(1.0 / 3.0)]);
    }

    #[test]
    fn cached_solve_component_matches_general_complex_nonsymmetric_solve() {
        let expression = solve(
            matrix([
                [event_scalar("x") + 2.0, Complex64::I.into()],
                [Complex64::new(2.0, -1.0).into(), 3.0.into()],
            ]),
            vector([
                parameter!("p", initial: 1.5),
                parameter!("q", initial: -0.25),
            ]),
        )
        .component(1);
        let model = CompiledModel::from_expr(&expression).unwrap();
        let params = Arc::new(model.params().clone()).default_values();
        let plan = CpuBackend.prepare(&model);
        assert!(plan.solve_components.iter().any(Option::is_some));

        let event = HashMap::from([("x".to_owned(), Complex64::from(0.75))]);
        let direct = plan
            .evaluate_with_event_and_gradient(&params, &event)
            .unwrap();
        let batch = EventBatch::from_events(
            Arc::new(Schema::new(std::iter::empty::<&str>(), ["x"], false).unwrap()),
            [OwnedEvent::new(vec![], vec![0.75])],
        )
        .unwrap();
        let cache = plan.cache_event_batch(&batch).unwrap();
        let cached = plan
            .evaluate_cache_row_with_gradient(&params, &cache, 0)
            .unwrap();

        assert!((cached.value() - direct.value()).norm() < 1.0e-12);
        for (cached, direct) in cached.gradient().iter().zip(direct.gradient()) {
            assert!((cached - direct).norm() < 1.0e-12);
        }
    }

    #[cfg(feature = "jit")]
    #[test]
    fn jit_gradient_reduction_handles_cached_solve_rows() {
        let expression = solve(
            matrix([
                [event_scalar("x") + 2.0, Complex64::I.into()],
                [Complex64::new(2.0, -1.0).into(), 3.0.into()],
            ]),
            vector([
                parameter!("p", initial: 1.5),
                parameter!("q", initial: -0.25),
            ]),
        )
        .component(1);
        let model = CompiledModel::from_expr(&expression).unwrap();
        let params = Arc::new(model.params().clone()).default_values();
        let automatic = CpuBackend.prepare(&model);
        let interpreted =
            CpuBackend.prepare_with_execution_mode(&model, CpuExecutionMode::Interpreter);
        assert!(matches!(
            automatic.gradient_executor,
            GradientExecutor::Jit(_)
        ));

        let dataset = Dataset::from_batch(
            EventBatch::from_events(
                Arc::new(Schema::new(std::iter::empty::<&str>(), ["x"], false).unwrap()),
                [
                    OwnedEvent::new(vec![], vec![0.25]),
                    OwnedEvent::new(vec![], vec![0.75]),
                    OwnedEvent::new(vec![], vec![1.25]),
                ],
            )
            .unwrap(),
        );
        let execution = CpuExecution::local(crate::CpuExecutionOptions {
            threads: crate::ThreadPolicy::Fixed(2),
            ..crate::CpuExecutionOptions::default()
        })
        .unwrap();
        let automatic_data = automatic.prepare_dataset(&execution, &dataset).unwrap();
        let interpreted_data = interpreted.prepare_dataset(&execution, &dataset).unwrap();
        let actual = automatic
            .reduce_with_gradient(
                &execution,
                &params,
                &automatic_data,
                ReductionPlan::weighted_real(),
            )
            .unwrap();
        let expected = interpreted
            .reduce_with_gradient(
                &execution,
                &params,
                &interpreted_data,
                ReductionPlan::weighted_real(),
            )
            .unwrap();

        assert!((actual.value() - expected.value()).abs() < 1.0e-12);
        for (actual, expected) in actual.gradient().iter().zip(expected.gradient()) {
            assert!((actual - expected).abs() < 1.0e-12);
        }
    }

    #[test]
    fn batch_cache_reports_missing_event_columns() {
        let expr = event_scalar("missing");
        let model = CompiledModel::from_expr(&expr).unwrap();
        let plan = CpuBackend.prepare(&model);
        let batch = EventBatch::from_events(
            Arc::new(Schema::new(std::iter::empty::<&str>(), ["x"], false).unwrap()),
            [OwnedEvent::new(vec![], vec![0.5])],
        )
        .unwrap();

        assert!(matches!(
            plan.cache_event_batch(&batch),
            Err(RuntimeError::MissingEventColumn(name)) if name == "missing"
        ));
    }

    #[test]
    fn cached_dataset_preserves_transformed_batches_and_weights() {
        let expr = event_scalar("x") * parameter!("scale", initial: 2.0);
        let model = CompiledModel::from_expr(&expr).unwrap();
        let params = Arc::new(model.params().clone()).default_values();
        let plan = CpuBackend.prepare(&model);
        let schema = Arc::new(Schema::new(std::iter::empty::<&str>(), ["x"], true).unwrap());
        let batch = EventBatch::from_events(
            schema,
            [
                OwnedEvent::weighted(vec![], vec![0.5], 2.0),
                OwnedEvent::weighted(vec![], vec![1.0], 3.0),
            ],
        )
        .unwrap();
        let dataset = Dataset::from_batch(batch).filter(|event| event.scalar(0) > 0.75);
        let cached = plan.cache_dataset(&dataset).unwrap();

        assert_eq!(cached.len(), 1);
        assert_eq!(cached.batches()[0].weights(), &[3.0]);
        assert_eq!(cached.batches()[0].sum_weights(), 3.0);
        assert_eq!(
            plan.evaluate_cached_dataset(&params, &cached).unwrap(),
            vec![Complex64::from(2.0)]
        );
    }

    #[test]
    fn cached_dataset_weighted_reductions_match_dataset_path() {
        let expr = event_scalar("x") * parameter!("scale", initial: 2.0);
        let model = CompiledModel::from_expr(&expr).unwrap();
        let params = Arc::new(model.params().clone()).default_values();
        let plan = CpuBackend.prepare(&model);
        let schema = Arc::new(Schema::new(std::iter::empty::<&str>(), ["x"], true).unwrap());
        let first = EventBatch::from_events(
            Arc::clone(&schema),
            [
                OwnedEvent::weighted(vec![], vec![1.0], 2.0),
                OwnedEvent::weighted(vec![], vec![2.0], 3.0),
            ],
        )
        .unwrap();
        let second =
            EventBatch::from_events(schema, [OwnedEvent::weighted(vec![], vec![3.0], 4.0)])
                .unwrap();
        let dataset = Dataset::from_batches(vec![first, second]).unwrap();
        let cached = plan.cache_dataset(&dataset).unwrap();

        let expected = dataset.weighted_sum(|event| 2.0 * event.scalar(0)).unwrap();
        assert_eq!(cached.sum_weights(), dataset.sum_weights().unwrap());
        assert_eq!(
            plan.weighted_sum_cached(&params, &cached, |value| value.re)
                .unwrap(),
            expected
        );
        assert_eq!(
            plan.weighted_complex_sum_cached(&params, &cached, |value| value * Complex64::I)
                .unwrap(),
            Complex64::I * expected
        );
        assert_eq!(
            plan.par_weighted_sum_cached(&params, &cached, |value| value.re)
                .unwrap(),
            expected
        );
        assert_eq!(
            plan.par_weighted_complex_sum_cached(&params, &cached, |value| value * Complex64::I)
                .unwrap(),
            Complex64::I * expected
        );
        let serial_gradient = plan
            .try_weighted_real_sum_with_gradient_cached(&params, &cached, |value| {
                Ok::<_, RuntimeError>((value.re.powi(2), 2.0 * value.re))
            })
            .unwrap();
        let parallel_gradient = plan
            .par_try_weighted_real_sum_with_gradient_cached(&params, &cached, |value| {
                Ok::<_, RuntimeError>((value.re.powi(2), 2.0 * value.re))
            })
            .unwrap();
        assert_eq!(serial_gradient, parallel_gradient);
    }

    #[test]
    fn reduction_plans_match_across_storage_and_thread_policies() {
        let expr = event_scalar("x") * parameter!("scale", initial: 2.0) + 1.0;
        let model = CompiledModel::from_expr(&expr).unwrap();
        let params = Arc::new(model.params().clone()).default_values();
        let plan = CpuBackend.prepare(&model);
        let schema = Arc::new(Schema::new(std::iter::empty::<&str>(), ["x"], true).unwrap());
        let first = EventBatch::from_events(
            Arc::clone(&schema),
            [
                OwnedEvent::weighted(vec![], vec![1.0], 2.0),
                OwnedEvent::weighted(vec![], vec![2.0], 3.0),
            ],
        )
        .unwrap();
        let second =
            EventBatch::from_events(schema, [OwnedEvent::weighted(vec![], vec![3.0], 4.0)])
                .unwrap();
        let resident = Dataset::from_batches(vec![first, second]).unwrap();
        let datasets = [resident.clone(), resident.streaming()];
        let executions = [
            CpuExecution::local(crate::CpuExecutionOptions {
                threads: crate::ThreadPolicy::Serial,
                ..crate::CpuExecutionOptions::default()
            })
            .unwrap(),
            CpuExecution::local(crate::CpuExecutionOptions {
                threads: crate::ThreadPolicy::Fixed(2),
                ..crate::CpuExecutionOptions::default()
            })
            .unwrap(),
        ];
        let expected_real = 49.0;
        let expected_log = 2.0 * 3.0_f64.ln() + 3.0 * 5.0_f64.ln() + 4.0 * 7.0_f64.ln();
        let expected_log_gradient = 2.0 / 3.0 + 6.0 / 5.0 + 12.0 / 7.0;

        for execution in &executions {
            for dataset in &datasets {
                let prepared = plan.prepare_dataset(execution, dataset).unwrap();
                assert_eq!(
                    plan.reduce(
                        execution,
                        &params,
                        &prepared,
                        ReductionPlan::weighted_real(),
                    )
                    .unwrap(),
                    expected_real
                );
                assert_eq!(
                    plan.reduce(
                        execution,
                        &params,
                        &prepared,
                        ReductionPlan::weighted_positive_real(),
                    )
                    .unwrap(),
                    expected_real
                );
                let evaluation = plan
                    .reduce_with_gradient(
                        execution,
                        &params,
                        &prepared,
                        ReductionPlan::weighted_log_positive_real(),
                    )
                    .unwrap();
                assert!((evaluation.value() - expected_log).abs() < 1.0e-12);
                assert!((evaluation.gradient()[0] - expected_log_gradient).abs() < 1.0e-12);
            }
        }
    }

    #[test]
    fn positive_reduction_reports_the_invalid_value() {
        let model = CompiledModel::from_expr(&event_scalar("x")).unwrap();
        let params = Arc::new(model.params().clone()).default_values();
        let plan = CpuBackend.prepare(&model);
        let dataset = Dataset::from_batch(
            EventBatch::from_events(
                Arc::new(Schema::new(std::iter::empty::<&str>(), ["x"], false).unwrap()),
                [OwnedEvent::new(vec![], vec![-2.0])],
            )
            .unwrap(),
        );
        let execution = CpuExecution::default();
        let prepared = plan.prepare_dataset(&execution, &dataset).unwrap();

        assert!(matches!(
            plan.reduce(
                &execution,
                &params,
                &prepared,
                ReductionPlan::weighted_log_positive_real(),
            ),
            Err(RuntimeError::Reduction(
                laddu_compile::ReductionError::NonPositiveValue {
                    transform: laddu_compile::ReductionTransform::LogPositiveReal,
                    value: -2.0,
                }
            ))
        ));
    }

    #[test]
    fn evaluates_linear_algebra_nodes() {
        let a = matrix([[2.0, 0.0], [0.0, 4.0]]);
        let b = vector([8.0, 12.0]);
        let x = solve(a, b);
        let expr = dot(&x, vector([1.0, 1.0]));
        let model = CompiledModel::from_expr(&expr).unwrap();
        let params = Arc::new(model.params().clone()).default_values();
        let plan = CpuBackend.prepare_with_execution_mode(&model, CpuExecutionMode::Interpreter);

        assert_eq!(plan.evaluate(&params).unwrap(), Complex64::from(7.0));
        assert_eq!(plan.constant_factors.len(), 1);
        assert!(plan.constant_factors[0].get().is_some());
    }

    #[cfg(feature = "jit")]
    #[test]
    fn auto_jit_matches_interpreter_for_complex_linear_algebra() {
        let diagonal = laddu_expr::Expr::from(parameter!("diagonal", initial: 4.0));
        let matrix = matrix([
            [complex(2.0, 0.5), complex(0.25, -0.1)],
            [complex(-0.2, 0.3), diagonal],
        ]);
        let rhs = vector([complex(8.0, 1.0), complex(12.0, -0.5)]);
        let solution = solve(matrix, rhs);
        let expression = dot(solution, vector([complex(1.0, -0.2), complex(0.5, 0.3)])).exp();
        let model = CompiledModel::from_expr(&expression).unwrap();
        let params = Arc::new(model.params().clone()).default_values();
        let automatic = CpuBackend.prepare(&model);
        let interpreted =
            CpuBackend.prepare_with_execution_mode(&model, CpuExecutionMode::Interpreter);

        assert!(matches!(
            automatic.scalar_executor,
            Some(ScalarExecutor::Jit(_))
        ));
        let actual = automatic.evaluate(&params).unwrap();
        let expected = interpreted.evaluate(&params).unwrap();
        assert!(
            (actual - expected).norm() < 1.0e-12,
            "{actual} != {expected}"
        );
    }

    #[cfg(feature = "jit")]
    #[test]
    fn auto_jit_executes_event_block_with_parameter_dependent_solve() {
        let x = event_scalar("x");
        let coupling = laddu_expr::Expr::from(parameter!("coupling", initial: 0.2));
        let matrix = matrix([
            [x.clone() + 2.0, complex(coupling.clone(), 0.1)],
            [complex(-0.3, coupling), 3.0.into()],
        ]);
        let expression = solve(matrix, vector([x.sin(), complex(x.cos(), 0.5)]))
            .component(1)
            .norm_sqr();
        let model = CompiledModel::from_expr(&expression).unwrap();
        let params = Arc::new(model.params().clone()).default_values();
        let automatic = CpuBackend.prepare(&model);
        let interpreted =
            CpuBackend.prepare_with_execution_mode(&model, CpuExecutionMode::Interpreter);
        let batch = EventBatch::from_events(
            Arc::new(Schema::new(std::iter::empty::<&str>(), ["x"], false).unwrap()),
            [
                OwnedEvent::new(vec![], vec![0.25]),
                OwnedEvent::new(vec![], vec![0.75]),
                OwnedEvent::new(vec![], vec![1.25]),
            ],
        )
        .unwrap();
        let automatic_cache = automatic.cache_event_batch(&batch).unwrap();
        let interpreted_cache = interpreted.cache_event_batch(&batch).unwrap();

        assert!(matches!(
            automatic.scalar_executor,
            Some(ScalarExecutor::Jit(_))
        ));
        assert!(matches!(
            automatic.gradient_executor,
            GradientExecutor::Interpreter
        ));
        let actual = automatic.evaluate_cache(&params, &automatic_cache).unwrap();
        let expected = interpreted
            .evaluate_cache(&params, &interpreted_cache)
            .unwrap();
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (*actual - expected).norm() < 1.0e-12,
                "{actual} != {expected}"
            );
        }
        let actual = automatic
            .evaluate_cache_with_gradient(&params, &automatic_cache)
            .unwrap();
        let expected = interpreted
            .evaluate_cache_with_gradient(&params, &interpreted_cache)
            .unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn optimized_and_unoptimized_plans_evaluate_the_same_expression() {
        let solved = solve(matrix([[2.0, 0.0], [0.0, 4.0]]), vector([8.0, 12.0]));
        let complex_offset = complex(
            parameter!("offset_re", initial: 1.5),
            parameter!("offset_im", initial: -0.5),
        );
        let polar_product = polar_complex(
            parameter!("mag1", initial: 2.0),
            parameter!("phase1", initial: 0.25),
        ) * polar_complex(
            parameter!("mag2", initial: 3.0),
            parameter!("phase2", initial: -0.5),
        );
        let expr = ((laddu_expr::event_scalar("mass") + 0.0) * 1.0
            + dot(solved, vector([1.0, 1.0]))
            + complex_offset.conj().real()
            + polar_product.real()
            + parameter!("unused", initial: 3.0) * 0.0)
            .norm_sqr();
        let no_optimization = CompileOptions::without_optimizations();
        let optimized = CompiledModel::from_expr(&expr).unwrap();
        let unoptimized = CompiledModel::from_expr_with_options(&expr, &no_optimization).unwrap();
        let optimized_params = Arc::new(optimized.params().clone()).default_values();
        let unoptimized_params = Arc::new(unoptimized.params().clone()).default_values();
        let event = HashMap::from([("mass".to_owned(), Complex64::from(2.0))]);

        let optimized = CpuBackend
            .prepare(&optimized)
            .evaluate_with_event_and_gradient(&optimized_params, &event)
            .unwrap();
        let unoptimized = CpuBackend
            .prepare(&unoptimized)
            .evaluate_with_event_and_gradient(&unoptimized_params, &event)
            .unwrap();
        assert_eq!(optimized.value(), unoptimized.value());
        for (optimized, unoptimized) in optimized.gradient().iter().zip(unoptimized.gradient()) {
            assert!((optimized - unoptimized).norm() < 1.0e-12);
        }
    }
}
