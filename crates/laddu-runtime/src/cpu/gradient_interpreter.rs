use laddu_autodiff::{AutodiffResult, gradient_ir};
use laddu_expr::parameters::{ParamId, ParamValues};
use laddu_kernel::ir::{
    GradientKernelIr, KernelInstruction, KernelValueKind, OutputComponent, ScalarKernelIr,
};
use nalgebra::{DMatrix, DVector};
use num::complex::Complex64;

use super::{
    CpuBatchCache, RuntimeError, RuntimeResult, ScalarEvaluationPlan, ScalarEventWorkspace,
    ScalarInvariantValues, ScalarSlot, Value, eval_binary, eval_unary, evaluate_scalar_cache_block,
    matrix_at, matrix_values_row_major, scalar_at, vector_at,
};

#[derive(Clone, Debug)]
pub(super) struct GradientInterpreter {
    real: GradientComponent,
    imag: Option<GradientComponent>,
}

#[derive(Clone, Debug)]
enum GradientComponent {
    Scalar(ScalarEvaluationPlan),
    Generic(GradientKernelIr),
}

#[derive(Clone)]
pub(super) struct GradientBlockState<'a> {
    plan: &'a ScalarEvaluationPlan,
    invariant: ScalarInvariantValues,
    workspace: ScalarEventWorkspace,
    output: Vec<Complex64>,
}

impl GradientBlockState<'_> {
    pub(super) fn evaluate(
        &mut self,
        cache: &CpuBatchCache,
        start: usize,
        end: usize,
    ) -> RuntimeResult<&[Complex64]> {
        evaluate_scalar_cache_block(
            cache,
            start,
            end,
            self.plan,
            &self.invariant,
            &mut self.workspace,
            &mut self.output,
        )?;
        Ok(&self.output)
    }

    pub(super) fn output_count(&self) -> usize {
        self.plan.outputs.len()
    }
}

impl GradientInterpreter {
    pub(super) fn new(primal: &ScalarKernelIr, free_params: &[ParamId]) -> AutodiffResult<Self> {
        let imag = (primal.values()[primal.root().index()].kind == KernelValueKind::Complex)
            .then(|| gradient_ir(primal, free_params, OutputComponent::Imag))
            .transpose()?
            .map(|ir| Self::component(ir, false));
        let real = gradient_ir(primal, free_params, OutputComponent::Real)?;
        Ok(Self {
            real: Self::component(real, true),
            imag,
        })
    }

    fn component(ir: GradientKernelIr, include_value: bool) -> GradientComponent {
        let mut outputs = Vec::with_capacity(ir.outputs().len() + usize::from(include_value));
        if include_value {
            outputs.push(ir.primal_root());
        }
        outputs.extend_from_slice(ir.outputs());
        ScalarEvaluationPlan::from_kernel_values(ir.values(), &outputs)
            .map(GradientComponent::Scalar)
            .unwrap_or(GradientComponent::Generic(ir))
    }

    pub(super) fn evaluate(
        &self,
        params: &ParamValues,
        cache: Option<(&CpuBatchCache, usize)>,
    ) -> RuntimeResult<(Complex64, Vec<Complex64>)> {
        let real = self.evaluate_component(&self.real, params, cache)?;
        let value = real[0];
        let real = real
            .into_iter()
            .skip(1)
            .map(|value| value.re)
            .collect::<Vec<_>>();
        let imag = if let Some(imag) = &self.imag {
            self.evaluate_component(imag, params, cache)?
                .into_iter()
                .map(|value| value.re)
                .collect()
        } else {
            vec![0.0; real.len()]
        };
        Ok((
            value,
            real.into_iter()
                .zip(imag)
                .map(|(re, im)| Complex64::new(re, im))
                .collect(),
        ))
    }

    pub(super) fn prepare_real_blocks(
        &self,
        params: &ParamValues,
    ) -> RuntimeResult<Option<GradientBlockState<'_>>> {
        let GradientComponent::Scalar(plan) = &self.real else {
            return Ok(None);
        };
        Ok(Some(GradientBlockState {
            plan,
            invariant: Self::evaluate_invariants(plan, params)?,
            workspace: ScalarEventWorkspace::default(),
            output: Vec::new(),
        }))
    }

    fn evaluate_component(
        &self,
        component: &GradientComponent,
        params: &ParamValues,
        cache: Option<(&CpuBatchCache, usize)>,
    ) -> RuntimeResult<Vec<Complex64>> {
        match component {
            GradientComponent::Scalar(plan) => self.evaluate_scalar(plan, params, cache),
            GradientComponent::Generic(ir) => {
                let (value, outputs) = self.evaluate_generic(ir, params, cache)?;
                let mut result = Vec::with_capacity(outputs.len() + 1);
                if ir.component() == OutputComponent::Real {
                    result.push(value);
                }
                result.extend(outputs.into_iter().map(Complex64::from));
                Ok(result)
            }
        }
    }

    fn evaluate_scalar(
        &self,
        plan: &ScalarEvaluationPlan,
        params: &ParamValues,
        cache: Option<(&CpuBatchCache, usize)>,
    ) -> RuntimeResult<Vec<Complex64>> {
        let invariant = Self::evaluate_invariants(plan, params)?;
        let mut event = ScalarEventWorkspace {
            real: vec![[0.0; super::SCALAR_BLOCK_SIZE]; plan.event_real_slot_count],
            complex: vec![
                [Complex64::ZERO; super::SCALAR_BLOCK_SIZE];
                plan.event_complex_slot_count
            ],
        };
        if let Some(cache) = cache {
            for instruction in &plan.event_instructions {
                match instruction.output_slot {
                    ScalarSlot::Real(slot) => {
                        event.real[slot][0] = instruction.instruction.evaluate_real(
                            None,
                            Some(cache),
                            &invariant,
                            &event,
                        )?;
                    }
                    ScalarSlot::Complex(slot) => {
                        event.complex[slot][0] = instruction.instruction.evaluate_complex(
                            None,
                            Some(cache),
                            &invariant,
                            &event,
                        )?;
                    }
                }
            }
        } else if !plan.event_instructions.is_empty() {
            return Err(RuntimeError::InvalidShape {
                index: 0,
                message: "event-dependent gradient requires an event cache".into(),
            });
        }
        Ok(plan
            .outputs
            .iter()
            .map(|output| output.complex_value(&invariant, &event))
            .collect())
    }

    fn evaluate_invariants(
        plan: &ScalarEvaluationPlan,
        params: &ParamValues,
    ) -> RuntimeResult<ScalarInvariantValues> {
        let mut invariant = ScalarInvariantValues {
            real: vec![0.0; plan.invariant_real_slot_count],
            complex: vec![Complex64::ZERO; plan.invariant_complex_slot_count],
        };
        let event = ScalarEventWorkspace::default();
        for instruction in &plan.invariant_instructions {
            match instruction.output_slot {
                ScalarSlot::Real(slot) => {
                    invariant.real[slot] = instruction.instruction.evaluate_real(
                        Some(params),
                        None,
                        &invariant,
                        &event,
                    )?;
                }
                ScalarSlot::Complex(slot) => {
                    invariant.complex[slot] = instruction.instruction.evaluate_complex(
                        Some(params),
                        None,
                        &invariant,
                        &event,
                    )?;
                }
            }
        }
        Ok(invariant)
    }

    fn evaluate_generic(
        &self,
        ir: &GradientKernelIr,
        params: &ParamValues,
        cache: Option<(&CpuBatchCache, usize)>,
    ) -> RuntimeResult<(Complex64, Vec<f64>)> {
        let mut values = Vec::with_capacity(ir.values().len());
        for (index, value) in ir.values().iter().enumerate() {
            let evaluated = match &value.instruction {
                KernelInstruction::Cached(slot) => {
                    let (cache, row) = cache.ok_or_else(|| RuntimeError::InvalidShape {
                        index,
                        message: "cached instruction requires an event cache".into(),
                    })?;
                    cache
                        .slots
                        .get(*slot)
                        .ok_or_else(|| RuntimeError::InvalidShape {
                            index,
                            message: format!("cache slot {slot} is out of bounds"),
                        })?
                        .value(row)?
                }
                KernelInstruction::RealConstant(value) => Value::Scalar((*value).into()),
                KernelInstruction::ComplexConstant(value) => Value::Scalar(*value),
                KernelInstruction::Parameter(id) => Value::Scalar(
                    params
                        .get(*id)
                        .map_err(|error| RuntimeError::Parameter(error.to_string()))?
                        .into(),
                ),
                KernelInstruction::Unary { op, input } => {
                    Value::Scalar(eval_unary(*op, scalar_at(&values, input.index())?))
                }
                KernelInstruction::Binary { op, lhs, rhs } => Value::Scalar(eval_binary(
                    *op,
                    scalar_at(&values, lhs.index())?,
                    scalar_at(&values, rhs.index())?,
                )),
                KernelInstruction::Add(terms) => Value::Scalar(
                    terms
                        .iter()
                        .map(|term| scalar_at(&values, term.index()))
                        .sum::<RuntimeResult<Complex64>>()?,
                ),
                KernelInstruction::Mul(factors) => {
                    Value::Scalar(factors.iter().try_fold(Complex64::ONE, |product, factor| {
                        Ok::<_, RuntimeError>(product * scalar_at(&values, factor.index())?)
                    })?)
                }
                KernelInstruction::Complex { re, im } => Value::Scalar(Complex64::new(
                    scalar_at(&values, re.index())?.re,
                    scalar_at(&values, im.index())?.re,
                )),
                KernelInstruction::Vector(entries) => Value::Vector(
                    entries
                        .iter()
                        .map(|entry| scalar_at(&values, entry.index()))
                        .collect::<RuntimeResult<_>>()?,
                ),
                KernelInstruction::Matrix {
                    rows,
                    cols,
                    elements,
                } => Value::Matrix {
                    rows: *rows,
                    cols: *cols,
                    values: elements
                        .iter()
                        .map(|entry| scalar_at(&values, entry.index()))
                        .collect::<RuntimeResult<_>>()?,
                },
                KernelInstruction::Component {
                    input,
                    index: element,
                } => {
                    let vector = vector_at(&values, input.index())?;
                    Value::Scalar(*vector.get(*element).ok_or_else(|| {
                        RuntimeError::InvalidShape {
                            index,
                            message: format!(
                                "component index {element} is out of bounds for len {}",
                                vector.len()
                            ),
                        }
                    })?)
                }
                KernelInstruction::MatrixElement { input, row, col } => {
                    let (rows, cols, matrix) = matrix_at(&values, input.index())?;
                    if *row >= rows || *col >= cols {
                        return Err(RuntimeError::InvalidShape {
                            index,
                            message: format!(
                                "matrix element ({row}, {col}) is out of bounds for {rows}x{cols}"
                            ),
                        });
                    }
                    Value::Scalar(matrix[row * cols + col])
                }
                KernelInstruction::MatMul { lhs, rhs } => {
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
                    let output = DMatrix::from_row_slice(lhs_rows, lhs_cols, lhs)
                        * DMatrix::from_row_slice(rhs_rows, rhs_cols, rhs);
                    Value::Matrix {
                        rows: output.nrows(),
                        cols: output.ncols(),
                        values: matrix_values_row_major(&output),
                    }
                }
                KernelInstruction::MatVec { matrix, vector } => {
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
                    let output = DMatrix::from_row_slice(rows, cols, matrix)
                        * DVector::from_row_slice(vector);
                    Value::Vector(output.iter().copied().collect())
                }
                KernelInstruction::Dot { lhs, rhs } => {
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
                KernelInstruction::Solve { matrix, rhs } => {
                    let (rows, cols, matrix) = matrix_at(&values, matrix.index())?;
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
                    let solution = DMatrix::from_row_slice(rows, cols, matrix)
                        .lu()
                        .solve(&DVector::from_row_slice(rhs))
                        .ok_or(RuntimeError::SingularMatrix(index))?;
                    Value::Vector(solution.iter().copied().collect())
                }
                KernelInstruction::SolveRow { row_slot, rhs } => {
                    let (cache, row) = cache.ok_or_else(|| RuntimeError::InvalidShape {
                        index,
                        message: "solve-row instruction requires an event cache".into(),
                    })?;
                    let inverse = cache.solve_row(*row_slot, row)?;
                    if inverse.len() != rhs.len() {
                        return Err(RuntimeError::InvalidShape {
                            index,
                            message: format!(
                                "specialized solve row has len {}, expected {}",
                                inverse.len(),
                                rhs.len()
                            ),
                        });
                    }
                    Value::Scalar(
                        inverse
                            .iter()
                            .zip(rhs)
                            .map(|(coefficient, rhs)| {
                                Ok::<_, RuntimeError>(
                                    coefficient * scalar_at(&values, rhs.index())?,
                                )
                            })
                            .sum::<RuntimeResult<Complex64>>()?,
                    )
                }
                KernelInstruction::SolveRowAdjointElement {
                    row_slot,
                    index: element,
                    len,
                    adjoint,
                } => {
                    let (cache, row) = cache.ok_or_else(|| RuntimeError::InvalidShape {
                        index,
                        message: "solve-row adjoint instruction requires an event cache".into(),
                    })?;
                    let inverse = cache.solve_row(*row_slot, row)?;
                    if inverse.len() != *len {
                        return Err(RuntimeError::InvalidShape {
                            index,
                            message: format!(
                                "specialized solve row has len {}, expected {len}",
                                inverse.len()
                            ),
                        });
                    }
                    Value::Scalar(scalar_at(&values, adjoint.index())? * inverse[*element].conj())
                }
            };
            values.push(evaluated);
        }
        let value = scalar_at(&values, ir.primal_root().index())?;
        let outputs = ir
            .outputs()
            .iter()
            .map(|output| Ok(scalar_at(&values, output.index())?.re))
            .collect::<RuntimeResult<_>>()?;
        Ok((value, outputs))
    }
}
