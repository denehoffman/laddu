use laddu_data::{data::EventBatch, schema::Schema};
use laddu_expr::{BinaryOp, ExprId, ExprNode, P4Component, UnaryOp, parameters::ParamValues};
use laddu_kernel::ir::{GradientKernelIr, KernelInstruction, KernelValue, KernelValueKind};
use nalgebra::{DMatrix, DVector};
use num::complex::{Complex32, Complex64};

use super::layout::{
    F32Value, Value, eval_binary, eval_unary, f32_matrix_at, f32_scalar_at, f32_vector_at,
    matrix_at, matrix_at_optional, matrix_values_row_major, matrix_values_row_major_f32, scalar_at,
    scalar_at_optional, vector_at, vector_at_optional,
};
use super::scalar::{
    OperandRun, SCALAR_BLOCK_SIZE, ScalarEvaluationPlan, ScalarEventWorkspace, ScalarInstruction,
    ScalarInvariantValues, ScalarSlot,
};
use super::{
    CpuBatchCache, CpuPlan, EventLookup, Precision, RuntimeError, RuntimeResult, ValueGradient,
};

#[cfg(feature = "jit")]
use crate::jit::{JitCacheView, JitScalarKernel};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) enum EventColumn {
    Scalar(usize),
    P4Component { col: usize, component: P4Component },
}

#[derive(Clone, Copy)]
pub(super) enum F32KernelInput<'a> {
    Cache(Option<(&'a CpuBatchCache, usize)>),
    Event(&'a dyn EventLookup),
}

impl<'a> F32KernelInput<'a> {
    pub(super) fn cache(self) -> Option<(&'a CpuBatchCache, usize)> {
        match self {
            Self::Cache(cache) => cache,
            Self::Event(_) => None,
        }
    }
}

impl CpuPlan {
    pub(super) fn parameter_value(&self, params: &ParamValues, node: usize) -> RuntimeResult<f64> {
        let id = self.parameter_slots[node].ok_or_else(|| RuntimeError::InvalidShape {
            index: node,
            message: "node is not a parameter".into(),
        })?;
        params
            .get(id)
            .map_err(|err| RuntimeError::Parameter(err.to_string()))
    }

    pub(super) fn evaluate_inner(
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
        if self.precision == Precision::F32 {
            let input = match event {
                Some(event) => F32KernelInput::Event(event),
                None => F32KernelInput::Cache(None),
            };
            return self.evaluate_f32_scalar(params, input);
        }
        let values = self.evaluate_values(params, event)?;
        scalar_at(&values, self.graph.root().index())
    }

    /// Evaluates an event-independent model and its free-parameter gradient.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when parameters are incompatible, the model
    /// requires event data, differentiation or evaluation fails, or a solve is
    /// singular.
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
        if self.precision == Precision::F32 {
            return self.evaluate_f32_gradient(params, F32KernelInput::Cache(None));
        }
        self.require_f64_gradient()?;
        if let Some(interpreter) = self.gradient_interpreter() {
            let (value, gradient) = interpreter.evaluate(params, None)?;
            return Ok(ValueGradient { value, gradient });
        }
        let values = self.evaluate_values(params, None)?;
        self.value_gradient(values, None)
    }

    pub(super) fn require_f64_gradient(&self) -> RuntimeResult<()> {
        if self.precision == Precision::F32 {
            return Err(crate::ExecutionError::UnsupportedCpuF32Gradient.into());
        }
        Ok(())
    }

    pub(super) fn evaluate_f32_scalar(
        &self,
        params: &ParamValues,
        input: F32KernelInput<'_>,
    ) -> RuntimeResult<Complex64> {
        let kernel = self
            .scalar_kernel
            .as_ref()
            .ok_or(crate::ExecutionError::UnsupportedCpuF32Model)?;
        let values = self.evaluate_f32_kernel_values(kernel.values(), params, input)?;
        let value = f32_scalar_at(&values, kernel.root())?;
        Ok(Complex64::new(value.re as f64, value.im as f64))
    }

    pub(super) fn evaluate_f32_kernel_values(
        &self,
        kernel_values: &[KernelValue],
        params: &ParamValues,
        input: F32KernelInput<'_>,
    ) -> RuntimeResult<Vec<F32Value>> {
        let mut values = Vec::with_capacity(kernel_values.len());
        for (index, value) in kernel_values.iter().enumerate() {
            let result = match &value.instruction {
                KernelInstruction::Cached(slot) => self.evaluate_f32_cached_value(
                    *slot,
                    params,
                    input,
                    crate::ExecutionError::UnsupportedCpuF32Model,
                )?,
                KernelInstruction::RealConstant(value) => {
                    F32Value::Scalar(Complex32::from(*value as f32))
                }
                KernelInstruction::ComplexConstant(value) => {
                    F32Value::Scalar(Complex32::new(value.re as f32, value.im as f32))
                }
                KernelInstruction::Parameter(id) => F32Value::Scalar(Complex32::from(
                    params
                        .get(*id)
                        .map_err(|error| RuntimeError::Parameter(error.to_string()))?
                        as f32,
                )),
                KernelInstruction::Unary { op, input } => {
                    F32Value::Scalar(eval_unary(*op, f32_scalar_at(&values, *input)?))
                }
                KernelInstruction::Binary { op, lhs, rhs } => F32Value::Scalar(eval_binary(
                    *op,
                    f32_scalar_at(&values, *lhs)?,
                    f32_scalar_at(&values, *rhs)?,
                )),
                KernelInstruction::Add(terms) => F32Value::Scalar(
                    terms
                        .iter()
                        .map(|id| f32_scalar_at(&values, *id))
                        .sum::<RuntimeResult<Complex32>>()?,
                ),
                KernelInstruction::Mul(factors) => {
                    F32Value::Scalar(factors.iter().try_fold(Complex32::ONE, |product, id| {
                        Ok::<_, RuntimeError>(product * f32_scalar_at(&values, *id)?)
                    })?)
                }
                KernelInstruction::Complex { re, im } => F32Value::Scalar(Complex32::new(
                    f32_scalar_at(&values, *re)?.re,
                    f32_scalar_at(&values, *im)?.re,
                )),
                KernelInstruction::Vector(elements) => F32Value::Vector(
                    elements
                        .iter()
                        .map(|id| f32_scalar_at(&values, *id))
                        .collect::<RuntimeResult<_>>()?,
                ),
                KernelInstruction::Matrix {
                    rows,
                    cols,
                    elements,
                } => F32Value::Matrix {
                    rows: *rows,
                    cols: *cols,
                    values: elements
                        .iter()
                        .map(|id| f32_scalar_at(&values, *id))
                        .collect::<RuntimeResult<_>>()?,
                },
                KernelInstruction::Component {
                    input,
                    index: element,
                } => {
                    let vector = f32_vector_at(&values, *input)?;
                    F32Value::Scalar(*vector.get(*element).ok_or_else(|| {
                        RuntimeError::InvalidShape {
                            index,
                            message: format!(
                                "component index {element} out of bounds for len {}",
                                vector.len()
                            ),
                        }
                    })?)
                }
                KernelInstruction::MatrixElement { input, row, col } => {
                    let (rows, cols, matrix) = f32_matrix_at(&values, *input)?;
                    let Some(offset) = (KernelValueKind::Matrix { rows, cols })
                        .checked_row_major_index(*row, *col)
                    else {
                        return Err(RuntimeError::InvalidShape {
                            index,
                            message: format!(
                                "matrix element ({row}, {col}) out of bounds for shape {rows}x{cols}"
                            ),
                        });
                    };
                    F32Value::Scalar(matrix[offset])
                }
                KernelInstruction::Dot { lhs, rhs } => {
                    let lhs = f32_vector_at(&values, *lhs)?;
                    let rhs = f32_vector_at(&values, *rhs)?;
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
                    F32Value::Scalar(lhs.iter().zip(rhs).map(|(lhs, rhs)| lhs * rhs).sum())
                }
                KernelInstruction::MatVec { matrix, vector } => {
                    let (rows, cols, matrix) = f32_matrix_at(&values, *matrix)?;
                    let vector = f32_vector_at(&values, *vector)?;
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
                    F32Value::Vector(output.iter().copied().collect())
                }
                KernelInstruction::MatMul { lhs, rhs } => {
                    let (lhs_rows, lhs_cols, lhs) = f32_matrix_at(&values, *lhs)?;
                    let (rhs_rows, rhs_cols, rhs) = f32_matrix_at(&values, *rhs)?;
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
                    F32Value::Matrix {
                        rows: output.nrows(),
                        cols: output.ncols(),
                        values: matrix_values_row_major_f32(&output),
                    }
                }
                KernelInstruction::Solve { matrix, rhs } => {
                    let (rows, cols, matrix) = f32_matrix_at(&values, *matrix)?;
                    let rhs = f32_vector_at(&values, *rhs)?;
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
                    F32Value::Vector(solution.iter().copied().collect())
                }
                KernelInstruction::SolveRow { row_slot, rhs } => {
                    let (cache, row) = input
                        .cache()
                        .ok_or(crate::ExecutionError::UnsupportedCpuF32Model)?;
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
                    F32Value::Scalar(
                        inverse
                            .iter()
                            .zip(rhs)
                            .map(|(coefficient, rhs)| {
                                Ok::<_, RuntimeError>(
                                    Complex32::new(coefficient.re as f32, coefficient.im as f32)
                                        * f32_scalar_at(&values, *rhs)?,
                                )
                            })
                            .sum::<RuntimeResult<Complex32>>()?,
                    )
                }
                KernelInstruction::SolveRowAdjointElement {
                    row_slot,
                    index: element,
                    len,
                    adjoint,
                } => {
                    let (cache, row) = input
                        .cache()
                        .ok_or(crate::ExecutionError::UnsupportedCpuF32Model)?;
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
                    let coefficient = inverse[*element];
                    F32Value::Scalar(
                        f32_scalar_at(&values, *adjoint)?
                            * Complex32::new(coefficient.re as f32, coefficient.im as f32).conj(),
                    )
                }
            };
            values.push(result);
        }
        Ok(values)
    }

    pub(super) fn evaluate_f32_cached_value(
        &self,
        slot: usize,
        params: &ParamValues,
        input: F32KernelInput<'_>,
        missing_input: crate::ExecutionError,
    ) -> RuntimeResult<F32Value> {
        match input {
            F32KernelInput::Cache(Some((cache, row))) => {
                Ok(F32Value::from_value(cache.value(slot, row)?))
            }
            F32KernelInput::Cache(None) => Err(missing_input.into()),
            F32KernelInput::Event(event) => {
                let entry =
                    self.cache_plan
                        .entries()
                        .get(slot)
                        .ok_or(RuntimeError::InvalidShape {
                            index: self.graph.root().index(),
                            message: format!("cache slot {slot} is out of bounds"),
                        })?;
                let values = self.evaluate_values(params, Some(event))?;
                let value = values
                    .get(entry.node().index())
                    .ok_or(RuntimeError::InvalidShape {
                        index: entry.node().index(),
                        message: "cached node is out of bounds".into(),
                    })?
                    .clone();
                Ok(F32Value::from_value(value))
            }
        }
    }

    pub(super) fn evaluate_f32_gradient(
        &self,
        params: &ParamValues,
        input: F32KernelInput<'_>,
    ) -> RuntimeResult<ValueGradient> {
        let real_ir = self
            .f32_gradient_fallback_real
            .as_ref()
            .ok_or(crate::ExecutionError::UnsupportedCpuF32Model)?;
        let mut real = Vec::new();
        let (value, _) =
            self.evaluate_f32_gradient_component_prepared(real_ir, params, input, &mut real)?;
        let imag = if let Some(imag_ir) = self.f32_gradient_fallback_imag.as_ref() {
            let mut imag = Vec::new();
            self.evaluate_f32_gradient_component_prepared(imag_ir, params, input, &mut imag)?;
            imag
        } else {
            vec![0.0; real.len()]
        };
        Ok(ValueGradient {
            value,
            gradient: real
                .into_iter()
                .zip(imag)
                .map(|(re, im)| Complex64::new(re as f64, im as f64))
                .collect(),
        })
    }

    pub(super) fn evaluate_f32_gradient_component_prepared<'a>(
        &self,
        ir: &GradientKernelIr,
        params: &ParamValues,
        input: F32KernelInput<'_>,
        gradient: &'a mut Vec<f32>,
    ) -> RuntimeResult<(Complex64, &'a [f32])> {
        let values = self.evaluate_f32_kernel_values(ir.values(), params, input)?;
        let value = f32_scalar_at(&values, ir.primal_root())?;
        gradient.clear();
        gradient.reserve(ir.outputs().len());
        for output in ir.outputs() {
            gradient.push(f32_scalar_at(&values, *output)?.re);
        }
        Ok((Complex64::new(value.re as f64, value.im as f64), gradient))
    }

    pub(super) fn solve_primal(
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

    pub(super) fn event_columns(&self, schema: &Schema) -> RuntimeResult<Vec<Option<EventColumn>>> {
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

    pub(super) fn evaluate_cache_values_for_row(
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
                        P4Component::Px => p4.px,
                        P4Component::Py => p4.py,
                        P4Component::Pz => p4.pz,
                        P4Component::E => p4.e,
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

    pub(super) fn evaluate_values(
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
                    Value::Scalar(Complex64::from(
                        event
                            .scalar(name)
                            .ok_or_else(|| RuntimeError::MissingEventScalar(name.to_string()))?,
                    ))
                }
                ExprNode::EventP4Component { name, component } => {
                    let Some(event) = event else {
                        return Err(RuntimeError::MissingEventScalar(format!(
                            "{name}.{}",
                            component.label()
                        )));
                    };
                    Value::Scalar(Complex64::from(
                        event.p4_component(name, *component).ok_or_else(|| {
                            RuntimeError::MissingEventScalar(format!(
                                "{name}.{}",
                                component.label()
                            ))
                        })?,
                    ))
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

    pub(super) fn evaluate_values_from_cache(
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
                        let inverse_row = cache.solve_row(plan.row_slot(), row)?;
                        if inverse_row.len() != plan.dimension() {
                            return Err(RuntimeError::InvalidShape {
                                index,
                                message: format!(
                                    "specialized solve expected row len {}, got {}",
                                    plan.dimension(),
                                    inverse_row.len()
                                ),
                            });
                        }
                        if let Some(elements) = &self.solve_rhs_elements[plan.rhs().index()] {
                            if elements.len() != plan.dimension() {
                                return Err(RuntimeError::InvalidShape {
                                    index,
                                    message: format!(
                                        "specialized solve expected {} RHS elements, got {}",
                                        plan.dimension(),
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
                            let rhs = self.cached_vector_at(&values, plan.rhs())?;
                            if rhs.len() != plan.dimension() {
                                return Err(RuntimeError::InvalidShape {
                                    index,
                                    message: format!(
                                        "specialized solve expected RHS len {}, got {}",
                                        plan.dimension(),
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

    pub(super) fn cached_value_slot(&self, id: ExprId) -> RuntimeResult<usize> {
        self.cached_value_slots[id.index()].ok_or_else(|| RuntimeError::InvalidShape {
            index: id.index(),
            message: "node is not part of the cached evaluation schedule".into(),
        })
    }

    pub(super) fn cached_scalar_at(
        &self,
        values: &[Value],
        id: ExprId,
    ) -> RuntimeResult<Complex64> {
        scalar_at(values, self.cached_value_slot(id)?)
    }

    pub(super) fn cached_vector_at<'a>(
        &self,
        values: &'a [Value],
        id: ExprId,
    ) -> RuntimeResult<&'a [Complex64]> {
        vector_at(values, self.cached_value_slot(id)?)
    }

    pub(super) fn cached_matrix_at<'a>(
        &self,
        values: &'a [Value],
        id: ExprId,
    ) -> RuntimeResult<(usize, usize, &'a [Complex64])> {
        matrix_at(values, self.cached_value_slot(id)?)
    }

    pub(super) fn check_batch_cache(&self, cache: &CpuBatchCache) -> RuntimeResult<()> {
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

    pub(super) fn scalar_invariant_values(
        &self,
        params: &ParamValues,
    ) -> RuntimeResult<Option<ScalarInvariantValues>> {
        if self.precision == Precision::F32 {
            return Ok(None);
        }
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

    pub(super) fn evaluate_cache_row_unchecked(
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

    /// Evaluates every row in a materialized batch cache.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when parameters or cache layout are
    /// incompatible, evaluation fails, or a matrix is singular.
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

    pub(super) fn evaluate_cache_row_prepared(
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
        if self.precision == Precision::F32 {
            return self.evaluate_f32_scalar(params, F32KernelInput::Cache(Some((cache, row))));
        }
        if let (Some(plan), Some(invariant)) = (self.scalar_interpreter_plan(), invariant) {
            return self.evaluate_scalar_cache_row(cache, row, plan, invariant, workspace);
        }
        let values = self.evaluate_values_from_cache(params, cache, row)?;
        self.cached_scalar_at(&values, self.graph.root())
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
        Ok(plan.root().complex_value(invariant, values))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn evaluate_cache_block_prepared(
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
        if self.precision == Precision::F32 {
            output.clear();
            output.reserve(end - start);
            for row in start..end {
                output.push(
                    self.evaluate_f32_scalar(params, F32KernelInput::Cache(Some((cache, row))))?,
                );
            }
            return Ok(());
        }
        if let (Some(plan), Some(invariant)) = (self.scalar_interpreter_plan(), invariant) {
            return evaluate_scalar_cache_block(
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

    pub(super) fn evaluate_cache_row_with_gradient_unchecked(
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
        if self.precision == Precision::F32 {
            return self.evaluate_f32_gradient(params, F32KernelInput::Cache(Some((cache, row))));
        }
        if self.autodiff.mode() == laddu_autodiff::AutodiffMode::Reverse {
            self.require_f64_gradient()?;
            let values = self.evaluate_values_from_cache(params, cache, row)?;
            return self.value_gradient(values, Some((cache, row)));
        }
        if let Some(interpreter) = self.gradient_interpreter() {
            let (value, gradient) = interpreter.evaluate(params, Some((cache, row)))?;
            return Ok(ValueGradient { value, gradient });
        }
        let values = self.evaluate_values_from_cache(params, cache, row)?;
        self.value_gradient(values, Some((cache, row)))
    }

    /// Evaluates every cached row and its free-parameter gradient.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when parameters or cache layout are
    /// incompatible, or differentiation or evaluation fails.
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
        if self.precision == Precision::F32 {
            return (0..cache.len())
                .map(|row| {
                    self.evaluate_f32_gradient(params, F32KernelInput::Cache(Some((cache, row))))
                })
                .collect();
        }
        self.require_f64_gradient()?;
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
}

pub(super) fn evaluate_scalar_cache_block(
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
                                UnaryOp::Neg => -input.block_real_value(invariant, workspace, lane),
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
                                OperandRun::InvariantComplex(_) | OperandRun::EventComplex(_) => {
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
                                OperandRun::InvariantComplex(_) | OperandRun::EventComplex(_) => {
                                    unreachable!("complex operand appeared in real multiply")
                                }
                            }
                        }
                    }
                    ScalarInstruction::Constant(_)
                    | ScalarInstruction::Parameter(_)
                    | ScalarInstruction::Complex { .. }
                    | ScalarInstruction::SolveRow { .. }
                    | ScalarInstruction::SolveRowAdjointElement { .. } => {
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
                                    lhs * operand.block_complex_value(invariant, workspace, lane)
                                })
                                .sum();
                        }
                    }
                    ScalarInstruction::SolveRowAdjointElement {
                        row_slot,
                        index,
                        len,
                        adjoint,
                    } => {
                        for lane in 0..block_len {
                            let inverse_row = cache.solve_row(*row_slot, start + lane)?;
                            if inverse_row.len() != *len {
                                return Err(RuntimeError::InvalidShape {
                                    index: start + lane,
                                    message: format!(
                                        "specialized solve row has len {}, expected {len}",
                                        inverse_row.len()
                                    ),
                                });
                            }
                            workspace.complex[output_slot][lane] = adjoint
                                .block_complex_value(invariant, workspace, lane)
                                * inverse_row[*index].conj();
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
    output.reserve(block_len * plan.outputs.len());
    for lane in 0..block_len {
        for output_operand in &plan.outputs {
            output.push(output_operand.block_complex_value(invariant, workspace, lane));
        }
    }
    Ok(())
}
