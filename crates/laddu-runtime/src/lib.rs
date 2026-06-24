use std::collections::HashMap;

use laddu_compile::{CachePlan, CompiledModel};
use laddu_data::{
    data::{Dataset, EventBatch},
    schema::Schema,
};
use laddu_expr::{
    BinaryOp, ExprGraph, ExprId, ExprNode, UnaryOp, ValueKind,
    parameters::{ParamLayout, ParamValues},
};
use nalgebra::{DMatrix, DVector};
use num::complex::Complex64;
use thiserror::Error;

pub type RuntimeResult<T> = Result<T, RuntimeError>;

#[derive(Clone, Debug, Error, PartialEq)]
pub enum RuntimeError {
    #[error("event scalar `{0}` was requested, but no event lookup was provided")]
    MissingEventScalar(String),
    #[error("node #{index} expected {expected}, got {actual}")]
    TypeMismatch {
        index: usize,
        expected: &'static str,
        actual: &'static str,
    },
    #[error("node #{index} has invalid shape: {message}")]
    InvalidShape { index: usize, message: String },
    #[error("matrix solve failed at node #{0}")]
    SingularMatrix(usize),
    #[error("event cache has {actual} slots, expected {expected}")]
    InvalidCache { expected: usize, actual: usize },
    #[error("event cache was built for a different cache layout")]
    InvalidCacheLayout,
    #[error("event scalar `{0}` was not found in the event batch schema")]
    MissingEventColumn(String),
    #[error("data error: {0}")]
    Data(String),
    #[error("parameter error: {0}")]
    Parameter(String),
}

pub trait EventLookup {
    fn scalar(&self, name: &str) -> Option<Complex64>;
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

#[derive(Clone, Debug)]
pub struct CpuPlan {
    graph: ExprGraph,
    params: ParamLayout,
    cache_plan: CachePlan,
    cache_slots: Vec<Option<usize>>,
    cache_required_nodes: Vec<bool>,
}

impl CpuBackend {
    pub fn prepare(&self, model: &CompiledModel) -> CpuPlan {
        let cache_plan = model.cache_plan().clone();
        let mut cache_slots = vec![None; model.graph().nodes().len()];
        for (slot, entry) in cache_plan.entries().iter().enumerate() {
            cache_slots[entry.node().index()] = Some(slot);
        }
        let cache_required_nodes = cache_required_nodes(model.graph(), &cache_plan);
        CpuPlan {
            graph: model.graph().clone(),
            params: model.params().clone(),
            cache_plan,
            cache_slots,
            cache_required_nodes,
        }
    }
}

impl CpuPlan {
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

    pub fn evaluate_with_event(
        &self,
        params: &ParamValues,
        event: &impl EventLookup,
    ) -> RuntimeResult<Complex64> {
        self.evaluate_inner(params, Some(event))
    }

    pub fn cache_event_batch(&self, batch: &EventBatch) -> RuntimeResult<CpuBatchCache> {
        let event_columns = self.event_scalar_columns(batch.schema())?;
        let mut cache = CpuBatchCache::new(&self.cache_plan, batch.len());
        for row in 0..batch.len() {
            let values = self.evaluate_cache_values_for_row(batch, row, &event_columns)?;
            for (slot, entry) in self.cache_plan.entries().iter().enumerate() {
                let value = values[entry.node().index()]
                    .as_ref()
                    .expect("cacheable node should have been evaluated")
                    .clone();
                cache.push(slot, value)?;
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
        let mut out = Vec::with_capacity(cache.len());
        for row in 0..cache.len() {
            let values = self.evaluate_values_from_cache(params, cache, row)?;
            out.push(scalar_at(&values, self.graph.root().index())?);
        }
        Ok(out)
    }

    pub fn evaluate_batch(
        &self,
        params: &ParamValues,
        batch: &EventBatch,
    ) -> RuntimeResult<Vec<Complex64>> {
        let cache = self.cache_event_batch(batch)?;
        self.evaluate_cache(params, &cache)
    }

    pub fn cache_dataset(&self, dataset: &Dataset) -> RuntimeResult<CpuCachedDataset> {
        let mut batches = Vec::new();
        for batch in dataset
            .batches()
            .map_err(|err| RuntimeError::Data(err.to_string()))?
        {
            let batch = batch.map_err(|err| RuntimeError::Data(err.to_string()))?;
            batches.push(CpuCachedBatch {
                cache: self.cache_event_batch(&batch)?,
            });
        }
        Ok(CpuCachedDataset { batches })
    }

    pub fn evaluate_cached_dataset(
        &self,
        params: &ParamValues,
        dataset: &CpuCachedDataset,
    ) -> RuntimeResult<Vec<Complex64>> {
        let total_len = dataset.batches.iter().map(CpuCachedBatch::len).sum();
        let mut out = Vec::with_capacity(total_len);
        for batch in &dataset.batches {
            out.extend(self.evaluate_cache(params, batch.cache())?);
        }
        Ok(out)
    }

    fn evaluate_inner(
        &self,
        params: &ParamValues,
        event: Option<&dyn EventLookup>,
    ) -> RuntimeResult<Complex64> {
        let values = self.evaluate_values(params, event)?;
        scalar_at(&values, self.graph.root().index())
    }

    fn event_scalar_columns(&self, schema: &Schema) -> RuntimeResult<Vec<Option<usize>>> {
        self.graph
            .nodes()
            .iter()
            .map(|node| {
                if let ExprNode::EventScalar(name) = node {
                    Ok(Some(schema.scalar_index(name).ok_or_else(|| {
                        RuntimeError::MissingEventColumn(name.to_string())
                    })?))
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
        event_columns: &[Option<usize>],
    ) -> RuntimeResult<Vec<Option<Value>>> {
        let mut values = vec![None; self.graph.nodes().len()];

        for (index, node) in self.graph.nodes().iter().enumerate() {
            if !self.cache_required_nodes[index] {
                continue;
            }
            let value = match node {
                ExprNode::RealConst(value) => Value::Scalar(Complex64::from(*value)),
                ExprNode::ComplexConst(value) => Value::Scalar(*value),
                ExprNode::EventScalar(name) => {
                    let col = event_columns[index]
                        .ok_or_else(|| RuntimeError::MissingEventColumn(name.to_string()))?;
                    Value::Scalar(Complex64::from(batch.scalar_at(col, row)))
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
                    let (rows, cols, matrix) = matrix_at_optional(&values, matrix.index())?;
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
                    let matrix = DMatrix::from_row_slice(rows, cols, matrix);
                    let rhs = DVector::from_row_slice(rhs);
                    let solution = matrix
                        .lu()
                        .solve(&rhs)
                        .ok_or(RuntimeError::SingularMatrix(index))?;
                    Value::Vector(solution.iter().copied().collect())
                }
                ExprNode::ScalarParam(_)
                | ExprNode::ComplexScalarParam { .. }
                | ExprNode::PolarComplexScalarParam { .. } => {
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
                ExprNode::ScalarParam(parameter) => Value::Scalar(Complex64::from(param_value(
                    params,
                    &self.params,
                    parameter.name(),
                )?)),
                ExprNode::ComplexScalarParam { re, im } => Value::Scalar(Complex64::new(
                    param_value(params, &self.params, re.name())?,
                    param_value(params, &self.params, im.name())?,
                )),
                ExprNode::PolarComplexScalarParam { mag, phase } => {
                    let mag = param_value(params, &self.params, mag.name())?;
                    let phase = param_value(params, &self.params, phase.name())?;
                    Value::Scalar(Complex64::cis(phase) * mag)
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
                    let matrix = DMatrix::from_row_slice(rows, cols, matrix);
                    let rhs = DVector::from_row_slice(rhs);
                    let solution = matrix
                        .lu()
                        .solve(&rhs)
                        .ok_or(RuntimeError::SingularMatrix(index))?;
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
        let mut values = Vec::with_capacity(self.graph.nodes().len());

        for (index, node) in self.graph.nodes().iter().enumerate() {
            if let Some(slot) = self.cache_slots[index] {
                values.push(cache.value(slot, row)?);
                continue;
            }
            let value = match node {
                ExprNode::RealConst(value) => Value::Scalar(Complex64::from(*value)),
                ExprNode::ComplexConst(value) => Value::Scalar(*value),
                ExprNode::ScalarParam(parameter) => Value::Scalar(Complex64::from(param_value(
                    params,
                    &self.params,
                    parameter.name(),
                )?)),
                ExprNode::ComplexScalarParam { re, im } => Value::Scalar(Complex64::new(
                    param_value(params, &self.params, re.name())?,
                    param_value(params, &self.params, im.name())?,
                )),
                ExprNode::PolarComplexScalarParam { mag, phase } => {
                    let mag = param_value(params, &self.params, mag.name())?;
                    let phase = param_value(params, &self.params, phase.name())?;
                    Value::Scalar(Complex64::cis(phase) * mag)
                }
                ExprNode::EventScalar(name) => {
                    return Err(RuntimeError::MissingEventScalar(name.to_string()));
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
                    let matrix = DMatrix::from_row_slice(rows, cols, matrix);
                    let rhs = DVector::from_row_slice(rhs);
                    let solution = matrix
                        .lu()
                        .solve(&rhs)
                        .ok_or(RuntimeError::SingularMatrix(index))?;
                    Value::Vector(solution.iter().copied().collect())
                }
            };
            values.push(value);
        }

        Ok(values)
    }

    fn check_batch_cache(&self, cache: &CpuBatchCache) -> RuntimeResult<()> {
        if cache.nodes
            == self
                .cache_plan
                .entries()
                .iter()
                .map(|entry| entry.node())
                .collect::<Vec<_>>()
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

#[derive(Clone, Debug, PartialEq)]
pub struct CpuBatchCache {
    len: usize,
    weights: Vec<f64>,
    nodes: Vec<ExprId>,
    slots: Vec<CachedSlot>,
}

impl CpuBatchCache {
    fn new(cache_plan: &CachePlan, len: usize) -> Self {
        Self {
            len,
            weights: vec![1.0; len],
            nodes: cache_plan
                .entries()
                .iter()
                .map(|entry| entry.node())
                .collect(),
            slots: cache_plan
                .entries()
                .iter()
                .map(|entry| CachedSlot::new(entry.value_kind()))
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

    fn set_weights(&mut self, weights: Vec<f64>) {
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
}

#[derive(Clone, Debug, PartialEq)]
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
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct CpuCachedDataset {
    batches: Vec<CpuCachedBatch>,
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
}

#[derive(Clone, Debug, PartialEq)]
enum CachedSlot {
    Scalar(Vec<Complex64>),
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
    fn new(kind: ValueKind) -> Self {
        match kind {
            ValueKind::Real | ValueKind::Complex => Self::Scalar(Vec::new()),
            ValueKind::Vector { len } => Self::Vector {
                len,
                values: Vec::new(),
            },
            ValueKind::Matrix { rows, cols } => Self::Matrix {
                rows,
                cols,
                values: Vec::new(),
            },
        }
    }

    fn push(&mut self, value: Value) -> RuntimeResult<()> {
        match (self, value) {
            (Self::Scalar(values), Value::Scalar(value)) => {
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
            Self::Scalar(values) => values.get(row).copied().map(Value::Scalar).ok_or_else(|| {
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

fn param_value(params: &ParamValues, layout: &ParamLayout, name: &str) -> RuntimeResult<f64> {
    let id = layout
        .id(name)
        .ok_or_else(|| RuntimeError::Parameter(format!("unknown parameter `{name}`")))?;
    params
        .get(id)
        .map_err(|err| RuntimeError::Parameter(err.to_string()))
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

fn cache_required_nodes(graph: &ExprGraph, cache_plan: &CachePlan) -> Vec<bool> {
    let mut required = vec![false; graph.nodes().len()];
    for entry in cache_plan.entries() {
        mark_required(graph, entry.node(), &mut required);
    }
    required
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
    match node {
        ExprNode::Unary { input, .. }
        | ExprNode::Component { input, .. }
        | ExprNode::MatrixElement { input, .. } => vec![*input],
        ExprNode::Binary { lhs, rhs, .. }
        | ExprNode::Complex { re: lhs, im: rhs }
        | ExprNode::MatMul { lhs, rhs }
        | ExprNode::Dot { lhs, rhs } => vec![*lhs, *rhs],
        ExprNode::MatVec { matrix, vector }
        | ExprNode::Solve {
            matrix,
            rhs: vector,
        } => vec![*matrix, *vector],
        ExprNode::NaryAdd { terms } => terms.clone(),
        ExprNode::NaryMul { factors } => factors.clone(),
        ExprNode::Vector { elements } | ExprNode::Matrix { elements, .. } => elements.clone(),
        ExprNode::RealConst(_)
        | ExprNode::ComplexConst(_)
        | ExprNode::ScalarParam(_)
        | ExprNode::ComplexScalarParam { .. }
        | ExprNode::PolarComplexScalarParam { .. }
        | ExprNode::EventScalar(_) => Vec::new(),
    }
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
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use laddu_compile::{CompileOptions, CompiledModel};
    use laddu_data::{
        data::{Dataset, EventBatch, OwnedEvent},
        schema::Schema,
    };
    use laddu_expr::{complex, dot, event_scalar, matrix, parameter, polar_complex, solve, vector};

    use super::*;

    fn evaluate(expr: &laddu_expr::Expr) -> Complex64 {
        let model = CompiledModel::from_expr(expr).unwrap();
        let params = Arc::new(model.params().clone()).default_values();
        CpuBackend.prepare(&model).evaluate(&params).unwrap()
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
    fn batch_cache_evaluates_without_original_event_batch() {
        let expr = event_scalar("x").sin() * parameter!("scale", initial: 2.0);
        let model = CompiledModel::from_expr(&expr).unwrap();
        let layout = Arc::new(model.params().clone());
        let mut params = layout.default_values();
        let plan = CpuBackend.prepare(&model);
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
        assert_eq!(
            plan.evaluate_cache(&params, &cache).unwrap(),
            vec![
                Complex64::from(2.0 * 0.5_f64.sin()),
                Complex64::from(2.0 * 1.0_f64.sin())
            ]
        );

        let scale = layout.id("scale").unwrap();
        params.set_full(scale, 3.0).unwrap();
        assert_eq!(
            plan.evaluate_cache(&params, &cache).unwrap(),
            vec![
                Complex64::from(3.0 * 0.5_f64.sin()),
                Complex64::from(3.0 * 1.0_f64.sin())
            ]
        );
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
        assert_eq!(
            plan.evaluate_cached_dataset(&params, &cached).unwrap(),
            vec![Complex64::from(2.0)]
        );
    }

    #[test]
    fn evaluates_linear_algebra_nodes() {
        let a = matrix([[2.0, 0.0], [0.0, 4.0]]);
        let b = vector([8.0, 12.0]);
        let x = solve(a, b);
        let expr = dot(&x, vector([1.0, 1.0]));

        assert_eq!(evaluate(&expr), Complex64::from(7.0));
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

        assert_eq!(
            CpuBackend
                .prepare(&optimized)
                .evaluate_with_event(&optimized_params, &event)
                .unwrap(),
            CpuBackend
                .prepare(&unoptimized)
                .evaluate_with_event(&unoptimized_params, &event)
                .unwrap()
        );
    }
}
