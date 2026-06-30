use std::{
    collections::HashMap,
    mem::size_of,
    sync::{Arc, OnceLock},
};

use laddu_autodiff::{AutodiffMode, AutodiffPlan, AutodiffResult, SeedKind};
use laddu_compile::{CachePlan, CompiledModel};
use laddu_data::{
    data::accurate::{AccurateComplex64, AccurateF64},
    data::{Dataset, EventBatch},
    schema::Schema,
};
use laddu_expr::{
    BinaryOp, ExprGraph, ExprId, ExprNode, P4Component, UnaryOp, ValueKind,
    parameters::{ParamLayout, ParamValues},
};
use nalgebra::{DMatrix, DVector, Dyn, LU};
use num::complex::Complex64;
use rayon::prelude::*;
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

#[derive(Clone, Debug)]
pub struct CpuPlan {
    graph: ExprGraph,
    params: ParamLayout,
    autodiff: AutodiffPlan,
    cache_plan: CachePlan,
    cache_slots: Vec<Option<usize>>,
    cached_evaluation_nodes: Vec<bool>,
    cache_required_nodes: Vec<bool>,
    factor_matrix_slots: Vec<Option<usize>>,
    factor_matrices: Vec<(ExprId, usize)>,
    constant_factor_slots: Vec<Option<usize>>,
    constant_factors: Vec<Arc<OnceLock<DynamicLu>>>,
}

impl CpuBackend {
    pub fn prepare(&self, model: &CompiledModel) -> CpuPlan {
        self.prepare_with_autodiff_mode(model, AutodiffMode::Forward)
            .expect("forward autodiff supports every compiled expression node")
    }

    pub fn prepare_with_autodiff_mode(
        &self,
        model: &CompiledModel,
        mode: AutodiffMode,
    ) -> AutodiffResult<CpuPlan> {
        let cache_plan = model.cache_plan().clone();
        let mut cache_slots = vec![None; model.graph().nodes().len()];
        for (slot, entry) in cache_plan.entries().iter().enumerate() {
            cache_slots[entry.node().index()] = Some(slot);
        }
        let cached_evaluation_nodes = cached_evaluation_nodes(model.graph(), &cache_slots);
        let cache_required_nodes = cache_required_nodes(model.graph(), &cache_plan);
        let mut factor_matrix_slots = vec![None; model.graph().nodes().len()];
        let mut factor_matrices = Vec::new();
        let mut constant_factor_slots = vec![None; model.graph().nodes().len()];
        let mut constant_factors = Vec::new();
        for node in model.graph().nodes() {
            let ExprNode::Solve { matrix, .. } = node else {
                continue;
            };
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
            autodiff: AutodiffPlan::from_model(model, mode)?,
            cache_plan,
            cache_slots,
            cached_evaluation_nodes,
            cache_required_nodes,
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
        let values = self.evaluate_values(params, None)?;
        self.value_gradient(params, values, None)
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
        self.value_gradient(params, values, None)
    }

    pub fn cache_event_batch(&self, batch: &EventBatch) -> RuntimeResult<CpuBatchCache> {
        let event_columns = self.event_columns(batch.schema())?;
        let mut cache = CpuBatchCache::new(&self.cache_plan, &self.factor_matrices, batch.len());
        for row in 0..batch.len() {
            let values = self.evaluate_cache_values_for_row(batch, row, &event_columns)?;
            for (slot, entry) in self.cache_plan.entries().iter().enumerate() {
                let value = values[entry.node().index()]
                    .as_ref()
                    .expect("cacheable node should have been evaluated")
                    .clone();
                cache.push(slot, value)?;
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
        let mut out = Vec::with_capacity(cache.len());
        for row in 0..cache.len() {
            out.push(self.evaluate_cache_row_unchecked(params, cache, row)?);
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
        let values = self.evaluate_values_from_cache(params, cache, row)?;
        scalar_at(&values, self.graph.root().index())
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
        let values = self.evaluate_values_from_cache(params, cache, row)?;
        self.value_gradient(params, values, Some((cache, row)))
    }

    pub fn evaluate_cache_with_gradient(
        &self,
        params: &ParamValues,
        cache: &CpuBatchCache,
    ) -> RuntimeResult<Vec<ValueGradient>> {
        self.check_batch_cache(cache)?;
        (0..cache.len())
            .map(|row| self.evaluate_cache_row_with_gradient_unchecked(params, cache, row))
            .collect()
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
        let mut batches = Vec::new();
        let mut sum_weights = 0.0;
        for batch in dataset
            .batches()
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

    pub fn try_weighted_sum_cached<E, F>(
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
        for batch in dataset.batches() {
            self.check_batch_cache(batch.cache())?;
            for row in 0..batch.len() {
                let value = self.evaluate_cache_row_unchecked(params, batch.cache(), row)?;
                sum += batch.weights()[row] * f(value)?;
            }
        }
        Ok(sum)
    }

    pub fn weighted_sum_cached<F>(
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

    pub fn try_weighted_real_sum_with_gradient_cached<E, F>(
        &self,
        params: &ParamValues,
        dataset: &CpuCachedDataset,
        mut transform: F,
    ) -> Result<(f64, Vec<f64>), E>
    where
        E: From<RuntimeError>,
        F: FnMut(Complex64) -> Result<(f64, f64), E>,
    {
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

    pub fn try_weighted_complex_sum_cached<E, F>(
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
        for batch in dataset.batches() {
            self.check_batch_cache(batch.cache())?;
            for row in 0..batch.len() {
                let value = self.evaluate_cache_row_unchecked(params, batch.cache(), row)?;
                sum += f(value)? * batch.weights()[row];
            }
        }
        Ok(sum)
    }

    pub fn weighted_complex_sum_cached<F>(
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

    pub fn par_try_weighted_sum_cached<E, F>(
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
        for batch in dataset.batches() {
            self.check_batch_cache(batch.cache())?;
            let partial = (0..batch.len())
                .into_par_iter()
                .try_fold(AccurateF64::zero, |mut acc, row| {
                    let value = self.evaluate_cache_row_unchecked(params, batch.cache(), row)?;
                    acc.push(batch.weights()[row] * f(value)?);
                    Ok::<AccurateF64, E>(acc)
                })
                .try_reduce(AccurateF64::zero, |mut a, b| {
                    a.merge(b);
                    Ok::<AccurateF64, E>(a)
                })?;
            total.merge(partial);
        }
        Ok(total.finish())
    }

    pub fn par_weighted_sum_cached<F>(
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

    pub fn par_try_weighted_real_sum_with_gradient_cached<E, F>(
        &self,
        params: &ParamValues,
        dataset: &CpuCachedDataset,
        transform: F,
    ) -> Result<(f64, Vec<f64>), E>
    where
        E: From<RuntimeError> + Send,
        F: Fn(Complex64) -> Result<(f64, f64), E> + Send + Sync,
    {
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

    pub fn par_try_weighted_complex_sum_cached<E, F>(
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
        for batch in dataset.batches() {
            self.check_batch_cache(batch.cache())?;
            let partial = (0..batch.len())
                .into_par_iter()
                .try_fold(AccurateComplex64::zero, |mut acc, row| {
                    let value = self.evaluate_cache_row_unchecked(params, batch.cache(), row)?;
                    acc.push(f(value)? * batch.weights()[row]);
                    Ok::<AccurateComplex64, E>(acc)
                })
                .try_reduce(AccurateComplex64::zero, |mut a, b| {
                    a.merge(b);
                    Ok::<AccurateComplex64, E>(a)
                })?;
            total.merge(partial);
        }
        Ok(total.finish())
    }

    pub fn par_weighted_complex_sum_cached<F>(
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
        let values = self.evaluate_values(params, event)?;
        scalar_at(&values, self.graph.root().index())
    }

    fn value_gradient(
        &self,
        params: &ParamValues,
        values: Vec<Value>,
        cached_factors: Option<(&CpuBatchCache, usize)>,
    ) -> RuntimeResult<ValueGradient> {
        let value = scalar_at(&values, self.graph.root().index())?;
        let gradient =
            DerivativeWorkspace::new(self, params, &values, cached_factors).gradient()?;
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
        let mut values = vec![Value::Scalar(Complex64::ZERO); self.graph.nodes().len()];

        for (index, node) in self.graph.nodes().iter().enumerate() {
            if !self.cached_evaluation_nodes[index] {
                continue;
            }
            if let Some(slot) = self.cache_slots[index] {
                values[index] = cache.value(slot, row)?;
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
                ExprNode::EventP4Component { name, component } => {
                    return Err(RuntimeError::MissingEventScalar(format!(
                        "{name}.{}",
                        component.label()
                    )));
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
            values[index] = value;
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
            && cache.factor_nodes
                == self
                    .factor_matrices
                    .iter()
                    .map(|(node, _)| *node)
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

type DynamicLu = LU<Complex64, Dyn, Dyn>;

struct DerivativeWorkspace<'a> {
    plan: &'a CpuPlan,
    params: &'a ParamValues,
    primals: &'a [Value],
    tangents: Vec<Option<Value>>,
    factors: HashMap<usize, DynamicLu>,
    cached_factors: Option<(&'a CpuBatchCache, usize)>,
}

impl<'a> DerivativeWorkspace<'a> {
    fn new(
        plan: &'a CpuPlan,
        params: &'a ParamValues,
        primals: &'a [Value],
        cached_factors: Option<(&'a CpuBatchCache, usize)>,
    ) -> Self {
        Self {
            plan,
            params,
            primals,
            tangents: vec![None; primals.len()],
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
                self.differentiate_node(*id, parameter)?;
            }
            gradient.push(self.scalar_tangent(self.plan.graph.root())?);
            for id in active {
                self.tangents[id.index()] = None;
            }
        }
        Ok(gradient)
    }

    fn differentiate_node(&mut self, id: ExprId, parameter: usize) -> RuntimeResult<()> {
        let index = id.index();
        let node = self.plan.graph.nodes()[index].clone();
        let tangent = match node {
            ExprNode::ScalarParam(_)
            | ExprNode::ComplexScalarParam { .. }
            | ExprNode::PolarComplexScalarParam { .. } => {
                Value::Scalar(self.parameter_seed(id, parameter, &node)?)
            }
            ExprNode::Unary { op, input } => {
                let input_value = scalar_at(self.primals, input.index())?;
                let output_value = scalar_at(self.primals, index)?;
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
                let lhs_value = scalar_at(self.primals, lhs.index())?;
                let rhs_value = scalar_at(self.primals, rhs.index())?;
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
                    let value = scalar_at(self.primals, factor.index())?;
                    derivative = derivative * value + product * self.scalar_tangent(factor)?;
                    product *= value;
                }
                Value::Scalar(derivative)
            }
            ExprNode::Complex { re, im } => Value::Scalar(Complex64::new(
                self.scalar_tangent(re)?.re,
                self.scalar_tangent(im)?.re,
            )),
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
                let vector = self.vector_tangent(input)?;
                Value::Scalar(*vector.get(i).ok_or_else(|| RuntimeError::InvalidShape {
                    index,
                    message: format!("component index {i} out of bounds for len {}", vector.len()),
                })?)
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
                let (lhs_rows, lhs_cols, lhs_value) = matrix_at(self.primals, lhs.index())?;
                let (rhs_rows, rhs_cols, rhs_value) = matrix_at(self.primals, rhs.index())?;
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
                let (rows, cols, matrix_value) = matrix_at(self.primals, matrix.index())?;
                let vector_value = vector_at(self.primals, vector.index())?;
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
                let lhs_value = vector_at(self.primals, lhs.index())?;
                let rhs_value = vector_at(self.primals, rhs.index())?;
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
                let (rows, cols, matrix_value) = matrix_at(self.primals, matrix.index())?;
                let solution = vector_at(self.primals, index)?;
                let rhs_value = vector_at(self.primals, rhs.index())?;
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
                    self.factors
                        .entry(matrix.index())
                        .or_insert_with(|| DMatrix::from_row_slice(rows, cols, matrix_value).lu())
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

    fn parameter_seed(
        &self,
        id: ExprId,
        parameter: usize,
        node: &ExprNode,
    ) -> RuntimeResult<Complex64> {
        let seed = self
            .plan
            .autodiff
            .seed_kind(id, parameter)
            .expect("active parameter node has a seed");
        Ok(match seed {
            SeedKind::Real | SeedKind::ComplexReal => Complex64::ONE,
            SeedKind::ComplexImag => Complex64::I,
            SeedKind::PolarMagnitude => {
                let ExprNode::PolarComplexScalarParam { phase, .. } = node else {
                    unreachable!("polar magnitude seed belongs to a polar parameter")
                };
                Complex64::cis(param_value(self.params, &self.plan.params, phase.name())?)
            }
            SeedKind::PolarPhase => Complex64::I * scalar_at(self.primals, id.index())?,
        })
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
    slots: Vec<CachedSlot>,
    factor_nodes: Vec<ExprId>,
    factor_slots: Vec<CachedFactorSlot>,
}

impl CpuBatchCache {
    fn new(cache_plan: &CachePlan, factor_matrices: &[(ExprId, usize)], len: usize) -> Self {
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
                .map(|entry| CachedSlot::new(entry.value_kind()))
                .collect(),
            factor_nodes: factor_matrices.iter().map(|(node, _)| *node).collect(),
            factor_slots: factor_matrices
                .iter()
                .map(|(_, dimension)| CachedFactorSlot::new(*dimension))
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

    fn resident_bytes(&self) -> usize {
        match self {
            Self::Scalar(values) => values.capacity() * size_of::<Complex64>(),
            Self::Vector { values, .. } | Self::Matrix { values, .. } => {
                values.capacity() * size_of::<Complex64>()
            }
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

fn cached_evaluation_nodes(graph: &ExprGraph, cache_slots: &[Option<usize>]) -> Vec<bool> {
    let mut required = vec![false; graph.nodes().len()];
    mark_cached_evaluation_node(graph, graph.root(), cache_slots, &mut required);
    required
}

fn mark_cached_evaluation_node(
    graph: &ExprGraph,
    id: ExprId,
    cache_slots: &[Option<usize>],
    required: &mut [bool],
) {
    if required[id.index()] {
        return;
    }
    required[id.index()] = true;
    if cache_slots[id.index()].is_some() {
        return;
    }
    if let Some(node) = graph.node(id) {
        for child in node_children(node) {
            mark_cached_evaluation_node(graph, child, cache_slots, required);
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
        | ExprNode::EventScalar(_)
        | ExprNode::EventP4Component { .. } => Vec::new(),
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
        let value = params.get(id).unwrap();
        plus.set_full(id, value + h).unwrap();
        minus.set_full(id, value - h).unwrap();
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
    fn event_only_solve_matrices_are_factorized_in_the_batch_cache() {
        let expression = solve(
            matrix([[event_scalar("x") + 2.0]]),
            vector([parameter!("rhs", initial: 3.0)]),
        )
        .component(0);
        let model = CompiledModel::from_expr(&expression).unwrap();
        let params = Arc::new(model.params().clone()).default_values();
        let plan = CpuBackend.prepare(&model);
        let batch = EventBatch::from_events(
            Arc::new(Schema::new(std::iter::empty::<&str>(), ["x"], false).unwrap()),
            [
                OwnedEvent::new(vec![], vec![0.0]),
                OwnedEvent::new(vec![], vec![1.0]),
            ],
        )
        .unwrap();
        let cache = plan.cache_event_batch(&batch).unwrap();

        assert_eq!(cache.factor_slots.len(), 1);
        assert_eq!(cache.factor_slots[0].factors.len(), 2);
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
    fn evaluates_linear_algebra_nodes() {
        let a = matrix([[2.0, 0.0], [0.0, 4.0]]);
        let b = vector([8.0, 12.0]);
        let x = solve(a, b);
        let expr = dot(&x, vector([1.0, 1.0]));
        let model = CompiledModel::from_expr(&expr).unwrap();
        let params = Arc::new(model.params().clone()).default_values();
        let plan = CpuBackend.prepare(&model);

        assert_eq!(plan.evaluate(&params).unwrap(), Complex64::from(7.0));
        assert_eq!(plan.constant_factors.len(), 1);
        assert!(plan.constant_factors[0].get().is_some());
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
