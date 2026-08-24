use std::sync::Arc;

use laddu_compile::CompiledQuery;
use laddu_data::{
    LadduDataError, LadduDataResult,
    data::{Dataset, EventBatch},
    io::{EventBatchIter, EventSource, ReadPlan, SourceCapabilities, memory::MemorySource},
    schema::Schema,
};
use laddu_expr::{Expr, ValueKind};
use num::complex::Complex64;
use serde::{Deserialize, Deserializer, Serialize};

use crate::{Execution, PreparedModel, RuntimeError, RuntimeResult};

/// Comparison operation used by a dataset predicate.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Comparison {
    /// Less than.
    Lt,
    /// Less than or equal to.
    Le,
    /// Greater than.
    Gt,
    /// Greater than or equal to.
    Ge,
    /// Equal to.
    Eq,
    /// Not equal to.
    Ne,
}

/// Determines which endpoints are included by an interval predicate.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntervalClosure {
    /// Exclude both endpoints.
    Open,
    /// Include only the lower endpoint.
    LeftClosed,
    /// Include only the upper endpoint.
    RightClosed,
    /// Include both endpoints.
    #[default]
    Closed,
}

/// Boolean expression used to select events from a dataset.
#[derive(Clone, Debug)]
pub enum Predicate {
    /// Compare two scalar expressions.
    Compare {
        /// Left-hand expression.
        lhs: Expr,
        /// Comparison operation.
        op: Comparison,
        /// Right-hand expression.
        rhs: Expr,
    },
    /// Require both child predicates to hold.
    And(Box<Self>, Box<Self>),
    /// Require either child predicate to hold.
    Or(Box<Self>, Box<Self>),
    /// Negate a predicate.
    Not(Box<Self>),
    /// Test whether a value lies between two bounds.
    Between {
        /// Expression whose value is tested.
        value: Expr,
        /// Lower bound expression.
        lower: Expr,
        /// Upper bound expression.
        upper: Expr,
        /// Endpoint inclusion policy.
        closure: IntervalClosure,
    },
}

impl Predicate {
    /// Creates a comparison predicate.
    pub fn compare(lhs: impl Into<Expr>, op: Comparison, rhs: impl Into<Expr>) -> Self {
        Self::Compare {
            lhs: lhs.into(),
            op,
            rhs: rhs.into(),
        }
    }

    /// Creates a less-than predicate.
    pub fn lt(lhs: impl Into<Expr>, rhs: impl Into<Expr>) -> Self {
        Self::compare(lhs, Comparison::Lt, rhs)
    }
    /// Creates a less-than-or-equal predicate.
    pub fn le(lhs: impl Into<Expr>, rhs: impl Into<Expr>) -> Self {
        Self::compare(lhs, Comparison::Le, rhs)
    }
    /// Creates a greater-than predicate.
    pub fn gt(lhs: impl Into<Expr>, rhs: impl Into<Expr>) -> Self {
        Self::compare(lhs, Comparison::Gt, rhs)
    }
    /// Creates a greater-than-or-equal predicate.
    pub fn ge(lhs: impl Into<Expr>, rhs: impl Into<Expr>) -> Self {
        Self::compare(lhs, Comparison::Ge, rhs)
    }
    /// Creates an equality predicate.
    pub fn eq(lhs: impl Into<Expr>, rhs: impl Into<Expr>) -> Self {
        Self::compare(lhs, Comparison::Eq, rhs)
    }
    /// Creates an inequality predicate.
    pub fn ne(lhs: impl Into<Expr>, rhs: impl Into<Expr>) -> Self {
        Self::compare(lhs, Comparison::Ne, rhs)
    }
    /// Combines this predicate with `rhs` using logical AND.
    pub fn and(self, rhs: Self) -> Self {
        Self::And(Box::new(self), Box::new(rhs))
    }
    /// Combines this predicate with `rhs` using logical OR.
    pub fn or(self, rhs: Self) -> Self {
        Self::Or(Box::new(self), Box::new(rhs))
    }
    /// Creates a closed-interval predicate.
    pub fn between(value: impl Into<Expr>, lower: impl Into<Expr>, upper: impl Into<Expr>) -> Self {
        Self::between_with(value, lower, upper, IntervalClosure::Closed)
    }
    /// Creates an interval predicate with an explicit endpoint policy.
    pub fn between_with(
        value: impl Into<Expr>,
        lower: impl Into<Expr>,
        upper: impl Into<Expr>,
        closure: IntervalClosure,
    ) -> Self {
        Self::Between {
            value: value.into(),
            lower: lower.into(),
            upper: upper.into(),
            closure,
        }
    }
}

impl std::ops::Not for Predicate {
    type Output = Self;
    fn not(self) -> Self::Output {
        Self::Not(Box::new(self))
    }
}

/// Validated, monotonically increasing bin edges.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(transparent)]
pub struct BinSpec {
    edges: Arc<[f64]>,
}

impl<'de> Deserialize<'de> for BinSpec {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let edges = Vec::<f64>::deserialize(deserializer)?;
        Self::edges(edges).map_err(serde::de::Error::custom)
    }
}

impl BinSpec {
    /// Creates `count` uniformly spaced bins spanning `[min, max]`.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when `count` is zero or the bounds are
    /// non-finite or not increasing.
    pub fn uniform(count: usize, min: f64, max: f64) -> RuntimeResult<Self> {
        if count == 0 || !min.is_finite() || !max.is_finite() || min >= max {
            return Err(query_error(
                "uniform bins require a positive count and finite min < max",
            ));
        }
        let width = (max - min) / count as f64;
        Self::edges((0..=count).map(|i| min + i as f64 * width))
    }

    /// Creates bins from explicit, strictly increasing finite edges.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when fewer than two edges are supplied or an
    /// edge is non-finite or not strictly increasing.
    pub fn edges(edges: impl IntoIterator<Item = f64>) -> RuntimeResult<Self> {
        let edges: Vec<_> = edges.into_iter().collect();
        if edges.len() < 2
            || edges.iter().any(|x| !x.is_finite())
            || edges.windows(2).any(|w| w[0] >= w[1])
        {
            return Err(query_error(
                "bin edges must contain at least two finite, strictly increasing values",
            ));
        }
        Ok(Self {
            edges: edges.into(),
        })
    }

    /// Returns the number of bins.
    pub fn bin_count(&self) -> usize {
        self.edges.len() - 1
    }
    /// Returns the validated bin edges.
    pub fn edges_slice(&self) -> &[f64] {
        &self.edges
    }

    fn index(&self, value: f64) -> Option<usize> {
        if !value.is_finite() || value < self.edges[0] || value > *self.edges.last()? {
            return None;
        }
        if value == *self.edges.last()? {
            return Some(self.bin_count() - 1);
        }
        let upper = self.edges.partition_point(|edge| *edge <= value);
        upper
            .checked_sub(1)
            .filter(|index| *index < self.bin_count())
    }
}

/// A lazily filtered dataset corresponding to one bin.
#[derive(Clone)]
pub struct DatasetBin {
    index: usize,
    lower: f64,
    upper: f64,
    dataset: Dataset,
}

impl DatasetBin {
    /// Returns the zero-based bin index.
    pub fn index(&self) -> usize {
        self.index
    }
    /// Returns the bin's lower edge.
    pub fn lower(&self) -> f64 {
        self.lower
    }
    /// Returns the bin's upper edge.
    pub fn upper(&self) -> f64 {
        self.upper
    }
    /// Returns the dataset containing events in this bin.
    pub fn dataset(&self) -> &Dataset {
        &self.dataset
    }
    /// Consumes the bin and returns its dataset.
    pub fn into_dataset(self) -> Dataset {
        self.dataset
    }
}

/// Expression-based query operations for datasets.
pub trait DatasetExprExt {
    /// Evaluates a scalar expression for every event.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when compilation, dataset reading, or
    /// evaluation fails, or the expression is not scalar.
    fn evaluate_expr(&self, expr: &Expr, execution: &Execution) -> RuntimeResult<Vec<Complex64>>;
    /// Evaluates a real scalar expression for every event.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when compilation, dataset reading, or
    /// evaluation fails, or the expression is not real scalar-valued.
    fn evaluate_real(&self, expr: &Expr, execution: &Execution) -> RuntimeResult<Vec<f64>>;
    /// Creates a lazily filtered dataset containing events that satisfy `predicate`.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when predicate compilation or evaluation
    /// fails, or its expression is not real scalar-valued.
    fn select(&self, predicate: &Predicate, execution: &Execution) -> RuntimeResult<Dataset>;
    /// Partitions the dataset in one pass according to an expression and bin specification.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when expression compilation or evaluation
    /// fails, or the expression is not real scalar-valued.
    fn bin_by(
        &self,
        expr: &Expr,
        bins: BinSpec,
        execution: &Execution,
    ) -> RuntimeResult<Vec<DatasetBin>>;
}

impl DatasetExprExt for Dataset {
    fn evaluate_expr(&self, expr: &Expr, execution: &Execution) -> RuntimeResult<Vec<Complex64>> {
        let query = QueryExprSet::prepare(vec![expr.clone()], execution, false)?;
        let mut output = Vec::new();
        for batch in self.batches().map_err(data_error)? {
            output.extend(
                query.evaluate_batch(&batch.map_err(data_error)?)?[0]
                    .iter()
                    .copied(),
            );
        }
        Ok(output)
    }

    fn evaluate_real(&self, expr: &Expr, execution: &Execution) -> RuntimeResult<Vec<f64>> {
        let query = QueryExprSet::prepare(vec![expr.clone()], execution, true)?;
        let mut output = Vec::new();
        for batch in self.batches().map_err(data_error)? {
            output.extend(
                query.evaluate_batch(&batch.map_err(data_error)?)?[0]
                    .iter()
                    .copied()
                    .map(|v| v.re),
            );
        }
        Ok(output)
    }

    fn select(&self, predicate: &Predicate, execution: &Execution) -> RuntimeResult<Dataset> {
        let compiled = CompiledPredicate::prepare(predicate, execution)?;
        Ok(self.with_derived_source(QuerySource {
            source: self.clone(),
            filter: QueryFilter::Predicate(Arc::new(compiled)),
        }))
    }

    fn bin_by(
        &self,
        expr: &Expr,
        bins: BinSpec,
        execution: &Execution,
    ) -> RuntimeResult<Vec<DatasetBin>> {
        let query = QueryExprSet::prepare(vec![expr.clone()], execution, true)?;
        let schema = self.schema().map_err(data_error)?;
        let mut partitions = vec![Vec::new(); bins.bin_count()];
        for batch in self.batches().map_err(data_error)? {
            let batch = batch.map_err(data_error)?;
            let mut rows = vec![Vec::new(); bins.bin_count()];
            for (row, value) in query.evaluate_batch(&batch)?[0].iter().copied().enumerate() {
                if let Some(index) = bins.index(value.re) {
                    rows[index].push(row);
                }
            }
            for (partition, rows) in partitions.iter_mut().zip(rows) {
                if !rows.is_empty() {
                    partition.push(batch.select(&rows));
                }
            }
        }

        partitions
            .into_iter()
            .enumerate()
            .map(|(index, batches)| {
                let source = if batches.is_empty() {
                    MemorySource::empty(Arc::clone(&schema))
                } else {
                    MemorySource::from_batches(batches).map_err(data_error)?
                };
                Ok(DatasetBin {
                    index,
                    lower: bins.edges[index],
                    upper: bins.edges[index + 1],
                    dataset: self.with_derived_source(source),
                })
            })
            .collect()
    }
}

struct QueryExpr {
    model: PreparedModel,
    params: laddu_expr::parameters::ParamValues,
    outputs: Vec<laddu_expr::ExprId>,
}

struct QueryExprSet {
    shared: QueryExprStorage,
    outputs: usize,
}

enum QueryExprStorage {
    Shared(QueryExpr),
    Separate(Vec<QueryExpr>),
}

impl QueryExprSet {
    fn prepare(
        expressions: Vec<Expr>,
        execution: &Execution,
        require_real: bool,
    ) -> RuntimeResult<Self> {
        let expression_count = expressions.len();
        let compiled = CompiledQuery::from_exprs(expressions.clone())
            .map_err(|error| query_error(error.to_string()))?;
        let model = compiled.model();
        let outputs = compiled.outputs();
        if outputs.len() != expression_count {
            return Err(query_error(
                "compiled query output count changed during lowering",
            ));
        }
        for element in outputs {
            let value_kind = model
                .node_facts(*element)
                .map(|facts| facts.value_kind)
                .ok_or_else(|| query_error("compiled expression facts are incomplete"))?;
            if value_kind != ValueKind::Real
                && (require_real || !matches!(value_kind, ValueKind::Complex))
            {
                return Err(query_error(if require_real {
                    "this dataset operation requires a real-valued expression"
                } else {
                    "dataset expressions must be scalar"
                }));
            }
        }
        if model.params().n_free() != 0 {
            return Err(query_error(
                "dataset expressions cannot contain free parameters",
            ));
        }
        let params = model.params().default_values();
        let plan = match PreparedModel::prepare(model, execution) {
            Ok(plan) => QueryExprStorage::Shared(QueryExpr {
                model: plan,
                params,
                outputs: outputs.to_vec(),
            }),
            Err(shared_error) => {
                if !may_fallback_to_scalar(execution, &shared_error) {
                    return Err(shared_error);
                }
                let separate = expressions
                    .iter()
                    .map(|expr| QueryExpr::prepare(expr, execution, require_real))
                    .collect::<RuntimeResult<Vec<_>>>();
                QueryExprStorage::Separate(separate?)
            }
        };
        Ok(Self {
            shared: plan,
            outputs: expression_count,
        })
    }

    fn evaluate_batch(&self, batch: &EventBatch) -> RuntimeResult<Vec<Vec<Complex64>>> {
        let values = match &self.shared {
            QueryExprStorage::Shared(query) => query.evaluate_outputs(batch)?,
            QueryExprStorage::Separate(queries) => queries
                .iter()
                .map(|query| query.evaluate_batch(batch))
                .collect::<RuntimeResult<Vec<_>>>()?,
        };
        if values.len() != self.outputs {
            return Err(query_error(
                "compiled query returned an unexpected output count",
            ));
        }
        Ok(values)
    }
}

impl QueryExpr {
    fn prepare(expr: &Expr, execution: &Execution, require_real: bool) -> RuntimeResult<Self> {
        if expr.shape().map_err(|e| query_error(e.to_string()))? != laddu_expr::ExprShape::Scalar {
            return Err(query_error("dataset expressions must be scalar"));
        }
        let compiled = laddu_compile::CompiledModel::from_expr(expr)
            .map_err(|error| query_error(error.to_string()))?;
        let value_kind = compiled
            .node_facts(compiled.graph().root())
            .map(|facts| facts.value_kind)
            .ok_or_else(|| query_error("compiled expression facts are incomplete"))?;
        if require_real && value_kind == ValueKind::Complex {
            return Err(query_error(
                "this dataset operation requires a real-valued expression",
            ));
        }
        if compiled.params().n_free() != 0 {
            return Err(query_error(
                "dataset expressions cannot contain free parameters",
            ));
        }
        let params = compiled.params().default_values();
        let model = PreparedModel::prepare(&compiled, execution)?;
        Ok(Self {
            model,
            params,
            outputs: Vec::new(),
        })
    }

    fn evaluate_batch(&self, batch: &EventBatch) -> RuntimeResult<Vec<Complex64>> {
        self.model.evaluate_batch(&self.params, batch)
    }

    fn evaluate_outputs(&self, batch: &EventBatch) -> RuntimeResult<Vec<Vec<Complex64>>> {
        self.model
            .evaluate_batch_outputs(&self.params, batch, &self.outputs)
    }
}

struct CompiledPredicate {
    expressions: QueryExprSet,
    program: PredicateProgram,
}

enum PredicateProgram {
    Compare {
        lhs: usize,
        op: Comparison,
        rhs: usize,
    },
    And(Box<Self>, Box<Self>),
    Or(Box<Self>, Box<Self>),
    Not(Box<Self>),
    Between {
        value: usize,
        lower: usize,
        upper: usize,
        closure: IntervalClosure,
    },
}

impl CompiledPredicate {
    fn prepare(predicate: &Predicate, execution: &Execution) -> RuntimeResult<Self> {
        let mut expressions = Vec::new();
        let program = Self::compile_program(predicate, &mut expressions);
        Ok(Self {
            expressions: QueryExprSet::prepare(expressions, execution, true)?,
            program,
        })
    }

    fn compile_program(predicate: &Predicate, expressions: &mut Vec<Expr>) -> PredicateProgram {
        let leaf = |expr: &Expr, expressions: &mut Vec<Expr>| {
            let index = expressions.len();
            expressions.push(expr.clone());
            index
        };
        match predicate {
            Predicate::Compare { lhs, op, rhs } => PredicateProgram::Compare {
                lhs: leaf(lhs, expressions),
                op: *op,
                rhs: leaf(rhs, expressions),
            },
            Predicate::And(lhs, rhs) => PredicateProgram::And(
                Box::new(Self::compile_program(lhs, expressions)),
                Box::new(Self::compile_program(rhs, expressions)),
            ),
            Predicate::Or(lhs, rhs) => PredicateProgram::Or(
                Box::new(Self::compile_program(lhs, expressions)),
                Box::new(Self::compile_program(rhs, expressions)),
            ),
            Predicate::Not(inner) => {
                PredicateProgram::Not(Box::new(Self::compile_program(inner, expressions)))
            }
            Predicate::Between {
                value,
                lower,
                upper,
                closure,
            } => PredicateProgram::Between {
                value: leaf(value, expressions),
                lower: leaf(lower, expressions),
                upper: leaf(upper, expressions),
                closure: *closure,
            },
        }
    }

    fn evaluate_batch(&self, batch: &EventBatch) -> RuntimeResult<Vec<usize>> {
        let values = self.expressions.evaluate_batch(batch)?;
        Ok((0..batch.len())
            .filter(|row| Self::evaluate_row(&self.program, &values, *row))
            .collect())
    }

    fn evaluate_row(program: &PredicateProgram, values: &[Vec<Complex64>], row: usize) -> bool {
        match program {
            PredicateProgram::Compare { lhs, op, rhs } => {
                compare(values[*lhs][row].re, *op, values[*rhs][row].re)
            }
            PredicateProgram::And(lhs, rhs) => {
                Self::evaluate_row(lhs, values, row) && Self::evaluate_row(rhs, values, row)
            }
            PredicateProgram::Or(lhs, rhs) => {
                Self::evaluate_row(lhs, values, row) || Self::evaluate_row(rhs, values, row)
            }
            PredicateProgram::Not(inner) => !Self::evaluate_row(inner, values, row),
            PredicateProgram::Between {
                value,
                lower,
                upper,
                closure,
            } => {
                let lower_op = match closure {
                    IntervalClosure::Open | IntervalClosure::RightClosed => Comparison::Gt,
                    IntervalClosure::LeftClosed | IntervalClosure::Closed => Comparison::Ge,
                };
                let upper_op = match closure {
                    IntervalClosure::Open | IntervalClosure::LeftClosed => Comparison::Lt,
                    IntervalClosure::RightClosed | IntervalClosure::Closed => Comparison::Le,
                };
                compare(values[*value][row].re, lower_op, values[*lower][row].re)
                    && compare(values[*value][row].re, upper_op, values[*upper][row].re)
            }
        }
    }
}

fn compare(lhs: f64, op: Comparison, rhs: f64) -> bool {
    if lhs.is_nan() || rhs.is_nan() {
        return false;
    }
    match op {
        Comparison::Lt => lhs < rhs,
        Comparison::Le => lhs <= rhs,
        Comparison::Gt => lhs > rhs,
        Comparison::Ge => lhs >= rhs,
        Comparison::Eq => lhs == rhs,
        Comparison::Ne => lhs != rhs,
    }
}

#[derive(Clone)]
struct QuerySource {
    source: Dataset,
    filter: QueryFilter,
}

#[derive(Clone)]
enum QueryFilter {
    Predicate(Arc<CompiledPredicate>),
}

impl EventSource for QuerySource {
    fn schema(&self) -> LadduDataResult<Arc<Schema>> {
        self.source.schema()
    }

    fn capabilities(&self) -> SourceCapabilities {
        let source = self.source.capabilities();
        SourceCapabilities {
            exact_len: false,
            exact_weighted_total: false,
            random_access: false,
            deterministic_partitioning: source.deterministic_partitioning,
            predicate_pushdown: false,
            projection_pushdown: false,
            streaming: true,
        }
    }

    fn batches(&self, plan: ReadPlan) -> LadduDataResult<EventBatchIter> {
        let batches = self.source.stream_with_plan(plan)?;
        let filter = self.filter.clone();
        Ok(Box::new(batches.filter_map(move |batch| {
            let batch = match batch {
                Ok(batch) => batch,
                Err(error) => return Some(Err(error)),
            };
            let rows = match filter.rows(&batch) {
                Ok(rows) => rows,
                Err(error) => return Some(Err(LadduDataError::Source(error.to_string()))),
            };
            (!rows.is_empty()).then(|| Ok(batch.select(&rows)))
        })))
    }
}

impl QueryFilter {
    fn rows(&self, batch: &EventBatch) -> RuntimeResult<Vec<usize>> {
        match self {
            Self::Predicate(predicate) => predicate.evaluate_batch(batch),
        }
    }
}

fn query_error(message: impl Into<String>) -> RuntimeError {
    RuntimeError::InvalidShape {
        index: 0,
        message: message.into(),
    }
}

fn may_fallback_to_scalar(execution: &Execution, error: &RuntimeError) -> bool {
    let cpu_f32 = matches!(
        error,
        RuntimeError::Execution(crate::ExecutionError::UnsupportedCpuF32Model)
    );
    #[cfg(feature = "wgpu")]
    {
        cpu_f32 || (execution.wgpu_context().is_some() && matches!(error, RuntimeError::Wgpu(_)))
    }
    #[cfg(not(feature = "wgpu"))]
    {
        let _ = execution;
        cpu_f32
    }
}

fn data_error(error: impl ToString) -> RuntimeError {
    RuntimeError::Data(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuOptions, Device, ExecutionOptions, Precision};
    use laddu_compile::CompiledModel;
    use laddu_data::{
        data::{EventBatch, OwnedEvent},
        io::{EventSource, ReadPlan, SourceCapabilities, memory::MemorySource},
        schema::Schema,
    };
    use laddu_expr::{complex, event_scalar};
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn bin_spec_roundtrip_preserves_validation() {
        let bins = BinSpec::edges([-1.0, 0.0, 2.0]).unwrap();
        let json = serde_json::to_string(&bins).unwrap();
        assert_eq!(serde_json::from_str::<BinSpec>(&json).unwrap(), bins);
        assert!(serde_json::from_str::<BinSpec>("[0.0,0.0]").is_err());
    }

    #[derive(Clone)]
    struct CountingSource {
        inner: MemorySource,
        reads: Arc<AtomicUsize>,
    }

    impl EventSource for CountingSource {
        fn schema(&self) -> LadduDataResult<Arc<Schema>> {
            EventSource::schema(&self.inner)
        }

        fn capabilities(&self) -> SourceCapabilities {
            self.inner.capabilities()
        }

        fn batches(&self, plan: ReadPlan) -> LadduDataResult<EventBatchIter> {
            self.reads.fetch_add(1, Ordering::Relaxed);
            self.inner.batches(plan)
        }
    }

    #[derive(Clone)]
    struct FailingSource {
        schema: Arc<Schema>,
    }

    impl EventSource for FailingSource {
        fn schema(&self) -> LadduDataResult<Arc<Schema>> {
            Ok(Arc::clone(&self.schema))
        }

        fn capabilities(&self) -> SourceCapabilities {
            SourceCapabilities {
                exact_len: false,
                exact_weighted_total: false,
                random_access: false,
                deterministic_partitioning: true,
                predicate_pushdown: false,
                projection_pushdown: false,
                streaming: true,
            }
        }

        fn batches(&self, _plan: ReadPlan) -> LadduDataResult<EventBatchIter> {
            Ok(Box::new(std::iter::once(Err(LadduDataError::Source(
                "query source failed".into(),
            )))))
        }
    }

    fn capability_tuple(
        capabilities: SourceCapabilities,
    ) -> (bool, bool, bool, bool, bool, bool, bool) {
        (
            capabilities.exact_len,
            capabilities.exact_weighted_total,
            capabilities.random_access,
            capabilities.deterministic_partitioning,
            capabilities.predicate_pushdown,
            capabilities.projection_pushdown,
            capabilities.streaming,
        )
    }

    fn dataset() -> Dataset {
        let schema = Arc::new(Schema::new(std::iter::empty::<&str>(), ["x"], true).unwrap());
        Dataset::from_events(
            schema,
            [
                OwnedEvent::weighted(vec![], vec![-1.0], 0.5),
                OwnedEvent::weighted(vec![], vec![0.0], 1.0),
                OwnedEvent::weighted(vec![], vec![1.0], 1.5),
                OwnedEvent::weighted(vec![], vec![2.0], 2.0),
            ],
        )
        .unwrap()
    }

    #[test]
    fn evaluates_selects_and_bins_dataset_expressions() {
        let dataset = dataset().chunked(1).unwrap();
        let execution = Execution::default();
        let x = event_scalar("x");
        assert_eq!(
            dataset.evaluate_real(&x, &execution).unwrap(),
            vec![-1.0, 0.0, 1.0, 2.0]
        );

        let selected = dataset
            .select(
                &Predicate::ge(x.clone(), 0.0).and(Predicate::lt(x.clone(), 2.0)),
                &execution,
            )
            .unwrap();
        assert_eq!(
            selected.map_events(|event| event.scalar(0)).unwrap(),
            vec![0.0, 1.0]
        );
        assert_eq!(selected.sum_weights().unwrap(), 2.5);

        let bins = dataset
            .bin_by(&x, BinSpec::uniform(2, 0.0, 2.0).unwrap(), &execution)
            .unwrap();
        assert_eq!(bins.len(), 2);
        assert_eq!(
            bins[0]
                .dataset()
                .map_events(|event| event.scalar(0))
                .unwrap(),
            vec![0.0]
        );
        assert_eq!(
            bins[1]
                .dataset()
                .map_events(|event| event.scalar(0))
                .unwrap(),
            vec![1.0, 2.0]
        );
    }

    #[test]
    fn empty_batches_are_valid_query_inputs() {
        let execution = Execution::default();
        let x = event_scalar("x");

        let empty_batch_schema =
            Arc::new(Schema::new(std::iter::empty::<&str>(), ["x"], true).unwrap());
        let empty_batch = Dataset::from_batch(
            EventBatch::from_events(empty_batch_schema, std::iter::empty::<OwnedEvent>()).unwrap(),
        );
        for empty in [empty_batch, dataset().empty_derived().unwrap()] {
            assert!(empty.evaluate_real(&x, &execution).unwrap().is_empty());
            assert!(
                empty
                    .select(&Predicate::ge(x.clone(), 0.0), &execution)
                    .unwrap()
                    .map_events(|event| event.scalar(0))
                    .unwrap()
                    .is_empty()
            );
        }
    }

    #[test]
    fn event_column_nan_comparisons_are_false() {
        let schema = Arc::new(Schema::new(std::iter::empty::<&str>(), ["x"], true).unwrap());
        let dataset = Dataset::from_events(
            schema,
            [
                OwnedEvent::weighted(vec![], vec![f64::NAN], 1.0),
                OwnedEvent::weighted(vec![], vec![0.0], 1.0),
                OwnedEvent::weighted(vec![], vec![1.0], 1.0),
            ],
        )
        .unwrap();
        let x = event_scalar("x");
        let selected = dataset
            .select(&Predicate::ne(x, 0.0), &Execution::default())
            .unwrap();

        assert_eq!(
            selected.map_events(|event| event.scalar(0)).unwrap(),
            vec![1.0]
        );
    }

    #[test]
    fn query_propagates_source_batch_errors() {
        let schema = Arc::new(Schema::new(std::iter::empty::<&str>(), ["x"], true).unwrap());
        let dataset = Dataset::new(FailingSource { schema });

        let error = dataset
            .evaluate_real(&event_scalar("x"), &Execution::default())
            .unwrap_err();
        assert!(
            matches!(error, RuntimeError::Data(message) if message.contains("query source failed"))
        );
    }

    #[test]
    fn all_empty_bins_retain_valid_empty_derived_sources() {
        let source = dataset();
        let before = capability_tuple(source.capabilities());
        let bins = source
            .bin_by(
                &event_scalar("x"),
                BinSpec::edges([10.0, 20.0, 30.0]).unwrap(),
                &Execution::default(),
            )
            .unwrap();

        assert_eq!(capability_tuple(source.capabilities()), before);
        assert_eq!(bins.len(), 2);
        for bin in bins {
            assert_eq!(bin.dataset().num_events().unwrap(), Some(0));
            assert!(
                bin.dataset()
                    .evaluate_real(&event_scalar("x"), &Execution::default())
                    .unwrap()
                    .is_empty()
            );
        }
    }

    #[test]
    fn traversing_all_bins_reads_the_source_once() {
        let reads = Arc::new(AtomicUsize::new(0));
        let source = CountingSource {
            inner: match dataset().batches().unwrap().next().unwrap() {
                Ok(batch) => MemorySource::new(batch),
                Err(error) => panic!("unexpected source error: {error}"),
            },
            reads: Arc::clone(&reads),
        };
        let dataset = Dataset::new(source).chunked(1).unwrap();
        let bins = dataset
            .bin_by(
                &event_scalar("x"),
                BinSpec::uniform(4, -1.0, 3.0).unwrap(),
                &Execution::default(),
            )
            .unwrap();

        let values = bins
            .into_iter()
            .map(|bin| {
                bin.into_dataset()
                    .map_events(|event| event.scalar(0))
                    .unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(values, [vec![-1.0], vec![0.0], vec![1.0], vec![2.0]]);
        assert_eq!(reads.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn between_predicates_have_explicit_endpoint_semantics() {
        let dataset = dataset();
        let execution = Execution::default();
        let x = event_scalar("x");

        let closed = dataset
            .select(&Predicate::between(x.clone(), 0.0, 1.0), &execution)
            .unwrap();
        assert_eq!(
            closed.map_events(|event| event.scalar(0)).unwrap(),
            vec![0.0, 1.0]
        );

        let open = dataset
            .select(
                &Predicate::between_with(x, -1.0, 1.0, IntervalClosure::Open),
                &execution,
            )
            .unwrap();
        assert_eq!(open.map_events(|event| event.scalar(0)).unwrap(), vec![0.0]);
    }

    #[test]
    fn real_queries_reject_complex_and_free_parameter_expressions() {
        let dataset = dataset();
        let execution = Execution::default();
        assert!(
            dataset
                .evaluate_real(&complex(1.0, 1.0), &execution)
                .is_err()
        );
        let parameter = Expr::from(laddu_expr::parameters::Parameter::free("p"));
        assert!(dataset.evaluate_expr(&parameter, &execution).is_err());
    }

    #[test]
    fn compiled_query_outputs_preserve_order_and_values() {
        let source = dataset();
        let batch = source.batches().unwrap().next().unwrap().unwrap();
        let x = event_scalar("x");
        let query = QueryExprSet::prepare(
            vec![x.clone() + 1.0, x.clone() * 2.0, x],
            &Execution::default(),
            false,
        )
        .unwrap();
        let values = query.evaluate_batch(&batch).unwrap();
        assert_eq!(
            values[0].iter().map(|v| v.re).collect::<Vec<_>>(),
            [0.0, 1.0, 2.0, 3.0]
        );
        assert_eq!(
            values[1].iter().map(|v| v.re).collect::<Vec<_>>(),
            [-2.0, 0.0, 2.0, 4.0]
        );
        assert_eq!(
            values[2].iter().map(|v| v.re).collect::<Vec<_>>(),
            [-1.0, 0.0, 1.0, 2.0]
        );
    }

    #[test]
    fn repeated_predicate_leaves_are_evaluated_once() {
        let x = event_scalar("x");
        let selected = dataset()
            .select(
                &Predicate::ge(x.clone() + 1.0, 0.0).and(Predicate::lt(x + 1.0, 2.0)),
                &Execution::default(),
            )
            .unwrap();
        assert_eq!(
            selected.map_events(|event| event.scalar(0)).unwrap(),
            [-1.0, 0.0]
        );
    }

    #[test]
    fn f32_queries_match_f64_query_results() {
        let x = event_scalar("x");
        let f64_values = dataset().evaluate_real(&x, &Execution::default()).unwrap();
        let f32_execution = Execution::local(ExecutionOptions {
            device: Device::Cpu(CpuOptions::default()),
            precision: Precision::F32,
            ..ExecutionOptions::default()
        })
        .unwrap();
        let f32_values = dataset().evaluate_real(&x, &f32_execution).unwrap();
        assert_eq!(f32_values, f64_values);
    }

    #[test]
    fn bin_edges_validate_and_nan_predicates_are_false() {
        assert!(BinSpec::edges([0.0, 0.0]).is_err());
        assert!(!compare(f64::NAN, Comparison::Ne, 0.0));
    }

    #[test]
    fn selection_is_lazy_and_one_pass_binning_preserves_streaming_policy() {
        let source = dataset();
        let batch = source.batches().unwrap().next().unwrap().unwrap();
        let reads = Arc::new(AtomicUsize::new(0));
        let dataset = Dataset::new(CountingSource {
            inner: MemorySource::new(batch),
            reads: Arc::clone(&reads),
        })
        .streaming();
        let execution = Execution::default();
        let x = event_scalar("x");

        let selected = dataset
            .select(&Predicate::ge(x.clone(), 0.0), &execution)
            .unwrap();
        let bins = dataset
            .bin_by(&x, BinSpec::uniform(2, 0.0, 2.0).unwrap(), &execution)
            .unwrap();
        assert_eq!(reads.load(Ordering::Relaxed), 1);
        assert_eq!(
            selected.cache_storage(),
            laddu_data::data::CacheStorage::Streaming
        );

        assert_eq!(
            selected.map_events(|event| event.scalar(0)).unwrap(),
            vec![0.0, 1.0, 2.0]
        );
        assert_eq!(reads.load(Ordering::Relaxed), 2);
        assert_eq!(
            bins[0]
                .dataset()
                .map_events(|event| event.scalar(0))
                .unwrap(),
            vec![0.0]
        );
        assert_eq!(reads.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn unknown_cardinality_fastest_discovers_and_retains_small_selection() {
        let source = dataset();
        let batch = source.batches().unwrap().next().unwrap().unwrap();
        let reads = Arc::new(AtomicUsize::new(0));
        let dataset = Dataset::new(CountingSource {
            inner: MemorySource::new(batch),
            reads: Arc::clone(&reads),
        });
        let execution = Execution::default();
        let x = event_scalar("x");
        let selected = dataset
            .select(&Predicate::ge(x.clone(), 0.0), &execution)
            .unwrap();
        let compiled = CompiledModel::from_expr(&x).unwrap();
        let params = compiled.params().default_values();
        let model = PreparedModel::prepare(&compiled, &execution).unwrap();
        let prepared = model.prepare_dataset(&execution, &selected).unwrap();

        #[cfg(not(feature = "wgpu"))]
        let crate::PreparedDataset::Cpu(prepared_cpu) = &prepared;
        #[cfg(feature = "wgpu")]
        let crate::PreparedDataset::Cpu(prepared_cpu) = &prepared else {
            panic!("default execution prepares CPU datasets");
        };
        assert_eq!(
            prepared_cpu.stats().storage(),
            laddu_data::data::CacheStorage::Resident
        );
        assert_eq!(prepared_cpu.stats().local_events(), 3);
        assert_eq!(reads.load(Ordering::Relaxed), 2);

        for _ in 0..2 {
            assert_eq!(
                model
                    .reduce(
                        &execution,
                        &params,
                        &prepared,
                        laddu_compile::ReductionPlan::weighted_real(),
                    )
                    .unwrap(),
                5.5
            );
        }
        assert_eq!(reads.load(Ordering::Relaxed), 2);
    }
}
