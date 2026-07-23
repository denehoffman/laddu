use std::sync::Arc;

use laddu_compile::CompiledModel;
use laddu_data::{
    LadduDataError, LadduDataResult,
    data::{Dataset, EventBatch},
    io::{EventBatchIter, EventSource, ReadPlan, SourceCapabilities},
    schema::Schema,
};
use laddu_expr::{Expr, ExprShape, ValueKind};
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
    fn evaluate_expr(&self, expr: &Expr, execution: &Execution) -> RuntimeResult<Vec<Complex64>>;
    /// Evaluates a real scalar expression for every event.
    fn evaluate_real(&self, expr: &Expr, execution: &Execution) -> RuntimeResult<Vec<f64>>;
    /// Creates a lazily filtered dataset containing events that satisfy `predicate`.
    fn select(&self, predicate: &Predicate, execution: &Execution) -> RuntimeResult<Dataset>;
    /// Splits the dataset into lazy datasets according to an expression and bin specification.
    fn bin_by(
        &self,
        expr: &Expr,
        bins: BinSpec,
        execution: &Execution,
    ) -> RuntimeResult<Vec<DatasetBin>>;
}

impl DatasetExprExt for Dataset {
    fn evaluate_expr(&self, expr: &Expr, execution: &Execution) -> RuntimeResult<Vec<Complex64>> {
        let query = QueryExpr::prepare(expr, execution, false)?;
        let mut output = Vec::new();
        for batch in self.batches().map_err(data_error)? {
            output.extend(query.evaluate_batch(&batch.map_err(data_error)?)?);
        }
        Ok(output)
    }

    fn evaluate_real(&self, expr: &Expr, execution: &Execution) -> RuntimeResult<Vec<f64>> {
        let query = QueryExpr::prepare(expr, execution, true)?;
        let mut output = Vec::new();
        for batch in self.batches().map_err(data_error)? {
            output.extend(
                query
                    .evaluate_batch(&batch.map_err(data_error)?)?
                    .into_iter()
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
        let query = Arc::new(QueryExpr::prepare(expr, execution, true)?);
        Ok((0..bins.bin_count())
            .map(|index| DatasetBin {
                index,
                lower: bins.edges[index],
                upper: bins.edges[index + 1],
                dataset: self.with_derived_source(QuerySource {
                    source: self.clone(),
                    filter: QueryFilter::Bin {
                        query: Arc::clone(&query),
                        bins: bins.clone(),
                        index,
                    },
                }),
            })
            .collect())
    }
}

struct QueryExpr {
    model: PreparedModel,
    params: laddu_expr::parameters::ParamValues,
}

impl QueryExpr {
    fn prepare(expr: &Expr, execution: &Execution, require_real: bool) -> RuntimeResult<Self> {
        if expr.shape().map_err(|e| query_error(e.to_string()))? != ExprShape::Scalar {
            return Err(query_error("dataset expressions must be scalar"));
        }
        let compiled = CompiledModel::from_expr(expr).map_err(|e| query_error(e.to_string()))?;
        if compiled.params().n_free() != 0 {
            return Err(query_error(
                "dataset expressions cannot contain free parameters",
            ));
        }
        if require_real
            && compiled
                .node_facts(compiled.graph().root())
                .is_some_and(|facts| facts.value_kind == ValueKind::Complex)
        {
            return Err(query_error(
                "this dataset operation requires a real-valued expression",
            ));
        }
        let params = compiled.params().default_values();
        let model = PreparedModel::prepare(&compiled, execution)?;
        Ok(Self { model, params })
    }

    fn evaluate_batch(&self, batch: &EventBatch) -> RuntimeResult<Vec<Complex64>> {
        self.model.evaluate_batch(&self.params, batch)
    }
}

enum CompiledPredicate {
    Compare {
        lhs: Box<QueryExpr>,
        op: Comparison,
        rhs: Box<QueryExpr>,
    },
    And(Box<Self>, Box<Self>),
    Or(Box<Self>, Box<Self>),
    Not(Box<Self>),
    Between {
        value: Box<QueryExpr>,
        lower: Box<QueryExpr>,
        upper: Box<QueryExpr>,
        closure: IntervalClosure,
    },
}

impl CompiledPredicate {
    fn prepare(predicate: &Predicate, execution: &Execution) -> RuntimeResult<Self> {
        Ok(match predicate {
            Predicate::Compare { lhs, op, rhs } => Self::Compare {
                lhs: Box::new(QueryExpr::prepare(lhs, execution, true)?),
                op: *op,
                rhs: Box::new(QueryExpr::prepare(rhs, execution, true)?),
            },
            Predicate::And(lhs, rhs) => Self::And(
                Box::new(Self::prepare(lhs, execution)?),
                Box::new(Self::prepare(rhs, execution)?),
            ),
            Predicate::Or(lhs, rhs) => Self::Or(
                Box::new(Self::prepare(lhs, execution)?),
                Box::new(Self::prepare(rhs, execution)?),
            ),
            Predicate::Not(inner) => Self::Not(Box::new(Self::prepare(inner, execution)?)),
            Predicate::Between {
                value,
                lower,
                upper,
                closure,
            } => Self::Between {
                value: Box::new(QueryExpr::prepare(value, execution, true)?),
                lower: Box::new(QueryExpr::prepare(lower, execution, true)?),
                upper: Box::new(QueryExpr::prepare(upper, execution, true)?),
                closure: *closure,
            },
        })
    }

    fn evaluate_batch(&self, batch: &EventBatch) -> RuntimeResult<Vec<bool>> {
        Ok(match self {
            Self::Compare { lhs, op, rhs } => lhs
                .evaluate_batch(batch)?
                .into_iter()
                .zip(rhs.evaluate_batch(batch)?)
                .map(|(lhs, rhs)| compare(lhs.re, *op, rhs.re))
                .collect(),
            Self::And(lhs, rhs) => lhs
                .evaluate_batch(batch)?
                .into_iter()
                .zip(rhs.evaluate_batch(batch)?)
                .map(|(l, r)| l && r)
                .collect(),
            Self::Or(lhs, rhs) => lhs
                .evaluate_batch(batch)?
                .into_iter()
                .zip(rhs.evaluate_batch(batch)?)
                .map(|(l, r)| l || r)
                .collect(),
            Self::Not(inner) => inner
                .evaluate_batch(batch)?
                .into_iter()
                .map(|v| !v)
                .collect(),
            Self::Between {
                value,
                lower,
                upper,
                closure,
            } => value
                .evaluate_batch(batch)?
                .into_iter()
                .zip(lower.evaluate_batch(batch)?)
                .zip(upper.evaluate_batch(batch)?)
                .map(|((value, lower), upper)| {
                    let lower_op = match closure {
                        IntervalClosure::Open | IntervalClosure::RightClosed => Comparison::Gt,
                        IntervalClosure::LeftClosed | IntervalClosure::Closed => Comparison::Ge,
                    };
                    let upper_op = match closure {
                        IntervalClosure::Open | IntervalClosure::LeftClosed => Comparison::Lt,
                        IntervalClosure::RightClosed | IntervalClosure::Closed => Comparison::Le,
                    };
                    compare(value.re, lower_op, lower.re) && compare(value.re, upper_op, upper.re)
                })
                .collect(),
        })
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
    Bin {
        query: Arc<QueryExpr>,
        bins: BinSpec,
        index: usize,
    },
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
        let batches = self.source.batches_with_plan(plan)?;
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
        // TODO: cache evaluated_batch indices somewhere
        match self {
            Self::Predicate(predicate) => Ok(predicate
                .evaluate_batch(batch)?
                .into_iter()
                .enumerate()
                .filter_map(|(row, keep)| keep.then_some(row))
                .collect()),
            Self::Bin { query, bins, index } => Ok(query
                .evaluate_batch(batch)?
                .into_iter()
                .enumerate()
                .filter_map(|(row, value)| (bins.index(value.re) == Some(*index)).then_some(row))
                .collect()),
        }
    }
}

fn query_error(message: impl Into<String>) -> RuntimeError {
    RuntimeError::InvalidShape {
        index: 0,
        message: message.into(),
    }
}
fn data_error(error: impl ToString) -> RuntimeError {
    RuntimeError::Data(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use laddu_data::{
        data::OwnedEvent,
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
    fn bin_edges_validate_and_nan_predicates_are_false() {
        assert!(BinSpec::edges([0.0, 0.0]).is_err());
        assert!(!compare(f64::NAN, Comparison::Ne, 0.0));
    }

    #[test]
    fn selection_and_binning_are_lazy_and_preserve_streaming_policy() {
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
        assert_eq!(reads.load(Ordering::Relaxed), 0);
        assert_eq!(
            selected.cache_storage(),
            laddu_data::data::CacheStorage::Streaming
        );

        assert_eq!(
            selected.map_events(|event| event.scalar(0)).unwrap(),
            vec![0.0, 1.0, 2.0]
        );
        assert_eq!(reads.load(Ordering::Relaxed), 1);
        assert_eq!(
            bins[0]
                .dataset()
                .map_events(|event| event.scalar(0))
                .unwrap(),
            vec![0.0]
        );
        assert_eq!(reads.load(Ordering::Relaxed), 2);
    }
}
