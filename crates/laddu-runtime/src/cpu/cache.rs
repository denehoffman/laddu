#[cfg(feature = "jit")]
use std::marker::PhantomData;
use std::{mem::size_of, sync::Arc};

use laddu_compile::CachePlan;
use laddu_data::{
    data::{CacheStorage, Dataset, EventBatch},
    io::ReadPlan,
};
use laddu_expr::{ExprId, ValueKind};
use nalgebra::{DMatrix, DVector};
use num::complex::Complex64;

use super::layout::{FlatRows, matrix_at_optional};
use super::{CpuPlan, DynamicLu, PreparedDatasetStats, RuntimeError, RuntimeResult, Value};
use crate::MemoryLease;

/// Raw cache payload metadata consumed by compiled JIT kernels.
#[cfg(feature = "jit")]
#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct CacheDescriptor {
    pub(crate) values: *const u8,
    pub(crate) width: usize,
}

#[cfg(feature = "jit")]
pub(crate) struct JitDescriptorSet<'a> {
    pub(crate) values: Vec<CacheDescriptor>,
    pub(crate) solve_rows: Vec<CacheDescriptor>,
    pub(crate) _cache: PhantomData<&'a CpuBatchCache>,
}

/// Materialized event-dependent values for one batch.
#[derive(Clone, Debug)]
pub struct CpuBatchCache {
    pub(super) len: usize,
    pub(super) weights: Vec<f64>,
    pub(super) sum_weights: f64,
    pub(super) nodes: Vec<ExprId>,
    pub(crate) slots: Vec<CachedSlot>,
    pub(super) factor_nodes: Vec<ExprId>,
    pub(super) factor_slots: Vec<CachedFactorSlot>,
    pub(super) solve_row_keys: Vec<(ExprId, usize, usize)>,
    pub(crate) solve_row_slots: Vec<CachedSolveRowSlot>,
}

impl CpuBatchCache {
    pub(super) fn new(
        cache_plan: &CachePlan,
        factor_matrices: &[(ExprId, usize)],
        solve_row_keys: &[(ExprId, usize, usize)],
        len: usize,
    ) -> RuntimeResult<Self> {
        let slots = cache_plan
            .entries()
            .iter()
            .map(|entry| CachedSlot::new(entry.value_kind(), len))
            .collect::<RuntimeResult<Vec<_>>>()?;
        let solve_row_slots = solve_row_keys
            .iter()
            .map(|(_, _, dimension)| CachedSolveRowSlot::new(*dimension, len))
            .collect::<RuntimeResult<Vec<_>>>()?;
        Ok(Self {
            len,
            weights: vec![1.0; len],
            sum_weights: len as f64,
            nodes: cache_plan
                .entries()
                .iter()
                .map(|entry| entry.node())
                .collect(),
            slots,
            factor_nodes: factor_matrices.iter().map(|(node, _)| *node).collect(),
            factor_slots: factor_matrices
                .iter()
                .map(|(_, dimension)| CachedFactorSlot::new(*dimension))
                .collect(),
            solve_row_keys: solve_row_keys.to_vec(),
            solve_row_slots,
        })
    }

    /// Returns the number of cached events.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns whether the cache contains no events.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns per-event weights.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Returns the sum of event weights.
    pub fn sum_weights(&self) -> f64 {
        self.sum_weights
    }

    /// Returns the raw cache payload metadata required by the JIT ABI.
    ///
    /// The returned pointers borrow this cache and are valid only while it is
    /// immutably borrowed.  Keeping this projection here prevents the JIT
    /// backend from depending on cache-slot representation details.
    #[cfg(feature = "jit")]
    #[cfg(feature = "jit")]
    pub(crate) fn jit_descriptors(&self) -> JitDescriptorSet<'_> {
        JitDescriptorSet {
            values: self
                .slots
                .iter()
                .map(|slot| CacheDescriptor {
                    values: slot.values_ptr(),
                    width: slot.width(),
                })
                .collect(),
            solve_rows: self
                .solve_row_slots
                .iter()
                .map(|slot| CacheDescriptor {
                    values: slot.values.as_ptr().cast(),
                    width: slot.dimension,
                })
                .collect(),
            _cache: PhantomData,
        }
    }

    /// Estimates heap memory retained by this cache, in bytes.
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

    pub(super) fn set_weights(&mut self, weights: Vec<f64>) {
        self.sum_weights = weights.iter().sum();
        self.weights = weights;
    }

    pub(super) fn push(&mut self, slot: usize, value: Value) -> RuntimeResult<()> {
        let len = self.slots.len();
        self.slots
            .get_mut(slot)
            .ok_or(RuntimeError::InvalidCache {
                expected: len,
                actual: slot + 1,
            })?
            .push(value)
    }

    pub(super) fn value(&self, slot: usize, row: usize) -> RuntimeResult<Value> {
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

    pub(super) fn scalar(&self, slot: usize, row: usize) -> RuntimeResult<Complex64> {
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

    pub(super) fn real_range(
        &self,
        slot: usize,
        start: usize,
        end: usize,
    ) -> RuntimeResult<&[f64]> {
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

    pub(super) fn complex_range(
        &self,
        slot: usize,
        start: usize,
        end: usize,
    ) -> RuntimeResult<&[Complex64]> {
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

    pub(super) fn push_factor(&mut self, slot: usize, factor: DynamicLu) -> RuntimeResult<()> {
        let len = self.factor_slots.len();
        self.factor_slots
            .get_mut(slot)
            .ok_or(RuntimeError::InvalidCache {
                expected: len,
                actual: slot + 1,
            })?
            .push(factor)
    }

    pub(super) fn factor(&self, slot: usize, row: usize) -> RuntimeResult<&DynamicLu> {
        self.factor_slots
            .get(slot)
            .ok_or(RuntimeError::InvalidCache {
                expected: self.factor_slots.len(),
                actual: slot + 1,
            })?
            .factor(row)
    }

    pub(super) fn push_solve_row(
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

    pub(super) fn solve_row(&self, slot: usize, row: usize) -> RuntimeResult<&[Complex64]> {
        self.solve_row_slots
            .get(slot)
            .ok_or(RuntimeError::InvalidCache {
                expected: self.solve_row_slots.len(),
                actual: slot + 1,
            })?
            .row(row)
    }
}

impl CpuPlan {
    pub(super) fn materialize_cache_event_batch(
        &self,
        batch: &EventBatch,
    ) -> RuntimeResult<CpuBatchCache> {
        let event_columns = self.event_columns(batch.schema())?;
        let mut cache = CpuBatchCache::new(
            &self.cache_plan,
            &self.factor_matrices,
            &self.solve_row_keys,
            batch.len(),
        )?;
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
                let (rows, cols, values) = matrix_at_optional(&values, plan.matrix().index())?;
                if rows != plan.dimension() || cols != plan.dimension() {
                    return Err(RuntimeError::InvalidShape {
                        index: plan.matrix().index(),
                        message: format!(
                            "specialized solve expected a {}x{} matrix, got {rows}x{cols}",
                            plan.dimension(),
                            plan.dimension()
                        ),
                    });
                }
                let transpose_factor = DMatrix::from_row_slice(rows, cols, values).transpose().lu();
                for (slot, index) in plan.rows() {
                    let mut basis = DVector::zeros(plan.dimension());
                    basis[*index] = Complex64::ONE;
                    let inverse_row = transpose_factor
                        .solve(&basis)
                        .ok_or(RuntimeError::SingularMatrix(plan.matrix().index()))?;
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
}

/// A cached event batch and its associated weights.
#[derive(Clone, Debug)]
pub struct CpuCachedBatch {
    pub(super) cache: CpuBatchCache,
}

impl CpuCachedBatch {
    pub(crate) fn from_cache(cache: CpuBatchCache) -> Self {
        Self { cache }
    }

    /// Returns the underlying materialized cache.
    pub fn cache(&self) -> &CpuBatchCache {
        &self.cache
    }

    /// Returns the number of events.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Returns whether the batch contains no events.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Returns per-event weights.
    pub fn weights(&self) -> &[f64] {
        self.cache.weights()
    }

    /// Returns the sum of event weights.
    pub fn sum_weights(&self) -> f64 {
        self.cache.sum_weights()
    }

    /// Estimates retained heap memory, in bytes.
    pub fn resident_bytes(&self) -> usize {
        self.cache.resident_bytes()
    }
}

/// A dataset whose event-dependent model values are fully cached in memory.
#[derive(Clone, Debug, Default)]
pub struct CpuCachedDataset {
    pub(super) batches: Vec<CpuCachedBatch>,
    pub(super) sum_weights: f64,
}

impl PreparedDatasetStats {
    pub(crate) fn new(
        local_events: usize,
        global_events: usize,
        local_batches: usize,
        sum_weights: f64,
        resident_bytes: usize,
        storage: CacheStorage,
    ) -> Self {
        Self {
            local_events,
            global_events,
            local_batches,
            sum_weights,
            resident_bytes,
            storage,
        }
    }

    /// Returns the number of events assigned to this rank.
    pub fn local_events(&self) -> usize {
        self.local_events
    }

    /// Returns the total number of events across all ranks.
    pub fn global_events(&self) -> usize {
        self.global_events
    }

    /// Returns the number of batches assigned to this rank.
    pub fn local_batches(&self) -> usize {
        self.local_batches
    }

    /// Returns the total event-weight sum across all ranks.
    pub fn sum_weights(&self) -> f64 {
        self.sum_weights
    }

    /// Returns the number of bytes retained for prepared data on this rank.
    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }

    /// Returns the dataset's cache-storage policy.
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
    /// A dataset whose event caches are resident in memory.
    Resident {
        /// Fully cached dataset.
        dataset: Arc<CpuCachedDataset>,
        /// Preparation statistics.
        stats: PreparedDatasetStats,
        /// Persistent host-memory reservation shared by clones.
        memory_lease: MemoryLease,
    },
    /// A dataset whose event caches are rebuilt while streaming.
    Streaming {
        /// Source dataset.
        dataset: Dataset,
        /// Read plan used for each pass.
        read_plan: ReadPlan,
        /// Preparation statistics.
        stats: PreparedDatasetStats,
        /// Peak transient bytes reserved during each reduction.
        transient_bytes: u64,
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
    /// Returns statistics collected while preparing the dataset.
    pub fn stats(&self) -> &PreparedDatasetStats {
        match self {
            Self::Resident { stats, .. } | Self::Streaming { stats, .. } => stats,
        }
    }
}

impl CpuCachedDataset {
    pub(crate) fn from_parts(batches: Vec<CpuCachedBatch>, sum_weights: f64) -> Self {
        Self {
            batches,
            sum_weights,
        }
    }

    /// Returns the cached batches.
    pub fn batches(&self) -> &[CpuCachedBatch] {
        &self.batches
    }

    /// Returns the total number of cached events.
    pub fn len(&self) -> usize {
        self.batches.iter().map(CpuCachedBatch::len).sum()
    }

    /// Returns whether the dataset contains no events.
    pub fn is_empty(&self) -> bool {
        self.batches.iter().all(CpuCachedBatch::is_empty)
    }

    /// Returns the sum of all event weights.
    pub fn sum_weights(&self) -> f64 {
        self.sum_weights
    }

    /// Estimates retained heap memory, in bytes.
    pub fn resident_bytes(&self) -> usize {
        self.batches
            .iter()
            .map(CpuCachedBatch::resident_bytes)
            .sum()
    }
}

#[derive(Clone, Debug)]
pub(super) struct CachedFactorSlot {
    dimension: usize,
    factors: Vec<DynamicLu>,
}

#[derive(Clone, Debug)]
pub(crate) struct CachedSolveRowSlot {
    #[cfg_attr(not(feature = "jit"), allow(dead_code))]
    pub(crate) dimension: usize,
    pub(crate) values: FlatRows<Complex64>,
}

impl CachedSolveRowSlot {
    fn new(dimension: usize, events: usize) -> RuntimeResult<Self> {
        Ok(Self {
            dimension,
            values: FlatRows::try_with_capacity(dimension, events)?,
        })
    }

    fn push(&mut self, values: impl IntoIterator<Item = Complex64>) -> RuntimeResult<()> {
        self.values.push_row(values)
    }

    fn row(&self, row: usize) -> RuntimeResult<&[Complex64]> {
        self.values.row(row)
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
        values: FlatRows<Complex64>,
    },
    Matrix {
        rows: usize,
        cols: usize,
        values: FlatRows<Complex64>,
    },
}

impl CachedSlot {
    #[cfg(feature = "jit")]
    pub(crate) fn values_ptr(&self) -> *const u8 {
        match self {
            Self::Real(values) => values.as_ptr().cast(),
            Self::Complex(values) => values.as_ptr().cast(),
            Self::Vector { values, .. } | Self::Matrix { values, .. } => values.as_ptr().cast(),
        }
    }

    #[cfg(feature = "jit")]
    pub(crate) fn width(&self) -> usize {
        match self {
            Self::Real(_) | Self::Complex(_) => 1,
            Self::Vector { values, .. } | Self::Matrix { values, .. } => values.width(),
        }
    }

    fn new(kind: ValueKind, events: usize) -> RuntimeResult<Self> {
        Ok(match kind {
            ValueKind::Real => Self::Real(Vec::with_capacity(events)),
            ValueKind::Complex => Self::Complex(Vec::with_capacity(events)),
            ValueKind::Vector { len } => Self::Vector {
                len,
                values: FlatRows::try_with_capacity(len, events)?,
            },
            ValueKind::Matrix { rows, cols } => Self::Matrix {
                rows,
                cols,
                values: FlatRows::try_with_capacity(
                    rows.checked_mul(cols)
                        .ok_or_else(|| RuntimeError::InvalidShape {
                            index: rows,
                            message: format!("matrix width overflowed for {rows}x{cols}"),
                        })?,
                    events,
                )?,
            },
        })
    }

    pub(crate) fn resident_bytes(&self) -> usize {
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
                values.push_row(value)
            }
            (
                Self::Matrix { rows, cols, values },
                Value::Matrix {
                    rows: value_rows,
                    cols: value_cols,
                    values: value,
                },
            ) if *rows == value_rows && *cols == value_cols => values.push_row(value),
            (_, value) => Err(RuntimeError::InvalidShape {
                index: 0,
                message: format!("cached value kind did not match slot: {}", value.kind()),
            }),
        }
    }

    pub(super) fn value(&self, row: usize) -> RuntimeResult<Value> {
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
            Self::Vector { values, .. } => {
                values.row(row).map(|value| Value::Vector(value.to_vec()))
            }
            Self::Matrix { rows, cols, values } => values.row(row).map(|value| Value::Matrix {
                rows: *rows,
                cols: *cols,
                values: value.to_vec(),
            }),
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
