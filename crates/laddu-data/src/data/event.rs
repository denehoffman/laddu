use std::fmt;
use std::sync::Arc;

use laddu_physics::vectors::RealVec4;

use crate::{
    BatchLayout, LadduDataError, LadduDataResult,
    schema::{Precision, Schema},
};

#[derive(Clone, Debug)]
struct BatchParts {
    p4s: Arc<[Arc<[RealVec4]>]>,
    scalars: Arc<[Arc<[f64]>]>,
    weights: Weights,
}

#[derive(Clone, Debug)]
enum Weights {
    ImplicitUnit,
    Explicit(Arc<[f64]>),
}

impl Weights {
    fn from_option(weights: Option<Arc<[f64]>>) -> Self {
        match weights {
            Some(weights) => Self::Explicit(weights),
            None => Self::ImplicitUnit,
        }
    }

    fn as_slice(&self) -> Option<&[f64]> {
        match self {
            Self::ImplicitUnit => None,
            Self::Explicit(weights) => Some(weights),
        }
    }

    fn at(&self, row: usize) -> f64 {
        self.as_slice().map_or(1.0, |weights| weights[row])
    }

    fn is_explicit(&self) -> bool {
        matches!(self, Self::Explicit(_))
    }

    fn select(&self, rows: &[usize]) -> Self {
        match self {
            Self::ImplicitUnit => Self::ImplicitUnit,
            Self::Explicit(weights) => {
                let selected: Arc<[f64]> = rows.iter().map(|&row| weights[row]).collect();
                Self::Explicit(selected)
            }
        }
    }

    fn slice(&self, start: usize, end: usize) -> Self {
        match self {
            Self::ImplicitUnit => Self::ImplicitUnit,
            Self::Explicit(weights) => Self::Explicit(Arc::from(&weights[start..end])),
        }
    }

    fn reweight<F>(&self, len: usize, f: F) -> Self
    where
        F: Fn(usize, f64) -> f64,
    {
        let weights: Arc<[f64]> = (0..len).map(|i| f(i, self.at(i))).collect();
        Self::Explicit(weights)
    }
}

impl BatchParts {
    fn from_columns(p4s: Vec<Arc<[RealVec4]>>, scalars: Vec<Arc<[f64]>>, weights: Weights) -> Self {
        Self {
            p4s: p4s.into(),
            scalars: scalars.into(),
            weights,
        }
    }

    fn validate(&self, schema: &Schema, expected_len: Option<usize>) -> LadduDataResult<usize> {
        if self.p4s.len() != schema.n_p4s() {
            return Err(LadduDataError::Schema(
                "wrong number of vec4 columns".into(),
            ));
        }

        if self.scalars.len() != schema.n_scalars() {
            return Err(LadduDataError::Schema(
                "wrong number of scalar columns".into(),
            ));
        }

        let len = infer_len(&self.p4s, &self.scalars, self.weights.as_slice())?;
        if let Some(expected_len) = expected_len {
            let has_columns =
                !self.p4s.is_empty() || !self.scalars.is_empty() || self.weights.is_explicit();
            if has_columns && len != expected_len {
                return Err(LadduDataError::Schema("inconsistent batch length".into()));
            }
            return Ok(expected_len);
        }
        Ok(len)
    }

    fn select(&self, rows: &[usize]) -> Self {
        let p4s = self
            .p4s
            .iter()
            .map(|col| rows.iter().map(|&i| col[i]).collect())
            .collect();
        let scalars = self
            .scalars
            .iter()
            .map(|col| rows.iter().map(|&i| col[i]).collect())
            .collect();

        Self {
            p4s,
            scalars,
            weights: self.weights.select(rows),
        }
    }

    fn slice(&self, start: usize, end: usize) -> Self {
        let p4s = self
            .p4s
            .iter()
            .map(|col| Arc::<[RealVec4]>::from(&col[start..end]))
            .collect();
        let scalars = self
            .scalars
            .iter()
            .map(|col| Arc::<[f64]>::from(&col[start..end]))
            .collect();

        Self {
            p4s,
            scalars,
            weights: self.weights.slice(start, end),
        }
    }

    fn reweight<F>(&self, len: usize, f: F) -> Self
    where
        F: Fn(usize, f64) -> f64,
    {
        Self {
            p4s: Arc::clone(&self.p4s),
            scalars: Arc::clone(&self.scalars),
            weights: self.weights.reweight(len, f),
        }
    }

    fn concat(batches: &[(&Self, usize)]) -> Self {
        let len: usize = batches.iter().map(|(_, len)| *len).sum();
        let n_p4s = batches.first().map_or(0, |(batch, _)| batch.p4s.len());
        let n_scalars = batches.first().map_or(0, |(batch, _)| batch.scalars.len());

        let mut p4s = Vec::with_capacity(n_p4s);
        for col in 0..n_p4s {
            let mut out = Vec::with_capacity(len);
            for (batch, _) in batches {
                out.extend_from_slice(&batch.p4s[col]);
            }
            p4s.push(Arc::from(out));
        }

        let mut scalars = Vec::with_capacity(n_scalars);
        for col in 0..n_scalars {
            let mut out = Vec::with_capacity(len);
            for (batch, _) in batches {
                out.extend_from_slice(&batch.scalars[col]);
            }
            scalars.push(Arc::from(out));
        }

        let weights = if batches.iter().any(|(batch, _)| batch.weights.is_explicit()) {
            let mut out = Vec::with_capacity(len);
            for (batch, batch_len) in batches {
                for row in 0..*batch_len {
                    out.push(batch.weights.at(row));
                }
            }
            Weights::Explicit(Arc::from(out))
        } else {
            Weights::ImplicitUnit
        };

        Self::from_columns(p4s, scalars, weights)
    }
}

#[derive(Default)]
enum WeightAssembler {
    #[default]
    ImplicitUnit,
    Explicit(Vec<f64>),
}

impl WeightAssembler {
    fn push(&mut self, weight: Option<f64>, len: usize) -> LadduDataResult<()> {
        match self {
            Self::Explicit(weights) => match weight {
                Some(weight) => weights.push(weight),
                None => {
                    return Err(LadduDataError::InvalidArgument(
                        "cannot mix weighted and unweighted events in one batch",
                    ));
                }
            },
            Self::ImplicitUnit => match weight {
                Some(weight) if len == 0 => *self = Self::Explicit(vec![weight]),
                Some(_) => {
                    return Err(LadduDataError::InvalidArgument(
                        "cannot mix unweighted and weighted events in one batch",
                    ));
                }
                None => {}
            },
        }

        Ok(())
    }

    fn finish(self) -> Weights {
        match self {
            Self::ImplicitUnit => Weights::ImplicitUnit,
            Self::Explicit(weights) => Weights::Explicit(Arc::from(weights)),
        }
    }
}

/// Immutable columnar batch of events sharing one schema.
#[derive(Clone)]
pub struct EventBatch {
    schema: Arc<Schema>,
    len: usize,
    parts: BatchParts,
}

impl fmt::Debug for EventBatch {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EventBatch")
            .field("schema", &self.schema)
            .field("len", &self.len)
            .field("p4s", &self.parts.p4s)
            .field("scalars", &self.parts.scalars)
            .field("weights", &self.parts.weights.as_slice())
            .finish()
    }
}

impl EventBatch {
    /// Validates column counts and lengths and constructs a batch.
    ///
    /// # Errors
    ///
    /// Returns [`LadduDataError`] when column counts do not match `schema` or
    /// column and weight lengths are inconsistent.
    pub fn new(
        schema: Arc<Schema>,
        p4s: Vec<Arc<[RealVec4]>>,
        scalars: Vec<Arc<[f64]>>,
        weights: Option<Arc<[f64]>>,
    ) -> LadduDataResult<Self> {
        BatchAssembler::from_columns(schema, p4s, scalars, weights)
    }

    fn from_parts(schema: Arc<Schema>, parts: BatchParts) -> LadduDataResult<Self> {
        let len = parts.validate(&schema, None)?;
        Ok(Self { schema, len, parts })
    }

    fn from_parts_with_len(
        schema: Arc<Schema>,
        parts: BatchParts,
        expected_len: usize,
    ) -> LadduDataResult<Self> {
        let len = parts.validate(&schema, Some(expected_len))?;
        Ok(Self { schema, len, parts })
    }

    /// Collects owned row events into a columnar batch.
    ///
    /// # Errors
    ///
    /// Returns [`LadduDataError`] when an event has the wrong number of values
    /// or weighted and unweighted events are mixed.
    pub fn from_events<I>(schema: Arc<Schema>, events: I) -> LadduDataResult<Self>
    where
        I: IntoIterator<Item = OwnedEvent>,
    {
        let mut builder = EventBatchBuilder::new(schema);
        builder.extend(events)?;
        builder.finish()
    }

    /// Returns the shared logical schema.
    pub fn schema(&self) -> &Arc<Schema> {
        &self.schema
    }

    /// Returns the number of rows.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns the logical payload bytes per event represented by this batch.
    pub fn bytes_per_event(&self) -> usize {
        BatchLayout::from_batch(self)
            .bytes_per_event(Precision::F64)
            .ok()
            .and_then(|bytes| usize::try_from(bytes).ok())
            .unwrap_or(usize::MAX)
    }

    /// Returns the retained column payload size in bytes.
    ///
    /// Shared schema metadata, allocation headers, and other owners of shared
    /// columns are not included.
    pub fn resident_bytes(&self) -> usize {
        BatchLayout::from_batch(self)
            .footprint(Precision::F64)
            .and_then(|footprint| footprint.checked_peak_bytes(self.len))
            .ok()
            .and_then(|bytes| usize::try_from(bytes).ok())
            .unwrap_or(usize::MAX)
    }

    /// Returns whether the batch contains no rows.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns a four-momentum column by index.
    pub fn vec4_column(&self, index: usize) -> &[RealVec4] {
        &self.parts.p4s[index]
    }

    /// Returns a scalar column by index.
    pub fn scalar_column(&self, index: usize) -> &[f64] {
        &self.parts.scalars[index]
    }

    /// Returns the optional explicit weight column.
    pub fn weights_column(&self) -> Option<&[f64]> {
        self.parts.weights.as_slice()
    }

    /// Returns a four-momentum column by logical name.
    pub fn vec4_column_named(&self, name: &str) -> Option<&[RealVec4]> {
        let i = self.schema.p4_index(name)?;
        Some(self.vec4_column(i))
    }

    /// Returns a scalar column by logical name.
    pub fn scalar_column_named(&self, name: &str) -> Option<&[f64]> {
        let i = self.schema.scalar_index(name)?;
        Some(self.scalar_column(i))
    }

    /// Returns one four-momentum cell.
    pub fn p4_at(&self, col: usize, row: usize) -> RealVec4 {
        self.parts.p4s[col][row]
    }

    /// Returns one scalar cell.
    pub fn scalar_at(&self, col: usize, row: usize) -> f64 {
        self.parts.scalars[col][row]
    }

    /// Returns the explicit row weight, or one when weights are absent.
    pub fn weights_at(&self, row: usize) -> f64 {
        self.parts.weights.at(row)
    }

    /// Returns a borrowed view of one row.
    pub fn event(&self, row: usize) -> BatchEvent<'_> {
        BatchEvent { batch: self, row }
    }

    /// Iterates over borrowed event views.
    pub fn iter(&self) -> impl Iterator<Item = BatchEvent<'_>> {
        (0..self.len()).map(|i| self.event(i))
    }

    /// Copies selected rows into a new batch in the requested order.
    ///
    /// # Panics
    ///
    /// Panics when a selected row is outside the batch.
    pub fn select(&self, rows: &[usize]) -> Self {
        Self::from_parts_with_len(
            Arc::clone(&self.schema),
            self.parts.select(rows),
            rows.len(),
        )
        .expect("select preserves EventBatch invariants")
    }

    /// Copies rows satisfying `keep` into a new batch.
    pub fn filter<F>(&self, keep: F) -> Self
    where
        F: Fn(BatchEvent<'_>) -> bool,
    {
        let rows: Vec<usize> = (0..self.len).filter(|&i| keep(self.event(i))).collect();

        self.select(&rows)
    }

    /// Returns a batch sharing value columns with newly computed weights.
    ///
    /// # Panics
    ///
    /// Panics if an internal batch invariant is violated while rebuilding the
    /// batch.
    pub fn reweight<F>(&self, f: F) -> Self
    where
        F: Fn(usize, f64) -> f64,
    {
        Self::from_parts(Arc::clone(&self.schema), self.parts.reweight(self.len, f))
            .expect("reweight preserves EventBatch invariants")
    }

    /// Copies the half-open row range `start..end` into a new batch.
    ///
    /// # Panics
    ///
    /// Panics when `start > end` or `end` exceeds the batch length.
    pub fn slice(&self, start: usize, end: usize) -> Self {
        assert!(start <= end);
        assert!(end <= self.len);

        if start == 0 && end == self.len {
            return self.clone();
        }

        Self::from_parts_with_len(
            Arc::clone(&self.schema),
            self.parts.slice(start, end),
            end - start,
        )
        .expect("slice preserves EventBatch invariants")
    }

    /// Concatenates schema-compatible batches.
    ///
    /// # Errors
    ///
    /// Returns [`LadduDataError`] when `batches` is empty or contains
    /// incompatible schemas.
    pub fn concat(batches: &[Self]) -> LadduDataResult<Self> {
        if batches.is_empty() {
            return Err(LadduDataError::InvalidArgument(
                "cannot concatenate zero batches",
            ));
        }

        let schema = Arc::clone(&batches[0].schema);

        for batch in batches {
            if schema != batch.schema {
                return Err(LadduDataError::Schema(
                    "cannot concatenate batches with different schemas".into(),
                ));
            }
        }

        let parts = batches
            .iter()
            .map(|batch| (&batch.parts, batch.len))
            .collect::<Vec<_>>();
        let len = batches.iter().map(|batch| batch.len).sum();
        Self::from_parts_with_len(schema, BatchParts::concat(&parts), len)
    }
}

fn infer_len(
    vec4s: &[Arc<[RealVec4]>],
    scalars: &[Arc<[f64]>],
    weight: Option<&[f64]>,
) -> LadduDataResult<usize> {
    let len = vec4s
        .first()
        .map(|c| c.len())
        .or_else(|| scalars.first().map(|c| c.len()))
        .or_else(|| weight.map(|w| w.len()))
        .unwrap_or(0);

    for col in vec4s {
        if col.len() != len {
            return Err(LadduDataError::Schema(
                "inconsistent vec4 column length".into(),
            ));
        }
    }

    for col in scalars {
        if col.len() != len {
            return Err(LadduDataError::Schema(
                "inconsistent scalar column length".into(),
            ));
        }
    }

    if let Some(w) = weight
        && w.len() != len
    {
        return Err(LadduDataError::Schema("inconsistent weight length".into()));
    }

    Ok(len)
}

/// Borrowed view of one row in an [`EventBatch`].
#[derive(Copy, Clone, Debug)]
pub struct BatchEvent<'a> {
    batch: &'a EventBatch,
    row: usize,
}

impl<'a> BatchEvent<'a> {
    /// Returns the row index.
    pub fn row(&self) -> usize {
        self.row
    }

    /// Returns the backing batch.
    pub fn batch(&self) -> &'a EventBatch {
        self.batch
    }

    /// Returns a four-momentum value by column index.
    pub fn p4(&self, col: usize) -> RealVec4 {
        self.batch.p4_at(col, self.row)
    }

    /// Returns a scalar value by column index.
    pub fn scalar(&self, col: usize) -> f64 {
        self.batch.scalar_at(col, self.row)
    }

    /// Returns the row weight, defaulting to one.
    pub fn weight(&self) -> f64 {
        self.batch.weights_at(self.row)
    }

    /// Returns a four-momentum value by logical name.
    pub fn p4_named(&self, name: &str) -> Option<RealVec4> {
        let col = self.batch.schema.p4_index(name)?;
        Some(self.p4(col))
    }

    /// Returns a scalar value by logical name.
    pub fn scalar_named(&self, name: &str) -> Option<f64> {
        let col = self.batch.schema.scalar_index(name)?;
        Some(self.scalar(col))
    }
}

/// Borrowed event view with a possibly transformed weight.
#[derive(Copy, Clone, Debug)]
pub struct Event<'a> {
    pub(super) batch: &'a EventBatch,
    pub(super) row: usize,
    pub(super) weight: f64,
}

impl<'a> Event<'a> {
    /// Returns the row index in the backing batch.
    pub fn row(&self) -> usize {
        self.row
    }

    /// Returns a four-momentum value by column index.
    pub fn p4(&self, col: usize) -> RealVec4 {
        self.batch.p4_at(col, self.row)
    }

    /// Returns a scalar value by column index.
    pub fn scalar(&self, col: usize) -> f64 {
        self.batch.scalar_at(col, self.row)
    }

    /// Returns this view's effective weight.
    pub fn weight(&self) -> f64 {
        self.weight
    }

    /// Returns a four-momentum value by logical name.
    pub fn p4_named(&self, name: &str) -> Option<RealVec4> {
        let col = self.batch.schema.p4_index(name)?;
        Some(self.p4(col))
    }

    /// Returns a scalar value by logical name.
    pub fn scalar_named(&self, name: &str) -> Option<f64> {
        let col = self.batch.schema.scalar_index(name)?;
        Some(self.scalar(col))
    }
}

/// Owned row-oriented event used while constructing batches.
#[derive(Clone, Debug)]
pub struct OwnedEvent {
    /// Four-momentum values in schema order.
    pub p4s: Vec<RealVec4>,
    /// Scalar values in schema order.
    pub scalars: Vec<f64>,
    /// Optional explicit event weight.
    pub weight: Option<f64>,
}

impl OwnedEvent {
    /// Creates an unweighted owned event.
    pub fn new(p4s: Vec<RealVec4>, scalars: Vec<f64>) -> Self {
        Self {
            p4s,
            scalars,
            weight: None,
        }
    }

    /// Creates an owned event with an explicit weight.
    pub fn weighted(p4s: Vec<RealVec4>, scalars: Vec<f64>, weight: f64) -> Self {
        Self {
            p4s,
            scalars,
            weight: Some(weight),
        }
    }
}

/// Shared checked assembly for row-oriented batch producers.
pub(crate) struct BatchAssembler {
    schema: Arc<Schema>,
    p4s: Vec<Vec<RealVec4>>,
    scalars: Vec<Vec<f64>>,
    weights: WeightAssembler,
    len: usize,
}

impl BatchAssembler {
    pub(crate) fn new(schema: Arc<Schema>, capacity: usize) -> Self {
        let p4s = (0..schema.n_p4s())
            .map(|_| Vec::with_capacity(capacity))
            .collect();
        let scalars = (0..schema.n_scalars())
            .map(|_| Vec::with_capacity(capacity))
            .collect();

        Self {
            schema,
            p4s,
            scalars,
            weights: WeightAssembler::default(),
            len: 0,
        }
    }

    pub(crate) fn with_weight_mode(
        schema: Arc<Schema>,
        capacity: usize,
        explicit_weights: bool,
    ) -> Self {
        let mut assembler = Self::new(schema, capacity);
        if explicit_weights {
            assembler.weights = WeightAssembler::Explicit(Vec::with_capacity(capacity));
        }
        assembler
    }

    pub(crate) fn from_columns(
        schema: Arc<Schema>,
        p4s: Vec<Arc<[RealVec4]>>,
        scalars: Vec<Arc<[f64]>>,
        weights: Option<Arc<[f64]>>,
    ) -> LadduDataResult<EventBatch> {
        EventBatch::from_parts(
            schema,
            BatchParts::from_columns(p4s, scalars, Weights::from_option(weights)),
        )
    }

    fn push_owned(&mut self, event: OwnedEvent) -> LadduDataResult<()> {
        if event.p4s.len() != self.schema.n_p4s() {
            return Err(LadduDataError::Schema(
                "wrong number of event vec4 values".into(),
            ));
        }

        if event.scalars.len() != self.schema.n_scalars() {
            return Err(LadduDataError::Schema(
                "wrong number of event scalar values".into(),
            ));
        }

        self.weights.push(event.weight, self.len)?;

        for (col, value) in event.p4s.into_iter().enumerate() {
            self.p4s[col].push(value);
        }
        for (col, value) in event.scalars.into_iter().enumerate() {
            self.scalars[col].push(value);
        }

        self.len += 1;
        Ok(())
    }

    pub(crate) fn push_borrowed(
        &mut self,
        event: Event<'_>,
        explicit_weight: bool,
    ) -> LadduDataResult<()> {
        if event.batch.schema().n_p4s() != self.schema.n_p4s() {
            return Err(LadduDataError::Schema(
                "wrong number of event vec4 values".into(),
            ));
        }

        if event.batch.schema().n_scalars() != self.schema.n_scalars() {
            return Err(LadduDataError::Schema(
                "wrong number of event scalar values".into(),
            ));
        }

        self.weights
            .push(explicit_weight.then_some(event.weight()), self.len)?;

        for col in 0..self.schema.n_p4s() {
            self.p4s[col].push(event.p4(col));
        }
        for col in 0..self.schema.n_scalars() {
            self.scalars[col].push(event.scalar(col));
        }

        self.len += 1;
        Ok(())
    }

    pub(crate) fn finish(self) -> LadduDataResult<EventBatch> {
        let parts = BatchParts::from_columns(
            self.p4s.into_iter().map(Arc::from).collect(),
            self.scalars.into_iter().map(Arc::from).collect(),
            self.weights.finish(),
        );
        EventBatch::from_parts_with_len(self.schema, parts, self.len)
    }
}

/// Incremental builder for a columnar [`EventBatch`].
pub struct EventBatchBuilder {
    assembler: BatchAssembler,
}

impl EventBatchBuilder {
    /// Creates an empty builder.
    pub fn new(schema: Arc<Schema>) -> Self {
        Self::with_capacity(schema, 0)
    }

    /// Creates an empty builder with per-column capacity.
    pub fn with_capacity(schema: Arc<Schema>, capacity: usize) -> Self {
        Self {
            assembler: BatchAssembler::new(schema, capacity),
        }
    }

    /// Appends an unweighted event from ordered values.
    ///
    /// # Errors
    ///
    /// Returns [`LadduDataError`] when value counts do not match the schema or
    /// the builder already contains weighted events.
    pub fn push<P, S>(&mut self, p4s: P, scalars: S) -> LadduDataResult<&mut Self>
    where
        P: IntoIterator<Item = RealVec4>,
        S: IntoIterator<Item = f64>,
    {
        self.push_event(OwnedEvent::new(
            p4s.into_iter().collect(),
            scalars.into_iter().collect(),
        ))
    }

    /// Appends a weighted event from ordered values.
    ///
    /// # Errors
    ///
    /// Returns [`LadduDataError`] when value counts do not match the schema or
    /// the builder already contains unweighted events.
    pub fn push_weighted<P, S>(
        &mut self,
        p4s: P,
        scalars: S,
        weight: f64,
    ) -> LadduDataResult<&mut Self>
    where
        P: IntoIterator<Item = RealVec4>,
        S: IntoIterator<Item = f64>,
    {
        self.push_event(OwnedEvent::weighted(
            p4s.into_iter().collect(),
            scalars.into_iter().collect(),
            weight,
        ))
    }

    /// Validates and appends one owned event.
    ///
    /// # Errors
    ///
    /// Returns [`LadduDataError`] when the event shape does not match the
    /// schema or its weight presence differs from prior events.
    pub fn push_event(&mut self, event: OwnedEvent) -> LadduDataResult<&mut Self> {
        self.assembler.push_owned(event)?;
        Ok(self)
    }

    /// Appends all owned events from an iterator.
    ///
    /// # Errors
    ///
    /// Returns the first [`LadduDataError`] produced by an event whose shape or
    /// weight presence is incompatible with the builder.
    pub fn extend<I>(&mut self, events: I) -> LadduDataResult<&mut Self>
    where
        I: IntoIterator<Item = OwnedEvent>,
    {
        for event in events {
            self.push_event(event)?;
        }

        Ok(self)
    }

    /// Finalizes the builder into an immutable batch.
    ///
    /// # Errors
    ///
    /// Returns [`LadduDataError`] if the accumulated columns or weights have
    /// inconsistent lengths.
    pub fn finish(self) -> LadduDataResult<EventBatch> {
        self.assembler.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(x: f64) -> RealVec4 {
        RealVec4 {
            e: x + 0.3,
            px: x,
            py: x + 0.1,
            pz: x + 0.2,
        }
    }

    fn schema_with_weight() -> Arc<Schema> {
        Arc::new(Schema::new(["p"], ["x"], true).unwrap())
    }

    fn weighted_batch(start: usize, len: usize) -> EventBatch {
        let schema = schema_with_weight();

        let events = (start..start + len)
            .map(|i| OwnedEvent::weighted(vec![v(i as f64)], vec![i as f64], 10.0 + i as f64));

        EventBatch::from_events(schema, events).unwrap()
    }

    fn scalar_values(batch: &EventBatch) -> Vec<f64> {
        batch.scalar_column(0).to_vec()
    }

    #[test]
    fn event_batch_rejects_shape_mismatches_and_builder_rejects_mixed_weights() {
        let schema = schema_with_weight();

        let bad_vec4_count = EventBatch::new(
            Arc::clone(&schema),
            vec![],
            vec![Arc::from([1.0, 2.0])],
            Some(Arc::from([1.0, 2.0])),
        );

        assert!(matches!(bad_vec4_count, Err(LadduDataError::Schema(_))));

        let bad_lengths = EventBatch::new(
            Arc::clone(&schema),
            vec![Arc::from([v(1.0), v(2.0)])],
            vec![Arc::from([1.0])],
            Some(Arc::from([1.0, 2.0])),
        );

        assert!(matches!(bad_lengths, Err(LadduDataError::Schema(_))));

        let mut builder = EventBatchBuilder::new(schema);
        builder.push([v(1.0)], [1.0]).unwrap();

        let mixed = builder.push_weighted([v(2.0)], [2.0], 2.0);
        assert!(matches!(mixed, Err(LadduDataError::InvalidArgument(_))));
    }

    #[test]
    fn select_slice_filter_reweight_and_concat_preserve_columns_and_weight_semantics() {
        let weighted = weighted_batch(0, 4);
        let selected = weighted.select(&[3, 1]);

        assert_eq!(scalar_values(&selected), vec![3.0, 1.0]);
        assert_eq!(selected.weights_column().unwrap(), &[13.0, 11.0]);
        assert_eq!(selected.p4_at(0, 0).px, 3.0);
        assert_eq!(selected.p4_at(0, 1).e, 1.3);

        let sliced = weighted.slice(1, 3);
        assert_eq!(scalar_values(&sliced), vec![1.0, 2.0]);
        assert_eq!(sliced.weights_column().unwrap(), &[11.0, 12.0]);

        let filtered = weighted.filter(|ev| ev.scalar(0) >= 2.0);
        assert_eq!(scalar_values(&filtered), vec![2.0, 3.0]);

        let reweighted = filtered.reweight(|i, w| w + 100.0 + i as f64);
        assert_eq!(reweighted.weights_column().unwrap(), &[112.0, 114.0]);

        let schema = schema_with_weight();

        let unweighted_with_weight_schema = EventBatch::from_events(
            Arc::clone(&schema),
            [
                OwnedEvent::new(vec![v(100.0)], vec![100.0]),
                OwnedEvent::new(vec![v(101.0)], vec![101.0]),
            ],
        )
        .unwrap();

        let weighted_tail = EventBatch::from_events(
            schema,
            [
                OwnedEvent::weighted(vec![v(200.0)], vec![200.0], 5.0),
                OwnedEvent::weighted(vec![v(201.0)], vec![201.0], 6.0),
            ],
        )
        .unwrap();

        let concatenated =
            EventBatch::concat(&[unweighted_with_weight_schema, weighted_tail]).unwrap();

        assert_eq!(
            scalar_values(&concatenated),
            vec![100.0, 101.0, 200.0, 201.0]
        );
        assert_eq!(
            concatenated.weights_column().unwrap(),
            &[1.0, 1.0, 5.0, 6.0]
        );
    }

    #[test]
    fn implicit_unit_weights_survive_assembly_and_row_transforms() {
        let schema = Arc::new(Schema::new(["p"], ["x"], false).unwrap());
        let batch = EventBatch::from_events(
            Arc::clone(&schema),
            (0..3).map(|i| OwnedEvent::new(vec![v(i as f64)], vec![i as f64])),
        )
        .unwrap();

        assert!(batch.weights_column().is_none());
        assert_eq!(batch.weights_at(2), 1.0);

        let selected = batch.select(&[2, 0]);
        let sliced = batch.slice(1, 3);
        let filtered = batch.filter(|event| event.scalar(0) > 0.0);
        let concatenated = EventBatch::concat(&[selected, sliced]).unwrap();

        assert!(filtered.weights_column().is_none());
        assert!(concatenated.weights_column().is_none());
        assert_eq!(concatenated.weights_at(3), 1.0);

        let reweighted = batch.reweight(|row, weight| weight + row as f64);
        assert_eq!(reweighted.weights_column().unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn shared_assembler_preserves_weight_mode_and_rejects_transitions() {
        let schema = Arc::new(Schema::new(["p"], ["x"], true).unwrap());
        let source = EventBatch::from_events(
            Arc::clone(&schema),
            [
                OwnedEvent::weighted(vec![v(1.0)], vec![1.0], 2.0),
                OwnedEvent::weighted(vec![v(2.0)], vec![2.0], 3.0),
            ],
        )
        .unwrap();

        let mut explicit = BatchAssembler::new(Arc::clone(&schema), 2);
        let first = Event {
            batch: &source,
            row: 0,
            weight: source.weights_at(0),
        };
        explicit.push_borrowed(first, true).unwrap();
        let second = Event {
            batch: &source,
            row: 1,
            weight: source.weights_at(1),
        };
        let transition = explicit.push_borrowed(second, false);
        assert!(matches!(
            transition,
            Err(LadduDataError::InvalidArgument(_))
        ));
        let explicit = explicit.finish().unwrap();
        assert_eq!(explicit.len(), 1);
        assert_eq!(explicit.weights_column().unwrap(), &[2.0]);

        let unweighted = EventBatch::from_events(
            Arc::clone(&schema),
            [OwnedEvent::new(vec![v(3.0)], vec![3.0])],
        )
        .unwrap();
        let mut implicit = BatchAssembler::new(schema, 1);
        let event = Event {
            batch: &unweighted,
            row: 0,
            weight: unweighted.weights_at(0),
        };
        implicit.push_borrowed(event, false).unwrap();
        let implicit = implicit.finish().unwrap();
        assert!(implicit.weights_column().is_none());
        assert_eq!(implicit.weights_at(0), 1.0);
    }

    #[test]
    fn assembly_table_covers_column_shapes_and_observable_sharing() {
        let cases = [
            (
                Arc::new(Schema::new(Vec::<&str>::new(), Vec::<&str>::new(), false).unwrap()),
                false,
            ),
            (
                Arc::new(Schema::new(["p"], Vec::<&str>::new(), false).unwrap()),
                false,
            ),
            (
                Arc::new(Schema::new(Vec::<&str>::new(), ["x"], false).unwrap()),
                false,
            ),
            (
                Arc::new(Schema::new(Vec::<&str>::new(), Vec::<&str>::new(), true).unwrap()),
                true,
            ),
            (Arc::new(Schema::new(["p"], ["x"], true).unwrap()), true),
        ];

        for (schema, weighted) in cases {
            let events = (0..2).map(|i| {
                let p4s = if schema.n_p4s() == 0 {
                    Vec::new()
                } else {
                    vec![v(i as f64)]
                };
                let scalars = if schema.n_scalars() == 0 {
                    Vec::new()
                } else {
                    vec![i as f64]
                };
                if weighted {
                    OwnedEvent::weighted(p4s, scalars, 2.0 + i as f64)
                } else {
                    OwnedEvent::new(p4s, scalars)
                }
            });
            let batch = EventBatch::from_events(Arc::clone(&schema), events).unwrap();
            assert_eq!(batch.len(), 2);
            assert_eq!(batch.weights_column().is_some(), weighted);

            let selected = batch.select(&(0..batch.len()).collect::<Vec<_>>());
            assert_eq!(selected.len(), batch.len());
            for col in 0..schema.n_p4s() {
                assert_eq!(selected.vec4_column(col), batch.vec4_column(col));
            }
            for col in 0..schema.n_scalars() {
                assert_eq!(selected.scalar_column(col), batch.scalar_column(col));
            }
            assert_eq!(selected.weights_column(), batch.weights_column());

            let concatenated = EventBatch::concat(&[batch.slice(0, 1), batch.slice(1, 2)]).unwrap();
            assert_eq!(concatenated.len(), batch.len());
            for col in 0..schema.n_p4s() {
                assert_eq!(concatenated.vec4_column(col), batch.vec4_column(col));
            }
            for col in 0..schema.n_scalars() {
                assert_eq!(concatenated.scalar_column(col), batch.scalar_column(col));
            }
            assert_eq!(concatenated.weights_column(), batch.weights_column());

            if schema.n_p4s() > 0 || schema.n_scalars() > 0 {
                let reweighted = batch.reweight(|_, weight| weight + 1.0);
                if schema.n_p4s() > 0 {
                    assert_eq!(
                        reweighted.vec4_column(0).as_ptr(),
                        batch.vec4_column(0).as_ptr()
                    );
                }
                if schema.n_scalars() > 0 {
                    assert_eq!(
                        reweighted.scalar_column(0).as_ptr(),
                        batch.scalar_column(0).as_ptr()
                    );
                }
            }
        }
    }
}
