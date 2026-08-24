use std::sync::Arc;

use crate::{
    LadduDataError, LadduDataResult,
    data::{EventBatch, OwnedEvent},
    io::{
        DataFragment, EventSink, EventSource, FragmentedSource, ReadPlan, SinkState,
        SourceCapabilities, WritePlan, fragmented_batches,
    },
    schema::Schema,
};

/// Replayable in-memory event source backed by one or more batches.
#[derive(Clone, Debug)]
pub struct MemorySource {
    schema: Arc<Schema>,
    batches: Arc<[EventBatch]>,
}

/// Key identifying one batch fragment in a [`MemorySource`].
#[derive(Clone, Copy, Debug)]
pub struct MemoryFragmentKey {
    batch_index: usize,
}

impl MemorySource {
    /// Creates an empty replayable source with the supplied schema.
    ///
    /// Empty derived datasets retain schema and source capabilities while
    /// yielding no batches. This is useful for partitioning operations where
    /// a valid empty result is distinct from an invalid source definition.
    pub fn empty(schema: Arc<Schema>) -> Self {
        Self {
            schema,
            batches: Arc::from([]),
        }
    }

    /// Creates a source containing one batch.
    pub fn new(batch: EventBatch) -> Self {
        Self {
            schema: Arc::clone(batch.schema()),
            batches: Arc::from([batch]),
        }
    }

    /// Validates and creates a source from nonempty schema-compatible batches.
    ///
    /// # Errors
    ///
    /// Returns [`LadduDataError`] when `batches` is empty or contains
    /// incompatible schemas.
    pub fn from_batches(batches: Vec<EventBatch>) -> LadduDataResult<Self> {
        if batches.is_empty() {
            return Err(LadduDataError::InvalidArgument(
                "memory source requires at least one batch",
            ));
        }

        let schema = Arc::clone(batches[0].schema());

        for batch in &batches {
            if batch.schema().as_ref() != schema.as_ref() {
                return Err(LadduDataError::Schema(
                    "memory source batches have different schemas".into(),
                ));
            }
        }

        Ok(Self {
            schema,
            batches: batches.into(),
        })
    }

    /// Collects owned events into an in-memory source.
    ///
    /// # Errors
    ///
    /// Returns [`LadduDataError`] when an event does not match `schema` or
    /// weighted and unweighted events are mixed.
    pub fn from_events<I>(schema: Arc<Schema>, events: I) -> LadduDataResult<Self>
    where
        I: IntoIterator<Item = OwnedEvent>,
    {
        let batch = EventBatch::from_events(schema, events)?;
        Ok(Self::new(batch))
    }

    /// Returns the shared schema.
    pub fn schema_arc(&self) -> &Arc<Schema> {
        &self.schema
    }

    /// Returns the backing batches.
    pub fn batches_slice(&self) -> &[EventBatch] {
        &self.batches
    }

    /// Consumes the source and returns its shared batches.
    pub fn into_batches(self) -> Arc<[EventBatch]> {
        self.batches
    }

    /// Consumes and concatenates all batches.
    ///
    /// # Errors
    ///
    /// Returns [`LadduDataError`] when no batches are present or their schemas
    /// are incompatible.
    pub fn into_batch(self) -> LadduDataResult<EventBatch> {
        EventBatch::concat(&self.batches)
    }
}

impl EventSource for MemorySource {
    fn schema(&self) -> LadduDataResult<Arc<Schema>> {
        Ok(Arc::clone(&self.schema))
    }

    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            exact_len: true,
            exact_weighted_total: true,
            random_access: true,
            deterministic_partitioning: true,
            predicate_pushdown: false,
            projection_pushdown: false,
            streaming: false,
        }
    }

    fn num_events(&self) -> LadduDataResult<Option<u64>> {
        Ok(Some(self.batches.iter().map(|b| b.len() as u64).sum()))
    }

    fn weighted_total(&self) -> LadduDataResult<Option<f64>> {
        let total = self
            .batches
            .iter()
            .map(|batch| (0..batch.len()).map(|i| batch.weights_at(i)).sum::<f64>())
            .sum();

        Ok(Some(total))
    }

    fn batches(
        &self,
        plan: ReadPlan,
    ) -> LadduDataResult<Box<dyn Iterator<Item = LadduDataResult<EventBatch>> + Send>> {
        fragmented_batches(Arc::new(self.clone()), plan)
    }
}

impl FragmentedSource for MemorySource {
    type Key = MemoryFragmentKey;

    fn fragments(&self) -> LadduDataResult<Vec<DataFragment<Self::Key>>> {
        let mut fragments = Vec::with_capacity(self.batches.len());
        let mut global_start = 0_u64;

        for (batch_index, batch) in self.batches.iter().enumerate() {
            let rows = batch.len() as u64;

            fragments.push(DataFragment {
                key: MemoryFragmentKey { batch_index },
                global_start,
                rows,
            });

            global_start += rows;
        }

        Ok(fragments)
    }

    fn read_fragment_range(
        &self,
        key: &Self::Key,
        local_start: usize,
        local_len: usize,
        chunk_size: Option<usize>,
    ) -> LadduDataResult<Box<dyn Iterator<Item = LadduDataResult<EventBatch>> + Send>> {
        if matches!(chunk_size, Some(0)) {
            return Err(LadduDataError::InvalidArgument(
                "chunk_size must be nonzero",
            ));
        }

        let batch = self
            .batches
            .get(key.batch_index)
            .ok_or_else(|| LadduDataError::Source("invalid memory batch index".into()))?
            .clone();

        let end = local_start
            .checked_add(local_len)
            .ok_or(LadduDataError::InvalidArgument(
                "slice range overflows usize",
            ))?;

        if end > batch.len() {
            return Err(LadduDataError::InvalidArgument(
                "memory fragment range exceeds batch length",
            ));
        }

        Ok(Box::new(MemoryRangeIter {
            batch,
            pos: local_start,
            end,
            chunk_size: chunk_size.unwrap_or(local_len.max(1)),
        }))
    }
}

struct MemoryRangeIter {
    batch: EventBatch,
    pos: usize,
    end: usize,
    chunk_size: usize,
}

impl Iterator for MemoryRangeIter {
    type Item = LadduDataResult<EventBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.end {
            return None;
        }

        let next = (self.pos + self.chunk_size).min(self.end);
        let batch = self.batch.slice(self.pos, next);
        self.pos = next;

        Some(Ok(batch))
    }
}

/// Event sink that collects written batches in memory.
#[derive(Clone, Debug, Default)]
pub struct MemorySink {
    schema: Option<Arc<Schema>>,
    batches: Vec<EventBatch>,
    state: SinkState,
}

impl MemorySink {
    /// Creates an empty sink.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the schema supplied to the most recent write.
    pub fn schema(&self) -> Option<&Arc<Schema>> {
        self.schema.as_ref()
    }

    /// Returns collected batches.
    pub fn batches(&self) -> &[EventBatch] {
        &self.batches
    }

    /// Consumes the sink and returns collected batches.
    pub fn into_batches(self) -> Vec<EventBatch> {
        self.batches
    }

    /// Consumes the sink and builds an in-memory source.
    ///
    /// # Errors
    ///
    /// Returns [`LadduDataError`] when the sink contains no batches or
    /// incompatible schemas.
    pub fn into_source(self) -> LadduDataResult<MemorySource> {
        MemorySource::from_batches(self.batches)
    }

    /// Consumes the sink and concatenates collected batches.
    ///
    /// # Errors
    ///
    /// Returns [`LadduDataError`] when no batches were collected or their
    /// schemas are incompatible.
    pub fn into_batch(self) -> LadduDataResult<EventBatch> {
        EventBatch::concat(&self.batches)
    }

    /// Clears schema, batches, and completion state.
    pub fn clear(&mut self) {
        self.schema = None;
        self.batches.clear();
        self.state = SinkState::Idle;
    }
}

impl EventSink for MemorySink {
    fn retains_batches(&self) -> bool {
        true
    }

    fn begin(&mut self, schema: Arc<Schema>, _plan: WritePlan) -> LadduDataResult<()> {
        match self.state {
            SinkState::Idle => {}
            SinkState::Writing => {
                return Err(LadduDataError::Sink(
                    "memory sink already initialized".into(),
                ));
            }
            SinkState::Failed => {
                return Err(LadduDataError::Sink(
                    "memory sink requires abort after failure".into(),
                ));
            }
        }

        self.schema = Some(schema);
        self.batches.clear();
        self.state = SinkState::Writing;
        Ok(())
    }

    fn write_batch(&mut self, batch: &EventBatch) -> LadduDataResult<()> {
        if !matches!(self.state, SinkState::Writing) {
            return Err(LadduDataError::Sink(
                match self.state {
                    SinkState::Idle => "memory sink not initialized",
                    SinkState::Failed => "memory sink requires abort after failure",
                    SinkState::Writing => unreachable!(),
                }
                .into(),
            ));
        }

        let schema = self
            .schema
            .as_ref()
            .ok_or_else(|| LadduDataError::Sink("memory sink not initialized".into()))?;

        if schema.as_ref() != batch.schema().as_ref() {
            return Err(LadduDataError::Sink(
                "batch schema does not match memory sink schema".into(),
            ));
        }

        self.batches.push(batch.clone());
        Ok(())
    }

    fn finish(&mut self) -> LadduDataResult<()> {
        if matches!(self.state, SinkState::Failed) {
            return Err(LadduDataError::Sink(
                "memory sink requires abort after failure".into(),
            ));
        }
        self.state = SinkState::Idle;
        Ok(())
    }

    fn abort(&mut self) -> LadduDataResult<()> {
        self.schema = None;
        self.batches.clear();
        self.state = SinkState::Idle;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use laddu_physics::vectors::RealVec4;

    use super::*;
    use crate::data::EventBatchBuilder;

    fn v(x: f64) -> RealVec4 {
        RealVec4 {
            e: x,
            px: x,
            py: x,
            pz: x,
        }
    }

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(["p"], ["id"], true).unwrap())
    }

    fn batch(start: usize, len: usize) -> EventBatch {
        let schema = schema();
        let mut builder = EventBatchBuilder::with_capacity(schema, len);

        for i in start..start + len {
            builder
                .push_weighted([v(i as f64)], [i as f64], 1.0 + i as f64)
                .unwrap();
        }

        builder.finish().unwrap()
    }

    fn concat_scalars(batches: Vec<EventBatch>) -> Vec<f64> {
        EventBatch::concat(&batches)
            .unwrap()
            .scalar_column(0)
            .to_vec()
    }

    #[test]
    fn memory_source_reports_exact_capabilities_counts_and_weighted_total() {
        let source = MemorySource::from_batches(vec![batch(0, 2), batch(2, 3)]).unwrap();

        let caps = source.capabilities();

        assert!(caps.exact_len);
        assert!(caps.exact_weighted_total);
        assert!(caps.random_access);
        assert!(caps.deterministic_partitioning);
        assert!(!caps.streaming);

        assert_eq!(source.num_events().unwrap(), Some(5));
        assert_eq!(
            source.weighted_total().unwrap(),
            Some(1.0 + 2.0 + 3.0 + 4.0 + 5.0)
        );
    }

    #[test]
    fn memory_source_coalesces_by_default_but_respects_chunked_read_plan() {
        let source = MemorySource::from_batches(vec![batch(0, 2), batch(2, 3)]).unwrap();

        let default_batches: Vec<EventBatch> = source
            .batches(ReadPlan::default())
            .unwrap()
            .map(Result::unwrap)
            .collect();

        assert_eq!(default_batches.len(), 1);
        assert_eq!(
            default_batches[0].scalar_column(0),
            &[0.0, 1.0, 2.0, 3.0, 4.0]
        );

        let chunked_batches: Vec<EventBatch> = source
            .batches(ReadPlan {
                chunk_size: Some(2),
                #[cfg(feature = "mpi")]
                distribution: Default::default(),
            })
            .unwrap()
            .map(Result::unwrap)
            .collect();

        assert_eq!(
            chunked_batches
                .iter()
                .map(EventBatch::len)
                .collect::<Vec<_>>(),
            vec![2, 2, 1]
        );
        assert_eq!(
            concat_scalars(chunked_batches),
            vec![0.0, 1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn memory_source_rejects_empty_or_schema_mismatched_batches() {
        assert!(matches!(
            MemorySource::from_batches(vec![]),
            Err(LadduDataError::InvalidArgument(_))
        ));

        let first = batch(0, 1);

        let other_schema = Arc::new(Schema::new(["q"], ["id"], true).unwrap());
        let mut builder = EventBatchBuilder::new(other_schema);
        builder.push_weighted([v(10.0)], [10.0], 1.0).unwrap();
        let second = builder.finish().unwrap();

        assert!(matches!(
            MemorySource::from_batches(vec![first, second]),
            Err(LadduDataError::Schema(_))
        ));
    }

    #[test]
    fn memory_sink_validates_lifecycle_schema_and_can_be_reused() {
        let mut sink = MemorySink::new();
        let first = batch(0, 2);

        assert!(matches!(
            sink.write_batch(&first),
            Err(LadduDataError::Sink(_))
        ));

        sink.begin(Arc::clone(first.schema()), WritePlan::default())
            .unwrap();
        sink.write_batch(&first).unwrap();
        sink.finish().unwrap();

        assert_eq!(sink.batches().len(), 1);
        assert_eq!(sink.batches()[0].scalar_column(0), &[0.0, 1.0]);

        let mismatched_schema = Arc::new(Schema::new(["other"], ["id"], true).unwrap());
        let mut builder = EventBatchBuilder::new(mismatched_schema);
        builder.push_weighted([v(9.0)], [9.0], 9.0).unwrap();
        let mismatched = builder.finish().unwrap();

        assert!(matches!(
            sink.write_batch(&mismatched),
            Err(LadduDataError::Sink(_))
        ));

        sink.clear();
        assert!(sink.schema().is_none());
        assert!(sink.batches().is_empty());
    }

    #[test]
    fn memory_sink_into_source_roundtrips_captured_batches() {
        let mut sink = MemorySink::new();
        let first = batch(0, 2);
        let second = batch(2, 2);

        sink.begin(Arc::clone(first.schema()), WritePlan::default())
            .unwrap();
        sink.write_batch(&first).unwrap();
        sink.write_batch(&second).unwrap();
        sink.finish().unwrap();

        let source = sink.into_source().unwrap();
        let merged = source.into_batch().unwrap();

        assert_eq!(merged.scalar_column(0), &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(merged.weights_column().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn memory_sink_abort_discards_partial_batches_and_allows_reuse() {
        let mut sink = MemorySink::new();
        let first = batch(0, 2);

        sink.begin(Arc::clone(first.schema()), WritePlan::default())
            .unwrap();
        sink.write_batch(&first).unwrap();
        sink.abort().unwrap();
        assert!(sink.schema().is_none());
        assert!(sink.batches().is_empty());

        sink.begin(Arc::clone(first.schema()), WritePlan::default())
            .unwrap();
        sink.finish().unwrap();
    }
}
