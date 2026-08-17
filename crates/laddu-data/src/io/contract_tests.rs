use std::{
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use laddu_physics::vectors::RealVec4;

use super::{
    EventSink, EventSource, ReadPlan, WritePlan,
    memory::MemorySink,
    parquet::{ParquetSink, ParquetSource},
    root::{RootSink, RootSource},
};
use crate::{
    LadduDataError, LadduDataResult,
    data::{Dataset, EventBatch, EventBatchBuilder},
    schema::Schema,
};

static NEXT_TEMP_FILE_ID: AtomicU64 = AtomicU64::new(0);

struct TempFile(PathBuf);

impl TempFile {
    fn new(extension: &str) -> Self {
        let id = NEXT_TEMP_FILE_ID.fetch_add(1, Ordering::Relaxed);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock is after the Unix epoch")
            .as_nanos();
        Self(std::env::temp_dir().join(format!(
            "laddu-data-contract-{}-{nanos}-{id}.{extension}",
            std::process::id()
        )))
    }

    fn path(&self) -> &Path {
        &self.0
    }

    fn pattern(&self) -> &str {
        self.0.to_str().expect("temporary path is UTF-8")
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(["p"], ["id"], true).unwrap())
}

fn vector(value: f64) -> RealVec4 {
    RealVec4 {
        e: value + 0.3,
        px: value,
        py: value + 0.1,
        pz: value + 0.2,
    }
}

fn batch(schema: Arc<Schema>, start: usize, len: usize) -> EventBatch {
    let mut builder = EventBatchBuilder::with_capacity(schema, len);
    for id in start..start + len {
        builder
            .push_weighted([vector(id as f64)], [id as f64], 10.0 + id as f64)
            .unwrap();
    }
    builder.finish().unwrap()
}

fn fixture_batches() -> Vec<EventBatch> {
    let schema = schema();
    vec![
        batch(Arc::clone(&schema), 0, 2),
        batch(Arc::clone(&schema), 2, 3),
    ]
}

fn mismatched_batch() -> EventBatch {
    let schema = Arc::new(Schema::new(["other"], ["id"], true).unwrap());
    batch(schema, 0, 1)
}

fn write_batches(sink: &mut impl EventSink) {
    let batches = fixture_batches();
    sink.begin(Arc::clone(batches[0].schema()), WritePlan::default())
        .unwrap();
    for batch in &batches {
        sink.write_batch(batch).unwrap();
    }
    sink.finish().unwrap();
}

fn assert_sink_lifecycle(mut sink: impl EventSink) {
    let batches = fixture_batches();

    assert!(matches!(
        sink.write_batch(&batches[0]),
        Err(LadduDataError::Sink(_))
    ));

    sink.begin(Arc::clone(batches[0].schema()), WritePlan::default())
        .unwrap();
    assert!(matches!(
        sink.begin(Arc::clone(batches[0].schema()), WritePlan::default()),
        Err(LadduDataError::Sink(_))
    ));
    assert!(matches!(
        sink.write_batch(&mismatched_batch()),
        Err(LadduDataError::Sink(_))
    ));
    sink.write_batch(&batches[0]).unwrap();
    sink.finish().unwrap();
    assert!(matches!(
        sink.write_batch(&batches[0]),
        Err(LadduDataError::Sink(_))
    ));
    sink.finish().unwrap();

    sink.begin(Arc::clone(batches[0].schema()), WritePlan::default())
        .unwrap();
    sink.write_batch(&batches[0]).unwrap();
    sink.abort().unwrap();
    sink.begin(Arc::clone(batches[0].schema()), WritePlan::default())
        .unwrap();
    sink.finish().unwrap();
}

fn read_batches(source: &impl EventSource, chunk_size: Option<usize>) -> Vec<EventBatch> {
    source
        .batches(ReadPlan {
            chunk_size,
            #[cfg(feature = "mpi")]
            distribution: Default::default(),
        })
        .unwrap()
        .collect::<LadduDataResult<Vec<_>>>()
        .unwrap()
}

fn rows(batches: &[EventBatch]) -> Vec<(f64, f64, RealVec4)> {
    batches
        .iter()
        .flat_map(|batch| {
            (0..batch.len()).map(|row| {
                (
                    batch.scalar_at(0, row),
                    batch.weights_at(row),
                    batch.p4_at(0, row),
                )
            })
        })
        .collect()
}

fn expected_rows() -> Vec<(f64, f64, RealVec4)> {
    (0..5)
        .map(|id| (id as f64, 10.0 + id as f64, vector(id as f64)))
        .collect()
}

fn assert_source_and_round_trip_contract(source: impl EventSource + 'static) {
    assert_eq!(source.schema().unwrap().as_ref(), schema().as_ref());
    assert_eq!(source.num_events().unwrap(), Some(5));

    let native = read_batches(&source, None);
    assert_eq!(rows(&native), expected_rows());

    let chunked = read_batches(&source, Some(2));
    assert_eq!(
        chunked.iter().map(EventBatch::len).collect::<Vec<_>>(),
        [2, 2, 1]
    );
    assert_eq!(rows(&chunked), expected_rows());

    let repeated = read_batches(&source, Some(3));
    assert!(repeated.iter().all(|batch| batch.len() <= 3));
    assert_eq!(rows(&repeated), expected_rows());

    let zero_chunk_result = source
        .batches(ReadPlan {
            chunk_size: Some(0),
            #[cfg(feature = "mpi")]
            distribution: Default::default(),
        })
        .and_then(|mut batches| batches.next().transpose().map(|_| ()));
    assert!(matches!(
        zero_chunk_result,
        Err(LadduDataError::InvalidArgument(_))
    ));

    assert_dataset_traversal_contract(Dataset::new(source));
}

fn assert_dataset_traversal_contract(dataset: Dataset) {
    let dataset = dataset
        .chunked(2)
        .unwrap()
        .filter(|event| event.scalar(0) % 2.0 == 0.0);
    let expected = vec![(0.0, 10.0), (2.0, 12.0), (4.0, 14.0)];

    let from_batches = dataset
        .batches()
        .unwrap()
        .collect::<LadduDataResult<Vec<_>>>()
        .unwrap();
    assert_eq!(
        from_batches.iter().map(EventBatch::len).collect::<Vec<_>>(),
        [2, 1]
    );
    assert_eq!(
        from_batches
            .iter()
            .flat_map(|batch| {
                (0..batch.len()).map(|row| (batch.scalar_at(0, row), batch.weights_at(row)))
            })
            .collect::<Vec<_>>(),
        expected
    );

    assert_eq!(
        dataset
            .map_events(|event| (event.scalar(0), event.weight()))
            .unwrap(),
        expected
    );
    assert_eq!(
        dataset
            .fold_events(Vec::new(), |mut values, event| {
                values.push((event.scalar(0), event.weight()));
                values
            })
            .unwrap(),
        expected
    );
    assert_eq!(
        dataset
            .accumulate_events(Vec::new(), |values, event| {
                values.push((event.scalar(0), event.weight()));
            })
            .unwrap(),
        expected
    );
    assert_eq!(
        dataset.weighted_sum(|event| event.scalar(0)).unwrap(),
        2.0 * 12.0 + 4.0 * 14.0
    );
}

#[test]
fn memory_backend_obeys_shared_data_contracts() {
    assert_sink_lifecycle(MemorySink::new());

    let mut sink = MemorySink::new();
    write_batches(&mut sink);
    assert_source_and_round_trip_contract(sink.into_source().unwrap());
}

#[test]
fn parquet_backend_obeys_shared_data_contracts() {
    let lifecycle_file = TempFile::new("parquet");
    assert_sink_lifecycle(ParquetSink::create(lifecycle_file.path()));

    let round_trip_file = TempFile::new("parquet");
    let mut sink = ParquetSink::create(round_trip_file.path());
    write_batches(&mut sink);
    let source = ParquetSource::open(round_trip_file.pattern()).unwrap();
    assert_source_and_round_trip_contract(source);
}

#[test]
fn root_backend_obeys_shared_data_contracts() {
    let lifecycle_file = TempFile::new("root");
    assert_sink_lifecycle(
        RootSink::builder(lifecycle_file.path())
            .tree("events")
            .build(),
    );

    let round_trip_file = TempFile::new("root");
    let mut sink = RootSink::builder(round_trip_file.path())
        .tree("events")
        .build();
    write_batches(&mut sink);
    let source = RootSource::builder(round_trip_file.pattern())
        .tree("events")
        .build()
        .unwrap();
    assert_source_and_round_trip_contract(source);
}
