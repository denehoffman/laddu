use std::{
    fs::File,
    path::{Path, PathBuf},
    sync::Arc,
};

use arrow::datatypes::SchemaRef;
use parquet::{
    arrow::{ArrowWriter, arrow_reader::ParquetRecordBatchReaderBuilder},
    file::properties::WriterProperties,
};

use crate::{
    LadduDataError, LadduDataResult,
    data::EventBatch,
    io::{
        DataFragment, EventSink, EventSource, FragmentedSource, OutputMode, OutputPath, ReadPlan,
        SinkState, SourceBuild, SourceBuildOptions, SourceCapabilities, WritePlan, build_source,
        fragmented_batches, sink_error, source_error,
    },
    schema::{
        Precision, Schema, SchemaColumnNames, SchemaInferenceOptions, SchemaWriteOptions,
        WriteWeightColumn,
    },
};

mod decode;
mod encode;

#[cfg(test)]
use decode::record_batch_to_event_batch;
use decode::{arrow_columns, open_parquet_fragment_reader, parquet_arrow_schema};
use encode::{arrow_schema_from_event_schema, event_batch_to_record_batch};

/// Event source backed by one or more Parquet files.
#[derive(Clone, Debug)]
pub struct ParquetSource {
    files: Arc<[Arc<PathBuf>]>,
    schema: Arc<Schema>,
    options: ParquetReadOptions,
}

/// Schema inference, validation, null, and glob options for Parquet reads.
#[derive(Clone, Debug)]
pub struct ParquetReadOptions {
    /// Infer a logical schema when none is supplied.
    pub infer_schema: bool,
    /// Validate required columns in every matched file.
    pub validate_all_files: bool,
    /// Policy for null floating-point cells.
    pub null_handling: NullHandling,
    /// Sort glob results for deterministic global row order.
    pub sort_glob: bool,
    /// Logical schema inference options.
    pub schema_inference: SchemaInferenceOptions,
}

impl Default for ParquetReadOptions {
    fn default() -> Self {
        Self {
            infer_schema: true,
            validate_all_files: true,
            null_handling: NullHandling::Error,
            sort_glob: true,
            schema_inference: SchemaInferenceOptions::default(),
        }
    }
}

/// Policy for null floating-point cells in Parquet input.
#[derive(Clone, Copy, Debug)]
pub enum NullHandling {
    /// Return an error on the first null.
    Error,
    /// Convert nulls to NaN.
    NaN,
}

/// Key identifying one Parquet row group.
#[derive(Clone, Debug)]
pub struct ParquetFragmentKey {
    /// Input file path.
    pub file: Arc<PathBuf>,
    /// Zero-based row-group index.
    pub row_group: usize,
}

impl ParquetSource {
    /// Opens files matching a glob with default options.
    ///
    /// # Errors
    ///
    /// Returns [`LadduDataError`] when the glob is invalid or empty, a file
    /// cannot be read, or schemas are incompatible.
    pub fn open(pattern: impl AsRef<str>) -> LadduDataResult<Self> {
        Self::builder(pattern).build()
    }

    /// Creates a configurable source builder for a file glob.
    pub fn builder(pattern: impl AsRef<str>) -> ParquetSourceBuilder {
        ParquetSourceBuilder {
            pattern: pattern.as_ref().to_owned(),
            schema: None,
            options: ParquetReadOptions::default(),
        }
    }

    /// Returns matched files in global row order.
    pub fn files(&self) -> &[Arc<PathBuf>] {
        &self.files
    }
}

/// Builder for a [`ParquetSource`].
pub struct ParquetSourceBuilder {
    pattern: String,
    schema: Option<Arc<Schema>>,
    options: ParquetReadOptions,
}

impl ParquetSourceBuilder {
    /// Supplies an explicit logical schema and disables inference.
    pub fn schema(mut self, schema: Arc<Schema>) -> Self {
        self.schema = Some(schema);
        self.options.infer_schema = false;
        self
    }

    /// Enables or disables logical schema inference.
    pub fn infer_schema(mut self, value: bool) -> Self {
        self.options.infer_schema = value;
        self
    }

    /// Requires a physical weight column during inference.
    pub fn require_weight(mut self, value: bool) -> Self {
        self.options.schema_inference.require_weight = value;
        self
    }

    /// Chooses whether every matched file is schema-validated eagerly.
    pub fn validate_all_files(mut self, value: bool) -> Self {
        self.options.validate_all_files = value;
        self
    }

    /// Converts null floating-point cells to NaN.
    pub fn nulls_as_nan(mut self) -> Self {
        self.options.null_handling = NullHandling::NaN;
        self
    }

    /// Returns an error on null floating-point cells.
    pub fn error_on_nulls(mut self) -> Self {
        self.options.null_handling = NullHandling::Error;
        self
    }

    /// Chooses whether matched paths are sorted.
    pub fn sort_glob(mut self, value: bool) -> Self {
        self.options.sort_glob = value;
        self
    }

    /// Replaces logical schema inference options.
    pub fn schema_inference(mut self, options: SchemaInferenceOptions) -> Self {
        self.options.schema_inference = options;
        self
    }

    /// Resolves files, validates schema, and builds the source.
    ///
    /// # Errors
    ///
    /// Returns [`LadduDataError`] when the glob is invalid or empty, Parquet
    /// metadata cannot be read, schema inference fails, or files disagree.
    pub fn build(self) -> LadduDataResult<ParquetSource> {
        let ParquetSourceBuilder {
            pattern,
            schema: explicit_schema,
            options,
        } = self;
        let infer_options = options.schema_inference.clone();
        let validate_options = options.schema_inference.clone();
        let SourceBuild { files, schema, .. } = build_source(
            SourceBuildOptions {
                pattern: &pattern,
                sort: options.sort_glob,
                format: "parquet",
                explicit_schema,
                infer_schema: options.infer_schema,
                validate_all_files: options.validate_all_files,
            },
            |_| Ok(()),
            move |path, _| {
                let arrow_schema = parquet_arrow_schema(path)?;
                Schema::infer_from_columns(arrow_columns(&arrow_schema), &infer_options)
            },
            move |path, schema, _| {
                let arrow_schema = parquet_arrow_schema(path)?;
                schema.validate_required_columns(arrow_columns(&arrow_schema), &validate_options)
            },
        )?;

        Ok(ParquetSource {
            files,
            schema,
            options,
        })
    }
}

impl EventSource for ParquetSource {
    fn schema(&self) -> LadduDataResult<Arc<Schema>> {
        Ok(Arc::clone(&self.schema))
    }

    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            exact_len: true,
            exact_weighted_total: false,
            random_access: false,
            deterministic_partitioning: true,
            predicate_pushdown: false,
            projection_pushdown: true,
            streaming: true,
        }
    }

    fn num_events(&self) -> LadduDataResult<Option<u64>> {
        Ok(Some(self.fragments()?.iter().map(|f| f.rows).sum()))
    }

    fn batches(
        &self,
        plan: ReadPlan,
    ) -> LadduDataResult<Box<dyn Iterator<Item = LadduDataResult<EventBatch>> + Send>> {
        fragmented_batches(Arc::new(self.clone()), plan)
    }
}

impl FragmentedSource for ParquetSource {
    type Key = ParquetFragmentKey;

    fn fragments(&self) -> LadduDataResult<Vec<DataFragment<Self::Key>>> {
        let mut fragments = Vec::new();
        let mut global_start = 0_u64;

        for path in self.files.iter() {
            let resource = path.as_ref().display().to_string();
            let file = File::open(path.as_ref())
                .map_err(|error| source_error("open Parquet file", &resource, error))?;

            let builder = ParquetRecordBatchReaderBuilder::try_new(file)
                .map_err(|error| source_error("read Parquet metadata", &resource, error))?;

            let metadata = builder.metadata();

            for row_group in 0..metadata.num_row_groups() {
                let rows = metadata.row_group(row_group).num_rows() as u64;

                fragments.push(DataFragment {
                    key: ParquetFragmentKey {
                        file: Arc::clone(path),
                        row_group,
                    },
                    global_start,
                    rows,
                });

                global_start += rows;
            }
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
        open_parquet_fragment_reader(
            Arc::clone(&self.schema),
            self.options.clone(),
            key.clone(),
            local_start,
            local_len,
            chunk_size,
        )
    }
}

/// Event sink that writes Arrow record batches to Parquet.
pub struct ParquetSink {
    output: OutputPath,
    writer: Option<ArrowWriter<File>>,
    arrow_schema: Option<SchemaRef>,
    event_schema: Option<Arc<Schema>>,
    options: ParquetWriteOptions,
    resolved_path: Option<PathBuf>,
    state: SinkState,
}

/// Parquet writer and physical schema options.
#[derive(Clone, Debug, Default)]
pub struct ParquetWriteOptions {
    /// Optional low-level Parquet writer properties.
    pub writer_properties: Option<WriterProperties>,
    /// Physical schema write options.
    pub schema_write: SchemaWriteOptions,
}

impl ParquetSink {
    /// Creates a sink with default options.
    pub fn create(path: impl Into<PathBuf>) -> Self {
        Self::builder(path).build()
    }

    /// Creates a configurable sink builder.
    pub fn builder(path: impl Into<PathBuf>) -> ParquetSinkBuilder {
        ParquetSinkBuilder {
            output: OutputPath::new(path),
            options: ParquetWriteOptions::default(),
        }
    }

    /// Returns the concrete path after writing has begun.
    pub fn resolved_path(&self) -> Option<&Path> {
        self.resolved_path.as_deref()
    }
}

/// Builder for a [`ParquetSink`].
pub struct ParquetSinkBuilder {
    output: OutputPath,
    options: ParquetWriteOptions,
}

impl ParquetSinkBuilder {
    /// Sets the output path mode.
    pub fn output_mode(mut self, mode: OutputMode) -> Self {
        self.output = self.output.with_mode(mode);
        self
    }

    /// Selects single-file output.
    pub fn single_file(self) -> Self {
        self.output_mode(OutputMode::SingleFile)
    }

    /// Selects one output file per rank.
    pub fn per_rank_files(self) -> Self {
        self.output_mode(OutputMode::PerRankFiles)
    }

    /// Selects output mode from the write plan.
    pub fn auto_output(self) -> Self {
        self.output_mode(OutputMode::Auto)
    }

    /// Replaces physical schema write options.
    pub fn schema_write(mut self, options: SchemaWriteOptions) -> Self {
        self.options.schema_write = options;
        self
    }

    /// Sets physical column naming conventions.
    pub fn column_names(mut self, column_names: SchemaColumnNames) -> Self {
        self.options.schema_write.column_names = column_names;
        self
    }

    /// Sets floating-point output precision.
    pub fn precision(mut self, precision: Precision) -> Self {
        self.options.schema_write.precision = precision;
        self
    }

    /// Sets low-level Parquet writer properties.
    pub fn writer_properties(mut self, props: WriterProperties) -> Self {
        self.options.writer_properties = Some(props);
        self
    }

    /// Sets the weight-column emission policy.
    pub fn write_weight_column(mut self, value: WriteWeightColumn) -> Self {
        self.options.schema_write.write_weight_column = value;
        self
    }

    /// Builds the sink.
    pub fn build(self) -> ParquetSink {
        ParquetSink {
            output: self.output,
            writer: None,
            arrow_schema: None,
            event_schema: None,
            options: self.options,
            resolved_path: None,
            state: SinkState::Idle,
        }
    }
}

impl EventSink for ParquetSink {
    fn begin(&mut self, schema: Arc<Schema>, plan: WritePlan) -> LadduDataResult<()> {
        match self.state {
            SinkState::Idle => {}
            SinkState::Writing => {
                return Err(LadduDataError::Sink(
                    "parquet sink already initialized".into(),
                ));
            }
            SinkState::Failed => {
                return Err(LadduDataError::Sink(
                    "parquet sink requires abort after failure".into(),
                ));
            }
        }

        let path = self.output.resolve(plan, "parquet")?;
        OutputPath::create_parent_dirs(&path)?;

        let arrow_schema = Arc::new(arrow_schema_from_event_schema(
            &schema,
            self.options.schema_write.write_weight_column,
            &self.options.schema_write,
        ));

        let file = File::create(&path)
            .map_err(|error| sink_error("create Parquet file", path.display(), error))?;

        let writer = ArrowWriter::try_new(
            file,
            Arc::clone(&arrow_schema),
            self.options.writer_properties.clone(),
        )
        .map_err(|error| sink_error("initialize Parquet writer", path.display(), error))?;

        self.arrow_schema = Some(arrow_schema);
        self.event_schema = Some(schema);
        self.writer = Some(writer);
        self.resolved_path = Some(path);
        self.state = SinkState::Writing;

        Ok(())
    }

    fn write_batch(&mut self, batch: &EventBatch) -> LadduDataResult<()> {
        if !matches!(self.state, SinkState::Writing) {
            return Err(LadduDataError::Sink(
                match self.state {
                    SinkState::Idle => "parquet sink not initialized",
                    SinkState::Failed => "parquet sink requires abort after failure",
                    SinkState::Writing => unreachable!(),
                }
                .into(),
            ));
        }

        let arrow_schema = self
            .arrow_schema
            .as_ref()
            .ok_or_else(|| LadduDataError::Sink("parquet sink not initialized".into()))?;

        let event_schema = self
            .event_schema
            .as_ref()
            .ok_or_else(|| LadduDataError::Sink("parquet sink not initialized".into()))?;
        if event_schema.as_ref() != batch.schema().as_ref() {
            return Err(LadduDataError::Sink(
                "batch schema does not match parquet sink schema".into(),
            ));
        }

        let rb = match event_batch_to_record_batch(
            batch,
            Arc::clone(arrow_schema),
            self.options.schema_write.write_weight_column,
            &self.options.schema_write,
            self.options.schema_write.precision,
        ) {
            Ok(rb) => rb,
            Err(error) => {
                self.state = SinkState::Failed;
                let resource = self.resolved_path.as_deref().map_or_else(
                    || "Parquet sink".to_owned(),
                    |path| path.display().to_string(),
                );
                return Err(sink_error("encode Parquet batch", resource, error));
            }
        };

        let resource = self.resolved_path.as_deref().map_or_else(
            || "Parquet sink".to_owned(),
            |path| path.display().to_string(),
        );
        let result = self
            .writer
            .as_mut()
            .ok_or_else(|| LadduDataError::Sink("parquet sink not initialized".into()))?
            .write(&rb)
            .map_err(|error| sink_error("write Parquet batch", resource, error));
        if result.is_err() {
            self.state = SinkState::Failed;
        }
        result
    }

    fn finish(&mut self) -> LadduDataResult<()> {
        if matches!(self.state, SinkState::Idle) {
            return Ok(());
        }
        if matches!(self.state, SinkState::Failed) {
            return Err(LadduDataError::Sink(
                "parquet sink requires abort after failure".into(),
            ));
        }

        let resource = self.resolved_path.as_deref().map_or_else(
            || "Parquet sink".to_owned(),
            |path| path.display().to_string(),
        );
        if let Some(writer) = self.writer.take()
            && let Err(error) = writer.close()
        {
            self.state = SinkState::Failed;
            return Err(sink_error("finalize Parquet file", resource, error));
        }

        self.arrow_schema = None;
        self.event_schema = None;
        self.state = SinkState::Idle;
        Ok(())
    }

    fn abort(&mut self) -> LadduDataResult<()> {
        // Dropping ArrowWriter closes its file handle without attempting to
        // finalize the Parquet footer. The concrete path is intentionally
        // retained so callers can identify the potentially incomplete file.
        self.writer.take();
        self.arrow_schema = None;
        self.event_schema = None;
        self.state = SinkState::Idle;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use arrow::{
        array::{ArrayRef, Float32Array, Float64Array},
        datatypes::{DataType, Field, Schema as ArrowSchema},
        record_batch::RecordBatch,
    };
    use laddu_physics::vectors::RealVec4;

    use super::*;
    use crate::data::{Dataset, EventBatchBuilder};

    fn temp_path(ext: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();

        std::env::temp_dir().join(format!(
            "laddu-parquet-test-{}-{nanos}.{ext}",
            std::process::id()
        ))
    }

    fn v(x: f64) -> RealVec4 {
        RealVec4 {
            e: x + 0.3,
            px: x,
            py: x + 0.1,
            pz: x + 0.2,
        }
    }

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(["p"], ["mass"], true).unwrap())
    }

    fn batch() -> EventBatch {
        let schema = schema();
        let mut builder = EventBatchBuilder::new(schema);

        for i in 0..4 {
            builder
                .push_weighted([v(i as f64)], [100.0 + i as f64], 10.0 + i as f64)
                .unwrap();
        }

        builder.finish().unwrap()
    }

    #[test]
    fn parquet_sink_and_source_roundtrip_with_f32_write_and_schema_inference() {
        let path = temp_path("parquet");
        let batch = batch();

        let mut sink = ParquetSink::builder(path.clone())
            .precision(Precision::F32)
            .build();

        sink.begin(Arc::clone(batch.schema()), WritePlan::default())
            .unwrap();
        sink.write_batch(&batch).unwrap();
        sink.finish().unwrap();

        let source = ParquetSource::builder(path.to_str().unwrap())
            .infer_schema(true)
            .validate_all_files(true)
            .build()
            .unwrap();

        let inferred = source.schema().unwrap();
        assert_eq!(
            inferred
                .p4s()
                .iter()
                .map(|n| n.to_string())
                .collect::<Vec<_>>(),
            vec!["p"]
        );
        assert_eq!(
            inferred
                .scalars()
                .iter()
                .map(|n| n.to_string())
                .collect::<Vec<_>>(),
            vec!["mass"]
        );
        assert!(inferred.has_weight());

        let read_batches: Vec<EventBatch> = source
            .batches(ReadPlan {
                chunk_size: Some(2),
                #[cfg(feature = "mpi")]
                distribution: Default::default(),
            })
            .unwrap()
            .map(Result::unwrap)
            .collect();

        assert_eq!(
            read_batches.iter().map(EventBatch::len).collect::<Vec<_>>(),
            vec![2, 2]
        );

        let read = EventBatch::concat(&read_batches).unwrap();

        assert_eq!(read.scalar_column(0), &[100.0, 101.0, 102.0, 103.0]);
        assert_eq!(read.weights_column().unwrap(), &[10.0, 11.0, 12.0, 13.0]);
        assert!((read.p4_at(0, 2).e - 2.3).abs() < 1.0e-6);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn parquet_source_require_weight_fails_when_written_without_weight_column() {
        let path = temp_path("parquet");
        let schema = Arc::new(Schema::new(["p"], ["mass"], false).unwrap());

        let mut builder = EventBatchBuilder::new(Arc::clone(&schema));
        builder.push([v(1.0)], [5.0]).unwrap();
        let batch = builder.finish().unwrap();

        let mut sink = ParquetSink::builder(path.clone())
            .write_weight_column(WriteWeightColumn::OnlyIfPresent)
            .build();

        sink.begin(schema, WritePlan::default()).unwrap();
        sink.write_batch(&batch).unwrap();
        sink.finish().unwrap();

        let err = ParquetSource::builder(path.to_str().unwrap())
            .require_weight(true)
            .build()
            .unwrap_err();

        assert!(matches!(err, LadduDataError::MissingColumn(name) if name.as_ref() == "weight"));

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn record_batch_to_event_batch_handles_nulls_as_error_or_nan_and_reads_f32_as_f64() {
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("p_e", DataType::Float64, true),
            Field::new("p_px", DataType::Float32, true),
            Field::new("p_py", DataType::Float64, true),
            Field::new("p_pz", DataType::Float32, true),
            Field::new("mass", DataType::Float32, true),
            Field::new("weight", DataType::Float64, true),
        ]));

        let rb = RecordBatch::try_new(
            Arc::clone(&arrow_schema),
            vec![
                Arc::new(Float64Array::from(vec![Some(1.0), None])) as ArrayRef,
                Arc::new(Float32Array::from(vec![Some(0.1), Some(0.2)])) as ArrayRef,
                Arc::new(Float64Array::from(vec![Some(0.3), Some(0.4)])) as ArrayRef,
                Arc::new(Float32Array::from(vec![Some(0.5), Some(0.6)])) as ArrayRef,
                Arc::new(Float32Array::from(vec![Some(2.0), Some(3.0)])) as ArrayRef,
                Arc::new(Float64Array::from(vec![Some(4.0), Some(5.0)])) as ArrayRef,
            ],
        )
        .unwrap();

        let schema = schema();

        let error_options = ParquetReadOptions {
            null_handling: NullHandling::Error,
            ..ParquetReadOptions::default()
        };

        let err = record_batch_to_event_batch(rb.clone(), Arc::clone(&schema), &error_options)
            .unwrap_err();

        assert!(matches!(err, LadduDataError::Source(msg) if msg.contains("null in column p_e")));

        let nan_options = ParquetReadOptions {
            null_handling: NullHandling::NaN,
            ..ParquetReadOptions::default()
        };

        let batch = record_batch_to_event_batch(rb, schema, &nan_options).unwrap();

        assert!(batch.p4_at(0, 1).e.is_nan());
        assert!((batch.p4_at(0, 1).px - 0.2).abs() < 1.0e-6);
        assert_eq!(batch.scalar_column(0), &[2.0, 3.0]);
        assert_eq!(batch.weights_column().unwrap(), &[4.0, 5.0]);
    }

    #[test]
    fn parquet_sink_rejects_batches_with_different_schema() {
        let path = temp_path("parquet");
        let batch = batch();

        let mut sink = ParquetSink::builder(path.clone()).build();
        sink.begin(Arc::clone(batch.schema()), WritePlan::default())
            .unwrap();

        let other_schema = Arc::new(Schema::new(["q"], ["mass"], true).unwrap());
        let mut builder = EventBatchBuilder::new(other_schema);
        builder.push_weighted([v(1.0)], [1.0], 1.0).unwrap();
        let other = builder.finish().unwrap();

        let err = sink.write_batch(&other).unwrap_err();
        assert!(matches!(err, LadduDataError::Sink(msg) if msg.contains("schema")));

        sink.finish().unwrap();
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn dataset_write_to_parquet_applies_dataset_transformations_before_writing() {
        let path = temp_path("parquet");

        let dataset = Dataset::from_batch(batch()).filter(|ev| ev.scalar(0) >= 102.0);

        let mut sink = ParquetSink::builder(path.clone()).build();
        dataset.write_to(&mut sink).unwrap();

        let source = ParquetSource::open(path.to_str().unwrap()).unwrap();
        let read = EventBatch::concat(
            &source
                .batches(ReadPlan::default())
                .unwrap()
                .map(Result::unwrap)
                .collect::<Vec<_>>(),
        )
        .unwrap();

        assert_eq!(read.scalar_column(0), &[102.0, 103.0]);
        assert_eq!(read.weights_column().unwrap(), &[12.0, 13.0]);

        let _ = std::fs::remove_file(path);
    }
}
