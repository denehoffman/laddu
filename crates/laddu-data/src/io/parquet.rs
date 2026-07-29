use std::{
    fs::File,
    path::{Path, PathBuf},
    sync::Arc,
};

use arrow::{
    array::{Array, ArrayRef, Float32Array, Float64Array},
    datatypes::{DataType, Field, Schema as ArrowSchema, SchemaRef},
    record_batch::RecordBatch,
};
use laddu_physics::vectors::RealVec4;
use parquet::{
    arrow::{ArrowWriter, ProjectionMask, arrow_reader::ParquetRecordBatchReaderBuilder},
    file::properties::WriterProperties,
    schema::types::SchemaDescriptor,
};

use crate::{
    LadduDataError, LadduDataResult, Name,
    data::EventBatch,
    io::{
        DataFragment, EventSink, EventSource, FragmentedSource, OutputMode, OutputPath, ReadPlan,
        SliceBatchIter, SourceCapabilities, WritePlan, fragmented_batches,
    },
    schema::{
        ColumnInfo, ColumnType, Precision, Schema, SchemaColumnNames, SchemaInferenceOptions,
        SchemaWriteOptions, WriteWeightColumn,
    },
};

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
        let mut files: Vec<PathBuf> = glob::glob(&self.pattern)
            .map_err(|e| LadduDataError::Source(e.to_string()))?
            .collect::<std::result::Result<_, _>>()
            .map_err(|e| LadduDataError::Source(e.to_string()))?;

        if self.options.sort_glob {
            files.sort();
        }

        if files.is_empty() {
            return Err(LadduDataError::Source(
                "no parquet files matched glob".into(),
            ));
        }

        let files: Arc<[Arc<PathBuf>]> = files.into_iter().map(Arc::new).collect();

        let schema = match self.schema {
            Some(schema) => schema,
            None if self.options.infer_schema => Arc::new({
                let path: &Path = files[0].as_ref();
                let options: &ParquetReadOptions = &self.options;
                let arrow_schema = parquet_arrow_schema(path)?;
                Schema::infer_from_columns(arrow_columns(&arrow_schema), &options.schema_inference)
            }?),
            None => return Err(LadduDataError::InvalidArgument("schema required")),
        };

        if self.options.validate_all_files {
            for file in files.iter() {
                {
                    let path: &Path = file.as_ref();
                    let schema: &Schema = &schema;
                    let options: &ParquetReadOptions = &self.options;
                    let arrow_schema = parquet_arrow_schema(path)?;
                    schema.validate_required_columns(
                        arrow_columns(&arrow_schema),
                        &options.schema_inference,
                    )
                }?;
            }
        }

        Ok(ParquetSource {
            files,
            schema,
            options: self.options,
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
            let file =
                File::open(path.as_ref()).map_err(|e| LadduDataError::Source(e.to_string()))?;

            let builder = ParquetRecordBatchReaderBuilder::try_new(file)
                .map_err(|e| LadduDataError::Source(e.to_string()))?;

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

fn open_parquet_fragment_reader(
    schema: Arc<Schema>,
    options: ParquetReadOptions,
    key: ParquetFragmentKey,
    local_start: usize,
    local_len: usize,
    chunk_size: Option<usize>,
) -> LadduDataResult<Box<dyn Iterator<Item = LadduDataResult<EventBatch>> + Send>> {
    let file = File::open(key.file.as_ref()).map_err(|e| LadduDataError::Source(e.to_string()))?;

    let mut builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| LadduDataError::Source(e.to_string()))?
        .with_row_groups(vec![key.row_group]);

    let projection = parquet_projection_for_schema(
        builder.parquet_schema(),
        builder.schema().as_ref(),
        &schema,
        &options.schema_inference.column_names,
    )?;
    builder = builder.with_projection(projection);

    if matches!(chunk_size, Some(0)) {
        return Err(LadduDataError::InvalidArgument(
            "chunk_size must be nonzero",
        ));
    }
    let end = local_start
        .checked_add(local_len)
        .ok_or(LadduDataError::InvalidArgument(
            "slice range overflows usize",
        ))?;
    let batch_size = chunk_size.unwrap_or(end.max(1));
    builder = builder.with_batch_size(batch_size);

    let reader = builder
        .build()
        .map_err(|e| LadduDataError::Source(e.to_string()))?;

    let batches = reader.map(move |rb| {
        let rb = rb.map_err(|e| LadduDataError::Source(e.to_string()))?;
        record_batch_to_event_batch(rb, Arc::clone(&schema), &options)
    });

    Ok(Box::new(SliceBatchIter::new(
        batches,
        local_start,
        local_len,
    )?))
}

fn parquet_projection_for_schema(
    parquet_schema: &SchemaDescriptor,
    arrow_schema: &ArrowSchema,
    schema: &Schema,
    column_names: &SchemaColumnNames,
) -> LadduDataResult<ProjectionMask> {
    let mut indices = Vec::new();

    for name in schema.physical_columns(column_names) {
        let index = arrow_schema
            .index_of(name.as_ref())
            .map_err(|_| LadduDataError::MissingColumn(name))?;

        indices.push(index);
    }

    indices.sort_unstable();
    indices.dedup();

    Ok(ProjectionMask::roots(parquet_schema, indices))
}

fn parquet_arrow_schema(path: &Path) -> LadduDataResult<SchemaRef> {
    let file = File::open(path).map_err(|e| LadduDataError::Source(e.to_string()))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| LadduDataError::Source(e.to_string()))?;

    Ok(builder.schema().clone())
}

fn arrow_columns(schema: &ArrowSchema) -> impl Iterator<Item = ColumnInfo<'_>> {
    schema.fields().iter().map(|field| ColumnInfo {
        name: field.name().as_str(),
        dtype: arrow_column_type(field.data_type()),
    })
}

fn arrow_column_type(data_type: &DataType) -> ColumnType {
    match data_type {
        DataType::Float64 => ColumnType::F64,
        DataType::Float32 => ColumnType::F32,
        _ => ColumnType::Other,
    }
}

fn record_batch_to_event_batch(
    rb: RecordBatch,
    schema: Arc<Schema>,
    options: &ParquetReadOptions,
) -> LadduDataResult<EventBatch> {
    let arrow_schema = rb.schema();

    let mut p4s = Vec::with_capacity(schema.n_p4s());
    let mut scalars = Vec::with_capacity(schema.n_scalars());

    for name in schema.p4s() {
        let [e_name, px_name, py_name, pz_name] = options
            .schema_inference
            .column_names
            .p4_suffixes
            .physical_p4_names(name);

        let e = read_f64_column(&rb, &arrow_schema, &e_name, options)?;
        let px = read_f64_column(&rb, &arrow_schema, &px_name, options)?;
        let py = read_f64_column(&rb, &arrow_schema, &py_name, options)?;
        let pz = read_f64_column(&rb, &arrow_schema, &pz_name, options)?;

        let col: Arc<[RealVec4]> = (0..rb.num_rows())
            .map(|i| RealVec4 {
                e: e[i],
                px: px[i],
                py: py[i],
                pz: pz[i],
            })
            .collect();

        p4s.push(col);
    }

    for name in schema.scalars() {
        scalars.push(read_f64_column(&rb, &arrow_schema, name, options)?.into());
    }

    let weights = if schema.has_weight() {
        Some(
            read_f64_column(
                &rb,
                &arrow_schema,
                &options.schema_inference.column_names.weight_column,
                options,
            )?
            .into(),
        )
    } else {
        None
    };

    EventBatch::new(schema, p4s, scalars, weights)
}

fn read_f64_column(
    rb: &RecordBatch,
    arrow_schema: &SchemaRef,
    name: &str,
    options: &ParquetReadOptions,
) -> LadduDataResult<Vec<f64>> {
    let index = arrow_schema
        .index_of(name)
        .map_err(|_| LadduDataError::MissingColumn(Name::from(name)))?;

    let array = rb.column(index);

    match array.data_type() {
        DataType::Float64 => {
            let array = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| LadduDataError::Source(format!("failed to read {name} as f64")))?;

            collect_f64(array, name, options.null_handling)
        }

        DataType::Float32 => {
            let array = array
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| LadduDataError::Source(format!("failed to read {name} as f32")))?;

            collect_f32(array, name, options.null_handling)
        }

        other => Err(LadduDataError::Source(format!(
            "column {name} has unsupported type {other:?}"
        ))),
    }
}

fn collect_f64(array: &Float64Array, name: &str, nulls: NullHandling) -> LadduDataResult<Vec<f64>> {
    let mut out = Vec::with_capacity(array.len());

    for i in 0..array.len() {
        if array.is_null(i) {
            match nulls {
                NullHandling::Error => {
                    return Err(LadduDataError::Source(format!("null in column {name}")));
                }
                NullHandling::NaN => out.push(f64::NAN),
            }
        } else {
            out.push(array.value(i));
        }
    }

    Ok(out)
}

fn collect_f32(array: &Float32Array, name: &str, nulls: NullHandling) -> LadduDataResult<Vec<f64>> {
    let mut out = Vec::with_capacity(array.len());

    for i in 0..array.len() {
        if array.is_null(i) {
            match nulls {
                NullHandling::Error => {
                    return Err(LadduDataError::Source(format!("null in column {name}")));
                }
                NullHandling::NaN => out.push(f64::NAN),
            }
        } else {
            out.push(array.value(i) as f64);
        }
    }

    Ok(out)
}

/// Event sink that writes Arrow record batches to Parquet.
pub struct ParquetSink {
    output: OutputPath,
    writer: Option<ArrowWriter<File>>,
    arrow_schema: Option<SchemaRef>,
    event_schema: Option<Arc<Schema>>,
    options: ParquetWriteOptions,
    resolved_path: Option<PathBuf>,
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
        }
    }
}

impl EventSink for ParquetSink {
    fn begin(&mut self, schema: Arc<Schema>, plan: WritePlan) -> LadduDataResult<()> {
        let path = self.output.resolve(plan, "parquet")?;
        OutputPath::create_parent_dirs(&path)?;

        let arrow_schema = Arc::new(arrow_schema_from_event_schema(
            &schema,
            self.options.schema_write.write_weight_column,
            &self.options.schema_write,
        ));

        let file = File::create(&path).map_err(|e| LadduDataError::Sink(e.to_string()))?;

        let writer = ArrowWriter::try_new(
            file,
            Arc::clone(&arrow_schema),
            self.options.writer_properties.clone(),
        )
        .map_err(|e| LadduDataError::Sink(e.to_string()))?;

        self.arrow_schema = Some(arrow_schema);
        self.event_schema = Some(schema);
        self.writer = Some(writer);
        self.resolved_path = Some(path);

        Ok(())
    }

    fn write_batch(&mut self, batch: &EventBatch) -> LadduDataResult<()> {
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

        let rb = event_batch_to_record_batch(
            batch,
            Arc::clone(arrow_schema),
            self.options.schema_write.write_weight_column,
            self.options.schema_write.precision,
        )?;

        self.writer
            .as_mut()
            .ok_or_else(|| LadduDataError::Sink("parquet sink not initialized".into()))?
            .write(&rb)
            .map_err(|e| LadduDataError::Sink(e.to_string()))
    }

    fn finish(&mut self) -> LadduDataResult<()> {
        if let Some(writer) = self.writer.take() {
            writer
                .close()
                .map_err(|e| LadduDataError::Sink(e.to_string()))?;
        }

        Ok(())
    }
}

fn arrow_schema_from_event_schema(
    schema: &Schema,
    write_weight: WriteWeightColumn,
    options: &SchemaWriteOptions,
) -> ArrowSchema {
    let should_write_weight =
        matches!(write_weight, WriteWeightColumn::Always) || schema.has_weight();

    let mut fields = Vec::with_capacity(
        4 * schema.n_p4s() + schema.n_scalars() + usize::from(should_write_weight),
    );

    let data_type = match options.precision {
        Precision::F64 => DataType::Float64,
        Precision::F32 => DataType::Float32,
    };

    for name in schema.p4s() {
        let [e, px, py, pz] = options.column_names.p4_suffixes.physical_p4_names(name);
        fields.push(Field::new(e, data_type.clone(), false));
        fields.push(Field::new(px, data_type.clone(), false));
        fields.push(Field::new(py, data_type.clone(), false));
        fields.push(Field::new(pz, data_type.clone(), false));
    }

    for name in schema.scalars() {
        fields.push(Field::new(name.as_ref(), data_type.clone(), false));
    }

    if should_write_weight {
        fields.push(Field::new(
            options.column_names.weight_column.as_ref(),
            data_type,
            false,
        ));
    }

    ArrowSchema::new(fields)
}

fn array_from_iter<I: IntoIterator<Item = f64>>(iter: I, precision: Precision) -> ArrayRef {
    match precision {
        Precision::F64 => Arc::new(Float64Array::from_iter_values(iter)),
        Precision::F32 => Arc::new(Float32Array::from_iter_values(
            iter.into_iter().map(|f| f as f32),
        )),
    }
}

fn event_batch_to_record_batch(
    batch: &EventBatch,
    arrow_schema: SchemaRef,
    write_weight: WriteWeightColumn,
    precision: Precision,
) -> LadduDataResult<RecordBatch> {
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(arrow_schema.fields().len());

    for col in 0..batch.schema().n_p4s() {
        let p = batch.vec4_column(col);

        columns.push(array_from_iter(p.iter().map(|x| x.e), precision));
        columns.push(array_from_iter(p.iter().map(|x| x.px), precision));
        columns.push(array_from_iter(p.iter().map(|x| x.py), precision));
        columns.push(array_from_iter(p.iter().map(|x| x.pz), precision));
    }

    for col in 0..batch.schema().n_scalars() {
        columns.push(array_from_iter(
            batch.scalar_column(col).iter().copied(),
            precision,
        ));
    }

    let should_write_weight =
        matches!(write_weight, WriteWeightColumn::Always) || batch.schema().has_weight();

    if should_write_weight {
        columns.push(array_from_iter(
            (0..batch.len()).map(|i| batch.weights_at(i)),
            precision,
        ));
    }

    RecordBatch::try_new(arrow_schema, columns).map_err(|e| LadduDataError::Sink(e.to_string()))
}

#[cfg(test)]
mod tests {
    use arrow::array::{Float32Array, Float64Array};

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
                ..ReadPlan::default()
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
