use std::{
    path::{Path, PathBuf},
    sync::{
        Arc,
        mpsc::{self, Receiver, Sender},
    },
    thread::{self, JoinHandle},
};

use laddu_physics::vectors::RealVec4;
use oxyroot::{Branch, ReaderTree, RootFile, WriterTree};

use crate::{
    LadduDataError, LadduDataResult, Name,
    data::EventBatch,
    io::{
        DataFragment, EventSink, EventSource, FragmentedSource, OutputMode, OutputPath, ReadPlan,
        SourceCapabilities, WritePlan, fragmented_batches,
    },
    schema::{
        ColumnInfo, ColumnType, Precision, Schema, SchemaColumnNames, SchemaInferenceOptions,
        SchemaWriteOptions, WriteWeightColumn,
    },
};

/// Event source backed by one or more ROOT TTrees.
#[derive(Clone, Debug)]
pub struct RootSource {
    files: Arc<[Arc<PathBuf>]>,
    tree_name: Name,
    schema: Arc<Schema>,
    options: RootReadOptions,
}

/// Schema, validation, glob, and tree-selection options for ROOT reads.
#[derive(Clone, Debug)]
pub struct RootReadOptions {
    /// Infer a logical schema when none is supplied.
    pub infer_schema: bool,
    /// Validate required columns in every matched file.
    pub validate_all_files: bool,
    /// Sort glob results for deterministic global row order.
    pub sort_glob: bool,
    /// TTree selection policy.
    pub tree: RootTreeSelection,
    /// Logical schema inference options.
    pub schema_inference: SchemaInferenceOptions,
}

impl Default for RootReadOptions {
    fn default() -> Self {
        Self {
            infer_schema: true,
            validate_all_files: true,
            sort_glob: true,
            tree: RootTreeSelection::First,
            schema_inference: SchemaInferenceOptions::default(),
        }
    }
}

/// Policy for selecting a TTree from each ROOT file.
#[derive(Clone, Debug, Default)]
pub enum RootTreeSelection {
    /// Select the first TTree.
    #[default]
    First,
    /// Select a TTree by name.
    Named(Name),
}

/// Key identifying one TTree within a ROOT file.
#[derive(Clone, Debug)]
pub struct RootFragmentKey {
    /// Input file path.
    pub file: Arc<PathBuf>,
    /// TTree name.
    pub tree_name: Name,
}

/// Introspection metadata for one ROOT branch.
#[derive(Clone, Debug)]
pub struct RootColumnInfo {
    /// Branch name.
    pub name: Name,
    /// Rust item type reported by the reader.
    pub item_type_name: String,
    /// ROOT interpretation string.
    pub interpretation: String,
    /// Number of branch entries.
    pub entries: i64,
}

impl RootSource {
    /// Opens files matching a glob with default options.
    pub fn open(pattern: impl AsRef<str>) -> LadduDataResult<Self> {
        Self::builder(pattern).build()
    }

    /// Creates a configurable source builder for a file glob.
    pub fn builder(pattern: impl AsRef<str>) -> RootSourceBuilder {
        RootSourceBuilder {
            pattern: pattern.as_ref().to_owned(),
            schema: None,
            options: RootReadOptions::default(),
        }
    }

    /// Returns matched files in global row order.
    pub fn files(&self) -> &[Arc<PathBuf>] {
        &self.files
    }

    /// Returns the selected TTree name.
    pub fn tree_name(&self) -> &str {
        self.tree_name.as_ref()
    }

    /// Lists TTrees in one ROOT file.
    pub fn tree_names(path: impl AsRef<Path>) -> LadduDataResult<Vec<Name>> {
        let mut file = RootFile::open(path.as_ref()).map_err(root_source_error)?;
        let key_names: Vec<String> = file.keys_name().map(str::to_owned).collect();

        let mut out = Vec::new();

        for name in key_names {
            if file.get_tree(&name).is_ok() {
                out.push(Name::from(name));
            }
        }

        Ok(out)
    }

    /// Lists branch metadata for a selected or first TTree.
    pub fn columns(
        path: impl AsRef<Path>,
        tree: Option<&str>,
    ) -> LadduDataResult<Vec<RootColumnInfo>> {
        let mut file = RootFile::open(path.as_ref()).map_err(root_source_error)?;

        let tree_name = match tree {
            Some(name) => Name::from(name),
            None => first_tree_name(&mut file)?,
        };

        let tree = file
            .get_tree(tree_name.as_ref())
            .map_err(root_source_error)?;

        Ok(tree
            .branches_r()
            .into_iter()
            .map(|branch| RootColumnInfo {
                name: Name::from(branch.name()),
                item_type_name: branch.item_type_name(),
                interpretation: branch.interpretation(),
                entries: branch.entries(),
            })
            .collect())
    }
}

/// Builder for a [`RootSource`].
pub struct RootSourceBuilder {
    pattern: String,
    schema: Option<Arc<Schema>>,
    options: RootReadOptions,
}

impl RootSourceBuilder {
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

    /// Selects a TTree by name.
    pub fn tree(mut self, name: impl Into<Name>) -> Self {
        self.options.tree = RootTreeSelection::Named(name.into());
        self
    }

    /// Selects the first TTree.
    pub fn first_tree(mut self) -> Self {
        self.options.tree = RootTreeSelection::First;
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

    /// Resolves files and tree, validates schema, and builds the source.
    pub fn build(self) -> LadduDataResult<RootSource> {
        let RootSourceBuilder {
            pattern,
            schema,
            options,
        } = self;

        let mut files: Vec<PathBuf> = glob::glob(&pattern)
            .map_err(|e| LadduDataError::Source(e.to_string()))?
            .collect::<std::result::Result<_, _>>()
            .map_err(|e| LadduDataError::Source(e.to_string()))?;

        if options.sort_glob {
            files.sort();
        }

        if files.is_empty() {
            return Err(LadduDataError::Source("no ROOT files matched glob".into()));
        }

        let files: Arc<[Arc<PathBuf>]> = files.into_iter().map(Arc::new).collect();

        let tree_name = {
            let path: &Path = files[0].as_ref();
            resolve_tree_name(path, &options.tree)?
        };

        let schema = match schema {
            Some(schema) => schema,
            None if options.infer_schema => {
                let path: &Path = files[0].as_ref();
                let columns = root_columns(path, tree_name.as_ref())?;

                Arc::new(Schema::infer_from_columns(
                    columns.iter().map(OwnedColumnInfo::as_column_info),
                    &options.schema_inference,
                )?)
            }
            None => return Err(LadduDataError::InvalidArgument("schema required")),
        };

        if options.validate_all_files {
            for file in files.iter() {
                let path: &Path = file.as_ref();
                validate_root_file(path, tree_name.as_ref(), &schema, &options.schema_inference)?;
            }
        }

        Ok(RootSource {
            files,
            tree_name,
            schema,
            options,
        })
    }
}

impl EventSource for RootSource {
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

impl FragmentedSource for RootSource {
    type Key = RootFragmentKey;

    fn fragments(&self) -> LadduDataResult<Vec<DataFragment<Self::Key>>> {
        let mut fragments = Vec::new();
        let mut global_start = 0_u64;

        for path in self.files.iter() {
            let mut file = RootFile::open(path.as_ref()).map_err(root_source_error)?;
            let tree = file
                .get_tree(self.tree_name.as_ref())
                .map_err(root_source_error)?;

            let rows = usize_from_i64(tree.entries(), "negative TTree entry count")? as u64;

            fragments.push(DataFragment {
                key: RootFragmentKey {
                    file: Arc::clone(path),
                    tree_name: self.tree_name.clone(),
                },
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

        Ok(Box::new(RootBatchIter::spawn(
            Arc::clone(&self.schema),
            self.options.clone(),
            key.clone(),
            local_start,
            local_len,
            chunk_size,
        )))
    }
}

struct RootBatchIter {
    rx: Receiver<LadduDataResult<EventBatch>>,
    handle: Option<JoinHandle<()>>,
    joined: bool,
}

impl RootBatchIter {
    fn spawn(
        schema: Arc<Schema>,
        options: RootReadOptions,
        key: RootFragmentKey,
        local_start: usize,
        local_len: usize,
        chunk_size: Option<usize>,
    ) -> Self {
        let (tx, rx) = mpsc::channel();

        let handle = thread::spawn(move || {
            if let Err(err) = read_root_range_and_send_batches(
                schema,
                options,
                key,
                local_start,
                local_len,
                chunk_size,
                tx.clone(),
            ) {
                let _ = tx.send(Err(err));
            }
        });

        Self {
            rx,
            handle: Some(handle),
            joined: false,
        }
    }

    fn join_if_needed(&mut self) -> Option<LadduDataResult<EventBatch>> {
        if self.joined {
            return None;
        }

        self.joined = true;

        if let Some(handle) = self.handle.take()
            && handle.join().is_err()
        {
            return Some(Err(LadduDataError::Source(
                "ROOT reader thread panicked".into(),
            )));
        }

        None
    }
}

impl Iterator for RootBatchIter {
    type Item = LadduDataResult<EventBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.rx.recv() {
            Ok(item) => Some(item),
            Err(_) => self.join_if_needed(),
        }
    }
}

fn read_root_range_and_send_batches(
    schema: Arc<Schema>,
    options: RootReadOptions,
    key: RootFragmentKey,
    local_start: usize,
    local_len: usize,
    chunk_size: Option<usize>,
    tx: Sender<LadduDataResult<EventBatch>>,
) -> LadduDataResult<()> {
    let mut file = RootFile::open(key.file.as_ref()).map_err(root_source_error)?;
    let tree = file
        .get_tree(key.tree_name.as_ref())
        .map_err(root_source_error)?;

    let mut readers =
        RootColumnReaders::new(&tree, &schema, &options.schema_inference.column_names)?;

    for _ in 0..local_start {
        readers.skip_one()?;
    }

    let mut remaining = local_len;
    let batch_size = chunk_size.unwrap_or(local_len.max(1));

    while remaining > 0 {
        let take = remaining.min(batch_size);
        let batch = readers.read_batch(Arc::clone(&schema), take)?;

        tx.send(Ok(batch))
            .map_err(|e| LadduDataError::Source(e.to_string()))?;

        remaining -= take;
    }

    Ok(())
}

struct RootColumnReaders<'a> {
    p4s: Vec<[RootFloatIter<'a>; 4]>,
    scalars: Vec<RootFloatIter<'a>>,
    weights: Option<RootFloatIter<'a>>,
}

impl<'a> RootColumnReaders<'a> {
    fn new(
        tree: &'a ReaderTree,
        schema: &Schema,
        column_names: &SchemaColumnNames,
    ) -> LadduDataResult<Self> {
        let mut p4s = Vec::with_capacity(schema.n_p4s());
        let mut scalars = Vec::with_capacity(schema.n_scalars());

        for name in schema.p4s() {
            let [e, px, py, pz] = column_names.p4_suffixes.physical_p4_names(name);

            p4s.push([
                open_float_reader(tree, e.as_ref())?,
                open_float_reader(tree, px.as_ref())?,
                open_float_reader(tree, py.as_ref())?,
                open_float_reader(tree, pz.as_ref())?,
            ]);
        }

        for name in schema.scalars() {
            scalars.push(open_float_reader(tree, name.as_ref())?);
        }

        let weights = if schema.has_weight() {
            Some(open_float_reader(
                tree,
                column_names.weight_column.as_ref(),
            )?)
        } else {
            None
        };

        Ok(Self {
            p4s,
            scalars,
            weights,
        })
    }

    fn skip_one(&mut self) -> LadduDataResult<()> {
        for [e, px, py, pz] in self.p4s.iter_mut() {
            e.next_f64()?;
            px.next_f64()?;
            py.next_f64()?;
            pz.next_f64()?;
        }

        for scalar in self.scalars.iter_mut() {
            scalar.next_f64()?;
        }

        if let Some(weights) = self.weights.as_mut() {
            weights.next_f64()?;
        }

        Ok(())
    }

    fn read_batch(&mut self, schema: Arc<Schema>, len: usize) -> LadduDataResult<EventBatch> {
        let mut p4s = Vec::with_capacity(schema.n_p4s());
        let mut scalars = Vec::with_capacity(schema.n_scalars());

        for [e, px, py, pz] in self.p4s.iter_mut() {
            let mut col = Vec::with_capacity(len);

            for _ in 0..len {
                col.push(RealVec4 {
                    e: e.next_f64()?,
                    px: px.next_f64()?,
                    py: py.next_f64()?,
                    pz: pz.next_f64()?,
                });
            }

            p4s.push(Arc::from(col));
        }

        for reader in self.scalars.iter_mut() {
            let mut col = Vec::with_capacity(len);

            for _ in 0..len {
                col.push(reader.next_f64()?);
            }

            scalars.push(Arc::from(col));
        }

        let weights = if let Some(reader) = self.weights.as_mut() {
            let mut col = Vec::with_capacity(len);

            for _ in 0..len {
                col.push(reader.next_f64()?);
            }

            Some(Arc::from(col))
        } else {
            None
        };

        EventBatch::new(schema, p4s, scalars, weights)
    }
}

enum RootFloatIter<'a> {
    F64(Box<dyn Iterator<Item = f64> + 'a>),
    F32(Box<dyn Iterator<Item = f32> + 'a>),
}

impl<'a> RootFloatIter<'a> {
    fn next_f64(&mut self) -> LadduDataResult<f64> {
        match self {
            Self::F64(iter) => iter
                .next()
                .ok_or_else(|| LadduDataError::Source("ROOT branch ended early".into())),
            Self::F32(iter) => iter
                .next()
                .map(f64::from)
                .ok_or_else(|| LadduDataError::Source("ROOT branch ended early".into())),
        }
    }
}

fn open_float_reader<'a>(tree: &'a ReaderTree, name: &str) -> LadduDataResult<RootFloatIter<'a>> {
    let branch =
        find_branch(tree, name).ok_or_else(|| LadduDataError::MissingColumn(Name::from(name)))?;

    match root_column_type(branch) {
        ColumnType::F64 => Ok(RootFloatIter::F64(Box::new(
            branch.as_iter::<f64>().map_err(root_source_error)?,
        ))),
        ColumnType::F32 => Ok(RootFloatIter::F32(Box::new(
            branch.as_iter::<f32>().map_err(root_source_error)?,
        ))),
        ColumnType::Other => Err(LadduDataError::Source(format!(
            "column {name} has unsupported ROOT type {} interpreted as {}",
            branch.item_type_name(),
            branch.interpretation()
        ))),
    }
}

fn find_branch<'a>(tree: &'a ReaderTree, name: &str) -> Option<&'a Branch> {
    tree.branch(name).or_else(|| {
        tree.branches_r()
            .into_iter()
            .find(|branch| branch.name() == name)
    })
}

#[derive(Clone, Debug)]
struct OwnedColumnInfo {
    name: Name,
    dtype: ColumnType,
}

impl OwnedColumnInfo {
    fn as_column_info(&self) -> ColumnInfo<'_> {
        ColumnInfo {
            name: self.name.as_ref(),
            dtype: self.dtype,
        }
    }
}

fn root_columns(path: &Path, tree_name: &str) -> LadduDataResult<Vec<OwnedColumnInfo>> {
    let mut file = RootFile::open(path).map_err(root_source_error)?;
    let tree = file.get_tree(tree_name).map_err(root_source_error)?;

    Ok(tree
        .branches_r()
        .into_iter()
        .map(|branch| OwnedColumnInfo {
            name: Name::from(branch.name()),
            dtype: root_column_type(branch),
        })
        .collect())
}

fn validate_root_file(
    path: &Path,
    tree_name: &str,
    schema: &Schema,
    options: &SchemaInferenceOptions,
) -> LadduDataResult<()> {
    let columns = root_columns(path, tree_name)?;

    schema.validate_required_columns(columns.iter().map(OwnedColumnInfo::as_column_info), options)
}

fn root_column_type(branch: &Branch) -> ColumnType {
    match branch.interpretation().as_str() {
        "f64" => ColumnType::F64,
        "f32" => ColumnType::F32,
        _ => match branch.item_type_name().as_str() {
            "double" | "Double_t" | "ROOT::Double_t" => ColumnType::F64,
            "float" | "Float_t" | "ROOT::Float_t" => ColumnType::F32,
            _ => ColumnType::Other,
        },
    }
}

fn resolve_tree_name(path: &Path, selection: &RootTreeSelection) -> LadduDataResult<Name> {
    let mut file = RootFile::open(path).map_err(root_source_error)?;

    match selection {
        RootTreeSelection::Named(name) => {
            file.get_tree(name.as_ref()).map_err(root_source_error)?;
            Ok(name.clone())
        }
        RootTreeSelection::First => first_tree_name(&mut file),
    }
}

fn first_tree_name(file: &mut RootFile) -> LadduDataResult<Name> {
    let key_names: Vec<String> = file.keys_name().map(str::to_owned).collect();

    for name in key_names {
        if file.get_tree(&name).is_ok() {
            return Ok(Name::from(name));
        }
    }

    Err(LadduDataError::Source("no TTree found in ROOT file".into()))
}

fn usize_from_i64(value: i64, message: &'static str) -> LadduDataResult<usize> {
    if value < 0 {
        return Err(LadduDataError::Source(message.into()));
    }

    usize::try_from(value).map_err(|_| LadduDataError::Source("entry count overflows usize".into()))
}

fn root_source_error(e: impl std::fmt::Display) -> LadduDataError {
    LadduDataError::Source(e.to_string())
}

fn root_sink_error(e: impl std::fmt::Display) -> LadduDataError {
    LadduDataError::Sink(e.to_string())
}

/// Event sink that writes a ROOT TTree on a background thread.
pub struct RootSink {
    output: OutputPath,
    options: RootWriteOptions,
    resolved_path: Option<PathBuf>,
    event_schema: Option<Arc<Schema>>,
    senders: Option<RootColumnSenders>,
    writer_thread: Option<JoinHandle<LadduDataResult<()>>>,
}

/// ROOT tree and physical schema write options.
#[derive(Clone, Debug)]
pub struct RootWriteOptions {
    /// Output TTree name.
    pub tree_name: Name,
    /// Physical schema write options.
    pub schema_write: SchemaWriteOptions,
}

impl Default for RootWriteOptions {
    fn default() -> Self {
        Self {
            tree_name: Name::from("tree"),
            schema_write: SchemaWriteOptions::default(),
        }
    }
}

impl RootSink {
    /// Creates a sink with default options.
    pub fn create(path: impl Into<PathBuf>) -> Self {
        Self::builder(path).build()
    }

    /// Creates a configurable sink builder.
    pub fn builder(path: impl Into<PathBuf>) -> RootSinkBuilder {
        RootSinkBuilder {
            output: OutputPath::new(path),
            options: RootWriteOptions::default(),
        }
    }

    /// Returns the concrete path after writing has begun.
    pub fn resolved_path(&self) -> Option<&Path> {
        self.resolved_path.as_deref()
    }
}

/// Builder for a [`RootSink`].
pub struct RootSinkBuilder {
    output: OutputPath,
    options: RootWriteOptions,
}

impl RootSinkBuilder {
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

    /// Sets the output TTree name.
    pub fn tree(mut self, name: impl Into<Name>) -> Self {
        self.options.tree_name = name.into();
        self
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

    /// Sets the weight-column emission policy.
    pub fn write_weight_column(mut self, value: WriteWeightColumn) -> Self {
        self.options.schema_write.write_weight_column = value;
        self
    }

    /// Builds the sink.
    pub fn build(self) -> RootSink {
        RootSink {
            output: self.output,
            options: self.options,
            resolved_path: None,
            event_schema: None,
            senders: None,
            writer_thread: None,
        }
    }
}

impl EventSink for RootSink {
    fn begin(&mut self, schema: Arc<Schema>, plan: WritePlan) -> LadduDataResult<()> {
        if self.writer_thread.is_some() {
            return Err(LadduDataError::Sink("ROOT sink already initialized".into()));
        }

        let path = self.output.resolve(plan, "root")?;
        OutputPath::create_parent_dirs(&path)?;

        let columns = root_output_columns(
            &schema,
            self.options.schema_write.write_weight_column,
            &self.options.schema_write,
        );

        let (senders, receivers) = root_channels(&columns, self.options.schema_write.precision);

        let writer_path = path.clone();
        let tree_name = self.options.tree_name.clone();

        let handle = thread::spawn(move || write_root_tree(writer_path, tree_name, receivers));

        self.resolved_path = Some(path);
        self.event_schema = Some(schema);
        self.senders = Some(senders);
        self.writer_thread = Some(handle);

        Ok(())
    }

    fn write_batch(&mut self, batch: &EventBatch) -> LadduDataResult<()> {
        let event_schema = self
            .event_schema
            .as_ref()
            .ok_or_else(|| LadduDataError::Sink("ROOT sink not initialized".into()))?;

        if event_schema.as_ref() != batch.schema().as_ref() {
            return Err(LadduDataError::Sink(
                "batch schema does not match ROOT sink schema".into(),
            ));
        }

        let senders = self
            .senders
            .as_ref()
            .ok_or_else(|| LadduDataError::Sink("ROOT sink not initialized".into()))?;

        let should_write_weight = matches!(
            self.options.schema_write.write_weight_column,
            WriteWeightColumn::Always
        ) || batch.schema().has_weight();

        for row in 0..batch.len() {
            let mut index = 0;

            for col in 0..batch.schema().n_p4s() {
                let p = batch.p4_at(col, row);

                senders.send(index, p.e)?;
                index += 1;

                senders.send(index, p.px)?;
                index += 1;

                senders.send(index, p.py)?;
                index += 1;

                senders.send(index, p.pz)?;
                index += 1;
            }

            for col in 0..batch.schema().n_scalars() {
                senders.send(index, batch.scalar_at(col, row))?;
                index += 1;
            }

            if should_write_weight {
                senders.send(index, batch.weights_at(row))?;
            }
        }

        Ok(())
    }

    fn finish(&mut self) -> LadduDataResult<()> {
        self.senders.take();

        if let Some(handle) = self.writer_thread.take() {
            match handle.join() {
                Ok(result) => result?,
                Err(_) => return Err(LadduDataError::Sink("ROOT writer thread panicked".into())),
            }
        }

        Ok(())
    }
}

impl Drop for RootSink {
    fn drop(&mut self) {
        let _ = self.finish();
    }
}

enum RootColumnSenders {
    F64(Vec<Sender<f64>>),
    F32(Vec<Sender<f32>>),
}

impl RootColumnSenders {
    fn send(&self, index: usize, value: f64) -> LadduDataResult<()> {
        match self {
            Self::F64(senders) => senders[index]
                .send(value)
                .map_err(|e| LadduDataError::Sink(e.to_string())),
            Self::F32(senders) => senders[index]
                .send(value as f32)
                .map_err(|e| LadduDataError::Sink(e.to_string())),
        }
    }
}

enum RootColumnReceivers {
    F64(Vec<(Name, Receiver<f64>)>),
    F32(Vec<(Name, Receiver<f32>)>),
}

fn root_channels(
    columns: &[Name],
    precision: Precision,
) -> (RootColumnSenders, RootColumnReceivers) {
    match precision {
        Precision::F64 => {
            let mut senders = Vec::with_capacity(columns.len());
            let mut receivers = Vec::with_capacity(columns.len());

            for name in columns {
                let (tx, rx) = mpsc::channel();
                senders.push(tx);
                receivers.push((name.clone(), rx));
            }

            (
                RootColumnSenders::F64(senders),
                RootColumnReceivers::F64(receivers),
            )
        }
        Precision::F32 => {
            let mut senders = Vec::with_capacity(columns.len());
            let mut receivers = Vec::with_capacity(columns.len());

            for name in columns {
                let (tx, rx) = mpsc::channel();
                senders.push(tx);
                receivers.push((name.clone(), rx));
            }

            (
                RootColumnSenders::F32(senders),
                RootColumnReceivers::F32(receivers),
            )
        }
    }
}

fn write_root_tree(
    path: PathBuf,
    tree_name: Name,
    receivers: RootColumnReceivers,
) -> LadduDataResult<()> {
    let mut file = RootFile::create(&path).map_err(root_sink_error)?;
    let mut tree = WriterTree::new(tree_name.as_ref());

    match receivers {
        RootColumnReceivers::F64(receivers) => {
            for (name, rx) in receivers {
                tree.new_branch(name.as_ref(), rx.into_iter());
            }
        }
        RootColumnReceivers::F32(receivers) => {
            for (name, rx) in receivers {
                tree.new_branch(name.as_ref(), rx.into_iter());
            }
        }
    }

    tree.write(&mut file).map_err(root_sink_error)?;
    file.close().map_err(root_sink_error)?;

    Ok(())
}

fn root_output_columns(
    schema: &Schema,
    write_weight: WriteWeightColumn,
    options: &SchemaWriteOptions,
) -> Vec<Name> {
    let should_write_weight =
        matches!(write_weight, WriteWeightColumn::Always) || schema.has_weight();

    let mut columns = Vec::with_capacity(
        4 * schema.n_p4s() + schema.n_scalars() + usize::from(should_write_weight),
    );

    for name in schema.p4s() {
        let [e, px, py, pz] = options.column_names.p4_suffixes.physical_p4_names(name);
        columns.push(e.into());
        columns.push(px.into());
        columns.push(py.into());
        columns.push(pz.into());
    }

    for name in schema.scalars() {
        columns.push(name.clone());
    }

    if should_write_weight {
        columns.push(options.column_names.weight_column.clone());
    }

    columns
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{Dataset, EventBatchBuilder};

    fn temp_path(ext: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();

        std::env::temp_dir().join(format!(
            "laddu-root-test-{}-{nanos}.{ext}",
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
    fn root_sink_and_source_roundtrip_named_tree_with_f32_precision() {
        let path = temp_path("root");
        let batch = batch();

        let mut sink = RootSink::builder(path.clone())
            .tree("events")
            .precision(Precision::F32)
            .build();

        sink.begin(Arc::clone(batch.schema()), WritePlan::default())
            .unwrap();
        sink.write_batch(&batch).unwrap();
        sink.finish().unwrap();

        let tree_names = RootSource::tree_names(&path).unwrap();
        assert!(tree_names.iter().any(|name| name.as_ref() == "events"));

        let columns = RootSource::columns(&path, Some("events")).unwrap();
        let names = columns
            .iter()
            .map(|col| col.name.to_string())
            .collect::<Vec<_>>();

        for expected in ["p_e", "p_px", "p_py", "p_pz", "mass", "weight"] {
            assert!(
                names.iter().any(|name| name == expected),
                "missing {expected}"
            );
        }

        let source = RootSource::builder(path.to_str().unwrap())
            .tree("events")
            .build()
            .unwrap();

        assert_eq!(source.tree_name(), "events");
        assert_eq!(source.num_events().unwrap(), Some(4));

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
    fn root_source_infers_first_tree_when_no_tree_is_named() {
        let path = temp_path("root");
        let batch = batch();

        let mut sink = RootSink::builder(path.clone()).tree("first_tree").build();

        sink.begin(Arc::clone(batch.schema()), WritePlan::default())
            .unwrap();
        sink.write_batch(&batch).unwrap();
        sink.finish().unwrap();

        let source = RootSource::builder(path.to_str().unwrap())
            .first_tree()
            .build()
            .unwrap();

        assert_eq!(source.tree_name(), "first_tree");

        let read = EventBatch::concat(
            &source
                .batches(ReadPlan::default())
                .unwrap()
                .map(Result::unwrap)
                .collect::<Vec<_>>(),
        )
        .unwrap();

        assert_eq!(read.scalar_column(0), &[100.0, 101.0, 102.0, 103.0]);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn root_source_named_missing_tree_fails() {
        let path = temp_path("root");
        let batch = batch();

        let mut sink = RootSink::builder(path.clone()).tree("events").build();

        sink.begin(Arc::clone(batch.schema()), WritePlan::default())
            .unwrap();
        sink.write_batch(&batch).unwrap();
        sink.finish().unwrap();

        let err = RootSource::builder(path.to_str().unwrap())
            .tree("missing")
            .build()
            .unwrap_err();

        assert!(matches!(err, LadduDataError::Source(_)));

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn root_sink_rejects_batches_with_different_schema() {
        let path = temp_path("root");
        let batch = batch();

        let mut sink = RootSink::builder(path.clone()).tree("events").build();

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
    fn dataset_write_to_root_applies_dataset_transformations_before_writing() {
        let path = temp_path("root");

        let dataset = Dataset::from_batch(batch()).filter(|ev| ev.scalar(0) >= 102.0);

        let mut sink = RootSink::builder(path.clone()).tree("events").build();

        dataset.write_to(&mut sink).unwrap();

        let source = RootSource::builder(path.to_str().unwrap())
            .tree("events")
            .build()
            .unwrap();

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
