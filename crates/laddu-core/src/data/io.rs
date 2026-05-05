//! Dataset I/O implementations and shared column-inference helpers.

#[cfg(feature = "mpi")]
use std::{cell::RefCell, rc::Rc};
use std::{
    fs::File,
    ops::Range,
    path::{Path, PathBuf},
    sync::Arc,
};

use arrow::{
    array::{Float32Array, Float64Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use indexmap::{IndexMap, IndexSet};
#[cfg(feature = "mpi")]
use mpi::topology::SimpleCommunicator;
#[cfg(feature = "mpi")]
use mpi::traits::*;
use oxyroot::{Branch, Named, ReaderTree, RootFile, WriterTree};
use parquet::arrow::{arrow_reader::ParquetRecordBatchReaderBuilder, ArrowWriter};
#[cfg(feature = "mpi")]
use parquet::file::metadata::ParquetMetaData;

use super::{dataset::*, event::*, metadata::*};
#[cfg(feature = "mpi")]
use crate::mpi::LadduMPI;
use crate::{LadduError, LadduResult};

fn canonicalize_dataset_path(file_path: &str) -> LadduResult<PathBuf> {
    Ok(Path::new(&*shellexpand::full(file_path)?).canonicalize()?)
}

fn expand_output_path(file_path: &str) -> LadduResult<PathBuf> {
    Ok(PathBuf::from(&*shellexpand::full(file_path)?))
}

/// Load a [`Dataset`] from a Parquet file.
pub fn read_parquet(file_path: &str, options: &DatasetReadOptions) -> LadduResult<Arc<Dataset>> {
    let storage = read_parquet_storage(file_path, options)?;
    Ok(Arc::new(storage.to_dataset()))
}

/// Load a Parquet file as an iterator of chunked [`Dataset`] values.
pub fn read_parquet_chunks(
    file_path: &str,
    options: &DatasetReadOptions,
    chunk_size: usize,
) -> LadduResult<impl Iterator<Item = LadduResult<Arc<Dataset>>>> {
    let chunks = read_parquet_storage_chunks(file_path, options, chunk_size)?;
    Ok(chunks.map(|chunk| chunk.map(|storage| Arc::new(storage.to_dataset()))))
}

/// Load a Parquet file as chunked [`Dataset`] values using the chunk size from [`DatasetReadOptions`].
pub fn read_parquet_chunks_with_options(
    file_path: &str,
    options: &DatasetReadOptions,
) -> LadduResult<impl Iterator<Item = LadduResult<Arc<Dataset>>>> {
    read_parquet_chunks(
        file_path,
        options,
        options.chunk_size.unwrap_or(DEFAULT_READ_CHUNK_SIZE),
    )
}

pub(crate) fn read_parquet_storage(
    file_path: &str,
    options: &DatasetReadOptions,
) -> LadduResult<Arc<DatasetStorage>> {
    let path = canonicalize_dataset_path(file_path)?;
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema = builder.schema();
    let float_cols: Vec<&str> = schema
        .fields()
        .iter()
        .filter(|f| matches!(f.data_type(), DataType::Float32 | DataType::Float64))
        .map(|f| f.name().as_str())
        .collect();
    let (detected_p4_names, detected_aux_names) = infer_p4_and_aux_names(&float_cols);
    let metadata = options.resolve_metadata(detected_p4_names, detected_aux_names)?;

    #[cfg(feature = "mpi")]
    {
        if let Some(world) = crate::mpi::get_world() {
            return read_parquet_columnar_mpi(builder, metadata, &world);
        }
    }

    read_parquet_columnar_local(builder, metadata)
}

pub(crate) fn read_parquet_storage_chunks(
    file_path: &str,
    options: &DatasetReadOptions,
    chunk_size: usize,
) -> LadduResult<impl Iterator<Item = LadduResult<Arc<DatasetStorage>>>> {
    let path = canonicalize_dataset_path(file_path)?;
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema = builder.schema();
    let float_cols: Vec<&str> = schema
        .fields()
        .iter()
        .filter(|f| matches!(f.data_type(), DataType::Float32 | DataType::Float64))
        .map(|f| f.name().as_str())
        .collect();
    let (detected_p4_names, detected_aux_names) = infer_p4_and_aux_names(&float_cols);
    let metadata = options.resolve_metadata(detected_p4_names, detected_aux_names)?;
    #[cfg(feature = "mpi")]
    {
        if let Some(world) = crate::mpi::get_world() {
            let plan = configure_parquet_mpi_read(builder, &world);
            if plan.local_rows == 0 {
                return Ok(read_parquet_storage_chunks_from_reader(
                    None,
                    metadata,
                    chunk_size,
                    0,
                    Some(0),
                ));
            }
            let reader = plan.builder.build()?;
            return Ok(read_parquet_storage_chunks_from_reader(
                Some(reader),
                metadata,
                chunk_size,
                plan.skip_rows,
                Some(plan.local_rows),
            ));
        }
    }

    let reader = builder.build()?;
    Ok(read_parquet_storage_chunks_from_reader(
        Some(reader),
        metadata,
        chunk_size,
        0,
        None,
    ))
}

fn read_parquet_storage_chunks_from_reader(
    mut reader: Option<impl Iterator<Item = arrow::error::Result<RecordBatch>>>,
    metadata: Arc<DatasetMetadata>,
    chunk_size: usize,
    mut remaining_skip: usize,
    mut remaining_rows: Option<usize>,
) -> impl Iterator<Item = LadduResult<Arc<DatasetStorage>>> {
    let rows_per_chunk = chunk_size.max(1);
    let mut pending_batch: Option<RecordBatch> = None;
    let mut pending_offset = 0usize;

    std::iter::from_fn(move || {
        if matches!(remaining_rows, Some(0)) {
            return None;
        }

        let mut p4 = (0..metadata.p4_names.len())
            .map(|_| ColumnarP4Column::with_capacity(rows_per_chunk))
            .collect::<Vec<_>>();
        let mut aux = (0..metadata.aux_names.len())
            .map(|_| Vec::with_capacity(rows_per_chunk))
            .collect::<Vec<_>>();
        let mut weights = Vec::with_capacity(rows_per_chunk);

        while weights.len() < rows_per_chunk {
            if matches!(remaining_rows, Some(0)) {
                break;
            }

            if pending_batch.is_none() {
                let next_batch = match reader.as_mut().and_then(Iterator::next) {
                    Some(batch) => batch,
                    None => break,
                };
                let batch = match next_batch {
                    Ok(batch) => batch,
                    Err(err) => return Some(Err(err.into())),
                };
                if remaining_skip >= batch.num_rows() {
                    remaining_skip -= batch.num_rows();
                    continue;
                }
                pending_offset = remaining_skip;
                pending_batch = Some(batch);
                remaining_skip = 0;
            }

            let batch = pending_batch
                .as_ref()
                .expect("pending batch should exist after refill");
            let available_rows = batch.num_rows().saturating_sub(pending_offset);
            if available_rows == 0 {
                pending_batch = None;
                pending_offset = 0;
                continue;
            }

            let remaining_chunk_rows = rows_per_chunk - weights.len();
            let remaining_window_rows = remaining_rows.unwrap_or(usize::MAX);
            let take_rows = available_rows
                .min(remaining_chunk_rows)
                .min(remaining_window_rows);
            let end_row = pending_offset + take_rows;
            if let Err(err) = append_record_batch_range_to_columnar(
                batch,
                metadata.as_ref(),
                pending_offset,
                end_row,
                &mut p4,
                &mut aux,
                &mut weights,
            ) {
                return Some(Err(err));
            }
            if let Some(rows_left) = remaining_rows.as_mut() {
                *rows_left -= take_rows;
            }
            pending_offset = end_row;
            if pending_offset >= batch.num_rows() || matches!(remaining_rows, Some(0)) {
                pending_batch = None;
                pending_offset = 0;
            }
        }

        if weights.is_empty() {
            return None;
        }

        Some(Ok(Arc::new(DatasetStorage {
            metadata: metadata.clone(),
            p4,
            aux,
            weights,
        })))
    })
}

fn append_record_batch_range_to_columnar(
    batch: &RecordBatch,
    metadata: &DatasetMetadata,
    start_row: usize,
    end_row: usize,
    p4_columns_out: &mut [ColumnarP4Column],
    aux_columns_out: &mut [Vec<f64>],
    weights_out: &mut Vec<f64>,
) -> LadduResult<()> {
    let p4_columns: Vec<P4Columns<'_>> = metadata
        .p4_names
        .iter()
        .map(|name| prepare_p4_columns(batch, name))
        .collect::<Result<_, _>>()?;
    let aux_columns: Vec<FloatColumn<'_>> = metadata
        .aux_names
        .iter()
        .map(|name| prepare_float_column(batch, name))
        .collect::<Result<_, _>>()?;
    let weight_column = find_float_column_from_candidates(batch, &["weight".to_string()])?;

    for row in start_row..end_row {
        for (target, source) in p4_columns_out.iter_mut().zip(&p4_columns) {
            target.px.push(source.px.value(row));
            target.py.push(source.py.value(row));
            target.pz.push(source.pz.value(row));
            target.e.push(source.e.value(row));
        }
        for (target, source) in aux_columns_out.iter_mut().zip(&aux_columns) {
            target.push(source.value(row));
        }
        weights_out.push(
            weight_column
                .as_ref()
                .map(|column| column.value(row))
                .unwrap_or(1.0),
        );
    }
    Ok(())
}

fn read_parquet_columnar_local(
    builder: ParquetRecordBatchReaderBuilder<File>,
    metadata: Arc<DatasetMetadata>,
) -> LadduResult<Arc<DatasetStorage>> {
    let total_rows = builder.metadata().file_metadata().num_rows() as usize;
    if total_rows == 0 {
        return Ok(Arc::new(empty_dataset_columnar(metadata)));
    }

    let reader = builder.build()?;
    let mut p4 = (0..metadata.p4_names.len())
        .map(|_| ColumnarP4Column::with_capacity(total_rows))
        .collect::<Vec<_>>();
    let mut aux = (0..metadata.aux_names.len())
        .map(|_| Vec::with_capacity(total_rows))
        .collect::<Vec<_>>();
    let mut weights = Vec::with_capacity(total_rows);

    append_record_batch_stream(reader, metadata.as_ref(), &mut p4, &mut aux, &mut weights)?;

    Ok(Arc::new(DatasetStorage {
        metadata,
        p4,
        aux,
        weights,
    }))
}

#[cfg(feature = "mpi")]
fn read_parquet_columnar_mpi(
    builder: ParquetRecordBatchReaderBuilder<File>,
    metadata: Arc<DatasetMetadata>,
    world: &SimpleCommunicator,
) -> LadduResult<Arc<DatasetStorage>> {
    let plan = configure_parquet_mpi_read(builder, world);
    if plan.local_rows == 0 {
        return Ok(Arc::new(empty_dataset_columnar(metadata)));
    }

    let reader = plan.builder.build()?;
    let mut p4 = (0..metadata.p4_names.len())
        .map(|_| ColumnarP4Column::with_capacity(plan.local_rows + plan.skip_rows))
        .collect::<Vec<_>>();
    let mut aux = (0..metadata.aux_names.len())
        .map(|_| Vec::with_capacity(plan.local_rows + plan.skip_rows))
        .collect::<Vec<_>>();
    let mut weights = Vec::with_capacity(plan.local_rows + plan.skip_rows);
    append_record_batch_stream(reader, metadata.as_ref(), &mut p4, &mut aux, &mut weights)?;
    let mut columnar = DatasetStorage {
        metadata,
        p4,
        aux,
        weights,
    };

    trim_columnar_rows(&mut columnar, plan.skip_rows, plan.local_rows);
    if columnar.n_events() != plan.local_rows {
        return Err(LadduError::LengthMismatch {
            context: format!("Loaded rows for MPI rank {}", plan.rank),
            expected: plan.local_rows,
            actual: columnar.n_events(),
        });
    }

    Ok(Arc::new(columnar))
}

#[cfg(feature = "mpi")]
struct ParquetMpiReadPlan {
    builder: ParquetRecordBatchReaderBuilder<File>,
    skip_rows: usize,
    local_rows: usize,
    rank: usize,
}

#[cfg(feature = "mpi")]
fn configure_parquet_mpi_read(
    mut builder: ParquetRecordBatchReaderBuilder<File>,
    world: &SimpleCommunicator,
) -> ParquetMpiReadPlan {
    let parquet_metadata = builder.metadata().clone();
    let total_rows = parquet_metadata.file_metadata().num_rows() as usize;
    let rank = world.rank() as usize;
    if total_rows == 0 {
        return ParquetMpiReadPlan {
            builder,
            skip_rows: 0,
            local_rows: 0,
            rank,
        };
    }

    let partition = world.partition(total_rows);
    let local_range = partition.range_for_rank(rank);
    let local_rows = local_range.len();
    if local_rows == 0 {
        return ParquetMpiReadPlan {
            builder,
            skip_rows: 0,
            local_rows: 0,
            rank,
        };
    }

    let (row_groups, first_row_start) =
        row_groups_for_range(&parquet_metadata, local_range.start, local_range.end);
    if !row_groups.is_empty() {
        builder = builder.with_row_groups(row_groups);
    }

    ParquetMpiReadPlan {
        builder,
        skip_rows: local_range.start.saturating_sub(first_row_start),
        local_rows,
        rank,
    }
}

#[cfg(feature = "mpi")]
fn row_groups_for_range(
    metadata: &Arc<ParquetMetaData>,
    start: usize,
    end: usize,
) -> (Vec<usize>, usize) {
    let mut selected = Vec::new();
    let mut first_row_start = start;
    let mut offset = 0usize;
    for (idx, row_group) in metadata.row_groups().iter().enumerate() {
        let group_start = offset;
        let rows = row_group.num_rows() as usize;
        let group_end = group_start + rows;
        offset = group_end;
        if group_end <= start {
            continue;
        }
        if group_start >= end {
            break;
        }
        if selected.is_empty() {
            first_row_start = group_start;
        }
        selected.push(idx);
        if group_end >= end {
            break;
        }
    }
    (selected, first_row_start)
}

fn empty_dataset_columnar(metadata: Arc<DatasetMetadata>) -> DatasetStorage {
    DatasetStorage {
        p4: (0..metadata.p4_names.len())
            .map(|_| ColumnarP4Column::with_capacity(0))
            .collect(),
        aux: (0..metadata.aux_names.len())
            .map(|_| Vec::with_capacity(0))
            .collect(),
        weights: Vec::new(),
        metadata,
    }
}

fn append_record_batch_to_columnar(
    batch: &RecordBatch,
    metadata: &DatasetMetadata,
    p4_columns_out: &mut [ColumnarP4Column],
    aux_columns_out: &mut [Vec<f64>],
    weights_out: &mut Vec<f64>,
) -> LadduResult<()> {
    let p4_columns: Vec<P4Columns<'_>> = metadata
        .p4_names
        .iter()
        .map(|name| prepare_p4_columns(batch, name))
        .collect::<Result<_, _>>()?;
    let aux_columns: Vec<FloatColumn<'_>> = metadata
        .aux_names
        .iter()
        .map(|name| prepare_float_column(batch, name))
        .collect::<Result<_, _>>()?;
    let weight_column = find_float_column_from_candidates(batch, &["weight".to_string()])?;

    for row in 0..batch.num_rows() {
        for (target, source) in p4_columns_out.iter_mut().zip(&p4_columns) {
            target.px.push(source.px.value(row));
            target.py.push(source.py.value(row));
            target.pz.push(source.pz.value(row));
            target.e.push(source.e.value(row));
        }
        for (target, source) in aux_columns_out.iter_mut().zip(&aux_columns) {
            target.push(source.value(row));
        }
        weights_out.push(
            weight_column
                .as_ref()
                .map(|column| column.value(row))
                .unwrap_or(1.0),
        );
    }

    Ok(())
}

fn append_record_batch_stream<I, E>(
    reader: I,
    metadata: &DatasetMetadata,
    p4_columns_out: &mut [ColumnarP4Column],
    aux_columns_out: &mut [Vec<f64>],
    weights_out: &mut Vec<f64>,
) -> LadduResult<()>
where
    I: IntoIterator<Item = Result<RecordBatch, E>>,
    E: Into<LadduError>,
{
    for batch in reader {
        let batch = batch.map_err(Into::into)?;
        append_record_batch_to_columnar(
            &batch,
            metadata,
            p4_columns_out,
            aux_columns_out,
            weights_out,
        )?;
    }
    Ok(())
}

#[cfg(feature = "mpi")]
fn trim_columnar_rows(columnar: &mut DatasetStorage, drop_front: usize, expected_len: usize) {
    if drop_front > 0 {
        for column in &mut columnar.p4 {
            column.px.drain(0..drop_front);
            column.py.drain(0..drop_front);
            column.pz.drain(0..drop_front);
            column.e.drain(0..drop_front);
        }
        for column in &mut columnar.aux {
            column.drain(0..drop_front);
        }
        columnar.weights.drain(0..drop_front);
    }

    if columnar.n_events() > expected_len {
        for column in &mut columnar.p4 {
            column.px.truncate(expected_len);
            column.py.truncate(expected_len);
            column.pz.truncate(expected_len);
            column.e.truncate(expected_len);
        }
        for column in &mut columnar.aux {
            column.truncate(expected_len);
        }
        columnar.weights.truncate(expected_len);
    }
}

/// Load a [`Dataset`] from a ROOT TTree using the oxyroot backend.
pub fn read_root(file_path: &str, options: &DatasetReadOptions) -> LadduResult<Arc<Dataset>> {
    let storage = read_root_storage(file_path, options)?;
    Ok(Arc::new(storage.to_dataset()))
}

pub(crate) fn read_root_storage(
    file_path: &str,
    options: &DatasetReadOptions,
) -> LadduResult<Arc<DatasetStorage>> {
    let root_data = read_root_columns(file_path, options)?;
    let p4 = root_data
        .p4_columns
        .into_iter()
        .map(|columns| ColumnarP4Column {
            px: columns.px,
            py: columns.py,
            pz: columns.pz,
            e: columns.e,
        })
        .collect::<Vec<_>>();
    Ok(Arc::new(DatasetStorage {
        metadata: root_data.metadata,
        p4,
        aux: root_data.aux_columns,
        weights: root_data.weight_values,
    }))
}

struct RootP4Columns {
    px: Vec<f64>,
    py: Vec<f64>,
    pz: Vec<f64>,
    e: Vec<f64>,
}

struct RootReadColumns {
    metadata: Arc<DatasetMetadata>,
    p4_columns: Vec<RootP4Columns>,
    aux_columns: Vec<Vec<f64>>,
    weight_values: Vec<f64>,
}

#[cfg(feature = "mpi")]
fn root_entry_range(total_entries: usize, world: &SimpleCommunicator) -> Range<usize> {
    let partition = world.partition(total_entries);
    partition.range_for_rank(world.rank() as usize)
}

fn read_root_columns(
    file_path: &str,
    options: &DatasetReadOptions,
) -> LadduResult<RootReadColumns> {
    let path = canonicalize_dataset_path(file_path)?;
    let mut file = RootFile::open(&path).map_err(|err| {
        LadduError::Custom(format!(
            "Failed to open ROOT file '{}': {err}",
            path.display()
        ))
    })?;

    let (tree, tree_name) = resolve_root_tree(&mut file, options.tree.as_deref())?;
    let branches: Vec<&Branch> = tree.branches().collect();
    let mut lookup: BranchLookup<'_> = IndexMap::new();
    for &branch in &branches {
        if let Some(kind) = branch_scalar_kind(branch) {
            lookup.insert(branch.name(), (kind, branch));
        }
    }
    if lookup.is_empty() {
        return Err(LadduError::Custom(format!(
            "No float or double branches found in ROOT tree '{tree_name}'"
        )));
    }

    let column_names: Vec<&str> = lookup.keys().copied().collect();
    let (detected_p4_names, detected_aux_names) = infer_p4_and_aux_names(&column_names);
    let metadata = options.resolve_metadata(detected_p4_names, detected_aux_names)?;
    let total_entries = tree.entries().max(0) as usize;
    let entry_range = {
        #[cfg(feature = "mpi")]
        {
            if let Some(world) = crate::mpi::get_world() {
                root_entry_range(total_entries, &world)
            } else {
                0..total_entries
            }
        }
        #[cfg(not(feature = "mpi"))]
        {
            0..total_entries
        }
    };
    let expected_entries = entry_range.len();
    if total_entries == 0 || expected_entries == 0 {
        return Ok(RootReadColumns {
            metadata,
            p4_columns: Vec::new(),
            aux_columns: Vec::new(),
            weight_values: Vec::new(),
        });
    }

    // Keep ROOT reads sequential for now: oxyroot branch handles are not `Sync`,
    // so sharing this lookup across worker threads is unsound.
    let p4_columns = metadata
        .p4_names
        .iter()
        .map(|name| {
            let px = read_branch_values_from_candidates(
                &lookup,
                &component_candidates(name, "px"),
                &format!("{name}_px"),
                &entry_range,
            )?;
            let py = read_branch_values_from_candidates(
                &lookup,
                &component_candidates(name, "py"),
                &format!("{name}_py"),
                &entry_range,
            )?;
            let pz = read_branch_values_from_candidates(
                &lookup,
                &component_candidates(name, "pz"),
                &format!("{name}_pz"),
                &entry_range,
            )?;
            let e = read_branch_values_from_candidates(
                &lookup,
                &component_candidates(name, "e"),
                &format!("{name}_e"),
                &entry_range,
            )?;
            Ok(RootP4Columns { px, py, pz, e })
        })
        .collect::<LadduResult<Vec<_>>>()?;

    let aux_columns = metadata
        .aux_names
        .iter()
        .map(|name| read_branch_values(&lookup, name, &entry_range))
        .collect::<LadduResult<Vec<_>>>()?;

    if p4_columns.is_empty() && aux_columns.is_empty() {
        return Err(LadduError::Custom(
            "Unable to determine event count; dataset has no four-momentum or auxiliary columns"
                .to_string(),
        ));
    }

    let weight_values = match read_branch_values_optional(&lookup, "weight", &entry_range)? {
        Some(values) => {
            if values.len() != expected_entries {
                return Err(LadduError::LengthMismatch {
                    context: "Column 'weight'".to_string(),
                    expected: expected_entries,
                    actual: values.len(),
                });
            }
            values
        }
        None => vec![1.0; expected_entries],
    };

    for (name, columns) in metadata.p4_names.iter().zip(p4_columns.iter()) {
        for (component, values) in [
            ("px", &columns.px),
            ("py", &columns.py),
            ("pz", &columns.pz),
            ("e", &columns.e),
        ] {
            if values.len() != expected_entries {
                return Err(LadduError::LengthMismatch {
                    context: format!("Column '{name}_{component}'"),
                    expected: expected_entries,
                    actual: values.len(),
                });
            }
        }
    }
    for (name, values) in metadata.aux_names.iter().zip(aux_columns.iter()) {
        if values.len() != expected_entries {
            return Err(LadduError::LengthMismatch {
                context: format!("Column '{name}'"),
                expected: expected_entries,
                actual: values.len(),
            });
        }
    }

    Ok(RootReadColumns {
        metadata,
        p4_columns,
        aux_columns,
        weight_values,
    })
}

/// Persist a [`Dataset`] to a Parquet file.
pub fn write_parquet(
    dataset: &Dataset,
    file_path: &str,
    options: &DatasetWriteOptions,
) -> LadduResult<()> {
    let path = expand_output_path(file_path)?;
    dataset.write_parquet_impl(path, options)
}

#[cfg(test)]
pub(crate) fn write_parquet_storage(
    dataset: &DatasetStorage,
    file_path: &str,
    options: &DatasetWriteOptions,
) -> LadduResult<()> {
    let path = expand_output_path(file_path)?;
    dataset.write_parquet_impl(path, options)
}

/// Persist a [`Dataset`] to a ROOT file using the oxyroot backend.
pub fn write_root(
    dataset: &Dataset,
    file_path: &str,
    options: &DatasetWriteOptions,
) -> LadduResult<()> {
    let path = expand_output_path(file_path)?;
    dataset.write_root_impl(path, options)
}

#[cfg(feature = "mpi")]
#[cfg(test)]
pub(crate) fn write_root_with_chunk_size_for_test(
    dataset: &Dataset,
    file_path: &str,
    options: &DatasetWriteOptions,
    fetch_chunk_size: usize,
) -> LadduResult<()> {
    let path = expand_output_path(file_path)?;
    dataset.write_root_impl_with_chunk_size(path, options, Some(fetch_chunk_size.max(1)))
}

impl DatasetStorage {
    pub(super) fn write_parquet_impl(
        &self,
        file_path: PathBuf,
        options: &DatasetWriteOptions,
    ) -> LadduResult<()> {
        let batch_size = options.batch_size.max(1);
        let precision = options.precision;
        let schema = Arc::new(build_parquet_schema(&self.metadata, precision));
        let file = File::create(&file_path)?;
        let mut writer = ArrowWriter::try_new(file, schema.clone(), None)
            .map_err(|err| LadduError::Custom(format!("Failed to create Parquet writer: {err}")))?;

        let n_rows = self.n_events();
        let mut start = 0usize;
        while start < n_rows {
            let end = (start + batch_size).min(n_rows);
            let batch = columnar_range_to_record_batch(self, start, end, schema.clone(), precision)
                .map_err(|err| {
                    LadduError::Custom(format!("Failed to build Parquet batch: {err}"))
                })?;
            writer.write(&batch).map_err(|err| {
                LadduError::Custom(format!("Failed to write Parquet batch: {err}"))
            })?;
            start = end;
        }

        writer
            .close()
            .map_err(|err| LadduError::Custom(format!("Failed to finalise Parquet file: {err}")))?;

        Ok(())
    }
}

impl Dataset {
    pub(super) fn write_parquet_impl(
        &self,
        file_path: PathBuf,
        options: &DatasetWriteOptions,
    ) -> LadduResult<()> {
        #[cfg(feature = "mpi")]
        {
            if let Some(world) = crate::mpi::get_world() {
                let is_root = world.rank() == crate::mpi::ROOT_RANK;
                let batch_size = options.batch_size.max(1);
                let precision = options.precision;
                let schema = Arc::new(build_parquet_schema(&self.metadata, precision));

                let mut local_counts = vec![0_i32; world.size() as usize];
                let local_n_events = self.n_events_local() as i32;
                world.all_gather_into(&local_n_events, &mut local_counts);
                let mut writer = if is_root {
                    let file = File::create(&file_path)?;
                    Some(
                        ArrowWriter::try_new(file, schema.clone(), None).map_err(|err| {
                            LadduError::Custom(format!("Failed to create Parquet writer: {err}"))
                        })?,
                    )
                } else {
                    None
                };

                for (source_rank, source_count) in local_counts.iter().enumerate() {
                    let source_total_rows = *source_count as usize;
                    let mut source_start = 0usize;
                    while source_start < source_total_rows {
                        let source_end = (source_start + batch_size).min(source_total_rows);
                        let source_chunk_rows = source_end - source_start;
                        let local_chunk_rows = if world.rank() as usize == source_rank {
                            source_chunk_rows as i32
                        } else {
                            0
                        };
                        let mut chunk_counts = vec![0_i32; world.size() as usize];
                        world.all_gather_into(&local_chunk_rows, &mut chunk_counts);
                        let mut chunk_displs = vec![0_i32; chunk_counts.len()];
                        for i in 1..chunk_displs.len() {
                            chunk_displs[i] = chunk_displs[i - 1] + chunk_counts[i - 1];
                        }
                        let mut gathered_p4 = Vec::with_capacity(self.columnar.p4.len());
                        for p4 in &self.columnar.p4 {
                            let px_local = if world.rank() as usize == source_rank {
                                &p4.px[source_start..source_end]
                            } else {
                                &[]
                            };
                            let py_local = if world.rank() as usize == source_rank {
                                &p4.py[source_start..source_end]
                            } else {
                                &[]
                            };
                            let pz_local = if world.rank() as usize == source_rank {
                                &p4.pz[source_start..source_end]
                            } else {
                                &[]
                            };
                            let e_local = if world.rank() as usize == source_rank {
                                &p4.e[source_start..source_end]
                            } else {
                                &[]
                            };
                            gathered_p4.push(ColumnarP4Column {
                                px: world.all_gather_with_counts(
                                    px_local,
                                    &chunk_counts,
                                    &chunk_displs,
                                ),
                                py: world.all_gather_with_counts(
                                    py_local,
                                    &chunk_counts,
                                    &chunk_displs,
                                ),
                                pz: world.all_gather_with_counts(
                                    pz_local,
                                    &chunk_counts,
                                    &chunk_displs,
                                ),
                                e: world.all_gather_with_counts(
                                    e_local,
                                    &chunk_counts,
                                    &chunk_displs,
                                ),
                            });
                        }

                        let mut gathered_aux = Vec::with_capacity(self.columnar.aux.len());
                        for aux_column in &self.columnar.aux {
                            let aux_local = if world.rank() as usize == source_rank {
                                &aux_column[source_start..source_end]
                            } else {
                                &[]
                            };
                            gathered_aux.push(world.all_gather_with_counts(
                                aux_local,
                                &chunk_counts,
                                &chunk_displs,
                            ));
                        }

                        let weights_local = if world.rank() as usize == source_rank {
                            &self.columnar.weights[source_start..source_end]
                        } else {
                            &[]
                        };
                        let gathered_weights = world.all_gather_with_counts(
                            weights_local,
                            &chunk_counts,
                            &chunk_displs,
                        );

                        if is_root {
                            let chunk = DatasetStorage {
                                metadata: self.metadata.clone(),
                                p4: gathered_p4,
                                aux: gathered_aux,
                                weights: gathered_weights,
                            };
                            let batch = columnar_range_to_record_batch(
                                &chunk,
                                0,
                                chunk.n_events(),
                                schema.clone(),
                                precision,
                            )
                            .map_err(|err| {
                                LadduError::Custom(format!("Failed to build Parquet batch: {err}"))
                            })?;
                            if let Some(ref mut active_writer) = writer {
                                active_writer.write(&batch).map_err(|err| {
                                    LadduError::Custom(format!(
                                        "Failed to write Parquet batch: {err}"
                                    ))
                                })?;
                            }
                        }

                        source_start = source_end;
                    }
                }

                if let Some(active_writer) = writer {
                    active_writer.close().map_err(|err| {
                        LadduError::Custom(format!("Failed to finalise Parquet file: {err}"))
                    })?;
                }
                return Ok(());
            }
        }
        self.columnar.write_parquet_impl(file_path, options)
    }

    pub(super) fn write_root_impl(
        &self,
        file_path: PathBuf,
        options: &DatasetWriteOptions,
    ) -> LadduResult<()> {
        self.write_root_impl_with_chunk_size(file_path, options, None)
    }

    fn write_root_impl_with_chunk_size(
        &self,
        file_path: PathBuf,
        options: &DatasetWriteOptions,
        fetch_chunk_size: Option<usize>,
    ) -> LadduResult<()> {
        let tree_name = options.tree.clone().unwrap_or_else(|| "events".to_string());

        #[cfg(feature = "mpi")]
        let mut world_opt = crate::mpi::get_world();
        #[cfg(feature = "mpi")]
        let is_root = world_opt.as_ref().is_none_or(|world| world.rank() == 0);
        #[cfg(not(feature = "mpi"))]
        let is_root = true;

        #[cfg(feature = "mpi")]
        let world: Option<WorldHandle> = world_opt.take();
        #[cfg(not(feature = "mpi"))]
        let world: Option<WorldHandle> = None;

        let total_events = self.n_events();
        let dataset_arc = Arc::new(self.clone());

        match options.precision {
            FloatPrecision::F64 => self.write_root_with_type::<f64>(
                dataset_arc,
                world,
                is_root,
                &file_path,
                &tree_name,
                total_events,
                fetch_chunk_size,
            ),
            FloatPrecision::F32 => self.write_root_with_type::<f32>(
                dataset_arc,
                world,
                is_root,
                &file_path,
                &tree_name,
                total_events,
                fetch_chunk_size,
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn write_root_with_type<T>(
        &self,
        dataset: Arc<Dataset>,
        world: Option<WorldHandle>,
        is_root: bool,
        file_path: &Path,
        tree_name: &str,
        total_events: usize,
        fetch_chunk_size: Option<usize>,
    ) -> LadduResult<()>
    where
        T: RootWriteValue,
    {
        if world.is_none() {
            let columns = build_root_local_column_buffers::<T>(&self.columnar);
            return write_root_file_from_local_columns(columns, file_path, tree_name);
        }

        #[cfg(feature = "mpi")]
        {
            write_root_with_type_mpi::<T>(
                dataset,
                world.expect("MPI world should exist for distributed ROOT writing"),
                is_root,
                file_path,
                tree_name,
                total_events,
                fetch_chunk_size,
            )
        }

        #[cfg(not(feature = "mpi"))]
        {
            let _ = (dataset, is_root, total_events, fetch_chunk_size);
            Ok(())
        }
    }
}

#[cfg(feature = "mpi")]
fn write_root_with_type_mpi<T>(
    dataset: Arc<Dataset>,
    world: WorldHandle,
    is_root: bool,
    file_path: &Path,
    tree_name: &str,
    total_events: usize,
    fetch_chunk_size: Option<usize>,
) -> LadduResult<()>
where
    T: RootWriteValue,
{
    let mut iterators =
        build_root_column_iterators::<T>(dataset, world, total_events, fetch_chunk_size);

    if is_root {
        let mut file = RootFile::create(file_path).map_err(|err| {
            LadduError::Custom(format!(
                "Failed to create ROOT file '{}': {err}",
                file_path.display()
            ))
        })?;

        let mut tree = WriterTree::new(tree_name);
        for (name, iterator) in iterators {
            tree.new_branch(name, iterator);
        }

        tree.write(&mut file).map_err(|err| {
            LadduError::Custom(format!(
                "Failed to write ROOT tree '{tree_name}' to '{}': {err}",
                file_path.display()
            ))
        })?;

        file.close().map_err(|err| {
            LadduError::Custom(format!(
                "Failed to close ROOT file '{}': {err}",
                file_path.display()
            ))
        })?;
    } else {
        drain_column_iterators(&mut iterators, total_events);
    }

    Ok(())
}

/// Canonical four-momentum component suffixes used for column discovery.
pub const P4_COMPONENT_SUFFIXES: [&str; 4] = ["_px", "_py", "_pz", "_e"];

#[cfg(feature = "python")]
/// Infer p4 and auxiliary names from a list of floating-point column names.
pub fn infer_p4_and_aux_names_from_columns(column_names: &[String]) -> (Vec<String>, Vec<String>) {
    let cols = column_names.iter().map(String::as_str).collect::<Vec<_>>();
    infer_p4_and_aux_names(&cols)
}

#[cfg(feature = "python")]
/// Resolve logical column names against available columns using case-insensitive matching.
pub fn resolve_columns_case_insensitive(
    column_names: &[String],
    logical_names: &[String],
) -> LadduResult<Vec<String>> {
    logical_names
        .iter()
        .map(|name| {
            resolve_column_name_case_insensitive(column_names, name)
                .ok_or_else(|| LadduError::MissingColumn { name: name.clone() })
        })
        .collect()
}

#[cfg(feature = "python")]
/// Resolve p4 component columns (`_px`, `_py`, `_pz`, `_e`) for each requested p4 name.
pub fn resolve_p4_component_columns(
    column_names: &[String],
    p4_names: &[String],
) -> LadduResult<Vec<[String; 4]>> {
    p4_names
        .iter()
        .map(|name| {
            let mut components = [String::new(), String::new(), String::new(), String::new()];
            for (component_idx, suffix) in P4_COMPONENT_SUFFIXES.iter().enumerate() {
                let logical_name = format!("{name}{suffix}");
                components[component_idx] =
                    resolve_column_name_case_insensitive(column_names, &logical_name)
                        .ok_or(LadduError::MissingColumn { name: logical_name })?;
            }
            Ok(components)
        })
        .collect()
}

#[cfg(feature = "python")]
/// Resolve an optional `weight` column using case-insensitive matching.
pub fn resolve_optional_weight_column(column_names: &[String]) -> Option<String> {
    resolve_column_name_case_insensitive(column_names, "weight")
}

fn infer_p4_and_aux_names(float_cols: &[&str]) -> (Vec<String>, Vec<String>) {
    let suffix_set: IndexSet<&str> = P4_COMPONENT_SUFFIXES.iter().copied().collect();
    let mut groups: IndexMap<String, (String, IndexSet<&str>)> = IndexMap::new();
    for col in float_cols {
        if let Some((prefix_key, prefix_display, suffix)) = split_p4_component_case_insensitive(col)
        {
            groups
                .entry(prefix_key)
                .and_modify(|(_, present)| {
                    present.insert(suffix);
                })
                .or_insert_with(|| {
                    let mut present = IndexSet::new();
                    present.insert(suffix);
                    (prefix_display, present)
                });
        }
    }

    let mut p4_names: Vec<String> = Vec::new();
    let mut p4_prefixes: IndexSet<String> = IndexSet::new();
    for (prefix_key, (prefix_display, suffixes)) in &groups {
        if suffixes.len() == suffix_set.len() {
            p4_names.push(prefix_display.clone());
            p4_prefixes.insert(prefix_key.clone());
        }
    }

    let mut aux_names: Vec<String> = Vec::new();
    for col in float_cols {
        if col.eq_ignore_ascii_case("weight") {
            continue;
        }
        if let Some((prefix_key, _, _)) = split_p4_component_case_insensitive(col) {
            if p4_prefixes.contains(&prefix_key) {
                continue;
            }
        }
        aux_names.push((*col).to_string());
    }
    (p4_names, aux_names)
}

fn split_p4_component_case_insensitive(
    column_name: &str,
) -> Option<(String, String, &'static str)> {
    let lower = column_name.to_ascii_lowercase();
    for suffix in P4_COMPONENT_SUFFIXES {
        if lower.ends_with(suffix) && column_name.len() > suffix.len() {
            let prefix = column_name[..column_name.len() - suffix.len()].to_string();
            let key = prefix.to_ascii_lowercase();
            return Some((key, prefix, suffix));
        }
    }
    None
}

#[cfg(feature = "python")]
fn resolve_column_name_case_insensitive(
    column_names: &[String],
    logical_name: &str,
) -> Option<String> {
    if let Some(exact) = column_names
        .iter()
        .find(|name| name.as_str() == logical_name)
    {
        return Some(exact.clone());
    }
    column_names
        .iter()
        .find(|name| name.eq_ignore_ascii_case(logical_name))
        .cloned()
}

type BranchLookup<'a> = IndexMap<&'a str, (RootScalarKind, &'a Branch)>;

#[derive(Clone, Copy)]
enum RootScalarKind {
    F32,
    F64,
}

fn branch_scalar_kind(branch: &Branch) -> Option<RootScalarKind> {
    let type_name = branch.item_type_name();
    let lower = type_name.to_ascii_lowercase();
    if lower.contains("vector") {
        return None;
    }
    match lower.as_str() {
        "float" | "float_t" | "float32_t" => Some(RootScalarKind::F32),
        "double" | "double_t" | "double32_t" => Some(RootScalarKind::F64),
        _ => None,
    }
}

fn read_branch_values<'a>(
    lookup: &BranchLookup<'a>,
    column_name: &str,
    range: &Range<usize>,
) -> LadduResult<Vec<f64>> {
    let (kind, branch) =
        lookup
            .get(column_name)
            .copied()
            .ok_or_else(|| LadduError::MissingColumn {
                name: column_name.to_string(),
            })?;
    let total_entries = branch.entries().max(0) as usize;
    if range.end > total_entries {
        return Err(LadduError::LengthMismatch {
            context: format!("Column '{column_name}'"),
            expected: total_entries,
            actual: range.end,
        });
    }
    let values = match kind {
        RootScalarKind::F32 => branch
            .as_iter::<f32>()
            .map_err(|err| map_root_error(&format!("Failed to read branch '{column_name}'"), err))?
            .skip(range.start)
            .take(range.len())
            .map(|value| value as f64)
            .collect(),
        RootScalarKind::F64 => branch
            .as_iter::<f64>()
            .map_err(|err| map_root_error(&format!("Failed to read branch '{column_name}'"), err))?
            .skip(range.start)
            .take(range.len())
            .collect(),
    };
    Ok(values)
}

fn read_branch_values_optional<'a>(
    lookup: &BranchLookup<'a>,
    column_name: &str,
    range: &Range<usize>,
) -> LadduResult<Option<Vec<f64>>> {
    if lookup.contains_key(column_name) {
        read_branch_values(lookup, column_name, range).map(Some)
    } else {
        Ok(None)
    }
}

fn read_branch_values_from_candidates<'a>(
    lookup: &BranchLookup<'a>,
    candidates: &[String],
    logical_name: &str,
    range: &Range<usize>,
) -> LadduResult<Vec<f64>> {
    for candidate in candidates {
        if lookup.contains_key(candidate.as_str()) {
            return read_branch_values(lookup, candidate, range);
        }
    }
    Err(LadduError::MissingColumn {
        name: logical_name.to_string(),
    })
}

fn resolve_root_tree(
    file: &mut RootFile,
    requested: Option<&str>,
) -> LadduResult<(ReaderTree, String)> {
    if let Some(name) = requested {
        let tree = file
            .get_tree(name)
            .map_err(|err| map_root_error(&format!("Failed to open ROOT tree '{name}'"), err))?;
        return Ok((tree, name.to_string()));
    }

    let tree_names: Vec<String> = file
        .keys()
        .into_iter()
        .filter(|key| key.class_name() == "TTree")
        .map(|key| key.name().to_string())
        .collect();

    if tree_names.is_empty() {
        return Err(LadduError::Custom(
            "ROOT file does not contain any TTrees".to_string(),
        ));
    }
    if tree_names.len() > 1 {
        return Err(LadduError::Custom(format!(
            "Multiple TTrees found ({:?}); specify DatasetReadOptions::tree to disambiguate",
            tree_names
        )));
    }

    let selected = &tree_names[0];
    let tree = file
        .get_tree(selected)
        .map_err(|err| map_root_error(&format!("Failed to open ROOT tree '{selected}'"), err))?;
    Ok((tree, selected.clone()))
}

fn map_root_error<E: std::fmt::Display>(context: &str, err: E) -> LadduError {
    LadduError::Custom(format!("{context}: {err}"))
}

#[derive(Clone, Copy)]
enum FloatColumn<'a> {
    F32(&'a Float32Array),
    F64(&'a Float64Array),
}

impl<'a> FloatColumn<'a> {
    fn value(&self, row: usize) -> f64 {
        match self {
            Self::F32(array) => array.value(row) as f64,
            Self::F64(array) => array.value(row),
        }
    }
}

struct P4Columns<'a> {
    px: FloatColumn<'a>,
    py: FloatColumn<'a>,
    pz: FloatColumn<'a>,
    e: FloatColumn<'a>,
}

fn prepare_float_column<'a>(batch: &'a RecordBatch, name: &str) -> LadduResult<FloatColumn<'a>> {
    prepare_float_column_from_candidates(batch, &[name.to_string()], name)
}

fn prepare_p4_columns<'a>(batch: &'a RecordBatch, name: &str) -> LadduResult<P4Columns<'a>> {
    Ok(P4Columns {
        px: prepare_float_column_from_candidates(
            batch,
            &component_candidates(name, "px"),
            &format!("{name}_px"),
        )?,
        py: prepare_float_column_from_candidates(
            batch,
            &component_candidates(name, "py"),
            &format!("{name}_py"),
        )?,
        pz: prepare_float_column_from_candidates(
            batch,
            &component_candidates(name, "pz"),
            &format!("{name}_pz"),
        )?,
        e: prepare_float_column_from_candidates(
            batch,
            &component_candidates(name, "e"),
            &format!("{name}_e"),
        )?,
    })
}

fn component_candidates(name: &str, suffix: &str) -> Vec<String> {
    let mut candidates = Vec::with_capacity(3);
    let base = format!("{name}_{suffix}");
    candidates.push(base.clone());

    let mut capitalized = suffix.to_string();
    if let Some(first) = capitalized.get_mut(0..1) {
        first.make_ascii_uppercase();
    }
    if capitalized != suffix {
        candidates.push(format!("{name}_{capitalized}"));
    }

    let upper = suffix.to_ascii_uppercase();
    if upper != suffix && upper != capitalized {
        candidates.push(format!("{name}_{upper}"));
    }
    candidates
}

fn find_float_column_from_candidates<'a>(
    batch: &'a RecordBatch,
    candidates: &[String],
) -> LadduResult<Option<FloatColumn<'a>>> {
    use arrow::datatypes::DataType;
    for candidate in candidates {
        if let Some(column) = batch.column_by_name(candidate) {
            return match column.data_type() {
                DataType::Float32 => Ok(Some(FloatColumn::F32(
                    column
                        .as_any()
                        .downcast_ref::<Float32Array>()
                        .expect("Column advertised as Float32 but could not be downcast"),
                ))),
                DataType::Float64 => Ok(Some(FloatColumn::F64(
                    column
                        .as_any()
                        .downcast_ref::<Float64Array>()
                        .expect("Column advertised as Float64 but could not be downcast"),
                ))),
                other => Err(LadduError::InvalidColumnType {
                    name: candidate.clone(),
                    datatype: other.to_string(),
                }),
            };
        }
    }
    Ok(None)
}

fn prepare_float_column_from_candidates<'a>(
    batch: &'a RecordBatch,
    candidates: &[String],
    logical_name: &str,
) -> LadduResult<FloatColumn<'a>> {
    find_float_column_from_candidates(batch, candidates)?.ok_or_else(|| LadduError::MissingColumn {
        name: logical_name.to_string(),
    })
}

fn columnar_range_to_record_batch(
    dataset: &DatasetStorage,
    start: usize,
    end: usize,
    schema: Arc<Schema>,
    precision: FloatPrecision,
) -> arrow::error::Result<RecordBatch> {
    let mut columns: Vec<arrow::array::ArrayRef> = Vec::new();
    match precision {
        FloatPrecision::F64 => {
            for p4 in &dataset.p4 {
                columns.push(Arc::new(Float64Array::from(p4.px[start..end].to_vec())));
                columns.push(Arc::new(Float64Array::from(p4.py[start..end].to_vec())));
                columns.push(Arc::new(Float64Array::from(p4.pz[start..end].to_vec())));
                columns.push(Arc::new(Float64Array::from(p4.e[start..end].to_vec())));
            }
            for aux in &dataset.aux {
                columns.push(Arc::new(Float64Array::from(aux[start..end].to_vec())));
            }
            columns.push(Arc::new(Float64Array::from(
                dataset.weights[start..end].to_vec(),
            )));
        }
        FloatPrecision::F32 => {
            for p4 in &dataset.p4 {
                columns.push(Arc::new(Float32Array::from(
                    p4.px[start..end]
                        .iter()
                        .map(|v| *v as f32)
                        .collect::<Vec<_>>(),
                )));
                columns.push(Arc::new(Float32Array::from(
                    p4.py[start..end]
                        .iter()
                        .map(|v| *v as f32)
                        .collect::<Vec<_>>(),
                )));
                columns.push(Arc::new(Float32Array::from(
                    p4.pz[start..end]
                        .iter()
                        .map(|v| *v as f32)
                        .collect::<Vec<_>>(),
                )));
                columns.push(Arc::new(Float32Array::from(
                    p4.e[start..end]
                        .iter()
                        .map(|v| *v as f32)
                        .collect::<Vec<_>>(),
                )));
            }
            for aux in &dataset.aux {
                columns.push(Arc::new(Float32Array::from(
                    aux[start..end]
                        .iter()
                        .map(|v| *v as f32)
                        .collect::<Vec<_>>(),
                )));
            }
            columns.push(Arc::new(Float32Array::from(
                dataset.weights[start..end]
                    .iter()
                    .map(|v| *v as f32)
                    .collect::<Vec<_>>(),
            )));
        }
    }
    RecordBatch::try_new(schema, columns)
}

fn build_parquet_schema(metadata: &DatasetMetadata, precision: FloatPrecision) -> Schema {
    let dtype = match precision {
        FloatPrecision::F64 => DataType::Float64,
        FloatPrecision::F32 => DataType::Float32,
    };
    let mut fields = Vec::new();
    for name in &metadata.p4_names {
        for suffix in P4_COMPONENT_SUFFIXES {
            fields.push(Field::new(format!("{name}{suffix}"), dtype.clone(), false));
        }
    }
    for name in &metadata.aux_names {
        fields.push(Field::new(name.clone(), dtype.clone(), false));
    }
    fields.push(Field::new("weight", dtype, false));
    Schema::new(fields)
}

pub(crate) trait FromF64 {
    fn from_f64(value: f64) -> Self;
}

impl FromF64 for f64 {
    fn from_f64(value: f64) -> Self {
        value
    }
}

impl FromF64 for f32 {
    fn from_f64(value: f64) -> Self {
        value as f32
    }
}

#[cfg(feature = "mpi")]
pub(super) trait RootWriteValue:
    FromF64 + oxyroot::Marshaler + Equivalence + Default + Clone + 'static
{
}

#[cfg(feature = "mpi")]
impl<T> RootWriteValue for T where
    T: FromF64 + oxyroot::Marshaler + Equivalence + Default + Clone + 'static
{
}

#[cfg(not(feature = "mpi"))]
pub(super) trait RootWriteValue: FromF64 + oxyroot::Marshaler + 'static {}

#[cfg(not(feature = "mpi"))]
impl<T> RootWriteValue for T where T: FromF64 + oxyroot::Marshaler + 'static {}

pub(crate) fn build_root_local_column_buffers<T>(storage: &DatasetStorage) -> Vec<(String, Vec<T>)>
where
    T: FromF64,
{
    let metadata = storage.metadata();
    let mut columns =
        Vec::with_capacity(metadata.p4_names().len() * 4 + metadata.aux_names().len() + 1);
    for (p4_index, name) in metadata.p4_names().iter().enumerate() {
        let p4 = &storage.p4[p4_index];
        columns.push((
            format!("{name}_px"),
            p4.px.iter().map(|value| T::from_f64(*value)).collect(),
        ));
        columns.push((
            format!("{name}_py"),
            p4.py.iter().map(|value| T::from_f64(*value)).collect(),
        ));
        columns.push((
            format!("{name}_pz"),
            p4.pz.iter().map(|value| T::from_f64(*value)).collect(),
        ));
        columns.push((
            format!("{name}_e"),
            p4.e.iter().map(|value| T::from_f64(*value)).collect(),
        ));
    }
    for (aux_index, name) in metadata.aux_names().iter().enumerate() {
        columns.push((
            name.clone(),
            storage.aux[aux_index]
                .iter()
                .map(|value| T::from_f64(*value))
                .collect(),
        ));
    }
    columns.push((
        "weight".to_string(),
        storage
            .weights
            .iter()
            .map(|value| T::from_f64(*value))
            .collect(),
    ));
    columns
}

fn write_root_file_from_local_columns<T>(
    columns: Vec<(String, Vec<T>)>,
    file_path: &Path,
    tree_name: &str,
) -> LadduResult<()>
where
    T: oxyroot::Marshaler + 'static,
{
    let mut file = RootFile::create(file_path).map_err(|err| {
        LadduError::Custom(format!(
            "Failed to create ROOT file '{}': {err}",
            file_path.display()
        ))
    })?;

    let mut tree = WriterTree::new(tree_name);
    for (name, values) in columns {
        tree.new_branch(name, values.into_iter());
    }

    tree.write(&mut file).map_err(|err| {
        LadduError::Custom(format!(
            "Failed to write ROOT tree '{tree_name}' to '{}': {err}",
            file_path.display()
        ))
    })?;

    file.close().map_err(|err| {
        LadduError::Custom(format!(
            "Failed to close ROOT file '{}': {err}",
            file_path.display()
        ))
    })?;

    Ok(())
}

#[cfg(feature = "mpi")]
struct SharedColumnChunkFetcher<T> {
    columns: Vec<(String, Vec<T>)>,
    world: WorldHandle,
    total: usize,
    chunk_size: usize,
    chunk_start: usize,
    chunk_end: usize,
    gathered_columns: Vec<Vec<T>>,
}

#[cfg(feature = "mpi")]
impl<T> SharedColumnChunkFetcher<T>
where
    T: Equivalence + Default + Clone,
{
    fn new(
        columns: Vec<(String, Vec<T>)>,
        world: WorldHandle,
        total: usize,
        fetch_chunk_size: Option<usize>,
    ) -> Self {
        let chunk_size = fetch_chunk_size
            .map(|size| size.max(1))
            .unwrap_or_else(|| resolve_mpi_event_fetch_chunk_size(total));
        Self {
            columns,
            world,
            total,
            chunk_size,
            chunk_start: 0,
            chunk_end: 0,
            gathered_columns: Vec::new(),
        }
    }

    fn value_for_index(&mut self, column_index: usize, global_index: usize) -> Option<T> {
        if global_index >= self.total {
            return None;
        }
        if global_index < self.chunk_start || global_index >= self.chunk_end {
            self.refresh_chunk(global_index);
        }
        self.gathered_columns
            .get(column_index)?
            .get(global_index - self.chunk_start)
            .cloned()
    }

    fn refresh_chunk(&mut self, chunk_start: usize) {
        let chunk_end = (chunk_start + self.chunk_size).min(self.total);
        let chunk_range = chunk_start..chunk_end;
        let partition = self.world.partition(self.total);
        let this_rank = self.world.rank() as usize;
        let this_range = partition.range_for_rank(this_rank);
        let local_start = chunk_range.start.max(this_range.start);
        let local_end = chunk_range.end.min(this_range.end);
        let local_slice = if local_start < local_end {
            Some((local_start - this_range.start)..(local_end - this_range.start))
        } else {
            None
        };

        let mut counts = Vec::with_capacity(partition.n_ranks());
        let mut displs = Vec::with_capacity(partition.n_ranks());
        let mut offset = 0_i32;
        for rank in 0..partition.n_ranks() {
            let rank_range = partition.range_for_rank(rank);
            let overlap_start = chunk_range.start.max(rank_range.start);
            let overlap_end = chunk_range.end.min(rank_range.end);
            let count = overlap_end.saturating_sub(overlap_start) as i32;
            counts.push(count);
            displs.push(offset);
            offset += count;
        }

        self.gathered_columns = self
            .columns
            .iter()
            .map(|(_, column)| {
                let local_values = local_slice
                    .as_ref()
                    .map(|indices| &column[indices.clone()])
                    .unwrap_or(&[]);
                self.world
                    .all_gather_with_counts(local_values, &counts, &displs)
            })
            .collect();
        self.chunk_start = chunk_start;
        self.chunk_end = chunk_end;
    }
}

#[cfg(feature = "mpi")]
struct ColumnIterator<T> {
    fetcher: Rc<RefCell<SharedColumnChunkFetcher<T>>>,
    index: usize,
    column_index: usize,
}

#[cfg(feature = "mpi")]
impl<T> ColumnIterator<T> {
    fn new(fetcher: Rc<RefCell<SharedColumnChunkFetcher<T>>>, column_index: usize) -> Self {
        Self {
            fetcher,
            index: 0,
            column_index,
        }
    }
}

#[cfg(feature = "mpi")]
impl<T> Iterator for ColumnIterator<T>
where
    T: Equivalence + Default + Clone,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let value = self
            .fetcher
            .borrow_mut()
            .value_for_index(self.column_index, self.index)?;
        self.index += 1;
        Some(value)
    }
}

#[cfg(feature = "mpi")]
fn build_root_column_iterators<T>(
    dataset: Arc<Dataset>,
    world: WorldHandle,
    total: usize,
    fetch_chunk_size: Option<usize>,
) -> Vec<(String, ColumnIterator<T>)>
where
    T: FromF64 + Equivalence + Default + Clone,
{
    let columns = build_root_local_column_buffers::<T>(&dataset.columnar);
    let fetcher = Rc::new(RefCell::new(SharedColumnChunkFetcher::new(
        columns,
        world,
        total,
        fetch_chunk_size,
    )));
    let column_names = fetcher
        .borrow()
        .columns
        .iter()
        .map(|(name, _)| name.clone())
        .collect::<Vec<_>>();
    column_names
        .into_iter()
        .enumerate()
        .map(|(column_index, name)| (name, ColumnIterator::new(fetcher.clone(), column_index)))
        .collect()
}

#[cfg(feature = "mpi")]
fn drain_column_iterators<T>(iterators: &mut [(String, ColumnIterator<T>)], n_events: usize)
where
    T: Equivalence + Default + Clone,
{
    for _ in 0..n_events {
        for (_name, iterator) in iterators.iter_mut() {
            let _ = iterator.next();
        }
    }
}
