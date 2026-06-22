use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

use crate::{LadduDataError, LadduDataResult, data::EventBatch, schema::Schema};

pub mod memory;
pub mod parquet;
pub mod root;

#[cfg(feature = "mpi")]
#[derive(Clone, Copy, Debug, Default)]
pub enum Distribution {
    #[default]
    Serial,
    Mpi {
        rank: usize,
        nranks: usize,
        partitioning: Partitioning,
    },
}

#[cfg(feature = "mpi")]
impl Distribution {
    pub fn serial() -> Self {
        Self::Serial
    }

    pub fn from_world<C>(world: &C) -> Self
    where
        C: mpi::topology::Communicator,
    {
        Self::Mpi {
            rank: world.rank() as usize,
            nranks: world.size() as usize,
            partitioning: Partitioning::default(),
        }
    }

    pub fn rank(self) -> usize {
        match self {
            Self::Serial => 0,
            Self::Mpi { rank, .. } => rank,
        }
    }

    pub fn nranks(self) -> usize {
        match self {
            Self::Serial => 1,
            Self::Mpi { nranks, .. } => nranks,
        }
    }

    pub fn partitioning(self) -> Partitioning {
        match self {
            Self::Serial => Partitioning::Contiguous,
            Self::Mpi { partitioning, .. } => partitioning,
        }
    }

    pub fn with_partitioning(self, partitioning: Partitioning) -> Self {
        match self {
            Self::Serial => Self::Serial,
            Self::Mpi { rank, nranks, .. } => Self::Mpi {
                rank,
                nranks,
                partitioning,
            },
        }
    }
}

#[cfg(feature = "mpi")]
#[derive(Clone, Copy, Debug, Default)]
pub enum Partitioning {
    /// Each rank reads a contiguous global row range.
    #[default]
    Contiguous,

    /// Each rank reads whole source fragments, such as files or row groups, round-robin.
    FileGroups,

    /// Rank r keeps rows where global_row % nranks == r.
    /// Deterministic, but usually slower.
    Rows,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ReadPlan {
    pub chunk_size: Option<usize>,

    #[cfg(feature = "mpi")]
    pub distribution: Distribution,
}

impl ReadPlan {
    pub fn serial() -> Self {
        Self::default()
    }

    pub fn rank(&self) -> usize {
        #[cfg(feature = "mpi")]
        {
            self.distribution.rank()
        }

        #[cfg(not(feature = "mpi"))]
        {
            0
        }
    }

    pub fn nranks(&self) -> usize {
        #[cfg(feature = "mpi")]
        {
            self.distribution.nranks()
        }

        #[cfg(not(feature = "mpi"))]
        {
            1
        }
    }

    pub fn is_distributed(&self) -> bool {
        self.nranks() > 1
    }

    pub fn fragment_partitioning(&self) -> FragmentPartitioning {
        #[cfg(feature = "mpi")]
        {
            match self.distribution.partitioning() {
                Partitioning::Contiguous => FragmentPartitioning::Contiguous,
                Partitioning::FileGroups => FragmentPartitioning::RoundRobinFragments,
                Partitioning::Rows => FragmentPartitioning::StridedRows,
            }
        }

        #[cfg(not(feature = "mpi"))]
        {
            FragmentPartitioning::Contiguous
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct WritePlan {
    #[cfg(feature = "mpi")]
    pub distribution: Distribution,
}

impl From<ReadPlan> for WritePlan {
    #[cfg_attr(not(feature = "mpi"), allow(unused_variables))]
    fn from(plan: ReadPlan) -> Self {
        Self {
            #[cfg(feature = "mpi")]
            distribution: plan.distribution,
        }
    }
}

impl WritePlan {
    pub fn rank(&self) -> usize {
        #[cfg(feature = "mpi")]
        {
            self.distribution.rank()
        }

        #[cfg(not(feature = "mpi"))]
        {
            0
        }
    }

    pub fn nranks(&self) -> usize {
        #[cfg(feature = "mpi")]
        {
            self.distribution.nranks()
        }

        #[cfg(not(feature = "mpi"))]
        {
            1
        }
    }

    pub fn is_distributed(&self) -> bool {
        self.nranks() > 1
    }
}

#[derive(Clone, Copy, Debug)]
pub enum FragmentPartitioning {
    Contiguous,
    RoundRobinFragments,
    StridedRows,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SourceCapabilities {
    pub exact_len: bool,
    pub exact_weighted_total: bool,
    pub random_access: bool,
    pub deterministic_partitioning: bool,
    pub predicate_pushdown: bool,
    pub projection_pushdown: bool,
    pub streaming: bool,
}

pub type EventBatchIter = Box<dyn Iterator<Item = LadduDataResult<EventBatch>> + Send>;

pub trait EventSource: Send + Sync {
    fn schema(&self) -> LadduDataResult<Arc<Schema>>;

    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities::default()
    }

    fn num_events(&self) -> LadduDataResult<Option<u64>> {
        Ok(None)
    }

    fn weighted_total(&self) -> LadduDataResult<Option<f64>> {
        Ok(None)
    }

    fn batches(&self, plan: ReadPlan) -> LadduDataResult<EventBatchIter>;
}

pub trait EventSink: Send {
    fn begin(&mut self, schema: Arc<Schema>, plan: WritePlan) -> LadduDataResult<()>;

    fn write_batch(&mut self, batch: &EventBatch) -> LadduDataResult<()>;

    fn finish(&mut self) -> LadduDataResult<()>;
}

#[derive(Clone, Debug)]
pub struct DataFragment<K> {
    pub key: K,
    pub global_start: u64,
    pub rows: u64,
}

#[derive(Clone, Debug)]
pub struct FragmentRead<K> {
    pub key: K,
    pub selection: FragmentSelection,
}

#[derive(Clone, Copy, Debug)]
pub enum FragmentSelection {
    Range {
        local_start: usize,
        local_len: usize,
    },
    StridedRows {
        global_start: u64,
        rows: usize,
        rank: usize,
        nranks: usize,
    },
}

pub trait FragmentedSource: Send + Sync {
    type Key: Clone + Send + Sync + 'static;

    fn fragments(&self) -> LadduDataResult<Vec<DataFragment<Self::Key>>>;

    fn read_fragment_range(
        &self,
        key: &Self::Key,
        local_start: usize,
        local_len: usize,
        chunk_size: Option<usize>,
    ) -> LadduDataResult<EventBatchIter>;
}

pub fn fragmented_batches<S>(source: Arc<S>, plan: ReadPlan) -> LadduDataResult<EventBatchIter>
where
    S: FragmentedSource + 'static,
{
    let iter = FragmentBatchIter::new(source, plan)?;

    if plan.chunk_size.is_none() {
        Ok(Box::new(CoalescedBatchIter::new(iter)))
    } else {
        Ok(Box::new(iter))
    }
}

pub fn plan_fragments<K: Clone>(
    fragments: &[DataFragment<K>],
    plan: ReadPlan,
) -> LadduDataResult<Vec<FragmentRead<K>>> {
    let total_rows: u64 = fragments.iter().map(|f| f.rows).sum();
    let rank = plan.rank();
    let nranks = plan.nranks();

    if nranks == 1 {
        return fragments
            .iter()
            .map(|f| {
                Ok(FragmentRead {
                    key: f.key.clone(),
                    selection: FragmentSelection::Range {
                        local_start: 0,
                        local_len: usize_from_u64(f.rows)?,
                    },
                })
            })
            .collect();
    }

    match plan.fragment_partitioning() {
        FragmentPartitioning::Contiguous => contiguous_plan(fragments, total_rows, rank, nranks),
        FragmentPartitioning::RoundRobinFragments => {
            round_robin_fragment_plan(fragments, rank, nranks)
        }
        FragmentPartitioning::StridedRows => strided_row_plan(fragments, rank, nranks),
    }
}

fn contiguous_plan<K: Clone>(
    fragments: &[DataFragment<K>],
    total_rows: u64,
    rank: usize,
    nranks: usize,
) -> LadduDataResult<Vec<FragmentRead<K>>> {
    let rank_start = total_rows * rank as u64 / nranks as u64;
    let rank_end = total_rows * (rank as u64 + 1) / nranks as u64;

    let mut out = Vec::new();

    for f in fragments {
        let frag_start = f.global_start;
        let frag_end = f.global_start + f.rows;

        let start = rank_start.max(frag_start);
        let end = rank_end.min(frag_end);

        if start < end {
            out.push(FragmentRead {
                key: f.key.clone(),
                selection: FragmentSelection::Range {
                    local_start: usize_from_u64(start - frag_start)?,
                    local_len: usize_from_u64(end - start)?,
                },
            });
        }
    }

    Ok(out)
}

fn round_robin_fragment_plan<K: Clone>(
    fragments: &[DataFragment<K>],
    rank: usize,
    nranks: usize,
) -> LadduDataResult<Vec<FragmentRead<K>>> {
    let mut out = Vec::new();

    for (i, f) in fragments.iter().enumerate() {
        if i % nranks == rank {
            out.push(FragmentRead {
                key: f.key.clone(),
                selection: FragmentSelection::Range {
                    local_start: 0,
                    local_len: usize_from_u64(f.rows)?,
                },
            });
        }
    }

    Ok(out)
}

fn strided_row_plan<K: Clone>(
    fragments: &[DataFragment<K>],
    rank: usize,
    nranks: usize,
) -> LadduDataResult<Vec<FragmentRead<K>>> {
    fragments
        .iter()
        .map(|f| {
            Ok(FragmentRead {
                key: f.key.clone(),
                selection: FragmentSelection::StridedRows {
                    global_start: f.global_start,
                    rows: usize_from_u64(f.rows)?,
                    rank,
                    nranks,
                },
            })
        })
        .collect()
}

fn usize_from_u64(value: u64) -> LadduDataResult<usize> {
    usize::try_from(value).map_err(|_| LadduDataError::InvalidArgument("row count exceeds usize"))
}

pub(crate) struct FragmentBatchIter<S>
where
    S: FragmentedSource,
{
    source: Arc<S>,
    reads: Vec<FragmentRead<S::Key>>,
    read_index: usize,
    current: Option<EventBatchIter>,
    chunk_size: Option<usize>,
}

impl<S> FragmentBatchIter<S>
where
    S: FragmentedSource,
{
    pub(crate) fn new(source: Arc<S>, plan: ReadPlan) -> LadduDataResult<Self> {
        let fragments = source.fragments()?;
        let reads = plan_fragments(&fragments, plan)?;

        Ok(Self {
            source,
            reads,
            read_index: 0,
            current: None,
            chunk_size: plan.chunk_size,
        })
    }
}

impl<S> Iterator for FragmentBatchIter<S>
where
    S: FragmentedSource,
{
    type Item = LadduDataResult<EventBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(iter) = self.current.as_mut() {
                match iter.next() {
                    Some(batch) => return Some(batch),
                    None => self.current = None,
                }
            }

            let read = self.reads.get(self.read_index)?.clone();
            self.read_index += 1;

            let next_iter = match read.selection {
                FragmentSelection::Range {
                    local_start,
                    local_len,
                } => self.source.read_fragment_range(
                    &read.key,
                    local_start,
                    local_len,
                    self.chunk_size,
                ),

                FragmentSelection::StridedRows {
                    global_start,
                    rows,
                    rank,
                    nranks,
                } => {
                    let inner =
                        self.source
                            .read_fragment_range(&read.key, 0, rows, self.chunk_size);

                    inner.and_then(|iter| {
                        let iter = StridedRowsBatchIter::new(iter, global_start, rank, nranks)?;
                        Ok(Box::new(iter) as EventBatchIter)
                    })
                }
            };

            match next_iter {
                Ok(iter) => self.current = Some(iter),
                Err(err) => return Some(Err(err)),
            }
        }
    }
}

pub(crate) struct SliceBatchIter<I> {
    inner: I,
    start: usize,
    end: usize,
    consumed: usize,
}

impl<I> SliceBatchIter<I> {
    pub(crate) fn new(inner: I, start: usize, len: usize) -> LadduDataResult<Self> {
        let end = start
            .checked_add(len)
            .ok_or(LadduDataError::InvalidArgument(
                "slice range overflows usize",
            ))?;
        Ok(Self {
            inner,
            start,
            end,
            consumed: 0,
        })
    }
}

impl<I> Iterator for SliceBatchIter<I>
where
    I: Iterator<Item = LadduDataResult<EventBatch>>,
{
    type Item = LadduDataResult<EventBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        while self.consumed < self.end {
            let batch = match self.inner.next()? {
                Ok(batch) => batch,
                Err(err) => return Some(Err(err)),
            };

            let batch_start = self.consumed;
            let batch_end = batch_start + batch.len();
            self.consumed = batch_end;

            let lo = self.start.max(batch_start);
            let hi = self.end.min(batch_end);

            if lo >= hi {
                continue;
            }

            let local_lo = lo - batch_start;
            let local_hi = hi - batch_start;

            return Some(Ok(batch.slice(local_lo, local_hi)));
        }

        None
    }
}

pub(crate) struct StridedRowsBatchIter<I> {
    inner: I,
    global_start: u64,
    consumed: u64,
    rank: usize,
    nranks: usize,
}

impl<I> StridedRowsBatchIter<I> {
    pub(crate) fn new(
        inner: I,
        global_start: u64,
        rank: usize,
        nranks: usize,
    ) -> LadduDataResult<Self> {
        if nranks == 0 {
            return Err(LadduDataError::InvalidArgument("nranks must be nonzero"));
        }
        if rank >= nranks {
            return Err(LadduDataError::InvalidArgument(
                "rank must be less than nranks",
            ));
        }
        Ok(Self {
            inner,
            global_start,
            consumed: 0,
            rank,
            nranks,
        })
    }
}

impl<I> Iterator for StridedRowsBatchIter<I>
where
    I: Iterator<Item = LadduDataResult<EventBatch>>,
{
    type Item = LadduDataResult<EventBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let batch = match self.inner.next()? {
                Ok(batch) => batch,
                Err(err) => return Some(Err(err)),
            };

            let batch_global_start = self.global_start + self.consumed;
            self.consumed = self.consumed.saturating_add(batch.len() as u64);

            let rows: Vec<usize> = (0..batch.len())
                .filter(|&i| {
                    ((batch_global_start + i as u64) % self.nranks as u64) == self.rank as u64
                })
                .collect();

            if rows.is_empty() {
                continue;
            }

            return Some(Ok(batch.select(&rows)));
        }
    }
}

pub(crate) struct CoalescedBatchIter<I> {
    inner: Option<I>,
    emitted: bool,
}

impl<I> CoalescedBatchIter<I> {
    pub(crate) fn new(inner: I) -> Self {
        Self {
            inner: Some(inner),
            emitted: false,
        }
    }
}

impl<I> Iterator for CoalescedBatchIter<I>
where
    I: Iterator<Item = LadduDataResult<EventBatch>>,
{
    type Item = LadduDataResult<EventBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.emitted {
            return None;
        }

        self.emitted = true;

        let inner = self.inner.as_mut()?;
        let mut batches = Vec::new();

        for batch in inner {
            match batch {
                Ok(batch) => batches.push(batch),
                Err(err) => return Some(Err(err)),
            }
        }

        if batches.is_empty() {
            None
        } else {
            Some(EventBatch::concat(&batches))
        }
    }
}

#[derive(Clone, Debug)]
pub struct OutputPath {
    base: PathBuf,
    mode: OutputMode,
}

#[derive(Clone, Copy, Debug, Default)]
pub enum OutputMode {
    #[default]
    Auto,
    SingleFile,
    PerRankFiles,
}

impl OutputPath {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            base: path.into(),
            mode: OutputMode::Auto,
        }
    }

    pub fn with_mode(mut self, mode: OutputMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn base(&self) -> &Path {
        &self.base
    }

    pub fn mode(&self) -> OutputMode {
        self.mode
    }

    pub fn resolve(&self, plan: WritePlan, default_extension: &str) -> LadduDataResult<PathBuf> {
        let mode = match self.mode {
            OutputMode::Auto if plan.is_distributed() => OutputMode::PerRankFiles,
            OutputMode::Auto => OutputMode::SingleFile,
            mode => mode,
        };

        match mode {
            OutputMode::SingleFile => {
                if plan.is_distributed() {
                    return Err(LadduDataError::Sink(
                        "single-file output is unsafe with multiple MPI ranks; use per-rank output"
                            .into(),
                    ));
                }

                Ok(self.base.clone())
            }

            OutputMode::PerRankFiles => Ok(per_rank_path(
                &self.base,
                plan.rank(),
                plan.nranks(),
                default_extension,
            )),

            OutputMode::Auto => unreachable!(),
        }
    }

    pub fn create_parent_dirs(path: &Path) -> LadduDataResult<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|e| LadduDataError::Sink(e.to_string()))?;
            }
        }

        Ok(())
    }
}

fn per_rank_path(base: &Path, rank: usize, nranks: usize, default_extension: &str) -> PathBuf {
    if base.extension().is_none() {
        let ext = default_extension.trim_start_matches('.');
        return base.join(format!("part-rank{rank:05}-of{nranks:05}.{ext}"));
    }

    let parent = base.parent().unwrap_or_else(|| Path::new(""));
    let stem = base.file_stem().unwrap_or_default().to_string_lossy();
    let ext = base.extension().unwrap_or_default().to_string_lossy();

    parent.join(format!("{stem}.rank{rank:05}-of{nranks:05}.{ext}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        data::{EventBatch, EventBatchBuilder},
        schema::Schema,
    };

    fn v(x: f64) -> Vec4 {
        Vec4 {
            x: x,
            y: x,
            z: x,
            t: x,
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
                .push_weighted([v(i as f64)], [i as f64], 100.0 + i as f64)
                .unwrap();
        }

        builder.finish().unwrap()
    }

    fn concat_values(batches: Vec<EventBatch>) -> Vec<f64> {
        EventBatch::concat(&batches)
            .unwrap()
            .scalar_column(0)
            .to_vec()
    }

    #[test]
    fn slice_batch_iter_slices_across_batch_boundaries_without_losing_alignment() {
        let inner = vec![Ok(batch(0, 3)), Ok(batch(3, 2)), Ok(batch(5, 4))].into_iter();

        let out: Vec<EventBatch> = SliceBatchIter::new(inner, 2, 5)
            .unwrap()
            .map(Result::unwrap)
            .collect();

        let values = concat_values(out);
        assert_eq!(values, vec![2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn strided_rows_batch_iter_uses_global_row_numbers_across_batches() {
        let inner = vec![Ok(batch(0, 4)), Ok(batch(4, 5))].into_iter();

        let out: Vec<EventBatch> = StridedRowsBatchIter::new(inner, 1, 1, 3)
            .unwrap()
            .map(Result::unwrap)
            .collect();

        // Global rows are 1..=9 because global_start = 1.
        // Rank 1 of 3 keeps global rows 1, 4, 7.
        // Those correspond to local scalar ids 0, 3, 6.
        assert_eq!(concat_values(out), vec![0.0, 3.0, 6.0]);
    }

    #[test]
    fn coalesced_batch_iter_concatenates_successes_and_propagates_first_error() {
        let success_inner = vec![Ok(batch(0, 2)), Ok(batch(2, 3))].into_iter();
        let mut success = CoalescedBatchIter::new(success_inner);

        let merged = success.next().unwrap().unwrap();
        assert_eq!(merged.scalar_column(0), &[0.0, 1.0, 2.0, 3.0, 4.0]);
        assert!(success.next().is_none());

        let error_inner = vec![
            Ok(batch(0, 1)),
            Err(LadduDataError::Source("boom".into())),
            Ok(batch(1, 1)),
        ]
        .into_iter();

        let err = CoalescedBatchIter::new(error_inner)
            .next()
            .unwrap()
            .unwrap_err();

        assert!(matches!(err, LadduDataError::Source(msg) if msg == "boom"));
    }

    #[test]
    fn output_path_resolves_single_file_and_per_rank_names() {
        let plan = WritePlan::default();

        let single = OutputPath::new(PathBuf::from("events.parquet"))
            .resolve(plan, "parquet")
            .unwrap();

        assert_eq!(single, PathBuf::from("events.parquet"));

        let per_rank_with_extension = OutputPath::new(PathBuf::from("events.parquet"))
            .with_mode(OutputMode::PerRankFiles)
            .resolve(plan, "parquet")
            .unwrap();

        assert_eq!(
            per_rank_with_extension,
            PathBuf::from("events.rank00000-of00001.parquet")
        );

        let per_rank_without_extension = OutputPath::new(PathBuf::from("events"))
            .with_mode(OutputMode::PerRankFiles)
            .resolve(plan, "root")
            .unwrap();

        assert_eq!(
            per_rank_without_extension,
            PathBuf::from("events").join("part-rank00000-of00001.root")
        );
    }

    #[test]
    fn plan_fragments_serial_mode_keeps_all_fragments_in_order() {
        let fragments = vec![
            DataFragment {
                key: "a",
                global_start: 0,
                rows: 2,
            },
            DataFragment {
                key: "b",
                global_start: 2,
                rows: 3,
            },
        ];

        let reads = plan_fragments(&fragments, ReadPlan::default()).unwrap();

        assert_eq!(reads.len(), 2);

        match &reads[0].selection {
            FragmentSelection::Range {
                local_start,
                local_len,
            } => {
                assert_eq!((*local_start, *local_len), (0, 2));
            }
            _ => panic!("expected range read"),
        }

        match &reads[1].selection {
            FragmentSelection::Range {
                local_start,
                local_len,
            } => {
                assert_eq!((*local_start, *local_len), (0, 3));
            }
            _ => panic!("expected range read"),
        }
    }

    use laddu_physics::vectors::Vec4;
    #[cfg(feature = "mpi")]
    use mpi::traits::*;
    #[cfg(feature = "mpi")]
    use mpi_test::mpi_test;

    #[cfg(feature = "mpi")]
    fn distributed_plan(
        partitioning: Partitioning,
        world: &impl mpi::topology::Communicator,
    ) -> ReadPlan {
        ReadPlan {
            chunk_size: None,
            distribution: Distribution::from_world(world).with_partitioning(partitioning),
        }
    }

    #[cfg(feature = "mpi")]
    fn expected_contiguous_global_range(total_rows: u64, rank: usize, nranks: usize) -> (u64, u64) {
        let start = total_rows * rank as u64 / nranks as u64;
        let end = total_rows * (rank as u64 + 1) / nranks as u64;
        (start, end)
    }

    #[cfg(feature = "mpi")]
    #[mpi_test(np = [2, 3, 4])]
    fn mpi_contiguous_plan_assigns_disjoint_ranges_covering_all_rows() {
        let universe = mpi::initialize().unwrap();
        let world = universe.world();

        let rank = world.rank() as usize;
        let nranks = world.size() as usize;

        let fragments = vec![
            DataFragment {
                key: "a",
                global_start: 0,
                rows: 4,
            },
            DataFragment {
                key: "b",
                global_start: 4,
                rows: 5,
            },
            DataFragment {
                key: "c",
                global_start: 9,
                rows: 3,
            },
        ];

        let total_rows = fragments.iter().map(|f| f.rows).sum::<u64>();
        let plan = distributed_plan(Partitioning::Contiguous, &world);
        let reads = plan_fragments(&fragments, plan).unwrap();

        let assigned_rows: u64 = reads
            .iter()
            .map(|read| match read.selection {
                FragmentSelection::Range { local_len, .. } => local_len as u64,
                FragmentSelection::StridedRows { .. } => panic!("expected range selection"),
            })
            .sum();

        let (expected_start, expected_end) =
            expected_contiguous_global_range(total_rows, rank, nranks);

        assert_eq!(assigned_rows, expected_end - expected_start);

        for read in reads {
            let fragment = fragments
                .iter()
                .find(|fragment| fragment.key == read.key)
                .unwrap();

            match read.selection {
                FragmentSelection::Range {
                    local_start,
                    local_len,
                } => {
                    let global_start = fragment.global_start + local_start as u64;
                    let global_end = global_start + local_len as u64;

                    assert!(expected_start <= global_start);
                    assert!(global_end <= expected_end);
                    assert!(fragment.global_start <= global_start);
                    assert!(global_end <= fragment.global_start + fragment.rows);
                }
                FragmentSelection::StridedRows { .. } => panic!("expected range selection"),
            }
        }
    }

    #[cfg(feature = "mpi")]
    #[mpi_test(np = [2, 3])]
    fn mpi_file_group_plan_assigns_fragment_by_rank_round_robin() {
        let universe = mpi::initialize().unwrap();
        let world = universe.world();

        let rank = world.rank() as usize;
        let nranks = world.size() as usize;

        let fragments = (0..8)
            .map(|i| DataFragment {
                key: i,
                global_start: 10 * i as u64,
                rows: 10,
            })
            .collect::<Vec<_>>();

        let plan = distributed_plan(Partitioning::FileGroups, &world);
        let reads = plan_fragments(&fragments, plan).unwrap();

        let keys = reads.iter().map(|read| read.key).collect::<Vec<_>>();
        let expected = (0..8).filter(|i| i % nranks == rank).collect::<Vec<_>>();

        assert_eq!(keys, expected);

        for read in reads {
            match read.selection {
                FragmentSelection::Range {
                    local_start,
                    local_len,
                } => {
                    assert_eq!(local_start, 0);
                    assert_eq!(local_len, 10);
                }
                FragmentSelection::StridedRows { .. } => panic!("expected range selection"),
            }
        }
    }

    #[cfg(feature = "mpi")]
    #[mpi_test(np = [2, 3, 4])]
    fn mpi_rows_plan_assigns_strided_row_selection_with_world_rank() {
        let universe = mpi::initialize().unwrap();
        let world = universe.world();

        let rank = world.rank() as usize;
        let nranks = world.size() as usize;

        let fragments = vec![
            DataFragment {
                key: "a",
                global_start: 0,
                rows: 4,
            },
            DataFragment {
                key: "b",
                global_start: 4,
                rows: 5,
            },
        ];

        let plan = distributed_plan(Partitioning::Rows, &world);
        let reads = plan_fragments(&fragments, plan).unwrap();

        assert_eq!(reads.len(), fragments.len());

        for (read, fragment) in reads.iter().zip(fragments.iter()) {
            assert_eq!(read.key, fragment.key);

            match read.selection {
                FragmentSelection::StridedRows {
                    global_start,
                    rows,
                    rank: selected_rank,
                    nranks: selected_nranks,
                } => {
                    assert_eq!(global_start, fragment.global_start);
                    assert_eq!(rows, fragment.rows as usize);
                    assert_eq!(selected_rank, rank);
                    assert_eq!(selected_nranks, nranks);
                }
                FragmentSelection::Range { .. } => panic!("expected strided selection"),
            }
        }
    }

    #[cfg(feature = "mpi")]
    #[mpi_test(np = [2, 3])]
    fn mpi_read_plan_and_write_plan_reflect_world_distribution() {
        let universe = mpi::initialize().unwrap();
        let world = universe.world();

        let read_plan = ReadPlan {
            chunk_size: Some(7),
            distribution: Distribution::from_world(&world).with_partitioning(Partitioning::Rows),
        };

        assert!(read_plan.is_distributed());
        assert_eq!(read_plan.rank(), world.rank() as usize);
        assert_eq!(read_plan.nranks(), world.size() as usize);

        match read_plan.fragment_partitioning() {
            FragmentPartitioning::StridedRows => {}
            _ => panic!("expected strided row partitioning"),
        }

        let write_plan = WritePlan::from(read_plan);

        assert!(write_plan.is_distributed());
        assert_eq!(write_plan.rank(), world.rank() as usize);
        assert_eq!(write_plan.nranks(), world.size() as usize);
    }
}
