//! # laddu-core
//!
//! This is an internal crate used by `laddu`.
#![warn(clippy::perf, clippy::style, missing_docs)]
#![allow(clippy::excessive_precision)]
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]

use ganesh::core::{MCMCSummary, MinimizationSummary};
#[cfg(feature = "python")]
use pyo3::PyErr;

/// Re-exported alias for `std::f64` to ease dependent crates transitioning to the 64-bit
/// floating point API.
pub use std::f64;

/// MPI backend for `laddu`
///
/// Message Passing Interface (MPI) is a protocol which enables communication between multiple
/// CPUs in a high-performance computing environment. While [`rayon`] can parallelize tasks on a
/// single CPU, MPI can also parallelize tasks on multiple CPUs by running independent
/// processes on all CPUs at once (tasks) which are assigned ids (ranks) which tell each
/// process what to do and where to send results. This backend coordinates processes which would
/// typically be parallelized over the events in a [`Dataset`](`crate::data::Dataset`).
///
/// To use this backend, the library must be built with the `mpi` feature, which requires an
/// existing implementation of MPI like OpenMPI or MPICH. All processing code should be
/// sandwiched between calls to [`use_mpi`] and [`finalize_mpi`]:
/// ```ignore
/// fn main() {
///     laddu_core::mpi::use_mpi(true);
///     // laddu analysis code here
///     laddu_core::mpi::finalize_mpi();
/// }
/// ```
///
/// [`finalize_mpi`] must be called to trigger all the methods which clean up the MPI
/// environment. While these are called by default when the [`Universe`](`mpi::environment::Universe`) is dropped, `laddu` uses a static `Universe` that can be accessed by all of the methods that need it, rather than passing the context to each method. This simplifies the way programs can be converted to use MPI, but means that the `Universe` is not automatically dropped at the end of the program (so it must be dropped manually).
#[cfg(feature = "mpi")]
#[cfg_attr(coverage_nightly, coverage(off))]
pub mod mpi {
    use std::ops::Range;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::OnceLock;

    use lazy_static::lazy_static;
    use mpi::datatype::PartitionMut;
    use mpi::environment::Universe;
    use mpi::topology::{Process, SimpleCommunicator};
    use mpi::traits::{Communicator, CommunicatorCollectives, Equivalence};
    use parking_lot::RwLock;

    lazy_static! {
        static ref USE_MPI: AtomicBool = AtomicBool::new(false);
    }

    static MPI_UNIVERSE: OnceLock<RwLock<Option<Universe>>> = OnceLock::new();

    /// The default root rank for MPI processes
    pub const ROOT_RANK: i32 = 0;

    /// Check if the current MPI process is the root process
    pub fn is_root() -> bool {
        if let Some(world) = crate::mpi::get_world() {
            world.rank() == ROOT_RANK
        } else {
            true
        }
    }

    /// Shortcut method to just get the global MPI communicator without accessing `size` and `rank`
    /// directly
    pub fn get_world() -> Option<SimpleCommunicator> {
        if let Some(universe_lock) = MPI_UNIVERSE.get() {
            if let Some(universe) = &*universe_lock.read() {
                return Some(universe.world());
            }
        }
        None
    }

    /// Get the rank of the current process
    pub fn get_rank() -> i32 {
        get_world().map(|w| w.rank()).unwrap_or(ROOT_RANK)
    }

    /// Get number of available processes/ranks
    pub fn get_size() -> i32 {
        get_world().map(|w| w.size()).unwrap_or(1)
    }

    /// Use the MPI backend
    ///
    /// # Notes
    ///
    /// You must have MPI installed for this to work, and you must call the program with
    /// `mpirun <executable>`, or bad things will happen.
    ///
    /// MPI runs an identical program on each process, but gives the program an ID called its
    /// "rank". Only the results of methods on the root process (rank 0) should be
    /// considered valid, as other processes only contain portions of each dataset. To ensure
    /// you don't save or print data at other ranks, use the provided [`is_root()`]
    /// method to check if the process is the root process.
    ///
    /// Once MPI is enabled, it cannot be disabled. If MPI could be toggled (which it can't),
    /// the other processes will still run, but they will be independent of the root process
    /// and will no longer communicate with it. The root process stores no data, so it would
    /// be difficult (and convoluted) to get the results which were already processed via
    /// MPI.
    ///
    /// Additionally, MPI must be enabled at the beginning of a script, at least before any
    /// other `laddu` functions are called.
    ///
    /// If [`use_mpi()`] is called multiple times, the subsequent calls will have no
    /// effect.
    ///
    /// <div class="warning">
    ///
    /// You **must** call [`finalize_mpi()`] before your program exits for MPI to terminate
    /// smoothly.
    ///
    /// </div>
    ///
    /// # Examples
    ///
    /// ```ignore
    /// fn main() {
    ///     laddu_core::use_mpi();
    ///
    ///     // ... your code here ...
    ///
    ///     laddu_core::finalize_mpi();
    /// }
    ///
    /// ```
    pub fn use_mpi(trigger: bool) {
        if trigger {
            USE_MPI.store(true, Ordering::SeqCst);
            MPI_UNIVERSE.get_or_init(|| {
                #[cfg(feature = "rayon")]
                let threading = mpi::Threading::Funneled;
                #[cfg(not(feature = "rayon"))]
                let threading = mpi::Threading::Single;
                let (universe, _threading) = mpi::initialize_with_threading(threading).unwrap();
                let world = universe.world();
                if world.size() == 1 {
                    eprintln!("Warning: MPI is enabled, but only one process is available. MPI will not be used, but single-CPU parallelism may still be used if enabled.");
                    finalize_mpi();
                    USE_MPI.store(false, Ordering::SeqCst);
                    RwLock::new(None)
                } else {
                    RwLock::new(Some(universe))
                }
            });
        }
    }

    /// Drop the MPI universe and finalize MPI at the end of a program
    ///
    /// This function will do nothing if MPI is not initialized.
    ///
    /// <div class="warning">
    ///
    /// This should only be called once and should be called at the end of all `laddu`-related
    /// function calls. This must be called at the end of any program which uses MPI.
    ///
    /// </div>
    pub fn finalize_mpi() {
        if using_mpi() {
            let mut universe = MPI_UNIVERSE.get().unwrap().write();
            *universe = None;
        }
    }

    /// Check if MPI backend is enabled
    pub fn using_mpi() -> bool {
        USE_MPI.load(Ordering::SeqCst)
    }

    fn counts_displs(size: usize, total: usize, stride: usize) -> (Vec<i32>, Vec<i32>) {
        let mut counts = vec![0i32; size];
        let mut displs = vec![0i32; size];
        if size == 0 {
            return (counts, displs);
        }
        let base = total / size;
        let remainder = total % size;
        let mut offset = 0i32;
        for rank in 0..size {
            let n = if rank < remainder { base + 1 } else { base };
            let scaled = (n * stride) as i32;
            counts[rank] = scaled;
            displs[rank] = offset;
            offset += scaled;
        }
        (counts, displs)
    }

    #[inline]
    fn rank_local_from_global(i_global: usize, size: usize, total: usize) -> (usize, usize) {
        assert!(size > 0, "Communicator must have at least one rank");
        assert!(total > 0, "Cannot map global indices when dataset is empty");
        assert!(
            i_global < total,
            "Global index {} out of bounds for {} events",
            i_global,
            total
        );
        let base = total / size;
        let remainder = total % size;
        let big_block = base + 1;
        let threshold = remainder * big_block;
        if i_global < threshold {
            let rank = i_global / big_block;
            let local = i_global % big_block;
            (rank, local)
        } else {
            let adjusted = i_global - threshold;
            let rank = remainder + adjusted / base;
            let local = adjusted % base;
            (rank, local)
        }
    }

    /// Canonical partitioning information for distributing a dataset across MPI ranks.
    #[derive(Clone, Debug)]
    pub struct Partition {
        counts: Vec<i32>,
        displs: Vec<i32>,
        total: usize,
    }

    impl Partition {
        /// Build a new distribution for `total` items across `size` ranks.
        pub fn new(size: usize, total: usize) -> Self {
            assert!(size > 0, "Communicator must have at least one rank");
            let (counts, displs) = counts_displs(size, total, 1);
            Self {
                counts,
                displs,
                total,
            }
        }

        /// Total number of items tracked by this partition.
        pub fn total(&self) -> usize {
            self.total
        }

        /// Number of ranks described by this partition.
        pub fn n_ranks(&self) -> usize {
            self.counts.len()
        }

        /// Number of items assigned to `rank`.
        pub fn len_for_rank(&self, rank: usize) -> usize {
            self.counts[rank] as usize
        }

        /// Starting global index for `rank`.
        pub fn start_for_rank(&self, rank: usize) -> usize {
            self.displs[rank] as usize
        }

        /// Contiguous global range owned by `rank`.
        pub fn range_for_rank(&self, rank: usize) -> Range<usize> {
            let start = self.start_for_rank(rank);
            start..start + self.len_for_rank(rank)
        }

        /// Determine the owning rank and local index for a global dataset index.
        pub fn owner_of(&self, global_index: usize) -> (usize, usize) {
            assert!(
                self.total > 0,
                "Cannot map global indices when dataset is empty"
            );
            rank_local_from_global(global_index, self.n_ranks(), self.total)
        }

        /// Convert into raw `(counts, displacements)` buffers.
        pub fn into_raw(self) -> (Vec<i32>, Vec<i32>) {
            (self.counts, self.displs)
        }
    }

    /// A trait including some useful auxiliary methods for MPI
    pub trait LadduMPI {
        /// Get the process at the root rank
        fn process_at_root(&self) -> Process<'_>;
        /// Check if the current rank is the root rank
        fn is_root(&self) -> bool;
        /// Gather arbitrarily-sized local slices into a buffer ordered by the
        /// canonical dataset partition.
        fn all_gather_partitioned<T: Equivalence + Default + Clone>(
            &self,
            local: &[T],
            total: usize,
            stride: Option<usize>,
        ) -> Vec<T>;
        /// Gather batches corresponding to arbitrary global indices while
        /// preserving the order of `global_indices`.
        fn all_gather_batched_partitioned<T: Equivalence + Default + Clone>(
            &self,
            local: &[T],
            global_indices: &[usize],
            total: usize,
            stride: Option<usize>,
        ) -> Vec<T>;
        /// Return the `(rank, local_index)` pair owning `global_index` in a
        /// dataset containing `total` events.
        fn owner_of_global_index(&self, global_index: usize, total: usize) -> (i32, usize);
        /// Translate a list of global dataset indices into the corresponding
        /// local indices owned by this rank, preserving their original order.
        fn locals_from_globals(&self, global_indices: &[usize], total: usize) -> Vec<usize>;
        /// Get the counts/displacements for partitioning a buffer of length
        /// `buf_len`
        fn get_counts_displs(&self, buf_len: usize) -> (Vec<i32>, Vec<i32>);
        /// Build a [`Partition`] describing how `total` items are distributed
        /// across ranks.
        fn partition(&self, total: usize) -> Partition;
        /// Get the counts/displacements for partitioning a nested buffer (like
        /// a [`Vec<Vec<T>>`]). If the internal vectors all have the same length
        /// `internal_len` and there are `unflattened_len` elements in the
        /// outer vector, then this will give the correct counts/displacements for a
        /// flattened version of the nested buffer.
        fn get_flattened_counts_displs(
            &self,
            unflattened_len: usize,
            internal_len: usize,
        ) -> (Vec<i32>, Vec<i32>);
    }

    impl LadduMPI for SimpleCommunicator {
        fn process_at_root(&self) -> Process<'_> {
            self.process_at_rank(crate::mpi::ROOT_RANK)
        }

        fn is_root(&self) -> bool {
            self.rank() == crate::mpi::ROOT_RANK
        }

        /// Gather arbitrarily-sized local slices into a buffer ordered by the
        /// canonical dataset partition.
        fn all_gather_partitioned<T: Equivalence + Default + Clone>(
            &self,
            local: &[T],
            total: usize,
            stride: Option<usize>,
        ) -> Vec<T> {
            let size = self.size() as usize;
            let stride = stride.unwrap_or(1);
            assert!(stride > 0, "Stride must be greater than zero");
            let mut out = vec![T::default(); total * stride];
            if total == 0 || size == 0 {
                return out;
            }
            let (counts, displs) = counts_displs(size, total, stride);
            {
                let mut partition = PartitionMut::new(&mut out, counts, displs);
                self.all_gather_varcount_into(local, &mut partition);
            }
            out
        }

        /// Gather batches corresponding to arbitrary global indices while
        /// preserving the order of `global_indices`.
        fn all_gather_batched_partitioned<T: Equivalence + Default + Clone>(
            &self,
            local: &[T],
            global_indices: &[usize],
            total: usize,
            stride: Option<usize>,
        ) -> Vec<T> {
            let size = self.size() as usize;
            let stride = stride.unwrap_or(1);
            assert!(stride > 0, "Stride must be greater than zero");
            let n_indices = global_indices.len();
            let mut gathered = vec![T::default(); n_indices * stride];
            if n_indices == 0 || size == 0 {
                return gathered;
            }

            assert!(
                total > 0,
                "Cannot gather batched data from an empty dataset"
            );

            let partition = Partition::new(size, total);
            let mut locals_by_rank = vec![Vec::<usize>::new(); size];
            let mut targets_by_rank = vec![Vec::<usize>::new(); size];
            for (position, &global_index) in global_indices.iter().enumerate() {
                let (rank, local_index) = partition.owner_of(global_index);
                locals_by_rank[rank].push(local_index);
                targets_by_rank[rank].push(position);
            }

            let mut counts = vec![0i32; size];
            let mut displs = vec![0i32; size];
            for rank in 0..size {
                counts[rank] = (locals_by_rank[rank].len() * stride) as i32;
                displs[rank] = if rank == 0 {
                    0
                } else {
                    displs[rank - 1] + counts[rank - 1]
                };
            }

            let expected_local = locals_by_rank[self.rank() as usize].len() * stride;
            debug_assert_eq!(
                local.len(),
                expected_local,
                "Local buffer length does not match expected gathered size for rank {}",
                self.rank()
            );

            {
                let mut partition =
                    PartitionMut::new(&mut gathered, counts.clone(), displs.clone());
                self.all_gather_varcount_into(local, &mut partition);
            }

            let mut result = vec![T::default(); n_indices * stride];
            for rank in 0..size {
                let mut cursor = displs[rank] as usize;
                for &target in &targets_by_rank[rank] {
                    let dst = target * stride;
                    for offset in 0..stride {
                        result[dst + offset] = gathered[cursor + offset].clone();
                    }
                    cursor += stride;
                }
            }

            result
        }

        fn owner_of_global_index(&self, global_index: usize, total: usize) -> (i32, usize) {
            let partition = Partition::new(self.size() as usize, total);
            let (rank, local) = partition.owner_of(global_index);
            (rank as i32, local)
        }

        /// Translate a list of global dataset indices into the corresponding
        /// local indices owned by this rank, preserving their original order.
        fn locals_from_globals(&self, global_indices: &[usize], total: usize) -> Vec<usize> {
            let partition = Partition::new(self.size() as usize, total);
            let this_rank = self.rank() as usize;
            let mut locals = Vec::new();
            if total == 0 {
                return locals;
            }
            for &global_index in global_indices {
                let (rank, local_index) = partition.owner_of(global_index);
                if rank == this_rank {
                    locals.push(local_index);
                }
            }
            locals
        }
        fn get_counts_displs(&self, buf_len: usize) -> (Vec<i32>, Vec<i32>) {
            self.partition(buf_len).into_raw()
        }

        fn partition(&self, total: usize) -> Partition {
            Partition::new(self.size() as usize, total)
        }

        fn get_flattened_counts_displs(
            &self,
            unflattened_len: usize,
            internal_len: usize,
        ) -> (Vec<i32>, Vec<i32>) {
            let mut counts = vec![0; self.size() as usize];
            let mut displs = vec![0; self.size() as usize];
            let chunk_size = unflattened_len / self.size() as usize;
            let surplus = unflattened_len % self.size() as usize;
            for i in 0..self.size() as usize {
                counts[i] = if i < surplus {
                    (chunk_size + 1) * internal_len
                } else {
                    chunk_size * internal_len
                } as i32;
                displs[i] = if i == 0 {
                    0
                } else {
                    displs[i - 1] + counts[i - 1]
                };
            }
            (counts, displs)
        }
    }
}

use thiserror::Error;

/// [`Amplitude`](crate::amplitudes::Amplitude)s and methods for making and evaluating them.
pub mod amplitudes;
/// Methods for loading and manipulating [`EventData`]-based data.
pub mod data;
/// Utilities for tracking parameter state across expressions and likelihoods.
pub mod parameter_manager;
/// Structures for manipulating the cache and free parameters.
pub mod resources;
/// Utility functions, enums, and traits
pub mod utils;
/// Useful traits for all crate structs
pub mod traits {
    pub use crate::amplitudes::Amplitude;
    pub use crate::utils::variables::Variable;
    pub use crate::ReadWrite;
}

pub use crate::data::{
    BinnedDataset, Dataset, DatasetMetadata, DatasetReadOptions, Event, EventData,
};
pub use crate::resources::{
    Cache, ComplexMatrixID, ComplexScalarID, ComplexVectorID, MatrixID, ParameterID, Parameters,
    Resources, ScalarID, VectorID,
};
pub use crate::utils::enums::{Channel, Frame, Sign};
pub use crate::utils::variables::{
    Angles, CosTheta, Mandelstam, Mass, Phi, PolAngle, PolMagnitude, Polarization,
};
pub use crate::utils::vectors::{Vec3, Vec4};
pub use amplitudes::{constant, parameter, AmplitudeID, Evaluator, Expression, ParameterLike};

/// The mathematical constant $`\pi`$.
pub const PI: f64 = std::f64::consts::PI;

/// A [`Result`] type alias for [`LadduError`]s.
pub type LadduResult<T> = Result<T, LadduError>;

/// The error type used by all `laddu` internal methods
#[derive(Error, Debug)]
pub enum LadduError {
    /// An alias for [`std::io::Error`].
    #[error(transparent)]
    IOError(#[from] std::io::Error),
    /// An alias for [`parquet::errors::ParquetError`].
    #[error(transparent)]
    ParquetError(#[from] parquet::errors::ParquetError),
    /// An alias for [`arrow::error::ArrowError`].
    #[error(transparent)]
    ArrowError(#[from] arrow::error::ArrowError),
    /// An alias for [`shellexpand::LookupError`].
    #[error(transparent)]
    LookupError(#[from] shellexpand::LookupError<std::env::VarError>),
    /// An error which occurs when the user tries to register two amplitudes by the same name.
    #[error("An amplitude by the name \"{name}\" is already registered!")]
    RegistrationError {
        /// Name of amplitude which is already registered
        name: String,
    },
    /// An error which occurs when the user tries to use an unregistered amplitude.
    #[error("No registered amplitude with name \"{name}\"!")]
    AmplitudeNotFoundError {
        /// Name of amplitude which failed lookup
        name: String,
    },
    /// An error which occurs when the user tries to parse an invalid string of text, typically
    /// into an enum variant.
    #[error("Failed to parse string: \"{name}\" does not correspond to a valid \"{object}\"!")]
    ParseError {
        /// The string which was parsed
        name: String,
        /// The name of the object it failed to parse into
        object: String,
    },
    /// An error returned by the Rust encoder
    #[error(transparent)]
    EncodeError(#[from] bincode::error::EncodeError),
    /// An error returned by the Rust decoder
    #[error(transparent)]
    DecodeError(#[from] bincode::error::DecodeError),
    /// An error returned by the Python pickle (de)serializer
    #[error(transparent)]
    PickleError(#[from] serde_pickle::Error),
    /// An error which occurs when parameter definitions conflict or clash.
    #[error("Parameter \"{name}\" conflict: {reason}")]
    ParameterConflict {
        /// Name of parameter
        name: String,
        /// Description of conflict
        reason: String,
    },
    /// An error which occurs when attempting to use an unregistered or unnamed parameter.
    #[error("Parameter \"{name}\" could not be registered: {reason}")]
    UnregisteredParameter {
        /// Name of parameter
        name: String,
        /// Reason for failure
        reason: String,
    },
    /// An error type for [`rayon`] thread pools
    #[cfg(feature = "rayon")]
    #[error(transparent)]
    ThreadPoolError(#[from] rayon::ThreadPoolBuildError),
    /// An error type for [`numpy`]-related conversions
    #[cfg(feature = "numpy")]
    #[error(transparent)]
    NumpyError(#[from] numpy::FromVecError),
    /// A required column was not found in the input
    #[error("Required column \"{name}\" was not found in the dataset")]
    MissingColumn {
        /// Name of the missing column
        name: String,
    },
    /// A column has an unsupported type
    #[error("Column \"{name}\" has unsupported type \"{datatype}\"")]
    InvalidColumnType {
        /// Column name
        name: String,
        /// Detected data type
        datatype: String,
    },
    /// A value has an unexpected length.
    #[error("{context} length mismatch: expected {expected}, received {actual}")]
    LengthMismatch {
        /// Description of what length was validated.
        context: String,
        /// Expected length.
        expected: usize,
        /// Actual length observed.
        actual: usize,
    },
    /// A duplicate name was provided for p4 or aux data
    #[error("Duplicate {category} name \"{name}\" provided")]
    DuplicateName {
        /// Category (p4 or aux)
        category: &'static str,
        /// Duplicate name
        name: String,
    },
    /// An unknown name was referenced (e.g., for boosts)
    #[error("Unknown {category} name \"{name}\"")]
    UnknownName {
        /// Category (p4 or aux)
        category: &'static str,
        /// Name that could not be resolved
        name: String,
    },
    /// A custom fallback error for errors too complex or too infrequent to warrant their own error
    /// category.
    #[error("{0}")]
    Custom(String),
}

/// Validate the number of free parameters supplied to a public entrypoint.
pub fn validate_free_parameter_len(input_len: usize, expected_len: usize) -> LadduResult<()> {
    if input_len != expected_len {
        return Err(LadduError::LengthMismatch {
            context: "free parameter vector".to_string(),
            expected: expected_len,
            actual: input_len,
        });
    }
    Ok(())
}

impl Clone for LadduError {
    // This is a little hack because error types are rarely cloneable, but I need to store them in a
    // cloneable box for minimizers and MCMC methods
    fn clone(&self) -> Self {
        let err_string = self.to_string();
        LadduError::Custom(err_string)
    }
}

#[cfg(feature = "python")]
impl From<LadduError> for PyErr {
    fn from(err: LadduError) -> Self {
        use pyo3::exceptions::*;
        let err_string = err.to_string();
        match err {
            LadduError::LookupError(_)
            | LadduError::RegistrationError { .. }
            | LadduError::AmplitudeNotFoundError { .. }
            | LadduError::ParseError { .. } => PyValueError::new_err(err_string),
            LadduError::ParquetError(_)
            | LadduError::ArrowError(_)
            | LadduError::IOError(_)
            | LadduError::EncodeError(_)
            | LadduError::DecodeError(_)
            | LadduError::PickleError(_) => PyIOError::new_err(err_string),
            LadduError::MissingColumn { .. } | LadduError::UnknownName { .. } => {
                PyKeyError::new_err(err_string)
            }
            LadduError::InvalidColumnType { .. }
            | LadduError::LengthMismatch { .. }
            | LadduError::DuplicateName { .. }
            | LadduError::ParameterConflict { .. }
            | LadduError::UnregisteredParameter { .. } => PyValueError::new_err(err_string),
            LadduError::Custom(_) => PyRuntimeError::new_err(err_string),
            #[cfg(feature = "rayon")]
            LadduError::ThreadPoolError(_) => PyRuntimeError::new_err(err_string),
            #[cfg(feature = "numpy")]
            LadduError::NumpyError(_) => PyValueError::new_err(err_string),
        }
    }
}

use serde::{de::DeserializeOwned, Serialize};
use std::fmt::Debug;
/// A trait which allows structs with [`Serialize`] and [`Deserialize`](`serde::Deserialize`) to
/// have a null constructor which Python can fill with data. This allows such structs to be
/// pickle-able from the Python API.
pub trait ReadWrite: Serialize + DeserializeOwned {
    /// Create a null version of the object which acts as a shell into which Python's `pickle` module
    /// can load data. This generally shouldn't be used to construct the struct in regular code.
    fn create_null() -> Self;
}
impl ReadWrite for MCMCSummary {
    fn create_null() -> Self {
        MCMCSummary::default()
    }
}
impl ReadWrite for MinimizationSummary {
    fn create_null() -> Self {
        MinimizationSummary::default()
    }
}
