//! # laddu-core
//!
//! This is an internal crate used by `laddu`.
#![warn(clippy::perf, clippy::style, missing_docs)]
#![allow(clippy::excessive_precision)]

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
pub mod mpi {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::OnceLock;

    use lazy_static::lazy_static;
    use mpi::environment::Universe;
    use mpi::topology::{Process, SimpleCommunicator};
    use mpi::traits::Communicator;
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

    /// A trait including some useful auxiliary methods for MPI
    pub trait LadduMPI {
        /// Get the process at the root rank
        fn process_at_root(&self) -> Process<'_>;
        /// Check if the current rank is the root rank
        fn is_root(&self) -> bool;
        /// Get the counts/displacements for partitioning a buffer of length
        /// `buf_len`
        fn get_counts_displs(&self, buf_len: usize) -> (Vec<i32>, Vec<i32>);
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

        fn get_counts_displs(&self, buf_len: usize) -> (Vec<i32>, Vec<i32>) {
            let mut counts = vec![0; self.size() as usize];
            let mut displs = vec![0; self.size() as usize];
            let chunk_size = buf_len / self.size() as usize;
            let surplus = buf_len % self.size() as usize;
            for i in 0..self.size() as usize {
                counts[i] = if i < surplus {
                    chunk_size + 1
                } else {
                    chunk_size
                } as i32;
                displs[i] = if i == 0 {
                    0
                } else {
                    displs[i - 1] + counts[i - 1]
                };
            }
            (counts, displs)
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
pub use amplitudes::{
    constant, parameter, AmplitudeID, Evaluator, Expression, Manager, Model, ParameterLike,
};

/// The mathematical constant $`\pi`$.
pub const PI: f64 = std::f64::consts::PI;

/// A [`Result`] type alias for [`LadduError`]s.
pub type LadduResult<T> = Result<T, LadduError>;

/// The error type used by all `laddu` internal methods
#[derive(Error, Debug)]
pub enum LadduError {
    /// An alias for [`std::io::Error`].
    #[error("IO Error: {0}")]
    IOError(#[from] std::io::Error),
    /// An alias for [`parquet::errors::ParquetError`].
    #[error("Parquet Error: {0}")]
    ParquetError(#[from] parquet::errors::ParquetError),
    /// An alias for [`arrow::error::ArrowError`].
    #[error("Arrow Error: {0}")]
    ArrowError(#[from] arrow::error::ArrowError),
    /// An alias for [`shellexpand::LookupError`].
    #[error("Failed to expand path: {0}")]
    LookupError(#[from] shellexpand::LookupError<std::env::VarError>),
    /// An error which occurs when the user tries to register two amplitudes by the same name to
    /// the same [`Manager`].
    #[error("An amplitude by the name \"{name}\" is already registered by this manager!")]
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
    #[error("Encoder error: {0}")]
    EncodeError(#[from] bincode::error::EncodeError),
    /// An error returned by the Rust decoder
    #[error("Decoder error: {0}")]
    DecodeError(#[from] bincode::error::DecodeError),
    /// An error returned by the Python pickle (de)serializer
    #[error("Pickle conversion error: {0}")]
    PickleError(#[from] serde_pickle::Error),
    /// An error type for [`rayon`] thread pools
    #[cfg(feature = "rayon")]
    #[error("Error building thread pool: {0}")]
    ThreadPoolError(#[from] rayon::ThreadPoolBuildError),
    /// An error type for [`numpy`]-related conversions
    #[cfg(feature = "numpy")]
    #[error("Numpy error: {0}")]
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
            LadduError::InvalidColumnType { .. } | LadduError::DuplicateName { .. } => {
                PyValueError::new_err(err_string)
            }
            LadduError::Custom(_) => PyException::new_err(err_string),
            #[cfg(feature = "rayon")]
            LadduError::ThreadPoolError(_) => PyException::new_err(err_string),
            #[cfg(feature = "numpy")]
            LadduError::NumpyError(_) => PyException::new_err(err_string),
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
