//! # laddu-core
//!
//! This is an internal crate used by `laddu`.
#![warn(clippy::perf, clippy::style)]
// #![warn(missing_docs)]
#![allow(clippy::excessive_precision)]

use ganesh::core::{MCMCSummary, MinimizationSummary};
#[cfg(feature = "python")]
use pyo3::PyErr;

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
pub mod mpi {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::OnceLock;

    use lazy_static::lazy_static;
    #[cfg(feature = "mpi")]
    use mpi::{
        environment::Universe,
        topology::{Process, SimpleCommunicator},
        traits::Communicator,
    };
    use parking_lot::RwLock;

    lazy_static! {
        static ref USE_MPI: AtomicBool = AtomicBool::new(false);
    }

    pub struct MPIState {
        pub size: usize,
        pub rank: usize,
        #[cfg(feature = "mpi")]
        pub universe: Universe,
    }

    static MPI_STATE: OnceLock<RwLock<Option<MPIState>>> = OnceLock::new();

    /// The default root rank for MPI processes
    pub const ROOT_RANK: usize = 0;

    /// Check if the current MPI process is the root process
    pub fn is_root() -> bool {
        crate::mpi::rank() == ROOT_RANK
    }

    /// Shortcut method to just get the global MPI communicator without accessing `size` and `rank`
    /// directly
    #[cfg(feature = "mpi")]
    pub fn world() -> Option<SimpleCommunicator> {
        if let Some(mpi_state_lock) = MPI_STATE.get() {
            if let Some(mpi_state) = &*mpi_state_lock.read() {
                return Some(mpi_state.universe.world());
            }
        }
        None
    }

    /// Get the rank of the current process
    pub fn rank() -> usize {
        if let Some(mpi_state_lock) = MPI_STATE.get() {
            if let Some(mpi_state) = &*mpi_state_lock.read() {
                return mpi_state.rank;
            }
        }
        ROOT_RANK
    }

    /// Get number of available processes/ranks
    pub fn size() -> usize {
        if let Some(mpi_state_lock) = MPI_STATE.get() {
            if let Some(mpi_state) = &*mpi_state_lock.read() {
                return mpi_state.size;
            }
        }
        1
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
            MPI_STATE.get_or_init(|| {
                #[cfg(feature = "mpi")]
                {
                    #[cfg(feature = "rayon")]
                    let threading = mpi::Threading::Funneled;
                    #[cfg(not(feature = "rayon"))]
                    let threading = mpi::Threading::Single;
                    let (universe, _threading) = mpi::initialize_with_threading(threading).unwrap();
                    let world = universe.world();
                    RwLock::new(Some(MPIState {
                        size: world.size() as usize,
                        rank: world.rank() as usize,
                        universe,
                    }))
                }
                #[cfg(not(feature = "mpi"))]
                {
                    RwLock::new(Some(MPIState {
                        size: 1,
                        rank: ROOT_RANK,
                    }))
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
            let mut mpi_state = MPI_STATE.get().unwrap().write();
            *mpi_state = None;
        }
    }

    /// Check if MPI backend is enabled
    pub fn using_mpi() -> bool {
        USE_MPI.load(Ordering::SeqCst)
    }

    pub fn get_range_for_rank(total: usize) -> (usize, usize) {
        let base = total / crate::mpi::size();
        let rem = total % crate::mpi::size();
        if crate::mpi::rank() < rem {
            let count = base + 1;
            let start = crate::mpi::rank() * count;
            (start, count)
        } else {
            let count = base;
            let start = rem * (base + 1) + (crate::mpi::rank() - rem) * base;
            (start, count)
        }
    }
}

use thiserror::Error;

/// [`Amplitude`](crate::amplitudes::Amplitude)s and methods for making and evaluating them.
pub mod amplitudes;
/// Methods for loading and manipulating [`Event`]-based data.
pub mod data;
/// Structures for manipulating the cache and free parameters.
pub mod resources;
/// Utility functions, enums, and traits
pub mod utils;
/// Useful traits for all crate structs
pub mod traits {
    pub use crate::amplitudes::Amplitude;
    pub use crate::ReadWrite;
}

pub use crate::data::Dataset;
// pub use crate::resources::{
//     Cache, ComplexMatrixID, ComplexScalarID, ComplexVectorID, MatrixID, ParameterID, Parameters,
//     Resources, ScalarID, VectorID,
// };
pub use crate::utils::enums::{Channel, Frame, Sign};
pub use crate::utils::variables::{
    angles, costheta, mandelstam, mass, phi, pol_angle, pol_magnitude, polarization,
};
pub use crate::utils::vectors::{Vec3, Vec4};
// pub use amplitudes::{
//     constant, parameter, AmplitudeID, Evaluator, Expression, Manager, Model, ParameterLike,
// };

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
    /// An error type for [`polars`]-related operations
    #[error("Polars error: {0}")]
    PolarsError(#[from] polars::error::PolarsError),
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
            LadduError::Custom(_) => PyException::new_err(err_string),
            #[cfg(feature = "rayon")]
            LadduError::ThreadPoolError(_) => PyException::new_err(err_string),
            #[cfg(feature = "numpy")]
            LadduError::NumpyError(_) => PyException::new_err(err_string),
            LadduError::PolarsError(_) => PyException::new_err(err_string),
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
