//! # laddu-core
//!
//! This is an internal crate used by `laddu`.
#![warn(clippy::perf, clippy::style, missing_docs)]

use bincode::ErrorKind;
#[cfg(feature = "python")]
use pyo3::PyErr;

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
            false
        }
    }

    /// Shortcut method to just get the global MPI communicator without accessing `size` and `rank`
    /// directly
    pub fn get_world() -> Option<SimpleCommunicator> {
        if let Some(universe) = &*MPI_UNIVERSE.get().unwrap().read() {
            let world = universe.world();
            if world.size() == 1 {
                return None;
            }
            return Some(world);
        }
        None
    }

    /// Get the rank of the current process
    pub fn get_rank() -> Option<i32> {
        get_world().map(|w| w.rank())
    }

    /// Get number of available processes/ranks
    pub fn get_size() -> Option<i32> {
        get_world().map(|w| w.size())
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

    pub trait LadduMPI {
        fn process_at_root(&self) -> Process<'_>;
        fn is_root(&self) -> bool;
        fn get_counts_displs(&self, buf_len: usize) -> (Vec<i32>, Vec<i32>);
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
/// Methods for loading and manipulating [`Event`]-based data.
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

pub use crate::data::{open, BinnedDataset, Dataset, Event};
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

// Re-exports
pub use ganesh::{mcmc::Ensemble, Bound, Status};
pub use nalgebra::DVector;
pub use num::Complex;

/// A floating-point number type (defaults to [`f64`], see `f32` feature).
#[cfg(not(feature = "f32"))]
pub type Float = f64;

/// A floating-point number type (defaults to [`f64`], see `f32` feature).
#[cfg(feature = "f32")]
pub type Float = f32;

/// The mathematical constant $`\pi`$.
#[cfg(not(feature = "f32"))]
pub const PI: Float = std::f64::consts::PI;

/// The mathematical constant $`\pi`$.
#[cfg(feature = "f32")]
pub const PI: Float = std::f32::consts::PI;

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
    /// An error returned by the Rust de(serializer)
    #[error("(De)Serialization error: {0}")]
    SerdeError(#[from] Box<ErrorKind>),
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
            | LadduError::SerdeError(_)
            | LadduError::PickleError(_) => PyIOError::new_err(err_string),
            LadduError::Custom(_) => PyException::new_err(err_string),
            #[cfg(feature = "rayon")]
            LadduError::ThreadPoolError(_) => PyException::new_err(err_string),
            #[cfg(feature = "numpy")]
            LadduError::NumpyError(_) => PyException::new_err(err_string),
        }
    }
}

use serde::{de::DeserializeOwned, Serialize};
use std::{
    fmt::Debug,
    fs::File,
    io::{BufReader, BufWriter},
    path::Path,
};
/// A trait which allows structs with [`Serialize`] and [`Deserialize`](`serde::Deserialize`) to be
/// written and read from files with a certain set of types/extensions.
///
/// Currently, Python's pickle format is supported supported, since it's an easy-to-parse standard
/// that supports floating point values better that JSON or TOML
pub trait ReadWrite: Serialize + DeserializeOwned {
    /// Create a null version of the object which acts as a shell into which Python's `pickle` module
    /// can load data. This generally shouldn't be used to construct the struct in regular code.
    fn create_null() -> Self;
    /// Save a [`serde`]-object to a file path, using the extension to determine the file format
    fn save_as<T: AsRef<str>>(&self, file_path: T) -> Result<(), LadduError> {
        let expanded_path = shellexpand::full(file_path.as_ref())?;
        let file_path = Path::new(expanded_path.as_ref());
        let file = File::create(file_path)?;
        let mut writer = BufWriter::new(file);
        serde_pickle::to_writer(&mut writer, self, Default::default())?;
        Ok(())
    }
    /// Load a [`serde`]-object from a file path, using the extension to determine the file format
    fn load_from<T: AsRef<str>>(file_path: T) -> Result<Self, LadduError> {
        let file_path = Path::new(&*shellexpand::full(file_path.as_ref())?).canonicalize()?;
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
        serde_pickle::from_reader(reader, Default::default()).map_err(LadduError::from)
    }
}

impl ReadWrite for Status {
    fn create_null() -> Self {
        Status::default()
    }
}
impl ReadWrite for Ensemble {
    fn create_null() -> Self {
        Ensemble::new(Vec::default())
    }
}
impl ReadWrite for Model {
    fn create_null() -> Self {
        Model {
            manager: Manager::default(),
            expression: Expression::default(),
        }
    }
}
