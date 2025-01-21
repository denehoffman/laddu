//! # laddu-core
//!
//! This is an internal crate used by `laddu`.
#![warn(clippy::perf, clippy::style, missing_docs)]

use bincode::ErrorKind;
#[cfg(feature = "python")]
use pyo3::PyErr;

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
    pub use crate::utils::vectors::{FourMomentum, FourVector, ThreeMomentum, ThreeVector};
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
pub use amplitudes::{
    constant, parameter, AmplitudeID, Evaluator, Expression, Manager, Model, ParameterLike,
};

// Re-exports
pub use ganesh::{mcmc::Ensemble, Bound, Status};
pub use nalgebra::{DVector, Vector3, Vector4};
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
