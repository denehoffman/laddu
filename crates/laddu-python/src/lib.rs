#![warn(clippy::perf, clippy::style)]
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[cfg_attr(coverage_nightly, coverage(off))]
pub mod amplitudes;
#[cfg_attr(coverage_nightly, coverage(off))]
pub mod data;
#[cfg_attr(coverage_nightly, coverage(off))]
pub mod utils;

#[cfg_attr(coverage_nightly, coverage(off))]
pub mod mpi {
    use super::*;
    /// Use the Message Passing Interface (MPI) to run on a distributed system
    ///
    /// Notes
    /// -----
    /// You must have MPI installed for this to work, and you must call the program with
    /// ``mpirun <executable>``, or bad things will happen.
    ///
    /// MPI runs an identical program on each process, but gives the program an ID called its
    /// "rank". Only the results of methods on the root process (rank 0) should be
    /// considered valid, as other processes only contain portions of each dataset. To ensure
    /// you don't save or print data at other ranks, use the provided ``laddu.mpi.is_root()``
    /// method to check if the process is the root process.
    ///
    /// Once MPI is enabled, it cannot be disabled. If MPI could be toggled (which it can't),
    /// the other processes will still run, but they will be independent of the root process
    /// and will no longer communicate with it. The root process stores no data, so it would
    /// be difficult (and convoluted) to get the results which were already processed via
    /// MPI.
    ///
    /// Additionally, MPI must be enabled at the beginning of a script, at least before any
    /// other ``laddu`` functions are called. For this reason, it is suggested that you use the
    /// context manager ``laddu.mpi.MPI`` to ensure the MPI backend is used properly.
    ///
    /// If ``laddu.mpi.use_mpi()`` is called multiple times, the subsequent calls will have no
    /// effect.
    ///
    /// You **must** call ``laddu.mpi.finalize_mpi()`` before your program exits for MPI to terminate
    /// smoothly.
    ///
    /// See Also
    /// --------
    /// laddu.mpi.MPI
    /// laddu.mpi.using_mpi
    /// laddu.mpi.is_root
    /// laddu.mpi.get_rank
    /// laddu.mpi.get_size
    /// laddu.mpi.finalize_mpi
    ///
    #[pyfunction]
    pub fn use_mpi() {
        laddu_core::use_mpi();
    }

    /// Drop the MPI universe and finalize MPI at the end of a program
    ///
    /// This should only be called once and should be called at the end of all ``laddu``-related
    /// function calls. This **must** be called at the end of any program which uses MPI.
    ///
    /// See Also
    /// --------
    /// laddu.mpi.use_mpi
    ///
    #[pyfunction]
    pub fn finalize_mpi() {
        laddu_core::finalize_mpi();
    }

    /// Check if MPI is enabled
    ///
    /// This can be combined with ``laddu.mpi.is_root()`` to ensure valid results are only
    /// returned from the root rank process on the condition that MPI is enabled.
    ///
    /// See Also
    /// --------
    /// laddu.mpi.use_mpi
    /// laddu.mpi.is_root
    ///
    #[pyfunction]
    pub fn using_mpi() -> bool {
        laddu_core::using_mpi()
    }

    /// Check if the current MPI process is the root process
    ///
    /// This can be combined with ``laddu.mpi.using_mpi()`` to ensure valid results are only
    /// returned from the root rank process on the condition that MPI is enabled.
    ///
    /// See Also
    /// --------
    /// laddu.mpi.use_mpi
    /// laddu.mpi.using_mpi
    ///
    #[pyfunction]
    pub fn is_root() -> bool {
        laddu_core::is_root()
    }

    /// Get the rank of the current MPI process
    ///
    /// Returns ``None`` if MPI is not enabled
    ///
    /// See Also
    /// --------
    /// laddu.mpi.use_mpi
    ///
    #[pyfunction]
    pub fn get_rank() -> Option<i32> {
        if let Some((_, rank, _)) = laddu_core::get_world_rank_size() {
            Some(rank)
        } else {
            None
        }
    }

    /// Get the total number of MPI processes (including the root process)
    ///
    /// Returns ``None`` if MPI is not enabled
    ///
    /// See Also
    /// --------
    /// laddu.mpi.use_mpi
    ///
    #[pyfunction]
    pub fn get_size() -> Option<i32> {
        if let Some((_, _, size)) = laddu_core::get_world_rank_size() {
            Some(size)
        } else {
            None
        }
    }
}

pub trait GetStrExtractObj {
    fn get_extract<T>(&self, key: &str) -> PyResult<Option<T>>
    where
        T: for<'py> FromPyObject<'py>;
}

#[cfg_attr(coverage_nightly, coverage(off))]
impl GetStrExtractObj for Bound<'_, PyDict> {
    fn get_extract<T>(&self, key: &str) -> PyResult<Option<T>>
    where
        T: for<'py> FromPyObject<'py>,
    {
        self.get_item(key)?
            .map(|value| value.extract::<T>())
            .transpose()
    }
}
