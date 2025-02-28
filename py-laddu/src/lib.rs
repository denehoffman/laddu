use pyo3::prelude::*;

#[cfg(not(feature = "mpi"))]
use pyo3::exceptions::PyModuleNotFoundError;

#[pymodule]
mod laddu {
    use super::*;
    #[pyfunction]
    fn version() -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }

    /// Check if ``laddu`` was compiled with MPI support (returns ``True`` if it was).
    ///
    /// Since ``laddu-mpi`` has the same namespace as ``laddu`` (they both are imported with
    /// ``import laddu``), this method can be used to check if MPI capabilities are available
    /// without actually running any MPI code. While functions in the ``laddu.mpi`` module will
    /// raise an ``ModuleNotFoundError`` if MPI is not supported, its sometimes convenient to have
    /// a simple boolean check rather than a try-catch block, and this method provides that.
    ///
    #[cfg(feature = "mpi")]
    #[pyfunction]
    fn is_mpi_available() -> bool {
        true
    }
    /// Check if ``laddu`` was compiled with MPI support (returns ``True`` if it was).
    ///
    /// Since ``laddu-mpi`` has the same namespace as ``laddu`` (they both are imported with
    /// ``import laddu``), this method can be used to check if MPI capabilities are available
    /// without actually running any MPI code. While functions in the ``laddu.mpi`` module will
    /// raise an ``ModuleNotFoundError`` if MPI is not supported, its sometimes convenient to have
    /// a simple boolean check rather than a try-catch block, and this method provides that.
    ///
    #[cfg(not(feature = "mpi"))]
    #[pyfunction]
    fn is_mpi_available() -> bool {
        false
    }

    #[cfg(feature = "mpi")]
    #[pymodule_export]
    use laddu_python::mpi::finalize_mpi;
    #[cfg(not(feature = "mpi"))]
    #[pyfunction]
    fn finalize_mpi() -> PyResult<()> {
        Err(PyModuleNotFoundError::new_err(
            "`laddu` was not compiled with MPI support! Please use `laddu-mpi` instead.",
        ))
    }
    #[cfg(feature = "mpi")]
    #[pymodule_export]
    use laddu_python::mpi::get_rank;
    #[cfg(not(feature = "mpi"))]
    #[pyfunction]
    fn get_rank() -> PyResult<usize> {
        Err(PyModuleNotFoundError::new_err(
            "`laddu` was not compiled with MPI support! Please use `laddu-mpi` instead.",
        ))
    }
    #[cfg(feature = "mpi")]
    #[pymodule_export]
    use laddu_python::mpi::get_size;
    #[cfg(not(feature = "mpi"))]
    #[pyfunction]
    fn get_size() -> PyResult<usize> {
        Err(PyModuleNotFoundError::new_err(
            "`laddu` was not compiled with MPI support! Please use `laddu-mpi` instead.",
        ))
    }
    #[cfg(feature = "mpi")]
    #[pymodule_export]
    use laddu_python::mpi::is_root;
    #[cfg(not(feature = "mpi"))]
    #[pyfunction]
    fn is_root() -> PyResult<bool> {
        Err(PyModuleNotFoundError::new_err(
            "`laddu` was not compiled with MPI support! Please use `laddu-mpi` instead.",
        ))
    }
    #[cfg(feature = "mpi")]
    #[pymodule_export]
    use laddu_python::mpi::use_mpi;
    #[cfg(not(feature = "mpi"))]
    #[pyfunction]
    #[pyo3(signature = (*, trigger=true))]
    #[allow(unused_variables)]
    fn use_mpi(trigger: bool) -> PyResult<()> {
        Err(PyModuleNotFoundError::new_err(
            "`laddu` was not compiled with MPI support! Please use `laddu-mpi` instead.",
        ))
    }
    #[cfg(feature = "mpi")]
    #[pymodule_export]
    use laddu_python::mpi::using_mpi;
    #[cfg(not(feature = "mpi"))]
    #[pyfunction]
    fn using_mpi() -> PyResult<bool> {
        Err(PyModuleNotFoundError::new_err(
            "`laddu` was not compiled with MPI support! Please use `laddu-mpi` instead.",
        ))
    }

    #[pymodule_export]
    use laddu_python::utils::vectors::PyVector3;
    #[pymodule_export]
    use laddu_python::utils::vectors::PyVector4;

    #[pymodule_export]
    use laddu_python::utils::variables::PyAngles;
    #[pymodule_export]
    use laddu_python::utils::variables::PyCosTheta;
    #[pymodule_export]
    use laddu_python::utils::variables::PyMandelstam;
    #[pymodule_export]
    use laddu_python::utils::variables::PyMass;
    #[pymodule_export]
    use laddu_python::utils::variables::PyPhi;
    #[pymodule_export]
    use laddu_python::utils::variables::PyPolAngle;
    #[pymodule_export]
    use laddu_python::utils::variables::PyPolMagnitude;
    #[pymodule_export]
    use laddu_python::utils::variables::PyPolarization;

    #[pymodule_export]
    use laddu_python::data::py_open;
    #[pymodule_export]
    use laddu_python::data::PyBinnedDataset;
    #[pymodule_export]
    use laddu_python::data::PyDataset;
    #[pymodule_export]
    use laddu_python::data::PyEvent;

    #[pymodule_export]
    use laddu_python::amplitudes::py_constant;
    #[pymodule_export]
    use laddu_python::amplitudes::py_parameter;
    #[pymodule_export]
    use laddu_python::amplitudes::PyAmplitude;
    #[pymodule_export]
    use laddu_python::amplitudes::PyAmplitudeID;
    #[pymodule_export]
    use laddu_python::amplitudes::PyEvaluator;
    #[pymodule_export]
    use laddu_python::amplitudes::PyExpression;
    #[pymodule_export]
    use laddu_python::amplitudes::PyManager;
    #[pymodule_export]
    use laddu_python::amplitudes::PyModel;
    #[pymodule_export]
    use laddu_python::amplitudes::PyParameterLike;

    #[pymodule_export]
    use laddu_amplitudes::common::py_complex_scalar;
    #[pymodule_export]
    use laddu_amplitudes::common::py_polar_complex_scalar;
    #[pymodule_export]
    use laddu_amplitudes::common::py_scalar;

    #[pymodule_export]
    use laddu_amplitudes::piecewise::py_piecewise_complex_scalar;
    #[pymodule_export]
    use laddu_amplitudes::piecewise::py_piecewise_polar_complex_scalar;
    #[pymodule_export]
    use laddu_amplitudes::piecewise::py_piecewise_scalar;

    #[pymodule_export]
    use laddu_amplitudes::breit_wigner::py_breit_wigner;

    #[pymodule_export]
    use laddu_amplitudes::ylm::py_ylm;

    #[pymodule_export]
    use laddu_amplitudes::zlm::py_zlm;

    #[pymodule_export]
    use laddu_amplitudes::kmatrix::py_kopf_kmatrix_a0;
    #[pymodule_export]
    use laddu_amplitudes::kmatrix::py_kopf_kmatrix_a2;
    #[pymodule_export]
    use laddu_amplitudes::kmatrix::py_kopf_kmatrix_f0;
    #[pymodule_export]
    use laddu_amplitudes::kmatrix::py_kopf_kmatrix_f2;
    #[pymodule_export]
    use laddu_amplitudes::kmatrix::py_kopf_kmatrix_pi1;
    #[pymodule_export]
    use laddu_amplitudes::kmatrix::py_kopf_kmatrix_rho;

    #[pymodule_export]
    use laddu_extensions::likelihoods::py_likelihood_scalar;
    #[pymodule_export]
    use laddu_extensions::likelihoods::PyLikelihoodEvaluator;
    #[pymodule_export]
    use laddu_extensions::likelihoods::PyLikelihoodExpression;
    #[pymodule_export]
    use laddu_extensions::likelihoods::PyLikelihoodID;
    #[pymodule_export]
    use laddu_extensions::likelihoods::PyLikelihoodManager;
    #[pymodule_export]
    use laddu_extensions::likelihoods::PyLikelihoodTerm;
    #[pymodule_export]
    use laddu_extensions::likelihoods::PyNLL;

    #[pymodule_export]
    use laddu_extensions::ganesh_ext::py_ganesh::py_integrated_autocorrelation_times;
    #[pymodule_export]
    use laddu_extensions::ganesh_ext::py_ganesh::PyAutocorrelationObserver;
    #[pymodule_export]
    use laddu_extensions::ganesh_ext::py_ganesh::PyBound;
    #[pymodule_export]
    use laddu_extensions::ganesh_ext::py_ganesh::PyEnsemble;
    #[pymodule_export]
    use laddu_extensions::ganesh_ext::py_ganesh::PyMCMCObserver;
    #[pymodule_export]
    use laddu_extensions::ganesh_ext::py_ganesh::PyObserver;
    #[pymodule_export]
    use laddu_extensions::ganesh_ext::py_ganesh::PyStatus;

    #[pymodule_export]
    use laddu_extensions::experimental::py_binned_guide_term;
}
