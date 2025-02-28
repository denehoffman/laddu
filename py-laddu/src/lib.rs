use pyo3::prelude::*;

#[pymodule]
mod laddu {
    use super::*;
    #[pyfunction]
    fn version() -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }

    #[pymodule_export]
    use laddu_python::mpi::finalize_mpi;
    #[pymodule_export]
    use laddu_python::mpi::get_rank;
    #[pymodule_export]
    use laddu_python::mpi::get_size;
    #[pymodule_export]
    use laddu_python::mpi::is_mpi_available;
    #[pymodule_export]
    use laddu_python::mpi::is_root;
    #[pymodule_export]
    use laddu_python::mpi::use_mpi;
    #[pymodule_export]
    use laddu_python::mpi::using_mpi;

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
