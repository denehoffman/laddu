use pyo3::prelude::*;

#[pymodule]
mod laddu {
    use super::*;
    #[pyfunction]
    fn version() -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }

    #[pymodule_export]
    use laddu_python::{
        amplitudes::{
            py_amplitude_one, py_amplitude_product, py_amplitude_sum, py_amplitude_zero,
            py_constant, py_parameter, PyAmplitude, PyAmplitudeID, PyEvaluator, PyExpression,
            PyManager, PyModel, PyParameterLike,
        },
        data::{py_open, PyBinnedDataset, PyDataset, PyEvent},
        mpi::{finalize_mpi, get_rank, get_size, is_mpi_available, is_root, use_mpi, using_mpi},
        utils::{
            variables::{
                PyAngles, PyCosTheta, PyMandelstam, PyMass, PyPhi, PyPolAngle, PyPolMagnitude,
                PyPolarization,
            },
            vectors::{PyVector3, PyVector4},
        },
    };

    #[pymodule_export]
    use laddu_amplitudes::{
        breit_wigner::py_breit_wigner,
        common::{py_complex_scalar, py_polar_complex_scalar, py_scalar},
        kmatrix::{
            py_kopf_kmatrix_a0, py_kopf_kmatrix_a2, py_kopf_kmatrix_f0, py_kopf_kmatrix_f2,
            py_kopf_kmatrix_pi1, py_kopf_kmatrix_rho,
        },
        phase_space::py_phase_space_factor,
        piecewise::{
            py_piecewise_complex_scalar, py_piecewise_polar_complex_scalar, py_piecewise_scalar,
        },
        ylm::py_ylm,
        zlm::py_zlm,
    };

    #[pymodule_export]
    use laddu_extensions::{
        ganesh_ext::py_ganesh::{
            py_integrated_autocorrelation_times, PyAIES, PyAIESMove, PyAutocorrelationObserver,
            PyBound, PyESS, PyESSMove, PyEnsemble, PyLBFGSB, PyMCMCObserver, PyNelderMead,
            PyObserver, PyPSO, PyStatus, PySwarm, PySwarmObserver, PySwarmPositionInitializer,
            PySwarmVelocityInitializer,
        },
        likelihoods::{
            py_likelihood_one, py_likelihood_product, py_likelihood_scalar, py_likelihood_sum,
            py_likelihood_zero, PyLikelihoodEvaluator, PyLikelihoodExpression, PyLikelihoodID,
            PyLikelihoodManager, PyLikelihoodTerm, PyNLL,
        },
    };

    #[pymodule_export]
    use laddu_extensions::experimental::{py_binned_guide_term, py_regularizer};
}
