use pyo3::prelude::*;

#[cfg_attr(feature = "mpi", pymodule(name = "laddu_mpi"))]
#[cfg_attr(not(feature = "mpi"), pymodule(name = "laddu_cpu"))]
mod laddu {
    use super::*;
    #[pyfunction]
    fn version() -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }

    #[pymodule_export]
    use laddu_amplitudes::{
        breit_wigner::{py_breit_wigner, py_breit_wigner_non_relativistic},
        common::{py_complex_scalar, py_polar_complex_scalar, py_scalar, py_variable_scalar},
        flatte::py_flatte,
        kmatrix::{
            py_kopf_kmatrix_a0, py_kopf_kmatrix_a2, py_kopf_kmatrix_f0, py_kopf_kmatrix_f2,
            py_kopf_kmatrix_pi1, py_kopf_kmatrix_rho, KopfKMatrixA0Channel, KopfKMatrixA2Channel,
            KopfKMatrixF0Channel, KopfKMatrixF2Channel, KopfKMatrixPi1Channel,
            KopfKMatrixRhoChannel,
        },
        lookup_table::{
            py_lookup_table, py_lookup_table_complex, py_lookup_table_polar, py_lookup_table_scalar,
        },
        phase_space::py_phase_space_factor,
        spin_factors::{
            py_blatt_weisskopf, py_clebsch_gordan, py_photon_sdme, py_wigner_3j, py_wigner_d,
        },
        voigt::py_voigt,
        ylm::py_ylm,
        zlm::{py_polphase, py_zlm},
    };
    #[pymodule_export]
    use laddu_extensions::experimental::{py_binned_guide_term, py_regularizer};
    #[pymodule_export]
    use laddu_extensions::{
        ganesh_ext::py_ganesh::{py_integrated_autocorrelation_times, PyControlFlow},
        likelihoods::{
            py_likelihood_one, py_likelihood_product, py_likelihood_scalar, py_likelihood_sum,
            py_likelihood_zero, PyLikelihoodExpression, PyNLL, PyStochasticNLL,
        },
    };
    #[pymodule_export]
    use laddu_python::data::{
        from_columns, read_parquet, read_parquet_chunked, read_root, write_parquet, write_root,
    };
    #[pymodule_export]
    use laddu_python::{
        amplitudes::{
            py_expr_one, py_expr_product, py_expr_sum, py_expr_zero, py_parameter,
            py_test_amplitude, PyCompiledExpression, PyEvaluator, PyExpression, PyParameter,
        },
        available_parallelism,
        data::{PyBinnedDataset, PyDataset, PyEvent, PyParquetChunkIter},
        get_threads,
        mpi::{finalize_mpi, get_rank, get_size, is_mpi_available, is_root, use_mpi, using_mpi},
        set_threads,
        utils::{
            angular_momentum::{py_allowed_projections, py_helicity_combinations},
            variables::{
                PyAngles, PyCosTheta, PyDecay, PyMandelstam, PyMass, PyParticle, PyPhi, PyPolAngle,
                PyPolMagnitude, PyPolarization, PyReaction, PyVariableExpression,
            },
            vectors::{PyVec3, PyVec4},
        },
    };
}
