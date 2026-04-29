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
    use laddu_python::data::{
        from_columns, read_parquet, read_parquet_chunked, read_root, write_parquet, write_root,
    };
    #[pymodule_export]
    use laddu_python::extensions::{
        callbacks::{py_integrated_autocorrelation_times, PyControlFlow},
        experimental::{py_binned_guide_term, py_regularizer},
        likelihood::{
            py_likelihood_one, py_likelihood_product, py_likelihood_scalar, py_likelihood_sum,
            py_likelihood_zero, PyLikelihoodExpression, PyNLL, PyStochasticNLL,
        },
    };
    #[pymodule_export]
    use laddu_python::{
        amplitudes::{
            py_blatt_weisskopf, py_breit_wigner, py_breit_wigner_non_relativistic,
            py_clebsch_gordan, py_complex_scalar, py_expr_one, py_expr_product, py_expr_sum,
            py_expr_zero, py_flatte, py_kopf_kmatrix_a0, py_kopf_kmatrix_a2, py_kopf_kmatrix_f0,
            py_kopf_kmatrix_f2, py_kopf_kmatrix_pi1, py_kopf_kmatrix_rho, py_lookup_table,
            py_lookup_table_complex, py_lookup_table_polar, py_lookup_table_scalar, py_parameter,
            py_phase_space_factor, py_photon_sdme, py_polar_complex_scalar, py_polphase, py_scalar,
            py_test_amplitude, py_variable_scalar, py_voigt, py_wigner_3j, py_wigner_d, py_ylm,
            py_zlm, PyCompiledExpression, PyEvaluator, PyExpression, PyKopfKMatrixA0Channel,
            PyKopfKMatrixA2Channel, PyKopfKMatrixF0Channel, PyKopfKMatrixF2Channel,
            PyKopfKMatrixPi1Channel, PyKopfKMatrixRhoChannel, PyParameter,
        },
        available_parallelism,
        data::{PyBinnedDataset, PyDataset, PyEvent, PyParquetChunkIter},
        generation::{
            PyCompositeGenerator, PyDistribution, PyEventGenerator, PyGeneratedBatch,
            PyGeneratedBatchIter, PyGeneratedEventLayout, PyGeneratedParticle,
            PyGeneratedParticleLayout, PyGeneratedReaction, PyGeneratedVertexLayout,
            PyInitialGenerator, PyMandelstamTDistribution, PyReconstruction, PyStableGenerator,
        },
        get_threads,
        mpi::{finalize_mpi, get_rank, get_size, is_mpi_available, is_root, use_mpi, using_mpi},
        quantum::angular_momentum::{py_allowed_projections, py_helicity_combinations},
        set_threads,
        variables::{
            PyAngles, PyCosTheta, PyDecay, PyMandelstam, PyMass, PyParticle, PyPhi, PyPolAngle,
            PyPolMagnitude, PyPolarization, PyReaction, PyVariableExpression,
        },
        vectors::{PyVec3, PyVec4},
    };
}
