//! Python bindings for the public laddu analysis API.

// Python-facing fallibility is documented with NumPy-style ``Raises`` sections.
#![allow(clippy::missing_errors_doc)]

/// Python functions for constructing standard amplitude models.
pub mod amplitude;
/// Python vector, rotation, and angular-momentum helpers.
pub mod angular;
/// Python dataset sources, sinks, transformations, and binning.
pub mod data;
/// Conversion between Rust errors and Python exceptions.
pub mod error;
/// Python symbolic-expression classes and constructors.
pub mod expr;
#[cfg(feature = "fit")]
/// Python optimizer adapters and fitting helpers.
pub mod fit;
#[cfg(feature = "generation")]
/// Python event-generation proposals and generators.
pub mod generation;
/// Python histogram container and filling operations.
pub mod histogram;
#[cfg(feature = "likelihood")]
/// Python likelihood terms, objectives, and projections.
pub mod likelihood;
/// Python mathematical and phase-space functions.
pub mod math;
/// Python compiled-model wrapper.
pub mod model;
/// Python particle definitions and built-in particle catalog.
pub mod particle;
/// Python spin, parity, isospin, and statistics types.
pub mod quantum;
/// Python predicates and dataset-bin specifications.
pub mod query;
/// Python execution-backend configuration and capability discovery.
pub mod runtime;
/// Python reaction-channel, edge, vertex, and frame types.
pub mod topology;

pub use laddu_fit::ganesh::python::ganesh;

/// Define a native module containing laddu's complete Python API.
///
/// Distribution crates invoke this macro so Maturin can extract the same static
/// type metadata for every backend without maintaining Python or stub files.
#[macro_export]
macro_rules! laddu_python_module {
    ($name:ident, $backend:expr $(, $initializer:item)?) => {
        #[pyo3::pymodule]
        #[doc = "laddu's native Python analysis API."]
        pub mod $name {
            use pyo3::prelude::*;

            #[pymodule_export]
            use $crate::python::ganesh;

            #[pymodule_export]
            use $crate::python::amplitude::{
                amplitudes, blatt_weisskopf_barriers, breit_wigner, f_vector, k_matrix, kopf_pi1,
                kopf_rho, p_vector, relativistic_breit_wigner,
                relativistic_breit_wigner_custom,
            };
            #[pymodule_export]
            use $crate::python::angular::{
                PyVec3 as Vec3, PyVec4 as Vec4, PyWignerD as WignerD, clebsch_gordan
            };
            #[pymodule_export]
            use $crate::python::data::{
                PyBinDataset as BinnedDataset, PyDataset as Dataset,
                PyParquetSink as ParquetSink, PyParquetSource as ParquetSource,
                PyRootSink as RootSink, PyRootSource as RootSource, read_parquet, read_root,
            };
            #[pymodule_export]
            use $crate::python::error::LadduError;
            #[pymodule_export]
            use $crate::python::expr::{
                PyExpr as Expr, acos, atan2, cis, complex, dot, matmul, matrix, matvec, parameter,
                polar_complex, scalar, solve, vector,
            };
            #[pymodule_export]
            use $crate::python::generation::{
                PyGenerationReport as GenerationReport, PyGenerator as Generator,
                PyInitialMomentum as InitialMomentum, PyMassProposal as MassProposal,
                PyVertexProposal as VertexProposal,
            };
            #[pymodule_export]
            use $crate::python::histogram::PyHistogram as Histogram;
            #[pymodule_export]
            use $crate::python::likelihood::{
                PyExtendedNll as ExtendedNLL, PyLassoPenalty as LassoPenalty,
                PyLikelihood as Likelihood, PyLikelihoodProjection as LikelihoodProjection,
                PyNll as NLL, PyRidgePenalty as RidgePenalty,
            };
            #[pymodule_export]
            use $crate::python::math::{
                PyBarrierKind as BarrierKind, PySheet as Sheet, blatt_weisskopf,
                blatt_weisskopf_custom, chew_mandelstam, q, rho, spherical_harmonic,
            };
            #[pymodule_export]
            use $crate::python::model::PyModel as Model;
            #[pymodule_export]
            use $crate::python::particle::{PyParticle as Particle, particles};
            #[pymodule_export]
            use $crate::python::quantum::{
                PyAllowedPartialWave as AllowedPartialWave, PyIsospin as Isospin, PyJ as J,
                PyL as L, PyM as M, PyMandelstamChannel as MandelstamChannel,
                PyPartialWave as PartialWave, PyParity as Parity, PyRuleCheck as RuleCheck,
                PyRuleReport as RuleReport, PyRuleSet as RuleSet, PyS as S,
                PySelectionRules as SelectionRules, PyStatistics as Statistics,
            };
            #[pymodule_export]
            use $crate::python::query::{PyBin as Bin, PyPredicate as Predicate};
            #[pymodule_export]
            use $crate::python::runtime::{PyExecution as Execution, capabilities, gpu};
            #[pymodule_export]
            use $crate::python::topology::{
                PyChannel as Channel, PyEdge as Edge, PyVertex as Vertex,
                PyVertexFrame as VertexFrame,
            };

            /// Return the native backend selected for this extension.
            #[pyfunction]
            fn backend() -> &'static str {
                $backend
            }

            $($initializer)?
        }
    };
}

laddu_python_module!(
    api,
    if cfg!(feature = "mpi") {
        "mpi"
    } else {
        "local"
    }
);

#[cfg(test)]
mod tests {
    use pyo3::prelude::*;

    #[test]
    fn python_module_exports_the_analysis_spine() {
        Python::initialize();
        Python::attach(|py| {
            let module = pyo3::wrap_pymodule!(super::api)(py);
            for name in [
                "Expr",
                "Particle",
                "Channel",
                "VertexFrame",
                "Vec3",
                "Vec4",
                "J",
                "L",
                "S",
                "M",
                "Parity",
                "Dataset",
                "Execution",
                "Model",
                "Likelihood",
                "Generator",
                "ganesh",
                "clebsch_gordan",
                "WignerD",
            ] {
                assert!(
                    module.getattr(py, name).is_ok(),
                    "missing Python export {name}"
                );
            }
            assert!(module.getattr(py, "particles").is_ok());
            assert!(module.getattr(py, "gpu").is_ok());
            assert!(module.getattr(py, "ganesh").is_ok());
        });
    }
}
