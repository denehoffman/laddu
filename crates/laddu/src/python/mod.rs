//! Python bindings for the public laddu analysis API.

// Python-facing fallibility is documented with NumPy-style ``Raises`` sections.
#![allow(clippy::missing_errors_doc)]

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::PyAny,
};
use serde::{Serialize, de::DeserializeOwned};

fn to_json<T: Serialize>(value: &T) -> PyResult<String> {
    serde_json::to_string(value).map_err(|error| PyValueError::new_err(error.to_string()))
}

fn from_json<T: DeserializeOwned>(json: &str) -> PyResult<T> {
    serde_json::from_str(json).map_err(|error| PyValueError::new_err(error.to_string()))
}

macro_rules! impl_json_methods {
    ($py_type:ty) => {
        #[pymethods]
        impl $py_type {
            /// Serialize this object to a JSON string.
            #[allow(clippy::wrong_self_convention)]
            fn to_json(&self) -> PyResult<String> {
                super::to_json(&self.inner)
            }

            /// Construct an object from a JSON string.
            #[staticmethod]
            fn from_json(json: &str) -> PyResult<Self> {
                Ok(Self {
                    inner: super::from_json(json)?,
                })
            }
        }
    };
}

/// Convert a one-dimensional Python float sequence or NumPy array to `f64`.
pub(crate) fn float_vec(values: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    if let Ok(values) = values.extract::<PyReadonlyArray1<'_, f64>>() {
        return Ok(values.as_array().iter().copied().collect());
    }
    if let Ok(values) = values.extract::<PyReadonlyArray1<'_, f32>>() {
        return Ok(values
            .as_array()
            .iter()
            .map(|&value| f64::from(value))
            .collect());
    }
    values.extract::<Vec<f64>>().map_err(|_| {
        PyTypeError::new_err(
            "expected a one-dimensional float sequence or float32/float64 NumPy array",
        )
    })
}

/// Convert a two-dimensional NumPy float array to nested `f64` vectors.
pub(crate) fn float_matrix(values: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f64>>> {
    if let Ok(values) = values.extract::<PyReadonlyArray2<'_, f64>>() {
        return Ok(values
            .as_array()
            .outer_iter()
            .map(|row| row.to_vec())
            .collect());
    }
    if let Ok(values) = values.extract::<PyReadonlyArray2<'_, f32>>() {
        return Ok(values
            .as_array()
            .outer_iter()
            .map(|row| row.iter().map(|&value| f64::from(value)).collect())
            .collect());
    }
    values.extract::<Vec<Vec<f64>>>().map_err(|_| {
        PyTypeError::new_err(
            "expected a two-dimensional float sequence or float32/float64 NumPy array",
        )
    })
}

/// Convert a three-dimensional NumPy float array to nested `f64` vectors.
pub(crate) fn float_tensor3(values: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<Vec<f64>>>> {
    if let Ok(values) = values.extract::<PyReadonlyArray3<'_, f64>>() {
        return Ok(values
            .as_array()
            .outer_iter()
            .map(|matrix| matrix.outer_iter().map(|row| row.to_vec()).collect())
            .collect());
    }
    if let Ok(values) = values.extract::<PyReadonlyArray3<'_, f32>>() {
        return Ok(values
            .as_array()
            .outer_iter()
            .map(|matrix| {
                matrix
                    .outer_iter()
                    .map(|row| row.iter().map(|&value| f64::from(value)).collect())
                    .collect()
            })
            .collect());
    }
    values.extract::<Vec<Vec<Vec<f64>>>>().map_err(|_| {
        PyTypeError::new_err(
            "expected a three-dimensional float sequence or float32/float64 NumPy array",
        )
    })
}

/// Python functions for constructing standard amplitude models.
pub mod amplitude;
/// Python vector, rotation, and angular-momentum helpers.
pub mod angular;
/// Python cross-section analyses, uncertainty ensembles, and estimates.
pub mod cross_section;
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
/// Python expression-visualization colors, selectors, and style rules.
pub mod visualization;

pub use laddu_fit::ganesh::python::ganesh;

/// Define a native module containing laddu's complete Python API.
///
/// Distribution crates invoke this macro so Maturin can extract the same static
/// type metadata for every backend without maintaining Python or stub files.
#[macro_export]
macro_rules! laddu_python_module {
    ($name:ident, $backend:expr $(, $initializer:item)?) => {
        #[pyo3::pymodule(gil_used = false)]
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
            use $crate::python::cross_section::{
                PyAxis as Axis, PyBinnedEstimate as BinnedEstimate,
                PyCrossSection as CrossSection,
                PyDifferentialCrossSection as DifferentialCrossSection, PyEnsemble as Ensemble,
                PyEstimate as Estimate,
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
                PyProvenEnvelopeReport as ProvenEnvelopeReport, PyScalarSource as ScalarSource,
                PyVertexProposal as VertexProposal,
            };
            #[pymodule_export]
            use $crate::python::histogram::PyHistogram as Histogram;
            #[pymodule_export]
            use $crate::python::likelihood::{
                PyCrossSectionIntegrals as CrossSectionIntegrals,
                PyDatasetDiagnostics as DatasetDiagnostics,
                PyExtendedNll as ExtendedNLL, PyLassoPenalty as LassoPenalty,
                PyLikelihood as Likelihood, PyLikelihoodDiagnostics as LikelihoodDiagnostics,
                PyLikelihoodProjection as LikelihoodProjection,
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
            use $crate::python::runtime::{
                PyExecution as Execution, PyMemoryBudget as MemoryBudget,
                PyMemoryPlan as MemoryPlan, PyMemoryResource as MemoryResource,
                PyMemoryState as MemoryState, capabilities, gpu
            };
            #[pymodule_export]
            use $crate::python::topology::{
                PyChannel as Channel, PyEdge as Edge, PyVertex as Vertex,
                PyVertexFrame as VertexFrame,
            };
            #[pymodule_export]
            use $crate::python::visualization::{
                PyDisplayColor as DisplayColor, PyExprNodeKind as ExprNodeKind,
                PyNodeSelector as NodeSelector, PyNodeStyle as NodeStyle,
                PyNodeStyleRule as NodeStyleRule,
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
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::{PyDict, PyList};

    #[test]
    fn python_json_methods_round_trip_and_reject_invalid_json() {
        Python::initialize();
        Python::attach(|py| {
            let module = pyo3::wrap_pymodule!(super::api)(py);
            for name in [
                "Expr",
                "Particle",
                "Channel",
                "Vec3",
                "Vec4",
                "WignerD",
                "MassProposal",
                "InitialMomentum",
                "VertexProposal",
                "GenerationReport",
                "Histogram",
                "Sheet",
                "BarrierKind",
                "Model",
                "J",
                "S",
                "L",
                "M",
                "Parity",
                "Isospin",
                "Statistics",
                "MandelstamChannel",
                "RuleCheck",
                "RuleReport",
                "RuleSet",
                "PartialWave",
                "AllowedPartialWave",
                "Bin",
                "MemoryBudget",
                "MemoryPlan",
                "MemoryResource",
            ] {
                let class = module.getattr(py, name).unwrap();
                assert!(
                    class.getattr(py, "from_json").is_ok(),
                    "missing {name}.from_json"
                );
                assert!(
                    class.getattr(py, "to_json").is_ok(),
                    "missing {name}.to_json"
                );
            }
            let class = module.getattr(py, "J").unwrap();
            let value = class.call1(py, (1.5,)).unwrap();
            let json = value
                .call_method0(py, "to_json")
                .unwrap()
                .extract::<String>(py)
                .unwrap();
            let restored = class.call_method1(py, "from_json", (json,)).unwrap();

            assert_eq!(
                restored
                    .getattr(py, "value")
                    .unwrap()
                    .extract::<f64>(py)
                    .unwrap(),
                1.5
            );
            let error = class
                .call_method1(py, "from_json", ("not JSON",))
                .unwrap_err();
            assert!(error.is_instance_of::<PyValueError>(py));
        });
    }

    #[test]
    fn python_edges_are_outputs_by_default() {
        Python::initialize();
        Python::attach(|py| {
            let module = pyo3::wrap_pymodule!(super::api)(py);
            let class = module.getattr(py, "Edge").unwrap();
            let included = class.call1(py, ("included",)).unwrap();
            assert!(
                included
                    .getattr(py, "output")
                    .unwrap()
                    .extract::<bool>(py)
                    .unwrap()
            );

            let kwargs = PyDict::new(py);
            kwargs.set_item("output", false).unwrap();
            let excluded = class.call(py, ("excluded",), Some(&kwargs)).unwrap();
            assert!(
                !excluded
                    .getattr(py, "output")
                    .unwrap()
                    .extract::<bool>(py)
                    .unwrap()
            );
        });
    }

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
                "Axis",
                "Ensemble",
                "Estimate",
                "CrossSection",
                "BinnedEstimate",
                "DifferentialCrossSection",
                "Execution",
                "Model",
                "Likelihood",
                "CrossSectionIntegrals",
                "LikelihoodProjection",
                "Generator",
                "ganesh",
                "clebsch_gordan",
                "WignerD",
                "ExprNodeKind",
                "DisplayColor",
                "NodeStyle",
                "NodeSelector",
                "NodeStyleRule",
            ] {
                assert!(
                    module.getattr(py, name).is_ok(),
                    "missing Python export {name}"
                );
            }
            assert!(module.getattr(py, "particles").is_ok());
            assert!(module.getattr(py, "gpu").is_ok());
            assert!(module.getattr(py, "ganesh").is_ok());
            for class_name in ["Expr", "Model"] {
                let class = module.getattr(py, class_name).unwrap();
                for method in ["equation", "latex", "tree", "dot", "svg"] {
                    assert!(
                        class.getattr(py, method).is_ok(),
                        "missing {class_name}.{method}"
                    );
                }
            }
            let cross_sections = module.getattr(py, "CrossSectionIntegrals").unwrap();
            for method in [
                "accepted_integral",
                "generated_integral",
                "acceptance",
                "full_accepted_integral",
                "acceptance_corrected_yield",
                "observed_cross_section",
                "fitted_cross_section",
                "cross_section",
            ] {
                assert!(
                    cross_sections.getattr(py, method).is_ok(),
                    "missing CrossSectionIntegrals.{method}"
                );
            }
            let projection = module.getattr(py, "LikelihoodProjection").unwrap();
            for method in [
                "accepted_integral",
                "generated_integral",
                "acceptance",
                "full_accepted_integral",
                "acceptance_corrected_yield",
                "observed_cross_section",
                "fitted_cross_section",
                "cross_section",
                "intensities",
                "weights",
            ] {
                assert!(
                    projection.getattr(py, method).is_ok(),
                    "missing LikelihoodProjection.{method}"
                );
            }
            let cross_section = module.getattr(py, "CrossSection").unwrap();
            for method in [
                "total",
                "observed_total",
                "fitted_total",
                "acceptance",
                "corrected_yield",
                "differential",
                "combine",
            ] {
                assert!(
                    cross_section.getattr(py, method).is_ok(),
                    "missing CrossSection.{method}"
                );
            }
            let ensemble = module.getattr(py, "Ensemble").unwrap();
            for method in ["from_arrays", "from_mcmc"] {
                assert!(
                    ensemble.getattr(py, method).is_ok(),
                    "missing Ensemble.{method}"
                );
            }
        });
    }

    #[test]
    fn python_visualization_rules_customize_all_rendering_modes() {
        Python::initialize();
        Python::attach(|py| {
            let module = pyo3::wrap_pymodule!(super::api)(py);
            let color = module
                .getattr(py, "DisplayColor")
                .unwrap()
                .call1(py, (1, 2, 3))
                .unwrap();
            let fill = module
                .getattr(py, "DisplayColor")
                .unwrap()
                .call1(py, (4, 5, 6))
                .unwrap();
            let border = module
                .getattr(py, "DisplayColor")
                .unwrap()
                .call1(py, (7, 8, 9))
                .unwrap();
            let style_kwargs = PyDict::new(py);
            style_kwargs.set_item("foreground", color).unwrap();
            style_kwargs.set_item("fill", fill).unwrap();
            style_kwargs.set_item("border", border).unwrap();
            let style = module
                .getattr(py, "NodeStyle")
                .unwrap()
                .call(py, (), Some(&style_kwargs))
                .unwrap();
            let event_kind = module
                .getattr(py, "ExprNodeKind")
                .unwrap()
                .getattr(py, "EVENT_SCALAR")
                .unwrap();
            let selector = module
                .getattr(py, "NodeSelector")
                .unwrap()
                .call_method1(py, "kind", (event_kind,))
                .unwrap();
            let rule = module
                .getattr(py, "NodeStyleRule")
                .unwrap()
                .call1(py, (selector, style))
                .unwrap();
            let rules = PyList::new(py, [rule]).unwrap();
            let expression = module
                .getattr(py, "scalar")
                .unwrap()
                .call1(py, ("debug_node",))
                .unwrap();
            let kwargs = PyDict::new(py);
            kwargs.set_item("style_rules", rules).unwrap();

            let equation = expression
                .call_method(py, "equation", (), Some(&kwargs))
                .unwrap()
                .extract::<String>(py)
                .unwrap();
            let tree = expression
                .call_method(py, "tree", (), Some(&kwargs))
                .unwrap()
                .extract::<String>(py)
                .unwrap();
            let latex = expression
                .call_method(py, "latex", (), Some(&kwargs))
                .unwrap()
                .extract::<String>(py)
                .unwrap();
            let dot = expression
                .call_method(py, "dot", (), Some(&kwargs))
                .unwrap()
                .extract::<String>(py)
                .unwrap();
            let svg_path = std::env::temp_dir().join(format!(
                "laddu-python-visualization-{}.svg",
                std::process::id()
            ));
            expression
                .call_method(py, "svg", (svg_path.clone(),), Some(&kwargs))
                .unwrap();
            let svg = std::fs::read_to_string(&svg_path).unwrap();
            std::fs::remove_file(svg_path).unwrap();

            assert!(equation.contains("\x1b[38;2;1;2;3m"));
            assert!(tree.contains("\x1b[48;2;4;5;6m"));
            assert!(latex.contains("\\color[RGB]{1,2,3}"));
            assert!(dot.contains("fontcolor=\"#010203\""));
            assert!(dot.contains("fillcolor=\"#040506\""));
            assert!(dot.contains("color=\"#070809\""));
            assert!(svg.contains("<svg"));
            assert!(svg.contains("</svg>"));
        });
    }
}
