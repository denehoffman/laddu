//! Parameter edits and immutable inspection records for Python models.

use laddu_expr::parameters::{Bounds, InitialSpec, ParamState, Parameter, ParameterUpdate};
use pyo3::{prelude::*, types::PyDict};

use super::error::to_py_err;

// Unlike Option<T> arguments, this keeps an explicitly supplied None distinct
// from an omitted keyword. Only the Rust default constructs the absent case.
struct UpdateArg<T>(Option<T>);

impl<'a, 'py, T: FromPyObject<'a, 'py>> FromPyObject<'a, 'py> for UpdateArg<T> {
    type Error = T::Error;

    const INPUT_TYPE: pyo3::inspect::PyStaticExpr = T::INPUT_TYPE;

    fn extract(value: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let value = T::extract(value)?;
        Ok(Self(Some(value)))
    }
}

#[derive(Clone, FromPyObject, IntoPyObject, IntoPyObjectRef)]
enum InitialValue {
    Value(f64),
    Uniform((f64, f64)),
}

fn initial_value(initial: &InitialSpec) -> Option<InitialValue> {
    match initial {
        InitialSpec::Default => None,
        InitialSpec::Value(value) => Some(InitialValue::Value(*value)),
        InitialSpec::Uniform { min, max } => Some(InitialValue::Uniform((*min, *max))),
    }
}

#[pyclass(
    name = "ParameterUpdate",
    module = "laddu",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
/// Immutable edits to a named parameter's definition.
///
/// Omitted keywords leave their settings unchanged. Explicit ``None`` clears
/// or resets a setting. Apply one or more updates with :meth:`Model.with_parameters`.
/// Names belong to that method's mapping, so an update can be reused for several
/// parameters or models.
///
/// Parameters
/// ----------
/// fixed : float or None, optional
///     Fix at this value, or make free with ``None``. Fixing also sets the initial
///     value unless ``initial`` is explicitly supplied in the same update.
/// initial : float, tuple of float, or None, optional
///     Initial value, uniform ``(minimum, maximum)`` range, or ``None`` to restore
///     the default initialization rule. Does not change whether a parameter is fixed.
/// bounds : tuple of float or None, or None, optional
///     Replace both inclusive bounds. Each endpoint may be ``None``; passing
///     ``None`` for the whole field removes both bounds.
/// periodic : bool, optional
///     Whether the parameter is periodic. Requires finite ordered bounds on the
///     resulting parameter; ``None`` is not accepted.
/// scale : float or None, optional
///     Positive finite optimizer scale, or ``None`` to remove it.
/// unit, latex, description : str or None, optional
///     Replace the corresponding metadata, or clear it with ``None``.
///
/// Raises
/// ------
/// TypeError
///     If a keyword is unknown or a value has the wrong type.
/// ValueError
///     If a bounds tuple does not contain exactly two endpoints.
/// LadduError
///     If a supplied value, range, or scale is invalid. Checks involving existing
///     parameter settings happen when applying the update to a model.
pub struct PyParameterUpdate {
    pub(crate) inner: ParameterUpdate,
}

#[pymethods]
impl PyParameterUpdate {
    #[new]
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    #[pyo3(signature = (
        *,
        fixed: "float | None" = UpdateArg(None),
        initial: "float | tuple[float, float] | None" = UpdateArg(None),
        bounds: "tuple[float | None, float | None] | None" = UpdateArg(None),
        periodic: "bool" = UpdateArg(None),
        scale: "float | None" = UpdateArg(None),
        unit: "str | None" = UpdateArg(None),
        latex: "str | None" = UpdateArg(None),
        description: "str | None" = UpdateArg(None)
    ))]
    fn new(
        fixed: UpdateArg<Option<f64>>,
        initial: UpdateArg<Option<InitialValue>>,
        bounds: UpdateArg<Option<(Option<f64>, Option<f64>)>>,
        periodic: UpdateArg<bool>,
        scale: UpdateArg<Option<f64>>,
        unit: UpdateArg<Option<String>>,
        latex: UpdateArg<Option<String>>,
        description: UpdateArg<Option<String>>,
    ) -> PyResult<Self> {
        let inner = ParameterUpdate {
            state: fixed.0.map(|value| match value {
                Some(value) => ParamState::Fixed(value),
                None => ParamState::Free,
            }),
            initial: initial.0.map(|value| match value {
                Some(InitialValue::Value(value)) => InitialSpec::Value(value),
                Some(InitialValue::Uniform((min, max))) => InitialSpec::Uniform { min, max },
                None => InitialSpec::Default,
            }),
            bounds: bounds.0.map(|value| match value {
                Some((min, max)) => Bounds { min, max },
                None => Bounds::default(),
            }),
            periodic: periodic.0,
            scale: scale.0,
            unit: unit.0,
            latex: latex.0,
            description: description.0,
        };
        inner.validate().map_err(to_py_err)?;
        Ok(Self { inner })
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let fields = PyDict::new(py);
        if let Some(state) = &self.inner.state {
            fields.set_item(
                "fixed",
                match state {
                    ParamState::Free => None,
                    ParamState::Fixed(value) => Some(*value),
                },
            )?;
        }
        if let Some(initial) = &self.inner.initial {
            fields.set_item("initial", initial_value(initial))?;
        }
        if let Some(bounds) = &self.inner.bounds {
            fields.set_item(
                "bounds",
                (bounds.min.is_some() || bounds.max.is_some()).then_some((bounds.min, bounds.max)),
            )?;
        }
        if let Some(periodic) = self.inner.periodic {
            fields.set_item("periodic", periodic)?;
        }
        if let Some(scale) = self.inner.scale {
            fields.set_item("scale", scale)?;
        }
        for (name, value) in [
            ("unit", &self.inner.unit),
            ("latex", &self.inner.latex),
            ("description", &self.inner.description),
        ] {
            if let Some(value) = value {
                fields.set_item(name, value)?;
            }
        }
        let fields = fields
            .iter()
            .map(|(key, value)| Ok(format!("{}={}", key, value.repr()?)))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(format!("ParameterUpdate({})", fields.join(", ")))
    }
}

#[pyclass(
    name = "ParameterSpec",
    module = "laddu",
    frozen,
    get_all,
    skip_from_py_object
)]
#[derive(Clone)]
/// An immutable snapshot of a named parameter's complete definition.
///
/// Obtain records from :attr:`Model.parameter_specs`. They include fixed
/// parameters even when compilation has folded their contributions into constants.
/// To change a definition, construct a :class:`ParameterUpdate` and apply it with
/// :meth:`Model.with_parameters`.
pub struct PyParameterSpec {
    /// str: Unique parameter name.
    name: String,
    /// float or None: Fixed value, or ``None`` if the parameter is free.
    fixed: Option<f64>,
    /// float, tuple of float, or None: Initialization rule; ``None`` means the default.
    initial: Option<InitialValue>,
    /// tuple or None: Inclusive bounds; ``None`` means completely unbounded.
    bounds: Option<(Option<f64>, Option<f64>)>,
    /// bool: Whether the parameter is periodic over its bounds.
    periodic: bool,
    /// float or None: Characteristic optimizer scale.
    scale: Option<f64>,
    /// str or None: Human-readable unit label.
    unit: Option<String>,
    /// str or None: LaTeX label.
    latex: Option<String>,
    /// str or None: Human-readable description.
    description: Option<String>,
}

impl From<&Parameter> for PyParameterSpec {
    fn from(parameter: &Parameter) -> Self {
        let bounds = parameter.bounds_spec();
        Self {
            name: parameter.name().to_owned(),
            fixed: match parameter.state() {
                ParamState::Free => None,
                ParamState::Fixed(value) => Some(*value),
            },
            initial: initial_value(parameter.initial_spec()),
            bounds: (bounds.min.is_some() || bounds.max.is_some())
                .then_some((bounds.min, bounds.max)),
            periodic: parameter.is_periodic(),
            scale: parameter.scale(),
            unit: parameter.unit_label().map(str::to_owned),
            latex: parameter.latex_label().map(str::to_owned),
            description: parameter.description_text().map(str::to_owned),
        }
    }
}

#[pymethods]
impl PyParameterSpec {
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let fields = PyDict::new(py);
        fields.set_item("name", &self.name)?;
        fields.set_item("fixed", self.fixed)?;
        fields.set_item("initial", &self.initial)?;
        fields.set_item("bounds", self.bounds)?;
        fields.set_item("periodic", self.periodic)?;
        fields.set_item("scale", self.scale)?;
        fields.set_item("unit", &self.unit)?;
        fields.set_item("latex", &self.latex)?;
        fields.set_item("description", &self.description)?;
        let fields = fields
            .iter()
            .map(|(key, value)| Ok(format!("{}={}", key, value.repr()?)))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(format!("ParameterSpec({})", fields.join(", ")))
    }
}
