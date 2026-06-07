//! Parameter handles, assembled parameter storage, and parameter identifiers.

use std::{hash::Hash, sync::Arc};

use fastrand_contrib::RngExt;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

#[derive(Clone, Default, Serialize, Deserialize, Debug)]
struct ParameterMetadata {
    /// The name of the parameter.
    name: String,
    /// If `Some`, this parameter is fixed to the given value. If `None`, it is free.
    fixed: Option<f64>,
    /// If `Some`, this is used to generate initial values for fits. If `None`, map-level helpers
    /// use their documented default.
    initial: Option<InitialValue>,
    /// Optional bounds which may be automatically used by optimizers. `None` represents no bound
    /// in the given direction.
    bounds: (Option<f64>, Option<f64>),
    /// An optional unit string which may be used to annotate the parameter.
    unit: Option<String>,
    /// Optional LaTeX representation of the parameter.
    latex: Option<String>,
    /// Optional description of the parameter.
    description: Option<String>,
}

/// An enum containing either a named free parameter or a constant value.
#[derive(Clone, Default, Serialize, Deserialize, Debug)]
pub struct Parameter(Arc<Mutex<ParameterMetadata>>);

impl PartialEq for Parameter {
    fn eq(&self, other: &Self) -> bool {
        self.0.lock().name == other.0.lock().name
    }
}
impl Eq for Parameter {}
impl Hash for Parameter {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.lock().name.hash(state);
    }
}

/// A serializable specification for generating a parameter's initial fit value.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum InitialValue {
    /// Always use the same initial value.
    Fixed(f64),
    /// Sample uniformly from the closed interval `[min, max]`.
    Uniform {
        /// Lower end of the interval.
        min: f64,
        /// Upper end of the interval.
        max: f64,
    },
}

impl InitialValue {
    /// Construct a fixed initial value.
    pub fn fixed(value: f64) -> Self {
        Self::Fixed(value)
    }

    /// Construct a uniform initial value.
    pub fn uniform(min: f64, max: f64) -> Self {
        Self::Uniform { min, max }
    }

    /// Sample an initial value from this specification.
    pub fn sample(self, rng: &mut fastrand::Rng) -> f64 {
        match self {
            Self::Fixed(value) => value,
            Self::Uniform { min, max } => rng.f64_range(min..=max),
        }
    }
}

/// Helper trait to convert values into [`InitialValue`] specifications.
pub trait IntoInitialValue {
    /// Convert to an initial-value specification.
    fn into_initial_value(self) -> InitialValue;
}

impl IntoInitialValue for f64 {
    fn into_initial_value(self) -> InitialValue {
        InitialValue::Fixed(self)
    }
}

impl IntoInitialValue for (f64, f64) {
    fn into_initial_value(self) -> InitialValue {
        InitialValue::Uniform {
            min: self.0,
            max: self.1,
        }
    }
}

impl IntoInitialValue for InitialValue {
    fn into_initial_value(self) -> InitialValue {
        self
    }
}

/// Helper trait to convert values to bounds-like [`Option<f64>`].
pub trait IntoBound {
    /// Convert to a bound.
    fn into_bound(self) -> Option<f64>;
}
impl IntoBound for f64 {
    fn into_bound(self) -> Option<f64> {
        Some(self)
    }
}
impl IntoBound for Option<f64> {
    fn into_bound(self) -> Option<f64> {
        self
    }
}

impl Parameter {
    /// Create a free (floating) parameter with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self(Arc::new(Mutex::new(ParameterMetadata {
            name: name.into(),
            ..Default::default()
        })))
    }

    /// Create a fixed parameter with the given name and value.
    pub fn new_fixed(name: impl Into<String>, value: f64) -> Self {
        Self(Arc::new(Mutex::new(ParameterMetadata {
            name: name.into(),
            fixed: Some(value),
            initial: Some(InitialValue::Fixed(value)),
            ..Default::default()
        })))
    }

    /// Return the current parameter name.
    pub fn name(&self) -> String {
        self.0.lock().name.clone()
    }

    /// Return the fixed value when the parameter is fixed.
    pub fn fixed(&self) -> Option<f64> {
        self.0.lock().fixed
    }

    /// Return the current initial value, if one is set.
    pub fn initial(&self) -> Option<InitialValue> {
        self.0.lock().initial
    }

    /// Return the current lower and upper bounds.
    pub fn bounds(&self) -> (Option<f64>, Option<f64>) {
        self.0.lock().bounds
    }

    /// Return the optional unit label.
    pub fn unit(&self) -> Option<String> {
        self.0.lock().unit.clone()
    }

    /// Return the optional LaTeX label.
    pub fn latex(&self) -> Option<String> {
        self.0.lock().latex.clone()
    }

    /// Return the optional human-readable description.
    pub fn description(&self) -> Option<String> {
        self.0.lock().description.clone()
    }

    /// Helper method to set the name of a parameter.
    pub(crate) fn set_name(&self, name: impl Into<String>) {
        self.0.lock().name = name.into();
    }

    /// Helper method to set the fixed value of a parameter.
    pub fn set_fixed_value(&self, value: Option<f64>) {
        let mut guard = self.0.lock();
        if let Some(value) = value {
            guard.fixed = Some(value);
            guard.initial = Some(InitialValue::Fixed(value));
        } else {
            guard.fixed = None;
        }
    }

    /// Helper method to set the initial value of a parameter.
    ///
    /// # Panics
    ///
    /// This method panics if the parameter is fixed.
    pub fn set_initial<I: IntoInitialValue>(&self, value: I) {
        assert!(
            self.is_free(),
            "cannot manually set `initial` on a fixed parameter"
        );
        self.0.lock().initial = Some(value.into_initial_value());
    }

    /// Helper method to set the bounds of a parameter.
    pub fn set_bounds<L, U>(&self, min: L, max: U)
    where
        L: IntoBound,
        U: IntoBound,
    {
        self.0.lock().bounds = (IntoBound::into_bound(min), IntoBound::into_bound(max));
    }

    /// Helper method to set the unit of a parameter.
    pub fn set_unit(&self, unit: impl Into<String>) {
        self.0.lock().unit = Some(unit.into());
    }

    /// Helper method to set the LaTeX representation of a parameter.
    pub fn set_latex(&self, latex: impl Into<String>) {
        self.0.lock().latex = Some(latex.into());
    }

    /// Helper method to set the description of a parameter.
    pub fn set_description(&self, description: impl Into<String>) {
        self.0.lock().description = Some(description.into());
    }

    /// Is this parameter free?
    pub fn is_free(&self) -> bool {
        self.0.lock().fixed.is_none()
    }

    /// Is this parameter fixed?
    pub fn is_fixed(&self) -> bool {
        self.0.lock().fixed.is_some()
    }
}

/// Convenience macro for creating parameters. Usage:
/// `parameter!("name")` for a free parameter, or `parameter!("name", 1.0)` for a fixed one.
#[macro_export]
macro_rules! parameter {
    ($name:expr) => {{
        $crate::parameters::Parameter::new($name)
    }};

    ($name:expr, $value:expr) => {{
        let p = $crate::parameters::Parameter::new($name);
        p.set_fixed_value(Some($value));
        p
    }};

    ($name:expr, $($rest:tt)+) => {{
        let p = $crate::parameters::Parameter::new($name);
        $crate::parameter!(@parse p, [fixed = false, initial = false]; $($rest)+);
        p
    }};

    (@parse $p:ident, [fixed = $f:tt, initial = $i:tt]; ) => {};

    (@parse $p:ident, [fixed = false, initial = false]; fixed : $value:expr $(, $($rest:tt)*)?) => {{
        $p.set_fixed_value(Some($value));
        $crate::parameter!(@parse $p, [fixed = true, initial = false]; $($($rest)*)?);
    }};

    (@parse $p:ident, [fixed = false, initial = false]; initial : $value:expr $(, $($rest:tt)*)?) => {{
        $p.set_initial($value);
        $crate::parameter!(@parse $p, [fixed = false, initial = true]; $($($rest)*)?);
    }};

    (@parse $p:ident, [fixed = true, initial = false]; initial : $value:expr $(, $($rest:tt)*)?) => {
        compile_error!("parameter!: cannot specify both `fixed` and `initial`");
    };

    (@parse $p:ident, [fixed = false, initial = true]; fixed : $value:expr $(, $($rest:tt)*)?) => {
        compile_error!("parameter!: cannot specify both `fixed` and `initial`");
    };

    (@parse $p:ident, [fixed = $f:tt, initial = $i:tt]; bounds : ($min:expr, $max:expr) $(, $($rest:tt)*)?) => {{
        $p.set_bounds($min, $max);
        $crate::parameter!(@parse $p, [fixed = $f, initial = $i]; $($($rest)*)?);
    }};

    (@parse $p:ident, [fixed = $f:tt, initial = $i:tt]; unit : $value:expr $(, $($rest:tt)*)?) => {{
        $p.set_unit($value);
        $crate::parameter!(@parse $p, [fixed = $f, initial = $i]; $($($rest)*)?);
    }};

    (@parse $p:ident, [fixed = $f:tt, initial = $i:tt]; latex : $value:expr $(, $($rest:tt)*)?) => {{
        $p.set_latex($value);
        $crate::parameter!(@parse $p, [fixed = $f, initial = $i]; $($($rest)*)?);
    }};

    (@parse $p:ident, [fixed = $f:tt, initial = $i:tt]; description : $value:expr $(, $($rest:tt)*)?) => {{
        $p.set_description($value);
        $crate::parameter!(@parse $p, [fixed = $f, initial = $i]; $($($rest)*)?);
    }};
}
