use std::{
    fmt::{Debug, Display},
    sync::Arc,
};

use auto_ops::*;
use dyn_clone::DynClone;
use nalgebra::{ComplexField, DVector};
use num::Complex;

use parking_lot::RwLock;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    data::{Dataset, Event},
    resources::{Cache, Parameters, Resources},
    Float, LadduError,
};

/// The Breit-Wigner amplitude.
pub mod breit_wigner;
/// Common amplitudes (like a scalar value which just contains a single free parameter).
pub mod common;
/// Amplitudes related to the K-Matrix formalism.
pub mod kmatrix;
/// A spherical harmonic amplitude.
pub mod ylm;
/// A polarized spherical harmonic amplitude.
pub mod zlm;

/// An enum containing either a named free parameter or a constant value.
#[derive(Clone, Default, Serialize, Deserialize)]
pub enum ParameterLike {
    /// A named free parameter.
    Parameter(String),
    /// A constant value.
    Constant(Float),
    /// An uninitialized parameter-like structure (typically used as the value given in an
    /// [`Amplitude`] constructor before the [`Amplitude::register`] method is called).
    #[default]
    Uninit,
}

/// Shorthand for generating a named free parameter.
pub fn parameter(name: &str) -> ParameterLike {
    ParameterLike::Parameter(name.to_string())
}

/// Shorthand for generating a constant value (which acts like a fixed parameter).
pub fn constant(value: Float) -> ParameterLike {
    ParameterLike::Constant(value)
}

/// This is the only required trait for writing new amplitude-like structures for this
/// crate. Users need only implement the [`register`](Amplitude::register)
/// method to register parameters, cached values, and the amplitude itself with an input
/// [`Resources`] struct and the [`compute`](Amplitude::compute) method to actually carry
/// out the calculation. [`Amplitude`]-implementors are required to implement [`Clone`] and can
/// optionally implement a [`precompute`](Amplitude::precompute) method to calculate and
/// cache values which do not depend on free parameters.
///
/// See [`BreitWigner`](breit_wigner::BreitWigner), [`Ylm`](ylm::Ylm), and [`Zlm`](zlm::Zlm) for examples which use all of these features.
#[typetag::serde(tag = "type")]
pub trait Amplitude: DynClone + Send + Sync {
    /// This method should be used to tell the [`Resources`] manager about all of
    /// the free parameters and cached values used by this [`Amplitude`]. It should end by
    /// returning an [`AmplitudeID`], which can be obtained from the
    /// [`Resources::register_amplitude`] method.
    fn register(&mut self, resources: &mut Resources) -> Result<AmplitudeID, LadduError>;
    /// This method can be used to do some critical calculations ahead of time and
    /// store them in a [`Cache`]. These values can only depend on the data in an [`Event`],
    /// not on any free parameters in the fit. This method is opt-in since it is not required
    /// to make a functioning [`Amplitude`].
    #[allow(unused_variables)]
    fn precompute(&self, event: &Event, cache: &mut Cache) {}
    /// Evaluates [`Amplitude::precompute`] over ever [`Event`] in a [`Dataset`].
    #[cfg(feature = "rayon")]
    fn precompute_all(&self, dataset: &Dataset, resources: &mut Resources) {
        dataset
            .events
            .par_iter()
            .zip(resources.caches.par_iter_mut())
            .for_each(|(event, cache)| {
                self.precompute(event, cache);
            })
    }
    /// Evaluates [`Amplitude::precompute`] over ever [`Event`] in a [`Dataset`].
    #[cfg(not(feature = "rayon"))]
    fn precompute_all(&self, dataset: &Dataset, resources: &mut Resources) {
        dataset
            .events
            .iter()
            .zip(resources.caches.iter_mut())
            .for_each(|(event, cache)| self.precompute(event, cache))
    }
    /// This method constitutes the main machinery of an [`Amplitude`], returning the actual
    /// calculated value for a particular [`Event`] and set of [`Parameters`]. See those
    /// structs, as well as [`Cache`], for documentation on their available methods. For the
    /// most part, [`Event`]s can be interacted with via
    /// [`Variable`](crate::utils::variables::Variable)s, while [`Parameters`] and the
    /// [`Cache`] are more like key-value storage accessed by
    /// [`ParameterID`](crate::resources::ParameterID)s and several different types of cache
    /// IDs.
    fn compute(&self, parameters: &Parameters, event: &Event, cache: &Cache) -> Complex<Float>;

    /// This method yields the gradient of a particular [`Amplitude`] at a point specified
    /// by a particular [`Event`] and set of [`Parameters`]. See those structs, as well as
    /// [`Cache`], for documentation on their available methods. For the most part,
    /// [`Event`]s can be interacted with via [`Variable`](crate::utils::variables::Variable)s,
    /// while [`Parameters`] and the [`Cache`] are more like key-value storage accessed by
    /// [`ParameterID`](crate::resources::ParameterID)s and several different types of cache
    /// IDs. If the analytic version of the gradient is known, this method can be overwritten to
    /// improve performance for some derivative-using methods of minimization. The default
    /// implementation calculates a central finite difference across all parameters, regardless of
    /// whether or not they are used in the [`Amplitude`].
    ///
    /// In the future, it may be possible to automatically implement this with the indices of
    /// registered free parameters, but until then, the [`Amplitude::central_difference_with_indices`]
    /// method can be used to conveniently only calculate central differences for the parameters
    /// which are used by the [`Amplitude`].
    fn compute_gradient(
        &self,
        parameters: &Parameters,
        event: &Event,
        cache: &Cache,
        gradient: &mut DVector<Complex<Float>>,
    ) {
        self.central_difference_with_indices(
            &Vec::from_iter(0..parameters.len()),
            parameters,
            event,
            cache,
            gradient,
        )
    }

    /// A helper function to implement a central difference only on indices which correspond to
    /// free parameters in the [`Amplitude`]. For example, if an [`Amplitude`] contains free
    /// parameters registered to indices 1, 3, and 5 of the its internal parameters array, then
    /// running this with those indices will compute a central finite difference derivative for
    /// those coordinates only, since the rest can be safely assumed to be zero.
    fn central_difference_with_indices(
        &self,
        indices: &[usize],
        parameters: &Parameters,
        event: &Event,
        cache: &Cache,
        gradient: &mut DVector<Complex<Float>>,
    ) {
        let x = parameters.parameters.to_owned();
        let constants = parameters.constants.to_owned();
        let h: DVector<Float> = x
            .iter()
            .map(|&xi| Float::cbrt(Float::EPSILON) * (xi.abs() + 1.0))
            .collect::<Vec<_>>()
            .into();
        for i in indices {
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();
            x_plus[*i] += h[*i];
            x_minus[*i] -= h[*i];
            let f_plus = self.compute(&Parameters::new(&x_plus, &constants), event, cache);
            let f_minus = self.compute(&Parameters::new(&x_minus, &constants), event, cache);
            gradient[*i] = (f_plus - f_minus) / (2.0 * h[*i]);
        }
    }
}

dyn_clone::clone_trait_object!(Amplitude);

#[derive(Debug)]
pub(crate) struct AmplitudeValues(pub(crate) Vec<Complex<Float>>);

#[derive(Debug)]
pub(crate) struct GradientValues(pub(crate) Vec<DVector<Complex<Float>>>);

/// A tag which refers to a registered [`Amplitude`]. This is the base object which can be used to
/// build [`Expression`]s and should be obtained from the [`Manager::register`] method.
#[derive(Clone, Default, Debug, Serialize, Deserialize)]
pub struct AmplitudeID(pub(crate) String, pub(crate) usize);

impl Display for AmplitudeID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<AmplitudeID> for Expression {
    fn from(value: AmplitudeID) -> Self {
        Self::Amp(value)
    }
}

/// An expression tree which contains [`AmplitudeID`]s and operators over them.
#[derive(Clone, Serialize, Deserialize, Default)]
pub enum Expression {
    #[default]
    /// A expression equal to zero.
    Zero,
    /// A registered [`Amplitude`] referenced by an [`AmplitudeID`].
    Amp(AmplitudeID),
    /// The sum of two [`Expression`]s.
    Add(Box<Expression>, Box<Expression>),
    /// The product of two [`Expression`]s.
    Mul(Box<Expression>, Box<Expression>),
    /// The real part of an [`Expression`].
    Real(Box<Expression>),
    /// The imaginary part of an [`Expression`].
    Imag(Box<Expression>),
    /// The absolute square of an [`Expression`].
    NormSqr(Box<Expression>),
}

impl Debug for Expression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.write_tree(f, "", "", "")
    }
}

impl Display for Expression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl_op_ex!(+ |a: &Expression, b: &Expression| -> Expression { Expression::Add(Box::new(a.clone()), Box::new(b.clone()))});
impl_op_ex!(*|a: &Expression, b: &Expression| -> Expression {
    Expression::Mul(Box::new(a.clone()), Box::new(b.clone()))
});
impl_op_ex_commutative!(+ |a: &AmplitudeID, b: &Expression| -> Expression { Expression::Add(Box::new(Expression::Amp(a.clone())), Box::new(b.clone()))});
impl_op_ex_commutative!(*|a: &AmplitudeID, b: &Expression| -> Expression {
    Expression::Mul(Box::new(Expression::Amp(a.clone())), Box::new(b.clone()))
});
impl_op_ex!(+ |a: &AmplitudeID, b: &AmplitudeID| -> Expression { Expression::Add(Box::new(Expression::Amp(a.clone())), Box::new(Expression::Amp(b.clone())))});
impl_op_ex!(*|a: &AmplitudeID, b: &AmplitudeID| -> Expression {
    Expression::Mul(
        Box::new(Expression::Amp(a.clone())),
        Box::new(Expression::Amp(b.clone())),
    )
});

impl AmplitudeID {
    /// Takes the real part of the given [`Amplitude`].
    pub fn real(&self) -> Expression {
        Expression::Real(Box::new(Expression::Amp(self.clone())))
    }
    /// Takes the imaginary part of the given [`Amplitude`].
    pub fn imag(&self) -> Expression {
        Expression::Imag(Box::new(Expression::Amp(self.clone())))
    }
    /// Takes the absolute square of the given [`Amplitude`].
    pub fn norm_sqr(&self) -> Expression {
        Expression::NormSqr(Box::new(Expression::Amp(self.clone())))
    }
}

impl Expression {
    pub(crate) fn evaluate(&self, amplitude_values: &AmplitudeValues) -> Complex<Float> {
        match self {
            Expression::Amp(aid) => amplitude_values.0[aid.1],
            Expression::Add(a, b) => a.evaluate(amplitude_values) + b.evaluate(amplitude_values),
            Expression::Mul(a, b) => a.evaluate(amplitude_values) * b.evaluate(amplitude_values),
            Expression::Real(a) => Complex::new(a.evaluate(amplitude_values).re, 0.0),
            Expression::Imag(a) => Complex::new(a.evaluate(amplitude_values).im, 0.0),
            Expression::NormSqr(a) => Complex::new(a.evaluate(amplitude_values).norm_sqr(), 0.0),
            Expression::Zero => Complex::ZERO,
        }
    }
    pub(crate) fn evaluate_gradient(
        &self,
        amplitude_values: &AmplitudeValues,
        gradient_values: &GradientValues,
    ) -> DVector<Complex<Float>> {
        match self {
            Expression::Amp(aid) => gradient_values.0[aid.1].clone(),
            Expression::Add(a, b) => {
                a.evaluate_gradient(amplitude_values, gradient_values)
                    + b.evaluate_gradient(amplitude_values, gradient_values)
            }
            Expression::Mul(a, b) => {
                let f_a = a.evaluate(amplitude_values);
                let f_b = b.evaluate(amplitude_values);
                b.evaluate_gradient(amplitude_values, gradient_values)
                    .map(|g| g * f_a)
                    + a.evaluate_gradient(amplitude_values, gradient_values)
                        .map(|g| g * f_b)
            }
            Expression::Real(a) => a
                .evaluate_gradient(amplitude_values, gradient_values)
                .map(|g| Complex::new(g.re, 0.0)),
            Expression::Imag(a) => a
                .evaluate_gradient(amplitude_values, gradient_values)
                .map(|g| Complex::new(g.im, 0.0)),
            Expression::NormSqr(a) => {
                let conj_f_a = a.evaluate(amplitude_values).conjugate();
                a.evaluate_gradient(amplitude_values, gradient_values)
                    .map(|g| Complex::new(2.0 * (g * conj_f_a).re, 0.0))
            }
            Expression::Zero => DVector::zeros(0),
        }
    }
    /// Takes the real part of the given [`Expression`].
    pub fn real(&self) -> Self {
        Self::Real(Box::new(self.clone()))
    }
    /// Takes the imaginary part of the given [`Expression`].
    pub fn imag(&self) -> Self {
        Self::Imag(Box::new(self.clone()))
    }
    /// Takes the absolute square of the given [`Expression`].
    pub fn norm_sqr(&self) -> Self {
        Self::NormSqr(Box::new(self.clone()))
    }

    /// Credit to Daniel Janus: <https://blog.danieljanus.pl/2023/07/20/iterating-trees/>
    fn write_tree(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        parent_prefix: &str,
        immediate_prefix: &str,
        parent_suffix: &str,
    ) -> std::fmt::Result {
        let display_string = match self {
            Self::Amp(aid) => aid.0.clone(),
            Self::Add(_, _) => "+".to_string(),
            Self::Mul(_, _) => "*".to_string(),
            Self::Real(_) => "Re".to_string(),
            Self::Imag(_) => "Im".to_string(),
            Self::NormSqr(_) => "NormSqr".to_string(),
            Self::Zero => "0".to_string(),
        };
        writeln!(f, "{}{}{}", parent_prefix, immediate_prefix, display_string)?;
        match self {
            Self::Amp(_) | Self::Zero => {}
            Self::Add(a, b) | Self::Mul(a, b) => {
                let terms = [a, b];
                let mut it = terms.iter().peekable();
                let child_prefix = format!("{}{}", parent_prefix, parent_suffix);
                while let Some(child) = it.next() {
                    match it.peek() {
                        Some(_) => child.write_tree(f, &child_prefix, "├─ ", "│  "),
                        None => child.write_tree(f, &child_prefix, "└─ ", "   "),
                    }?;
                }
            }
            Self::Real(a) | Self::Imag(a) | Self::NormSqr(a) => {
                let child_prefix = format!("{}{}", parent_prefix, parent_suffix);
                a.write_tree(f, &child_prefix, "└─ ", "   ")?;
            }
        }
        Ok(())
    }
}

/// A manager which can be used to register [`Amplitude`]s with [`Resources`]. This structure is
/// essential to any analysis and should be constructed using the [`Manager::default()`] method.
#[derive(Default, Clone, Serialize, Deserialize)]
pub struct Manager {
    amplitudes: Vec<Box<dyn Amplitude>>,
    resources: Resources,
}

impl Manager {
    /// Get the list of parameter names in the order they appear in the [`Manager`]'s [`Resources`] field.
    pub fn parameters(&self) -> Vec<String> {
        self.resources.parameters.iter().cloned().collect()
    }
    /// Register the given [`Amplitude`] and return an [`AmplitudeID`] that can be used to build
    /// [`Expression`]s.
    ///
    /// # Errors
    ///
    /// The [`Amplitude`]'s name must be unique and not already
    /// registered, else this will return a [`RegistrationError`][LadduError::RegistrationError].
    pub fn register(&mut self, amplitude: Box<dyn Amplitude>) -> Result<AmplitudeID, LadduError> {
        let mut amp = amplitude.clone();
        let aid = amp.register(&mut self.resources)?;
        self.amplitudes.push(amp);
        Ok(aid)
    }
    /// Turns an [`Expression`] made from registered [`Amplitude`]s into a [`Model`].
    pub fn model(&self, expression: &Expression) -> Model {
        Model {
            manager: self.clone(),
            expression: expression.clone(),
        }
    }
}

/// A struct which contains a set of registerd [`Amplitude`]s (inside a [`Manager`])
/// and an [`Expression`].
///
/// This struct implements [`serde::Serialize`] and [`serde::Deserialize`] and is intended
/// to be used to store models to disk.
#[derive(Clone, Serialize, Deserialize)]
pub struct Model {
    pub(crate) manager: Manager,
    pub(crate) expression: Expression,
}

impl Model {
    /// Get the list of parameter names in the order they appear in the [`Model`]'s [`Manager`] field.
    pub fn parameters(&self) -> Vec<String> {
        self.manager.parameters()
    }
    /// Create an [`Evaluator`] which can compute the result of the internal [`Expression`] built on
    /// registered [`Amplitude`]s over the given [`Dataset`]. This method precomputes any relevant
    /// information over the [`Event`]s in the [`Dataset`].
    pub fn load(&self, dataset: &Arc<Dataset>) -> Evaluator {
        let loaded_resources = Arc::new(RwLock::new(self.manager.resources.clone()));
        loaded_resources.write().reserve_cache(dataset.len());
        for amplitude in &self.manager.amplitudes {
            amplitude.precompute_all(dataset, &mut loaded_resources.write());
        }
        Evaluator {
            amplitudes: self.manager.amplitudes.clone(),
            resources: loaded_resources.clone(),
            dataset: dataset.clone(),
            expression: self.expression.clone(),
        }
    }
}

/// A structure which can be used to evaluate the stored [`Expression`] built on registered
/// [`Amplitude`]s. This contains a [`Resources`] struct which already contains cached values for
/// precomputed [`Amplitude`]s and any relevant free parameters and constants.
#[derive(Clone)]
pub struct Evaluator {
    pub(crate) amplitudes: Vec<Box<dyn Amplitude>>,
    pub(crate) resources: Arc<RwLock<Resources>>,
    pub(crate) dataset: Arc<Dataset>,
    pub(crate) expression: Expression,
}

impl Evaluator {
    /// Get the list of parameter names in the order they appear in the [`Evaluator::evaluate`]
    /// method.
    pub fn parameters(&self) -> Vec<String> {
        self.resources.read().parameters.iter().cloned().collect()
    }
    /// Activate an [`Amplitude`] by name.
    pub fn activate<T: AsRef<str>>(&self, name: T) -> Result<(), LadduError> {
        self.resources.write().activate(name)
    }
    /// Activate several [`Amplitude`]s by name.
    pub fn activate_many<T: AsRef<str>>(&self, names: &[T]) -> Result<(), LadduError> {
        self.resources.write().activate_many(names)
    }
    /// Activate all registered [`Amplitude`]s.
    pub fn activate_all(&self) {
        self.resources.write().activate_all();
    }
    /// Dectivate an [`Amplitude`] by name.
    pub fn deactivate<T: AsRef<str>>(&self, name: T) -> Result<(), LadduError> {
        self.resources.write().deactivate(name)
    }
    /// Deactivate several [`Amplitude`]s by name.
    pub fn deactivate_many<T: AsRef<str>>(&self, names: &[T]) -> Result<(), LadduError> {
        self.resources.write().deactivate_many(names)
    }
    /// Deactivate all registered [`Amplitude`]s.
    pub fn deactivate_all(&self) {
        self.resources.write().deactivate_all();
    }
    /// Isolate an [`Amplitude`] by name (deactivate the rest).
    pub fn isolate<T: AsRef<str>>(&self, name: T) -> Result<(), LadduError> {
        self.resources.write().isolate(name)
    }
    /// Isolate several [`Amplitude`]s by name (deactivate the rest).
    pub fn isolate_many<T: AsRef<str>>(&self, names: &[T]) -> Result<(), LadduError> {
        self.resources.write().isolate_many(names)
    }
    /// Evaluate the stored [`Expression`] over the events in the [`Dataset`] stored by the
    /// [`Evaluator`] with the given values for free parameters.
    #[cfg(feature = "rayon")]
    pub fn evaluate(&self, parameters: &[Float]) -> Vec<Complex<Float>> {
        let resources = self.resources.read();
        let parameters = Parameters::new(parameters, &resources.constants);
        let amplitude_values_vec: Vec<AmplitudeValues> = self
            .dataset
            .events
            .par_iter()
            .zip(resources.caches.par_iter())
            .map(|(event, cache)| {
                AmplitudeValues(
                    self.amplitudes
                        .iter()
                        .zip(resources.active.iter())
                        .map(|(amp, active)| {
                            if *active {
                                amp.compute(&parameters, event, cache)
                            } else {
                                Complex::new(0.0, 0.0)
                            }
                        })
                        .collect(),
                )
            })
            .collect();
        amplitude_values_vec
            .par_iter()
            .map(|amplitude_values| self.expression.evaluate(amplitude_values))
            .collect()
    }
    /// Evaluate the stored [`Expression`] over the events in the [`Dataset`] stored by the
    /// [`Evaluator`] with the given values for free parameters.
    #[cfg(not(feature = "rayon"))]
    pub fn evaluate(&self, parameters: &[Float]) -> Vec<Complex<Float>> {
        let resources = self.resources.read();
        let parameters = Parameters::new(parameters, &resources.constants);
        let amplitude_values_vec: Vec<AmplitudeValues> = self
            .dataset
            .events
            .iter()
            .zip(resources.caches.iter())
            .map(|(event, cache)| {
                AmplitudeValues(
                    self.amplitudes
                        .iter()
                        .zip(resources.active.iter())
                        .map(|(amp, active)| {
                            if *active {
                                amp.compute(&parameters, event, cache)
                            } else {
                                Complex::new(0.0, 0.0)
                            }
                        })
                        .collect(),
                )
            })
            .collect();
        amplitude_values_vec
            .iter()
            .map(|amplitude_values| self.expression.evaluate(amplitude_values))
            .collect()
    }
    /// Evaluate gradient of the stored [`Expression`] over the events in the [`Dataset`] stored by the
    /// [`Evaluator`] with the given values for free parameters.
    #[cfg(feature = "rayon")]
    pub fn evaluate_gradient(&self, parameters: &[Float]) -> Vec<DVector<Complex<Float>>> {
        let resources = self.resources.read();
        let parameters = Parameters::new(parameters, &resources.constants);
        let amplitude_values_and_gradient_vec: Vec<(AmplitudeValues, GradientValues)> = self
            .dataset
            .events
            .par_iter()
            .zip(resources.caches.par_iter())
            .map(|(event, cache)| {
                let mut gradient_values =
                    vec![DVector::zeros(parameters.len()); self.amplitudes.len()];
                self.amplitudes
                    .iter()
                    .zip(resources.active.iter())
                    .zip(gradient_values.iter_mut())
                    .for_each(|((amp, active), grad)| {
                        if *active {
                            amp.compute_gradient(&parameters, event, cache, grad)
                        }
                    });
                (
                    AmplitudeValues(
                        self.amplitudes
                            .iter()
                            .zip(resources.active.iter())
                            .map(|(amp, active)| {
                                if *active {
                                    amp.compute(&parameters, event, cache)
                                } else {
                                    Complex::new(0.0, 0.0)
                                }
                            })
                            .collect(),
                    ),
                    GradientValues(gradient_values),
                )
            })
            .collect();
        amplitude_values_and_gradient_vec
            .par_iter()
            .map(|(amplitude_values, gradient_values)| {
                self.expression
                    .evaluate_gradient(amplitude_values, gradient_values)
            })
            .collect()
    }
    /// Evaluate the stored [`Expression`] over the events in the [`Dataset`] stored by the
    /// [`Evaluator`] with the given values for free parameters.
    #[cfg(not(feature = "rayon"))]
    pub fn evaluate_gradient(&self, parameters: &[Float]) -> Vec<DVector<Complex<Float>>> {
        let resources = self.resources.read();
        let parameters = Parameters::new(parameters, &resources.constants);
        let amplitude_values_and_gradient_vec: Vec<(AmplitudeValues, GradientValues)> = self
            .dataset
            .events
            .iter()
            .zip(resources.caches.iter())
            .map(|(event, cache)| {
                let mut gradient_values =
                    vec![DVector::zeros(parameters.len()); self.amplitudes.len()];
                self.amplitudes
                    .iter()
                    .zip(resources.active.iter())
                    .zip(gradient_values.iter_mut())
                    .for_each(|((amp, active), grad)| {
                        if *active {
                            amp.compute_gradient(&parameters, event, cache, grad)
                        }
                    });
                (
                    AmplitudeValues(
                        self.amplitudes
                            .iter()
                            .zip(resources.active.iter())
                            .map(|(amp, active)| {
                                if *active {
                                    amp.compute(&parameters, event, cache)
                                } else {
                                    Complex::new(0.0, 0.0)
                                }
                            })
                            .collect(),
                    ),
                    GradientValues(gradient_values),
                )
            })
            .collect();
        amplitude_values_and_gradient_vec
            .iter()
            .map(|(amplitude_values, gradient_values)| {
                self.expression
                    .evaluate_gradient(amplitude_values, gradient_values)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::data::{test_dataset, test_event};

    use super::*;
    use approx::assert_relative_eq;
    use common::{ComplexScalar, Scalar};

    #[test]
    fn test_constant_amplitude() {
        let mut manager = Manager::default();
        let amp = Scalar::new("constant", constant(2.0));
        let aid = manager.register(amp).unwrap();
        let dataset = Arc::new(Dataset {
            events: vec![Arc::new(test_event())],
        });
        let expr = Expression::Amp(aid);
        let model = manager.model(&expr);
        let evaluator = model.load(&dataset);
        let result = evaluator.evaluate(&[]);
        assert_eq!(result[0], Complex::from(2.0));
    }

    #[test]
    fn test_parametric_amplitude() {
        let mut manager = Manager::default();
        let amp = Scalar::new("parametric", parameter("test_param"));
        let aid = manager.register(amp).unwrap();
        let dataset = Arc::new(test_dataset());
        let expr = Expression::Amp(aid);
        let model = manager.model(&expr);
        let evaluator = model.load(&dataset);
        let result = evaluator.evaluate(&[3.0]);
        assert_eq!(result[0], Complex::from(3.0));
    }

    #[test]
    fn test_expression_operations() {
        let mut manager = Manager::default();
        let amp1 = ComplexScalar::new("const1", constant(2.0), constant(0.0));
        let amp2 = ComplexScalar::new("const2", constant(0.0), constant(1.0));
        let amp3 = ComplexScalar::new("const3", constant(3.0), constant(4.0));

        let aid1 = manager.register(amp1).unwrap();
        let aid2 = manager.register(amp2).unwrap();
        let aid3 = manager.register(amp3).unwrap();

        let dataset = Arc::new(test_dataset());

        // Test (amp) addition
        let expr_add = &aid1 + &aid2;
        let model_add = manager.model(&expr_add);
        let eval_add = model_add.load(&dataset);
        let result_add = eval_add.evaluate(&[]);
        assert_eq!(result_add[0], Complex::new(2.0, 1.0));

        // Test (amp) multiplication
        let expr_mul = &aid1 * &aid2;
        let model_mul = manager.model(&expr_mul);
        let eval_mul = model_mul.load(&dataset);
        let result_mul = eval_mul.evaluate(&[]);
        assert_eq!(result_mul[0], Complex::new(0.0, 2.0));

        // Test (expr) addition
        let expr_add2 = &expr_add + &expr_mul;
        let model_add2 = manager.model(&expr_add2);
        let eval_add2 = model_add2.load(&dataset);
        let result_add2 = eval_add2.evaluate(&[]);
        assert_eq!(result_add2[0], Complex::new(2.0, 3.0));

        // Test (expr) multiplication
        let expr_mul2 = &expr_add * &expr_mul;
        let model_mul2 = manager.model(&expr_mul2);
        let eval_mul2 = model_mul2.load(&dataset);
        let result_mul2 = eval_mul2.evaluate(&[]);
        assert_eq!(result_mul2[0], Complex::new(-2.0, 4.0));

        // Test (amp) real
        let expr_real = aid3.real();
        let model_real = manager.model(&expr_real);
        let eval_real = model_real.load(&dataset);
        let result_real = eval_real.evaluate(&[]);
        assert_eq!(result_real[0], Complex::new(3.0, 0.0));

        // Test (expr) real
        let expr_mul2_real = expr_mul2.real();
        let model_mul2_real = manager.model(&expr_mul2_real);
        let eval_mul2_real = model_mul2_real.load(&dataset);
        let result_mul2_real = eval_mul2_real.evaluate(&[]);
        assert_eq!(result_mul2_real[0], Complex::new(-2.0, 0.0));

        // Test (expr) imag
        let expr_mul2_imag = expr_mul2.imag();
        let model_mul2_imag = manager.model(&expr_mul2_imag);
        let eval_mul2_imag = model_mul2_imag.load(&dataset);
        let result_mul2_imag = eval_mul2_imag.evaluate(&[]);
        assert_eq!(result_mul2_imag[0], Complex::new(4.0, 0.0));

        // Test (amp) imag
        let expr_imag = aid3.imag();
        let model_imag = manager.model(&expr_imag);
        let eval_imag = model_imag.load(&dataset);
        let result_imag = eval_imag.evaluate(&[]);
        assert_eq!(result_imag[0], Complex::new(4.0, 0.0));

        // Test (amp) norm_sqr
        let expr_norm = aid1.norm_sqr();
        let model_norm = manager.model(&expr_norm);
        let eval_norm = model_norm.load(&dataset);
        let result_norm = eval_norm.evaluate(&[]);
        assert_eq!(result_norm[0], Complex::new(4.0, 0.0));

        // Test (expr) norm_sqr
        let expr_mul2_norm = expr_mul2.norm_sqr();
        let model_mul2_norm = manager.model(&expr_mul2_norm);
        let eval_mul2_norm = model_mul2_norm.load(&dataset);
        let result_mul2_norm = eval_mul2_norm.evaluate(&[]);
        assert_eq!(result_mul2_norm[0], Complex::new(20.0, 0.0));
    }

    #[test]
    fn test_amplitude_activation() {
        let mut manager = Manager::default();
        let amp1 = ComplexScalar::new("const1", constant(1.0), constant(0.0));
        let amp2 = ComplexScalar::new("const2", constant(2.0), constant(0.0));

        let aid1 = manager.register(amp1).unwrap();
        let aid2 = manager.register(amp2).unwrap();

        let dataset = Arc::new(test_dataset());
        let expr = &aid1 + &aid2;
        let model = manager.model(&expr);
        let evaluator = model.load(&dataset);

        // Test initial state (all active)
        let result = evaluator.evaluate(&[]);
        assert_eq!(result[0], Complex::new(3.0, 0.0));

        // Test deactivation
        evaluator.deactivate("const1").unwrap();
        let result = evaluator.evaluate(&[]);
        assert_eq!(result[0], Complex::new(2.0, 0.0));

        // Test isolation
        evaluator.isolate("const1").unwrap();
        let result = evaluator.evaluate(&[]);
        assert_eq!(result[0], Complex::new(1.0, 0.0));

        // Test reactivation
        evaluator.activate_all();
        let result = evaluator.evaluate(&[]);
        assert_eq!(result[0], Complex::new(3.0, 0.0));
    }

    #[test]
    fn test_gradient() {
        let mut manager = Manager::default();
        let amp = Scalar::new("parametric", parameter("test_param"));

        let aid = manager.register(amp).unwrap();
        let dataset = Arc::new(test_dataset());
        let expr = aid.norm_sqr();
        let model = manager.model(&expr);
        let evaluator = model.load(&dataset);

        let params = vec![2.0];
        let gradient = evaluator.evaluate_gradient(&params);

        // For |f(x)|^2 where f(x) = x, the derivative should be 2x
        assert_relative_eq!(gradient[0][0].re, 4.0);
        assert_relative_eq!(gradient[0][0].im, 0.0);
    }

    #[test]
    fn test_parameter_registration() {
        let mut manager = Manager::default();
        let amp = Scalar::new("parametric", parameter("test_param"));

        let aid = manager.register(amp).unwrap();
        let parameters = manager.parameters();
        let model = manager.model(&aid.into());
        let model_parameters = model.parameters();
        assert_eq!(parameters.len(), 1);
        assert_eq!(parameters[0], "test_param");
        assert_eq!(model_parameters.len(), 1);
        assert_eq!(model_parameters[0], "test_param");
    }

    #[test]
    fn test_duplicate_amplitude_registration() {
        let mut manager = Manager::default();
        let amp1 = ComplexScalar::new("same_name", constant(1.0), constant(0.0));
        let amp2 = ComplexScalar::new("same_name", constant(2.0), constant(0.0));
        manager.register(amp1).unwrap();
        assert!(manager.register(amp2).is_err());
    }
}
