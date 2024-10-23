use std::{
    fmt::{Debug, Display},
    sync::Arc,
};

use auto_ops::*;
use dyn_clone::DynClone;
use num::Complex;

use parking_lot::RwLock;
#[cfg(feature = "rayon")]
use rayon::prelude::*;

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
#[derive(Clone, Default)]
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
}

dyn_clone::clone_trait_object!(Amplitude);

#[derive(Debug)]
struct AmplitudeValues(Vec<Complex<Float>>);

/// A tag which refers to a registered [`Amplitude`]. This is the base object which can be used to
/// build [`Expression`]s and should be obtained from the [`Manager::register`] method.
#[derive(Clone, Default, Debug)]
pub struct AmplitudeID(pub(crate) String, pub(crate) usize);

impl Display for AmplitudeID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// An expression tree which contains [`AmplitudeID`]s and operators over them.
#[derive(Clone)]
pub enum Expression {
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
    fn evaluate(&self, amplitude_values: &AmplitudeValues) -> Complex<Float> {
        match self {
            Expression::Amp(aid) => amplitude_values.0[aid.1],
            Expression::Add(a, b) => a.evaluate(amplitude_values) + b.evaluate(amplitude_values),
            Expression::Mul(a, b) => a.evaluate(amplitude_values) * b.evaluate(amplitude_values),
            Expression::Real(a) => Complex::new(a.evaluate(amplitude_values).re, 0.0),
            Expression::Imag(a) => Complex::new(a.evaluate(amplitude_values).im, 0.0),
            Expression::NormSqr(a) => Complex::new(a.evaluate(amplitude_values).norm_sqr(), 0.0),
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
        };
        writeln!(f, "{}{}{}", parent_prefix, immediate_prefix, display_string)?;
        match self {
            Self::Amp(_) => {}
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
#[derive(Default, Clone)]
pub struct Manager {
    amplitudes: Vec<Box<dyn Amplitude>>,
    resources: Resources,
}

impl Manager {
    /// Register the given [`Amplitude`] and return an [`AmplitudeID`] that can be used to build
    /// [`Expression`]s.
    ///
    /// # Errors
    ///
    /// The [`Amplitude`](crate::amplitudes::Amplitude)'s name must be unique and not already
    /// registered, else this will return a [`RegistrationError`][LadduError::RegistrationError].
    pub fn register(&mut self, amplitude: Box<dyn Amplitude>) -> Result<AmplitudeID, LadduError> {
        let mut amp = amplitude.clone();
        let aid = amp.register(&mut self.resources)?;
        self.amplitudes.push(amp);
        Ok(aid)
    }
    /// Create an [`Evaluator`] which can compute the result of the given [`Expression`] built on
    /// registered [`Amplitude`]s over the given [`Dataset`]. This method precomputes any relevant
    /// information over the [`Event`]s in the [`Dataset`].
    pub fn load(&self, dataset: &Arc<Dataset>, expression: &Expression) -> Evaluator {
        let loaded_resources = Arc::new(RwLock::new(self.resources.clone()));
        loaded_resources.write().reserve_cache(dataset.len());
        for amplitude in &self.amplitudes {
            amplitude.precompute_all(dataset, &mut loaded_resources.write());
        }
        Evaluator {
            amplitudes: self.amplitudes.clone(),
            resources: loaded_resources.clone(),
            dataset: dataset.clone(),
            expression: expression.clone(),
        }
    }
}

/// A structure which can be used to evaluate the stored [`Expression`] built on registered
/// [`Amplitude`]s. This contains a [`Resources`] struct which already contains cached values for
/// precomputed [`Amplitude`]s and any relevant free parameters and constants.
#[derive(Clone)]
pub struct Evaluator {
    amplitudes: Vec<Box<dyn Amplitude>>,
    pub(crate) resources: Arc<RwLock<Resources>>,
    pub(crate) dataset: Arc<Dataset>,
    expression: Expression,
}

impl Evaluator {
    /// Get the list of parameter names in the order they appear in the [`Evaluator::evaluate`]
    /// method.
    pub fn parameters(&self) -> Vec<String> {
        self.resources.read().parameters.iter().cloned().collect()
    }
    /// Activate an [`Amplitude`] by name.
    pub fn activate<T: AsRef<str>>(&self, name: T) {
        self.resources.write().activate(name);
    }
    /// Activate several [`Amplitude`]s by name.
    pub fn activate_many<T: AsRef<str>>(&self, names: &[T]) {
        self.resources.write().activate_many(names);
    }
    /// Activate all registered [`Amplitude`]s.
    pub fn activate_all(&self) {
        self.resources.write().activate_all();
    }
    /// Dectivate an [`Amplitude`] by name.
    pub fn deactivate<T: AsRef<str>>(&self, name: T) {
        self.resources.write().deactivate(name);
    }
    /// Deactivate several [`Amplitude`]s by name.
    pub fn deactivate_many<T: AsRef<str>>(&self, names: &[T]) {
        self.resources.write().deactivate_many(names);
    }
    /// Deactivate all registered [`Amplitude`]s.
    pub fn deactivate_all(&self) {
        self.resources.write().deactivate_all();
    }
    /// Isolate an [`Amplitude`] by name (deactivate the rest).
    pub fn isolate<T: AsRef<str>>(&self, name: T) {
        self.resources.write().isolate(name);
    }
    /// Isolate several [`Amplitude`]s by name (deactivate the rest).
    pub fn isolate_many<T: AsRef<str>>(&self, names: &[T]) {
        self.resources.write().isolate_many(names);
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
}
