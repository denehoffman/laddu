use std::{
    fmt::{Debug, Display},
    sync::Arc,
};

use auto_ops::*;
use dyn_clone::DynClone;
use num::Complex;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::{
    data::{Dataset, Event},
    resources::{Cache, Parameters, Resources},
    Float,
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
    fn register(&mut self, resources: &mut Resources) -> AmplitudeID;
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
    pub fn register(&mut self, amplitude: Box<dyn Amplitude>) -> AmplitudeID {
        let mut amp = amplitude.clone();
        let aid = amp.register(&mut self.resources);
        self.amplitudes.push(amp);
        aid
    }
    /// Create an [`Evaluator`] which can compute the result of any [`Expression`] built on
    /// registered [`Amplitude`]s over the given [`Dataset`]. This method precomputes any relevant
    /// information over the [`Event`]s in the [`Dataset`].
    pub fn load(&mut self, dataset: &Arc<Dataset>) -> Evaluator {
        let mut loaded_resources = self.resources.clone();
        loaded_resources.reserve_cache(dataset.len());
        for amplitude in &self.amplitudes {
            amplitude.precompute_all(dataset, &mut loaded_resources);
        }
        Evaluator {
            amplitudes: self.amplitudes.clone(),
            resources: loaded_resources,
            dataset: dataset.clone(),
        }
    }
}

/// A structure which can be used to evaluate any [`Expression`] built on registered
/// [`Amplitude`]s. This contains a [`Resources`] struct which already contains cached values for
/// precomputed [`Amplitude`]s and any relevant free parameters and constants.
pub struct Evaluator {
    amplitudes: Vec<Box<dyn Amplitude>>,
    resources: Resources,
    dataset: Arc<Dataset>,
}

impl Evaluator {
    /// Get the list of parameter names in the order they appear in the [`Evaluator::evaluate`]
    /// method.
    pub fn parameters(&self) -> Vec<String> {
        self.resources.parameters.iter().cloned().collect()
    }
    /// Activate an [`Amplitude`] by name.
    pub fn activate(&mut self, name: &str) {
        self.resources.activate(name);
    }
    /// Activate several [`Amplitude`]s by name.
    pub fn activate_many<T: AsRef<str>>(&mut self, names: &[T]) {
        self.resources.activate_many(names);
    }
    /// Activate all registered [`Amplitude`]s.
    pub fn activate_all(&mut self) {
        self.resources.activate_all();
    }
    /// Dectivate an [`Amplitude`] by name.
    pub fn deactivate<T: AsRef<str>>(&mut self, name: T) {
        self.resources.deactivate(name);
    }
    /// Deactivate several [`Amplitude`]s by name.
    pub fn deactivate_many<T: AsRef<str>>(&mut self, names: &[T]) {
        self.resources.deactivate_many(names);
    }
    /// Deactivate all registered [`Amplitude`]s.
    pub fn deactivate_all(&mut self) {
        self.resources.deactivate_all();
    }
    /// Isolate an [`Amplitude`] by name (deactivate the rest).
    pub fn isolate<T: AsRef<str>>(&mut self, name: T) {
        self.resources.isolate(name);
    }
    /// Isolate several [`Amplitude`]s by name (deactivate the rest).
    pub fn isolate_many<T: AsRef<str>>(&mut self, names: &[T]) {
        self.resources.isolate_many(names);
    }
    /// Evaluate the given [`Expression`] over the events in the [`Dataset`] stored by the
    /// [`Evaluator`] with the given values for free parameters.
    #[cfg(feature = "rayon")]
    pub fn evaluate(&self, parameters: &[Float], expression: &Expression) -> Vec<Complex<Float>> {
        let parameters = Parameters::new(parameters, &self.resources.constants);
        let amplitude_values_vec: Vec<AmplitudeValues> = self
            .dataset
            .events
            .par_iter()
            .zip(self.resources.caches.par_iter())
            .map(|(event, cache)| {
                AmplitudeValues(
                    self.amplitudes
                        .iter()
                        .zip(self.resources.active.iter())
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
            .map(|amplitude_values| expression.evaluate(amplitude_values))
            .collect()
    }
    /// Evaluate the given [`Expression`] over the events in the [`Dataset`] stored by the
    /// [`Evaluator`] with the given values for free parameters.
    #[cfg(not(feature = "rayon"))]
    pub fn evaluate(&self, parameters: &[Float], expression: &Expression) -> Vec<Complex<Float>> {
        let parameters = Parameters::new(parameters, &self.resources.constants);
        let amplitude_values_vec: Vec<AmplitudeValues> = self
            .dataset
            .events
            .iter()
            .zip(self.resources.caches.iter())
            .map(|(event, cache)| {
                AmplitudeValues(
                    self.amplitudes
                        .iter()
                        .zip(self.resources.active.iter())
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
            .map(|amplitude_values| expression.evaluate(amplitude_values))
            .collect()
    }
}

/// An extended, unbinned negative log-likelihood evaluator.
pub struct NLL {
    data_evaluator: Evaluator,
    mc_evaluator: Evaluator,
}

impl NLL {
    /// Construct an [`NLL`] from a [`Manager`] and two [`Dataset`]s (data and Monte Carlo). This
    /// is the equivalent of the [`Manager::load`] method, but for two [`Dataset`]s and a different
    /// method of evaluation.
    pub fn new(manager: &Manager, ds_data: &Arc<Dataset>, ds_mc: &Arc<Dataset>) -> Self {
        Self {
            data_evaluator: manager.clone().load(ds_data),
            mc_evaluator: manager.clone().load(ds_mc),
        }
    }
    /// Get the list of parameter names in the order they appear in the [`NLL::evaluate`]
    /// method.
    pub fn parameters(&self) -> Vec<String> {
        self.data_evaluator
            .resources
            .parameters
            .iter()
            .cloned()
            .collect()
    }
    /// Activate an [`Amplitude`] by name.
    pub fn activate<T: AsRef<str>>(&mut self, name: T) {
        self.data_evaluator.resources.activate(&name);
        self.mc_evaluator.resources.activate(name);
    }
    /// Activate several [`Amplitude`]s by name.
    pub fn activate_many<T: AsRef<str>>(&mut self, names: &[T]) {
        self.data_evaluator.resources.activate_many(names);
        self.mc_evaluator.resources.activate_many(names);
    }
    /// Activate all registered [`Amplitude`]s.
    pub fn activate_all(&mut self) {
        self.data_evaluator.resources.activate_all();
        self.mc_evaluator.resources.activate_all();
    }
    /// Dectivate an [`Amplitude`] by name.
    pub fn deactivate<T: AsRef<str>>(&mut self, name: T) {
        self.data_evaluator.resources.deactivate(&name);
        self.mc_evaluator.resources.deactivate(name);
    }
    /// Deactivate several [`Amplitude`]s by name.
    pub fn deactivate_many<T: AsRef<str>>(&mut self, names: &[T]) {
        self.data_evaluator.resources.deactivate_many(names);
        self.mc_evaluator.resources.deactivate_many(names);
    }
    /// Deactivate all registered [`Amplitude`]s.
    pub fn deactivate_all(&mut self) {
        self.data_evaluator.resources.deactivate_all();
        self.mc_evaluator.resources.deactivate_all();
    }
    /// Isolate an [`Amplitude`] by name (deactivate the rest).
    pub fn isolate<T: AsRef<str>>(&mut self, name: T) {
        self.data_evaluator.resources.isolate(&name);
        self.mc_evaluator.resources.isolate(name);
    }
    /// Isolate several [`Amplitude`]s by name (deactivate the rest).
    pub fn isolate_many<T: AsRef<str>>(&mut self, names: &[T]) {
        self.data_evaluator.resources.isolate_many(names);
        self.mc_evaluator.resources.isolate_many(names);
    }

    /// Evaluate the given [`Expression`] over the events in the [`Dataset`] stored by the
    /// [`Evaluator`] with the given values for free parameters. This method takes the
    /// real part of the given expression (discarding the imaginary part entirely, which
    /// does not matter if expressions are coherent sums wrapped in [`Expression::norm_sqr`]). The
    /// result is given by the following formula:
    ///
    /// ```math
    /// NLL(\vec{p}) = -2 \left(\sum_{e \in \text{Data}} \text{weight}(e) \ln(\mathcal{L}(e)) - \frac{N_{\text{Data}}}{N_{\text{MC}}} \sum_{e \in \text{MC}} \text{weight}(e) \mathcal{L}(e) \right)
    /// ```
    #[cfg(feature = "rayon")]
    pub fn evaluate(&self, parameters: &[Float], expression: &Expression) -> Float {
        let data_result = self.data_evaluator.evaluate(parameters, expression);
        let n_data = self.data_evaluator.dataset.weighted_len();
        let mc_result = self.mc_evaluator.evaluate(parameters, expression);
        let n_mc = self.mc_evaluator.dataset.weighted_len();
        let data_term: Float = data_result
            .par_iter()
            .zip(self.data_evaluator.dataset.par_iter())
            .map(|(l, e)| e.weight * Float::ln(l.re))
            .sum();
        let mc_term: Float = mc_result
            .par_iter()
            .zip(self.mc_evaluator.dataset.par_iter())
            .map(|(l, e)| e.weight * l.re)
            .sum();
        -2.0 * (data_term - (n_data / n_mc) * mc_term)
    }

    /// Evaluate the given [`Expression`] over the events in the [`Dataset`] stored by the
    /// [`Evaluator`] with the given values for free parameters. This method takes the
    /// real part of the given expression (discarding the imaginary part entirely, which
    /// does not matter if expressions are coherent sums wrapped in [`Expression::norm_sqr`]). The
    /// result is given by the following formula:
    ///
    /// ```math
    /// NLL(\vec{p}) = -2 \left(\sum_{e \in \text{Data}} \text{weight}(e) \ln(\mathcal{L}(e)) - \frac{N_{\text{Data}}}{N_{\text{MC}}} \sum_{e \in \text{MC}} \text{weight}(e) \mathcal{L}(e) \right)
    /// ```
    #[cfg(not(feature = "rayon"))]
    pub fn evaluate(&self, parameters: &[Float], expression: &Expression) -> Float {
        let data_result = self.data_evaluator.evaluate(parameters, expression);
        let n_data = self.data_evaluator.dataset.weighted_len();
        let mc_result = self.mc_evaluator.evaluate(parameters, expression);
        let n_mc = self.mc_evaluator.dataset.weighted_len();
        let data_term: Float = data_result
            .iter()
            .zip(self.data_evaluator.dataset.iter())
            .map(|(l, e)| e.weight * Float::ln(l.re))
            .sum();
        let mc_term: Float = mc_result
            .iter()
            .zip(self.mc_evaluator.dataset.iter())
            .map(|(l, e)| e.weight * l.re)
            .sum();
        -2.0 * (data_term - (n_data / n_mc) * mc_term)
    }

    /// Project the given [`Expression`] over the events in the [`Dataset`] stored by the
    /// [`Evaluator`] with the given values for free parameters to obtain weights for each
    /// Monte-Carlo event. This method takes the real part of the given expression (discarding
    /// the imaginary part entirely, which does not matter if expressions are coherent sums
    /// wrapped in [`Expression::norm_sqr`]). Event weights are determined by the following
    /// formula:
    ///
    /// ```math
    /// \text{weight}(\vec{p}; e) = \text{weight}(e) \mathcal{L}(e) \frac{N_{\text{Data}}}{N_{\text{MC}}}
    /// ```
    #[cfg(feature = "rayon")]
    pub fn project(&self, parameters: &[Float], expression: &Expression) -> Vec<Float> {
        let n_data = self.data_evaluator.dataset.weighted_len();
        let mc_result = self.mc_evaluator.evaluate(parameters, expression);
        let n_mc = self.mc_evaluator.dataset.weighted_len();
        mc_result
            .par_iter()
            .zip(self.mc_evaluator.dataset.par_iter())
            .map(|(l, e)| e.weight * l.re * (n_data / n_mc))
            .collect()
    }

    /// Project the given [`Expression`] over the events in the [`Dataset`] stored by the
    /// [`Evaluator`] with the given values for free parameters to obtain weights for each
    /// Monte-Carlo event. This method takes the real part of the given expression (discarding
    /// the imaginary part entirely, which does not matter if expressions are coherent sums
    /// wrapped in [`Expression::norm_sqr`]). Event weights are determined by the following
    /// formula:
    ///
    /// ```math
    /// \text{weight}(\vec{p}; e) = \text{weight}(e) \mathcal{L}(e) \frac{N_{\text{Data}}}{N_{\text{MC}}}
    /// ```
    #[cfg(not(feature = "rayon"))]
    pub fn project(&self, parameters: &[Float], expression: &Expression) -> Vec<Float> {
        let n_data = self.data_evaluator.dataset.weighted_len();
        let mc_result = self.mc_evaluator.evaluate(parameters, expression);
        let n_mc = self.mc_evaluator.dataset.weighted_len();
        mc_result
            .iter()
            .zip(self.mc_evaluator.dataset.iter())
            .map(|(l, e)| e.weight * l.re * (n_data / n_mc))
            .collect()
    }
}
