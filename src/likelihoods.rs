use std::{
    collections::HashMap,
    convert::Infallible,
    fmt::{Debug, Display},
    sync::Arc,
};

use crate::{
    amplitudes::{AmplitudeValues, Evaluator, Expression, GradientValues, Manager},
    data::Dataset,
    resources::Parameters,
    Float,
};
use accurate::{sum::Klein, traits::*};
use auto_ops::*;
use dyn_clone::DynClone;
use ganesh::{
    algorithms::LBFGSB, observers::DebugObserver, Algorithm, Function, Minimizer, Observer, Status,
};
use nalgebra::DVector;
use num::Complex;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// A trait which describes a term that can be used like a likelihood (more correctly, a negative
/// log-likelihood) in a minimization.
pub trait LikelihoodTerm: DynClone + Send + Sync {
    /// Evaluate the term given some input parameters.
    fn evaluate(&self, parameters: &[Float]) -> Float;
    /// Evaluate the gradient of the term given some input parameters.
    fn evaluate_gradient(&self, parameters: &[Float]) -> DVector<Float>;
    /// The list of names of the input parameters for [`LikelihoodTerm::evaluate`].
    fn parameters(&self) -> Vec<String>;
}

dyn_clone::clone_trait_object!(LikelihoodTerm);

/// An extended, unbinned negative log-likelihood evaluator.
#[derive(Clone)]
pub struct NLL {
    pub(crate) data_evaluator: Evaluator,
    pub(crate) mc_evaluator: Evaluator,
}

impl NLL {
    /// Construct an [`NLL`] from a [`Manager`] and two [`Dataset`]s (data and Monte Carlo), as
    /// well as an [`Expression`]. This is the equivalent of the [`Manager::load`] method,
    /// but for two [`Dataset`]s and a different method of evaluation.
    pub fn new(
        manager: &Manager,
        ds_data: &Arc<Dataset>,
        ds_mc: &Arc<Dataset>,
        expression: &Expression,
    ) -> Box<Self> {
        Self {
            data_evaluator: manager.clone().load(ds_data, expression),
            mc_evaluator: manager.clone().load(ds_mc, expression),
        }
        .into()
    }
    /// Activate an [`Amplitude`] by name.
    pub fn activate<T: AsRef<str>>(&self, name: T) {
        self.data_evaluator.activate(&name);
        self.mc_evaluator.activate(name);
    }
    /// Activate several [`Amplitude`]s by name.
    pub fn activate_many<T: AsRef<str>>(&self, names: &[T]) {
        self.data_evaluator.activate_many(names);
        self.mc_evaluator.activate_many(names);
    }
    /// Activate all registered [`Amplitude`]s.
    pub fn activate_all(&self) {
        self.data_evaluator.activate_all();
        self.mc_evaluator.activate_all();
    }
    /// Dectivate an [`Amplitude`] by name.
    pub fn deactivate<T: AsRef<str>>(&self, name: T) {
        self.data_evaluator.deactivate(&name);
        self.mc_evaluator.deactivate(name);
    }
    /// Deactivate several [`Amplitude`]s by name.
    pub fn deactivate_many<T: AsRef<str>>(&self, names: &[T]) {
        self.data_evaluator.deactivate_many(names);
        self.mc_evaluator.deactivate_many(names);
    }
    /// Deactivate all registered [`Amplitude`]s.
    pub fn deactivate_all(&self) {
        self.data_evaluator.deactivate_all();
        self.mc_evaluator.deactivate_all();
    }
    /// Isolate an [`Amplitude`] by name (deactivate the rest).
    pub fn isolate<T: AsRef<str>>(&self, name: T) {
        self.data_evaluator.isolate(&name);
        self.mc_evaluator.isolate(name);
    }
    /// Isolate several [`Amplitude`]s by name (deactivate the rest).
    pub fn isolate_many<T: AsRef<str>>(&self, names: &[T]) {
        self.data_evaluator.isolate_many(names);
        self.mc_evaluator.isolate_many(names);
    }

    /// Project the stored [`Expression`] over the events in the [`Dataset`] stored by the
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
    pub fn project(&self, parameters: &[Float]) -> Vec<Float> {
        let n_data = self.data_evaluator.dataset.weighted_len();
        let mc_result = self.mc_evaluator.evaluate(parameters);
        let n_mc = self.mc_evaluator.dataset.weighted_len();
        mc_result
            .par_iter()
            .zip(self.mc_evaluator.dataset.par_iter())
            .map(|(l, e)| e.weight * l.re * (n_data / n_mc))
            .collect()
    }

    /// Project the stored [`Expression`] over the events in the [`Dataset`] stored by the
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
    pub fn project(&self, parameters: &[Float]) -> Vec<Float> {
        let n_data = self.data_evaluator.dataset.weighted_len();
        let mc_result = self.mc_evaluator.evaluate(parameters);
        let n_mc = self.mc_evaluator.dataset.weighted_len();
        mc_result
            .iter()
            .zip(self.mc_evaluator.dataset.iter())
            .map(|(l, e)| e.weight * l.re * (n_data / n_mc))
            .collect()
    }
}

impl LikelihoodTerm for NLL {
    /// Get the list of parameter names in the order they appear in the [`NLL::evaluate`]
    /// method.
    fn parameters(&self) -> Vec<String> {
        self.data_evaluator
            .resources
            .read()
            .parameters
            .iter()
            .cloned()
            .collect()
    }
    /// Evaluate the stored [`Expression`] over the events in the [`Dataset`] stored by the
    /// [`Evaluator`] with the given values for free parameters. This method takes the
    /// real part of the given expression (discarding the imaginary part entirely, which
    /// does not matter if expressions are coherent sums wrapped in [`Expression::norm_sqr`]). The
    /// result is given by the following formula:
    ///
    /// ```math
    /// NLL(\vec{p}) = -2 \left(\sum_{e \in \text{Data}} \text{weight}(e) \ln(\mathcal{L}(e)) - \frac{N_{\text{Data}}}{N_{\text{MC}}} \sum_{e \in \text{MC}} \text{weight}(e) \mathcal{L}(e) \right)
    /// ```
    #[cfg(feature = "rayon")]
    fn evaluate(&self, parameters: &[Float]) -> Float {
        let data_result = self.data_evaluator.evaluate(parameters);
        let n_data = self.data_evaluator.dataset.weighted_len();
        let mc_result = self.mc_evaluator.evaluate(parameters);
        let n_mc = self.mc_evaluator.dataset.weighted_len();
        let data_term: Float = data_result
            .par_iter()
            .zip(self.data_evaluator.dataset.par_iter())
            .map(|(l, e)| e.weight * Float::ln(l.re))
            .parallel_sum_with_accumulator::<Klein<Float>>();
        let mc_term: Float = mc_result
            .par_iter()
            .zip(self.mc_evaluator.dataset.par_iter())
            .map(|(l, e)| e.weight * l.re)
            .parallel_sum_with_accumulator::<Klein<Float>>();
        -2.0 * (data_term - (n_data / n_mc) * mc_term)
    }

    /// Evaluate the stored [`Expression`] over the events in the [`Dataset`] stored by the
    /// [`Evaluator`] with the given values for free parameters. This method takes the
    /// real part of the given expression (discarding the imaginary part entirely, which
    /// does not matter if expressions are coherent sums wrapped in [`Expression::norm_sqr`]). The
    /// result is given by the following formula:
    ///
    /// ```math
    /// NLL(\vec{p}) = -2 \left(\sum_{e \in \text{Data}} \text{weight}(e) \ln(\mathcal{L}(e)) - \frac{N_{\text{Data}}}{N_{\text{MC}}} \sum_{e \in \text{MC}} \text{weight}(e) \mathcal{L}(e) \right)
    /// ```
    #[cfg(not(feature = "rayon"))]
    fn evaluate(&self, parameters: &[Float]) -> Float {
        let data_result = self.data_evaluator.evaluate(parameters);
        let n_data = self.data_evaluator.dataset.weighted_len();
        let mc_result = self.mc_evaluator.evaluate(parameters);
        let n_mc = self.mc_evaluator.dataset.weighted_len();
        let data_term: Float = data_result
            .iter()
            .zip(self.data_evaluator.dataset.iter())
            .map(|(l, e)| e.weight * Float::ln(l.re))
            .sum_with_accumulator::<Klein<Float>>();
        let mc_term: Float = mc_result
            .iter()
            .zip(self.mc_evaluator.dataset.iter())
            .map(|(l, e)| e.weight * l.re)
            .sum_with_accumulator::<Klein<Float>>();
        -2.0 * (data_term - (n_data / n_mc) * mc_term)
    }

    /// Evaluate the gradient of the stored [`Expression`] over the events in the [`Dataset`]
    /// stored by the [`Evaluator`] with the given values for free parameters. This method takes the
    /// real part of the given expression (discarding the imaginary part entirely, which
    /// does not matter if expressions are coherent sums wrapped in [`Expression::norm_sqr`]).
    /// ```
    #[cfg(feature = "rayon")]
    fn evaluate_gradient(&self, parameters: &[Float]) -> DVector<Float> {
        let data_resources = self.data_evaluator.resources.read();
        let data_parameters = Parameters::new(parameters, &data_resources.constants);
        let n_data = self.data_evaluator.dataset.weighted_len();
        let mc_resources = self.mc_evaluator.resources.read();
        let mc_parameters = Parameters::new(parameters, &mc_resources.constants);
        let n_mc = self.mc_evaluator.dataset.weighted_len();
        let zero: DVector<Float> = DVector::zeros(parameters.len());
        let data_term: DVector<Float> = self
            .data_evaluator
            .dataset
            .par_iter()
            .zip(data_resources.caches.par_iter())
            .map(|(event, cache)| {
                let mut gradient_values =
                    vec![DVector::zeros(parameters.len()); self.data_evaluator.amplitudes.len()];
                self.data_evaluator
                    .amplitudes
                    .iter()
                    .zip(data_resources.active.iter())
                    .zip(gradient_values.iter_mut())
                    .for_each(|((amp, active), grad)| {
                        if *active {
                            amp.compute_gradient(&data_parameters, event, cache, grad)
                        }
                    });
                (
                    event.weight,
                    AmplitudeValues(
                        self.data_evaluator
                            .amplitudes
                            .iter()
                            .zip(data_resources.active.iter())
                            .map(|(amp, active)| {
                                if *active {
                                    amp.compute(&data_parameters, event, cache)
                                } else {
                                    Complex::new(0.0, 0.0)
                                }
                            })
                            .collect(),
                    ),
                    GradientValues(gradient_values),
                )
            })
            .map(|(weight, amp_vals, grad_vals)| {
                (
                    weight,
                    self.data_evaluator.expression.evaluate(&amp_vals),
                    self.data_evaluator
                        .expression
                        .evaluate_gradient(&amp_vals, &grad_vals),
                )
            })
            .map(|(w, l, g)| g.map(|gi| gi.re * w / l.re))
            .collect::<Vec<DVector<Float>>>()
            .iter()
            .sum(); // TODO: replace with custom implementation of accurate crate's trait

        let mc_term: DVector<Float> = self
            .mc_evaluator
            .dataset
            .par_iter()
            .zip(mc_resources.caches.par_iter())
            .map(|(event, cache)| {
                let mut gradient_values =
                    vec![DVector::zeros(parameters.len()); self.mc_evaluator.amplitudes.len()];
                self.mc_evaluator
                    .amplitudes
                    .iter()
                    .zip(mc_resources.active.iter())
                    .zip(gradient_values.iter_mut())
                    .for_each(|((amp, active), grad)| {
                        if *active {
                            amp.compute_gradient(&mc_parameters, event, cache, grad)
                        }
                    });
                (
                    event.weight,
                    AmplitudeValues(
                        self.mc_evaluator
                            .amplitudes
                            .iter()
                            .zip(mc_resources.active.iter())
                            .map(|(amp, active)| {
                                if *active {
                                    amp.compute(&mc_parameters, event, cache)
                                } else {
                                    Complex::new(0.0, 0.0)
                                }
                            })
                            .collect(),
                    ),
                    GradientValues(gradient_values),
                )
            })
            .map(|(weight, amp_vals, grad_vals)| {
                (
                    weight,
                    self.mc_evaluator
                        .expression
                        .evaluate_gradient(&amp_vals, &grad_vals),
                )
            })
            .map(|(w, g)| w * g.map(|gi| gi.re))
            .collect::<Vec<DVector<Float>>>()
            .iter()
            .sum();
    }

    /// Evaluate the gradient of the stored [`Expression`] over the events in the [`Dataset`]
    /// stored by the [`Evaluator`] with the given values for free parameters. This method takes the
    /// real part of the given expression (discarding the imaginary part entirely, which
    /// does not matter if expressions are coherent sums wrapped in [`Expression::norm_sqr`]).
    /// ```
    #[cfg(not(feature = "rayon"))]
    fn evaluate_gradient(&self, parameters: &[Float]) -> DVector<Float> {
        let data_resources = self.data_evaluator.resources.read();
        let data_parameters = Parameters::new(parameters, &data_resources.constants);
        let n_data = self.data_evaluator.dataset.weighted_len();
        let mc_resources = self.mc_evaluator.resources.read();
        let mc_parameters = Parameters::new(parameters, &mc_resources.constants);
        let n_mc = self.mc_evaluator.dataset.weighted_len();
        let data_term: DVector<Float> = self
            .data_evaluator
            .dataset
            .iter()
            .zip(data_resources.caches.iter())
            .map(|(event, cache)| {
                let mut gradient_values =
                    vec![DVector::zeros(parameters.len()); self.data_evaluator.amplitudes.len()];
                self.data_evaluator
                    .amplitudes
                    .iter()
                    .zip(data_resources.active.iter())
                    .zip(gradient_values.iter_mut())
                    .for_each(|((amp, active), grad)| {
                        if *active {
                            amp.compute_gradient(&data_parameters, event, cache, grad)
                        }
                    });
                (
                    event.weight,
                    AmplitudeValues(
                        self.data_evaluator
                            .amplitudes
                            .iter()
                            .zip(data_resources.active.iter())
                            .map(|(amp, active)| {
                                if *active {
                                    amp.compute(&data_parameters, event, cache)
                                } else {
                                    Complex::new(0.0, 0.0)
                                }
                            })
                            .collect(),
                    ),
                    GradientValues(gradient_values),
                )
            })
            .map(|(weight, amp_vals, grad_vals)| {
                (
                    weight,
                    self.data_evaluator.expression.evaluate(&amp_vals),
                    self.data_evaluator
                        .expression
                        .evaluate_gradient(&amp_vals, &grad_vals),
                )
            })
            .map(|(w, l, g)| g.map(|gi| gi.re * w / l.re))
            .sum();

        let mc_term: DVector<Float> = self
            .mc_evaluator
            .dataset
            .iter()
            .zip(mc_resources.caches.iter())
            .map(|(event, cache)| {
                let mut gradient_values =
                    vec![DVector::zeros(parameters.len()); self.mc_evaluator.amplitudes.len()];
                self.mc_evaluator
                    .amplitudes
                    .iter()
                    .zip(mc_resources.active.iter())
                    .zip(gradient_values.iter_mut())
                    .for_each(|((amp, active), grad)| {
                        if *active {
                            amp.compute_gradient(&mc_parameters, event, cache, grad)
                        }
                    });
                (
                    event.weight,
                    AmplitudeValues(
                        self.mc_evaluator
                            .amplitudes
                            .iter()
                            .zip(mc_resources.active.iter())
                            .map(|(amp, active)| {
                                if *active {
                                    amp.compute(&mc_parameters, event, cache)
                                } else {
                                    Complex::new(0.0, 0.0)
                                }
                            })
                            .collect(),
                    ),
                    GradientValues(gradient_values),
                )
            })
            .map(|(weight, amp_vals, grad_vals)| {
                (
                    weight,
                    self.mc_evaluator
                        .expression
                        .evaluate_gradient(&amp_vals, &grad_vals),
                )
            })
            .map(|(w, g)| w * g.map(|gi| gi.re))
            .sum();
        -2.0 * (data_term - (n_data / n_mc) * mc_term)
    }
}

impl Function<Float, (), Infallible> for NLL {
    fn evaluate(&self, parameters: &[Float], _user_data: &mut ()) -> Result<Float, Infallible> {
        Ok(LikelihoodTerm::evaluate(self, parameters))
    }
    fn gradient(
        &self,
        parameters: &[Float],
        _user_data: &mut (),
    ) -> Result<DVector<Float>, Infallible> {
        Ok(LikelihoodTerm::evaluate_gradient(self, parameters))
    }
}

/// A set of options that are used when minimizations are performed.
pub struct MinimizerOptions {
    algorithm: Box<dyn ganesh::Algorithm<Float, (), Infallible>>,
    observers: Vec<Box<dyn Observer<Float, ()>>>,
    max_steps: usize,
}

impl Default for MinimizerOptions {
    fn default() -> Self {
        Self {
            algorithm: Box::new(LBFGSB::default()),
            observers: Default::default(),
            max_steps: 4000,
        }
    }
}

struct VerboseObserver {
    show_step: bool,
    show_x: bool,
    show_fx: bool,
}
impl Observer<Float, ()> for VerboseObserver {
    fn callback(&mut self, step: usize, status: &mut Status<Float>, _user_data: &mut ()) -> bool {
        if self.show_step {
            println!("Step: {}", step);
        }
        if self.show_x {
            println!("Current Best Position: {}", status.x.transpose());
        }
        if self.show_fx {
            println!("Current Best Value: {}", status.fx);
        }
        true
    }
}

impl MinimizerOptions {
    /// Adds the [`DebugObserver`] to the minimization.
    pub fn debug(self) -> Self {
        let mut observers = self.observers;
        observers.push(Box::new(DebugObserver));
        Self {
            algorithm: self.algorithm,
            observers,
            max_steps: self.max_steps,
        }
    }
    /// Adds a customizable [`VerboseObserver`] to the minimization.
    pub fn verbose(self, show_step: bool, show_x: bool, show_fx: bool) -> Self {
        let mut observers = self.observers;
        observers.push(Box::new(VerboseObserver {
            show_step,
            show_x,
            show_fx,
        }));
        Self {
            algorithm: self.algorithm,
            observers,
            max_steps: self.max_steps,
        }
    }
    /// Set the [`Algorithm`] to be used in the minimization (default: [`LBFGSB`] with default
    /// settings).
    pub fn with_algorithm<A: Algorithm<Float, (), Infallible> + 'static>(
        self,
        algorithm: A,
    ) -> Self {
        Self {
            algorithm: Box::new(algorithm),
            observers: self.observers,
            max_steps: self.max_steps,
        }
    }
    /// Add an [`Observer`] to the list of [`Observer`]s used in the minimization.
    pub fn with_observer<O: Observer<Float, ()> + 'static>(self, observer: O) -> Self {
        let mut observers = self.observers;
        observers.push(Box::new(observer));
        Self {
            algorithm: self.algorithm,
            observers,
            max_steps: self.max_steps,
        }
    }

    /// Set the maximum number of [`Algorithm`] steps for the minimization (default: 4000).
    pub fn with_max_steps(self, max_steps: usize) -> Self {
        Self {
            algorithm: self.algorithm,
            observers: self.observers,
            max_steps,
        }
    }
}

impl NLL {
    /// Minimizes the negative log-likelihood using the L-BFGS-B algorithm, a limited-memory
    /// quasi-Newton minimizer which supports bounded optimization.
    pub fn minimize(
        &self,
        p0: &[Float],
        bounds: Option<Vec<(Float, Float)>>,
        options: Option<MinimizerOptions>,
    ) -> Status<Float> {
        let options = options.unwrap_or_default();
        let mut m = Minimizer::new_from_box(options.algorithm, self.parameters().len())
            .with_bounds(bounds)
            .with_observers(options.observers)
            .with_max_steps(options.max_steps);
        m.minimize(self, p0, &mut ())
            .unwrap_or_else(|never| match never {});
        m.status
    }
}

/// An identifier that can be used like an [`AmplitudeID`] to combine registered
/// [`LikelihoodTerm`]s.
#[derive(Clone, Debug)]
pub struct LikelihoodID(usize);

impl Display for LikelihoodID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A [`Manager`] but for [`LikelihoodTerm`]s.
#[derive(Default, Clone)]
pub struct LikelihoodManager {
    terms: Vec<Box<dyn LikelihoodTerm>>,
    param_name_to_index: HashMap<String, usize>,
    param_names: Vec<String>,
    param_layouts: Vec<Vec<usize>>,
    param_counts: Vec<usize>,
}

impl LikelihoodManager {
    /// Register a [`LikelihoodTerm`] to get a [`LikelihoodID`] which can be combined with others
    /// to form [`LikelihoodExpression`]s which can be minimized.
    pub fn register(&mut self, term: Box<dyn LikelihoodTerm>) -> LikelihoodID {
        let term_idx = self.terms.len();
        for param_name in term.parameters() {
            if !self.param_name_to_index.contains_key(&param_name) {
                self.param_name_to_index
                    .insert(param_name.clone(), self.param_name_to_index.len());
                self.param_names.push(param_name.clone());
            }
        }
        let param_layout: Vec<usize> = term
            .parameters()
            .iter()
            .map(|name| self.param_name_to_index[name])
            .collect();
        let param_count = term.parameters().len();
        self.param_layouts.push(param_layout);
        self.param_counts.push(param_count);
        self.terms.push(term.clone());

        LikelihoodID(term_idx)
    }

    /// Return all of the parameter names of registered [`LikelihoodTerm`]s in order. This only
    /// returns the unique names in the order they should be input when evaluated with a
    /// [`LikelihoodEvaluator`].
    pub fn parameters(&self) -> Vec<String> {
        self.param_names.clone()
    }

    /// Load a [`LikelihoodExpression`] to generate a [`LikelihoodEvaluator`] that can be
    /// minimized.
    pub fn load(&self, likelihood_expression: &LikelihoodExpression) -> LikelihoodEvaluator {
        LikelihoodEvaluator {
            likelihood_manager: self.clone(),
            likelihood_expression: likelihood_expression.clone(),
        }
    }
}

#[derive(Debug)]
struct LikelihoodValues(Vec<Float>);

#[derive(Debug)]
struct LikelihoodGradients(Vec<DVector<Float>>);

/// A combination of [`LikelihoodTerm`]s as well as sums and products of them.
#[derive(Clone)]
pub enum LikelihoodExpression {
    /// A registered [`LikelihoodTerm`] referenced by an [`LikelihoodID`].
    Term(LikelihoodID),
    /// The sum of two [`LikelihoodExpression`]s.
    Add(Box<LikelihoodExpression>, Box<LikelihoodExpression>),
    /// The product of two [`LikelihoodExpression`]s.
    Mul(Box<LikelihoodExpression>, Box<LikelihoodExpression>),
}

impl Debug for LikelihoodExpression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.write_tree(f, "", "", "")
    }
}

impl Display for LikelihoodExpression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl LikelihoodExpression {
    fn evaluate(&self, likelihood_values: &LikelihoodValues) -> Float {
        match self {
            LikelihoodExpression::Term(lid) => likelihood_values.0[lid.0],
            LikelihoodExpression::Add(a, b) => {
                a.evaluate(likelihood_values) + b.evaluate(likelihood_values)
            }
            LikelihoodExpression::Mul(a, b) => {
                a.evaluate(likelihood_values) * b.evaluate(likelihood_values)
            }
        }
    }
    fn evaluate_gradient(
        &self,
        likelihood_values: &LikelihoodValues,
        likelihood_gradients: &LikelihoodGradients,
    ) -> DVector<Float> {
        match self {
            LikelihoodExpression::Term(lid) => likelihood_gradients.0[lid.0].clone(),
            LikelihoodExpression::Add(a, b) => {
                a.evaluate_gradient(likelihood_values, likelihood_gradients)
                    + b.evaluate_gradient(likelihood_values, likelihood_gradients)
            }
            LikelihoodExpression::Mul(a, b) => {
                a.evaluate_gradient(likelihood_values, likelihood_gradients)
                    * b.evaluate(likelihood_values)
                    + b.evaluate_gradient(likelihood_values, likelihood_gradients)
                        * a.evaluate(likelihood_values)
            }
        }
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
            // TODO: maybe come up with a way to name likelihood terms?
            Self::Term(lid) => format!("{}", lid.0),
            Self::Add(_, _) => "+".to_string(),
            Self::Mul(_, _) => "*".to_string(),
        };
        writeln!(f, "{}{}{}", parent_prefix, immediate_prefix, display_string)?;
        match self {
            Self::Term(_) => {}
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
        }
        Ok(())
    }
}

impl_op_ex!(+ |a: &LikelihoodExpression, b: &LikelihoodExpression| -> LikelihoodExpression { LikelihoodExpression::Add(Box::new(a.clone()), Box::new(b.clone()))});
impl_op_ex!(
    *|a: &LikelihoodExpression, b: &LikelihoodExpression| -> LikelihoodExpression {
        LikelihoodExpression::Mul(Box::new(a.clone()), Box::new(b.clone()))
    }
);
impl_op_ex_commutative!(+ |a: &LikelihoodID, b: &LikelihoodExpression| -> LikelihoodExpression { LikelihoodExpression::Add(Box::new(LikelihoodExpression::Term(a.clone())), Box::new(b.clone()))});
impl_op_ex_commutative!(
    *|a: &LikelihoodID, b: &LikelihoodExpression| -> LikelihoodExpression {
        LikelihoodExpression::Mul(
            Box::new(LikelihoodExpression::Term(a.clone())),
            Box::new(b.clone()),
        )
    }
);
impl_op_ex!(+ |a: &LikelihoodID, b: &LikelihoodID| -> LikelihoodExpression { LikelihoodExpression::Add(Box::new(LikelihoodExpression::Term(a.clone())), Box::new(LikelihoodExpression::Term(b.clone())))});
impl_op_ex!(
    *|a: &LikelihoodID, b: &LikelihoodID| -> LikelihoodExpression {
        LikelihoodExpression::Mul(
            Box::new(LikelihoodExpression::Term(a.clone())),
            Box::new(LikelihoodExpression::Term(b.clone())),
        )
    }
);

/// A structure to evaluate and minimize combinations of [`LikelihoodTerm`]s.
pub struct LikelihoodEvaluator {
    likelihood_manager: LikelihoodManager,
    likelihood_expression: LikelihoodExpression,
}

impl Function<Float, (), Infallible> for LikelihoodEvaluator {
    fn evaluate(&self, parameters: &[Float], _user_data: &mut ()) -> Result<Float, Infallible> {
        let mut param_buffers: Vec<Vec<Float>> = self
            .likelihood_manager
            .param_counts
            .iter()
            .map(|&count| vec![0.0; count])
            .collect();
        for (layout, buffer) in self
            .likelihood_manager
            .param_layouts
            .iter()
            .zip(param_buffers.iter_mut())
        {
            for (buffer_idx, &param_idx) in layout.iter().enumerate() {
                buffer[buffer_idx] = parameters[param_idx];
            }
        }
        let likelihood_values = LikelihoodValues(
            self.likelihood_manager
                .terms
                .iter()
                .zip(param_buffers.iter())
                .map(|(term, buffer)| term.evaluate(buffer))
                .collect(),
        );
        Ok(self.likelihood_expression.evaluate(&likelihood_values))
    }
    fn gradient(
        &self,
        parameters: &[Float],
        _user_data: &mut (),
    ) -> Result<DVector<Float>, Infallible> {
        let mut param_buffers: Vec<Vec<Float>> = self
            .likelihood_manager
            .param_counts
            .iter()
            .map(|&count| vec![0.0; count])
            .collect();
        for (layout, buffer) in self
            .likelihood_manager
            .param_layouts
            .iter()
            .zip(param_buffers.iter_mut())
        {
            for (buffer_idx, &param_idx) in layout.iter().enumerate() {
                buffer[buffer_idx] = parameters[param_idx];
            }
        }
        let likelihood_values = LikelihoodValues(
            self.likelihood_manager
                .terms
                .iter()
                .zip(param_buffers.iter())
                .map(|(term, buffer)| term.evaluate(buffer))
                .collect(),
        );
        let mut gradient_buffers: Vec<DVector<Float>> = (0..self.likelihood_manager.terms.len())
            .map(|_| DVector::zeros(self.likelihood_manager.param_names.len()))
            .collect();
        for (((term, param_buffer), gradient_buffer), layout) in self
            .likelihood_manager
            .terms
            .iter()
            .zip(param_buffers.iter())
            .zip(gradient_buffers.iter_mut())
            .zip(self.likelihood_manager.param_layouts.iter())
        {
            let term_gradient = term.evaluate_gradient(param_buffer); // This has a local layout
            for (term_idx, &buffer_idx) in layout.iter().enumerate() {
                gradient_buffer[buffer_idx] = term_gradient[term_idx] // This has a global layout
            }
        }
        let likelihood_gradients = LikelihoodGradients(gradient_buffers);
        Ok(self
            .likelihood_expression
            .evaluate_gradient(&likelihood_values, &likelihood_gradients))
    }
}

impl LikelihoodEvaluator {
    /// The parameter names used in [`LikelihoodEvaluator::evaluate`]'s input in order.
    pub fn parameters(&self) -> Vec<String> {
        self.likelihood_manager.parameters()
    }
    /// A function that can be called to evaluate the sum/product of the [`LikelihoodTerm`]s
    /// contained by this [`LikelihoodEvaluator`].
    pub fn evaluate(&self, parameters: &[Float]) -> Float {
        Function::evaluate(self, parameters, &mut ()).unwrap_or_else(|never| match never {})
    }

    /// A function that can be called to minimize the sum/product of the [`LikelihoodTerm`]s
    /// contained by this [`LikelihoodEvaluator`].
    ///
    /// See [`NLL::minimize`] for more details.
    pub fn minimize(
        &self,
        p0: &[Float],
        bounds: Option<Vec<(Float, Float)>>,
        options: Option<MinimizerOptions>,
    ) -> Status<Float> {
        let options = options.unwrap_or_default();
        let mut m = Minimizer::new_from_box(options.algorithm, self.parameters().len())
            .with_bounds(bounds)
            .with_observers(options.observers)
            .with_max_steps(options.max_steps);
        m.minimize(self, p0, &mut ())
            .unwrap_or_else(|never| match never {});
        m.status
    }
}

/// A [`LikelihoodTerm`] which represents a single scaling parameter.
#[derive(Clone)]
pub struct LikelihoodScalar(String);

impl LikelihoodScalar {
    /// Create a new [`LikelihoodScalar`] with a parameter with the given name.
    pub fn new<T: AsRef<str>>(name: T) -> Box<Self> {
        Self(name.as_ref().into()).into()
    }
}

impl LikelihoodTerm for LikelihoodScalar {
    fn evaluate(&self, parameters: &[Float]) -> Float {
        parameters[0]
    }

    fn evaluate_gradient(&self, _parameters: &[Float]) -> DVector<Float> {
        DVector::from_vec(vec![1.0])
    }

    fn parameters(&self) -> Vec<String> {
        vec![self.0.clone()]
    }
}
