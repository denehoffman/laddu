use dyn_clone::DynClone;
use nalgebra::DVector;
use num::complex::Complex64;

use crate::{
    amplitude::{AmplitudeID, AmplitudeSemanticKey},
    data::{Dataset, DatasetMetadata, Event},
    expression::{Expression, ExpressionDependence},
    resources::{Cache, Parameters, Resources},
    LadduResult,
};

/// This is the only required trait for writing new amplitude-like structures for this
/// crate. Users need only implement the [`register`](Amplitude::register)
/// method to register parameters, cached values, and the amplitude itself with an input
/// [`Resources`] struct and the [`compute`](Amplitude::compute) method to actually carry
/// out the calculation. [`Amplitude`]-implementors are required to implement [`Clone`] and can
/// optionally implement a [`precompute`](Amplitude::precompute) method to calculate and
/// cache values which do not depend on free parameters.
#[typetag::serde(tag = "type")]
pub trait Amplitude: DynClone + Send + Sync {
    /// This method should be used to tell the [`Resources`] manager about all of
    /// the free parameters and cached values used by this [`Amplitude`]. It should end by
    /// returning an [`AmplitudeID`], which can be obtained from the
    /// [`Resources::register_amplitude`] method.
    ///
    /// [`register`](Amplitude::register) is invoked once when an amplitude is first converted into
    /// an [`Expression`]. Use it to allocate parameter/cache state within [`Resources`] without assuming
    /// any dataset context.
    fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID>;

    /// Optional semantic identity key for deduplicating equivalent amplitude computations.
    ///
    /// Return `Some` only when two independently constructed instances with equal keys are
    /// interchangeable after registration, binding, precomputation, value evaluation, and gradient
    /// evaluation. The key should include the concrete amplitude type and all user-facing
    /// configuration, but must ignore registration-assigned IDs like
    /// [`ParameterID`](crate::resources::ParameterID)s and cache IDs.
    fn semantic_key(&self) -> Option<AmplitudeSemanticKey> {
        None
    }

    /// Bind this [`Amplitude`] to a concrete [`Dataset`] by using the provided metadata to wire up
    /// [`Variable`](crate::variables::Variable)s or other dataset-specific state. This will
    /// be invoked when a [`Expression`] is loaded with data, after [`register`](Amplitude::register)
    /// has already succeeded. The default implementation is a no-op for amplitudes that do not
    /// depend on metadata.
    fn bind(&mut self, _metadata: &DatasetMetadata) -> LadduResult<()> {
        Ok(())
    }

    /// Optional dependence hint used by expression-IR diagnostics/planning.
    ///
    /// The default returns [`ExpressionDependence::Mixed`] for backward compatibility.
    fn dependence_hint(&self) -> ExpressionDependence {
        ExpressionDependence::Mixed
    }

    /// Optional hint that this amplitude always evaluates to a purely real complex value.
    ///
    /// This must be conservative. Returning `true` allows `expression-ir` to erase
    /// redundant `imag`, `real`, and `conj` work under the assumption that the
    /// amplitude output always has zero imaginary component.
    fn real_valued_hint(&self) -> bool {
        false
    }

    /// This method can be used to do some critical calculations ahead of time and
    /// store them in a [`Cache`]. These values can only depend on event data,
    /// not on any free parameters in the fit. This method is opt-in since it is
    /// not required to make a functioning [`Amplitude`].
    #[allow(unused_variables)]
    fn precompute(&self, event: &Event<'_>, cache: &mut Cache) {}

    /// Evaluate [`Amplitude::precompute`] over columnar event views in a [`Dataset`].
    #[cfg(feature = "rayon")]
    fn precompute_all(&self, dataset: &Dataset, resources: &mut Resources) {
        use rayon::prelude::*;

        resources
            .caches
            .par_iter_mut()
            .enumerate()
            .for_each(|(event_index, cache)| {
                let event = dataset.event_view(event_index);
                self.precompute(&event, cache);
            });
    }

    /// Evaluate [`Amplitude::precompute`] over columnar event views in a [`Dataset`].
    #[cfg(not(feature = "rayon"))]
    fn precompute_all(&self, dataset: &Dataset, resources: &mut Resources) {
        dataset.for_each_event_local(|event_index, event| {
            let cache = &mut resources.caches[event_index];
            self.precompute(&event, cache);
        });
    }

    /// This method constitutes the main machinery of an [`Amplitude`], returning the actual
    /// calculated value for a particular set of [`Parameters`] and event [`Cache`].
    fn compute(&self, parameters: &Parameters, cache: &Cache) -> Complex64;

    /// This method yields the gradient of a particular [`Amplitude`] at a point specified
    /// by a set of [`Parameters`]. See those structs, as well as
    /// [`Cache`], for documentation on their available methods. For the most part,
    /// [`Parameters`] and the [`Cache`] are key-value storage accessed by
    /// [`ParameterID`](crate::resources::ParameterID)s and
    /// several different types of cache
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
        cache: &Cache,
        gradient: &mut DVector<Complex64>,
    ) {
        self.central_difference_with_indices(
            &Vec::from_iter(0..parameters.len()),
            parameters,
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
        cache: &Cache,
        gradient: &mut DVector<Complex64>,
    ) {
        let x = parameters.values().to_owned();
        let h: DVector<f64> = x
            .iter()
            .map(|&xi| f64::cbrt(f64::EPSILON) * (xi.abs() + 1.0))
            .collect::<Vec<_>>()
            .into();
        for i in indices {
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();
            x_plus[*i] += h[*i];
            x_minus[*i] -= h[*i];
            let f_plus = self.compute(&parameters.with_values(x_plus), cache);
            let f_minus = self.compute(&parameters.with_values(x_minus), cache);
            gradient[*i] = (f_plus - f_minus) / (2.0 * h[*i]);
        }
    }

    /// Convenience helper to wrap an amplitude into an [`Expression`].
    ///
    /// This allows amplitude constructors to return `LadduResult<Expression>` without duplicating
    /// boxing/registration boilerplate.
    fn into_expression(self) -> LadduResult<Expression>
    where
        Self: Sized + 'static,
    {
        Expression::from_amplitude(Box::new(self))
    }
}

dyn_clone::clone_trait_object!(Amplitude);
