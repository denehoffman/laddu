use std::collections::HashMap;

use dyn_clone::DynClone;
use laddu_core::{amplitude::ParameterMap, LadduResult};
use nalgebra::DVector;

use super::LikelihoodExpression;

/// A trait which describes a term that can be used like a likelihood (more correctly, a negative
/// log-likelihood) in a minimization.
pub trait LikelihoodTerm: DynClone + Send + Sync {
    /// Evaluate the term given some input parameters.
    fn evaluate(&self, parameters: &[f64]) -> LadduResult<f64>;
    /// Evaluate the gradient of the term given some input parameters.
    fn evaluate_gradient(&self, parameters: &[f64]) -> LadduResult<DVector<f64>>;
    /// Fix a named parameter local to this term.
    fn fix_parameter(&self, _name: &str, _value: f64) -> LadduResult<()> {
        Ok(())
    }
    /// Mark a named parameter local to this term as free.
    fn free_parameter(&self, _name: &str) -> LadduResult<()> {
        Ok(())
    }
    /// Rename a single parameter local to this term.
    fn rename_parameter(&self, _old: &str, _new: &str) -> LadduResult<()> {
        Ok(())
    }
    /// Rename multiple parameters local to this term.
    fn rename_parameters(&self, _mapping: &HashMap<String, String>) -> LadduResult<()> {
        Ok(())
    }
    /// Return the parameters owned by this term in local order.
    fn parameter_map(&self) -> ParameterMap {
        ParameterMap::default()
    }
    /// A method called every step of any minimization/MCMC algorithm.
    fn update(&self) {}

    /// Convenience helper to wrap a likelihood term into a [`LikelihoodExpression`].
    ///
    /// This allows term constructors to return expressions without exposing the manager
    /// machinery that previously performed registration.
    fn into_expression(self) -> LadduResult<LikelihoodExpression>
    where
        Self: Sized + 'static,
    {
        LikelihoodExpression::from_term(Box::new(self))
    }
}

dyn_clone::clone_trait_object!(LikelihoodTerm);
