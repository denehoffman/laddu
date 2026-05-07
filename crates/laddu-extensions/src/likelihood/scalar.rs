use std::{collections::HashMap, sync::Arc};

use laddu_core::{amplitude::ParameterMap, LadduResult, Parameter};
use nalgebra::DVector;
use parking_lot::RwLock;

use super::{LikelihoodExpression, LikelihoodTerm};

/// A [`LikelihoodTerm`] which represents a single scaling parameter.
#[derive(Clone)]
pub struct LikelihoodScalar {
    parameter_map: Arc<RwLock<ParameterMap>>,
}

impl LikelihoodScalar {
    /// Create a new [`LikelihoodScalar`] with a parameter with the given name and wrap it as a
    /// [`LikelihoodExpression`].
    #[allow(clippy::new_ret_no_self)]
    pub fn new<T: AsRef<str>>(name: T) -> LadduResult<LikelihoodExpression> {
        let mut parameter_map = ParameterMap::default();
        parameter_map.insert(Parameter::new(name.as_ref()));
        let term = Self {
            parameter_map: Arc::new(RwLock::new(parameter_map)),
        };
        term.into_expression()
    }
}

impl LikelihoodTerm for LikelihoodScalar {
    fn evaluate(&self, parameters: &[f64]) -> LadduResult<f64> {
        Ok(parameters[0])
    }

    fn evaluate_gradient(&self, _parameters: &[f64]) -> LadduResult<DVector<f64>> {
        Ok(DVector::from_vec(vec![1.0]))
    }

    fn fix_parameter(&self, name: &str, value: f64) -> LadduResult<()> {
        self.parameter_map.read().fix_parameter(name, value)
    }

    fn free_parameter(&self, name: &str) -> LadduResult<()> {
        self.parameter_map.read().free_parameter(name)
    }

    fn rename_parameter(&self, old: &str, new: &str) -> LadduResult<()> {
        self.parameter_map.write().rename_parameter(old, new)
    }

    fn rename_parameters(&self, mapping: &HashMap<String, String>) -> LadduResult<()> {
        self.parameter_map.write().rename_parameters(mapping)
    }

    fn parameter_map(&self) -> ParameterMap {
        self.parameter_map.read().clone()
    }
}
