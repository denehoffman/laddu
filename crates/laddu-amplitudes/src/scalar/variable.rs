use laddu_core::{
    amplitudes::{
        debug_key, display_key, Amplitude, AmplitudeID, AmplitudeSemanticKey, Expression,
        ExpressionDependence,
    },
    data::{DatasetMetadata, Event},
    resources::{Cache, Parameters, Resources},
    traits::Variable,
    LadduResult, ScalarID,
};
use nalgebra::DVector;
use num::complex::Complex64;
use serde::{Deserialize, Serialize};

/// A real-valued [`Amplitude`] which evaluates an event [`Variable`].
#[derive(Clone, Serialize, Deserialize)]
pub struct VariableScalar {
    name: String,
    variable: Box<dyn Variable>,
    value_id: ScalarID,
}

impl VariableScalar {
    /// Create a new [`VariableScalar`] that evaluates `variable` on each event.
    pub fn new<V: Variable + 'static>(name: &str, variable: &V) -> LadduResult<Expression> {
        Self {
            name: name.to_string(),
            variable: dyn_clone::clone_box(variable),
            value_id: ScalarID::default(),
        }
        .into_expression()
    }
}

/// Extension methods for building expressions from event [`Variable`]s.
pub trait VariableExpressionExt: Variable + 'static {
    /// Convert this variable into a real-valued [`Expression`].
    fn as_expression(&self, name: &str) -> LadduResult<Expression>
    where
        Self: Sized,
    {
        VariableScalar::new(name, self)
    }
}

impl<T: Variable + 'static> VariableExpressionExt for T {}

#[typetag::serde]
impl Amplitude for VariableScalar {
    fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID> {
        self.value_id = resources.register_scalar(Some(&format!("{}.value", self.name)));
        resources.register_amplitude(&self.name)
    }

    fn semantic_key(&self) -> Option<AmplitudeSemanticKey> {
        Some(
            AmplitudeSemanticKey::new("VariableScalar")
                .with_field("name", debug_key(&self.name))
                .with_field("variable", display_key(&self.variable)),
        )
    }

    fn dependence_hint(&self) -> ExpressionDependence {
        ExpressionDependence::CacheOnly
    }

    fn real_valued_hint(&self) -> bool {
        true
    }

    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        self.variable.bind(metadata)
    }

    fn precompute(&self, event: &Event<'_>, cache: &mut Cache) {
        cache.store_scalar(self.value_id, self.variable.value(event));
    }

    fn compute(&self, _parameters: &Parameters, cache: &Cache) -> Complex64 {
        cache.get_scalar(self.value_id).into()
    }

    fn compute_gradient(
        &self,
        _parameters: &Parameters,
        _cache: &Cache,
        _gradient: &mut DVector<Complex64>,
    ) {
    }
}
