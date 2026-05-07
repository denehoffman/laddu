//! Core amplitude traits, identifiers, and expression-facing compatibility exports.

mod amplitude_trait;
mod ids;
mod semantic;
mod tags;
mod test_amplitude;

pub use amplitude_trait::Amplitude;
pub(crate) use ids::AmplitudeUseSite;
pub use ids::{central_difference, AmplitudeID, AmplitudeValues, GradientValues};
pub use semantic::{
    debug_key, display_key, f64_key, parameter_array_key, parameter_key, parameter_pair_slice_key,
    parameter_slice_key, seed_key, AmplitudeSemanticField, AmplitudeSemanticKey,
};
pub use tags::{IntoTags, Tags};
pub use test_amplitude::TestAmplitude;

pub use crate::{
    expression::{
        CompiledExpression, CompiledExpressionNode, Evaluator, Expression,
        ExpressionCompileMetrics, ExpressionDependence, ExpressionRuntimeDiagnostics,
        ExpressionSpecializationMetrics, ExpressionSpecializationOrigin,
        ExpressionSpecializationStatus, NormalizationExecutionSetsExplain,
        NormalizationPlanExplain, PrecomputedCachedIntegral, PrecomputedCachedIntegralGradientTerm,
    },
    parameters::{Parameter, ParameterMap},
};
