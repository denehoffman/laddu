//! Core amplitude traits, identifiers, and expression-facing compatibility exports.

mod semantic_key;
mod test_amplitude;

pub use semantic_key::{
    debug_key, display_key, f64_key, parameter_array_key, parameter_key, parameter_pair_slice_key,
    parameter_slice_key, seed_key,
};
pub use test_amplitude::TestAmplitude;

pub use crate::{
    expression::{
        Amplitude, AmplitudeID, AmplitudeSemanticField, AmplitudeSemanticKey, CompiledExpression,
        CompiledExpressionNode, Evaluator, Expression, ExpressionCompileMetrics,
        ExpressionDependence, ExpressionRuntimeDiagnostics, ExpressionSpecializationMetrics,
        ExpressionSpecializationOrigin, ExpressionSpecializationStatus,
        NormalizationExecutionSetsExplain, NormalizationPlanExplain, PrecomputedCachedIntegral,
        PrecomputedCachedIntegralGradientTerm,
    },
    parameters::{Parameter, ParameterMap},
};
