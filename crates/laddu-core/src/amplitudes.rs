//! Core amplitude traits, identifiers, and expression-facing compatibility exports.

mod test_amplitude;

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
