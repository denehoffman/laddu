//! Composable likelihood objectives, penalties, projections, and cross-section utilities.

mod error;
mod fitted_cross_section;
mod likelihood;
mod measurement;
mod yield_context;
mod yield_projection;

pub use error::{LikelihoodError, LikelihoodResult};
pub use fitted_cross_section::{CombinationMember, CrossSection, CrossSectionProjection};
pub use likelihood::*;
pub use measurement::{
    AreaUnit, Axis, BinnedEstimate, BinnedEstimateUnit, BootstrapFitError, Ensemble, ErrorBudget,
    ErrorComponent, Estimate, Luminosity, ScalarErrorView, next_uncertainty_source_id,
};
pub use yield_context::{RateClosure, RateClosureStatus, Yield};
pub use yield_projection::{
    ComponentYieldProjection, Projection, YieldBinValidity, YieldHistogramView, YieldProjection,
    YieldProjectionDiagnostics, YieldProjectionSet,
};
