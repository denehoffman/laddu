//! Immutable scalar yield evaluation and rate-closure diagnostics.

use std::sync::Arc;

use laddu_data::data::Dataset;

use crate::{
    CrossSectionIntegrals, Ensemble, Estimate, Likelihood, LikelihoodError, LikelihoodResult,
};

const DEFAULT_ABSOLUTE_CLOSURE_TOLERANCE: f64 = 1.0e-9;
const DEFAULT_RELATIVE_CLOSURE_TOLERANCE: f64 = 1.0e-9;

/// Status of an absolute-rate closure comparison.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RateClosureStatus {
    /// Both closure comparisons are within the configured tolerance.
    Closed,
    /// At least one comparison is outside the configured tolerance.
    Failed,
    /// The likelihood term does not define an absolute fitted rate.
    NotApplicable,
}

/// Inspectable comparison of selected and fitted scalar yields.
#[derive(Clone, Debug, PartialEq)]
pub struct RateClosure {
    observed_selected: Estimate,
    accepted_fitted: Option<Estimate>,
    generated_fitted: Option<Estimate>,
    corrected_observed: Option<Estimate>,
    accepted_residual: Option<f64>,
    corrected_residual: Option<f64>,
    accepted_relative_residual: Option<f64>,
    corrected_relative_residual: Option<f64>,
    accepted_residual_draws: Vec<f64>,
    corrected_residual_draws: Vec<f64>,
    absolute_tolerance: f64,
    relative_tolerance: f64,
    status: RateClosureStatus,
    reason: Option<String>,
}

impl RateClosure {
    /// Returns the effective observed selected yield.
    pub fn selected_yield(&self) -> &Estimate {
        &self.observed_selected
    }

    /// Returns the accepted fitted yield, when available.
    pub fn accepted_fitted_yield(&self) -> Option<&Estimate> {
        self.accepted_fitted.as_ref()
    }

    /// Returns the generated fitted yield, when available.
    pub fn generated_fitted_yield(&self) -> Option<&Estimate> {
        self.generated_fitted.as_ref()
    }

    /// Returns the corrected observed yield, when available.
    pub fn corrected_observed_yield(&self) -> Option<&Estimate> {
        self.corrected_observed.as_ref()
    }

    /// Returns the central accepted-minus-selected residual.
    pub fn accepted_residual(&self) -> Option<f64> {
        self.accepted_residual
    }

    /// Returns the central absolute accepted-minus-selected residual.
    pub fn accepted_absolute_residual(&self) -> Option<f64> {
        self.accepted_residual.map(f64::abs)
    }

    /// Returns the central corrected-observed-minus-generated residual.
    pub fn corrected_residual(&self) -> Option<f64> {
        self.corrected_residual
    }

    /// Returns the central absolute corrected-observed-minus-generated residual.
    pub fn corrected_absolute_residual(&self) -> Option<f64> {
        self.corrected_residual.map(f64::abs)
    }

    /// Returns the central relative accepted-yield residual.
    pub fn accepted_relative_residual(&self) -> Option<f64> {
        self.accepted_relative_residual
    }

    /// Returns the central relative corrected-yield residual.
    pub fn corrected_relative_residual(&self) -> Option<f64> {
        self.corrected_relative_residual
    }

    /// Returns accepted-minus-selected residuals in ensemble draw order.
    pub fn accepted_residual_draws(&self) -> &[f64] {
        &self.accepted_residual_draws
    }

    /// Returns corrected-observed-minus-generated residuals in draw order.
    pub fn corrected_residual_draws(&self) -> &[f64] {
        &self.corrected_residual_draws
    }

    /// Returns the absolute tolerance used by the closure comparison.
    pub fn absolute_tolerance(&self) -> f64 {
        self.absolute_tolerance
    }

    /// Returns the relative tolerance scale used by the closure comparison.
    pub fn relative_tolerance(&self) -> f64 {
        self.relative_tolerance
    }

    /// Returns the closure status.
    pub fn status(&self) -> RateClosureStatus {
        self.status
    }

    /// Returns why closure was not applicable, if a reason was recorded.
    pub fn reason(&self) -> Option<&str> {
        self.reason.as_deref()
    }

    /// Returns whether the central values satisfy rate closure.
    pub fn is_closed(&self) -> bool {
        self.status == RateClosureStatus::Closed
    }
}

/// Immutable scalar yield evaluation context for one intensity likelihood term.
#[derive(Clone)]
pub struct Yield {
    likelihood: Arc<Likelihood>,
    term_name: String,
    observed_data: Dataset,
    accepted_mc: Dataset,
    generated_mc: Dataset,
    parameters: Vec<f64>,
    ensemble: Option<Ensemble>,
    has_absolute_rate: bool,
    scalars: YieldScalars,
}

#[derive(Clone, Debug)]
struct YieldScalars {
    selected: Estimate,
    rate: RateScalars,
    acceptance: Estimate,
    corrected: Estimate,
}

#[derive(Clone, Debug)]
enum RateScalars {
    Absolute {
        accepted: Estimate,
        generated: Estimate,
    },
    ShapeOnly,
}

impl std::fmt::Debug for Yield {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Yield")
            .field("term_name", &self.term_name)
            .field("parameters", &self.parameters)
            .field("ensemble", &self.ensemble)
            .finish_non_exhaustive()
    }
}

impl Yield {
    /// Constructs a scalar yield context with optional paired uncertainty draws.
    ///
    /// # Errors
    /// Returns an error when parameters, ensemble provenance, or integral
    /// preparation is invalid.
    pub fn with_ensemble(
        likelihood: Arc<Likelihood>,
        term_name: impl Into<String>,
        generated_mc: Dataset,
        parameters: Vec<f64>,
        ensemble: Option<Ensemble>,
    ) -> LikelihoodResult<Self> {
        likelihood.params().validate_free_values(&parameters)?;
        let term_name = term_name.into();
        let model_digest = likelihood.intensity_model_digest(&term_name)?;
        let (observed_data, accepted_mc) = likelihood.intensity_datasets(&term_name)?;
        let observed_data = observed_data.clone();
        let accepted_mc = accepted_mc.clone();
        let integrals = likelihood.cross_section_integrals(&term_name, &generated_mc)?;
        let has_absolute_rate = integrals.has_absolute_rate();
        let mut replica_integrals = Vec::new();
        if let Some(ensemble) = &ensemble {
            let names = likelihood
                .params()
                .free_params()
                .iter()
                .map(|id| likelihood.params().name(*id).map(str::to_owned))
                .collect::<Result<Vec<_>, _>>()?;
            if names != ensemble.parameter_names() {
                return Err(LikelihoodError::InvalidCrossSection(
                    "ensemble parameter names do not match the likelihood".to_owned(),
                ));
            }
            for parameters in ensemble.draws() {
                likelihood.params().validate_free_values(parameters)?;
            }
            for replica in ensemble.replicas() {
                if replica.intensity_model_digest(&term_name)? != model_digest {
                    return Err(LikelihoodError::InvalidCrossSection(
                        "ensemble replica model does not match the likelihood".to_owned(),
                    ));
                }
                let replica_names = replica
                    .params()
                    .free_params()
                    .iter()
                    .map(|id| replica.params().name(*id).map(str::to_owned))
                    .collect::<Result<Vec<_>, _>>()?;
                if replica_names != names {
                    return Err(LikelihoodError::InvalidCrossSection(
                        "ensemble replica parameter names do not match the likelihood".to_owned(),
                    ));
                }
                for parameters in ensemble.draws() {
                    replica.params().validate_free_values(parameters)?;
                }
                let replica_integral =
                    replica.cross_section_integrals(&term_name, &generated_mc)?;
                if replica_integral.has_absolute_rate() != has_absolute_rate {
                    return Err(LikelihoodError::InvalidCrossSection(
                        "ensemble replica rate semantics do not match the likelihood".to_owned(),
                    ));
                }
                replica_integrals.push(replica_integral);
            }
        }
        let selected = evaluate_estimate(
            &integrals,
            &parameters,
            ensemble.as_ref(),
            &replica_integrals,
            |integrals, _| finite("selected yield", integrals.data_weight_sum()),
        )?;
        let accepted_integral = evaluate_estimate(
            &integrals,
            &parameters,
            ensemble.as_ref(),
            &replica_integrals,
            |integrals, parameters| {
                positive_accepted(
                    integrals.accepted_integral(parameters)?,
                    integrals.accepted_mc_source().stats()?.events() > 0,
                )
            },
        )?;
        let generated_integral = evaluate_estimate(
            &integrals,
            &parameters,
            ensemble.as_ref(),
            &replica_integrals,
            |integrals, parameters| {
                positive_generated(
                    integrals.generated_integral(parameters)?,
                    integrals.generated_mc_source().stats()?.events() > 0,
                )
            },
        )?;
        let acceptance = finite_estimate(
            "fitted acceptance",
            &accepted_integral / &generated_integral,
        )?;
        let corrected = finite_estimate(
            "corrected observed yield",
            &(&selected * &generated_integral) / &accepted_integral,
        )?;
        let scalars = YieldScalars {
            selected,
            rate: if has_absolute_rate {
                RateScalars::Absolute {
                    accepted: accepted_integral,
                    generated: generated_integral,
                }
            } else {
                RateScalars::ShapeOnly
            },
            acceptance,
            corrected,
        };
        Ok(Self {
            likelihood,
            term_name,
            observed_data,
            accepted_mc,
            generated_mc,
            parameters,
            ensemble,
            has_absolute_rate,
            scalars,
        })
    }

    /// Returns the source likelihood.
    pub fn likelihood(&self) -> &Arc<Likelihood> {
        &self.likelihood
    }

    /// Returns the selected intensity-term name.
    pub fn term_name(&self) -> &str {
        &self.term_name
    }

    /// Returns the generated Monte Carlo source dataset.
    pub fn generated_mc(&self) -> &Dataset {
        &self.generated_mc
    }

    /// Returns the observed source dataset owned by the selected intensity term.
    pub fn observed_data(&self) -> &Dataset {
        &self.observed_data
    }

    /// Returns the accepted Monte Carlo source dataset owned by the selected term.
    pub fn accepted_mc(&self) -> &Dataset {
        &self.accepted_mc
    }

    /// Returns the central fitted free-parameter values.
    pub fn parameters(&self) -> &[f64] {
        &self.parameters
    }

    /// Returns the optional paired uncertainty ensemble.
    pub fn ensemble(&self) -> Option<&Ensemble> {
        self.ensemble.as_ref()
    }

    /// Returns whether the selected term determines an absolute fitted rate.
    pub fn has_absolute_rate(&self) -> bool {
        self.has_absolute_rate
    }

    /// Returns the effective observed selected yield, D.
    pub fn selected_yield(&self) -> &Estimate {
        &self.scalars.selected
    }

    /// Returns the accepted fitted yield, A.
    ///
    /// # Errors
    /// Returns an error when absolute rate is unavailable, accepted support is
    /// non-positive, or a paired draw cannot be evaluated.
    pub fn accepted_fitted_yield(&self) -> LikelihoodResult<Estimate> {
        match &self.scalars.rate {
            RateScalars::Absolute { accepted, .. } => Ok(accepted.clone()),
            RateScalars::ShapeOnly => Err(LikelihoodError::AbsoluteRateUnavailable(format!(
                "{} (accepted fitted yield)",
                self.term_name
            ))),
        }
    }

    /// Returns the generated fitted yield, G.
    ///
    /// # Errors
    /// Returns an error when absolute rate is unavailable, generated support is
    /// non-positive, or a paired draw cannot be evaluated.
    pub fn generated_fitted_yield(&self) -> LikelihoodResult<Estimate> {
        match &self.scalars.rate {
            RateScalars::Absolute { generated, .. } => Ok(generated.clone()),
            RateScalars::ShapeOnly => Err(LikelihoodError::AbsoluteRateUnavailable(format!(
                "{} (generated fitted yield)",
                self.term_name
            ))),
        }
    }

    /// Returns fitted acceptance, A/G.
    pub fn fitted_acceptance(&self) -> &Estimate {
        &self.scalars.acceptance
    }

    /// Returns the corrected observed yield, DG/A.
    pub fn corrected_observed_yield(&self) -> &Estimate {
        &self.scalars.corrected
    }

    /// Compares accepted fitted and corrected observed yields to their observed/generated counterparts.
    pub fn rate_closure(&self) -> RateClosure {
        let scalars = self.scalars.clone();
        let observed_selected = scalars.selected;
        let (accepted_fitted, generated_fitted) = match scalars.rate {
            RateScalars::ShapeOnly => {
                return RateClosure {
                    observed_selected,
                    accepted_fitted: None,
                    generated_fitted: None,
                    corrected_observed: Some(scalars.corrected),
                    accepted_residual: None,
                    corrected_residual: None,
                    accepted_relative_residual: None,
                    corrected_relative_residual: None,
                    accepted_residual_draws: Vec::new(),
                    corrected_residual_draws: Vec::new(),
                    absolute_tolerance: DEFAULT_ABSOLUTE_CLOSURE_TOLERANCE,
                    relative_tolerance: DEFAULT_RELATIVE_CLOSURE_TOLERANCE,
                    status: RateClosureStatus::NotApplicable,
                    reason: Some("likelihood term does not determine an absolute rate".to_owned()),
                };
            }
            RateScalars::Absolute {
                accepted,
                generated,
            } => (accepted, generated),
        };
        let corrected_observed = scalars.corrected;
        let accepted_residual = accepted_fitted.value() - observed_selected.value();
        let corrected_residual = corrected_observed.value() - generated_fitted.value();
        let accepted_relative_residual =
            relative_residual(accepted_residual, observed_selected.value());
        let corrected_relative_residual =
            relative_residual(corrected_residual, generated_fitted.value());
        let accepted_residual_draws = accepted_fitted
            .draws()
            .iter()
            .zip(observed_selected.draws())
            .map(|(accepted, observed)| accepted - observed)
            .collect::<Vec<_>>();
        let corrected_residual_draws = corrected_observed
            .draws()
            .iter()
            .zip(generated_fitted.draws())
            .map(|(corrected, generated)| corrected - generated)
            .collect::<Vec<_>>();
        let status = if within_tolerance(accepted_residual, observed_selected.value())
            && within_tolerance(corrected_residual, generated_fitted.value())
        {
            RateClosureStatus::Closed
        } else {
            RateClosureStatus::Failed
        };
        let reason = (status == RateClosureStatus::Failed)
            .then(|| "closure residual exceeds the configured tolerance".to_owned());
        RateClosure {
            observed_selected,
            accepted_fitted: Some(accepted_fitted),
            generated_fitted: Some(generated_fitted),
            corrected_observed: Some(corrected_observed),
            accepted_residual: Some(accepted_residual),
            corrected_residual: Some(corrected_residual),
            accepted_relative_residual,
            corrected_relative_residual,
            accepted_residual_draws,
            corrected_residual_draws,
            absolute_tolerance: DEFAULT_ABSOLUTE_CLOSURE_TOLERANCE,
            relative_tolerance: DEFAULT_RELATIVE_CLOSURE_TOLERANCE,
            status,
            reason,
        }
    }
}

fn evaluate_estimate(
    integrals: &CrossSectionIntegrals,
    parameters: &[f64],
    ensemble: Option<&Ensemble>,
    replica_integrals: &[CrossSectionIntegrals],
    function: impl Fn(&CrossSectionIntegrals, &[f64]) -> LikelihoodResult<f64>,
) -> LikelihoodResult<Estimate> {
    let central = function(integrals, parameters)?;
    let (draws, source_id) = match ensemble {
        Some(ensemble) => {
            let draws = ensemble
                .draws()
                .iter()
                .enumerate()
                .map(|(index, parameters)| {
                    let integrals = replica_integrals.get(index).unwrap_or(integrals);
                    function(integrals, parameters)
                })
                .collect::<LikelihoodResult<Vec<_>>>()?;
            (draws, Some(ensemble.source_id()))
        }
        None => (Vec::new(), None),
    };
    Estimate::with_source_id(central, draws, source_id)
}

fn finite(quantity: &'static str, value: f64) -> LikelihoodResult<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(LikelihoodError::NonFiniteYield { quantity, value })
    }
}

fn finite_estimate(quantity: &'static str, estimate: Estimate) -> LikelihoodResult<Estimate> {
    if let Some(value) = std::iter::once(estimate.value())
        .chain(estimate.draws().iter().copied())
        .find(|value| !value.is_finite())
    {
        Err(LikelihoodError::NonFiniteYield { quantity, value })
    } else {
        Ok(estimate)
    }
}

fn positive_accepted(value: f64, has_support: bool) -> LikelihoodResult<f64> {
    finite("accepted fitted yield", value)?;
    if value > 0.0 {
        Ok(value)
    } else if !has_support {
        Err(LikelihoodError::MissingAcceptedSupport)
    } else if value == 0.0 {
        Err(LikelihoodError::ZeroAcceptedIntegral)
    } else {
        Err(LikelihoodError::NonPositiveAcceptedIntegral(value))
    }
}

fn positive_generated(value: f64, has_support: bool) -> LikelihoodResult<f64> {
    finite("generated fitted yield", value)?;
    if value > 0.0 {
        Ok(value)
    } else if !has_support {
        Err(LikelihoodError::MissingGeneratedSupport)
    } else if value == 0.0 {
        Err(LikelihoodError::ZeroGeneratedIntegral)
    } else {
        Err(LikelihoodError::NonPositiveGeneratedIntegral(value))
    }
}

fn within_tolerance(residual: f64, reference: f64) -> bool {
    residual.abs()
        <= DEFAULT_ABSOLUTE_CLOSURE_TOLERANCE + DEFAULT_RELATIVE_CLOSURE_TOLERANCE * reference.abs()
}

fn relative_residual(residual: f64, reference: f64) -> Option<f64> {
    (reference != 0.0).then(|| residual.abs() / reference.abs())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use approx::assert_relative_eq;
    use laddu_compile::CompiledModel;
    use laddu_data::{
        data::{EventBatch, OwnedEvent},
        schema::Schema,
    };
    use laddu_expr::{event_scalar, parameter};
    #[cfg(feature = "jit")]
    use laddu_runtime::{CpuOptions, Device, ExecutionOptions, JitPolicy, Precision, ThreadPolicy};

    use super::*;
    use crate::{ExtendedNllTerm, Likelihood, NllTerm};
    use laddu_data::data::Dataset;

    fn weighted_dataset(values: &[(f64, f64)]) -> Dataset {
        let schema = Arc::new(Schema::new(std::iter::empty::<&str>(), ["x"], true).unwrap());
        let batch = EventBatch::from_events(
            schema,
            values
                .iter()
                .map(|(x, weight)| OwnedEvent::weighted(vec![], vec![*x], *weight)),
        )
        .unwrap();
        Dataset::from_batches(vec![batch]).unwrap()
    }

    #[test]
    fn scalar_yield_exposes_distinct_rate_quantities() {
        let model =
            CompiledModel::from_expr(&(event_scalar("x") * parameter!("scale", initial: 0.25)))
                .unwrap();
        let data = weighted_dataset(&[(2.0, 1.0), (3.0, 2.0)]);
        let accepted = weighted_dataset(&[(4.0, 1.0)]);
        let generated = weighted_dataset(&[(6.0, 1.0)]);
        let likelihood = Arc::new(
            Likelihood::new([ExtendedNllTerm::new("signal", &model, &data, &accepted).unwrap()])
                .unwrap(),
        );

        let yield_context =
            Yield::with_ensemble(likelihood, "signal", generated, vec![0.25], None).unwrap();

        assert_relative_eq!(yield_context.selected_yield().value(), 3.0);
        assert_relative_eq!(yield_context.accepted_fitted_yield().unwrap().value(), 1.0);
        assert_relative_eq!(yield_context.generated_fitted_yield().unwrap().value(), 1.5);
        assert_relative_eq!(yield_context.fitted_acceptance().value(), 2.0 / 3.0);
        assert_relative_eq!(yield_context.corrected_observed_yield().value(), 4.5);
        assert_eq!(
            yield_context.selected_yield(),
            yield_context.selected_yield()
        );
    }

    #[test]
    fn scalar_yield_reports_rate_closure_without_rescaling() {
        let model =
            CompiledModel::from_expr(&(event_scalar("x") * parameter!("scale", initial: 0.75)))
                .unwrap();
        let data = weighted_dataset(&[(2.0, 1.0), (3.0, 2.0)]);
        let accepted = weighted_dataset(&[(4.0, 1.0)]);
        let generated = weighted_dataset(&[(6.0, 1.0)]);
        let likelihood = Arc::new(
            Likelihood::new([ExtendedNllTerm::new("signal", &model, &data, &accepted).unwrap()])
                .unwrap(),
        );
        let yield_context =
            Yield::with_ensemble(likelihood, "signal", generated, vec![0.75], None).unwrap();

        let closure = yield_context.rate_closure();
        assert_eq!(closure.status(), RateClosureStatus::Closed);
        assert!(closure.is_closed());
        assert_relative_eq!(closure.selected_yield().value(), 3.0);
        assert_relative_eq!(closure.accepted_fitted_yield().unwrap().value(), 3.0);
        assert_relative_eq!(closure.generated_fitted_yield().unwrap().value(), 4.5);
        assert_relative_eq!(closure.corrected_observed_yield().unwrap().value(), 4.5);
        assert_relative_eq!(closure.accepted_residual().unwrap(), 0.0);
        assert_relative_eq!(closure.corrected_residual().unwrap(), 0.0);
    }

    #[test]
    fn scalar_yield_keeps_closure_failure_visible() {
        let model =
            CompiledModel::from_expr(&(event_scalar("x") * parameter!("scale", initial: 0.25)))
                .unwrap();
        let data = weighted_dataset(&[(2.0, 1.0), (3.0, 2.0)]);
        let accepted = weighted_dataset(&[(4.0, 1.0)]);
        let generated = weighted_dataset(&[(6.0, 1.0)]);
        let likelihood = Arc::new(
            Likelihood::new([ExtendedNllTerm::new("signal", &model, &data, &accepted).unwrap()])
                .unwrap(),
        );
        let closure = Yield::with_ensemble(likelihood, "signal", generated, vec![0.25], None)
            .unwrap()
            .rate_closure();

        assert_eq!(closure.status(), RateClosureStatus::Failed);
        assert!(!closure.is_closed());
        assert!(closure.reason().unwrap().contains("exceeds"));
        assert_relative_eq!(closure.accepted_residual().unwrap(), -2.0);
        assert_relative_eq!(closure.accepted_absolute_residual().unwrap(), 2.0);
        assert_relative_eq!(closure.corrected_absolute_residual().unwrap(), 3.0);
        assert_relative_eq!(closure.accepted_relative_residual().unwrap(), 2.0 / 3.0);
    }

    #[test]
    fn shape_only_yield_keeps_scale_free_quantities_and_marks_rate_unavailable() {
        let model = CompiledModel::from_expr(&event_scalar("x")).unwrap();
        let data = weighted_dataset(&[(2.0, 1.0), (3.0, 2.0)]);
        let accepted = weighted_dataset(&[(4.0, 1.0)]);
        let generated = weighted_dataset(&[(6.0, 1.0)]);
        let likelihood = Arc::new(
            Likelihood::new([NllTerm::new("shape", &model, &data, &accepted).unwrap()]).unwrap(),
        );
        let yield_context =
            Yield::with_ensemble(likelihood, "shape", generated, vec![], None).unwrap();

        assert!(matches!(
            yield_context.accepted_fitted_yield(),
            Err(LikelihoodError::AbsoluteRateUnavailable(_))
        ));
        assert!(matches!(
            yield_context.generated_fitted_yield(),
            Err(LikelihoodError::AbsoluteRateUnavailable(_))
        ));
        assert_relative_eq!(yield_context.fitted_acceptance().value(), 2.0 / 3.0);
        assert_relative_eq!(yield_context.corrected_observed_yield().value(), 4.5);
        let closure = yield_context.rate_closure();
        assert_eq!(closure.status(), RateClosureStatus::NotApplicable);
        assert!(closure.reason().unwrap().contains("absolute rate"));
    }

    #[test]
    fn scalar_yield_preserves_draw_order_and_provenance() {
        let model =
            CompiledModel::from_expr(&(event_scalar("x") * parameter!("scale", initial: 0.25)))
                .unwrap();
        let data = weighted_dataset(&[(2.0, 1.0), (3.0, 2.0)]);
        let accepted = weighted_dataset(&[(4.0, 1.0)]);
        let generated = weighted_dataset(&[(6.0, 1.0)]);
        let likelihood = Arc::new(
            Likelihood::new([ExtendedNllTerm::new("signal", &model, &data, &accepted).unwrap()])
                .unwrap(),
        );
        let ensemble =
            Ensemble::with_source_id(vec!["scale".to_owned()], vec![vec![0.5], vec![0.75]], 88)
                .unwrap();
        let yield_context =
            Yield::with_ensemble(likelihood, "signal", generated, vec![0.25], Some(ensemble))
                .unwrap();

        let accepted = yield_context.accepted_fitted_yield().unwrap();
        assert_eq!(accepted.source_id(), Some(88));
        assert_eq!(accepted.draws(), &[2.0, 3.0]);
        assert_eq!(yield_context.selected_yield().draws(), &[3.0, 3.0]);
        assert_eq!(
            yield_context.corrected_observed_yield().draws(),
            &[4.5, 4.5]
        );
    }

    #[test]
    fn paired_replica_draws_use_their_matching_observed_dataset() {
        let model =
            CompiledModel::from_expr(&(event_scalar("x") * parameter!("scale", initial: 0.5)))
                .unwrap();
        let accepted = weighted_dataset(&[(4.0, 1.0)]);
        let generated = weighted_dataset(&[(6.0, 1.0)]);
        let central = Arc::new(
            Likelihood::new([ExtendedNllTerm::new(
                "signal",
                &model,
                &weighted_dataset(&[(2.0, 1.0)]),
                &accepted,
            )
            .unwrap()])
            .unwrap(),
        );
        let replica = Arc::new(
            Likelihood::new([ExtendedNllTerm::new(
                "signal",
                &model,
                &weighted_dataset(&[(2.0, 2.0), (3.0, 3.0)]),
                &accepted,
            )
            .unwrap()])
            .unwrap(),
        );
        let ensemble =
            Ensemble::with_replicas(vec!["scale".to_owned()], vec![vec![0.5]], vec![replica])
                .unwrap();
        let yield_context =
            Yield::with_ensemble(central, "signal", generated, vec![0.5], Some(ensemble)).unwrap();

        assert_eq!(yield_context.selected_yield().draws(), &[5.0]);
        assert_eq!(
            yield_context.accepted_fitted_yield().unwrap().draws(),
            &[2.0]
        );
    }

    #[test]
    fn paired_replicas_must_preserve_absolute_rate_semantics() {
        let model =
            CompiledModel::from_expr(&(event_scalar("x") * parameter!("scale", initial: 0.5)))
                .unwrap();
        let data = weighted_dataset(&[(2.0, 1.0)]);
        let accepted = weighted_dataset(&[(4.0, 1.0)]);
        let generated = weighted_dataset(&[(6.0, 1.0)]);
        let central = Arc::new(
            Likelihood::new([ExtendedNllTerm::new("signal", &model, &data, &accepted).unwrap()])
                .unwrap(),
        );
        let replica = Arc::new(
            Likelihood::new([NllTerm::new("signal", &model, &data, &accepted).unwrap()]).unwrap(),
        );
        let ensemble =
            Ensemble::with_replicas(vec!["scale".to_owned()], vec![vec![0.5]], vec![replica])
                .unwrap();

        let error = Yield::with_ensemble(central, "signal", generated, vec![0.5], Some(ensemble))
            .unwrap_err();
        assert!(error.to_string().contains("rate semantics"));
    }

    #[test]
    fn paired_replicas_must_use_the_same_model_semantics() {
        let model =
            CompiledModel::from_expr(&(event_scalar("x") * parameter!("scale", initial: 0.5)))
                .unwrap();
        let different_model = CompiledModel::from_expr(
            &((event_scalar("x") + 1.0) * parameter!("scale", initial: 0.5)),
        )
        .unwrap();
        let data = weighted_dataset(&[(2.0, 1.0)]);
        let accepted = weighted_dataset(&[(4.0, 1.0)]);
        let generated = weighted_dataset(&[(6.0, 1.0)]);
        let central = Arc::new(
            Likelihood::new([ExtendedNllTerm::new("signal", &model, &data, &accepted).unwrap()])
                .unwrap(),
        );
        let replica = Arc::new(
            Likelihood::new([
                ExtendedNllTerm::new("signal", &different_model, &data, &accepted).unwrap(),
            ])
            .unwrap(),
        );
        let ensemble =
            Ensemble::with_replicas(vec!["scale".to_owned()], vec![vec![0.5]], vec![replica])
                .unwrap();

        let error = Yield::with_ensemble(central, "signal", generated, vec![0.5], Some(ensemble))
            .unwrap_err();
        assert!(error.to_string().contains("replica model"));
    }

    #[test]
    fn invalid_integral_support_is_reported_structurally() {
        let model =
            CompiledModel::from_expr(&(event_scalar("x") * parameter!("scale", initial: 0.5)))
                .unwrap();
        let data = weighted_dataset(&[(2.0, 1.0)]);
        let accepted_zero = weighted_dataset(&[(4.0, 0.0)]);
        let generated = weighted_dataset(&[(6.0, 1.0)]);
        let accepted_likelihood = Arc::new(
            Likelihood::new([
                ExtendedNllTerm::new("signal", &model, &data, &accepted_zero).unwrap(),
            ])
            .unwrap(),
        );
        assert!(matches!(
            Yield::with_ensemble(
                accepted_likelihood,
                "signal",
                generated.clone(),
                vec![0.5],
                None
            ),
            Err(LikelihoodError::ZeroAcceptedIntegral)
        ));

        let accepted = weighted_dataset(&[(4.0, 1.0)]);
        let generated_zero = weighted_dataset(&[(6.0, 0.0)]);
        let generated_likelihood = Arc::new(
            Likelihood::new([ExtendedNllTerm::new("signal", &model, &data, &accepted).unwrap()])
                .unwrap(),
        );
        assert!(matches!(
            Yield::with_ensemble(
                generated_likelihood,
                "signal",
                generated_zero,
                vec![0.5],
                None
            ),
            Err(LikelihoodError::ZeroGeneratedIntegral)
        ));

        let accepted_negative = weighted_dataset(&[(4.0, -1.0)]);
        let negative_likelihood = Arc::new(
            Likelihood::new([
                ExtendedNllTerm::new("signal", &model, &data, &accepted_negative).unwrap(),
            ])
            .unwrap(),
        );
        assert!(matches!(
            Yield::with_ensemble(
                negative_likelihood,
                "signal",
                generated.clone(),
                vec![0.5],
                None
            ),
            Err(LikelihoodError::NonPositiveAcceptedIntegral(value)) if value < 0.0
        ));

        let empty_accepted = weighted_dataset(&[]);
        let empty_likelihood = Arc::new(
            Likelihood::new([
                ExtendedNllTerm::new("signal", &model, &data, &empty_accepted).unwrap(),
            ])
            .unwrap(),
        );
        assert!(matches!(
            Yield::with_ensemble(empty_likelihood, "signal", generated, vec![0.5], None),
            Err(LikelihoodError::MissingAcceptedSupport)
        ));
    }

    #[test]
    fn zero_selected_yield_is_valid_when_acceptance_support_is_positive() {
        let model =
            CompiledModel::from_expr(&(event_scalar("x") * parameter!("scale", initial: 0.5)))
                .unwrap();
        let data = weighted_dataset(&[(2.0, 0.0)]);
        let accepted = weighted_dataset(&[(4.0, 1.0)]);
        let generated = weighted_dataset(&[(6.0, 1.0)]);
        let likelihood = Arc::new(
            Likelihood::new([ExtendedNllTerm::new("signal", &model, &data, &accepted).unwrap()])
                .unwrap(),
        );
        let yield_context =
            Yield::with_ensemble(likelihood, "signal", generated, vec![0.5], None).unwrap();

        assert_eq!(yield_context.selected_yield().value(), 0.0);
        assert_eq!(yield_context.corrected_observed_yield().value(), 0.0);
        assert_eq!(
            yield_context.rate_closure().accepted_relative_residual(),
            None
        );
    }

    #[test]
    fn relative_residual_uses_the_actual_reference_scale() {
        assert_relative_eq!(relative_residual(-0.2, 0.3).unwrap(), 2.0 / 3.0);
        assert_eq!(relative_residual(0.0, 0.0), None);
    }

    #[test]
    fn scalar_yields_match_across_memory_policies_and_chunking() {
        let model =
            CompiledModel::from_expr(&(event_scalar("x") * parameter!("scale", initial: 0.25)))
                .unwrap();
        let source_data = weighted_dataset(&[(2.0, 1.0), (3.0, 2.0)]);
        let source_accepted = weighted_dataset(&[(4.0, 1.0)]);
        let source_generated = weighted_dataset(&[(6.0, 1.0)]);
        let mut values = Vec::new();

        for (data, accepted, generated) in [
            (
                source_data.clone().resident(),
                source_accepted.clone().resident(),
                source_generated.clone().resident(),
            ),
            (
                source_data.clone().streaming(),
                source_accepted.clone().streaming(),
                source_generated.clone().streaming(),
            ),
            (
                source_data.chunked(1).unwrap(),
                source_accepted.chunked(1).unwrap(),
                source_generated.chunked(1).unwrap(),
            ),
        ] {
            let likelihood = Arc::new(
                Likelihood::new(
                    [ExtendedNllTerm::new("signal", &model, &data, &accepted).unwrap()],
                )
                .unwrap(),
            );
            let result =
                Yield::with_ensemble(likelihood, "signal", generated, vec![0.25], None).unwrap();
            values.push((
                result.selected_yield().value(),
                result.accepted_fitted_yield().unwrap().value(),
                result.generated_fitted_yield().unwrap().value(),
                result.fitted_acceptance().value(),
                result.corrected_observed_yield().value(),
            ));
        }

        assert!(values.windows(2).all(|pair| pair[0] == pair[1]));
    }

    #[cfg(feature = "jit")]
    #[test]
    fn scalar_yields_match_cpu_interpreter_and_jit_backends() {
        let model =
            CompiledModel::from_expr(&(event_scalar("x") * parameter!("scale", initial: 0.25)))
                .unwrap();
        let data = weighted_dataset(&[(2.0, 1.0), (3.0, 2.0)]);
        let accepted = weighted_dataset(&[(4.0, 1.0)]);
        let generated = weighted_dataset(&[(6.0, 1.0)]);
        let execution = |jit| {
            laddu_runtime::Execution::local(ExecutionOptions {
                device: Device::Cpu(CpuOptions {
                    threads: ThreadPolicy::Serial,
                    jit,
                }),
                precision: Precision::F64,
                ..ExecutionOptions::default()
            })
            .unwrap()
        };
        let mut values = Vec::new();

        for backend in [JitPolicy::Disabled, JitPolicy::Enabled] {
            let likelihood = Arc::new(
                Likelihood::with_execution(
                    [ExtendedNllTerm::new("signal", &model, &data, &accepted).unwrap()],
                    &execution(backend),
                )
                .unwrap(),
            );
            let result =
                Yield::with_ensemble(likelihood, "signal", generated.clone(), vec![0.25], None)
                    .unwrap();
            values.push((
                result.selected_yield().value(),
                result.accepted_fitted_yield().unwrap().value(),
                result.generated_fitted_yield().unwrap().value(),
                result.fitted_acceptance().value(),
                result.corrected_observed_yield().value(),
            ));
        }

        assert_eq!(values[0], values[1]);
    }
}
