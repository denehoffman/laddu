//! End-to-end statistical closure test for generation and fitting.

#[allow(dead_code)]
#[path = "../examples/common/mod.rs"]
mod common;

use std::f64::consts::PI;

use common::{
    closure::{ClosureConfig, run_closure, wrapped_phase_residual},
    ksks::{F2_MAGNITUDE_TRUTH, F2_PHASE_TRUTH},
};
use laddu::prelude::Execution;

#[test]
fn generated_two_wave_sample_recovers_injected_coupling() {
    let result = run_closure(
        ClosureConfig {
            data_events: 4_000,
            normalization_events: 40_000,
            pilot_proposals: 20_000,
            ..ClosureConfig::default()
        },
        Execution::default(),
    )
    .expect("generation-to-fit closure should succeed");

    assert_eq!(result.data_report.produced, 4_000);
    assert_eq!(result.normalization_report.produced, 40_000);
    assert!(result.initial_nll.is_finite());
    assert!(
        result
            .initial_gradient
            .iter()
            .all(|value| value.is_finite())
    );
    assert!(result.fit.fx.is_finite());
    assert!(result.fit.fx < result.initial_nll);

    let magnitude = result
        .fitted("f2_magnitude")
        .expect("fit should return f2_magnitude");
    let phase = result
        .fitted("f2_phase")
        .expect("fit should return f2_phase");
    assert!((0.0..=2.0).contains(&magnitude));
    assert!((-PI..PI).contains(&phase));
    assert!((magnitude - F2_MAGNITUDE_TRUTH).abs() < 0.15);
    assert!(wrapped_phase_residual(phase, F2_PHASE_TRUTH).abs() < 0.35);

    assert_eq!(result.projection.data.histogram.bin_edges().len(), 51);
    assert_eq!(result.projection.projections.len(), 3);
    let data_yield = result
        .projection
        .data
        .histogram
        .counts()
        .iter()
        .sum::<f64>();
    let fit_yield = result.projection.projections[0]
        .histogram
        .counts()
        .iter()
        .sum::<f64>();
    assert!((fit_yield - data_yield).abs() < 1e-9 * data_yield);
    assert!(
        result.projection.projections[1]
            .histogram
            .counts()
            .iter()
            .sum::<f64>()
            > 0.0
    );
    assert!(
        result.projection.projections[2]
            .histogram
            .counts()
            .iter()
            .sum::<f64>()
            > 0.0
    );
}
