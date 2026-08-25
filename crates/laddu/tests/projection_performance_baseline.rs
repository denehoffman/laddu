//! Public differential-cross-section reference behavior for projection benchmarks.

#[path = "../benches/support/projection.rs"]
mod projection;

use laddu::prelude::ThreadPolicy;
use laddu::prelude::{BinnedEstimate, DifferentialCrossSection};
use projection::{ProjectionFixture, Storage};

#[test]
fn representative_projection_fixture_exercises_the_public_differential_contract() {
    let fixture = ProjectionFixture::new(128, 3, Storage::Resident, ThreadPolicy::Serial)
        .expect("representative fixture should build");

    assert_eq!(fixture.member_count(), 4);
    assert_eq!(fixture.projection_count(), 4);
    assert_eq!(fixture.selection_count(), 3);
    assert_eq!(fixture.unique_selection_count(), 2);

    let projections = fixture
        .evaluate_combined(4)
        .expect("public differential calls should succeed");
    assert_eq!(projections.len(), 4);
    for projection in &projections {
        assert_eq!(projection.shape(), &[40]);
        assert_eq!(projection.data().draws().len(), 3);
        assert_eq!(projection.model().draws().len(), 3);
        assert_eq!(projection.components().len(), 3);
        assert_eq!(
            projection.components()["signal"].values(),
            projection.components()["signal_alias"].values()
        );
        assert_eq!(
            projection.components()["signal"].draws(),
            projection.components()["signal_alias"].draws()
        );
    }
}

#[test]
fn resident_and_streaming_fixtures_preserve_projection_results() {
    let resident = ProjectionFixture::new(96, 2, Storage::Resident, ThreadPolicy::Serial)
        .expect("resident fixture should build")
        .evaluate_combined(4)
        .expect("resident projections should evaluate");
    let streaming = ProjectionFixture::new(96, 2, Storage::Streaming, ThreadPolicy::Serial)
        .expect("streaming fixture should build")
        .evaluate_combined(4)
        .expect("streaming projections should evaluate");

    assert_eq!(resident.len(), streaming.len());
    for (resident, streaming) in resident.iter().zip(&streaming) {
        assert_projection_equal(resident, streaming);
    }
}

#[test]
fn single_member_differentials_preserve_resident_and_streaming_results() {
    let resident = ProjectionFixture::new(96, 2, Storage::Resident, ThreadPolicy::Serial)
        .expect("resident fixture should build")
        .evaluate_single(4)
        .expect("resident projections should evaluate");
    let streaming = ProjectionFixture::new(96, 2, Storage::Streaming, ThreadPolicy::Serial)
        .expect("streaming fixture should build")
        .evaluate_single(4)
        .expect("streaming projections should evaluate");

    assert_eq!(resident.len(), streaming.len());
    for (resident, streaming) in resident.iter().zip(&streaming) {
        assert_projection_equal(resident, streaming);
    }
}

#[test]
fn single_member_projection_sets_match_independent_calls_with_ensemble_draws() {
    for storage in [Storage::Resident, Storage::Streaming] {
        let fixture = ProjectionFixture::new(96, 3, storage, ThreadPolicy::Serial)
            .expect("projection fixture should build");
        let expected = fixture
            .evaluate_single(4)
            .expect("independent projections should evaluate");
        let actual = fixture
            .evaluate_single_set(4)
            .expect("projection set should evaluate");

        assert_eq!(actual.len(), expected.len());
        assert_eq!(
            actual.iter().map(|(name, _)| name).collect::<Vec<_>>(),
            vec![
                "projection_0",
                "projection_1",
                "projection_2",
                "projection_3"
            ]
        );
        for ((_, actual), expected) in actual.iter().zip(&expected) {
            assert_projection_equal(actual, expected);
        }
    }
}

#[test]
fn representative_fixture_captures_complete_legacy_reference_fingerprints() {
    let projections = ProjectionFixture::new(128, 3, Storage::Resident, ThreadPolicy::Serial)
        .expect("representative fixture should build")
        .evaluate_combined(4)
        .expect("projections should evaluate");
    let actual = projections.iter().map(fingerprint).collect::<Vec<_>>();
    let expected = [
        Fingerprint {
            finite_sum: 1922.5498761698045,
            weighted_sum: 507508.0526853,
            square_sum: 8802.073987413241,
            nan_hash: 14695981039346656037,
            value_count: 640,
        },
        Fingerprint {
            finite_sum: 624.0528287103864,
            weighted_sum: 166060.96750409793,
            square_sum: 896.42069045685,
            nan_hash: 14695981039346656037,
            value_count: 640,
        },
        Fingerprint {
            finite_sum: 1964.6290309499093,
            weighted_sum: 512833.96550840064,
            square_sum: 8729.792787230928,
            nan_hash: 14695981039346656037,
            value_count: 640,
        },
        Fingerprint {
            finite_sum: 593.7691271480444,
            weighted_sum: 155274.82300814957,
            square_sum: 809.6967765671711,
            nan_hash: 14695981039346656037,
            value_count: 640,
        },
    ];

    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(&expected) {
        assert_fingerprint_equal(*actual, *expected);
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Fingerprint {
    finite_sum: f64,
    weighted_sum: f64,
    square_sum: f64,
    nan_hash: u64,
    value_count: usize,
}

fn fingerprint(projection: &DifferentialCrossSection) -> Fingerprint {
    let estimates = [
        projection.data(),
        projection.model(),
        &projection.components()["background"],
        &projection.components()["signal"],
    ];
    let mut fingerprint = Fingerprint {
        nan_hash: 0xcbf2_9ce4_8422_2325,
        ..Fingerprint::default()
    };
    for estimate in estimates {
        fingerprint_values(&mut fingerprint, estimate.values());
        for draw in estimate.draws() {
            fingerprint_values(&mut fingerprint, draw);
        }
    }
    fingerprint
}

fn fingerprint_values(fingerprint: &mut Fingerprint, values: &[f64]) {
    for value in values {
        let index = fingerprint.value_count;
        fingerprint.value_count += 1;
        if value.is_nan() {
            fingerprint.nan_hash ^= index as u64;
            fingerprint.nan_hash = fingerprint.nan_hash.wrapping_mul(0x0000_0100_0000_01b3);
        } else {
            fingerprint.finite_sum += value;
            fingerprint.weighted_sum += (index + 1) as f64 * value;
            fingerprint.square_sum += value * value;
        }
    }
}

fn assert_fingerprint_equal(actual: Fingerprint, expected: Fingerprint) {
    assert_eq!(actual.nan_hash, expected.nan_hash);
    assert_eq!(actual.value_count, expected.value_count);
    assert_values_close(actual.finite_sum, expected.finite_sum);
    assert_values_close(actual.weighted_sum, expected.weighted_sum);
    assert_values_close(actual.square_sum, expected.square_sum);
}

fn assert_values_close(actual: f64, expected: f64) {
    let tolerance = 1e-10 + 1e-10 * actual.abs().max(expected.abs());
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual} != {expected}"
    );
}

fn assert_projection_equal(left: &DifferentialCrossSection, right: &DifferentialCrossSection) {
    assert_eq!(left.axes(), right.axes());
    assert_eq!(left.shape(), right.shape());
    assert_estimate_equal(left.data(), right.data());
    assert_estimate_equal(left.model(), right.model());
    assert_eq!(left.components().len(), right.components().len());
    for (name, estimate) in left.components() {
        assert_estimate_equal(estimate, &right.components()[name]);
    }
}

fn assert_estimate_equal(left: &BinnedEstimate, right: &BinnedEstimate) {
    assert_values_equal(left.values(), right.values());
    assert_eq!(left.draws().len(), right.draws().len());
    for (left, right) in left.draws().iter().zip(right.draws()) {
        assert_values_equal(left, right);
    }
}

fn assert_values_equal(left: &[f64], right: &[f64]) {
    assert_eq!(left.len(), right.len());
    for (left, right) in left.iter().zip(right) {
        if left.is_nan() || right.is_nan() {
            assert!(left.is_nan() && right.is_nan());
            continue;
        }
        let tolerance = 1e-10 + 1e-10 * left.abs().max(right.abs());
        assert!((left - right).abs() <= tolerance, "{left} != {right}");
    }
}
