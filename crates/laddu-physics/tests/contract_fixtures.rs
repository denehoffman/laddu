//! Stable serialization and public error-category fixtures.

use laddu_physics::{
    LadduPhysicsError,
    generation::{MassProposal, ProposalRng, ScalarSource},
    histogram::Histogram,
    quantum::ParticleProperties,
    vectors::{RealVec3, RealVec4},
};
use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;

fn assert_json_fixture<T>(value: &T, fixture: &str)
where
    T: Serialize + DeserializeOwned,
{
    let expected: Value = serde_json::from_str(fixture).unwrap();
    assert_eq!(serde_json::to_value(value).unwrap(), expected);
    let decoded = serde_json::from_value::<T>(expected.clone()).unwrap();
    assert_eq!(serde_json::to_value(decoded).unwrap(), expected);
}

#[test]
fn public_physics_types_match_stable_json_fixtures() {
    assert_json_fixture(
        &RealVec3::new(1.25, -2.5, 0.75),
        include_str!("fixtures/real_vec3.json"),
    );
    assert_json_fixture(
        &RealVec4::new(4.5, 1.25, -2.5, 0.75),
        include_str!("fixtures/real_vec4.json"),
    );
    assert_json_fixture(
        &MassProposal::uniform(0.25, 2.5),
        include_str!("fixtures/mass_proposal.json"),
    );
    assert_json_fixture(
        &ScalarSource::constant(8.5),
        include_str!("fixtures/scalar_source_fixed.json"),
    );
    assert_json_fixture(
        &ScalarSource::uniform(8.0, 9.0),
        include_str!("fixtures/scalar_source_uniform.json"),
    );
    assert_json_fixture(
        &ScalarSource::histogram(Histogram::new(vec![4.0, 9.0], vec![0.0, 1.0, 4.0]).unwrap()),
        include_str!("fixtures/scalar_source_histogram.json"),
    );
    assert_json_fixture(
        &Histogram::new_with_flow(vec![4.0, 9.0], vec![0.0, 1.0, 4.0], 0.5, 1.25).unwrap(),
        include_str!("fixtures/histogram.json"),
    );
    assert_json_fixture(
        &ParticleProperties::unknown()
            .with_name("contract particle")
            .with_mass(1.25),
        include_str!("fixtures/particle_properties.json"),
    );
}

#[test]
fn scalar_source_exposes_its_json_schema() {
    let schema = serde_json::to_string(&schemars::schema_for!(ScalarSource)).unwrap();
    for variant in ["fixed", "uniform", "histogram"] {
        assert!(schema.contains(&format!(r#""const":"{variant}""#)));
    }
}

#[test]
fn public_error_categories_and_messages_remain_classifiable() {
    let cases = [
        (
            RealVec3::zero().unit().unwrap_err(),
            "InvalidValue",
            "Invalid value for vector magnitude: expected positive when constructing unit vector, got 0",
        ),
        (
            Histogram::new(vec![1.0], vec![0.0]).unwrap_err(),
            "InvalidLength",
            "Invalid length for histogram bin edges: expected at least 2, got 1",
        ),
        (
            MassProposal::uniform(2.0, 1.0)
                .propose(0.0, 3.0, &mut ProposalRng::new(7))
                .unwrap_err(),
            "InvalidRelation",
            "Invalid relation: uniform mass proposal requires finite low < high, got [2, 1]",
        ),
        (
            RealVec4::try_from(vec![1.0, 2.0]).unwrap_err(),
            "Custom",
            "Attempted to convert Vec<f64> to RealVec4 for Vec with len != 4",
        ),
    ];

    for (error, category, message) in cases {
        let actual_category = match error {
            LadduPhysicsError::ConversionError(_) => "ConversionError",
            LadduPhysicsError::ParseError { .. } => "ParseError",
            LadduPhysicsError::MissingParticleProperty { .. } => "MissingParticleProperty",
            LadduPhysicsError::InvalidValue { .. } => "InvalidValue",
            LadduPhysicsError::InvalidLength { .. } => "InvalidLength",
            LadduPhysicsError::InvalidRelation { .. } => "InvalidRelation",
            LadduPhysicsError::UnsupportedValue { .. } => "UnsupportedValue",
            LadduPhysicsError::NumericOverflow { .. } => "NumericOverflow",
            LadduPhysicsError::Custom(_) => "Custom",
        };
        assert_eq!(actual_category, category);
        assert_eq!(error.to_string(), message);
    }
}
