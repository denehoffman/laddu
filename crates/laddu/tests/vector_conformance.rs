//! Cross-crate storage, expression, and runtime four-vector contracts.

use std::sync::Arc;

use laddu::{
    data::{
        data::{Dataset, OwnedEvent},
        schema::Schema,
    },
    physics::vectors::{RealVec4, Vec4},
    runtime::{DatasetExprExt, Execution},
};

#[test]
fn stored_vectors_preserve_lorentz_invariants_through_runtime_evaluation() {
    let vectors = [
        RealVec4::new(5.0, 1.0, -2.0, 0.5),
        RealVec4::new(3.5, -0.25, 0.75, 1.5),
        RealVec4::new(8.0, 2.5, 1.25, -3.0),
        RealVec4::new(1.0, 0.0, 0.0, 1.0),
    ];
    let schema = Arc::new(Schema::new(["p"], std::iter::empty::<&str>(), false).unwrap());
    let dataset = Dataset::from_events(
        schema,
        vectors
            .iter()
            .copied()
            .map(|p4| OwnedEvent::new(vec![p4], vec![])),
    )
    .unwrap();

    let p4 = Vec4::event("p");
    let evaluated = dataset
        .evaluate_real(&p4.m2(), &Execution::default())
        .unwrap();
    let expected = vectors.map(|p4| p4.m2());

    assert_eq!(evaluated.len(), expected.len());
    for (index, (actual, expected)) in evaluated.iter().zip(expected).enumerate() {
        let scale = expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= 1.0e-12 * scale,
            "event {index}: runtime m2 {actual} did not match stored m2 {expected}"
        );
    }
}
