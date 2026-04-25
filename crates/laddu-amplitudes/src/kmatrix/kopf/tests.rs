use std::sync::Arc;

use approx::assert_relative_eq;
use laddu_core::{data::test_dataset, parameter, variables::Mass};

use super::{KopfKMatrixA0, KopfKMatrixA0Channel};

#[test]
fn test_resampled_evaluation() {
    let res_mass = Mass::new(["kshort1", "kshort2"]);
    let expr = KopfKMatrixA0::new(
        "a0",
        [
            [parameter!("p0"), parameter!("p1")],
            [parameter!("p2"), parameter!("p3")],
        ],
        KopfKMatrixA0Channel::KKbar,
        &res_mass,
        Some(1),
    )
    .unwrap();

    let dataset = Arc::new(test_dataset());
    let evaluator = expr.load(&dataset).unwrap();

    let result = evaluator.evaluate(&[0.1, 0.2, 0.3, 0.4]).unwrap();

    assert_relative_eq!(result[0].re, -0.8428829840871046);
    assert_relative_eq!(result[0].im, -0.018842179274928372);
}

#[test]
fn test_resampled_gradient() {
    let res_mass = Mass::new(["kshort1", "kshort2"]);
    let expr = KopfKMatrixA0::new(
        "a0",
        [
            [parameter!("p0"), parameter!("p1")],
            [parameter!("p2"), parameter!("p3")],
        ],
        KopfKMatrixA0Channel::KKbar,
        &res_mass,
        Some(1),
    )
    .unwrap();

    let dataset = Arc::new(test_dataset());
    let evaluator = expr.load(&dataset).unwrap();

    let result = evaluator.evaluate_gradient(&[0.1, 0.2, 0.3, 0.4]).unwrap();

    assert_relative_eq!(result[0][0].re, 0.30662648055639463);
    assert_relative_eq!(result[0][0].im, -0.04825756855221591);
    assert_relative_eq!(result[0][1].re, -result[0][0].im);
    assert_relative_eq!(result[0][1].im, result[0][0].re);
    assert_relative_eq!(result[0][2].re, -1.180383324673402);
    assert_relative_eq!(result[0][2].im, 1.3227053711279164);
    assert_relative_eq!(result[0][3].re, -result[0][2].im);
    assert_relative_eq!(result[0][3].im, result[0][2].re);
}
