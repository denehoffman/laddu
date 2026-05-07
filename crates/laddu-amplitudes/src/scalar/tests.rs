use std::{f64, sync::Arc};

use approx::assert_relative_eq;
use laddu_core::{data::test_dataset, parameter, traits::Variable, variables::Mass, PI};

use super::{
    components::{ComplexScalar, PolarComplexScalar, Scalar},
    variable::{VariableExpressionExt, VariableScalar},
};

#[test]
fn test_scalar_creation_and_evaluation() {
    let dataset = Arc::new(test_dataset());
    let expr = Scalar::new("test_scalar", parameter!("test_param")).unwrap();
    let evaluator = expr.load(&dataset).unwrap();

    let params = vec![2.5];
    let result = evaluator.evaluate(&params).unwrap();

    assert_relative_eq!(result[0].re, 2.5);
    assert_relative_eq!(result[0].im, 0.0);
}

#[test]
fn test_scalar_gradient() {
    let dataset = Arc::new(test_dataset());
    let expr = Scalar::new("test_scalar", parameter!("test_param"))
        .unwrap()
        .norm_sqr();
    let evaluator = expr.load(&dataset).unwrap();

    let params = vec![2.0];
    let gradient = evaluator.evaluate_gradient(&params).unwrap();

    assert_relative_eq!(gradient[0][0].re, 4.0);
    assert_relative_eq!(gradient[0][0].im, 0.0);
}

#[test]
fn test_variable_scalar_evaluation() {
    let dataset = Arc::new(test_dataset());
    let mut variable = Mass::new(["kshort1", "kshort2"]);
    variable.bind(dataset.metadata()).unwrap();
    let expected = variable.value(&dataset.event_local(0).unwrap());

    let expr = VariableScalar::new("mass", &variable).unwrap();
    let evaluator = expr.load(&dataset).unwrap();
    let result = evaluator.evaluate(&[]).unwrap();

    assert_relative_eq!(result[0].re, expected);
    assert_relative_eq!(result[0].im, 0.0);
}

#[test]
fn test_variable_scalar_has_no_parameters() {
    let dataset = Arc::new(test_dataset());
    let variable = Mass::new(["kshort1", "kshort2"]);
    let expr = VariableScalar::new("mass", &variable).unwrap();
    let evaluator = expr.load(&dataset).unwrap();

    assert!(evaluator.parameters().is_empty());
    assert!(evaluator.parameters().free().names().is_empty());
    assert!(evaluator.parameters().fixed().names().is_empty());
}

#[test]
fn test_variable_as_expression() {
    let dataset = Arc::new(test_dataset());
    let mut variable = Mass::new(["kshort1", "kshort2"]);
    variable.bind(dataset.metadata()).unwrap();
    let expected = variable.value(&dataset.event_local(0).unwrap());

    let expr = variable.as_expression("mass").unwrap();
    let evaluator = expr.load(&dataset).unwrap();
    let result = evaluator.evaluate(&[]).unwrap();

    assert_relative_eq!(result[0].re, expected);
    assert_relative_eq!(result[0].im, 0.0);
}

#[test]
fn test_complex_scalar_evaluation() {
    let dataset = Arc::new(test_dataset());
    let expr = ComplexScalar::new(
        "test_complex",
        parameter!("re_param"),
        parameter!("im_param"),
    )
    .unwrap();
    let evaluator = expr.load(&dataset).unwrap();

    let params = vec![1.5, 2.5];
    let result = evaluator.evaluate(&params).unwrap();

    assert_relative_eq!(result[0].re, 1.5);
    assert_relative_eq!(result[0].im, 2.5);
}

#[test]
fn test_complex_scalar_gradient() {
    let dataset = Arc::new(test_dataset());
    let expr = ComplexScalar::new(
        "test_complex",
        parameter!("re_param"),
        parameter!("im_param"),
    )
    .unwrap()
    .norm_sqr();
    let evaluator = expr.load(&dataset).unwrap();

    let params = vec![3.0, 4.0];
    let gradient = evaluator.evaluate_gradient(&params).unwrap();

    assert_relative_eq!(gradient[0][0].re, 6.0);
    assert_relative_eq!(gradient[0][0].im, 0.0);
    assert_relative_eq!(gradient[0][1].re, 8.0);
    assert_relative_eq!(gradient[0][1].im, 0.0);
}

#[test]
fn test_semantic_key_deduplicates_matching_complex_scalar() {
    let dataset = Arc::new(test_dataset());
    let expr = ComplexScalar::new(
        "same_complex",
        parameter!("re_param"),
        parameter!("im_param"),
    )
    .unwrap()
        + ComplexScalar::new(
            "same_complex",
            parameter!("re_param"),
            parameter!("im_param"),
        )
        .unwrap();
    let evaluator = expr.load(&dataset).unwrap();

    let result = evaluator.evaluate(&[1.5, 2.5]).unwrap();

    assert_eq!(evaluator.amplitudes.len(), 1);
    assert_relative_eq!(result[0].re, 3.0);
    assert_relative_eq!(result[0].im, 5.0);
}

#[test]
fn test_semantic_key_keeps_mismatched_complex_scalar_fields_separate() {
    let dataset = Arc::new(test_dataset());
    let expr = ComplexScalar::new(
        "same_complex",
        parameter!("re_param"),
        parameter!("im_param"),
    )
    .unwrap()
        + ComplexScalar::new(
            "same_complex",
            parameter!("other_re"),
            parameter!("im_param"),
        )
        .unwrap();
    let evaluator = expr.load(&dataset).unwrap();

    assert_eq!(evaluator.amplitudes.len(), 2);
}

#[test]
fn test_polar_complex_scalar_evaluation() {
    let dataset = Arc::new(test_dataset());
    let expr = PolarComplexScalar::new(
        "test_polar",
        parameter!("r_param"),
        parameter!("theta_param"),
    )
    .unwrap();
    let evaluator = expr.load(&dataset).unwrap();

    let r = 2.0;
    let theta = PI / 4.3;
    let params = vec![r, theta];
    let result = evaluator.evaluate(&params).unwrap();

    assert_relative_eq!(result[0].re, r * theta.cos());
    assert_relative_eq!(result[0].im, r * theta.sin());
}

#[test]
fn test_polar_complex_scalar_gradient() {
    let dataset = Arc::new(test_dataset());
    let expr = PolarComplexScalar::new(
        "test_polar",
        parameter!("r_param"),
        parameter!("theta_param"),
    )
    .unwrap();
    let evaluator = expr.load(&dataset).unwrap();

    let r = 2.0;
    let theta = PI / 4.3;
    let params = vec![r, theta];
    let gradient = evaluator.evaluate_gradient(&params).unwrap();

    assert_relative_eq!(gradient[0][0].re, f64::cos(theta));
    assert_relative_eq!(gradient[0][0].im, f64::sin(theta));
    assert_relative_eq!(gradient[0][1].re, -r * f64::sin(theta));
    assert_relative_eq!(gradient[0][1].im, r * f64::cos(theta));
}
