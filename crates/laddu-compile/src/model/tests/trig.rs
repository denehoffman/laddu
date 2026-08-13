use super::*;

#[test]
fn exponential_products_merge_after_multiplication_reassociation() {
    let a = event_scalar("a");
    let b = event_scalar("b");
    let scale = Expr::from(parameter!("scale"));
    let compiled = CompiledModel::from_expr(&(a.exp() * scale * b.exp())).unwrap();

    assert_eq!(count_unary_op(&compiled, laddu_expr::UnaryOp::Exp), 1);
}

#[test]
fn exponential_product_partition_keeps_non_exponential_factors() {
    let a = event_scalar("a");
    let b = event_scalar("b");
    let scale = Expr::from(parameter!("scale"));
    let compiled = CompiledModel::from_expr(&(a.exp() * b.exp() * scale)).unwrap();

    assert_eq!(count_unary_op(&compiled, UnaryOp::Exp), 1);
    assert_eq!(count_nary_mul(&compiled), 1);

    let ExprNode::NaryMul { factors } = compiled
        .graph()
        .node(compiled.graph().root())
        .expect("root node exists")
    else {
        panic!("expected n-ary product root");
    };
    assert!(factors.iter().any(|id| matches!(
        compiled.graph().node(*id),
        Some(ExprNode::ScalarParam(parameter)) if parameter.name() == "scale"
    )));
    assert!(factors.iter().any(|id| matches!(
            compiled.graph().node(*id),
            Some(ExprNode::Unary {
                op: UnaryOp::Exp,
                input,
            }) if matches!(compiled.graph().node(*input), Some(ExprNode::NaryAdd { terms }) if terms.len() == 2)
        )));
}

#[test]
fn polar_complex_products_merge_exponential_phase_factors() {
    let lhs = polar_complex(parameter!("m1"), event_scalar("p1"));
    let rhs = polar_complex(parameter!("m2"), event_scalar("p2"));
    let compiled = CompiledModel::from_expr(&(lhs * rhs)).unwrap();

    assert_eq!(count_unary_op(&compiled, laddu_expr::UnaryOp::Exp), 1);
}

#[test]
fn common_complex_phase_factor_is_factored_from_sum() {
    let p1 = event_scalar("p1");
    let p2 = event_scalar("p2");
    let compiled = CompiledModel::from_expr(&(Complex64::I * p1 + Complex64::I * p2)).unwrap();

    assert!(matches!(
        compiled.graph().node(compiled.graph().root()),
        Some(ExprNode::NaryMul { factors }) if factors.len() == 2
            && factors.iter().any(|id| matches!(compiled.graph().node(*id), Some(ExprNode::ComplexConst(value)) if *value == Complex64::I))
            && factors.iter().any(|id| matches!(compiled.graph().node(*id), Some(ExprNode::NaryAdd { .. })))
    ));
}

#[test]
fn trig_identities_simplify_common_pythagorean_forms() {
    let phi = Expr::from(parameter!("phi"));
    let sin_cos = CompiledModel::from_expr(&(phi.sin().powi(2) + phi.cos().powi(2))).unwrap();
    assert!(matches!(
        sin_cos.graph().node(sin_cos.graph().root()),
        Some(ExprNode::RealConst(1.0))
    ));

    let one_minus_cos = CompiledModel::from_expr(&(1.0 - phi.cos().powi(2))).unwrap();
    assert!(matches!(
        one_minus_cos.graph().node(one_minus_cos.graph().root()),
        Some(ExprNode::Unary {
            op: UnaryOp::PowI(2),
            input,
        }) if matches!(
            one_minus_cos.graph().node(*input),
            Some(ExprNode::Unary {
                op: UnaryOp::Sin,
                ..
            })
        )
    ));
}

#[test]
fn trig_parity_normalizes_negative_real_arguments() {
    let phi = Expr::from(parameter!("phi"));
    let sin = CompiledModel::from_expr(&(-phi.clone()).sin()).unwrap();
    let cos = CompiledModel::from_expr(&(-phi).cos()).unwrap();

    assert!(matches!(
        sin.graph().node(sin.graph().root()),
        Some(ExprNode::Unary {
            op: UnaryOp::Neg,
            input,
        }) if matches!(
            sin.graph().node(*input),
            Some(ExprNode::Unary {
                op: UnaryOp::Sin,
                ..
            })
        )
    ));
    assert!(matches!(
        cos.graph().node(cos.graph().root()),
        Some(ExprNode::Unary {
            op: UnaryOp::Cos,
            input,
        }) if matches!(cos.graph().node(*input), Some(ExprNode::ScalarParam(parameter)) if parameter.name() == "phi")
    ));
}

#[test]
fn euler_forms_rewrite_to_exponentials() {
    let phi = Expr::from(parameter!("phi"));
    let positive = CompiledModel::from_expr(&(phi.cos() + Complex64::I * phi.sin())).unwrap();
    let phi = Expr::from(parameter!("phi"));
    let negative = CompiledModel::from_expr(&(phi.cos() - Complex64::I * phi.sin())).unwrap();

    for compiled in [positive, negative] {
        assert_eq!(count_unary_op(&compiled, UnaryOp::Exp), 1);
        assert_eq!(count_unary_op(&compiled, UnaryOp::Sin), 0);
        assert_eq!(count_unary_op(&compiled, UnaryOp::Cos), 0);
    }
}

#[test]
fn euler_forms_preserve_common_real_scalar_factor() {
    let phi = Expr::from(parameter!("phi"));
    let angle = 2.0 * phi;
    let compiled = CompiledModel::from_expr(
        &(0.6690465435572891 * angle.cos() + Complex64::new(0.0, 0.6690465435572891) * angle.sin()),
    )
    .unwrap();

    assert_eq!(count_unary_op(&compiled, UnaryOp::Exp), 1);
    assert_eq!(count_unary_op(&compiled, UnaryOp::Sin), 0);
    assert_eq!(count_unary_op(&compiled, UnaryOp::Cos), 0);
    assert!(has_real_const(&compiled, 0.6690465435572891));
}

#[test]
fn negative_angle_euler_form_rewrites_to_negative_phase_exponential() {
    let costheta = Expr::from(parameter!("costheta"));
    let phi = Expr::from(parameter!("phi"));
    let angle = -(costheta + phi);
    let compiled = CompiledModel::from_expr(&(Complex64::I * angle.sin() + angle.cos())).unwrap();

    assert_eq!(count_unary_op(&compiled, UnaryOp::Exp), 1);
    assert_eq!(count_unary_op(&compiled, UnaryOp::Sin), 0);
    assert_eq!(count_unary_op(&compiled, UnaryOp::Cos), 0);
}

#[test]
fn imaginary_exponential_phases_merge_under_single_i_factor() {
    let costheta = Expr::from(parameter!("costheta"));
    let phi = Expr::from(parameter!("phi"));
    let compiled = CompiledModel::from_expr(
        &((Complex64::I * (2.0 * phi.clone())).exp()
            * (Complex64::I * (-(costheta.clone() + phi))).exp()),
    )
    .unwrap();

    let ExprNode::Unary {
        op: UnaryOp::Exp,
        input,
    } = compiled.graph().node(compiled.graph().root()).unwrap()
    else {
        panic!("expected root exp node");
    };
    assert!(matches!(
        compiled.graph().node(*input),
        Some(ExprNode::NaryMul { factors }) if factors.iter().any(|factor| matches!(
            compiled.graph().node(*factor),
            Some(ExprNode::ComplexConst(value)) if *value == Complex64::I
        ))
    ));
    assert_eq!(count_unary_op(&compiled, UnaryOp::Exp), 1);
}

#[test]
fn linear_phase_terms_are_collected_after_phase_merging() {
    let costheta = Expr::from(parameter!("costheta"));
    let phi = Expr::from(parameter!("phi"));
    let compiled = CompiledModel::from_expr(
        &((Complex64::I * (2.0 * phi.clone())).exp()
            * (Complex64::I * (-(costheta.clone() + phi))).exp()),
    )
    .unwrap();

    assert_eq!(format!("{}", compiled.graph()), "exp(i * (phi - costheta))");
}

#[test]
fn sqrt_square_and_half_angle_identities_simplify() {
    let costheta = Expr::from(parameter!("costheta"));
    let phi = Expr::from(parameter!("phi"));
    let sqrt_square = CompiledModel::from_expr(&((1.0 - costheta.powi(2)).sqrt().powi(2))).unwrap();
    assert_eq!(count_unary_op(&sqrt_square, UnaryOp::Sqrt), 0);
    assert!(matches!(
        sqrt_square.graph().node(sqrt_square.graph().root()),
        Some(ExprNode::NaryAdd { .. })
    ));

    let half = CompiledModel::from_expr(&(0.5 * (0.5 * phi.clone()).sin().powi(2))).unwrap();
    assert_eq!(count_unary_op(&half, UnaryOp::Sin), 0);
    assert!(has_real_const(&half, 0.25));

    let polynomial = CompiledModel::from_expr(
        &(3.0 * (0.5 * phi.clone()).cos().powi(2) - (0.5 * phi).sin().powi(2)),
    )
    .unwrap();
    assert_eq!(count_unary_op(&polynomial, UnaryOp::Sin), 0);
    assert_eq!(count_unary_op(&polynomial, UnaryOp::Cos), 1);
    assert!(has_real_const(&polynomial, 1.0));
    assert!(has_real_const(&polynomial, 2.0));
}

#[test]
fn half_angle_fourth_power_polynomial_simplifies() {
    let phi = Expr::from(parameter!("phi"));
    let compiled = CompiledModel::from_expr(
        &(0.75 * (1.0 - phi.clone().cos()) * (1.0 + phi.clone().cos()) - (0.5 * phi).sin().powi(4)),
    )
    .unwrap();

    assert_eq!(count_unary_op(&compiled, UnaryOp::Sin), 0);
    assert_eq!(count_unary_op(&compiled, UnaryOp::Cos), 1);
    assert!(has_real_const(&compiled, 0.5));
    assert!(has_real_const(&compiled, 1.0));
    assert!(has_real_const(&compiled, 2.0));
}

#[test]
fn polar_complex_product_combines_phases_under_single_i_factor() {
    let lhs = polar_complex(parameter!("m1"), event_scalar("p1"));
    let rhs = polar_complex(parameter!("m2"), event_scalar("p2"));
    let compiled = CompiledModel::from_expr(&(lhs * rhs)).unwrap();
    let exp_input = compiled
        .graph()
        .nodes()
        .iter()
        .find_map(|node| match node {
            ExprNode::Unary {
                op: UnaryOp::Exp,
                input,
            } => Some(*input),
            _ => None,
        })
        .unwrap();

    assert!(matches!(
        compiled.graph().node(exp_input),
        Some(ExprNode::NaryMul { factors }) if factors.len() == 2
            && factors.iter().any(|id| matches!(compiled.graph().node(*id), Some(ExprNode::ComplexConst(value)) if *value == Complex64::I))
            && factors.iter().any(|id| matches!(compiled.graph().node(*id), Some(ExprNode::NaryAdd { .. })))
    ));
}
