use super::*;

#[test]
fn constant_folding_preserves_signed_zero_across_branch_cuts() {
    let expression = (Expr::from(Complex64::new(-1.0, -0.0)) * 1.0).sqrt();
    let compiled = CompiledModel::from_expr(&expression).unwrap();

    assert!(matches!(
        compiled.graph().node(compiled.graph().root()),
        Some(ExprNode::ComplexConst(value))
            if value.re == 0.0 && value.im == -1.0
    ));
}

#[test]
fn aggressive_scalar_identities_simplify_self_operations() {
    let x = Expr::from(parameter!("x"));

    let subtract = CompiledModel::from_expr(&(x.clone() - x.clone())).unwrap();
    assert!(matches!(
        subtract.graph().node(subtract.graph().root()),
        Some(ExprNode::RealConst(0.0))
    ));

    let divide = CompiledModel::from_expr(&(x.clone() / x.clone())).unwrap();
    assert!(matches!(
        divide.graph().node(divide.graph().root()),
        Some(ExprNode::RealConst(1.0))
    ));

    let negated = CompiledModel::from_expr(&(0.0 - x)).unwrap();
    assert!(matches!(
        negated.graph().node(negated.graph().root()),
        Some(ExprNode::Unary {
            op: laddu_expr::UnaryOp::Neg,
            ..
        })
    ));
}

#[test]
fn aggressive_unary_identities_simplify_nested_projections() {
    let z = complex(parameter!("a_re"), parameter!("a_im"));
    let compiled = CompiledModel::from_expr(&z.conj().conj()).unwrap();

    assert!(matches!(
        compiled.graph().node(compiled.graph().root()),
        Some(ExprNode::Complex { .. })
    ));

    let x = event_scalar("z");
    let compiled = CompiledModel::from_expr(&x.real().real()).unwrap();
    assert!(matches!(
        compiled.graph().node(compiled.graph().root()),
        Some(ExprNode::EventScalar(name)) if name.as_ref() == "z"
    ));
}

#[test]
fn complex_parameter_projections_simplify_to_component_parameters() {
    let z = complex(parameter!("a_re"), parameter!("a_im"));
    let real = CompiledModel::from_expr(&z.real()).unwrap();
    let imag = CompiledModel::from_expr(&z.imag()).unwrap();

    assert!(matches!(
        real.graph().node(real.graph().root()),
        Some(ExprNode::ScalarParam(parameter)) if parameter.name() == "a_re"
    ));
    assert!(matches!(
        imag.graph().node(imag.graph().root()),
        Some(ExprNode::ScalarParam(parameter)) if parameter.name() == "a_im"
    ));
}

#[test]
fn complex_conjugation_rewrites_to_complex_with_negated_imaginary_part() {
    let z = complex(parameter!("a_re"), parameter!("a_im"));
    let compiled = CompiledModel::from_expr(&z.conj()).unwrap();

    assert!(compiled.graph().nodes().iter().all(|node| !matches!(
        node,
        ExprNode::Unary {
            op: laddu_expr::UnaryOp::Conj,
            ..
        }
    )));
    assert!(matches!(
        compiled.graph().node(compiled.graph().root()),
        Some(ExprNode::Complex { .. })
    ));
    assert!(compiled.graph().nodes().iter().any(|node| matches!(
        node,
        ExprNode::Unary {
            op: laddu_expr::UnaryOp::Neg,
            ..
        }
    )));
}

#[test]
fn subtraction_normalizes_to_signed_nary_addition() {
    let x = Expr::from(parameter!("x"));
    let y = Expr::from(parameter!("y"));
    let z = Expr::from(parameter!("z"));
    let compiled = CompiledModel::from_expr(&(x + y - z)).unwrap();

    assert!(compiled.graph().nodes().iter().all(|node| !matches!(
        node,
        ExprNode::Binary {
            op: BinaryOp::Sub,
            ..
        }
    )));
    assert!(matches!(
        compiled.graph().node(compiled.graph().root()),
        Some(ExprNode::NaryAdd { terms }) if terms.len() == 3
            && terms.iter().any(|id| matches!(
                compiled.graph().node(*id),
                Some(ExprNode::NaryMul { factors }) if factors.len() == 2
                    && factors.iter().any(|factor| matches!(compiled.graph().node(*factor), Some(ExprNode::RealConst(-1.0))))
                    && factors.iter().any(|factor| matches!(compiled.graph().node(*factor), Some(ExprNode::ScalarParam(parameter)) if parameter.name() == "z"))
            ))
    ));
}

#[test]
fn product_normalization_absorbs_negated_factors() {
    let phi = Expr::from(parameter!("phi"));
    let compiled = CompiledModel::from_expr(&(Expr::from(-2.0) * (0.0 - phi))).unwrap();

    assert!(matches!(
        compiled.graph().node(compiled.graph().root()),
        Some(ExprNode::NaryMul { factors }) if factors.len() == 2
            && factors.iter().any(|id| matches!(compiled.graph().node(*id), Some(ExprNode::RealConst(2.0))))
            && factors.iter().any(|id| matches!(compiled.graph().node(*id), Some(ExprNode::ScalarParam(parameter)) if parameter.name() == "phi"))
    ));
}

#[test]
fn product_normalization_collects_repeated_factors_into_powers() {
    let x = Expr::from(parameter!("x"));
    let compiled = CompiledModel::from_expr(&(x.clone() * x.clone() * x)).unwrap();

    assert!(matches!(
        compiled.graph().node(compiled.graph().root()),
        Some(ExprNode::Unary {
            op: UnaryOp::PowI(3),
            input,
        }) if matches!(compiled.graph().node(*input), Some(ExprNode::ScalarParam(parameter)) if parameter.name() == "x")
    ));
}

#[test]
fn product_normalization_combines_same_power_factors() {
    let x = Expr::from(parameter!("x"));
    let y = Expr::from(parameter!("y"));
    let compiled = CompiledModel::from_expr(&(x.powi(2) * y.powi(2))).unwrap();

    let Some(ExprNode::Unary {
        op: UnaryOp::PowI(2),
        input,
    }) = compiled.graph().node(compiled.graph().root())
    else {
        panic!("expected combined square root");
    };
    assert!(matches!(
        compiled.graph().node(*input),
        Some(ExprNode::NaryMul { factors }) if factors.len() == 2
            && factors.iter().any(|factor| matches!(compiled.graph().node(*factor), Some(ExprNode::ScalarParam(parameter)) if parameter.name() == "x"))
            && factors.iter().any(|factor| matches!(compiled.graph().node(*factor), Some(ExprNode::ScalarParam(parameter)) if parameter.name() == "y"))
    ));
}

#[test]
fn nary_add_constant_terms_are_folded_without_requiring_all_constants() {
    let x = Expr::from(parameter!("x"));
    let compiled = CompiledModel::from_expr(&(x + 2.0 + 3.0)).unwrap();

    assert!(matches!(
        compiled.graph().node(compiled.graph().root()),
        Some(ExprNode::NaryAdd { terms }) if terms.len() == 2
            && terms.iter().any(|id| matches!(compiled.graph().node(*id), Some(ExprNode::RealConst(5.0))))
            && terms.iter().any(|id| matches!(compiled.graph().node(*id), Some(ExprNode::ScalarParam(parameter)) if parameter.name() == "x"))
    ));
}

#[test]
fn power_identities_simplify_integer_powers() {
    let x = Expr::from(parameter!("x"));

    let identity = CompiledModel::from_expr(&x.powi(1)).unwrap();
    assert!(matches!(
        identity.graph().node(identity.graph().root()),
        Some(ExprNode::ScalarParam(parameter)) if parameter.name() == "x"
    ));

    let one = CompiledModel::from_expr(&x.powi(0)).unwrap();
    assert!(matches!(
        one.graph().node(one.graph().root()),
        Some(ExprNode::RealConst(1.0))
    ));

    let nested = CompiledModel::from_expr(&x.powi(2).powi(3)).unwrap();
    assert!(matches!(
        nested.graph().node(nested.graph().root()),
        Some(ExprNode::Unary {
            op: UnaryOp::PowI(6),
            input,
        }) if matches!(nested.graph().node(*input), Some(ExprNode::ScalarParam(parameter)) if parameter.name() == "x")
    ));
}

#[test]
fn scalar_division_normalizes_to_inverse_powers() {
    let x = Expr::from(parameter!("x"));
    let compiled = CompiledModel::from_expr(&(x.clone().powi(3) / x)).unwrap();

    assert!(compiled.graph().nodes().iter().all(|node| !matches!(
        node,
        ExprNode::Binary {
            op: BinaryOp::Div,
            ..
        }
    )));
    assert!(matches!(
        compiled.graph().node(compiled.graph().root()),
        Some(ExprNode::Unary {
            op: UnaryOp::PowI(2),
            input,
        }) if matches!(compiled.graph().node(*input), Some(ExprNode::ScalarParam(parameter)) if parameter.name() == "x")
    ));
}

#[test]
fn partial_common_product_factorization_groups_subset_terms() {
    let x = Expr::from(parameter!("x"));
    let y = Expr::from(parameter!("y"));
    let direct =
        CompiledModel::from_expr(&(1.0 + Complex64::I * x.clone() + Complex64::I * y.clone()))
            .unwrap();

    assert!(matches!(
        direct.graph().node(direct.graph().root()),
        Some(ExprNode::NaryAdd { terms }) if terms.len() == 2
            && terms.iter().any(|term| matches!(direct.graph().node(*term), Some(ExprNode::RealConst(1.0))))
            && terms.iter().any(|term| matches!(
                direct.graph().node(*term),
                Some(ExprNode::NaryMul { factors }) if factors.iter().any(|factor| matches!(
                    direct.graph().node(*factor),
                    Some(ExprNode::ComplexConst(value)) if *value == Complex64::I
                )) && factors.iter().any(|factor| matches!(
                    direct.graph().node(*factor),
                    Some(ExprNode::NaryAdd { terms }) if terms.len() == 2
                ))
            ))
    ));

    let compiled =
        CompiledModel::from_expr(&(1.0 + Complex64::I * x + Complex64::I * y).exp()).unwrap();

    let ExprNode::Unary {
        op: UnaryOp::Exp,
        input,
    } = compiled.graph().node(compiled.graph().root()).unwrap()
    else {
        panic!("expected root exp node");
    };
    assert!(matches!(
        compiled.graph().node(*input),
        Some(ExprNode::NaryAdd { terms }) if terms.len() == 2
            && terms.iter().any(|term| matches!(compiled.graph().node(*term), Some(ExprNode::RealConst(1.0))))
            && terms.iter().any(|term| matches!(
                compiled.graph().node(*term),
                Some(ExprNode::NaryMul { factors }) if factors.iter().any(|factor| matches!(
                    compiled.graph().node(*factor),
                    Some(ExprNode::ComplexConst(value)) if *value == Complex64::I
                )) && factors.iter().any(|factor| matches!(
                    compiled.graph().node(*factor),
                    Some(ExprNode::NaryAdd { terms }) if terms.len() == 2
                ))
            ))
    ));
}

#[test]
fn like_terms_with_real_coefficients_are_combined() {
    let x = Expr::from(parameter!("x"));
    let compiled = CompiledModel::from_expr(&(2.0 * x.clone() + 3.0 * x)).unwrap();

    assert!(matches!(
        compiled.graph().node(compiled.graph().root()),
        Some(ExprNode::NaryMul { factors }) if factors.len() == 2
            && factors.iter().any(|id| matches!(compiled.graph().node(*id), Some(ExprNode::RealConst(5.0))))
            && factors.iter().any(|id| matches!(compiled.graph().node(*id), Some(ExprNode::ScalarParam(parameter)) if parameter.name() == "x"))
    ));
}

#[test]
fn like_terms_cancel_in_nary_additions() {
    let x = Expr::from(parameter!("x"));
    let y = Expr::from(parameter!("y"));
    let compiled = CompiledModel::from_expr(&(x.clone() + y.clone() - x)).unwrap();

    assert!(matches!(
        compiled.graph().node(compiled.graph().root()),
        Some(ExprNode::ScalarParam(parameter)) if parameter.name() == "y"
    ));
}

#[test]
fn common_product_factor_extraction_handles_nary_sums() {
    let a = Expr::from(parameter!("a"));
    let b = Expr::from(parameter!("b"));
    let c = Expr::from(parameter!("c"));
    let x = Expr::from(parameter!("x"));
    let compiled =
        CompiledModel::from_expr(&(a * x.clone() + b * x.clone() + c * x.clone())).unwrap();

    let Some(ExprNode::NaryMul { factors }) = compiled.graph().node(compiled.graph().root()) else {
        panic!("expected factored product root");
    };

    assert!(factors.iter().any(|id| matches!(
        compiled.graph().node(*id),
        Some(ExprNode::ScalarParam(parameter)) if parameter.name() == "x"
    )));
    assert!(factors.iter().any(|id| matches!(
            compiled.graph().node(*id),
            Some(ExprNode::NaryAdd { terms }) if terms.len() == 3
                && terms.iter().any(|term| matches!(compiled.graph().node(*term), Some(ExprNode::ScalarParam(parameter)) if parameter.name() == "a"))
                && terms.iter().any(|term| matches!(compiled.graph().node(*term), Some(ExprNode::ScalarParam(parameter)) if parameter.name() == "b"))
                && terms.iter().any(|term| matches!(compiled.graph().node(*term), Some(ExprNode::ScalarParam(parameter)) if parameter.name() == "c"))
        )));
}

#[test]
fn cost_aware_common_product_factor_extraction_keeps_useful_rewrites() {
    let a = Expr::from(parameter!("a"));
    let b = Expr::from(parameter!("b"));
    let x = Expr::from(parameter!("x"));
    let compiled = CompiledModel::from_expr_with_options(
        &(a * x.clone() + b * x.clone()),
        &CompileOptions::with_pipeline(
            OptimizationPipeline::new()
                .with_pass(CanonicalCsePass)
                .with_pass(RewritePass::factor_common_products()),
        ),
    )
    .unwrap();

    assert!(matches!(
        compiled.graph().node(compiled.graph().root()),
        Some(ExprNode::NaryMul { factors }) if factors.iter().any(|id| matches!(
            compiled.graph().node(*id),
            Some(ExprNode::ScalarParam(parameter)) if parameter.name() == "x"
        ))
    ));
}

#[test]
fn cost_aware_common_product_factor_extraction_rejects_more_expensive_rewrites() {
    let compiled = CompiledModel::from_expr_with_options(
        &(Expr::from(2.0) + 4.0),
        &CompileOptions::with_pipeline(
            OptimizationPipeline::new().with_pass(RewritePass::factor_common_products()),
        ),
    )
    .unwrap();

    assert!(matches!(
        compiled.graph().node(compiled.graph().root()),
        Some(ExprNode::Binary {
            op: BinaryOp::Add,
            ..
        })
    ));
}

#[test]
fn common_product_factor_extraction_handles_partial_powers() {
    let x = Expr::from(parameter!("x"));
    let a = Expr::from(parameter!("a"));
    let b = Expr::from(parameter!("b"));
    let compiled = CompiledModel::from_expr(&(a * x.clone().powi(3) - b * x.powi(2))).unwrap();

    let Some(ExprNode::NaryMul { factors }) = compiled.graph().node(compiled.graph().root()) else {
        panic!("expected factored product root");
    };

    assert!(factors.iter().any(|id| matches!(
            compiled.graph().node(*id),
            Some(ExprNode::Unary {
                op: UnaryOp::PowI(2),
                input,
            }) if matches!(compiled.graph().node(*input), Some(ExprNode::ScalarParam(parameter)) if parameter.name() == "x")
        )));
    assert!(factors.iter().any(|id| matches!(
            compiled.graph().node(*id),
            Some(ExprNode::NaryAdd { terms }) if terms.len() == 2
                && terms.iter().any(|term| matches!(
                    compiled.graph().node(*term),
                    Some(ExprNode::NaryMul { factors }) if factors.iter().any(|factor| matches!(compiled.graph().node(*factor), Some(ExprNode::ScalarParam(parameter)) if parameter.name() == "a"))
                        && factors.iter().any(|factor| matches!(compiled.graph().node(*factor), Some(ExprNode::ScalarParam(parameter)) if parameter.name() == "x"))
                ))
                && terms.iter().any(|term| matches!(
                    compiled.graph().node(*term),
                    Some(ExprNode::NaryMul { factors }) if factors.iter().any(|factor| matches!(compiled.graph().node(*factor), Some(ExprNode::RealConst(-1.0))))
                        && factors.iter().any(|factor| matches!(compiled.graph().node(*factor), Some(ExprNode::ScalarParam(parameter)) if parameter.name() == "b"))
                ))
        )));
}

#[test]
fn common_product_factor_extraction_runs_before_coefficient_folding() {
    let c = Expr::from(parameter!("c"));
    let d = Expr::from(parameter!("d"));
    let lhs = Expr::from(-1.0) * -3.0 * 5.0 * 7.0 * c.clone() * c.clone() * d.clone() * d.clone();
    let rhs = Expr::from(-1.0) * -3.0 * 5.0 * d.clone() * d.clone();
    let compiled = CompiledModel::from_expr(&(lhs - rhs)).unwrap();

    let Some(ExprNode::NaryMul { factors }) = compiled.graph().node(compiled.graph().root()) else {
        panic!("expected factored product root");
    };

    assert!(
        factors
            .iter()
            .any(|id| matches!(compiled.graph().node(*id), Some(ExprNode::RealConst(15.0))))
    );
    assert!(factors.iter().any(|id| matches!(
            compiled.graph().node(*id),
            Some(ExprNode::Unary {
                op: UnaryOp::PowI(2),
                input,
            }) if matches!(compiled.graph().node(*input), Some(ExprNode::ScalarParam(parameter)) if parameter.name() == "d")
        )));
    assert!(factors.iter().any(|id| matches!(
            compiled.graph().node(*id),
            Some(ExprNode::NaryAdd { terms }) if terms.len() == 2
                && terms.iter().any(|term| matches!(compiled.graph().node(*term), Some(ExprNode::RealConst(-1.0))))
                && terms.iter().any(|term| matches!(
                    compiled.graph().node(*term),
                    Some(ExprNode::NaryMul { factors }) if factors.len() == 2
                        && factors.iter().any(|factor| matches!(compiled.graph().node(*factor), Some(ExprNode::RealConst(7.0))))
                        && factors.iter().any(|factor| matches!(
                            compiled.graph().node(*factor),
                            Some(ExprNode::Unary {
                                op: UnaryOp::PowI(2),
                                input,
                            }) if matches!(compiled.graph().node(*input), Some(ExprNode::ScalarParam(parameter)) if parameter.name() == "c")
                        ))
                ))
        )));
}

#[test]
fn complex_fact_rule_simplifies_real_projection() {
    let model = Expr::from(parameter!("x")).imag();
    let compiled = CompiledModel::from_expr(&model).unwrap();

    assert!(matches!(
        compiled.graph().node(compiled.graph().root()),
        Some(ExprNode::RealConst(0.0))
    ));
    assert_eq!(
        compiled
            .node_facts(compiled.graph().root())
            .unwrap()
            .number_class,
        NumberClass::Real
    );
}
