use super::*;

#[derive(Copy, Clone, Debug)]
struct ReplaceTwoWithFour;

impl RewriteRule for ReplaceTwoWithFour {
    fn name(&self) -> &'static str {
        "replace-two-with-four"
    }

    fn rewrite(
        &self,
        node: &ExprNode,
        metadata: &ExprMetadata,
        _context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite> {
        if matches!(node, ExprNode::RealConst(2.0)) {
            Ok(Rewrite::Replace {
                node: ExprNode::RealConst(4.0),
                metadata: metadata.clone(),
            })
        } else {
            Ok(Rewrite::Keep)
        }
    }
}

#[test]
fn cost_gate_rejects_more_expensive_candidate_pipeline() {
    let x = Expr::from(parameter!("x"));
    let baseline =
        CompiledModel::from_expr_with_options(&x, &CompileOptions::without_optimizations())
            .unwrap();
    let gated = CompiledModel::from_expr_with_options(
        &x,
        &CompileOptions::with_pipeline(OptimizationPipeline::new().with_pass(CostGatePass::new(
            OptimizationPipeline::new().with_pass(WrapRootInExp),
        ))),
    )
    .unwrap();

    assert_eq!(baseline.graph().root(), gated.graph().root());
    assert_eq!(baseline.graph().nodes(), gated.graph().nodes());
    assert_eq!(baseline.cost(), gated.cost());
}

#[test]
fn cost_gated_norm_sqr_expansion_removes_unit_phase_when_cheaper() {
    let costheta = Expr::from(parameter!("costheta"));
    let phi = Expr::from(parameter!("phi"));
    let expr = ((Complex64::I * phi).exp() * (1.0 + costheta)).norm_sqr();
    let without_gate = CompiledModel::from_expr_with_options(
        &expr,
        &CompileOptions::with_pipeline(
            OptimizationPipeline::new()
                .with_pass(RewritePass::simplify())
                .with_pass(CanonicalCsePass)
                .with_pass(RewritePass::normalize_add_mul())
                .with_pass(CanonicalCsePass)
                .with_pass(RewritePass::combine_like_terms())
                .with_pass(CanonicalCsePass)
                .with_pass(RewritePass::factor_common_products())
                .with_pass(RewritePass::normalize_add_mul())
                .with_pass(CanonicalCsePass)
                .with_pass(RewritePass::exponential())
                .with_pass(RewritePass::simplify())
                .with_max_iterations(16),
        ),
    )
    .unwrap();
    let with_gate = CompiledModel::from_expr(&expr).unwrap();

    assert!(with_gate.cost().weighted_ops() < without_gate.cost().weighted_ops());
    assert_eq!(count_unary_op(&with_gate, UnaryOp::NormSqr), 0);
    assert_eq!(count_unary_op(&with_gate, UnaryOp::Exp), 0);
}

#[test]
fn custom_rewrite_rules_replace_local_node_patterns() {
    let options = CompileOptions::with_pipeline(
        OptimizationPipeline::new()
            .with_pass(RewritePass::new("custom").with_rule(ReplaceTwoWithFour)),
    );
    let model = Expr::from(2.0) + 1.0;
    let compiled = CompiledModel::from_expr_with_options(&model, &options).unwrap();

    assert!(
        compiled
            .graph()
            .nodes()
            .iter()
            .any(|node| matches!(node, ExprNode::RealConst(4.0)))
    );
}

#[test]
fn default_pipeline_simplifies_scalar_identities() {
    let model = (parameter!("x") + 0.0) * 1.0;
    let compiled = CompiledModel::from_expr(&model).unwrap();

    assert_eq!(compiled.graph().nodes().len(), 1);
    assert!(matches!(
        compiled.graph().node(compiled.graph().root()),
        Some(ExprNode::ScalarParam(parameter)) if parameter.name() == "x"
    ));
}

#[test]
fn default_pipeline_constant_folds_scalar_nodes() {
    let model = (Expr::from(2.0) + 3.0).powi(2);
    let compiled = CompiledModel::from_expr(&model).unwrap();

    assert!(matches!(
        compiled.graph().node(compiled.graph().root()),
        Some(ExprNode::RealConst(25.0))
    ));
}

#[test]
fn default_pipeline_reduces_structural_squared_norms() {
    let value = complex(event_scalar("x"), event_scalar("y"));
    let compiled = CompiledModel::from_expr(&value.clone().conj().norm_sqr()).unwrap();

    assert_eq!(count_unary_op(&compiled, UnaryOp::Conj), 0);
    assert_eq!(count_unary_op(&compiled, UnaryOp::NormSqr), 0);
    assert_eq!(count_unary_op(&compiled, UnaryOp::PowI(2)), 2);
    assert!(matches!(
        compiled.graph().node(compiled.graph().root()),
        Some(ExprNode::NaryAdd { terms }) if terms.len() == 2
    ));
}

#[test]
fn default_pipeline_uses_simplify_cse_simplify() {
    let x = Expr::from(parameter!("x"));
    let model = (x.clone() + 0.0) - x;
    let compiled = CompiledModel::from_expr(&model).unwrap();

    assert_eq!(compiled.graph().nodes().len(), 1);
    assert!(matches!(
        compiled.graph().node(compiled.graph().root()),
        Some(ExprNode::RealConst(0.0))
    ));
}

#[test]
fn custom_pipeline_can_include_canonical_cse() {
    let x = Expr::from(parameter!("x"));
    let y = Expr::from(parameter!("y"));
    let sum = x + y;
    let options =
        CompileOptions::with_pipeline(OptimizationPipeline::new().with_pass(CanonicalCsePass));
    let compiled = CompiledModel::from_expr_with_options(&(sum.clone() * sum), &options).unwrap();

    assert_eq!(count_nary_add(&compiled), 1);
}

#[test]
fn custom_pipeline_can_omit_canonical_cse() {
    let x = Expr::from(parameter!("x"));
    let y = Expr::from(parameter!("y"));
    let lhs = x.clone() + y.clone();
    let rhs = x + y;
    let options = CompileOptions::with_pipeline(
        OptimizationPipeline::new().with_pass(RewritePass::simplify()),
    );
    let compiled = CompiledModel::from_expr_with_options(&(lhs * rhs), &options).unwrap();

    assert_eq!(count_binary_op(&compiled, BinaryOp::Add), 2);
}

#[test]
fn fixed_point_pipeline_revisits_nodes_created_by_earlier_iterations() {
    let expr =
        (Complex64::I * event_scalar("p1")).exp() * (Complex64::I * event_scalar("p2")).exp();
    let one_iteration = CompileOptions::with_pipeline(
        OptimizationPipeline::new()
            .with_pass(CanonicalCsePass)
            .with_pass(RewritePass::factor_common_products())
            .with_pass(RewritePass::exponential()),
    );
    let fixed_point = CompileOptions::with_pipeline(
        OptimizationPipeline::new()
            .with_pass(CanonicalCsePass)
            .with_pass(RewritePass::factor_common_products())
            .with_pass(RewritePass::exponential())
            .with_max_iterations(4),
    );
    let one_iteration = CompiledModel::from_expr_with_options(&expr, &one_iteration).unwrap();
    let fixed_point = CompiledModel::from_expr_with_options(&expr, &fixed_point).unwrap();

    let ExprNode::Unary {
        op: UnaryOp::Exp,
        input: one_iteration_exp_input,
    } = one_iteration
        .graph()
        .node(one_iteration.graph().root())
        .unwrap()
    else {
        panic!("expected root exp node");
    };
    assert!(matches!(
        one_iteration.graph().node(*one_iteration_exp_input),
        Some(ExprNode::NaryMul { factors }) if factors.len() == 2
            && factors.iter().any(|id| matches!(one_iteration.graph().node(*id), Some(ExprNode::ComplexConst(value)) if *value == Complex64::I))
            && factors.iter().any(|id| matches!(one_iteration.graph().node(*id), Some(ExprNode::NaryAdd { .. })))
    ));

    let ExprNode::Unary {
        op: UnaryOp::Exp,
        input: fixed_point_exp_input,
    } = fixed_point
        .graph()
        .node(fixed_point.graph().root())
        .unwrap()
    else {
        panic!("expected root exp node");
    };
    assert!(matches!(
        fixed_point.graph().node(*fixed_point_exp_input),
        Some(ExprNode::NaryMul { factors }) if factors.len() == 2
            && factors.iter().any(|id| matches!(fixed_point.graph().node(*id), Some(ExprNode::ComplexConst(value)) if *value == Complex64::I))
            && factors.iter().any(|id| matches!(fixed_point.graph().node(*id), Some(ExprNode::NaryAdd { .. })))
    ));
}
