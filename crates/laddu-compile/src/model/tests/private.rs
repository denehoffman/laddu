use super::*;

#[test]
fn compile_options_select_search_or_source_graph() {
    assert!(CompileOptions::default().optimize);
    assert!(!CompileOptions::without_optimizations().optimize);
}

#[test]
fn compiler_bakes_fixed_parameters_before_search() {
    let source = (Expr::from(Parameter::fixed("scale", 2.0)) * event_scalar("x")).to_graph();
    let parameter_baked = Compiler::bake_parameters(&source);
    assert!(
        !parameter_baked
            .nodes()
            .iter()
            .any(|node| matches!(node, ExprNode::ScalarParam(_)))
    );
}

#[test]
fn normalization_submodel_disables_analysis() {
    let source = (event_scalar("x") + event_scalar("x")).to_graph();
    let compiled = CompiledModel::from_graph_without_normalization(source).unwrap();
    assert!(matches!(
        compiled.normalization_diagnostics().fallback_reason(),
        Some(
            crate::NormalizationFallbackReason::UnsupportedMixedOperation {
                operation: "normalization analysis disabled",
                ..
            }
        )
    ));
    assert_eq!(compiled.cache_plan().len(), 1);
    assert_eq!(
        compiled
            .optimization_diagnostics()
            .unwrap()
            .normalization_fallback(),
        Some("normalization analysis disabled")
    );
}

#[test]
fn normalization_cache_digest_includes_its_plan() {
    let x = event_scalar("x");
    let expr = (Expr::from(parameter!("scale")) * x).norm_sqr();
    let normal = CompiledModel::from_expr(&expr).unwrap();
    let disabled = CompiledModel::from_graph_without_normalization(expr.to_graph()).unwrap();
    assert_eq!(normal.graph().nodes(), disabled.graph().nodes());
    assert_ne!(normal.optimized_digest(), disabled.optimized_digest());
}

#[test]
fn compiled_query_deduplicates_structurally_repeated_outputs() {
    let x = event_scalar("x");
    let query = CompiledQuery::from_exprs([x.clone(), x, event_scalar("y")]).unwrap();
    assert_eq!(query.outputs().len(), 3);
    assert_eq!(query.outputs()[0], query.outputs()[1]);
    assert_eq!(
        query
            .model()
            .graph()
            .nodes()
            .iter()
            .filter(|node| matches!(node, ExprNode::EventScalar(name) if name.as_ref() == "x"))
            .count(),
        1
    );
}
