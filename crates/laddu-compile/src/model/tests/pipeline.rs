use super::*;

#[test]
fn search_selects_cheaper_equivalent() {
    let x = Expr::from(parameter!("x"));
    let expr = (x.clone() + 0.0) * 1.0;
    let compiled = CompiledModel::from_expr(&expr).unwrap();
    assert!(matches!(compiled.graph().node(compiled.graph().root()),
        Some(ExprNode::ScalarParam(parameter)) if parameter.name() == "x"));
    assert!(
        compiled
            .optimization_diagnostics()
            .unwrap()
            .execution_exact()
    );
}

#[test]
fn search_constant_folds_scalar_nodes() {
    let expr = (Expr::from(2.0) + 3.0).powi(2);
    let compiled = CompiledModel::from_expr(&expr).unwrap();
    assert!(matches!(
        compiled.graph().node(compiled.graph().root()),
        Some(ExprNode::RealConst(25.0))
    ));
}

#[test]
fn search_removes_unit_phase_from_squared_norm() {
    let costheta = Expr::from(parameter!("costheta"));
    let phi = Expr::from(parameter!("phi"));
    let expr = ((Complex64::I * phi).exp() * (1.0 + costheta)).norm_sqr();
    let source =
        CompiledModel::from_expr_with_options(&expr, &CompileOptions::without_optimizations())
            .unwrap();
    let compiled = CompiledModel::from_expr(&expr).unwrap();
    assert!(compiled.cost().is_no_worse_than(&source.cost()));
    assert_eq!(count_unary_op(&compiled, UnaryOp::Exp), 0);
}

#[test]
fn search_revisits_new_equivalents() {
    let x = Expr::from(parameter!("x"));
    let expr = (x.clone() + 0.0) - x;
    let compiled = CompiledModel::from_expr(&expr).unwrap();
    assert!(matches!(
        compiled.graph().node(compiled.graph().root()),
        Some(ExprNode::RealConst(0.0))
    ));
    assert!(compiled.optimization_diagnostics().unwrap().rounds() > 1);
}

#[test]
fn node_budget_retains_a_valid_graph() {
    let x = Expr::from(parameter!("x"));
    let expr = x.clone().sin().powi(2) + x.cos().powi(2) + 7.0;
    let options = CompileOptions::default().with_optimization_budget(crate::OptimizationBudget {
        rounds: 8,
        nodes: 1,
        ..Default::default()
    });
    let compiled = CompiledModel::from_expr_with_options(&expr, &options).unwrap();
    assert_eq!(
        compiled.optimization_diagnostics().unwrap().stop_reason(),
        "node budget"
    );
    assert!(compiled.graph().node(compiled.graph().root()).is_some());
}

#[test]
fn memory_budget_retains_a_valid_graph() {
    let expr = Expr::from(parameter!("x")).sin() + 1.0;
    let options = CompileOptions::default().with_optimization_budget(crate::OptimizationBudget {
        memory_bytes: 1,
        ..Default::default()
    });
    let compiled = CompiledModel::from_expr_with_options(&expr, &options).unwrap();
    let diagnostics = compiled.optimization_diagnostics().unwrap();
    assert_eq!(diagnostics.stop_reason(), "memory budget");
    assert!(diagnostics.peak_memory_bytes() > 1);
    assert!(compiled.graph().node(compiled.graph().root()).is_some());
}

#[test]
fn search_time_budget_retains_a_valid_graph() {
    let expr = Expr::from(parameter!("x")).sin() + 1.0;
    let options = CompileOptions::default().with_optimization_budget(crate::OptimizationBudget {
        search_seconds: 0.0,
        ..Default::default()
    });
    let compiled = CompiledModel::from_expr_with_options(&expr, &options).unwrap();
    assert_eq!(
        compiled.optimization_diagnostics().unwrap().stop_reason(),
        "time budget"
    );
    assert!(compiled.graph().node(compiled.graph().root()).is_some());
}

#[test]
fn solver_budget_uses_deterministic_fallback() {
    let x = Expr::from(parameter!("x"));
    let expr = (x.clone() + 0.0) * (x + 1.0);
    let options = CompileOptions::default().with_optimization_budget(crate::OptimizationBudget {
        solver_seconds: 0.0,
        ..Default::default()
    });
    let first = CompiledModel::from_expr_with_options(&expr, &options).unwrap();
    let second = CompiledModel::from_expr_with_options(&expr, &options).unwrap();
    assert_eq!(first.graph().nodes(), second.graph().nodes());
    let diagnostics = first.optimization_diagnostics().unwrap();
    assert!(!diagnostics.execution_exact());
    assert_eq!(diagnostics.execution_fallback(), Some("solver limit"));
}

#[test]
fn exact_extraction_is_deterministic() {
    let x = Expr::from(parameter!("x"));
    let y = Expr::from(parameter!("y"));
    let expr = (x.clone() + y.clone()) * (y + x);
    let first = CompiledModel::from_expr(&expr).unwrap();
    let second = CompiledModel::from_expr(&expr).unwrap();
    assert!(first.optimization_diagnostics().unwrap().execution_exact());
    assert_eq!(first.graph().nodes(), second.graph().nodes());
}
