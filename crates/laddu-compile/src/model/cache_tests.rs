use std::mem::size_of;

use laddu_expr::{
    Expr, ExprNode, UnaryOp, complex, event_scalar, parameter, parameters::Parameter,
};
use num::complex::Complex64;

use super::*;

#[test]
fn cache_policy_off_produces_no_cache_entries() {
    let model = event_scalar("x").sin() + event_scalar("x").sin();
    let compiled = CompiledModel::from_expr_with_options(
        &model,
        &CompileOptions::default().with_cache_policy(CachePolicy::Off),
    )
    .unwrap();

    assert!(compiled.cache_plan().is_empty());
}

#[test]
fn event_dependent_cache_policy_selects_parameter_boundary() {
    let model = parameter!("scale") * event_scalar("x").real().sin();
    let compiled = CompiledModel::from_expr(&model).unwrap();

    assert_eq!(compiled.cache_plan().len(), 1);
    assert!(compiled.cache_plan().entries().iter().all(|entry| {
        entry.evaluation_class() == EvaluationClass::PerEvent
            && entry.dependency().depends_on_event
            && !entry.dependency().depends_on_free_params
            && !entry.dependency().depends_on_fixed_params
    }));
    let entry = compiled.cache_plan().entries()[0];
    assert!(matches!(
        compiled.graph().node(entry.node()),
        Some(ExprNode::Unary {
            op: UnaryOp::Sin,
            input,
        }) if matches!(compiled.graph().node(*input),
            Some(ExprNode::EventScalar(name)) if name.as_ref() == "x")
    ));
    assert_eq!(entry.storage_kind(), CacheStorageKind::Real);
    assert_eq!(compiled.cache_plan().bytes_per_event(), size_of::<f64>());
    assert_eq!(compiled.cache_plan().materialization_nodes().len(), 2);
    assert_eq!(
        compiled.cache_plan().materialization_nodes().last(),
        Some(&entry.node())
    );
}

#[test]
fn cache_layout_distinguishes_real_and_complex_payloads() {
    let phase = event_scalar("x");
    let model = parameter!("scale") * complex(phase.clone().cos(), phase.sin());
    let compiled = CompiledModel::from_expr(&model).unwrap();

    assert_eq!(compiled.cache_plan().len(), 1);
    assert_eq!(
        compiled.cache_plan().entries()[0].storage_kind(),
        CacheStorageKind::Complex { width: 1 }
    );
    assert_eq!(
        compiled.cache_plan().bytes_per_event(),
        size_of::<Complex64>()
    );
    for window in compiled.cache_plan().materialization_nodes().windows(2) {
        assert!(window[0].index() < window[1].index());
    }
}

#[test]
fn event_dependent_cache_policy_excludes_parameter_dependent_values() {
    let x = event_scalar("x");
    let y = Expr::from(parameter!("y"));
    let fixed = Expr::from(Parameter::fixed("fixed", 2.0));
    let model = x.clone() * 2.0 + x.sin() + y.clone() * event_scalar("x") + fixed * x;
    let compiled = CompiledModel::from_expr(&model).unwrap();

    assert!(!compiled.cache_plan().is_empty());
    for entry in compiled.cache_plan().entries() {
        let facts = compiled.node_facts(entry.node()).unwrap();
        assert_eq!(facts.evaluation_class(), EvaluationClass::PerEvent);
        assert!(!facts.dependency.depends_on_free_params);
        assert!(!facts.dependency.depends_on_fixed_params);
    }
}
