//! Public compiler-model integration contracts.

use laddu_compile::{CompileError, CompileOptions, CompiledModel};
use laddu_expr::{
    BinaryOp, Expr, ExprNode, complex, event_scalar, parameter,
    parameters::{ParamError, Parameter},
};
use num::complex::Complex64;

#[test]
fn optimized_digest_uses_semantic_identity_and_excludes_expression_metadata() {
    let plain = CompiledModel::from_expr(&Expr::from(2.0)).unwrap();
    let named = CompiledModel::from_expr(&Expr::from(2.0).named("display-only")).unwrap();
    assert_eq!(plain.optimized_digest(), named.optimized_digest());

    let base = CompiledModel::from_expr(&Expr::from(Parameter::free("p"))).unwrap();
    let scaled = CompiledModel::from_expr(&Expr::from(
        Parameter::free("p")
            .with_scale(0.5)
            .with_unit("GeV")
            .with_description("semantic parameter metadata"),
    ))
    .unwrap();
    assert_ne!(base.optimized_digest(), scaled.optimized_digest());
}

#[test]
fn collects_parameters_in_graph_construction_order() {
    let model = (Complex64::new(0.0, 1.0) * parameter!("y", initial: 1.0, bounds: (0.0, 2.0))
        + parameter!("x"))
    .norm_sqr();
    let compiled = CompiledModel::from_expr(&model).unwrap();

    assert_eq!(
        compiled
            .params()
            .specs()
            .iter()
            .map(|spec| spec.name())
            .collect::<Vec<_>>(),
        vec!["y", "x"]
    );
}

#[test]
fn merges_reused_compatible_parameters() {
    let x = parameter!("x", initial: 1.0);
    let model = x.clone() + x;
    let compiled = CompiledModel::from_expr(&model).unwrap();

    assert_eq!(compiled.params().len(), 1);
    assert_eq!(compiled.params().specs()[0].name(), "x");
}

#[test]
fn rejects_reused_incompatible_parameters() {
    let model = parameter!("x", initial: 1.0) + parameter!("x", initial: 2.0);

    assert!(matches!(
        CompiledModel::from_expr(&model),
        Err(CompileError::Params(ParamError::ParameterConflict { name, .. }))
            if name == "x"
    ));
}

#[test]
fn tag_projection_prunes_unselected_parameters() {
    let selected = Expr::from(parameter!("selected", initial: 1.0)).tagged("selected");
    let removed = Expr::from(parameter!("removed", initial: 2.0)).tagged("removed");
    let model = CompiledModel::from_expr(&(selected + removed + 3.0)).unwrap();
    let projected = model.project_tags(["selected"]).unwrap();

    assert!(projected.params().id("selected").is_some());
    assert!(projected.params().id("removed").is_none());
}

#[test]
fn collects_complex_scalar_parameter_components() {
    let model = complex(parameter!("a_re"), parameter!("a_im"));
    let compiled = CompiledModel::from_expr(&model).unwrap();

    assert_eq!(
        compiled
            .params()
            .specs()
            .iter()
            .map(|spec| spec.name())
            .collect::<Vec<_>>(),
        vec!["a_re", "a_im"]
    );
}

#[test]
fn parameter_polynomial_degree_proves_quadratic_models_and_rejects_nonlinear_coefficients() {
    let linear = complex(parameter!("re"), parameter!("im")) * complex(event_scalar("x"), 1.0);
    let quadratic = CompiledModel::from_expr(&linear.norm_sqr()).unwrap();
    assert_eq!(quadratic.parameter_polynomial_degree(), Some(2));

    let nonlinear = CompiledModel::from_expr(&Expr::from(parameter!("phase")).sin()).unwrap();
    assert_eq!(nonlinear.parameter_polynomial_degree(), None);
}

#[test]
fn no_optimization_preserves_raw_graph_shape() {
    let model = (parameter!("x") + 0.0) * 1.0;
    let options = CompileOptions::without_optimizations();
    let compiled = CompiledModel::from_expr_with_options(&model, &options).unwrap();

    assert!(compiled.graph().nodes().iter().any(|node| matches!(
        node,
        ExprNode::Binary {
            op: BinaryOp::Add,
            ..
        }
    )));
    assert!(compiled.graph().nodes().iter().any(|node| matches!(
        node,
        ExprNode::Binary {
            op: BinaryOp::Mul,
            ..
        }
    )));
}

#[test]
fn optimization_does_not_change_original_parameter_layout() {
    let model = parameter!("x") * 0.0 + parameter!("y");
    let compiled = CompiledModel::from_expr(&model).unwrap();

    assert_eq!(
        compiled
            .params()
            .specs()
            .iter()
            .map(|spec| spec.name())
            .collect::<Vec<_>>(),
        vec!["x", "y"]
    );
    assert!(matches!(
        compiled.graph().node(compiled.graph().root()),
        Some(ExprNode::ScalarParam(parameter)) if parameter.name() == "y"
    ));
}

#[test]
fn parameter_conflicts_are_detected_before_optimization() {
    let model = parameter!("x", initial: 1.0) * 0.0 + parameter!("x", initial: 2.0);

    assert!(matches!(
        CompiledModel::from_expr(&model),
        Err(CompileError::Params(ParamError::ParameterConflict { name, .. }))
            if name == "x"
    ));
}

#[test]
fn fixed_parameters_are_baked_and_folded() {
    let expression = (parameter!("scale") + 1.0) * event_scalar("x");
    let compiled = CompiledModel::from_expr(&expression)
        .unwrap()
        .fix_parameter("scale", 2.0)
        .unwrap();

    assert_eq!(compiled.params().n_free(), 0);
    assert!(!compiled.graph().nodes().iter().any(
        |node| matches!(node, ExprNode::ScalarParam(parameter) if parameter.name() == "scale")
    ));
    assert!(
        compiled
            .graph()
            .nodes()
            .iter()
            .any(|node| matches!(node, ExprNode::RealConst(value) if *value == 3.0))
    );
}

#[test]
fn freeing_a_compiled_parameter_recompiles_from_the_source_graph() {
    let expression = parameter!("scale") * event_scalar("x");
    let fixed = CompiledModel::from_expr(&expression)
        .unwrap()
        .fix_parameter("scale", 2.0)
        .unwrap();
    let freed = fixed.free_parameter("scale").unwrap();

    assert_eq!(freed.params().n_free(), 1);
    assert!(freed.graph().nodes().iter().any(
            |node| matches!(node, ExprNode::ScalarParam(parameter) if parameter.name() == "scale" && parameter.is_free())
        ));
}
