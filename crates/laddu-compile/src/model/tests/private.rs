use super::*;

#[test]
fn compile_option_constructors_select_explicit_normalization_boundaries() {
    assert_eq!(
        CompileOptions::default().normalization_analysis,
        NormalizationAnalysisMode::BeforeExecutionLowering
    );
    assert_eq!(
        CompileOptions::without_optimizations().normalization_analysis,
        NormalizationAnalysisMode::ExecutionGraph
    );
    assert_eq!(
        CompileOptions::with_pipeline(OptimizationPipeline::new()).normalization_analysis,
        NormalizationAnalysisMode::ExecutionGraph
    );
}

#[test]
fn compiler_phase_methods_keep_normalization_and_execution_inputs_distinct() {
    let source = (Expr::from(Parameter::fixed("scale", 2.0)) * event_scalar("x")).to_graph();
    let parameter_baked = Compiler::bake_parameters(&source);
    assert!(
        !parameter_baked
            .nodes()
            .iter()
            .any(|node| matches!(node, ExprNode::ScalarParam(_)))
    );

    let prepared = Compiler::prepare_normalization(
        parameter_baked,
        NormalizationRecipe::AnalyzeBeforeExecution(OptimizationPipeline::new()),
    )
    .unwrap();
    assert!(matches!(prepared.plan, PreparedNormalizationPlan::Ready(_)));
    assert!(!matches!(
        prepared
            .execution_input
            .node(prepared.execution_input.root()),
        Some(ExprNode::Unary {
            op: UnaryOp::Exp,
            ..
        })
    ));

    let execution_pipeline = OptimizationPipeline::new().with_pass(WrapRootInExp);
    let execution_graph =
        Compiler::lower_execution(prepared.execution_input, &execution_pipeline).unwrap();
    assert!(matches!(
        execution_graph.node(execution_graph.root()),
        Some(ExprNode::Unary {
            op: UnaryOp::Exp,
            ..
        })
    ));
}

#[test]
fn normalization_submodel_recipe_disables_analysis_after_execution_lowering() {
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

    let query =
        CompiledQuery::from_exprs([event_scalar("x") + 1.0, event_scalar("x") + 2.0]).unwrap();
    let graph = query.model().graph();
    assert!(query.outputs().len() == 2 && query.outputs()[0] != query.outputs()[1]);
    assert_eq!(
        graph
            .nodes()
            .iter()
            .filter(|node| matches!(node, ExprNode::EventScalar(name) if name.as_ref() == "x"))
            .count(),
        1
    );
}
