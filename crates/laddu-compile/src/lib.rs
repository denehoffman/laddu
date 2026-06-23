use laddu_expr::{
    Expr, ExprGraph, ExprGraphError, ExprId, ExprNode,
    parameters::{ParamError, ParamLayout, ParamRegistry},
};
use thiserror::Error;

pub mod facts;
pub mod optimize;

pub use facts::{DependencyFacts, EvaluationClass, GraphFacts, NodeFacts, NumberClass};
pub use optimize::{
    AlgebraicIdentityRule, CanonicalCsePass, ComplexFactRule, ConstantFoldScalarRule,
    OptimizationPass, OptimizationPipeline, Rewrite, RewriteContext, RewritePass, RewriteRule,
};

pub type CompileResult<T> = Result<T, CompileError>;

#[derive(Clone, Debug, Error, PartialEq)]
pub enum CompileError {
    #[error(transparent)]
    Params(#[from] ParamError),
    #[error(transparent)]
    Graph(#[from] ExprGraphError),
    #[error("unsupported compile feature: {0}")]
    Unsupported(&'static str),
}

#[derive(Debug)]
pub struct CompileOptions {
    pipeline: OptimizationPipeline,
}

impl CompileOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn without_optimizations() -> Self {
        Self {
            pipeline: OptimizationPipeline::new(),
        }
    }

    pub fn with_pipeline(pipeline: OptimizationPipeline) -> Self {
        Self { pipeline }
    }

    pub fn pipeline(&self) -> &OptimizationPipeline {
        &self.pipeline
    }

    pub fn pipeline_mut(&mut self) -> &mut OptimizationPipeline {
        &mut self.pipeline
    }
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            pipeline: OptimizationPipeline::default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct CompiledModel {
    graph: ExprGraph,
    params: ParamLayout,
    facts: GraphFacts,
}

impl CompiledModel {
    pub fn from_expr(expr: &Expr) -> CompileResult<Self> {
        Self::from_expr_with_options(expr, &CompileOptions::default())
    }

    pub fn from_expr_with_options(expr: &Expr, options: &CompileOptions) -> CompileResult<Self> {
        Self::from_graph_with_options(expr.to_graph(), options)
    }

    pub fn from_graph(graph: ExprGraph) -> CompileResult<Self> {
        Self::from_graph_with_options(graph, &CompileOptions::default())
    }

    pub fn from_graph_with_options(
        graph: ExprGraph,
        options: &CompileOptions,
    ) -> CompileResult<Self> {
        let params = collect_params(&graph)?;
        let graph = options.pipeline.run(graph)?;
        let facts = GraphFacts::analyze(&graph);
        Ok(Self {
            graph,
            params,
            facts,
        })
    }

    pub fn graph(&self) -> &ExprGraph {
        &self.graph
    }

    pub fn params(&self) -> &ParamLayout {
        &self.params
    }

    pub fn facts(&self) -> &GraphFacts {
        &self.facts
    }

    pub fn node_facts(&self, id: ExprId) -> Option<&NodeFacts> {
        self.facts.get(id)
    }
}

pub fn collect_params(graph: &ExprGraph) -> CompileResult<ParamLayout> {
    let mut registry = ParamRegistry::new();
    for node in graph.nodes() {
        match node {
            ExprNode::ScalarParam(spec) => {
                registry.register(spec.clone())?;
            }
            ExprNode::ComplexScalarParam { re, im } => {
                registry.register(re.clone())?;
                registry.register(im.clone())?;
            }
            ExprNode::PolarComplexScalarParam { mag, phase } => {
                registry.register(mag.clone())?;
                registry.register(phase.clone())?;
            }
            _ => {}
        }
    }
    Ok(registry.layout()?)
}

#[cfg(test)]
mod tests {
    use laddu_expr::{
        BinaryOp, Expr, ExprId, ExprMetadata, ValueKind, complex, event_scalar, matrix, parameter,
        parameters::Parameter, vector,
    };
    use num::complex::Complex64;

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

    fn count_binary_op(compiled: &CompiledModel, op: BinaryOp) -> usize {
        compiled
            .graph()
            .nodes()
            .iter()
            .filter(|node| matches!(node, ExprNode::Binary { op: node_op, .. } if *node_op == op))
            .count()
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
    fn cse_merges_duplicate_subtrees() {
        let x = Expr::from(parameter!("x"));
        let y = Expr::from(parameter!("y"));
        let sum = x + y;
        let model = sum.clone() * sum;
        let compiled = CompiledModel::from_expr(&model).unwrap();

        assert_eq!(count_binary_op(&compiled, BinaryOp::Add), 1);
    }

    #[test]
    fn cse_canonicalizes_commutative_binary_operands() {
        let x = Expr::from(parameter!("x"));
        let y = Expr::from(parameter!("y"));
        let model = (x.clone() + y.clone()) + (y + x);
        let compiled = CompiledModel::from_expr(&model).unwrap();

        assert_eq!(count_binary_op(&compiled, BinaryOp::Add), 2);
        assert!(matches!(
            compiled.graph().node(compiled.graph().root()),
            Some(ExprNode::Binary {
                op: BinaryOp::Add,
                lhs,
                rhs,
            }) if lhs == rhs
        ));
    }

    #[test]
    fn cse_ignores_metadata_when_merging_duplicate_subtrees() {
        let x = Expr::from(parameter!("x"));
        let y = Expr::from(parameter!("y"));
        let lhs = (x.clone() + y.clone()).named("lhs");
        let rhs = (x + y).tagged("rhs");
        let compiled = CompiledModel::from_expr(&(lhs + rhs)).unwrap();

        assert_eq!(count_binary_op(&compiled, BinaryOp::Add), 2);
    }

    #[test]
    fn custom_pipeline_can_include_canonical_cse() {
        let x = Expr::from(parameter!("x"));
        let y = Expr::from(parameter!("y"));
        let sum = x + y;
        let options =
            CompileOptions::with_pipeline(OptimizationPipeline::new().with_pass(CanonicalCsePass));
        let compiled =
            CompiledModel::from_expr_with_options(&(sum.clone() * sum), &options).unwrap();

        assert_eq!(count_binary_op(&compiled, BinaryOp::Add), 1);
    }

    #[test]
    fn custom_pipeline_can_omit_canonical_cse() {
        let x = Expr::from(parameter!("x"));
        let y = Expr::from(parameter!("y"));
        let sum = x + y;
        let options = CompileOptions::with_pipeline(
            OptimizationPipeline::new().with_pass(RewritePass::simplify()),
        );
        let compiled =
            CompiledModel::from_expr_with_options(&(sum.clone() * sum), &options).unwrap();

        assert_eq!(count_binary_op(&compiled, BinaryOp::Add), 2);
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
            Some(ExprNode::Unary {
                op: laddu_expr::UnaryOp::Real,
                ..
            })
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
    fn vector_and_matrix_extraction_alias_selected_scalar() {
        let x = Expr::from(parameter!("x"));
        let y = Expr::from(parameter!("y"));
        let component =
            CompiledModel::from_expr(&vector([x.clone(), y.clone()]).component(1)).unwrap();
        let element = CompiledModel::from_expr(
            &matrix([[x, y.clone()], [3.0.into(), 4.0.into()]]).matrix_element(0, 1),
        )
        .unwrap();

        for compiled in [component, element] {
            assert_eq!(compiled.graph().nodes().len(), 1);
            assert!(matches!(
                compiled.graph().node(ExprId::from_index(0).unwrap()),
                Some(ExprNode::ScalarParam(parameter)) if parameter.name() == "y"
            ));
        }
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
    fn facts_track_number_class_and_dependencies() {
        let model =
            event_scalar("mass") * Expr::from(Parameter::fixed("scale", 2.0)) + parameter!("x");
        let options = CompileOptions::without_optimizations();
        let compiled = CompiledModel::from_expr_with_options(&model, &options).unwrap();

        let event_id = compiled
            .graph()
            .nodes()
            .iter()
            .position(|node| matches!(node, ExprNode::EventScalar(name) if name.as_ref() == "mass"))
            .and_then(ExprId::from_index)
            .unwrap();
        let fixed_id = compiled
            .graph()
            .nodes()
            .iter()
            .position(|node| matches!(node, ExprNode::ScalarParam(parameter) if parameter.name() == "scale"))
            .and_then(ExprId::from_index)
            .unwrap();
        let root_facts = compiled.node_facts(compiled.graph().root()).unwrap();

        assert_eq!(
            compiled.node_facts(event_id).unwrap().value_kind,
            ValueKind::Complex
        );
        assert!(
            compiled
                .node_facts(event_id)
                .unwrap()
                .dependency
                .depends_on_event
        );
        assert!(
            compiled
                .node_facts(fixed_id)
                .unwrap()
                .dependency
                .depends_on_fixed_params
        );
        assert!(root_facts.dependency.depends_on_free_params);
        assert!(root_facts.dependency.depends_on_event);
        assert_eq!(
            root_facts.evaluation_class(),
            EvaluationClass::PerEvaluation
        );
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
}
