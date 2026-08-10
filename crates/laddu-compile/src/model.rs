use std::mem::size_of;

#[cfg(test)]
use crate::CompileError;
use crate::CompileResult;
#[cfg(test)]
use laddu_expr::parameters::ParamError;
use laddu_expr::{
    BinaryOp, Expr, ExprGraph, ExprId, ExprNode, UnaryOp, ValueKind,
    parameters::{ParamLayout, ParamRegistry},
};
use serde::{Deserialize, Serialize};

#[cfg(test)]
use crate::facts::NumberClass;
use crate::{
    cost::OptimizationCost,
    facts::{DependencyFacts, EvaluationClass, GraphFacts, NodeFacts},
    optimize::*,
};

/// Options controlling graph optimization and event-cache planning.
#[derive(Debug, Default)]
pub struct CompileOptions {
    pipeline: OptimizationPipeline,
    cache_policy: CachePolicy,
}

impl CompileOptions {
    /// Creates the default optimized compilation options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates options with an empty optimization pipeline.
    pub fn without_optimizations() -> Self {
        Self {
            pipeline: OptimizationPipeline::new(),
            cache_policy: CachePolicy::default(),
        }
    }

    /// Creates options using a custom optimization pipeline.
    pub fn with_pipeline(pipeline: OptimizationPipeline) -> Self {
        Self {
            pipeline,
            cache_policy: CachePolicy::default(),
        }
    }

    /// Returns the optimization pipeline.
    pub fn pipeline(&self) -> &OptimizationPipeline {
        &self.pipeline
    }

    /// Returns the optimization pipeline for mutation.
    pub fn pipeline_mut(&mut self) -> &mut OptimizationPipeline {
        &mut self.pipeline
    }

    /// Returns the event-cache policy.
    pub fn cache_policy(&self) -> CachePolicy {
        self.cache_policy
    }

    /// Sets the event-cache policy.
    pub fn set_cache_policy(&mut self, cache_policy: CachePolicy) {
        self.cache_policy = cache_policy;
    }

    /// Returns these options with a new event-cache policy.
    pub fn with_cache_policy(mut self, cache_policy: CachePolicy) -> Self {
        self.set_cache_policy(cache_policy);
        self
    }
}

/// Policy for caching parameter-independent event expressions.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum CachePolicy {
    /// Disable event caching.
    Off,
    /// Cache the event-dependent frontier that is independent of parameters.
    #[default]
    EventDependent,
}

impl CachePolicy {
    fn accepts(self, facts: NodeFacts) -> bool {
        match self {
            Self::Off => false,
            Self::EventDependent => {
                facts.dependency.depends_on_event
                    && !facts.dependency.depends_on_free_params
                    && !facts.dependency.depends_on_fixed_params
            }
        }
    }
}

/// Ordered set of graph nodes materialized into the per-event cache.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CachePlan {
    entries: Vec<CacheEntry>,
    materialization_nodes: Vec<ExprId>,
}

impl CachePlan {
    pub(crate) fn new(graph: &ExprGraph, facts: &GraphFacts, policy: CachePolicy) -> Self {
        if policy == CachePolicy::Off {
            return Self::default();
        }

        let cacheable = graph
            .nodes()
            .iter()
            .enumerate()
            .map(|(index, _)| {
                let id = ExprId::from_index(index);
                let facts = *facts.get(id).expect("facts are complete for graph");
                policy.accepts(facts)
            })
            .collect::<Vec<_>>();
        let mut frontier = vec![false; graph.nodes().len()];
        if cacheable[graph.root().index()] {
            frontier[graph.root().index()] = true;
        }
        for (index, node) in graph.nodes().iter().enumerate() {
            if cacheable[index] {
                continue;
            }
            for child in node.child_ids() {
                if cacheable[child.index()] {
                    frontier[child.index()] = true;
                }
            }
        }

        let entries = cacheable
            .into_iter()
            .zip(frontier)
            .enumerate()
            .filter(|&(_index, (cacheable, frontier))| cacheable && frontier)
            .map(|(index, (_cacheable, _frontier))| {
                let id = ExprId::from_index(index);
                let facts = *facts.get(id).expect("facts are complete for graph");
                CacheEntry {
                    node: id,
                    value_kind: facts.value_kind,
                    evaluation_class: facts.evaluation_class(),
                    dependency: facts.dependency,
                }
            })
            .collect::<Vec<_>>();
        let mut required = vec![false; graph.nodes().len()];
        for entry in &entries {
            mark_cache_requirement(graph, entry.node, &mut required);
        }
        let materialization_nodes = required
            .into_iter()
            .enumerate()
            .filter(|&(_index, required)| required)
            .map(|(index, _required)| ExprId::from_index(index))
            .collect();
        Self {
            entries,
            materialization_nodes,
        }
    }

    /// Returns cache entries in slot order.
    pub fn entries(&self) -> &[CacheEntry] {
        &self.entries
    }

    /// Returns the number of cache slots.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns whether the plan contains no cache slots.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the cache slot assigned to `node`.
    pub fn node_slot(&self, node: ExprId) -> Option<usize> {
        self.entries.iter().position(|entry| entry.node == node)
    }

    /// Graph nodes required to materialize every cache slot, in evaluation order.
    pub fn materialization_nodes(&self) -> &[ExprId] {
        &self.materialization_nodes
    }

    /// Packed payload bytes required per event by ordinary cache slots.
    /// Returns the packed number of bytes required per event.
    pub fn bytes_per_event(&self) -> usize {
        self.entries
            .iter()
            .map(|entry| entry.storage_kind().bytes_per_event())
            .sum()
    }
}

fn mark_cache_requirement(graph: &ExprGraph, id: ExprId, required: &mut [bool]) {
    if required[id.index()] {
        return;
    }
    required[id.index()] = true;
    if let Some(node) = graph.node(id) {
        for child in node.child_ids() {
            mark_cache_requirement(graph, child, required);
        }
    }
}

/// In-memory representation used for one cache entry.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CacheStorageKind {
    /// One real scalar.
    Real,
    /// One or more complex scalar elements.
    Complex {
        /// Number of complex elements.
        width: usize,
    },
}

impl CacheStorageKind {
    /// Returns the number of logical scalar elements.
    pub fn width(self) -> usize {
        match self {
            Self::Real => 1,
            Self::Complex { width } => width,
        }
    }

    /// Returns the packed number of bytes required per event.
    pub fn bytes_per_event(self) -> usize {
        match self {
            Self::Real => size_of::<f64>(),
            Self::Complex { width } => width * size_of::<num::complex::Complex64>(),
        }
    }
}

/// Description of one planned event-cache slot.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CacheEntry {
    node: ExprId,
    value_kind: ValueKind,
    evaluation_class: EvaluationClass,
    dependency: DependencyFacts,
}

impl CacheEntry {
    /// Returns the graph node materialized into this slot.
    pub fn node(&self) -> ExprId {
        self.node
    }

    /// Returns the node's value kind.
    pub fn value_kind(&self) -> ValueKind {
        self.value_kind
    }

    /// Returns the node's evaluation frequency.
    pub fn evaluation_class(&self) -> EvaluationClass {
        self.evaluation_class
    }

    /// Returns the node's dependency facts.
    pub fn dependency(&self) -> DependencyFacts {
        self.dependency
    }

    /// Returns the packed cache storage representation.
    pub fn storage_kind(&self) -> CacheStorageKind {
        match self.value_kind {
            ValueKind::Real => CacheStorageKind::Real,
            ValueKind::Complex => CacheStorageKind::Complex { width: 1 },
            ValueKind::Vector { len } => CacheStorageKind::Complex { width: len },
            ValueKind::Matrix { rows, cols } => CacheStorageKind::Complex { width: rows * cols },
        }
    }
}

/// Optimized expression graph and the analyses required for execution.
#[derive(Clone, Debug)]
pub struct CompiledModel {
    source_graph: ExprGraph,
    graph: ExprGraph,
    params: ParamLayout,
    facts: GraphFacts,
    cache_plan: CachePlan,
}

impl Serialize for CompiledModel {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.source_graph.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for CompiledModel {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Self::from_graph(ExprGraph::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}

impl CompiledModel {
    /// Projects the source graph to selected tags and recompiles it.
    ///
    /// # Errors
    ///
    /// Returns [`CompileError`](crate::CompileError) when the projected graph
    /// has conflicting parameter definitions or an optimization pass fails.
    pub fn project_tags<'a>(&self, tags: impl IntoIterator<Item = &'a str>) -> CompileResult<Self> {
        Self::from_graph(self.source_graph.project_tags(tags))
    }

    /// Compiles an expression with default options.
    ///
    /// # Errors
    ///
    /// Returns [`CompileError`](crate::CompileError) when parameter collection
    /// or an optimization pass fails.
    pub fn from_expr(expr: &Expr) -> CompileResult<Self> {
        Self::from_expr_with_options(expr, &CompileOptions::default())
    }

    /// Compiles an expression with explicit options.
    ///
    /// # Errors
    ///
    /// Returns [`CompileError`](crate::CompileError) when parameter collection
    /// or an optimization pass fails.
    pub fn from_expr_with_options(expr: &Expr, options: &CompileOptions) -> CompileResult<Self> {
        Self::from_graph_with_options(expr.to_graph(), options)
    }

    /// Compiles a serialized graph with default options.
    ///
    /// # Errors
    ///
    /// Returns [`CompileError`](crate::CompileError) when parameter collection
    /// or an optimization pass fails.
    pub fn from_graph(graph: ExprGraph) -> CompileResult<Self> {
        Self::from_graph_with_options(graph, &CompileOptions::default())
    }

    /// Compiles a serialized graph with explicit options.
    ///
    /// # Errors
    ///
    /// Returns [`CompileError`](crate::CompileError) when parameter collection
    /// or an optimization pass fails.
    pub fn from_graph_with_options(
        graph: ExprGraph,
        options: &CompileOptions,
    ) -> CompileResult<Self> {
        let params = collect_params(&graph)?;
        let source_graph = graph;
        let graph = options.pipeline.run(bake_fixed_parameters(&source_graph))?;
        let facts = GraphFacts::analyze(&graph);
        let cache_plan = CachePlan::new(&graph, &facts, options.cache_policy);
        Ok(Self {
            source_graph,
            graph,
            params,
            facts,
            cache_plan,
        })
    }

    /// Returns the optimized graph.
    pub fn graph(&self) -> &ExprGraph {
        &self.graph
    }

    /// Returns the proven polynomial degree in free parameters, or `None` for a
    /// graph containing a parameter-dependent non-polynomial operation.
    pub fn parameter_polynomial_degree(&self) -> Option<usize> {
        let mut degrees: Vec<Option<usize>> = Vec::with_capacity(self.graph.nodes().len());
        for node in self.graph.nodes() {
            let child = |id: ExprId| degrees.get(id.index()).copied().flatten();
            let degree = match node {
                ExprNode::RealConst(_)
                | ExprNode::ComplexConst(_)
                | ExprNode::EventScalar(_)
                | ExprNode::EventP4Component { .. } => Some(0),
                ExprNode::ScalarParam(_) => Some(1),
                ExprNode::Unary { op, input } => {
                    let input = child(*input)?;
                    match op {
                        UnaryOp::Neg | UnaryOp::Real | UnaryOp::Imag | UnaryOp::Conj => Some(input),
                        UnaryOp::NormSqr => input.checked_mul(2),
                        UnaryOp::PowI(power) if *power >= 0 => input.checked_mul(*power as usize),
                        UnaryOp::Sqrt
                        | UnaryOp::Exp
                        | UnaryOp::Sin
                        | UnaryOp::Cos
                        | UnaryOp::Log
                        | UnaryOp::PowI(_) => (input == 0).then_some(0),
                    }
                }
                ExprNode::Binary { op, lhs, rhs } => {
                    let lhs = child(*lhs)?;
                    let rhs = child(*rhs)?;
                    match op {
                        BinaryOp::Add | BinaryOp::Sub => Some(lhs.max(rhs)),
                        BinaryOp::Mul => lhs.checked_add(rhs),
                        BinaryOp::Div if rhs == 0 => Some(lhs),
                        BinaryOp::Atan2 if lhs == 0 && rhs == 0 => Some(0),
                        BinaryOp::Div | BinaryOp::Atan2 => None,
                    }
                }
                ExprNode::NaryAdd { terms } => terms
                    .iter()
                    .map(|id| child(*id))
                    .collect::<Option<Vec<_>>>()?
                    .into_iter()
                    .max(),
                ExprNode::NaryMul { factors } => factors
                    .iter()
                    .try_fold(0usize, |degree, id| degree.checked_add(child(*id)?)),
                ExprNode::Complex { re, im } => Some(child(*re)?.max(child(*im)?)),
                ExprNode::Vector { elements } | ExprNode::Matrix { elements, .. } => elements
                    .iter()
                    .map(|id| child(*id))
                    .collect::<Option<Vec<_>>>()?
                    .into_iter()
                    .max(),
                ExprNode::Component { input, .. } | ExprNode::MatrixElement { input, .. } => {
                    child(*input)
                }
                ExprNode::MatMul { lhs, rhs } | ExprNode::Dot { lhs, rhs } => {
                    child(*lhs)?.checked_add(child(*rhs)?)
                }
                ExprNode::MatVec { matrix, vector } => child(*matrix)?.checked_add(child(*vector)?),
                ExprNode::Solve { matrix, rhs } if child(*matrix)? == 0 => child(*rhs),
                ExprNode::Solve { .. } => None,
            };
            degrees.push(degree);
        }
        degrees.get(self.graph.root().index()).copied().flatten()
    }

    /// Creates an indented-tree display of the optimized graph.
    pub fn display_tree(&self) -> laddu_expr::ExprGraphTreeDisplay<'_> {
        self.graph.display_tree()
    }

    /// Creates a Graphviz DOT display of the optimized graph.
    pub fn display_dot(&self) -> laddu_expr::ExprGraphDotDisplay<'_> {
        self.graph.display_dot()
    }

    /// Fix a parameter by name and recompile the model.
    ///
    /// # Errors
    ///
    /// Returns [`CompileError`](crate::CompileError) when the parameter is
    /// unknown, the fixed value is out of bounds, or recompilation fails.
    pub fn fix_parameter(&self, name: &str, value: f64) -> CompileResult<Self> {
        self.fix_parameter_with_options(name, value, &CompileOptions::default())
    }

    /// Fix a parameter by name and recompile with explicit options.
    ///
    /// # Errors
    ///
    /// Returns [`CompileError`](crate::CompileError) when the parameter is
    /// unknown, the fixed value is out of bounds, or recompilation fails.
    pub fn fix_parameter_with_options(
        &self,
        name: &str,
        value: f64,
        options: &CompileOptions,
    ) -> CompileResult<Self> {
        Self::from_graph_with_options(self.source_graph.fix_parameter(name, value)?, options)
    }

    /// Free a parameter by name and recompile the model.
    ///
    /// # Errors
    ///
    /// Returns [`CompileError`](crate::CompileError) when the parameter is
    /// unknown or recompilation fails.
    pub fn free_parameter(&self, name: &str) -> CompileResult<Self> {
        self.free_parameter_with_options(name, &CompileOptions::default())
    }

    /// Free a parameter by name and recompile with explicit options.
    ///
    /// # Errors
    ///
    /// Returns [`CompileError`](crate::CompileError) when the parameter is
    /// unknown or recompilation fails.
    pub fn free_parameter_with_options(
        &self,
        name: &str,
        options: &CompileOptions,
    ) -> CompileResult<Self> {
        Self::from_graph_with_options(self.source_graph.free_parameter(name)?, options)
    }

    /// Returns the validated parameter layout.
    pub fn params(&self) -> &ParamLayout {
        &self.params
    }

    /// Returns static facts for the optimized graph.
    pub fn facts(&self) -> &GraphFacts {
        &self.facts
    }

    /// Returns the event-cache plan.
    pub fn cache_plan(&self) -> &CachePlan {
        &self.cache_plan
    }

    /// Computes the optimized graph's operation cost.
    pub fn cost(&self) -> OptimizationCost {
        OptimizationCost::analyze(&self.graph)
    }

    /// Returns static facts for one optimized graph node.
    pub fn node_facts(&self, id: ExprId) -> Option<&NodeFacts> {
        self.facts.get(id)
    }
}

fn bake_fixed_parameters(graph: &ExprGraph) -> ExprGraph {
    let nodes = graph
        .nodes()
        .iter()
        .map(|node| match node {
            ExprNode::ScalarParam(parameter) => match parameter.state() {
                laddu_expr::parameters::ParamState::Fixed(value) => ExprNode::RealConst(*value),
                laddu_expr::parameters::ParamState::Free => node.clone(),
            },
            _ => node.clone(),
        })
        .collect();
    let metadata = (0..graph.nodes().len())
        .map(|index| {
            graph
                .metadata(ExprId::from_index(index))
                .expect("graph metadata is complete")
                .clone()
        })
        .collect();
    ExprGraph::from_parts(graph.root(), nodes, metadata).expect("source graph is valid")
}

/// Collects and validates all scalar parameters referenced by `graph`.
///
/// # Errors
///
/// Returns [`CompileError`](crate::CompileError) when parameter definitions
/// conflict or contain invalid metadata or values.
pub fn collect_params(graph: &ExprGraph) -> CompileResult<ParamLayout> {
    let mut registry = ParamRegistry::new();
    for node in graph.nodes() {
        if let ExprNode::ScalarParam(spec) = node {
            registry.register(spec.clone())?;
        }
    }
    Ok(registry.layout()?)
}

#[cfg(test)]
mod tests {
    use laddu_expr::{
        BinaryOp, Expr, ExprId, ExprMetadata, UnaryOp, ValueKind, complex, dot, event_scalar,
        matmul, matrix, matvec, parameter, parameters::Parameter, polar_complex, vector,
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

    #[derive(Copy, Clone, Debug)]
    struct WrapRootInExp;

    impl OptimizationPass for WrapRootInExp {
        fn name(&self) -> &'static str {
            "wrap-root-in-exp"
        }

        fn run(&self, graph: ExprGraph) -> CompileResult<ExprGraph> {
            let mut nodes = graph.nodes().to_vec();
            let mut metadata = graph
                .nodes()
                .iter()
                .enumerate()
                .map(|(index, _)| {
                    graph
                        .metadata(ExprId::from_index(index))
                        .expect("graph metadata length is validated")
                        .clone()
                })
                .collect::<Vec<_>>();
            let root = ExprId::from_index(nodes.len());
            nodes.push(ExprNode::Unary {
                op: UnaryOp::Exp,
                input: graph.root(),
            });
            metadata.push(ExprMetadata::new(laddu_expr::ExprSourceKind::Unary));
            Ok(ExprGraph::from_parts(root, nodes, metadata)?)
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

    fn count_nary_add(compiled: &CompiledModel) -> usize {
        compiled
            .graph()
            .nodes()
            .iter()
            .filter(|node| matches!(node, ExprNode::NaryAdd { .. }))
            .count()
    }

    fn count_nary_mul(compiled: &CompiledModel) -> usize {
        compiled
            .graph()
            .nodes()
            .iter()
            .filter(|node| matches!(node, ExprNode::NaryMul { .. }))
            .count()
    }

    fn count_unary_op(compiled: &CompiledModel, op: UnaryOp) -> usize {
        compiled
            .graph()
            .nodes()
            .iter()
            .filter(|node| matches!(node, ExprNode::Unary { op: node_op, .. } if *node_op == op))
            .count()
    }

    fn has_real_const(compiled: &CompiledModel, expected: f64) -> bool {
        compiled.graph().nodes().iter().any(|node| {
            matches!(node, ExprNode::RealConst(value) if (*value - expected).abs() <= f64::EPSILON * expected.abs().max(1.0) * 16.0)
        })
    }

    fn compile_cost(expr: &Expr, pipeline: OptimizationPipeline) -> OptimizationCost {
        CompiledModel::from_expr_with_options(expr, &CompileOptions::with_pipeline(pipeline))
            .unwrap()
            .cost()
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
    fn optimization_cost_reports_weighted_operation_breakdown() {
        let x = Expr::from(parameter!("x"));
        let compiled = CompiledModel::from_expr_with_options(
            &(x.sin() + x.exp() * x.powi(2)),
            &CompileOptions::without_optimizations(),
        )
        .unwrap();
        let cost = compiled.cost();

        assert_eq!(cost.transcendental_ops(), 2);
        assert_eq!(cost.power_ops(), 1);
        assert_eq!(cost.scalar_adds(), 1);
        assert_eq!(cost.scalar_muls(), 1);
        assert_eq!(cost.weighted_ops(), 46);
    }

    #[test]
    fn optimization_cost_compares_pipeline_effectiveness() {
        let phi = Expr::from(parameter!("phi"));
        let euler = phi.cos() + Complex64::I * phi.sin();
        let without_exponential = compile_cost(
            &euler,
            OptimizationPipeline::new()
                .with_pass(RewritePass::simplify())
                .with_pass(CanonicalCsePass)
                .with_pass(RewritePass::normalize_add_mul())
                .with_pass(CanonicalCsePass)
                .with_max_iterations(4),
        );
        let with_exponential = compile_cost(
            &euler,
            OptimizationPipeline::new()
                .with_pass(RewritePass::simplify())
                .with_pass(CanonicalCsePass)
                .with_pass(RewritePass::normalize_add_mul())
                .with_pass(CanonicalCsePass)
                .with_pass(RewritePass::exponential())
                .with_pass(RewritePass::simplify())
                .with_max_iterations(4),
        );

        assert!(with_exponential.weighted_ops() < without_exponential.weighted_ops());
        assert_eq!(with_exponential.transcendental_ops(), 1);
    }

    #[test]
    fn cost_gate_rejects_more_expensive_candidate_pipeline() {
        let x = Expr::from(parameter!("x"));
        let baseline =
            CompiledModel::from_expr_with_options(&x, &CompileOptions::without_optimizations())
                .unwrap();
        let gated = CompiledModel::from_expr_with_options(
            &x,
            &CompileOptions::with_pipeline(OptimizationPipeline::new().with_pass(
                CostGatePass::new(OptimizationPipeline::new().with_pass(WrapRootInExp)),
            )),
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
    fn parameter_polynomial_degree_proves_quadratic_models_and_rejects_nonlinear_coefficients() {
        let linear = complex(parameter!("re"), parameter!("im")) * complex(event_scalar("x"), 1.0);
        let quadratic = CompiledModel::from_expr(&linear.norm_sqr()).unwrap();
        assert_eq!(quadratic.parameter_polynomial_degree(), Some(2));

        let nonlinear = CompiledModel::from_expr(&Expr::from(parameter!("phase")).sin()).unwrap();
        assert_eq!(nonlinear.parameter_polynomial_degree(), None);
    }

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

        assert_eq!(count_nary_add(&compiled), 1);
    }

    #[test]
    fn cse_canonicalizes_commutative_binary_operands() {
        let x = Expr::from(parameter!("x"));
        let y = Expr::from(parameter!("y"));
        let model = (x.clone() + y.clone()) * (y + x);
        let compiled = CompiledModel::from_expr(&model).unwrap();

        assert_eq!(count_nary_add(&compiled), 1);
        assert!(matches!(
            compiled.graph().node(compiled.graph().root()),
            Some(ExprNode::Unary {
                op: UnaryOp::PowI(2),
                input,
            }) if matches!(compiled.graph().node(*input), Some(ExprNode::NaryAdd { .. }))
        ));
    }

    #[test]
    fn cse_canonicalizes_associative_addition_trees() {
        let x = Expr::from(parameter!("x"));
        let y = Expr::from(parameter!("y"));
        let z = Expr::from(parameter!("z"));
        let lhs = (x.clone() + y.clone()) + z.clone();
        let rhs = x + (z + y);
        let compiled = CompiledModel::from_expr(&(lhs * rhs)).unwrap();

        assert!(matches!(
            compiled.graph().node(compiled.graph().root()),
            Some(ExprNode::Unary {
                op: UnaryOp::PowI(2),
                input,
            }) if matches!(compiled.graph().node(*input), Some(ExprNode::NaryAdd { .. }))
        ));
        assert_eq!(count_nary_add(&compiled), 1);
    }

    #[test]
    fn cse_canonicalizes_associative_multiplication_trees() {
        let x = Expr::from(parameter!("x"));
        let y = Expr::from(parameter!("y"));
        let z = Expr::from(parameter!("z"));
        let lhs = (x.clone() * y.clone()) * z.clone();
        let rhs = z * (y * x);
        let options =
            CompileOptions::with_pipeline(OptimizationPipeline::new().with_pass(CanonicalCsePass));
        let compiled = CompiledModel::from_expr_with_options(&(lhs + rhs), &options).unwrap();

        assert!(matches!(
            compiled.graph().node(compiled.graph().root()),
            Some(ExprNode::NaryAdd { terms }) if terms.len() == 2 && terms[0] == terms[1]
        ));
    }

    #[test]
    fn cse_ignores_metadata_when_merging_duplicate_subtrees() {
        let x = Expr::from(parameter!("x"));
        let y = Expr::from(parameter!("y"));
        let lhs = (x.clone() + y.clone()).named("lhs");
        let rhs = (x + y).tagged("rhs");
        let compiled = CompiledModel::from_expr(&(lhs * rhs)).unwrap();

        assert_eq!(count_nary_add(&compiled), 1);
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
            &(0.6690465435572891 * angle.cos()
                + Complex64::new(0.0, 0.6690465435572891) * angle.sin()),
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
        let compiled =
            CompiledModel::from_expr(&(Complex64::I * angle.sin() + angle.cos())).unwrap();

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
        let sqrt_square =
            CompiledModel::from_expr(&((1.0 - costheta.powi(2)).sqrt().powi(2))).unwrap();
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
            &(0.75 * (1.0 - phi.clone().cos()) * (1.0 + phi.clone().cos())
                - (0.5 * phi).sin().powi(4)),
        )
        .unwrap();

        assert_eq!(count_unary_op(&compiled, UnaryOp::Sin), 0);
        assert_eq!(count_unary_op(&compiled, UnaryOp::Cos), 1);
        assert!(has_real_const(&compiled, 0.5));
        assert!(has_real_const(&compiled, 1.0));
        assert!(has_real_const(&compiled, 2.0));
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

        let Some(ExprNode::NaryMul { factors }) = compiled.graph().node(compiled.graph().root())
        else {
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

        let Some(ExprNode::NaryMul { factors }) = compiled.graph().node(compiled.graph().root())
        else {
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
        let lhs =
            Expr::from(-1.0) * -3.0 * 5.0 * 7.0 * c.clone() * c.clone() * d.clone() * d.clone();
        let rhs = Expr::from(-1.0) * -3.0 * 5.0 * d.clone() * d.clone();
        let compiled = CompiledModel::from_expr(&(lhs - rhs)).unwrap();

        let Some(ExprNode::NaryMul { factors }) = compiled.graph().node(compiled.graph().root())
        else {
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
                compiled.graph().node(ExprId::from_index(0)),
                Some(ExprNode::ScalarParam(parameter)) if parameter.name() == "y"
            ));
        }
    }

    #[test]
    fn matrix_vector_identities_and_zeroes_simplify() {
        let x = event_scalar("x");
        let y = event_scalar("y");
        let identity = matrix([[1.0, 0.0], [0.0, 1.0]]);
        let zero_matrix = matrix([[0.0, 0.0], [0.0, 0.0]]);
        let vector = vector([x, y]);

        let identity_product = CompiledModel::from_expr(&matvec(identity, vector.clone())).unwrap();
        assert!(matches!(
            identity_product.graph().node(identity_product.graph().root()),
            Some(ExprNode::Vector { elements }) if elements.len() == 2
        ));

        let zero_product = CompiledModel::from_expr(&matvec(zero_matrix, vector)).unwrap();
        assert!(matches!(
            zero_product.graph().node(zero_product.graph().root()),
            Some(ExprNode::Vector { elements }) if elements.len() == 2
                && elements.iter().all(|id| matches!(zero_product.graph().node(*id), Some(ExprNode::RealConst(0.0))))
        ));
    }

    #[test]
    fn dot_and_matvec_lower_to_scalar_arithmetic_when_cheaper() {
        let x = event_scalar("x");
        let y = event_scalar("y");
        let dot_product =
            CompiledModel::from_expr(&dot(vector([x.clone(), y.clone()]), vector([2.0, 3.0])))
                .unwrap();
        assert_eq!(
            dot_product
                .graph()
                .nodes()
                .iter()
                .filter(|node| matches!(node, ExprNode::Dot { .. }))
                .count(),
            0
        );
        assert!(matches!(
            dot_product.graph().node(dot_product.graph().root()),
            Some(ExprNode::NaryAdd { terms }) if terms.len() == 2
        ));

        let matrix_product =
            CompiledModel::from_expr(&matvec(matrix([[1.0, 2.0], [3.0, 4.0]]), vector([x, y])))
                .unwrap();
        assert_eq!(
            matrix_product
                .graph()
                .nodes()
                .iter()
                .filter(|node| matches!(node, ExprNode::MatVec { .. }))
                .count(),
            0
        );
        assert!(matches!(
            matrix_product.graph().node(matrix_product.graph().root()),
            Some(ExprNode::Vector { elements }) if elements.len() == 2
        ));
    }

    #[test]
    fn selected_aggregate_outputs_only_lower_required_contractions() {
        const N: usize = 8;
        let matrix_values = matrix::<N, N, Expr>(std::array::from_fn(|row| {
            std::array::from_fn(|col| event_scalar(format!("m{row}_{col}")))
        }));
        let vector_values = vector(std::array::from_fn::<Expr, N, _>(|index| {
            event_scalar(format!("v{index}"))
        }));
        let selected_row =
            CompiledModel::from_expr(&matvec(matrix_values, vector_values).component(3)).unwrap();

        assert!(matches!(
            selected_row.graph().node(selected_row.graph().root()),
            Some(ExprNode::NaryAdd { terms }) if terms.len() == N
        ));
        assert!(
            !selected_row
                .graph()
                .nodes()
                .iter()
                .any(|node| matches!(node, ExprNode::MatVec { .. } | ExprNode::Component { .. }))
        );
        let selected_names = selected_row
            .graph()
            .nodes()
            .iter()
            .filter_map(|node| match node {
                ExprNode::EventScalar(name) => Some(name.as_ref()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(selected_names.len(), 2 * N);
        assert!(
            selected_names
                .iter()
                .all(|name| { name.starts_with("v") || name.starts_with("m3_") })
        );

        let lhs = matrix::<N, N, Expr>(std::array::from_fn(|row| {
            std::array::from_fn(|col| event_scalar(format!("a{row}_{col}")))
        }));
        let rhs = matrix::<N, N, Expr>(std::array::from_fn(|row| {
            std::array::from_fn(|col| event_scalar(format!("b{row}_{col}")))
        }));
        let selected_element =
            CompiledModel::from_expr(&matmul(lhs, rhs).matrix_element(2, 5)).unwrap();

        assert!(matches!(
            selected_element.graph().node(selected_element.graph().root()),
            Some(ExprNode::NaryAdd { terms }) if terms.len() == N
        ));
        assert!(
            !selected_element.graph().nodes().iter().any(|node| matches!(
                node,
                ExprNode::MatMul { .. } | ExprNode::MatrixElement { .. }
            ))
        );
        let selected_names = selected_element
            .graph()
            .nodes()
            .iter()
            .filter_map(|node| match node {
                ExprNode::EventScalar(name) => Some(name.as_ref()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(selected_names.len(), 2 * N);
        assert!(
            selected_names
                .iter()
                .all(|name| { name.starts_with("a2_") || name.ends_with("_5") })
        );
    }

    #[test]
    fn matrix_multiplication_identity_and_zero_simplify() {
        let x = event_scalar("x");
        let matrix_value = matrix([[x, 2.0.into()], [3.0.into(), 4.0.into()]]);
        let identity = matrix([[1.0, 0.0], [0.0, 1.0]]);
        let identity_product = CompiledModel::from_expr(&matmul(identity, matrix_value)).unwrap();
        assert!(matches!(
            identity_product
                .graph()
                .node(identity_product.graph().root()),
            Some(ExprNode::Matrix {
                rows: 2,
                cols: 2,
                ..
            })
        ));

        let zero_product = CompiledModel::from_expr(&matmul(
            matrix([[0.0, 0.0], [0.0, 0.0]]),
            matrix([[1.0, 2.0], [3.0, 4.0]]),
        ))
        .unwrap();
        assert!(matches!(
            zero_product.graph().node(zero_product.graph().root()),
            Some(ExprNode::Matrix { rows: 2, cols: 2, elements }) if elements
                .iter()
                .all(|id| matches!(zero_product.graph().node(*id), Some(ExprNode::RealConst(0.0))))
        ));
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
            .map(ExprId::from_index)
            .unwrap();
        let root_facts = compiled.node_facts(compiled.graph().root()).unwrap();

        assert_eq!(
            compiled.node_facts(event_id).unwrap().value_kind,
            ValueKind::Real
        );
        assert!(
            compiled
                .node_facts(event_id)
                .unwrap()
                .dependency
                .depends_on_event
        );
        assert!(!compiled.graph().nodes().iter().any(
            |node| matches!(node, ExprNode::ScalarParam(parameter) if parameter.name() == "scale")
        ));
        assert!(
            compiled
                .graph()
                .nodes()
                .iter()
                .any(|node| matches!(node, ExprNode::RealConst(value) if *value == 2.0))
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
