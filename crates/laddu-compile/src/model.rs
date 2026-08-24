use std::{
    hash::{Hash, Hasher},
    mem::size_of,
};

use crate::CompileResult;
use laddu_expr::{
    BinaryOp, Expr, ExprGraph, ExprId, ExprNode, ParameterStructuralKey, UnaryOp, ValueKind,
    parameters::{ParamLayout, ParamRegistry},
    vector,
};
use serde::{Deserialize, Serialize};

#[cfg(test)]
use crate::facts::NumberClass;
use crate::{
    NormalizationDiagnostics, NormalizationPlan,
    cost::OptimizationCost,
    facts::{DependencyFacts, EvaluationClass, GraphFacts, NodeFacts},
    graph_utils::mark_reachable,
    optimize::*,
};

/// Options controlling graph optimization and event-cache planning.
#[derive(Debug)]
pub struct CompileOptions {
    pipeline: OptimizationPipeline,
    cache_policy: CachePolicy,
    normalization_analysis: NormalizationAnalysisMode,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum NormalizationAnalysisMode {
    BeforeExecutionLowering,
    ExecutionGraph,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            pipeline: OptimizationPipeline::normalization_target_lowering_passes(),
            cache_policy: CachePolicy::default(),
            normalization_analysis: NormalizationAnalysisMode::BeforeExecutionLowering,
        }
    }
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
            normalization_analysis: NormalizationAnalysisMode::ExecutionGraph,
        }
    }

    /// Creates options using a custom optimization pipeline.
    pub fn with_pipeline(pipeline: OptimizationPipeline) -> Self {
        Self {
            pipeline,
            cache_policy: CachePolicy::default(),
            normalization_analysis: NormalizationAnalysisMode::ExecutionGraph,
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

struct CompileRecipe<'a> {
    normalization: NormalizationRecipe,
    execution_pipeline: &'a OptimizationPipeline,
    cache_policy: CachePolicy,
}

enum NormalizationRecipe {
    AnalyzeBeforeExecution(OptimizationPipeline),
    AnalyzeExecutionGraph,
    Disabled,
}

impl<'a> CompileRecipe<'a> {
    fn from_options(options: &'a CompileOptions) -> Self {
        let normalization = match options.normalization_analysis {
            NormalizationAnalysisMode::BeforeExecutionLowering => {
                NormalizationRecipe::AnalyzeBeforeExecution(
                    OptimizationPipeline::normalization_analysis_passes(),
                )
            }
            NormalizationAnalysisMode::ExecutionGraph => NormalizationRecipe::AnalyzeExecutionGraph,
        };
        Self {
            normalization,
            execution_pipeline: &options.pipeline,
            cache_policy: options.cache_policy,
        }
    }

    fn normalization_submodel(execution_pipeline: &'a OptimizationPipeline) -> Self {
        Self {
            normalization: NormalizationRecipe::Disabled,
            execution_pipeline,
            cache_policy: CachePolicy::EventDependent,
        }
    }
}

struct Compiler<'a> {
    source_graph: ExprGraph,
    params: ParamLayout,
    recipe: CompileRecipe<'a>,
}

struct PreparedNormalization {
    execution_input: ExprGraph,
    plan: PreparedNormalizationPlan,
}

enum PreparedNormalizationPlan {
    Ready(NormalizationPlan),
    AnalyzeExecutionGraph,
    Disabled,
}

impl<'a> Compiler<'a> {
    fn new(source_graph: ExprGraph, recipe: CompileRecipe<'a>) -> CompileResult<Self> {
        let params = collect_params(&source_graph)?;
        Ok(Self {
            source_graph,
            params,
            recipe,
        })
    }

    fn compile(self) -> CompileResult<CompiledModel> {
        let Self {
            source_graph,
            params,
            recipe,
        } = self;
        let parameter_baked = Self::bake_parameters(&source_graph);
        let prepared = Self::prepare_normalization(parameter_baked, recipe.normalization)?;
        let execution_graph =
            Self::lower_execution(prepared.execution_input, recipe.execution_pipeline)?;
        let facts = GraphFacts::analyze(&execution_graph);
        let cache_plan = CachePlan::new(&execution_graph, &facts, recipe.cache_policy);
        let normalization_plan = match prepared.plan {
            PreparedNormalizationPlan::Ready(plan) => plan,
            PreparedNormalizationPlan::AnalyzeExecutionGraph => {
                NormalizationPlan::analyze(&execution_graph, &facts)
            }
            PreparedNormalizationPlan::Disabled => {
                NormalizationPlan::analyze_disabled(&execution_graph)
            }
        };
        Ok(CompiledModel {
            source_graph,
            graph: execution_graph,
            params,
            facts,
            cache_plan,
            normalization_plan,
        })
    }

    fn bake_parameters(source: &ExprGraph) -> ExprGraph {
        bake_fixed_parameters(source)
    }

    fn prepare_normalization(
        parameter_baked: ExprGraph,
        recipe: NormalizationRecipe,
    ) -> CompileResult<PreparedNormalization> {
        match recipe {
            NormalizationRecipe::AnalyzeBeforeExecution(pipeline) => {
                let normalization_input = pipeline.run(parameter_baked)?;
                let facts = GraphFacts::analyze(&normalization_input);
                let plan = NormalizationPlan::analyze(&normalization_input, &facts);
                Ok(PreparedNormalization {
                    execution_input: normalization_input,
                    plan: PreparedNormalizationPlan::Ready(plan),
                })
            }
            NormalizationRecipe::AnalyzeExecutionGraph => Ok(PreparedNormalization {
                execution_input: parameter_baked,
                plan: PreparedNormalizationPlan::AnalyzeExecutionGraph,
            }),
            NormalizationRecipe::Disabled => Ok(PreparedNormalization {
                execution_input: parameter_baked,
                plan: PreparedNormalizationPlan::Disabled,
            }),
        }
    }

    fn lower_execution(
        execution_input: ExprGraph,
        pipeline: &OptimizationPipeline,
    ) -> CompileResult<ExprGraph> {
        pipeline.run(execution_input)
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
            for child in node.children() {
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
        mark_reachable(graph, entries.iter().map(|entry| entry.node), &mut required);
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

    /// Returns the packed logical-element layout shared by backend consumers.
    ///
    /// # Panics
    ///
    /// Panics if the total packed width exceeds the addressable `usize` range.
    pub fn layout(&self) -> CacheLayout {
        let mut offsets = Vec::with_capacity(self.entries.len());
        let mut width: usize = 0;
        for entry in &self.entries {
            offsets.push(width);
            width = width
                .checked_add(entry.storage_kind().width())
                .expect("cache layout width exceeds addressable memory");
        }
        CacheLayout { offsets, width }
    }
}

/// Packed logical-element offsets for ordinary event-cache entries.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CacheLayout {
    offsets: Vec<usize>,
    width: usize,
}

impl CacheLayout {
    /// Returns each cache slot's first logical element offset.
    pub fn offsets(&self) -> &[usize] {
        &self.offsets
    }

    /// Returns the total logical elements stored per event.
    pub fn width(&self) -> usize {
        self.width
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
    normalization_plan: NormalizationPlan,
}

/// A set of compiled expression outputs that shares compilation work for
/// repeated roots.
///
/// Query consumers commonly need several scalar expressions over the same
/// event rows.  This artifact keeps the output order supplied by the caller
/// while retaining one compiled model for structurally repeated expressions.
/// It is intentionally additive; a single [`CompiledModel`] remains the
/// execution seam for ordinary model evaluation.
#[derive(Clone, Debug)]
pub struct CompiledQuery {
    model: CompiledModel,
    outputs: Vec<ExprId>,
}

impl CompiledQuery {
    /// Compiles expression outputs with default options.
    ///
    /// # Errors
    ///
    /// Returns [`CompileError`] when no expressions are supplied, parameter
    /// definitions conflict, or graph optimization fails.
    pub fn from_exprs<I>(exprs: I) -> CompileResult<Self>
    where
        I: IntoIterator<Item = Expr>,
    {
        Self::from_exprs_with_options(exprs, &CompileOptions::default())
    }

    /// Compiles expression outputs with explicit options.
    ///
    /// # Errors
    ///
    /// Returns [`CompileError`] when no expressions are supplied, parameter
    /// definitions conflict, or graph optimization fails.
    pub fn from_exprs_with_options<I>(exprs: I, options: &CompileOptions) -> CompileResult<Self>
    where
        I: IntoIterator<Item = Expr>,
    {
        let expressions = exprs.into_iter().collect::<Vec<_>>();
        if expressions.is_empty() {
            return Err(crate::CompileError::Unsupported(
                "multi-output query requires at least one expression",
            ));
        }
        let model = CompiledModel::from_expr_with_options(&vector(expressions), options)?;
        let outputs = match model.graph().node(model.graph().root()) {
            Some(ExprNode::Vector { elements }) => elements.clone(),
            _ => {
                return Err(crate::CompileError::InvalidExecutablePlan(
                    "compiled query root is not a vector".into(),
                ));
            }
        };
        Ok(Self { model, outputs })
    }

    /// Returns the one compiled graph containing every output root.
    pub fn model(&self) -> &CompiledModel {
        &self.model
    }

    /// Returns output node identifiers in the caller's requested order.
    pub fn outputs(&self) -> &[ExprId] {
        &self.outputs
    }
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
    /// Returns a process-local structural digest for execution-scoped caches.
    ///
    /// The digest is bit-exact and excludes expression metadata, but its hash
    /// algorithm and values are not a stable persisted format.
    ///
    /// This is a workspace-internal backend contract used by `laddu-runtime`.
    /// It is public only because Rust crate boundaries do not provide
    /// workspace-scoped visibility; downstream users must not persist or rely
    /// on its value.
    #[doc(hidden)]
    pub fn optimized_digest(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.graph.root().hash(&mut hasher);
        self.graph.nodes().len().hash(&mut hasher);
        for node in self.graph.nodes() {
            node.structural_key().hash(&mut hasher);
        }
        for parameter in self.params.specs() {
            ParameterStructuralKey::from(parameter).hash(&mut hasher);
        }
        hasher.finish()
    }

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
        Compiler::new(graph, CompileRecipe::from_options(options))?.compile()
    }

    pub(crate) fn from_graph_without_normalization(graph: ExprGraph) -> CompileResult<Self> {
        let execution_pipeline = OptimizationPipeline::new().with_pass(CanonicalCsePass);
        Compiler::new(
            graph,
            CompileRecipe::normalization_submodel(&execution_pipeline),
        )?
        .compile()
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

    /// Returns stable compiler-native normalization analysis diagnostics.
    pub fn normalization_diagnostics(&self) -> &NormalizationDiagnostics {
        self.normalization_plan.diagnostics()
    }

    /// Returns the exact normalization compiler artifact for runtime lowering.
    ///
    /// This is a workspace-internal backend contract used by `laddu-runtime`.
    /// It is public only because Rust crate boundaries do not provide
    /// workspace-scoped visibility and may change with the compiler/runtime
    /// implementation without becoming a supported user-facing API.
    #[doc(hidden)]
    pub fn normalization_plan(&self) -> &NormalizationPlan {
        &self.normalization_plan
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
#[path = "model/cache_tests.rs"]
mod cache_tests;

#[cfg(test)]
#[path = "model/tests.rs"]
mod tests;
