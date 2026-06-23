use std::{collections::HashMap, fmt};

use laddu_expr::{
    BinaryOp, ExprGraph, ExprId, ExprMetadata, ExprNode, ExprSourceKind, UnaryOp, ValueKind,
    parameters::{InitialSpec, ParamState, Parameter},
};

use crate::{
    CompileResult,
    facts::{NodeFacts, NumberClass},
};

const DEFAULT_MAX_ITERATIONS: usize = 16;

pub struct OptimizationPipeline {
    passes: Vec<Box<dyn OptimizationPass>>,
    max_iterations: usize,
}

impl OptimizationPipeline {
    pub fn new() -> Self {
        Self {
            passes: Vec::new(),
            max_iterations: 1,
        }
    }

    pub fn with_default_passes() -> Self {
        Self::new()
            .with_pass(RewritePass::simplify())
            .with_pass(CanonicalCsePass)
            .with_pass(RewritePass::factor_common_products())
            .with_pass(RewritePass::exponential())
            .with_pass(RewritePass::simplify())
            .with_max_iterations(DEFAULT_MAX_ITERATIONS)
    }

    pub fn add_pass(&mut self, pass: impl OptimizationPass + 'static) {
        self.passes.push(Box::new(pass));
    }

    pub fn with_pass(mut self, pass: impl OptimizationPass + 'static) -> Self {
        self.add_pass(pass);
        self
    }

    pub fn set_max_iterations(&mut self, max_iterations: usize) {
        self.max_iterations = max_iterations.max(1);
    }

    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.set_max_iterations(max_iterations);
        self
    }

    pub fn max_iterations(&self) -> usize {
        self.max_iterations
    }

    pub fn run(&self, mut graph: ExprGraph) -> CompileResult<ExprGraph> {
        for _ in 0..self.max_iterations {
            let previous = graph.clone();
            for pass in &self.passes {
                graph = pass.run(graph)?;
            }
            if graph_shape_eq(&previous, &graph) {
                break;
            }
        }
        Ok(graph)
    }

    pub fn is_empty(&self) -> bool {
        self.passes.is_empty()
    }
}

impl Default for OptimizationPipeline {
    fn default() -> Self {
        Self::with_default_passes()
    }
}

impl fmt::Debug for OptimizationPipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OptimizationPipeline")
            .field("passes", &self.passes.len())
            .field("max_iterations", &self.max_iterations)
            .finish()
    }
}

fn graph_shape_eq(lhs: &ExprGraph, rhs: &ExprGraph) -> bool {
    lhs.root() == rhs.root() && lhs.nodes() == rhs.nodes()
}

pub trait OptimizationPass: Send + Sync {
    fn name(&self) -> &'static str;
    fn run(&self, graph: ExprGraph) -> CompileResult<ExprGraph>;
}

#[derive(Copy, Clone, Debug, Default)]
pub struct CanonicalCsePass;

impl OptimizationPass for CanonicalCsePass {
    fn name(&self) -> &'static str {
        "canonical-cse"
    }

    fn run(&self, graph: ExprGraph) -> CompileResult<ExprGraph> {
        let mut old_to_new = Vec::with_capacity(graph.nodes().len());
        let mut nodes = Vec::with_capacity(graph.nodes().len());
        let mut metadata = Vec::with_capacity(graph.nodes().len());
        let mut keys = HashMap::with_capacity(graph.nodes().len());

        for (old_index, node) in graph.nodes().iter().enumerate() {
            let old_id = ExprId::from_index(old_index).expect("graph too large");
            let node_metadata = graph
                .metadata(old_id)
                .expect("graph metadata length is validated")
                .clone();
            let new_id = emit_canonical_node(
                remap_node(node, &old_to_new),
                node_metadata,
                &mut nodes,
                &mut metadata,
                &mut keys,
            );
            old_to_new.push(new_id);
        }

        let root = old_to_new[graph.root().index()];
        compact_graph(ExprGraph::from_parts(root, nodes, metadata)?)
    }
}

pub struct RewritePass {
    name: &'static str,
    rules: Vec<Box<dyn RewriteRule>>,
}

impl RewritePass {
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            rules: Vec::new(),
        }
    }

    pub fn simplify() -> Self {
        Self::new("simplify")
            .with_rule(ConstantFoldScalarRule)
            .with_rule(AlgebraicIdentityRule)
            .with_rule(ComplexFactRule)
    }

    pub fn factor_common_products() -> Self {
        Self::new("factor-common-products").with_rule(FactorCommonProductRule)
    }

    pub fn exponential() -> Self {
        Self::new("exponential").with_rule(ExponentialRule)
    }

    pub fn add_rule(&mut self, rule: impl RewriteRule + 'static) {
        self.rules.push(Box::new(rule));
    }

    pub fn with_rule(mut self, rule: impl RewriteRule + 'static) -> Self {
        self.add_rule(rule);
        self
    }
}

impl fmt::Debug for RewritePass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RewritePass")
            .field("name", &self.name)
            .field("rules", &self.rules.len())
            .finish()
    }
}

impl OptimizationPass for RewritePass {
    fn name(&self) -> &'static str {
        self.name
    }

    fn run(&self, graph: ExprGraph) -> CompileResult<ExprGraph> {
        let rewritten = RewriteBuilder::new(&self.rules).rewrite(graph)?;
        compact_graph(rewritten)
    }
}

pub trait RewriteRule: Send + Sync {
    fn name(&self) -> &'static str;

    fn rewrite(
        &self,
        node: &ExprNode,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite>;
}

#[derive(Clone, Debug, PartialEq)]
pub enum Rewrite {
    Keep,
    Alias(ExprId),
    Replace {
        node: ExprNode,
        metadata: ExprMetadata,
    },
    ReplaceMany {
        nodes: Vec<(ExprNode, ExprMetadata)>,
    },
}

pub struct RewriteContext<'a> {
    nodes: &'a [ExprNode],
    metadata: &'a [ExprMetadata],
    facts: &'a [NodeFacts],
}

impl<'a> RewriteContext<'a> {
    pub fn node(&self, id: ExprId) -> Option<&'a ExprNode> {
        self.nodes.get(id.index())
    }

    pub fn metadata(&self, id: ExprId) -> Option<&'a ExprMetadata> {
        self.metadata.get(id.index())
    }

    pub fn facts(&self, id: ExprId) -> Option<&'a NodeFacts> {
        self.facts.get(id.index())
    }

    pub fn next_id(&self) -> ExprId {
        ExprId::from_index(self.nodes.len()).expect("graph too large")
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct ConstantFoldScalarRule;

impl RewriteRule for ConstantFoldScalarRule {
    fn name(&self) -> &'static str {
        "constant-fold-scalar"
    }

    fn rewrite(
        &self,
        node: &ExprNode,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite> {
        match node {
            ExprNode::Unary { op, input } => {
                let Some(input) = context.node(*input).and_then(ExprNode::const_value) else {
                    return Ok(Rewrite::Keep);
                };
                Ok(Rewrite::Replace {
                    node: op.evaluate(input).into(),
                    metadata: metadata.clone(),
                })
            }
            ExprNode::Binary { op, lhs, rhs } => {
                let (Some(lhs), Some(rhs)) = (
                    context.node(*lhs).and_then(ExprNode::const_value),
                    context.node(*rhs).and_then(ExprNode::const_value),
                ) else {
                    return Ok(Rewrite::Keep);
                };
                Ok(Rewrite::Replace {
                    node: op.evaluate(lhs, rhs).into(),
                    metadata: metadata.clone(),
                })
            }
            _ => Ok(Rewrite::Keep),
        }
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct AlgebraicIdentityRule;

impl RewriteRule for AlgebraicIdentityRule {
    fn name(&self) -> &'static str {
        "algebraic-identity"
    }

    fn rewrite(
        &self,
        node: &ExprNode,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite> {
        match node {
            ExprNode::Binary {
                op: BinaryOp::Add,
                lhs,
                rhs,
            } if context.node(*lhs).is_some_and(ExprNode::is_zero) => {
                Ok(alias_or_preserve(*rhs, metadata, context))
            }
            ExprNode::Binary {
                op: BinaryOp::Add,
                lhs,
                rhs,
            } if context.node(*rhs).is_some_and(ExprNode::is_zero) => {
                Ok(alias_or_preserve(*lhs, metadata, context))
            }
            ExprNode::Binary {
                op: BinaryOp::Sub,
                lhs,
                rhs,
            } if lhs == rhs => Ok(Rewrite::Replace {
                node: ExprNode::RealConst(0.0),
                metadata: metadata.clone(),
            }),
            ExprNode::Binary {
                op: BinaryOp::Sub,
                lhs,
                rhs,
            } if context.node(*lhs).is_some_and(ExprNode::is_zero) => Ok(Rewrite::Replace {
                node: ExprNode::Unary {
                    op: UnaryOp::Neg,
                    input: *rhs,
                },
                metadata: metadata.clone(),
            }),
            ExprNode::Binary {
                op: BinaryOp::Sub,
                lhs,
                rhs,
            } if context.node(*rhs).is_some_and(ExprNode::is_zero) => {
                Ok(alias_or_preserve(*lhs, metadata, context))
            }
            ExprNode::Binary {
                op: BinaryOp::Mul,
                lhs,
                rhs,
            } if context.node(*lhs).is_some_and(ExprNode::is_one) => {
                Ok(alias_or_preserve(*rhs, metadata, context))
            }
            ExprNode::Binary {
                op: BinaryOp::Mul,
                lhs,
                rhs,
            } if context.node(*rhs).is_some_and(ExprNode::is_one) => {
                Ok(alias_or_preserve(*lhs, metadata, context))
            }
            ExprNode::Binary {
                op: BinaryOp::Mul,
                lhs,
                ..
            } if context.node(*lhs).is_some_and(ExprNode::is_zero) => {
                Ok(alias_or_preserve(*lhs, metadata, context))
            }
            ExprNode::Binary {
                op: BinaryOp::Mul,
                rhs,
                ..
            } if context.node(*rhs).is_some_and(ExprNode::is_zero) => {
                Ok(alias_or_preserve(*rhs, metadata, context))
            }
            ExprNode::Binary {
                op: BinaryOp::Div,
                lhs,
                rhs,
            } if lhs == rhs => Ok(Rewrite::Replace {
                node: ExprNode::RealConst(1.0),
                metadata: metadata.clone(),
            }),
            ExprNode::Binary {
                op: BinaryOp::Div,
                lhs,
                rhs,
            } if context.node(*rhs).is_some_and(ExprNode::is_one) => {
                Ok(alias_or_preserve(*lhs, metadata, context))
            }
            ExprNode::Unary {
                op: UnaryOp::Neg,
                input,
            } => {
                let Some(ExprNode::Unary {
                    op: UnaryOp::Neg,
                    input,
                }) = context.node(*input)
                else {
                    return Ok(Rewrite::Keep);
                };
                Ok(alias_or_preserve(*input, metadata, context))
            }
            ExprNode::Unary {
                op: UnaryOp::Conj,
                input,
            } => match context.node(*input) {
                Some(ExprNode::Unary {
                    op: UnaryOp::Conj,
                    input,
                }) => Ok(alias_or_preserve(*input, metadata, context)),
                Some(ExprNode::Complex { re, im }) => {
                    let neg_id = context.next_id();
                    Ok(Rewrite::ReplaceMany {
                        nodes: vec![
                            (
                                ExprNode::Unary {
                                    op: UnaryOp::Neg,
                                    input: *im,
                                },
                                ExprMetadata::new(ExprSourceKind::Unary),
                            ),
                            (
                                ExprNode::Complex {
                                    re: *re,
                                    im: neg_id,
                                },
                                metadata.clone(),
                            ),
                        ],
                    })
                }
                Some(ExprNode::ComplexScalarParam { re, im }) => {
                    let re_id = context.next_id();
                    let im_id = ExprId::from_index(re_id.index() + 1).expect("graph too large");
                    let neg_id = ExprId::from_index(re_id.index() + 2).expect("graph too large");
                    Ok(Rewrite::ReplaceMany {
                        nodes: vec![
                            (
                                ExprNode::ScalarParam(re.clone()),
                                ExprMetadata::new(ExprSourceKind::Param),
                            ),
                            (
                                ExprNode::ScalarParam(im.clone()),
                                ExprMetadata::new(ExprSourceKind::Param),
                            ),
                            (
                                ExprNode::Unary {
                                    op: UnaryOp::Neg,
                                    input: im_id,
                                },
                                ExprMetadata::new(ExprSourceKind::Unary),
                            ),
                            (
                                ExprNode::Complex {
                                    re: re_id,
                                    im: neg_id,
                                },
                                metadata.clone(),
                            ),
                        ],
                    })
                }
                _ => Ok(Rewrite::Keep),
            },
            ExprNode::Unary {
                op: UnaryOp::Real,
                input,
            } => match context.node(*input) {
                Some(ExprNode::Unary {
                    op: UnaryOp::Real, ..
                })
                | Some(ExprNode::Unary {
                    op: UnaryOp::Imag, ..
                }) => Ok(alias_or_preserve(*input, metadata, context)),
                Some(ExprNode::Complex { re, .. }) => Ok(alias_or_preserve(*re, metadata, context)),
                Some(ExprNode::ComplexScalarParam { re, .. }) => Ok(Rewrite::Replace {
                    node: ExprNode::ScalarParam(re.clone()),
                    metadata: metadata.clone(),
                }),
                _ => Ok(Rewrite::Keep),
            },
            ExprNode::Unary {
                op: UnaryOp::Imag,
                input,
            } => match context.node(*input) {
                Some(ExprNode::Unary {
                    op: UnaryOp::Real, ..
                }) => Ok(Rewrite::Replace {
                    node: ExprNode::RealConst(0.0),
                    metadata: metadata.clone(),
                }),
                Some(ExprNode::Complex { im, .. }) => Ok(alias_or_preserve(*im, metadata, context)),
                Some(ExprNode::ComplexScalarParam { im, .. }) => Ok(Rewrite::Replace {
                    node: ExprNode::ScalarParam(im.clone()),
                    metadata: metadata.clone(),
                }),
                _ => Ok(Rewrite::Keep),
            },
            ExprNode::Component { input, index } => {
                let Some(ExprNode::Vector { elements }) = context.node(*input) else {
                    return Ok(Rewrite::Keep);
                };
                if !elements.iter().all(|id| is_scalar_value(context, *id)) {
                    return Ok(Rewrite::Keep);
                }
                let Some(element) = elements.get(*index).copied() else {
                    return Ok(Rewrite::Keep);
                };
                Ok(alias_or_preserve(element, metadata, context))
            }
            ExprNode::MatrixElement { input, row, col } => {
                let Some(ExprNode::Matrix { cols, elements, .. }) = context.node(*input) else {
                    return Ok(Rewrite::Keep);
                };
                if !elements.iter().all(|id| is_scalar_value(context, *id)) {
                    return Ok(Rewrite::Keep);
                }
                let Some(index) = row
                    .checked_mul(*cols)
                    .and_then(|base| base.checked_add(*col))
                else {
                    return Ok(Rewrite::Keep);
                };
                let Some(element) = elements.get(index).copied() else {
                    return Ok(Rewrite::Keep);
                };
                Ok(alias_or_preserve(element, metadata, context))
            }
            _ => Ok(Rewrite::Keep),
        }
    }
}

fn is_scalar_value(context: &RewriteContext<'_>, id: ExprId) -> bool {
    context
        .facts(id)
        .is_some_and(|facts| matches!(facts.value_kind, ValueKind::Real | ValueKind::Complex))
}

#[derive(Copy, Clone, Debug, Default)]
pub struct FactorCommonProductRule;

impl RewriteRule for FactorCommonProductRule {
    fn name(&self) -> &'static str {
        "factor-common-product"
    }

    fn rewrite(
        &self,
        node: &ExprNode,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite> {
        match node {
            ExprNode::Binary {
                op: BinaryOp::Add,
                lhs,
                rhs,
            } => factor_common_product(*lhs, *rhs, metadata, context),
            _ => Ok(Rewrite::Keep),
        }
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct ExponentialRule;

impl RewriteRule for ExponentialRule {
    fn name(&self) -> &'static str {
        "exponential"
    }

    fn rewrite(
        &self,
        node: &ExprNode,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite> {
        match node {
            ExprNode::Unary {
                op: UnaryOp::Exp,
                input,
            } if context.node(*input).is_some_and(ExprNode::is_zero) => Ok(Rewrite::Replace {
                node: ExprNode::RealConst(1.0),
                metadata: metadata.clone(),
            }),
            ExprNode::Binary {
                op: BinaryOp::Mul,
                lhs,
                rhs,
            } => {
                let (Some(lhs_input), Some(rhs_input)) =
                    (exp_input(context, *lhs), exp_input(context, *rhs))
                else {
                    return Ok(Rewrite::Keep);
                };
                let add_id = context.next_id();
                Ok(Rewrite::ReplaceMany {
                    nodes: vec![
                        (
                            ExprNode::Binary {
                                op: BinaryOp::Add,
                                lhs: lhs_input,
                                rhs: rhs_input,
                            },
                            ExprMetadata::new(ExprSourceKind::Binary),
                        ),
                        (
                            ExprNode::Unary {
                                op: UnaryOp::Exp,
                                input: add_id,
                            },
                            metadata.clone(),
                        ),
                    ],
                })
            }
            _ => Ok(Rewrite::Keep),
        }
    }
}

fn factor_common_product(
    lhs: ExprId,
    rhs: ExprId,
    metadata: &ExprMetadata,
    context: &RewriteContext<'_>,
) -> CompileResult<Rewrite> {
    let Some(nodes) = common_product_factor_nodes(lhs, rhs, metadata, context) else {
        return Ok(Rewrite::Keep);
    };
    Ok(Rewrite::ReplaceMany { nodes })
}

fn common_product_factor_nodes(
    lhs: ExprId,
    rhs: ExprId,
    metadata: &ExprMetadata,
    context: &RewriteContext<'_>,
) -> Option<Vec<(ExprNode, ExprMetadata)>> {
    let (lhs_factor, lhs_term) = binary_product_terms(lhs, context)?;
    let (rhs_factor, rhs_term) = binary_product_terms(rhs, context)?;
    let (factor, lhs_term, rhs_term) = if lhs_factor == rhs_factor {
        (lhs_factor, lhs_term, rhs_term)
    } else if lhs_factor == rhs_term {
        (lhs_factor, lhs_term, rhs_factor)
    } else if lhs_term == rhs_factor {
        (lhs_term, lhs_factor, rhs_term)
    } else if lhs_term == rhs_term {
        (lhs_term, lhs_factor, rhs_factor)
    } else {
        return None;
    };
    let sum_id = context.next_id();

    Some(vec![
        (
            ExprNode::Binary {
                op: BinaryOp::Add,
                lhs: lhs_term,
                rhs: rhs_term,
            },
            ExprMetadata::new(ExprSourceKind::Binary),
        ),
        (
            ExprNode::Binary {
                op: BinaryOp::Mul,
                lhs: factor,
                rhs: sum_id,
            },
            metadata.clone(),
        ),
    ])
}

fn binary_product_terms(id: ExprId, context: &RewriteContext<'_>) -> Option<(ExprId, ExprId)> {
    match context.node(id) {
        Some(ExprNode::Binary {
            op: BinaryOp::Mul,
            lhs,
            rhs,
        }) => Some((*lhs, *rhs)),
        _ => None,
    }
}

fn exp_input(context: &RewriteContext<'_>, id: ExprId) -> Option<ExprId> {
    match context.node(id) {
        Some(ExprNode::Unary {
            op: UnaryOp::Exp,
            input,
        }) => Some(*input),
        _ => None,
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct ComplexFactRule;

impl RewriteRule for ComplexFactRule {
    fn name(&self) -> &'static str {
        "complex-fact"
    }

    fn rewrite(
        &self,
        node: &ExprNode,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite> {
        match node {
            ExprNode::Unary {
                op: UnaryOp::Real,
                input,
            } if context
                .facts(*input)
                .is_some_and(|facts| facts.number_class == NumberClass::Real) =>
            {
                Ok(alias_or_preserve(*input, metadata, context))
            }
            ExprNode::Unary {
                op: UnaryOp::Imag,
                input,
            } if context
                .facts(*input)
                .is_some_and(|facts| facts.number_class == NumberClass::Real) =>
            {
                Ok(Rewrite::Replace {
                    node: ExprNode::RealConst(0.0),
                    metadata: metadata.clone(),
                })
            }
            ExprNode::Unary {
                op: UnaryOp::Conj,
                input,
            } if context
                .facts(*input)
                .is_some_and(|facts| facts.number_class == NumberClass::Real) =>
            {
                Ok(alias_or_preserve(*input, metadata, context))
            }
            _ => Ok(Rewrite::Keep),
        }
    }
}

struct RewriteBuilder<'a> {
    rules: &'a [Box<dyn RewriteRule>],
}

impl<'a> RewriteBuilder<'a> {
    fn new(rules: &'a [Box<dyn RewriteRule>]) -> Self {
        Self { rules }
    }

    fn rewrite(&self, graph: ExprGraph) -> CompileResult<ExprGraph> {
        let mut nodes = Vec::with_capacity(graph.nodes().len());
        let mut metadata = Vec::with_capacity(graph.nodes().len());
        let mut facts = Vec::with_capacity(graph.nodes().len());
        let mut old_to_new = Vec::with_capacity(graph.nodes().len());

        for (old_index, node) in graph.nodes().iter().enumerate() {
            let node = remap_node(node, &old_to_new);
            let old_id = ExprId::from_index(old_index).expect("graph too large");
            let metadata_for_node = graph
                .metadata(old_id)
                .expect("graph metadata length is validated")
                .clone();
            let context = RewriteContext {
                nodes: &nodes,
                metadata: &metadata,
                facts: &facts,
            };

            let mut rewrite = Rewrite::Keep;
            for rule in self.rules {
                rewrite = rule.rewrite(&node, &metadata_for_node, &context)?;
                if !matches!(rewrite, Rewrite::Keep) {
                    break;
                }
            }

            let new_id = match rewrite {
                Rewrite::Keep => push_node(
                    node,
                    metadata_for_node,
                    &mut nodes,
                    &mut metadata,
                    &mut facts,
                ),
                Rewrite::Alias(id) => id,
                Rewrite::Replace {
                    node,
                    metadata: replacement_metadata,
                } => push_node(
                    node,
                    replacement_metadata,
                    &mut nodes,
                    &mut metadata,
                    &mut facts,
                ),
                Rewrite::ReplaceMany {
                    nodes: replacement_nodes,
                } => {
                    let mut root = None;
                    for (node, replacement_metadata) in replacement_nodes {
                        root = Some(push_node(
                            node,
                            replacement_metadata,
                            &mut nodes,
                            &mut metadata,
                            &mut facts,
                        ));
                    }
                    root.expect("replacement fragment must contain at least one node")
                }
            };
            old_to_new.push(new_id);
        }

        let root = old_to_new[graph.root().index()];
        Ok(ExprGraph::from_parts(root, nodes, metadata)?)
    }
}

fn push_node(
    node: ExprNode,
    node_metadata: ExprMetadata,
    nodes: &mut Vec<ExprNode>,
    metadata: &mut Vec<ExprMetadata>,
    facts: &mut Vec<NodeFacts>,
) -> ExprId {
    let id = ExprId::from_index(nodes.len()).expect("graph too large");
    facts.push(NodeFacts::for_node(&node, facts));
    nodes.push(node);
    metadata.push(node_metadata);
    id
}

fn compact_graph(graph: ExprGraph) -> CompileResult<ExprGraph> {
    let mut old_to_new = vec![None; graph.nodes().len()];
    let mut nodes = Vec::new();
    let mut metadata = Vec::new();
    let root = compact_visit(
        graph.root(),
        &graph,
        &mut old_to_new,
        &mut nodes,
        &mut metadata,
    );
    Ok(ExprGraph::from_parts(root, nodes, metadata)?)
}

fn compact_visit(
    old_id: ExprId,
    graph: &ExprGraph,
    old_to_new: &mut [Option<ExprId>],
    nodes: &mut Vec<ExprNode>,
    metadata: &mut Vec<ExprMetadata>,
) -> ExprId {
    if let Some(new_id) = old_to_new[old_id.index()] {
        return new_id;
    }

    for child in child_ids(graph.node(old_id).expect("valid graph")) {
        compact_visit(child, graph, old_to_new, nodes, metadata);
    }

    let node = remap_compacted_node(graph.node(old_id).expect("valid graph"), old_to_new);
    let new_id = ExprId::from_index(nodes.len()).expect("graph too large");
    nodes.push(node);
    metadata.push(
        graph
            .metadata(old_id)
            .expect("graph metadata length is validated")
            .clone(),
    );
    old_to_new[old_id.index()] = Some(new_id);
    new_id
}

fn alias_or_preserve(
    alias: ExprId,
    _metadata: &ExprMetadata,
    context: &RewriteContext<'_>,
) -> Rewrite {
    let _ = context.node(alias).expect("valid alias");
    Rewrite::Alias(alias)
}

fn emit_canonical_node(
    node: ExprNode,
    node_metadata: ExprMetadata,
    nodes: &mut Vec<ExprNode>,
    metadata: &mut Vec<ExprMetadata>,
    keys: &mut HashMap<StructuralKey, ExprId>,
) -> ExprId {
    match node {
        ExprNode::Binary {
            op: op @ (BinaryOp::Add | BinaryOp::Mul),
            lhs,
            rhs,
        } => {
            let mut operands = Vec::new();
            collect_associative_operands(op, lhs, nodes, &mut operands);
            collect_associative_operands(op, rhs, nodes, &mut operands);
            operands.sort_by(|lhs, rhs| {
                operand_sort_key(op, *lhs, nodes).cmp(&operand_sort_key(op, *rhs, nodes))
            });

            let operand_len = operands.len();
            let mut operands = operands.into_iter().enumerate();
            let (_, mut root) = operands.next().expect("binary node has operands");
            for (index, operand) in operands {
                let emitted_metadata = if index + 1 == operand_len {
                    node_metadata.clone()
                } else {
                    ExprMetadata::new(ExprSourceKind::Binary)
                };
                root = intern_canonical_node(
                    ExprNode::Binary {
                        op,
                        lhs: root,
                        rhs: operand,
                    },
                    emitted_metadata,
                    nodes,
                    metadata,
                    keys,
                );
            }
            root
        }
        node => intern_canonical_node(node, node_metadata, nodes, metadata, keys),
    }
}

fn intern_canonical_node(
    node: ExprNode,
    node_metadata: ExprMetadata,
    nodes: &mut Vec<ExprNode>,
    metadata: &mut Vec<ExprMetadata>,
    keys: &mut HashMap<StructuralKey, ExprId>,
) -> ExprId {
    let canonical = canonicalize_node(node);
    let key = StructuralKey::from_node(&canonical);
    if let Some(id) = keys.get(&key).copied() {
        return id;
    }

    let id = ExprId::from_index(nodes.len()).expect("graph too large");
    keys.insert(key, id);
    nodes.push(canonical);
    metadata.push(node_metadata);
    id
}

fn collect_associative_operands(
    op: BinaryOp,
    id: ExprId,
    nodes: &[ExprNode],
    operands: &mut Vec<ExprId>,
) {
    if let Some(ExprNode::Binary {
        op: child_op,
        lhs,
        rhs,
    }) = nodes.get(id.index())
        && *child_op == op
    {
        collect_associative_operands(op, *lhs, nodes, operands);
        collect_associative_operands(op, *rhs, nodes, operands);
        return;
    }
    operands.push(id);
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct OperandSortKey {
    category: u8,
    structural: String,
    id: usize,
}

fn operand_sort_key(op: BinaryOp, id: ExprId, nodes: &[ExprNode]) -> OperandSortKey {
    let node = nodes
        .get(id.index())
        .expect("associative operands are emitted nodes");
    let category = match (op, node) {
        (
            BinaryOp::Mul,
            ExprNode::Unary {
                op: UnaryOp::Exp, ..
            },
        ) => 0,
        _ => 1,
    };
    OperandSortKey {
        category,
        structural: format!("{:?}", StructuralKey::from_node(node)),
        id: id.index(),
    }
}

fn canonicalize_node(node: ExprNode) -> ExprNode {
    match node {
        ExprNode::Binary {
            op: op @ (BinaryOp::Add | BinaryOp::Mul),
            lhs,
            rhs,
        } if rhs.index() < lhs.index() => ExprNode::Binary {
            op,
            lhs: rhs,
            rhs: lhs,
        },
        node => node,
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum StructuralKey {
    RealConst(u64),
    ComplexConst {
        re: u64,
        im: u64,
    },
    ScalarParam(ParameterKey),
    ComplexScalarParam {
        re: ParameterKey,
        im: ParameterKey,
    },
    PolarComplexScalarParam {
        mag: ParameterKey,
        phase: ParameterKey,
    },
    EventScalar(String),
    Unary {
        op: UnaryKey,
        input: usize,
    },
    Binary {
        op: BinaryOp,
        lhs: usize,
        rhs: usize,
    },
    Complex {
        re: usize,
        im: usize,
    },
    Vector {
        elements: Vec<usize>,
    },
    Matrix {
        rows: usize,
        cols: usize,
        elements: Vec<usize>,
    },
    Component {
        input: usize,
        index: usize,
    },
    MatrixElement {
        input: usize,
        row: usize,
        col: usize,
    },
    MatMul {
        lhs: usize,
        rhs: usize,
    },
    MatVec {
        matrix: usize,
        vector: usize,
    },
    Dot {
        lhs: usize,
        rhs: usize,
    },
    Solve {
        matrix: usize,
        rhs: usize,
    },
}

impl StructuralKey {
    fn from_node(node: &ExprNode) -> Self {
        match node {
            ExprNode::RealConst(value) => Self::RealConst(value.to_bits()),
            ExprNode::ComplexConst(value) => Self::ComplexConst {
                re: value.re.to_bits(),
                im: value.im.to_bits(),
            },
            ExprNode::ScalarParam(parameter) => Self::ScalarParam(ParameterKey::from(parameter)),
            ExprNode::ComplexScalarParam { re, im } => Self::ComplexScalarParam {
                re: ParameterKey::from(re),
                im: ParameterKey::from(im),
            },
            ExprNode::PolarComplexScalarParam { mag, phase } => Self::PolarComplexScalarParam {
                mag: ParameterKey::from(mag),
                phase: ParameterKey::from(phase),
            },
            ExprNode::EventScalar(name) => Self::EventScalar(name.to_string()),
            ExprNode::Unary { op, input } => Self::Unary {
                op: UnaryKey::from(*op),
                input: input.index(),
            },
            ExprNode::Binary { op, lhs, rhs } => Self::Binary {
                op: *op,
                lhs: lhs.index(),
                rhs: rhs.index(),
            },
            ExprNode::Complex { re, im } => Self::Complex {
                re: re.index(),
                im: im.index(),
            },
            ExprNode::Vector { elements } => Self::Vector {
                elements: elements.iter().map(|id| id.index()).collect(),
            },
            ExprNode::Matrix {
                rows,
                cols,
                elements,
            } => Self::Matrix {
                rows: *rows,
                cols: *cols,
                elements: elements.iter().map(|id| id.index()).collect(),
            },
            ExprNode::Component { input, index } => Self::Component {
                input: input.index(),
                index: *index,
            },
            ExprNode::MatrixElement { input, row, col } => Self::MatrixElement {
                input: input.index(),
                row: *row,
                col: *col,
            },
            ExprNode::MatMul { lhs, rhs } => Self::MatMul {
                lhs: lhs.index(),
                rhs: rhs.index(),
            },
            ExprNode::MatVec { matrix, vector } => Self::MatVec {
                matrix: matrix.index(),
                vector: vector.index(),
            },
            ExprNode::Dot { lhs, rhs } => Self::Dot {
                lhs: lhs.index(),
                rhs: rhs.index(),
            },
            ExprNode::Solve { matrix, rhs } => Self::Solve {
                matrix: matrix.index(),
                rhs: rhs.index(),
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ParameterKey {
    name: String,
    state: ParamStateKey,
    initial: InitialSpecKey,
    bounds: BoundsKey,
    unit: Option<String>,
    latex: Option<String>,
    description: Option<String>,
}

impl From<&Parameter> for ParameterKey {
    fn from(parameter: &Parameter) -> Self {
        Self {
            name: parameter.name().to_owned(),
            state: ParamStateKey::from(parameter.state()),
            initial: InitialSpecKey::from(parameter.initial_spec()),
            bounds: BoundsKey {
                min: parameter.bounds_spec().min.map(f64::to_bits),
                max: parameter.bounds_spec().max.map(f64::to_bits),
            },
            unit: parameter.unit_label().map(str::to_owned),
            latex: parameter.latex_label().map(str::to_owned),
            description: parameter.description_text().map(str::to_owned),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum ParamStateKey {
    Free,
    Fixed(u64),
}

impl From<&ParamState> for ParamStateKey {
    fn from(state: &ParamState) -> Self {
        match state {
            ParamState::Free => Self::Free,
            ParamState::Fixed(value) => Self::Fixed(value.to_bits()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum InitialSpecKey {
    Default,
    Value(u64),
    Uniform { min: u64, max: u64 },
}

impl From<&InitialSpec> for InitialSpecKey {
    fn from(initial: &InitialSpec) -> Self {
        match initial {
            InitialSpec::Default => Self::Default,
            InitialSpec::Value(value) => Self::Value(value.to_bits()),
            InitialSpec::Uniform { min, max } => Self::Uniform {
                min: min.to_bits(),
                max: max.to_bits(),
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct BoundsKey {
    min: Option<u64>,
    max: Option<u64>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
enum UnaryKey {
    Neg,
    Real,
    Imag,
    Conj,
    NormSqr,
    Sqrt,
    Exp,
    Sin,
    Cos,
    Log,
    PowI(i32),
}

impl From<UnaryOp> for UnaryKey {
    fn from(op: UnaryOp) -> Self {
        match op {
            UnaryOp::Neg => Self::Neg,
            UnaryOp::Real => Self::Real,
            UnaryOp::Imag => Self::Imag,
            UnaryOp::Conj => Self::Conj,
            UnaryOp::NormSqr => Self::NormSqr,
            UnaryOp::Sqrt => Self::Sqrt,
            UnaryOp::Exp => Self::Exp,
            UnaryOp::Sin => Self::Sin,
            UnaryOp::Cos => Self::Cos,
            UnaryOp::Log => Self::Log,
            UnaryOp::PowI(power) => Self::PowI(power),
        }
    }
}

fn remap_node(node: &ExprNode, old_to_new: &[ExprId]) -> ExprNode {
    match node {
        ExprNode::RealConst(_)
        | ExprNode::ComplexConst(_)
        | ExprNode::ScalarParam(_)
        | ExprNode::ComplexScalarParam { .. }
        | ExprNode::PolarComplexScalarParam { .. }
        | ExprNode::EventScalar(_) => node.clone(),
        ExprNode::Unary { op, input } => ExprNode::Unary {
            op: *op,
            input: old_to_new[input.index()],
        },
        ExprNode::Binary { op, lhs, rhs } => ExprNode::Binary {
            op: *op,
            lhs: old_to_new[lhs.index()],
            rhs: old_to_new[rhs.index()],
        },
        ExprNode::Complex { re, im } => ExprNode::Complex {
            re: old_to_new[re.index()],
            im: old_to_new[im.index()],
        },
        ExprNode::Vector { elements } => ExprNode::Vector {
            elements: elements.iter().map(|id| old_to_new[id.index()]).collect(),
        },
        ExprNode::Matrix {
            rows,
            cols,
            elements,
        } => ExprNode::Matrix {
            rows: *rows,
            cols: *cols,
            elements: elements.iter().map(|id| old_to_new[id.index()]).collect(),
        },
        ExprNode::Component { input, index } => ExprNode::Component {
            input: old_to_new[input.index()],
            index: *index,
        },
        ExprNode::MatrixElement { input, row, col } => ExprNode::MatrixElement {
            input: old_to_new[input.index()],
            row: *row,
            col: *col,
        },
        ExprNode::MatMul { lhs, rhs } => ExprNode::MatMul {
            lhs: old_to_new[lhs.index()],
            rhs: old_to_new[rhs.index()],
        },
        ExprNode::MatVec { matrix, vector } => ExprNode::MatVec {
            matrix: old_to_new[matrix.index()],
            vector: old_to_new[vector.index()],
        },
        ExprNode::Dot { lhs, rhs } => ExprNode::Dot {
            lhs: old_to_new[lhs.index()],
            rhs: old_to_new[rhs.index()],
        },
        ExprNode::Solve { matrix, rhs } => ExprNode::Solve {
            matrix: old_to_new[matrix.index()],
            rhs: old_to_new[rhs.index()],
        },
    }
}

fn remap_compacted_node(node: &ExprNode, old_to_new: &[Option<ExprId>]) -> ExprNode {
    match node {
        ExprNode::RealConst(_)
        | ExprNode::ComplexConst(_)
        | ExprNode::ScalarParam(_)
        | ExprNode::ComplexScalarParam { .. }
        | ExprNode::PolarComplexScalarParam { .. }
        | ExprNode::EventScalar(_) => node.clone(),
        ExprNode::Unary { op, input } => ExprNode::Unary {
            op: *op,
            input: old_to_new[input.index()].expect("child was compacted first"),
        },
        ExprNode::Binary { op, lhs, rhs } => ExprNode::Binary {
            op: *op,
            lhs: old_to_new[lhs.index()].expect("child was compacted first"),
            rhs: old_to_new[rhs.index()].expect("child was compacted first"),
        },
        ExprNode::Complex { re, im } => ExprNode::Complex {
            re: old_to_new[re.index()].expect("child was compacted first"),
            im: old_to_new[im.index()].expect("child was compacted first"),
        },
        ExprNode::Vector { elements } => ExprNode::Vector {
            elements: elements
                .iter()
                .map(|id| old_to_new[id.index()].expect("child was compacted first"))
                .collect(),
        },
        ExprNode::Matrix {
            rows,
            cols,
            elements,
        } => ExprNode::Matrix {
            rows: *rows,
            cols: *cols,
            elements: elements
                .iter()
                .map(|id| old_to_new[id.index()].expect("child was compacted first"))
                .collect(),
        },
        ExprNode::Component { input, index } => ExprNode::Component {
            input: old_to_new[input.index()].expect("child was compacted first"),
            index: *index,
        },
        ExprNode::MatrixElement { input, row, col } => ExprNode::MatrixElement {
            input: old_to_new[input.index()].expect("child was compacted first"),
            row: *row,
            col: *col,
        },
        ExprNode::MatMul { lhs, rhs } => ExprNode::MatMul {
            lhs: old_to_new[lhs.index()].expect("child was compacted first"),
            rhs: old_to_new[rhs.index()].expect("child was compacted first"),
        },
        ExprNode::MatVec { matrix, vector } => ExprNode::MatVec {
            matrix: old_to_new[matrix.index()].expect("child was compacted first"),
            vector: old_to_new[vector.index()].expect("child was compacted first"),
        },
        ExprNode::Dot { lhs, rhs } => ExprNode::Dot {
            lhs: old_to_new[lhs.index()].expect("child was compacted first"),
            rhs: old_to_new[rhs.index()].expect("child was compacted first"),
        },
        ExprNode::Solve { matrix, rhs } => ExprNode::Solve {
            matrix: old_to_new[matrix.index()].expect("child was compacted first"),
            rhs: old_to_new[rhs.index()].expect("child was compacted first"),
        },
    }
}

fn child_ids(node: &ExprNode) -> Vec<ExprId> {
    match node {
        ExprNode::RealConst(_)
        | ExprNode::ComplexConst(_)
        | ExprNode::ScalarParam(_)
        | ExprNode::ComplexScalarParam { .. }
        | ExprNode::PolarComplexScalarParam { .. }
        | ExprNode::EventScalar(_) => Vec::new(),
        ExprNode::Unary { input, .. }
        | ExprNode::Component { input, .. }
        | ExprNode::MatrixElement { input, .. } => vec![*input],
        ExprNode::Complex { re, im } => vec![*re, *im],
        ExprNode::Binary { lhs, rhs, .. }
        | ExprNode::MatMul { lhs, rhs }
        | ExprNode::Dot { lhs, rhs } => vec![*lhs, *rhs],
        ExprNode::MatVec { matrix, vector } => vec![*matrix, *vector],
        ExprNode::Solve { matrix, rhs } => vec![*matrix, *rhs],
        ExprNode::Vector { elements } | ExprNode::Matrix { elements, .. } => elements.clone(),
    }
}
