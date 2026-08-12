use std::fmt;

use laddu_expr::{
    ExprGraph, ExprGraphRebuilder, ExprId, ExprMetadata, ExprNode, ExprNodeSemantics,
};

use crate::{
    CompileResult, cost::OptimizationCost, facts::NodeFacts, graph_utils::compact_to_root,
};

use super::rules::NormalizeAddMulRule;
use super::{
    AlgebraicIdentityRule, CombineLikeTermsRule, ComplexFactRule, ConjugationRule,
    ConstantFoldScalarRule, ExponentialRule, FactorCommonProductRule, MatrixVectorRule,
    NormSqrExpansionRule, NormSqrReductionRule, OptimizationPass, Rewrite, RewriteContext,
    RewritePass, RewriteRule, TrigIdentityRule,
};

impl RewritePass {
    /// Creates an empty named rewrite pass.
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            rules: Vec::new(),
        }
    }

    /// Creates the standard scalar and linear-algebra simplification pass.
    pub fn simplify() -> Self {
        Self::new("simplify")
            .with_rule(ConstantFoldScalarRule)
            .with_rule(AlgebraicIdentityRule)
            .with_rule(TrigIdentityRule)
            .with_rule(NormSqrReductionRule)
            .with_rule(ComplexFactRule)
            .with_rule(MatrixVectorRule)
    }

    /// Creates a common-product factoring pass.
    pub fn factor_common_products() -> Self {
        Self::new("factor-common-products").with_rule(FactorCommonProductRule)
    }

    /// Creates a like-term combination pass.
    pub fn combine_like_terms() -> Self {
        Self::new("combine-like-terms").with_rule(CombineLikeTermsRule)
    }

    /// Creates an associative addition/multiplication normalization pass.
    pub fn normalize_add_mul() -> Self {
        Self::new("normalize-add-mul").with_rule(NormalizeAddMulRule)
    }

    /// Creates an exponential-identity pass.
    pub fn exponential() -> Self {
        Self::new("exponential").with_rule(ExponentialRule)
    }

    /// Creates a conjugation simplification pass.
    pub fn conjugation() -> Self {
        Self::new("conjugation").with_rule(ConjugationRule)
    }

    /// Creates a norm-squared expansion pass.
    pub fn norm_sqr_expansion() -> Self {
        Self::new("norm-sqr-expansion").with_rule(NormSqrExpansionRule)
    }

    /// Appends a rewrite rule.
    pub fn add_rule(&mut self, rule: impl RewriteRule + 'static) {
        self.rules.push(Box::new(rule));
    }

    /// Returns this pass with an appended rewrite rule.
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
        let root = rewritten.root();
        compact_to_root(&rewritten, root)
    }
}

impl<'a> RewriteContext<'a> {
    /// Returns a previously emitted node.
    pub fn node(&self, id: ExprId) -> Option<&'a ExprNode> {
        self.nodes.get(id.index())
    }

    /// Returns metadata for a previously emitted node.
    pub fn metadata(&self, id: ExprId) -> Option<&'a ExprMetadata> {
        self.metadata.get(id.index())
    }

    /// Returns inferred facts for a previously emitted node.
    pub fn facts(&self, id: ExprId) -> Option<&'a NodeFacts> {
        self.facts.get(id.index())
    }

    /// Returns the identifier the next emitted node will receive.
    pub fn next_id(&self) -> ExprId {
        ExprId::from_index(self.nodes.len())
    }

    pub(super) fn local_node_cost(
        &self,
        node: ExprNode,
        metadata: ExprMetadata,
    ) -> CompileResult<OptimizationCost> {
        let root = self.next_id();
        let mut nodes = self.nodes.to_vec();
        let mut metadata_nodes = self.metadata.to_vec();
        nodes.push(node);
        metadata_nodes.push(metadata);
        let graph = ExprGraph::from_parts(root, nodes, metadata_nodes)?;
        let graph = compact_to_root(&graph, root)?;
        Ok(OptimizationCost::analyze(&graph))
    }

    pub(super) fn local_fragment_cost(
        &self,
        fragment: &[(ExprNode, ExprMetadata)],
    ) -> CompileResult<OptimizationCost> {
        let root = ExprId::from_index(self.nodes.len() + fragment.len() - 1);
        let mut nodes = self.nodes.to_vec();
        let mut metadata = self.metadata.to_vec();
        nodes.extend(fragment.iter().map(|(node, _)| node.clone()));
        metadata.extend(fragment.iter().map(|(_, metadata)| metadata.clone()));
        let graph = ExprGraph::from_parts(root, nodes, metadata)?;
        let graph = compact_to_root(&graph, root)?;
        Ok(OptimizationCost::analyze(&graph))
    }

    pub(super) fn ids_are_all_constants(&self, ids: &[ExprId]) -> bool {
        ids.iter().all(|id| {
            self.node(*id)
                .is_some_and(|node| node.const_value().is_some())
        })
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
        let mut rebuild = ExprGraphRebuilder::with_capacity(graph.nodes().len());
        let mut facts = Vec::with_capacity(graph.nodes().len());
        let mut semantics = Vec::with_capacity(graph.nodes().len());

        for (old_index, node) in graph.nodes().iter().enumerate() {
            let old_id = ExprId::from_index(old_index);
            let node = node.map_children(|child| {
                rebuild
                    .remapped(&child)
                    .expect("validated graph children precede their parents")
            });
            let metadata_for_node = graph
                .metadata(old_id)
                .expect("graph metadata length is validated")
                .clone();
            let context = RewriteContext {
                nodes: rebuild.nodes(),
                metadata: rebuild.metadata(),
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
                    &mut rebuild,
                    &mut facts,
                    &mut semantics,
                ),
                Rewrite::Alias(id) => id,
                Rewrite::Replace {
                    node,
                    metadata: replacement_metadata,
                } => push_node(
                    node,
                    replacement_metadata,
                    &mut rebuild,
                    &mut facts,
                    &mut semantics,
                ),
                Rewrite::ReplaceMany {
                    nodes: replacement_nodes,
                } => {
                    let mut root = None;
                    for (node, replacement_metadata) in replacement_nodes {
                        root = Some(push_node(
                            node,
                            replacement_metadata,
                            &mut rebuild,
                            &mut facts,
                            &mut semantics,
                        ));
                    }
                    root.expect("replacement fragment must contain at least one node")
                }
            };
            rebuild.alias(old_id, new_id);
        }

        let root = rebuild
            .remapped(&graph.root())
            .expect("the rebuilt graph includes its root");
        Ok(rebuild.finish(root)?)
    }
}

fn push_node(
    node: ExprNode,
    node_metadata: ExprMetadata,
    rebuild: &mut ExprGraphRebuilder<ExprId>,
    facts: &mut Vec<NodeFacts>,
    semantics: &mut Vec<ExprNodeSemantics>,
) -> ExprId {
    let id = ExprId::from_index(rebuild.nodes().len());
    let node_semantics = node.semantics(semantics);
    facts.push(NodeFacts::for_node(&node, facts, node_semantics));
    semantics.push(node_semantics);
    let emitted = rebuild.emit_anonymous(node, node_metadata);
    debug_assert_eq!(id, emitted);
    emitted
}

pub(super) fn alias_or_preserve(
    alias: ExprId,
    _metadata: &ExprMetadata,
    context: &RewriteContext<'_>,
) -> Rewrite {
    let _ = context.node(alias).expect("valid alias");
    Rewrite::Alias(alias)
}
