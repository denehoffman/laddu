use std::{collections::HashMap, fmt};

use laddu_expr::{
    BinaryOp, ExprGraph, ExprId, ExprMetadata, ExprNode, ExprSourceKind, P4Component, UnaryOp,
    ValueKind,
    parameters::{InitialSpec, ParamState, Parameter},
};
use num::complex::Complex64;

use crate::{
    CompileResult,
    cost::OptimizationCost,
    facts::{NodeFacts, NumberClass},
};

const DEFAULT_MAX_ITERATIONS: usize = 16;

/// Ordered optimization passes repeatedly applied until convergence.
pub struct OptimizationPipeline {
    passes: Vec<Box<dyn OptimizationPass>>,
    max_iterations: usize,
}

impl OptimizationPipeline {
    /// Creates an empty single-iteration pipeline.
    pub fn new() -> Self {
        Self {
            passes: Vec::new(),
            max_iterations: 1,
        }
    }

    /// Creates the default cost-aware optimization pipeline.
    pub fn with_default_passes() -> Self {
        Self::new()
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
            .with_pass(CostGatePass::new(Self::norm_sqr_expansion_candidate()))
            .with_max_iterations(DEFAULT_MAX_ITERATIONS)
    }

    /// Creates the candidate pipeline used to test norm-squared expansion.
    pub fn norm_sqr_expansion_candidate() -> Self {
        Self::new()
            .with_pass(RewritePass::norm_sqr_expansion())
            .with_pass(CanonicalCsePass)
            .with_pass(RewritePass::conjugation())
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
            .with_max_iterations(8)
    }

    /// Appends an optimization pass.
    pub fn add_pass(&mut self, pass: impl OptimizationPass + 'static) {
        self.passes.push(Box::new(pass));
    }

    /// Returns this pipeline with an appended pass.
    pub fn with_pass(mut self, pass: impl OptimizationPass + 'static) -> Self {
        self.add_pass(pass);
        self
    }

    /// Sets the maximum fixed-point iterations, clamped to at least one.
    pub fn set_max_iterations(&mut self, max_iterations: usize) {
        self.max_iterations = max_iterations.max(1);
    }

    /// Returns this pipeline with a new iteration limit.
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.set_max_iterations(max_iterations);
        self
    }

    /// Returns the fixed-point iteration limit.
    pub fn max_iterations(&self) -> usize {
        self.max_iterations
    }

    /// Runs all passes until convergence or the iteration limit.
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

    /// Returns whether the pipeline contains no passes.
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

/// One whole-graph transformation in an [`OptimizationPipeline`].
pub trait OptimizationPass: Send + Sync {
    /// Returns a stable diagnostic name.
    fn name(&self) -> &'static str;
    /// Transforms `graph`.
    fn run(&self, graph: ExprGraph) -> CompileResult<ExprGraph>;
}

/// Runs a candidate pipeline only when it strictly improves static cost.
pub struct CostGatePass {
    name: &'static str,
    candidate: OptimizationPipeline,
}

impl CostGatePass {
    /// Creates a cost gate named `"cost-gate"`.
    pub fn new(candidate: OptimizationPipeline) -> Self {
        Self::named("cost-gate", candidate)
    }

    /// Creates a cost gate with a diagnostic name.
    pub fn named(name: &'static str, candidate: OptimizationPipeline) -> Self {
        Self { name, candidate }
    }
}

impl fmt::Debug for CostGatePass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CostGatePass")
            .field("name", &self.name)
            .field("candidate", &self.candidate)
            .finish()
    }
}

impl OptimizationPass for CostGatePass {
    fn name(&self) -> &'static str {
        self.name
    }

    fn run(&self, graph: ExprGraph) -> CompileResult<ExprGraph> {
        let original_cost = OptimizationCost::analyze(&graph);
        let candidate = self.candidate.run(graph.clone())?;
        let candidate_cost = OptimizationCost::analyze(&candidate);
        if candidate_cost.is_better_than(&original_cost) {
            Ok(candidate)
        } else {
            Ok(graph)
        }
    }
}

/// Canonical common-subexpression-elimination pass.
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
            let old_id = ExprId::from_index(old_index);
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

/// Whole-graph pass applying local rewrite rules in order.
pub struct RewritePass {
    name: &'static str,
    rules: Vec<Box<dyn RewriteRule>>,
}

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
        compact_graph(rewritten)
    }
}

/// Local transformation of one expression node.
pub trait RewriteRule: Send + Sync {
    /// Returns a stable diagnostic name.
    fn name(&self) -> &'static str;

    /// Chooses how to emit the current node.
    fn rewrite(
        &self,
        node: &ExprNode,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite>;
}

/// Emission selected by a [`RewriteRule`].
#[derive(Clone, Debug, PartialEq)]
pub enum Rewrite {
    /// Emit the current node unchanged.
    Keep,
    /// Replace the current node with an existing node.
    Alias(ExprId),
    /// Replace the current node with one new node.
    Replace {
        /// Replacement node.
        node: ExprNode,
        /// Replacement metadata.
        metadata: ExprMetadata,
    },
    /// Replace the current node with a topologically ordered fragment.
    ReplaceMany {
        /// Replacement nodes paired with metadata; the final node is the result.
        nodes: Vec<(ExprNode, ExprMetadata)>,
    },
}

/// Read-only view of nodes already emitted during rewriting.
pub struct RewriteContext<'a> {
    nodes: &'a [ExprNode],
    metadata: &'a [ExprMetadata],
    facts: &'a [NodeFacts],
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

    fn local_node_cost(
        &self,
        node: ExprNode,
        metadata: ExprMetadata,
    ) -> CompileResult<OptimizationCost> {
        let root = self.next_id();
        let mut nodes = self.nodes.to_vec();
        let mut metadata_nodes = self.metadata.to_vec();
        nodes.push(node);
        metadata_nodes.push(metadata);
        let graph = compact_graph(ExprGraph::from_parts(root, nodes, metadata_nodes)?)?;
        Ok(OptimizationCost::analyze(&graph))
    }

    fn local_fragment_cost(
        &self,
        fragment: &[(ExprNode, ExprMetadata)],
    ) -> CompileResult<OptimizationCost> {
        let root = ExprId::from_index(self.nodes.len() + fragment.len() - 1);
        let mut nodes = self.nodes.to_vec();
        let mut metadata = self.metadata.to_vec();
        nodes.extend(fragment.iter().map(|(node, _)| node.clone()));
        metadata.extend(fragment.iter().map(|(_, metadata)| metadata.clone()));
        let graph = compact_graph(ExprGraph::from_parts(root, nodes, metadata)?)?;
        Ok(OptimizationCost::analyze(&graph))
    }

    fn ids_are_all_constants(&self, ids: &[ExprId]) -> bool {
        ids.iter().all(|id| {
            self.node(*id)
                .is_some_and(|node| node.const_value().is_some())
        })
    }
}

/// Folds scalar operations whose operands are constants.
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
                    node: ExprNode::from_folded_const(op.evaluate(input)),
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
                    node: ExprNode::from_folded_const(op.evaluate(lhs, rhs)),
                    metadata: metadata.clone(),
                })
            }
            ExprNode::NaryAdd { terms } => self.fold_nary_add_constants(terms, metadata, context),
            ExprNode::NaryMul { factors } => {
                let Some(product) = factors
                    .iter()
                    .map(|id| context.node(*id).and_then(ExprNode::const_value))
                    .try_fold(Complex64::ONE, |product, value| {
                        value.map(|value| product * value)
                    })
                else {
                    return Ok(Rewrite::Keep);
                };
                Ok(Rewrite::Replace {
                    node: ExprNode::from_folded_const(product),
                    metadata: metadata.clone(),
                })
            }
            _ => Ok(Rewrite::Keep),
        }
    }
}

impl ConstantFoldScalarRule {
    fn fold_nary_add_constants(
        &self,
        terms: &[ExprId],
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite> {
        let mut constant_sum = Complex64::ZERO;
        let mut nonconstant_terms = Vec::new();
        let mut constant_count = 0;

        for term in terms {
            if let Some(value) = context.node(*term).and_then(ExprNode::const_value) {
                constant_sum += value;
                constant_count += 1;
            } else {
                nonconstant_terms.push(*term);
            }
        }

        if constant_count == 0 {
            return Ok(Rewrite::Keep);
        }

        if nonconstant_terms.is_empty() {
            return Ok(Rewrite::Replace {
                node: ExprNode::from_folded_const(constant_sum),
                metadata: metadata.clone(),
            });
        }

        if constant_sum != Complex64::ZERO {
            if constant_count == 1 {
                return Ok(Rewrite::Keep);
            }
            let mut builder = ReplacementFragment::new(context);
            let constant = builder.push(
                ExprNode::from_folded_const(constant_sum),
                ExprMetadata::new(ExprSourceKind::Const),
            );
            nonconstant_terms.push(constant);
            builder.push(
                ExprNode::NaryAdd {
                    terms: nonconstant_terms,
                },
                metadata.clone(),
            );
            return Ok(builder.into_rewrite());
        }

        Ok(match nonconstant_terms.as_slice() {
            [term] => alias_or_preserve(*term, metadata, context),
            _ => Rewrite::Replace {
                node: ExprNode::NaryAdd {
                    terms: nonconstant_terms,
                },
                metadata: metadata.clone(),
            },
        })
    }
}

/// Eliminates scalar arithmetic identities such as adding zero.
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
            ExprNode::NaryAdd { terms } => self.simplify_nary_add(terms, metadata, context),
            ExprNode::NaryMul { factors } => self.simplify_nary_mul(factors, metadata, context),
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
                op: UnaryOp::PowI(0),
                ..
            } => Ok(Rewrite::Replace {
                node: ExprNode::RealConst(1.0),
                metadata: metadata.clone(),
            }),
            ExprNode::Unary {
                op: UnaryOp::PowI(1),
                input,
            } => Ok(alias_or_preserve(*input, metadata, context)),
            ExprNode::Unary {
                op: UnaryOp::PowI(outer_power),
                input,
            } if *outer_power == 2 => {
                let Some(ExprNode::Unary {
                    op: UnaryOp::Sqrt,
                    input,
                }) = context.node(*input)
                else {
                    return Ok(Rewrite::Keep);
                };
                Ok(alias_or_preserve(*input, metadata, context))
            }
            ExprNode::Unary {
                op: UnaryOp::PowI(outer_power),
                input,
            } => {
                let Some(ExprNode::Unary {
                    op: UnaryOp::PowI(inner_power),
                    input,
                }) = context.node(*input)
                else {
                    return Ok(Rewrite::Keep);
                };
                let Some(power) = inner_power.checked_mul(*outer_power) else {
                    return Ok(Rewrite::Keep);
                };
                Ok(Rewrite::Replace {
                    node: ExprNode::Unary {
                        op: UnaryOp::PowI(power),
                        input: *input,
                    },
                    metadata: metadata.clone(),
                })
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

impl AlgebraicIdentityRule {
    fn simplify_nary_add(
        &self,
        terms: &[ExprId],
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite> {
        let nonzero_terms: Vec<_> = terms
            .iter()
            .copied()
            .filter(|id| !context.node(*id).is_some_and(ExprNode::is_zero))
            .collect();

        if nonzero_terms.len() == terms.len() {
            return Ok(Rewrite::Keep);
        }

        Ok(match nonzero_terms.as_slice() {
            [] => Rewrite::Replace {
                node: ExprNode::RealConst(0.0),
                metadata: metadata.clone(),
            },
            [term] => alias_or_preserve(*term, metadata, context),
            _ => Rewrite::Replace {
                node: ExprNode::NaryAdd {
                    terms: nonzero_terms,
                },
                metadata: metadata.clone(),
            },
        })
    }

    fn simplify_nary_mul(
        &self,
        factors: &[ExprId],
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite> {
        if let Some(zero) = factors
            .iter()
            .copied()
            .find(|id| context.node(*id).is_some_and(ExprNode::is_zero))
        {
            return Ok(alias_or_preserve(zero, metadata, context));
        }

        let nonone_factors: Vec<_> = factors
            .iter()
            .copied()
            .filter(|id| !context.node(*id).is_some_and(ExprNode::is_one))
            .collect();

        if nonone_factors.len() == factors.len() {
            return Ok(Rewrite::Keep);
        }

        Ok(match nonone_factors.as_slice() {
            [] => Rewrite::Replace {
                node: ExprNode::RealConst(1.0),
                metadata: metadata.clone(),
            },
            [factor] => alias_or_preserve(*factor, metadata, context),
            _ => Rewrite::Replace {
                node: ExprNode::NaryMul {
                    factors: nonone_factors,
                },
                metadata: metadata.clone(),
            },
        })
    }
}

/// Simplifies trigonometric identities.
#[derive(Copy, Clone, Debug, Default)]
pub struct TrigIdentityRule;

impl RewriteRule for TrigIdentityRule {
    fn name(&self) -> &'static str {
        "trig-identity"
    }

    fn rewrite(
        &self,
        node: &ExprNode,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite> {
        match node {
            ExprNode::Unary {
                op: op @ (UnaryOp::Sin | UnaryOp::Cos),
                input,
            } => Ok(self.normalize_parity(*op, *input, metadata, context)),
            ExprNode::Unary {
                op: UnaryOp::PowI(power),
                input,
            } if *power > 0 && *power % 2 == 0 => {
                Ok(self.simplify_even_trig_power(*power, *input, metadata, context))
            }
            ExprNode::Binary {
                op: BinaryOp::Add,
                lhs,
                rhs,
            } => Ok(self.simplify_sum(&[*lhs, *rhs], metadata, context)),
            ExprNode::Binary {
                op: BinaryOp::Sub,
                lhs,
                rhs,
            } if context.node(*lhs).is_some_and(ExprNode::is_one) => {
                Ok(self.simplify_one_minus(*rhs, metadata, context))
            }
            ExprNode::NaryAdd { terms } => Ok(self.simplify_sum(terms, metadata, context)),
            _ => Ok(Rewrite::Keep),
        }
    }
}

impl TrigIdentityRule {
    fn simplify_sum(
        &self,
        terms: &[ExprId],
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> Rewrite {
        if let Some(rewrite) = self.simplify_affine_cos_pair(terms, metadata, context) {
            return rewrite;
        }
        if let Some(rewrite) = self.simplify_half_angle_pair(terms, metadata, context) {
            return rewrite;
        }
        if let Some(rewrite) = self.simplify_sin_cos_pair(terms, metadata, context) {
            return rewrite;
        }
        self.simplify_one_minus_trig_square(terms, metadata, context)
    }

    fn normalize_parity(
        &self,
        op: UnaryOp,
        input: ExprId,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> Rewrite {
        let Some(positive_input) = self.negated_input(input, context) else {
            return Rewrite::Keep;
        };
        match op {
            UnaryOp::Cos => Rewrite::Replace {
                node: ExprNode::Unary {
                    op: UnaryOp::Cos,
                    input: positive_input,
                },
                metadata: metadata.clone(),
            },
            UnaryOp::Sin => {
                let mut builder = ReplacementFragment::new(context);
                let sin = builder.push(
                    ExprNode::Unary {
                        op: UnaryOp::Sin,
                        input: positive_input,
                    },
                    ExprMetadata::new(ExprSourceKind::Unary),
                );
                builder.push(
                    ExprNode::Unary {
                        op: UnaryOp::Neg,
                        input: sin,
                    },
                    metadata.clone(),
                );
                builder.into_rewrite()
            }
            _ => Rewrite::Keep,
        }
    }

    fn simplify_affine_cos_pair(
        &self,
        terms: &[ExprId],
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> Option<Rewrite> {
        for (lhs_index, lhs) in terms.iter().enumerate() {
            let Some(lhs) = self.affine_cos_term(*lhs, context) else {
                continue;
            };
            for (rhs_index, rhs) in terms.iter().enumerate().skip(lhs_index + 1) {
                let Some(rhs) = self.affine_cos_term(*rhs, context) else {
                    continue;
                };
                if lhs.input == rhs.input {
                    return Some(self.replace_affine_cos_pair(
                        terms, lhs_index, lhs, rhs_index, rhs, metadata, context,
                    ));
                }
            }
        }
        None
    }

    #[allow(clippy::too_many_arguments)]
    fn replace_affine_cos_pair(
        &self,
        terms: &[ExprId],
        lhs_index: usize,
        lhs: AffineCosTerm,
        rhs_index: usize,
        rhs: AffineCosTerm,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> Rewrite {
        let constant = lhs.constant + rhs.constant;
        let cos_coefficient = lhs.cos_coefficient + rhs.cos_coefficient;

        let mut builder = ReplacementFragment::new(context);
        let mut replacement_terms = Vec::new();
        if !ProductTerm::approx_eq(constant, 0.0) {
            replacement_terms.push(builder.push(
                ExprNode::RealConst(constant),
                ExprMetadata::new(ExprSourceKind::Const),
            ));
        }
        if !ProductTerm::approx_eq(cos_coefficient, 0.0) {
            let cos = builder.push(
                ExprNode::Unary {
                    op: UnaryOp::Cos,
                    input: lhs.input,
                },
                ExprMetadata::new(ExprSourceKind::Unary),
            );
            replacement_terms.push(ProductTerm::push_parts(
                cos_coefficient,
                &[PowerFactor {
                    base: cos,
                    exponent: 1,
                }],
                &mut builder,
            ));
        }

        let replacement = match replacement_terms.as_slice() {
            [] => builder.push(
                ExprNode::RealConst(0.0),
                ExprMetadata::new(ExprSourceKind::Const),
            ),
            [term] => *term,
            _ => builder.push(
                ExprNode::NaryAdd {
                    terms: replacement_terms,
                },
                ExprMetadata::new(ExprSourceKind::Binary),
            ),
        };

        let mut new_terms: Vec<_> = terms
            .iter()
            .enumerate()
            .filter_map(|(index, term)| {
                if index == lhs_index || index == rhs_index {
                    None
                } else {
                    Some(*term)
                }
            })
            .collect();
        if new_terms.is_empty() {
            builder.into_rewrite()
        } else {
            new_terms.push(replacement);
            builder.push(ExprNode::NaryAdd { terms: new_terms }, metadata.clone());
            builder.into_rewrite()
        }
    }

    fn simplify_even_trig_power(
        &self,
        power: i32,
        input: ExprId,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> Rewrite {
        let Some((op, half_input)) = self
            .trig_call(input, context)
            .and_then(|(op, input)| self.half_angle_input(input, context).map(|half| (op, half)))
        else {
            return Rewrite::Keep;
        };
        self.replace_half_angle_power(op, half_input, power, metadata, context)
    }

    fn replace_half_angle_power(
        &self,
        op: TrigSquareOp,
        input: ExprId,
        power: i32,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> Rewrite {
        let mut builder = ReplacementFragment::new(context);
        let one = builder.push(
            ExprNode::RealConst(1.0),
            ExprMetadata::new(ExprSourceKind::Const),
        );
        let cos = builder.push(
            ExprNode::Unary {
                op: UnaryOp::Cos,
                input,
            },
            ExprMetadata::new(ExprSourceKind::Unary),
        );
        let signed_cos = match op {
            TrigSquareOp::Sin => builder.negated_term(cos),
            TrigSquareOp::Cos => cos,
        };
        let sum = builder.push(
            ExprNode::NaryAdd {
                terms: vec![one, signed_cos],
            },
            ExprMetadata::new(ExprSourceKind::Binary),
        );
        let half = builder.push(
            ExprNode::RealConst(0.5),
            ExprMetadata::new(ExprSourceKind::Const),
        );
        let square = builder.push(
            ExprNode::NaryMul {
                factors: vec![half, sum],
            },
            if power == 2 {
                metadata.clone()
            } else {
                ExprMetadata::new(ExprSourceKind::Binary)
            },
        );
        if power == 2 {
            return builder.into_rewrite();
        }
        builder.push(
            ExprNode::Unary {
                op: UnaryOp::PowI(power / 2),
                input: square,
            },
            metadata.clone(),
        );
        builder.into_rewrite()
    }

    fn simplify_half_angle_pair(
        &self,
        terms: &[ExprId],
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> Option<Rewrite> {
        for (lhs_index, lhs) in terms.iter().enumerate() {
            let Some(lhs) = self.half_angle_square_term(*lhs, context) else {
                continue;
            };
            for (rhs_index, rhs) in terms.iter().enumerate().skip(lhs_index + 1) {
                let Some(rhs) = self.half_angle_square_term(*rhs, context) else {
                    continue;
                };
                if lhs.input == rhs.input && lhs.op.is_complement(rhs.op) {
                    return Some(self.replace_half_angle_pair(
                        terms, lhs_index, lhs, rhs_index, rhs, metadata, context,
                    ));
                }
            }
        }
        None
    }

    #[allow(clippy::too_many_arguments)]
    fn replace_half_angle_pair(
        &self,
        terms: &[ExprId],
        lhs_index: usize,
        lhs: TrigSquareTerm,
        rhs_index: usize,
        rhs: TrigSquareTerm,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> Rewrite {
        let (sin_coefficient, cos_coefficient) = match (lhs.op, rhs.op) {
            (TrigSquareOp::Sin, TrigSquareOp::Cos) => (lhs.coefficient, rhs.coefficient),
            (TrigSquareOp::Cos, TrigSquareOp::Sin) => (rhs.coefficient, lhs.coefficient),
            _ => return Rewrite::Keep,
        };
        let constant = 0.5 * (sin_coefficient + cos_coefficient);
        let cosine_coefficient = 0.5 * (cos_coefficient - sin_coefficient);

        let mut builder = ReplacementFragment::new(context);
        let mut replacement_terms = Vec::new();
        if !ProductTerm::approx_eq(constant, 0.0) {
            replacement_terms.push(builder.push(
                ExprNode::RealConst(constant),
                ExprMetadata::new(ExprSourceKind::Const),
            ));
        }
        if !ProductTerm::approx_eq(cosine_coefficient, 0.0) {
            let cos = builder.push(
                ExprNode::Unary {
                    op: UnaryOp::Cos,
                    input: lhs.input,
                },
                ExprMetadata::new(ExprSourceKind::Unary),
            );
            replacement_terms.push(ProductTerm::push_parts(
                cosine_coefficient,
                &[PowerFactor {
                    base: cos,
                    exponent: 1,
                }],
                &mut builder,
            ));
        }

        let replacement = match replacement_terms.as_slice() {
            [] => builder.push(
                ExprNode::RealConst(0.0),
                ExprMetadata::new(ExprSourceKind::Const),
            ),
            [term] => *term,
            _ => builder.push(
                ExprNode::NaryAdd {
                    terms: replacement_terms,
                },
                ExprMetadata::new(ExprSourceKind::Binary),
            ),
        };

        let mut new_terms: Vec<_> = terms
            .iter()
            .enumerate()
            .filter_map(|(index, term)| {
                if index == lhs_index || index == rhs_index {
                    None
                } else {
                    Some(*term)
                }
            })
            .collect();
        if new_terms.is_empty() {
            builder.into_rewrite()
        } else {
            new_terms.push(replacement);
            builder.push(ExprNode::NaryAdd { terms: new_terms }, metadata.clone());
            builder.into_rewrite()
        }
    }

    fn simplify_sin_cos_pair(
        &self,
        terms: &[ExprId],
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> Option<Rewrite> {
        for (lhs_index, lhs) in terms.iter().enumerate() {
            let Some((lhs_op, lhs_input)) = self.trig_square(*lhs, context) else {
                continue;
            };
            for (rhs_index, rhs) in terms.iter().enumerate().skip(lhs_index + 1) {
                let Some((rhs_op, rhs_input)) = self.trig_square(*rhs, context) else {
                    continue;
                };
                if lhs_input == rhs_input && lhs_op.is_complement(rhs_op) {
                    return Some(
                        self.replace_pair_with_one(terms, lhs_index, rhs_index, metadata, context),
                    );
                }
            }
        }
        None
    }

    fn replace_pair_with_one(
        &self,
        terms: &[ExprId],
        lhs_index: usize,
        rhs_index: usize,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> Rewrite {
        let remaining: Vec<_> = terms
            .iter()
            .enumerate()
            .filter_map(|(index, term)| {
                if index == lhs_index || index == rhs_index {
                    None
                } else {
                    Some(*term)
                }
            })
            .collect();

        if remaining.is_empty() {
            return Rewrite::Replace {
                node: ExprNode::RealConst(1.0),
                metadata: metadata.clone(),
            };
        }

        let mut builder = ReplacementFragment::new(context);
        let one = builder.push(
            ExprNode::RealConst(1.0),
            ExprMetadata::new(ExprSourceKind::Const),
        );
        let mut terms = remaining;
        terms.push(one);
        builder.push(ExprNode::NaryAdd { terms }, metadata.clone());
        builder.into_rewrite()
    }

    fn simplify_one_minus_trig_square(
        &self,
        terms: &[ExprId],
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> Rewrite {
        let Some(one_index) = terms
            .iter()
            .position(|term| context.node(*term).is_some_and(ExprNode::is_one))
        else {
            return Rewrite::Keep;
        };

        for (index, term) in terms.iter().enumerate() {
            if index == one_index {
                continue;
            }
            let Some((op, input)) = self.negative_trig_square(*term, context) else {
                continue;
            };
            let remaining: Vec<_> = terms
                .iter()
                .enumerate()
                .filter_map(|(term_index, term)| {
                    if term_index == one_index || term_index == index {
                        None
                    } else {
                        Some(*term)
                    }
                })
                .collect();
            return self.replace_one_minus_term(remaining, op, input, metadata, context);
        }

        Rewrite::Keep
    }

    fn simplify_one_minus(
        &self,
        rhs: ExprId,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> Rewrite {
        let Some((op, input)) = self.trig_square(rhs, context) else {
            return Rewrite::Keep;
        };
        let mut builder = ReplacementFragment::new(context);
        self.push_complement_square(op, input, metadata.clone(), &mut builder);
        builder.into_rewrite()
    }

    fn replace_one_minus_term(
        &self,
        remaining: Vec<ExprId>,
        op: TrigSquareOp,
        input: ExprId,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> Rewrite {
        let mut builder = ReplacementFragment::new(context);
        if remaining.is_empty() {
            self.push_complement_square(op, input, metadata.clone(), &mut builder);
            return builder.into_rewrite();
        }

        let mut terms = remaining;
        let replacement = self.push_complement_square(
            op,
            input,
            ExprMetadata::new(ExprSourceKind::Unary),
            &mut builder,
        );
        terms.push(replacement);
        builder.push(ExprNode::NaryAdd { terms }, metadata.clone());
        builder.into_rewrite()
    }

    fn push_complement_square(
        &self,
        op: TrigSquareOp,
        input: ExprId,
        metadata: ExprMetadata,
        builder: &mut ReplacementFragment<'_>,
    ) -> ExprId {
        let trig = builder.push(
            ExprNode::Unary {
                op: op.complement().into_unary_op(),
                input,
            },
            ExprMetadata::new(ExprSourceKind::Unary),
        );
        builder.push(
            ExprNode::Unary {
                op: UnaryOp::PowI(2),
                input: trig,
            },
            metadata,
        )
    }

    fn trig_square(
        &self,
        id: ExprId,
        context: &RewriteContext<'_>,
    ) -> Option<(TrigSquareOp, ExprId)> {
        let ExprNode::Unary {
            op: UnaryOp::PowI(2),
            input,
        } = context.node(id)?
        else {
            return None;
        };
        self.trig_call(*input, context)
    }

    fn negative_trig_square(
        &self,
        id: ExprId,
        context: &RewriteContext<'_>,
    ) -> Option<(TrigSquareOp, ExprId)> {
        match context.node(id)? {
            ExprNode::Unary {
                op: UnaryOp::Neg,
                input,
            } => self.trig_square(*input, context),
            ExprNode::NaryMul { factors } => {
                let mut has_minus_one = false;
                let mut square = None;
                for factor in factors {
                    match context.node(*factor) {
                        Some(ExprNode::RealConst(-1.0)) if !has_minus_one => {
                            has_minus_one = true;
                        }
                        _ if square.is_none() => {
                            square = self.trig_square(*factor, context);
                        }
                        _ => return None,
                    }
                }
                has_minus_one.then_some(square).flatten()
            }
            _ => None,
        }
    }

    fn trig_call(
        &self,
        id: ExprId,
        context: &RewriteContext<'_>,
    ) -> Option<(TrigSquareOp, ExprId)> {
        match context.node(id)? {
            ExprNode::Unary {
                op: UnaryOp::Sin,
                input,
            } => Some((TrigSquareOp::Sin, *input)),
            ExprNode::Unary {
                op: UnaryOp::Cos,
                input,
            } => Some((TrigSquareOp::Cos, *input)),
            _ => None,
        }
    }

    fn half_angle_input(&self, id: ExprId, context: &RewriteContext<'_>) -> Option<ExprId> {
        let term = ProductTerm::from_id(id, context);
        if !ProductTerm::approx_eq(term.signed_coefficient(), 0.5) {
            return None;
        }
        match term.sorted_factor_key().as_slice() {
            [PowerFactor { base, exponent: 1 }] => Some(*base),
            _ => None,
        }
    }

    fn negated_input(&self, id: ExprId, context: &RewriteContext<'_>) -> Option<ExprId> {
        let term = ProductTerm::from_id(id, context);
        if !ProductTerm::approx_eq(term.signed_coefficient(), -1.0) {
            return None;
        }
        match term.sorted_factor_key().as_slice() {
            [PowerFactor { base, exponent: 1 }] => Some(*base),
            _ => None,
        }
    }

    fn half_angle_square_term(
        &self,
        id: ExprId,
        context: &RewriteContext<'_>,
    ) -> Option<TrigSquareTerm> {
        let term = ProductTerm::from_id(id, context);
        let factors = term.sorted_factor_key();
        let mut square = None;
        let mut remaining = Vec::new();
        for factor in factors {
            if square.is_none()
                && factor.exponent == 1
                && let Some((op, input)) =
                    self.trig_square(factor.base, context)
                        .and_then(|(op, input)| {
                            self.half_angle_input(input, context).map(|half| (op, half))
                        })
            {
                square = Some((op, input));
            } else {
                remaining.push(factor);
            }
        }
        if !remaining.is_empty() {
            return None;
        }
        let (op, input) = square?;
        Some(TrigSquareTerm {
            op,
            input,
            coefficient: term.signed_coefficient(),
        })
    }

    fn affine_cos_term(&self, id: ExprId, context: &RewriteContext<'_>) -> Option<AffineCosTerm> {
        let term = ProductTerm::from_id(id, context);
        let mut scale = term.signed_coefficient();
        let mut affine = None;
        for factor in term.sorted_factor_key() {
            if factor.exponent == 1
                && affine.is_none()
                && let Some(parsed) = self.affine_cos_sum(factor.base, context)
            {
                affine = Some(parsed);
            } else {
                return None;
            }
        }
        let mut affine = affine?;
        scale *= affine.scale;
        affine.scale = 1.0;
        Some(AffineCosTerm {
            input: affine.input,
            scale: 1.0,
            constant: scale * affine.constant,
            cos_coefficient: scale * affine.cos_coefficient,
        })
    }

    fn affine_cos_sum(&self, id: ExprId, context: &RewriteContext<'_>) -> Option<AffineCosTerm> {
        let Some(ExprNode::NaryAdd { terms }) = context.node(id) else {
            return None;
        };
        let mut constant = 0.0;
        let mut cos_coefficient = 0.0;
        let mut input = None;
        for term_id in terms {
            let term = ProductTerm::from_id(*term_id, context);
            let factors = term.sorted_factor_key();
            match factors.as_slice() {
                [] => constant += term.signed_coefficient(),
                [PowerFactor { base, exponent: 1 }] => {
                    let Some((TrigSquareOp::Cos, cos_input)) = self.trig_call(*base, context)
                    else {
                        return None;
                    };
                    if let Some(input) = input {
                        if input != cos_input {
                            return None;
                        }
                    } else {
                        input = Some(cos_input);
                    }
                    cos_coefficient += term.signed_coefficient();
                }
                _ => return None,
            }
        }
        Some(AffineCosTerm {
            input: input?,
            scale: 1.0,
            constant,
            cos_coefficient,
        })
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum TrigSquareOp {
    Sin,
    Cos,
}

impl TrigSquareOp {
    fn complement(self) -> Self {
        match self {
            Self::Sin => Self::Cos,
            Self::Cos => Self::Sin,
        }
    }

    fn is_complement(self, other: Self) -> bool {
        self.complement() == other
    }

    fn into_unary_op(self) -> UnaryOp {
        match self {
            Self::Sin => UnaryOp::Sin,
            Self::Cos => UnaryOp::Cos,
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct TrigSquareTerm {
    op: TrigSquareOp,
    input: ExprId,
    coefficient: f64,
}

#[derive(Copy, Clone, Debug)]
struct AffineCosTerm {
    input: ExprId,
    scale: f64,
    constant: f64,
    cos_coefficient: f64,
}

fn is_scalar_value(context: &RewriteContext<'_>, id: ExprId) -> bool {
    context
        .facts(id)
        .is_some_and(|facts| matches!(facts.value_kind, ValueKind::Real | ValueKind::Complex))
}

#[derive(Copy, Clone, Debug, Default)]
struct NormalizeAddMulRule;

impl RewriteRule for NormalizeAddMulRule {
    fn name(&self) -> &'static str {
        "normalize-add-mul"
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
            } => Ok(Rewrite::Replace {
                node: ExprNode::NaryAdd {
                    terms: vec![*lhs, *rhs],
                },
                metadata: metadata.clone(),
            }),
            ExprNode::Binary {
                op: BinaryOp::Sub,
                lhs,
                rhs,
            } => self.normalize_subtraction(*lhs, *rhs, metadata, context),
            ExprNode::Binary {
                op: BinaryOp::Mul,
                lhs,
                rhs,
            } => self.normalize_product(&[*lhs, *rhs], metadata, context),
            ExprNode::Binary {
                op: BinaryOp::Div,
                lhs,
                rhs,
            } if is_scalar_value(context, *lhs) && is_scalar_value(context, *rhs) => {
                self.normalize_division(*lhs, *rhs, metadata, context)
            }
            ExprNode::NaryAdd { terms } => self.normalize_sum(terms, metadata, context),
            ExprNode::NaryMul { factors } => self.normalize_product(factors, metadata, context),
            _ => Ok(Rewrite::Keep),
        }
    }
}

impl NormalizeAddMulRule {
    fn normalize_subtraction(
        &self,
        lhs: ExprId,
        rhs: ExprId,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite> {
        let mut builder = ReplacementFragment::new(context);
        let rhs = builder.negated_term(rhs);
        builder.push(
            ExprNode::NaryAdd {
                terms: vec![lhs, rhs],
            },
            metadata.clone(),
        );
        Ok(builder.into_rewrite())
    }

    fn normalize_division(
        &self,
        lhs: ExprId,
        rhs: ExprId,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite> {
        let mut builder = ReplacementFragment::new(context);
        let reciprocal = builder.push(
            ExprNode::Unary {
                op: UnaryOp::PowI(-1),
                input: rhs,
            },
            ExprMetadata::new(ExprSourceKind::Unary),
        );
        builder.push(
            ExprNode::NaryMul {
                factors: vec![lhs, reciprocal],
            },
            metadata.clone(),
        );
        Ok(builder.into_rewrite())
    }

    fn normalize_sum(
        &self,
        terms: &[ExprId],
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite> {
        let mut builder = ReplacementFragment::new(context);
        let mut normalized = Vec::new();
        for term in terms {
            match context.node(*term) {
                Some(ExprNode::NaryAdd { terms }) => normalized.extend(terms.iter().copied()),
                Some(node) if ExprNode::is_zero(node) => {}
                _ if let Some(scaled_terms) =
                    self.scaled_sum_terms(*term, context, &mut builder) =>
                {
                    normalized.extend(scaled_terms);
                }
                _ => normalized.push(*term),
            }
        }

        Ok(match normalized.as_slice() {
            [] => Rewrite::Replace {
                node: ExprNode::RealConst(0.0),
                metadata: metadata.clone(),
            },
            [term] if builder.is_empty() => alias_or_preserve(*term, metadata, context),
            [_] => builder.into_rewrite(),
            _ if normalized == terms && builder.is_empty() => Rewrite::Keep,
            _ if builder.is_empty() => Rewrite::Replace {
                node: ExprNode::NaryAdd { terms: normalized },
                metadata: metadata.clone(),
            },
            _ => {
                builder.push(ExprNode::NaryAdd { terms: normalized }, metadata.clone());
                builder.into_rewrite()
            }
        })
    }

    fn scaled_sum_terms(
        &self,
        id: ExprId,
        context: &RewriteContext<'_>,
        builder: &mut ReplacementFragment<'_>,
    ) -> Option<Vec<ExprId>> {
        let term = ProductTerm::from_id(id, context);
        let coefficient = term.signed_coefficient();
        let factors = term.sorted_factor_key();
        let [PowerFactor { base, exponent: 1 }] = factors.as_slice() else {
            return None;
        };
        let Some(ExprNode::NaryAdd { terms }) = context.node(*base) else {
            return None;
        };
        Some(
            terms
                .iter()
                .map(|term| {
                    if ProductTerm::approx_eq(coefficient, 1.0) {
                        *term
                    } else {
                        ProductTerm::push_parts(
                            coefficient,
                            &[PowerFactor {
                                base: *term,
                                exponent: 1,
                            }],
                            builder,
                        )
                    }
                })
                .collect(),
        )
    }

    fn normalize_product(
        &self,
        factors: &[ExprId],
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite> {
        if let Some(rewrite) =
            self.absorb_additive_factor_coefficient(factors, metadata, context)?
        {
            return Ok(rewrite);
        }
        if let Some(rewrite) = self.combine_same_power_factors(factors, metadata, context)? {
            return Ok(rewrite);
        }
        let mut collector = ProductCollector::new(context);
        collector.collect_all(factors);
        Ok(collector.into_rewrite(factors, metadata))
    }

    fn combine_same_power_factors(
        &self,
        factors: &[ExprId],
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Option<Rewrite>> {
        let Some((power, selected)) = self.largest_same_power_group(factors, context) else {
            return Ok(None);
        };
        let mut builder = ReplacementFragment::new(context);
        let inner_factors = selected.iter().map(|index| {
            let Some(ExprNode::Unary {
                op: UnaryOp::PowI(_),
                input,
            }) = context.node(factors[*index])
            else {
                unreachable!("selected factors are powers")
            };
            *input
        });
        let inner = builder.push(
            ExprNode::NaryMul {
                factors: inner_factors.collect(),
            },
            ExprMetadata::new(ExprSourceKind::Binary),
        );
        let combined = builder.push(
            ExprNode::Unary {
                op: UnaryOp::PowI(power),
                input: inner,
            },
            ExprMetadata::new(ExprSourceKind::Unary),
        );
        let first_selected = selected[0];
        let mut output_factors = Vec::with_capacity(factors.len() - selected.len() + 1);
        for (index, factor) in factors.iter().enumerate() {
            if index == first_selected {
                output_factors.push(combined);
            } else if !selected.contains(&index) {
                output_factors.push(*factor);
            }
        }
        builder.push(
            ExprNode::NaryMul {
                factors: output_factors,
            },
            metadata.clone(),
        );
        let Rewrite::ReplaceMany { nodes } = builder.into_rewrite() else {
            unreachable!("replacement builder always produces fragments")
        };
        let original = ExprNode::NaryMul {
            factors: factors.to_vec(),
        };
        let original_cost = context.local_node_cost(original, metadata.clone())?;
        let candidate_cost = context.local_fragment_cost(&nodes)?;
        if candidate_cost.is_better_than(&original_cost) {
            Ok(Some(Rewrite::ReplaceMany { nodes }))
        } else {
            Ok(None)
        }
    }

    fn largest_same_power_group(
        &self,
        factors: &[ExprId],
        context: &RewriteContext<'_>,
    ) -> Option<(i32, Vec<usize>)> {
        let mut best = None;
        for (index, factor) in factors.iter().enumerate() {
            let Some(ExprNode::Unary {
                op: UnaryOp::PowI(power),
                input,
            }) = context.node(*factor)
            else {
                continue;
            };
            if *power <= 1 || !is_scalar_value(context, *input) {
                continue;
            }
            let selected: Vec<_> = factors
                .iter()
                .enumerate()
                .skip(index)
                .filter_map(|(candidate_index, candidate)| {
                    matches!(
                        context.node(*candidate),
                        Some(ExprNode::Unary {
                            op: UnaryOp::PowI(candidate_power),
                            input,
                        }) if candidate_power == power && is_scalar_value(context, *input)
                    )
                    .then_some(candidate_index)
                })
                .collect();
            if selected.len() >= 2
                && best
                    .as_ref()
                    .is_none_or(|(_, best_selected): &(i32, Vec<usize>)| {
                        selected.len() > best_selected.len()
                    })
            {
                best = Some((*power, selected));
            }
        }
        best
    }

    fn absorb_additive_factor_coefficient(
        &self,
        factors: &[ExprId],
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Option<Rewrite>> {
        let Some((scalar_index, scalar)) =
            factors
                .iter()
                .enumerate()
                .find_map(|(index, factor)| match context.node(*factor) {
                    Some(ExprNode::RealConst(value)) if value.is_finite() => Some((index, *value)),
                    _ => None,
                })
        else {
            return Ok(None);
        };

        for (add_index, factor) in factors.iter().enumerate() {
            let Some(ExprNode::NaryAdd { terms }) = context.node(*factor) else {
                continue;
            };
            let mut product_terms: Vec<_> = terms
                .iter()
                .map(|term| ProductTerm::from_id(*term, context))
                .collect();
            let common_coefficient =
                ProductTerm::take_common_coefficient_from_all(&mut product_terms);
            if ProductTerm::approx_eq(common_coefficient, 1.0) {
                continue;
            }

            let mut builder = ReplacementFragment::new(context);
            let normalized_terms = product_terms
                .iter()
                .map(|term| term.push_remainder(&mut builder))
                .collect();
            let normalized_add = builder.push(
                ExprNode::NaryAdd {
                    terms: normalized_terms,
                },
                ExprMetadata::new(ExprSourceKind::Binary),
            );
            let absorbed_scalar = builder.push(
                ExprNode::RealConst(scalar * common_coefficient),
                ExprMetadata::new(ExprSourceKind::Const),
            );
            let normalized_factors = factors
                .iter()
                .enumerate()
                .map(|(index, factor)| {
                    if index == scalar_index {
                        absorbed_scalar
                    } else if index == add_index {
                        normalized_add
                    } else {
                        *factor
                    }
                })
                .collect();
            builder.push(
                ExprNode::NaryMul {
                    factors: normalized_factors,
                },
                metadata.clone(),
            );
            let Rewrite::ReplaceMany { nodes } = builder.into_rewrite() else {
                unreachable!("replacement builder always produces fragments")
            };
            let original = ExprNode::NaryMul {
                factors: factors.to_vec(),
            };
            let original_cost = context.local_node_cost(original, metadata.clone())?;
            let candidate_cost = context.local_fragment_cost(&nodes)?;
            if candidate_cost.is_no_worse_than(&original_cost) {
                return Ok(Some(Rewrite::ReplaceMany { nodes }));
            }
        }

        Ok(None)
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum ProductPiece {
    Power { base: ExprId, exponent: i32 },
    Const(Complex64),
}

struct ProductCollector<'a> {
    context: &'a RewriteContext<'a>,
    coefficient: Complex64,
    pieces: Vec<ProductPiece>,
    zero: bool,
    changed: bool,
}

impl<'a> ProductCollector<'a> {
    fn new(context: &'a RewriteContext<'a>) -> Self {
        Self {
            context,
            coefficient: Complex64::ONE,
            pieces: Vec::new(),
            zero: false,
            changed: false,
        }
    }

    fn collect_all(&mut self, factors: &[ExprId]) {
        for factor in factors {
            self.collect_factor(*factor);
        }
    }

    fn collect_factor(&mut self, id: ExprId) {
        if self.zero {
            return;
        }
        match self.context.node(id) {
            Some(ExprNode::NaryMul { factors }) => {
                self.changed = true;
                for factor in factors {
                    self.collect_factor(*factor);
                }
            }
            Some(ExprNode::Binary {
                op: BinaryOp::Mul,
                lhs,
                rhs,
            }) => {
                self.changed = true;
                self.collect_factor(*lhs);
                self.collect_factor(*rhs);
            }
            Some(ExprNode::Unary {
                op: UnaryOp::Neg,
                input,
            }) => {
                self.changed = true;
                self.coefficient = -self.coefficient;
                self.collect_factor(*input);
            }
            Some(ExprNode::Unary {
                op: UnaryOp::PowI(power),
                input,
            }) if self.is_scalar(*input) => {
                self.changed = true;
                self.collect_power(*input, *power);
            }
            Some(ExprNode::RealConst(value)) => self.collect_real_const(*value),
            Some(ExprNode::ComplexConst(value)) => {
                self.changed = true;
                self.coefficient *= *value;
                if self.coefficient == Complex64::ZERO {
                    self.zero = true;
                }
            }
            _ if self.is_scalar(id) => self.collect_power(id, 1),
            _ => self.pieces.push(ProductPiece::Power {
                base: id,
                exponent: 1,
            }),
        }
    }

    fn collect_power(&mut self, base: ExprId, exponent: i32) {
        if exponent == 0 {
            self.changed = true;
            return;
        }

        if let Some(ProductPiece::Power {
            exponent: existing,
            ..
        }) = self
            .pieces
            .iter_mut()
            .find(|piece| matches!(piece, ProductPiece::Power { base: candidate, .. } if *candidate == base))
        {
            *existing += exponent;
            self.changed = true;
        } else {
            self.pieces.push(ProductPiece::Power { base, exponent });
        }
    }

    fn collect_real_const(&mut self, value: f64) {
        if value == 0.0 {
            self.changed = true;
            self.zero = true;
            return;
        }

        self.changed = true;
        self.coefficient *= value;
    }

    fn into_rewrite(mut self, original: &[ExprId], metadata: &ExprMetadata) -> Rewrite {
        if self.zero || self.coefficient == Complex64::ZERO {
            return Rewrite::Replace {
                node: ExprNode::RealConst(0.0),
                metadata: metadata.clone(),
            };
        }

        self.remove_cancelled_powers();

        if self.coefficient != Complex64::ONE || self.pieces.is_empty() {
            self.pieces.insert(0, ProductPiece::Const(self.coefficient));
        }

        if self.has_original_shape(original) {
            return Rewrite::Keep;
        }

        match self.pieces.as_slice() {
            [] => {
                return Rewrite::Replace {
                    node: ExprNode::RealConst(1.0),
                    metadata: metadata.clone(),
                };
            }
            [ProductPiece::Power { base, exponent: 1 }] => {
                return alias_or_preserve(*base, metadata, self.context);
            }
            [ProductPiece::Const(value)] => {
                return Rewrite::Replace {
                    node: ExprNode::from(*value),
                    metadata: metadata.clone(),
                };
            }
            _ => {}
        }

        let mut builder = ReplacementFragment::new(self.context);
        let mut factors = Vec::with_capacity(self.pieces.len());
        for piece in self.pieces {
            match piece {
                ProductPiece::Power { base, exponent: 1 } => factors.push(base),
                ProductPiece::Power { base, exponent } => factors.push(builder.push(
                    ExprNode::Unary {
                        op: UnaryOp::PowI(exponent),
                        input: base,
                    },
                    ExprMetadata::new(ExprSourceKind::Unary),
                )),
                ProductPiece::Const(value) => factors.push(builder.push(
                    ExprNode::from(value),
                    ExprMetadata::new(ExprSourceKind::Const),
                )),
            }
        }
        builder.push(ExprNode::NaryMul { factors }, metadata.clone());
        builder.into_rewrite()
    }

    fn remove_cancelled_powers(&mut self) {
        self.pieces.retain(|piece| match piece {
            ProductPiece::Power { exponent, .. } if *exponent == 0 => {
                self.changed = true;
                false
            }
            _ => true,
        });
    }

    fn has_original_shape(&self, original: &[ExprId]) -> bool {
        !self.changed
            && self.pieces.len() == original.len()
            && self
                .pieces
                .iter()
                .zip(original)
                .all(|(piece, original)| piece.as_identity_factor() == Some(*original))
    }

    fn is_scalar(&self, id: ExprId) -> bool {
        is_scalar_value(self.context, id)
    }
}

impl ProductPiece {
    fn as_identity_factor(&self) -> Option<ExprId> {
        match self {
            Self::Power { base, exponent: 1 } => Some(*base),
            Self::Power { .. } | Self::Const(_) => None,
        }
    }
}

struct ReplacementFragment<'a> {
    context: &'a RewriteContext<'a>,
    nodes: Vec<(ExprNode, ExprMetadata)>,
}

impl<'a> ReplacementFragment<'a> {
    fn new(context: &'a RewriteContext<'a>) -> Self {
        Self {
            context,
            nodes: Vec::new(),
        }
    }

    fn push(&mut self, node: ExprNode, metadata: ExprMetadata) -> ExprId {
        let id = self.next_id();
        self.nodes.push((node, metadata));
        id
    }

    fn negated_term(&mut self, id: ExprId) -> ExprId {
        let minus_one = self.push(
            ExprNode::RealConst(-1.0),
            ExprMetadata::new(ExprSourceKind::Const),
        );
        self.push(
            ExprNode::NaryMul {
                factors: vec![minus_one, id],
            },
            ExprMetadata::new(ExprSourceKind::Binary),
        )
    }

    fn conjugated_term(&mut self, id: ExprId) -> ExprId {
        self.push(
            ExprNode::Unary {
                op: UnaryOp::Conj,
                input: id,
            },
            ExprMetadata::new(ExprSourceKind::Unary),
        )
    }

    fn next_id(&self) -> ExprId {
        ExprId::from_index(self.context.next_id().index() + self.nodes.len())
    }

    fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    fn into_rewrite(self) -> Rewrite {
        Rewrite::ReplaceMany { nodes: self.nodes }
    }
}

/// Combines additive terms with identical symbolic factors.
#[derive(Copy, Clone, Debug, Default)]
pub struct CombineLikeTermsRule;

impl RewriteRule for CombineLikeTermsRule {
    fn name(&self) -> &'static str {
        "combine-like-terms"
    }

    fn rewrite(
        &self,
        node: &ExprNode,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite> {
        match node {
            ExprNode::NaryAdd { terms } => Ok(self.combine_terms(terms, metadata, context)),
            _ => Ok(Rewrite::Keep),
        }
    }
}

impl CombineLikeTermsRule {
    fn combine_terms(
        &self,
        terms: &[ExprId],
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> Rewrite {
        let mut groups: Vec<LikeTermGroup> = Vec::new();
        let mut changed = false;

        for term in terms {
            let product = ProductTerm::from_id(*term, context);
            let coefficient = product.signed_coefficient();
            if coefficient == 0.0 {
                changed = true;
                continue;
            }

            let key = product.sorted_factor_key();
            if let Some(group) = groups.iter_mut().find(|group| group.factors == key) {
                group.coefficient += coefficient;
                changed = true;
            } else {
                groups.push(LikeTermGroup {
                    coefficient,
                    factors: key,
                });
            }
        }

        let mut builder = ReplacementFragment::new(context);
        let mut combined = Vec::new();
        for group in groups {
            if ProductTerm::approx_eq(group.coefficient, 0.0) {
                changed = true;
                continue;
            }
            combined.push(ProductTerm::push_parts(
                group.coefficient,
                &group.factors,
                &mut builder,
            ));
        }

        if !changed {
            return Rewrite::Keep;
        }

        match combined.as_slice() {
            [] => Rewrite::Replace {
                node: ExprNode::RealConst(0.0),
                metadata: metadata.clone(),
            },
            [term] if builder.is_empty() => alias_or_preserve(*term, metadata, context),
            [_] => builder.into_rewrite(),
            _ => {
                builder.push(ExprNode::NaryAdd { terms: combined }, metadata.clone());
                builder.into_rewrite()
            }
        }
    }
}

#[derive(Clone, Debug)]
struct LikeTermGroup {
    coefficient: f64,
    factors: Vec<PowerFactor>,
}

/// Factors products shared by additive terms.
#[derive(Copy, Clone, Debug, Default)]
pub struct FactorCommonProductRule;

#[derive(Clone, Debug)]
struct FactorCandidate {
    nodes: Vec<(ExprNode, ExprMetadata)>,
}

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
            } => self.factor_common_products(node, &[*lhs, *rhs], metadata, context),
            ExprNode::NaryAdd { terms } if terms.len() >= 2 => {
                self.factor_common_products(node, terms, metadata, context)
            }
            _ => Ok(Rewrite::Keep),
        }
    }
}

impl FactorCommonProductRule {
    fn factor_common_products(
        &self,
        original: &ExprNode,
        terms: &[ExprId],
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite> {
        let Some(nodes) =
            self.best_common_product_factor_nodes(original, terms, metadata, context)?
        else {
            return Ok(Rewrite::Keep);
        };
        Ok(Rewrite::ReplaceMany { nodes })
    }

    fn best_common_product_factor_nodes(
        &self,
        original: &ExprNode,
        term_ids: &[ExprId],
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Option<Vec<(ExprNode, ExprMetadata)>>> {
        let candidates = [
            self.common_product_factor_candidate(term_ids, metadata, context),
            self.partial_common_product_factor_candidate(term_ids, metadata, context),
        ];
        let original_cost = context.local_node_cost(original.clone(), metadata.clone())?;
        let mut best_cost = None;
        let mut best_nodes = None;

        for candidate in candidates.into_iter().flatten() {
            let candidate_cost = context.local_fragment_cost(&candidate.nodes)?;
            if best_cost
                .as_ref()
                .is_none_or(|best_cost| candidate_cost.is_better_than(best_cost))
            {
                best_cost = Some(candidate_cost);
                best_nodes = Some(candidate.nodes);
            }
        }

        let Some(best_cost) = best_cost else {
            return Ok(None);
        };
        if !best_cost.is_no_worse_than(&original_cost) && context.ids_are_all_constants(term_ids) {
            return Ok(None);
        }
        Ok(best_nodes)
    }

    fn common_product_factor_candidate(
        &self,
        term_ids: &[ExprId],
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> Option<FactorCandidate> {
        let mut terms: Vec<_> = term_ids
            .iter()
            .map(|term| ProductTerm::from_id(*term, context))
            .collect();
        let common = ProductTerm::take_common_factors_from_all(&mut terms);
        let common_coefficient = ProductTerm::take_common_coefficient_from_all(&mut terms);
        if common.is_empty() && ProductTerm::approx_eq(common_coefficient, 1.0) {
            return None;
        }

        let mut builder = ReplacementFragment::new(context);
        let remainder_terms = terms
            .iter()
            .map(|term| term.push_remainder(&mut builder))
            .collect();
        let sum = builder.push(
            ExprNode::NaryAdd {
                terms: remainder_terms,
            },
            ExprMetadata::new(ExprSourceKind::Binary),
        );
        let mut factors = Vec::new();
        if !ProductTerm::approx_eq(common_coefficient, 1.0) {
            factors.push(builder.push(
                ExprNode::RealConst(common_coefficient),
                ExprMetadata::new(ExprSourceKind::Const),
            ));
        }
        factors.extend(common.iter().map(|factor| factor.emit(&mut builder)));
        factors.push(sum);
        builder.push(ExprNode::NaryMul { factors }, metadata.clone());
        let Rewrite::ReplaceMany { nodes } = builder.into_rewrite() else {
            unreachable!("replacement builder always produces fragments")
        };
        Some(FactorCandidate { nodes })
    }

    fn partial_common_product_factor_candidate(
        &self,
        term_ids: &[ExprId],
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> Option<FactorCandidate> {
        let terms: Vec<_> = term_ids
            .iter()
            .map(|term| ProductTerm::from_id(*term, context))
            .collect();
        let common = Self::best_partial_common_factors(&terms)?;
        let selected_indices: Vec<_> = terms
            .iter()
            .enumerate()
            .filter_map(|(index, term)| term.has_factors(&common).then_some(index))
            .collect();
        if selected_indices.len() < 2 || selected_indices.len() == terms.len() {
            return None;
        }

        let mut remainders = terms.clone();
        for index in &selected_indices {
            for factor in &common {
                remainders[*index].remove_factor(*factor);
            }
        }

        let mut builder = ReplacementFragment::new(context);
        let subset_terms = selected_indices
            .iter()
            .map(|index| remainders[*index].push_remainder(&mut builder))
            .collect();
        let subset_sum = builder.push(
            ExprNode::NaryAdd {
                terms: subset_terms,
            },
            ExprMetadata::new(ExprSourceKind::Binary),
        );
        let mut factored_factors: Vec<_> = common
            .iter()
            .map(|factor| factor.emit(&mut builder))
            .collect();
        factored_factors.push(subset_sum);
        let factored_term = builder.push(
            ExprNode::NaryMul {
                factors: factored_factors,
            },
            ExprMetadata::new(ExprSourceKind::Binary),
        );

        let mut output_terms = Vec::with_capacity(term_ids.len() - selected_indices.len() + 1);
        for (index, term_id) in term_ids.iter().enumerate() {
            if !selected_indices.contains(&index) {
                output_terms.push(*term_id);
            }
        }
        output_terms.push(factored_term);
        builder.push(
            ExprNode::NaryAdd {
                terms: output_terms,
            },
            metadata.clone(),
        );
        let Rewrite::ReplaceMany { nodes } = builder.into_rewrite() else {
            unreachable!("replacement builder always produces fragments")
        };
        Some(FactorCandidate { nodes })
    }

    fn best_partial_common_factors(terms: &[ProductTerm]) -> Option<Vec<PowerFactor>> {
        let mut best = Vec::new();
        for lhs_index in 0..terms.len() {
            for rhs_index in (lhs_index + 1)..terms.len() {
                let mut pair = vec![terms[lhs_index].clone(), terms[rhs_index].clone()];
                let common = ProductTerm::take_common_factors_from_all(&mut pair);
                if common.len() > best.len() {
                    best = common;
                }
            }
        }
        (!best.is_empty()).then_some(best)
    }
}

#[derive(Clone, Debug)]
struct ProductTerm {
    sign: f64,
    coefficient: f64,
    factors: Vec<PowerFactor>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct PowerFactor {
    base: ExprId,
    exponent: i32,
}

impl ProductTerm {
    fn from_id(id: ExprId, context: &RewriteContext<'_>) -> Self {
        let mut term = Self {
            sign: 1.0,
            coefficient: 1.0,
            factors: Vec::new(),
        };
        term.collect(id, context);
        term
    }

    fn collect(&mut self, id: ExprId, context: &RewriteContext<'_>) {
        match context.node(id) {
            Some(ExprNode::NaryMul { factors }) => {
                for factor in factors {
                    self.collect(*factor, context);
                }
            }
            Some(ExprNode::Binary {
                op: BinaryOp::Mul,
                lhs,
                rhs,
            }) => {
                self.collect(*lhs, context);
                self.collect(*rhs, context);
            }
            Some(ExprNode::Unary {
                op: UnaryOp::Neg,
                input,
            }) => {
                self.sign *= -1.0;
                self.collect(*input, context);
            }
            Some(ExprNode::Unary {
                op: UnaryOp::PowI(power),
                input,
            }) if is_scalar_value(context, *input) => {
                if !self.collect_powered_product(*input, *power, context) {
                    self.collect_power(*input, *power);
                }
            }
            Some(ExprNode::RealConst(-1.0)) => {
                self.sign *= -1.0;
            }
            Some(ExprNode::RealConst(value)) if *value < 0.0 => {
                self.sign *= -1.0;
                self.coefficient *= -*value;
            }
            Some(ExprNode::RealConst(value)) => {
                self.coefficient *= *value;
            }
            _ => self.collect_power(id, 1),
        }
    }

    fn collect_power(&mut self, base: ExprId, exponent: i32) {
        if exponent == 0 {
            return;
        }

        if let Some(existing) = self.factors.iter_mut().find(|factor| factor.base == base)
            && let Some(exponent) = existing.exponent.checked_add(exponent)
        {
            existing.exponent = exponent;
            return;
        }

        self.factors.push(PowerFactor { base, exponent });
    }

    fn collect_powered_product(
        &mut self,
        input: ExprId,
        exponent: i32,
        context: &RewriteContext<'_>,
    ) -> bool {
        if !matches!(context.node(input), Some(ExprNode::NaryMul { .. })) {
            return false;
        }
        let inner = Self::from_id(input, context);
        let coefficient = inner.signed_coefficient().powi(exponent);
        if coefficient < 0.0 {
            self.sign *= -1.0;
            self.coefficient *= -coefficient;
        } else {
            self.coefficient *= coefficient;
        }
        for factor in inner.factors {
            let Some(exponent) = factor.exponent.checked_mul(exponent) else {
                return false;
            };
            self.collect_power(factor.base, exponent);
        }
        true
    }

    fn signed_coefficient(&self) -> f64 {
        self.sign * self.coefficient
    }

    fn sorted_factor_key(&self) -> Vec<PowerFactor> {
        let mut factors: Vec<_> = self
            .factors
            .iter()
            .copied()
            .filter(|factor| factor.exponent != 0)
            .collect();
        factors.sort_by_key(|factor| (factor.base.index(), factor.exponent));
        factors
    }

    fn push_remainder(&self, builder: &mut ReplacementFragment<'_>) -> ExprId {
        Self::push_parts(self.signed_coefficient(), &self.factors, builder)
    }

    fn push_parts(
        coefficient: f64,
        source_factors: &[PowerFactor],
        builder: &mut ReplacementFragment<'_>,
    ) -> ExprId {
        let mut factors = Vec::new();
        for factor in source_factors {
            if factor.exponent == 0 {
                continue;
            }
            factors.push(factor.emit(builder));
        }

        if !Self::approx_eq(coefficient, 1.0) || factors.is_empty() {
            let coefficient = builder.push(
                ExprNode::RealConst(coefficient),
                ExprMetadata::new(ExprSourceKind::Const),
            );
            factors.insert(0, coefficient);
        }

        match factors.as_slice() {
            [factor] => *factor,
            _ => builder.push(
                ExprNode::NaryMul { factors },
                ExprMetadata::new(ExprSourceKind::Binary),
            ),
        }
    }

    fn take_common_factors_from_all(terms: &mut [Self]) -> Vec<PowerFactor> {
        let Some((first, rest)) = terms.split_first() else {
            return Vec::new();
        };

        let mut common = first.sorted_factor_key();
        for term in rest {
            common = common
                .into_iter()
                .filter_map(|factor| {
                    term.factors
                        .iter()
                        .find_map(|candidate| factor.common_with(candidate))
                })
                .collect();
            if common.is_empty() {
                return common;
            }
        }

        for term in terms {
            for factor in &common {
                term.remove_factor(*factor);
            }
        }

        common
    }

    fn take_common_coefficient_from_all(terms: &mut [Self]) -> f64 {
        let Some((first, rest)) = terms.split_first() else {
            return 1.0;
        };

        let mut common = first.coefficient;
        for term in rest {
            common = Self::common_real_coefficient(common, term.coefficient);
            if Self::approx_eq(common, 1.0) {
                return 1.0;
            }
        }

        if !Self::approx_eq(common, 1.0) {
            for term in terms {
                term.coefficient /= common;
            }
        }
        common
    }

    fn remove_factor(&mut self, factor: PowerFactor) {
        let Some(index) = self.factors.iter().position(|candidate| {
            candidate.base == factor.base
                && candidate.exponent.signum() == factor.exponent.signum()
                && candidate.exponent.abs() >= factor.exponent.abs()
        }) else {
            return;
        };

        self.factors[index].exponent -= factor.exponent;
        if self.factors[index].exponent == 0 {
            self.factors.remove(index);
        }
    }

    fn has_factors(&self, factors: &[PowerFactor]) -> bool {
        factors.iter().all(|factor| {
            self.factors.iter().any(|candidate| {
                candidate.base == factor.base
                    && candidate.exponent.signum() == factor.exponent.signum()
                    && candidate.exponent.abs() >= factor.exponent.abs()
            })
        })
    }

    fn common_real_coefficient(lhs: f64, rhs: f64) -> f64 {
        if lhs == 0.0 || rhs == 0.0 || !lhs.is_finite() || !rhs.is_finite() {
            return 1.0;
        }

        let lhs = lhs.abs();
        let rhs = rhs.abs();
        let lhs_integer = lhs.round();
        let rhs_integer = rhs.round();
        if Self::approx_eq(lhs, lhs_integer)
            && Self::approx_eq(rhs, rhs_integer)
            && lhs_integer > 0.0
            && rhs_integer > 0.0
            && lhs_integer <= u64::MAX as f64
            && rhs_integer <= u64::MAX as f64
        {
            return Self::gcd_u64(lhs_integer as u64, rhs_integer as u64) as f64;
        }

        let smaller = lhs.min(rhs);
        let larger = lhs.max(rhs);
        let ratio = larger / smaller;
        if Self::approx_eq(ratio, ratio.round()) {
            smaller
        } else {
            1.0
        }
    }

    fn approx_eq(lhs: f64, rhs: f64) -> bool {
        (lhs - rhs).abs() <= f64::EPSILON * lhs.abs().max(rhs.abs()).max(1.0) * 16.0
    }

    fn gcd_u64(mut lhs: u64, mut rhs: u64) -> u64 {
        while rhs != 0 {
            let remainder = lhs % rhs;
            lhs = rhs;
            rhs = remainder;
        }
        lhs
    }
}

impl PowerFactor {
    fn common_with(&self, other: &Self) -> Option<Self> {
        if self.base != other.base || self.exponent.signum() != other.exponent.signum() {
            return None;
        }

        Some(Self {
            base: self.base,
            exponent: self.exponent.abs().min(other.exponent.abs()) * self.exponent.signum(),
        })
    }

    fn emit(&self, builder: &mut ReplacementFragment<'_>) -> ExprId {
        match self.exponent {
            1 => self.base,
            exponent => builder.push(
                ExprNode::Unary {
                    op: UnaryOp::PowI(exponent),
                    input: self.base,
                },
                ExprMetadata::new(ExprSourceKind::Unary),
            ),
        }
    }
}

/// Rewrites compatible exponential expressions.
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
            ExprNode::Unary {
                op: UnaryOp::Exp, ..
            } => Ok(Rewrite::Keep),
            ExprNode::Binary {
                op: BinaryOp::Mul,
                lhs,
                rhs,
            } => {
                let (Some(lhs_input), Some(rhs_input)) =
                    (self.exp_input(context, *lhs), self.exp_input(context, *rhs))
                else {
                    return Ok(Rewrite::Keep);
                };
                self.merge_product(vec![lhs_input, rhs_input], Vec::new(), metadata, context)
            }
            ExprNode::NaryMul { factors } => {
                let mut exp_inputs = Vec::new();
                let mut other_factors = Vec::new();
                for factor in factors {
                    if let Some(input) = self.exp_input(context, *factor) {
                        exp_inputs.push(input);
                    } else {
                        other_factors.push(*factor);
                    }
                }
                if exp_inputs.len() < 2 {
                    return Ok(Rewrite::Keep);
                }
                self.merge_product(exp_inputs, other_factors, metadata, context)
            }
            ExprNode::NaryAdd { terms } if terms.len() >= 2 => {
                Ok(self.rewrite_euler_sum(terms, metadata, context))
            }
            _ => Ok(Rewrite::Keep),
        }
    }
}

impl ExponentialRule {
    fn merge_product(
        &self,
        exp_inputs: Vec<ExprId>,
        mut other_factors: Vec<ExprId>,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite> {
        if let Some(rewrite) =
            self.merge_imaginary_phase_product(&exp_inputs, &other_factors, metadata, context)
        {
            return Ok(rewrite);
        }

        let mut builder = ReplacementFragment::new(context);
        let add_id = builder.push(
            ExprNode::NaryAdd { terms: exp_inputs },
            ExprMetadata::new(ExprSourceKind::Binary),
        );
        let exp_id = builder.push(
            ExprNode::Unary {
                op: UnaryOp::Exp,
                input: add_id,
            },
            if other_factors.is_empty() {
                metadata.clone()
            } else {
                ExprMetadata::new(ExprSourceKind::Unary)
            },
        );

        if !other_factors.is_empty() {
            other_factors.push(exp_id);
            builder.push(
                ExprNode::NaryMul {
                    factors: other_factors,
                },
                metadata.clone(),
            );
        }
        Ok(builder.into_rewrite())
    }

    fn rewrite_euler_sum(
        &self,
        terms: &[ExprId],
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> Rewrite {
        for (lhs_index, lhs) in terms.iter().enumerate() {
            let Some(lhs) = self.trig_product(*lhs, context) else {
                continue;
            };
            for (rhs_index, rhs) in terms.iter().enumerate().skip(lhs_index + 1) {
                let Some(rhs) = self.trig_product(*rhs, context) else {
                    continue;
                };
                let Some(euler) = lhs.euler_with(&rhs) else {
                    continue;
                };
                return self
                    .replace_euler_pair(terms, lhs_index, rhs_index, euler, metadata, context);
            }
        }
        Rewrite::Keep
    }

    fn replace_euler_pair(
        &self,
        terms: &[ExprId],
        lhs_index: usize,
        rhs_index: usize,
        euler: EulerProduct,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> Rewrite {
        let mut builder = ReplacementFragment::new(context);
        let phase = self.emit_imaginary_phase(euler.phase_sign, euler.input, &mut builder);
        let exp = builder.push(
            ExprNode::Unary {
                op: UnaryOp::Exp,
                input: phase,
            },
            ExprMetadata::new(ExprSourceKind::Unary),
        );
        let euler_term = euler.emit_with_exp(exp, &mut builder);
        let mut new_terms: Vec<_> = terms
            .iter()
            .enumerate()
            .filter_map(|(index, term)| {
                if index == lhs_index || index == rhs_index {
                    None
                } else {
                    Some(*term)
                }
            })
            .collect();
        if new_terms.is_empty() {
            builder.into_rewrite()
        } else {
            new_terms.push(euler_term);
            builder.push(ExprNode::NaryAdd { terms: new_terms }, metadata.clone());
            builder.into_rewrite()
        }
    }

    fn merge_imaginary_phase_product(
        &self,
        exp_inputs: &[ExprId],
        other_factors: &[ExprId],
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> Option<Rewrite> {
        let phases: Vec<_> = exp_inputs
            .iter()
            .map(|input| self.imaginary_phase(*input, context))
            .collect::<Option<_>>()?;
        let mut builder = ReplacementFragment::new(context);
        let phase_terms = phases
            .into_iter()
            .map(|phase| phase.emit(&mut builder))
            .collect();
        let phase_sum = builder.push(
            ExprNode::NaryAdd { terms: phase_terms },
            ExprMetadata::new(ExprSourceKind::Binary),
        );
        let exp_input = self.emit_imaginary_phase(1.0, phase_sum, &mut builder);
        let exp = builder.push(
            ExprNode::Unary {
                op: UnaryOp::Exp,
                input: exp_input,
            },
            if other_factors.is_empty() {
                metadata.clone()
            } else {
                ExprMetadata::new(ExprSourceKind::Unary)
            },
        );

        if other_factors.is_empty() {
            return Some(builder.into_rewrite());
        }

        let mut factors = other_factors.to_vec();
        factors.push(exp);
        builder.push(ExprNode::NaryMul { factors }, metadata.clone());
        Some(builder.into_rewrite())
    }

    fn emit_imaginary_phase(
        &self,
        sign: f64,
        input: ExprId,
        builder: &mut ReplacementFragment<'_>,
    ) -> ExprId {
        let i = builder.push(
            ExprNode::ComplexConst(Complex64::I),
            ExprMetadata::new(ExprSourceKind::Const),
        );
        let phase = if ProductTerm::approx_eq(sign, 1.0) {
            input
        } else {
            builder.negated_term(input)
        };
        builder.push(
            ExprNode::NaryMul {
                factors: vec![i, phase],
            },
            ExprMetadata::new(ExprSourceKind::Binary),
        )
    }

    fn trig_product(&self, id: ExprId, context: &RewriteContext<'_>) -> Option<TrigProduct> {
        let mut product = TrigProduct::new();
        product.collect(id, context)?;
        product.into_normalized()
    }

    fn imaginary_phase(&self, id: ExprId, context: &RewriteContext<'_>) -> Option<PhaseProduct> {
        let mut product = PhaseProduct::new();
        product.collect(id, context)?;
        product.into_normalized(context)
    }

    fn exp_input(&self, context: &RewriteContext<'_>, id: ExprId) -> Option<ExprId> {
        match context.node(id) {
            Some(ExprNode::Unary {
                op: UnaryOp::Exp,
                input,
            }) => Some(*input),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
struct TrigProduct {
    coefficient: Complex64,
    factors: Vec<PowerFactor>,
    trig: Option<(TrigSquareOp, ExprId)>,
}

impl TrigProduct {
    fn new() -> Self {
        Self {
            coefficient: Complex64::ONE,
            factors: Vec::new(),
            trig: None,
        }
    }

    fn collect(&mut self, id: ExprId, context: &RewriteContext<'_>) -> Option<()> {
        match context.node(id)? {
            ExprNode::NaryMul { factors } => {
                for factor in factors {
                    self.collect(*factor, context)?;
                }
            }
            ExprNode::Binary {
                op: BinaryOp::Mul,
                lhs,
                rhs,
            } => {
                self.collect(*lhs, context)?;
                self.collect(*rhs, context)?;
            }
            ExprNode::Unary {
                op: UnaryOp::Neg,
                input,
            } => {
                self.coefficient = -self.coefficient;
                self.collect(*input, context)?;
            }
            ExprNode::RealConst(value) => {
                self.coefficient *= *value;
            }
            ExprNode::ComplexConst(value) => {
                self.coefficient *= *value;
            }
            ExprNode::Unary { op, input } if matches!(op, UnaryOp::Sin | UnaryOp::Cos) => {
                if self.trig.is_some() {
                    return None;
                }
                let op = match op {
                    UnaryOp::Sin => TrigSquareOp::Sin,
                    UnaryOp::Cos => TrigSquareOp::Cos,
                    _ => unreachable!("matched trig ops"),
                };
                self.trig = Some((op, *input));
            }
            ExprNode::Unary {
                op: UnaryOp::PowI(power),
                input,
            } if is_scalar_value(context, *input) => {
                self.factors.push(PowerFactor {
                    base: *input,
                    exponent: *power,
                });
            }
            _ if is_scalar_value(context, id) => {
                self.factors.push(PowerFactor {
                    base: id,
                    exponent: 1,
                });
            }
            _ => return None,
        }
        Some(())
    }

    fn into_normalized(mut self) -> Option<Self> {
        let _ = self.trig?;
        self.factors
            .sort_by_key(|factor| (factor.base.index(), factor.exponent));
        Some(self)
    }

    fn euler_with(&self, other: &Self) -> Option<EulerProduct> {
        let (self_op, self_input) = self.trig?;
        let (other_op, other_input) = other.trig?;
        if self_input != other_input || self_op == other_op || self.factors != other.factors {
            return None;
        }

        let (cos, sin) = match (self_op, other_op) {
            (TrigSquareOp::Cos, TrigSquareOp::Sin) => (self, other),
            (TrigSquareOp::Sin, TrigSquareOp::Cos) => (other, self),
            _ => return None,
        };
        if !ProductTerm::approx_eq(cos.coefficient.im, 0.0)
            || !ProductTerm::approx_eq(sin.coefficient.re, 0.0)
            || !ProductTerm::approx_eq(cos.coefficient.re.abs(), sin.coefficient.im.abs())
        {
            return None;
        }
        let phase_sign = if (sin.coefficient.im / cos.coefficient.re).is_sign_positive() {
            1.0
        } else {
            -1.0
        };
        Some(EulerProduct {
            coefficient: cos.coefficient.re,
            factors: cos.factors.clone(),
            phase_sign,
            input: self_input,
        })
    }
}

#[derive(Clone, Debug)]
struct EulerProduct {
    coefficient: f64,
    factors: Vec<PowerFactor>,
    phase_sign: f64,
    input: ExprId,
}

impl EulerProduct {
    fn emit_with_exp(&self, exp: ExprId, builder: &mut ReplacementFragment<'_>) -> ExprId {
        let mut factors = self.factors.clone();
        factors.push(PowerFactor {
            base: exp,
            exponent: 1,
        });
        ProductTerm::push_parts(self.coefficient, &factors, builder)
    }
}

#[derive(Clone, Debug)]
struct PhaseProduct {
    coefficient: Complex64,
    factors: Vec<PowerFactor>,
}

impl PhaseProduct {
    fn new() -> Self {
        Self {
            coefficient: Complex64::ONE,
            factors: Vec::new(),
        }
    }

    fn collect(&mut self, id: ExprId, context: &RewriteContext<'_>) -> Option<()> {
        match context.node(id)? {
            ExprNode::NaryMul { factors } => {
                for factor in factors {
                    self.collect(*factor, context)?;
                }
            }
            ExprNode::Binary {
                op: BinaryOp::Mul,
                lhs,
                rhs,
            } => {
                self.collect(*lhs, context)?;
                self.collect(*rhs, context)?;
            }
            ExprNode::Unary {
                op: UnaryOp::Neg,
                input,
            } => {
                self.coefficient = -self.coefficient;
                self.collect(*input, context)?;
            }
            ExprNode::RealConst(value) => self.coefficient *= *value,
            ExprNode::ComplexConst(value) => self.coefficient *= *value,
            ExprNode::Unary {
                op: UnaryOp::PowI(power),
                input,
            } if is_scalar_value(context, *input) => {
                self.factors.push(PowerFactor {
                    base: *input,
                    exponent: *power,
                });
            }
            _ if context
                .facts(id)
                .is_some_and(|facts| facts.number_class == NumberClass::Real) =>
            {
                self.factors.push(PowerFactor {
                    base: id,
                    exponent: 1,
                });
            }
            _ => return None,
        }
        Some(())
    }

    fn into_normalized(mut self, _context: &RewriteContext<'_>) -> Option<Self> {
        if !ProductTerm::approx_eq(self.coefficient.re, 0.0)
            || ProductTerm::approx_eq(self.coefficient.im, 0.0)
        {
            return None;
        }
        self.factors
            .sort_by_key(|factor| (factor.base.index(), factor.exponent));
        Some(Self {
            coefficient: Complex64::from(self.coefficient.im),
            factors: self.factors,
        })
    }

    fn emit(&self, builder: &mut ReplacementFragment<'_>) -> ExprId {
        ProductTerm::push_parts(self.coefficient.re, &self.factors, builder)
    }
}

/// Expands squared norms into forms that may expose common subexpressions.
#[derive(Copy, Clone, Debug, Default)]
pub struct NormSqrExpansionRule;

impl RewriteRule for NormSqrExpansionRule {
    fn name(&self) -> &'static str {
        "norm-sqr-expansion"
    }

    fn rewrite(
        &self,
        node: &ExprNode,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite> {
        let ExprNode::Unary {
            op: UnaryOp::NormSqr,
            input,
        } = node
        else {
            return Ok(Rewrite::Keep);
        };
        let mut builder = ReplacementFragment::new(context);
        let conj = builder.push(
            ExprNode::Unary {
                op: UnaryOp::Conj,
                input: *input,
            },
            ExprMetadata::new(ExprSourceKind::Unary),
        );
        builder.push(
            ExprNode::NaryMul {
                factors: vec![*input, conj],
            },
            metadata.clone(),
        );
        Ok(builder.into_rewrite())
    }
}

/// Simplifies and distributes complex conjugation.
#[derive(Copy, Clone, Debug, Default)]
pub struct ConjugationRule;

impl RewriteRule for ConjugationRule {
    fn name(&self) -> &'static str {
        "conjugation"
    }

    fn rewrite(
        &self,
        node: &ExprNode,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite> {
        let ExprNode::Unary {
            op: UnaryOp::Conj,
            input,
        } = node
        else {
            return Ok(Rewrite::Keep);
        };
        Ok(match context.node(*input) {
            Some(ExprNode::NaryMul { factors }) => {
                self.push_conjugated_product(factors, metadata, context)
            }
            Some(ExprNode::Binary {
                op: BinaryOp::Mul,
                lhs,
                rhs,
            }) => self.push_conjugated_product(&[*lhs, *rhs], metadata, context),
            Some(ExprNode::NaryAdd { terms }) => self.push_conjugated_sum(terms, metadata, context),
            Some(ExprNode::Binary {
                op: BinaryOp::Add,
                lhs,
                rhs,
            }) => self.push_conjugated_sum(&[*lhs, *rhs], metadata, context),
            Some(ExprNode::Unary {
                op: UnaryOp::Exp,
                input,
            }) => self.push_conjugated_exp(*input, metadata, context),
            Some(ExprNode::Unary {
                op: UnaryOp::Neg,
                input,
            }) => self.push_conjugated_neg(*input, metadata, context),
            _ => Rewrite::Keep,
        })
    }
}

impl ConjugationRule {
    fn push_conjugated_product(
        &self,
        factors: &[ExprId],
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> Rewrite {
        let mut builder = ReplacementFragment::new(context);
        let factors = factors
            .iter()
            .map(|factor| builder.conjugated_term(*factor))
            .collect();
        builder.push(ExprNode::NaryMul { factors }, metadata.clone());
        builder.into_rewrite()
    }

    fn push_conjugated_sum(
        &self,
        terms: &[ExprId],
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> Rewrite {
        let mut builder = ReplacementFragment::new(context);
        let terms = terms
            .iter()
            .map(|term| builder.conjugated_term(*term))
            .collect();
        builder.push(ExprNode::NaryAdd { terms }, metadata.clone());
        builder.into_rewrite()
    }

    fn push_conjugated_exp(
        &self,
        input: ExprId,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> Rewrite {
        let mut builder = ReplacementFragment::new(context);
        let input = builder.conjugated_term(input);
        builder.push(
            ExprNode::Unary {
                op: UnaryOp::Exp,
                input,
            },
            metadata.clone(),
        );
        builder.into_rewrite()
    }

    fn push_conjugated_neg(
        &self,
        input: ExprId,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> Rewrite {
        let mut builder = ReplacementFragment::new(context);
        let input = builder.conjugated_term(input);
        builder.push(
            ExprNode::Unary {
                op: UnaryOp::Neg,
                input,
            },
            metadata.clone(),
        );
        builder.into_rewrite()
    }
}

/// Simplifies matrix and vector construction/extraction operations.
#[derive(Copy, Clone, Debug, Default)]
pub struct MatrixVectorRule;

impl RewriteRule for MatrixVectorRule {
    fn name(&self) -> &'static str {
        "matrix-vector"
    }

    fn rewrite(
        &self,
        node: &ExprNode,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite> {
        match node {
            ExprNode::MatMul { lhs, rhs } => self.rewrite_matmul(*lhs, *rhs, metadata, context),
            ExprNode::MatVec { matrix, vector } => {
                self.rewrite_matvec(*matrix, *vector, metadata, context)
            }
            ExprNode::Dot { lhs, rhs } => self.rewrite_dot(*lhs, *rhs, metadata, context),
            ExprNode::Component { input, index } => {
                self.rewrite_component(*input, *index, metadata, context)
            }
            ExprNode::MatrixElement { input, row, col } => {
                self.rewrite_matrix_element(*input, *row, *col, metadata, context)
            }
            _ => Ok(Rewrite::Keep),
        }
    }
}

impl MatrixVectorRule {
    fn rewrite_component(
        &self,
        input: ExprId,
        index: usize,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite> {
        let Some(ExprNode::MatVec { matrix, vector }) = context.node(input) else {
            return Ok(Rewrite::Keep);
        };
        if !matches!(context.node(*matrix), Some(ExprNode::Matrix { .. }))
            || !matches!(context.node(*vector), Some(ExprNode::Vector { .. }))
        {
            return Ok(Rewrite::Keep);
        }
        let (Some((rows, cols)), Some(len)) = (
            Self::matrix_dims(context, *matrix),
            Self::vector_len(context, *vector),
        ) else {
            return Ok(Rewrite::Keep);
        };
        if index >= rows || cols != len || cols == 0 {
            return Ok(Rewrite::Keep);
        }

        let mut builder = ReplacementFragment::new(context);
        let terms = (0..cols)
            .map(|col| {
                let lhs = Self::matrix_element(*matrix, index, col, &mut builder);
                let rhs = Self::vector_element(*vector, col, &mut builder);
                builder.push(
                    ExprNode::NaryMul {
                        factors: vec![lhs, rhs],
                    },
                    ExprMetadata::new(ExprSourceKind::Binary),
                )
            })
            .collect();
        builder.push(ExprNode::NaryAdd { terms }, metadata.clone());
        self.cost_gated_fragment(
            ExprNode::Component { input, index },
            metadata,
            builder,
            context,
        )
    }

    fn rewrite_matrix_element(
        &self,
        input: ExprId,
        row: usize,
        col: usize,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite> {
        let Some(ExprNode::MatMul { lhs, rhs }) = context.node(input) else {
            return Ok(Rewrite::Keep);
        };
        if !matches!(context.node(*lhs), Some(ExprNode::Matrix { .. }))
            || !matches!(context.node(*rhs), Some(ExprNode::Matrix { .. }))
        {
            return Ok(Rewrite::Keep);
        }
        let (Some((lhs_rows, lhs_cols)), Some((rhs_rows, rhs_cols))) = (
            Self::matrix_dims(context, *lhs),
            Self::matrix_dims(context, *rhs),
        ) else {
            return Ok(Rewrite::Keep);
        };
        if row >= lhs_rows || col >= rhs_cols || lhs_cols != rhs_rows || lhs_cols == 0 {
            return Ok(Rewrite::Keep);
        }

        let mut builder = ReplacementFragment::new(context);
        let terms = (0..lhs_cols)
            .map(|inner| {
                let lhs = Self::matrix_element(*lhs, row, inner, &mut builder);
                let rhs = Self::matrix_element(*rhs, inner, col, &mut builder);
                builder.push(
                    ExprNode::NaryMul {
                        factors: vec![lhs, rhs],
                    },
                    ExprMetadata::new(ExprSourceKind::Binary),
                )
            })
            .collect();
        builder.push(ExprNode::NaryAdd { terms }, metadata.clone());
        self.cost_gated_fragment(
            ExprNode::MatrixElement { input, row, col },
            metadata,
            builder,
            context,
        )
    }

    fn rewrite_matmul(
        &self,
        lhs: ExprId,
        rhs: ExprId,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite> {
        if Self::matrix_is_identity(context, lhs) && Self::matrix_dims(context, rhs).is_some() {
            return Ok(alias_or_preserve(rhs, metadata, context));
        }
        if Self::matrix_is_identity(context, rhs) && Self::matrix_dims(context, lhs).is_some() {
            return Ok(alias_or_preserve(lhs, metadata, context));
        }

        let (Some((lhs_rows, lhs_cols)), Some((rhs_rows, rhs_cols))) = (
            Self::matrix_dims(context, lhs),
            Self::matrix_dims(context, rhs),
        ) else {
            return Ok(Rewrite::Keep);
        };
        if lhs_cols != rhs_rows {
            return Ok(Rewrite::Keep);
        }
        if Self::matrix_is_zero(context, lhs) || Self::matrix_is_zero(context, rhs) {
            return Ok(Self::zero_matrix_rewrite(
                lhs_rows, rhs_cols, metadata, context,
            ));
        }
        Ok(Rewrite::Keep)
    }

    fn rewrite_matvec(
        &self,
        matrix: ExprId,
        vector: ExprId,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite> {
        if Self::matrix_is_identity(context, matrix) && Self::vector_len(context, vector).is_some()
        {
            return Ok(alias_or_preserve(vector, metadata, context));
        }

        let (Some((rows, cols)), Some(len)) = (
            Self::matrix_dims(context, matrix),
            Self::vector_len(context, vector),
        ) else {
            return Ok(Rewrite::Keep);
        };
        if cols != len {
            return Ok(Rewrite::Keep);
        }
        if Self::matrix_is_zero(context, matrix) || Self::vector_is_zero(context, vector) {
            return Ok(Self::zero_vector_rewrite(rows, metadata, context));
        }
        self.expand_matvec(matrix, vector, rows, cols, metadata, context)
    }

    fn rewrite_dot(
        &self,
        lhs: ExprId,
        rhs: ExprId,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite> {
        let (Some(lhs_len), Some(rhs_len)) = (
            Self::vector_len(context, lhs),
            Self::vector_len(context, rhs),
        ) else {
            return Ok(Rewrite::Keep);
        };
        if lhs_len != rhs_len {
            return Ok(Rewrite::Keep);
        }
        if Self::vector_is_zero(context, lhs) || Self::vector_is_zero(context, rhs) {
            return Ok(Rewrite::Replace {
                node: ExprNode::RealConst(0.0),
                metadata: metadata.clone(),
            });
        }
        self.expand_dot(lhs, rhs, lhs_len, metadata, context)
    }

    fn expand_dot(
        &self,
        lhs: ExprId,
        rhs: ExprId,
        len: usize,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite> {
        let mut builder = ReplacementFragment::new(context);
        let terms = (0..len)
            .map(|index| {
                let lhs = Self::vector_element(lhs, index, &mut builder);
                let rhs = Self::vector_element(rhs, index, &mut builder);
                builder.push(
                    ExprNode::NaryMul {
                        factors: vec![lhs, rhs],
                    },
                    ExprMetadata::new(ExprSourceKind::Binary),
                )
            })
            .collect();
        builder.push(ExprNode::NaryAdd { terms }, metadata.clone());
        self.cost_gated_fragment(ExprNode::Dot { lhs, rhs }, metadata, builder, context)
    }

    fn expand_matvec(
        &self,
        matrix: ExprId,
        vector: ExprId,
        rows: usize,
        cols: usize,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite> {
        let mut builder = ReplacementFragment::new(context);
        let elements = (0..rows)
            .map(|row| {
                let terms = (0..cols)
                    .map(|col| {
                        let lhs = Self::matrix_element(matrix, row, col, &mut builder);
                        let rhs = Self::vector_element(vector, col, &mut builder);
                        builder.push(
                            ExprNode::NaryMul {
                                factors: vec![lhs, rhs],
                            },
                            ExprMetadata::new(ExprSourceKind::Binary),
                        )
                    })
                    .collect();
                builder.push(
                    ExprNode::NaryAdd { terms },
                    ExprMetadata::new(ExprSourceKind::Binary),
                )
            })
            .collect();
        builder.push(ExprNode::Vector { elements }, metadata.clone());
        self.cost_gated_fragment(
            ExprNode::MatVec { matrix, vector },
            metadata,
            builder,
            context,
        )
    }

    fn cost_gated_fragment(
        &self,
        original: ExprNode,
        metadata: &ExprMetadata,
        builder: ReplacementFragment<'_>,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite> {
        let Rewrite::ReplaceMany { nodes } = builder.into_rewrite() else {
            unreachable!("replacement builder always produces fragments")
        };
        let original_cost = context.local_node_cost(original, metadata.clone())?;
        let candidate_cost = context.local_fragment_cost(&nodes)?;
        if candidate_cost.is_better_than(&original_cost) {
            Ok(Rewrite::ReplaceMany { nodes })
        } else {
            Ok(Rewrite::Keep)
        }
    }

    fn zero_vector_rewrite(
        len: usize,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> Rewrite {
        let mut builder = ReplacementFragment::new(context);
        let zero = builder.push(
            ExprNode::RealConst(0.0),
            ExprMetadata::new(ExprSourceKind::Const),
        );
        builder.push(
            ExprNode::Vector {
                elements: vec![zero; len],
            },
            metadata.clone(),
        );
        builder.into_rewrite()
    }

    fn zero_matrix_rewrite(
        rows: usize,
        cols: usize,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> Rewrite {
        let mut builder = ReplacementFragment::new(context);
        let zero = builder.push(
            ExprNode::RealConst(0.0),
            ExprMetadata::new(ExprSourceKind::Const),
        );
        builder.push(
            ExprNode::Matrix {
                rows,
                cols,
                elements: vec![zero; rows * cols],
            },
            metadata.clone(),
        );
        builder.into_rewrite()
    }

    fn vector_len(context: &RewriteContext<'_>, id: ExprId) -> Option<usize> {
        match context.facts(id)?.value_kind {
            ValueKind::Vector { len } => Some(len),
            _ => None,
        }
    }

    fn matrix_dims(context: &RewriteContext<'_>, id: ExprId) -> Option<(usize, usize)> {
        match context.facts(id)?.value_kind {
            ValueKind::Matrix { rows, cols } => Some((rows, cols)),
            _ => None,
        }
    }

    fn vector_element(id: ExprId, index: usize, builder: &mut ReplacementFragment<'_>) -> ExprId {
        if let Some(ExprNode::Vector { elements }) = builder.context.node(id)
            && let Some(element) = elements.get(index).copied()
        {
            return element;
        }
        builder.push(
            ExprNode::Component { input: id, index },
            ExprMetadata::new(ExprSourceKind::Vector),
        )
    }

    fn matrix_element(
        id: ExprId,
        row: usize,
        col: usize,
        builder: &mut ReplacementFragment<'_>,
    ) -> ExprId {
        if let Some(ExprNode::Matrix { cols, elements, .. }) = builder.context.node(id)
            && let Some(element) = row
                .checked_mul(*cols)
                .and_then(|base| base.checked_add(col))
                .and_then(|index| elements.get(index))
                .copied()
        {
            return element;
        }
        builder.push(
            ExprNode::MatrixElement {
                input: id,
                row,
                col,
            },
            ExprMetadata::new(ExprSourceKind::Matrix),
        )
    }

    fn vector_is_zero(context: &RewriteContext<'_>, id: ExprId) -> bool {
        let Some(ExprNode::Vector { elements }) = context.node(id) else {
            return false;
        };
        elements
            .iter()
            .all(|element| context.node(*element).is_some_and(ExprNode::is_zero))
    }

    fn matrix_is_zero(context: &RewriteContext<'_>, id: ExprId) -> bool {
        let Some(ExprNode::Matrix { elements, .. }) = context.node(id) else {
            return false;
        };
        elements
            .iter()
            .all(|element| context.node(*element).is_some_and(ExprNode::is_zero))
    }

    fn matrix_is_identity(context: &RewriteContext<'_>, id: ExprId) -> bool {
        let Some(ExprNode::Matrix {
            rows,
            cols,
            elements,
        }) = context.node(id)
        else {
            return false;
        };
        if rows != cols || elements.len() != rows * cols {
            return false;
        }
        for row in 0..*rows {
            for col in 0..*cols {
                let Some(node) = context.node(elements[row * cols + col]) else {
                    return false;
                };
                if row == col {
                    if !ExprNode::is_one(node) {
                        return false;
                    }
                } else if !ExprNode::is_zero(node) {
                    return false;
                }
            }
        }
        true
    }
}

/// Propagates known real, imaginary, and complex facts through expressions.
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
            let old_id = ExprId::from_index(old_index);
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
    let id = ExprId::from_index(nodes.len());
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
    let new_id = ExprId::from_index(nodes.len());
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
            emit_canonical_associative(op, operands, node_metadata, nodes, metadata, keys)
        }
        ExprNode::NaryAdd { terms } => {
            emit_canonical_associative(BinaryOp::Add, terms, node_metadata, nodes, metadata, keys)
        }
        ExprNode::NaryMul { factors } => {
            emit_canonical_associative(BinaryOp::Mul, factors, node_metadata, nodes, metadata, keys)
        }
        node => intern_canonical_node(node, node_metadata, nodes, metadata, keys),
    }
}

fn emit_canonical_associative(
    op: BinaryOp,
    operands: Vec<ExprId>,
    node_metadata: ExprMetadata,
    nodes: &mut Vec<ExprNode>,
    metadata: &mut Vec<ExprMetadata>,
    keys: &mut HashMap<StructuralKey, ExprId>,
) -> ExprId {
    let mut flattened = Vec::new();
    for operand in operands {
        collect_associative_operands(op, operand, nodes, &mut flattened);
    }
    flattened.sort_by(|lhs, rhs| {
        operand_sort_key(op, *lhs, nodes).cmp(&operand_sort_key(op, *rhs, nodes))
    });

    match flattened.as_slice() {
        [] => intern_canonical_node(
            identity_for_associative_op(op),
            node_metadata,
            nodes,
            metadata,
            keys,
        ),
        [operand] => *operand,
        _ => {
            let node = match op {
                BinaryOp::Add => ExprNode::NaryAdd { terms: flattened },
                BinaryOp::Mul => ExprNode::NaryMul { factors: flattened },
                BinaryOp::Sub | BinaryOp::Div | BinaryOp::Atan2 => {
                    unreachable!("only associative ops are canonicalized")
                }
            };
            intern_canonical_node(node, node_metadata, nodes, metadata, keys)
        }
    }
}

fn identity_for_associative_op(op: BinaryOp) -> ExprNode {
    match op {
        BinaryOp::Add => ExprNode::RealConst(0.0),
        BinaryOp::Mul => ExprNode::RealConst(1.0),
        BinaryOp::Sub | BinaryOp::Div | BinaryOp::Atan2 => {
            unreachable!("only associative ops have identities")
        }
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

    let id = ExprId::from_index(nodes.len());
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
    match nodes.get(id.index()) {
        Some(ExprNode::Binary {
            op: child_op,
            lhs,
            rhs,
        }) if *child_op == op => {
            collect_associative_operands(op, *lhs, nodes, operands);
            collect_associative_operands(op, *rhs, nodes, operands);
        }
        Some(ExprNode::NaryAdd { terms }) if op == BinaryOp::Add => {
            for term in terms {
                collect_associative_operands(op, *term, nodes, operands);
            }
        }
        Some(ExprNode::NaryMul { factors }) if op == BinaryOp::Mul => {
            for factor in factors {
                collect_associative_operands(op, *factor, nodes, operands);
            }
        }
        _ => operands.push(id),
    }
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
    let is_negative_add_operand = || match node {
        ExprNode::RealConst(value) => *value < 0.0,
        ExprNode::NaryMul { factors } => factors.iter().any(|factor| {
            matches!(
                nodes.get(factor.index()),
                Some(ExprNode::RealConst(value)) if *value < 0.0
            )
        }),
        _ => false,
    };
    let category = match (op, node) {
        (BinaryOp::Add, ExprNode::RealConst(value)) if *value >= 0.0 => 0,
        (BinaryOp::Add, _) if !is_negative_add_operand() => 1,
        (BinaryOp::Add, _) => 2,
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
    EventScalar(String),
    EventP4Component {
        name: String,
        component: P4Component,
    },
    Unary {
        op: UnaryKey,
        input: usize,
    },
    Binary {
        op: BinaryOp,
        lhs: usize,
        rhs: usize,
    },
    NaryAdd {
        terms: Vec<usize>,
    },
    NaryMul {
        factors: Vec<usize>,
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
            ExprNode::EventScalar(name) => Self::EventScalar(name.to_string()),
            ExprNode::EventP4Component { name, component } => Self::EventP4Component {
                name: name.to_string(),
                component: *component,
            },
            ExprNode::Unary { op, input } => Self::Unary {
                op: UnaryKey::from(*op),
                input: input.index(),
            },
            ExprNode::Binary { op, lhs, rhs } => Self::Binary {
                op: *op,
                lhs: lhs.index(),
                rhs: rhs.index(),
            },
            ExprNode::NaryAdd { terms } => Self::NaryAdd {
                terms: terms.iter().map(|id| id.index()).collect(),
            },
            ExprNode::NaryMul { factors } => Self::NaryMul {
                factors: factors.iter().map(|id| id.index()).collect(),
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
        | ExprNode::EventScalar(_)
        | ExprNode::EventP4Component { .. } => node.clone(),
        ExprNode::Unary { op, input } => ExprNode::Unary {
            op: *op,
            input: old_to_new[input.index()],
        },
        ExprNode::Binary { op, lhs, rhs } => ExprNode::Binary {
            op: *op,
            lhs: old_to_new[lhs.index()],
            rhs: old_to_new[rhs.index()],
        },
        ExprNode::NaryAdd { terms } => ExprNode::NaryAdd {
            terms: terms.iter().map(|id| old_to_new[id.index()]).collect(),
        },
        ExprNode::NaryMul { factors } => ExprNode::NaryMul {
            factors: factors.iter().map(|id| old_to_new[id.index()]).collect(),
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
        | ExprNode::EventScalar(_)
        | ExprNode::EventP4Component { .. } => node.clone(),
        ExprNode::Unary { op, input } => ExprNode::Unary {
            op: *op,
            input: old_to_new[input.index()].expect("child was compacted first"),
        },
        ExprNode::Binary { op, lhs, rhs } => ExprNode::Binary {
            op: *op,
            lhs: old_to_new[lhs.index()].expect("child was compacted first"),
            rhs: old_to_new[rhs.index()].expect("child was compacted first"),
        },
        ExprNode::NaryAdd { terms } => ExprNode::NaryAdd {
            terms: terms
                .iter()
                .map(|id| old_to_new[id.index()].expect("child was compacted first"))
                .collect(),
        },
        ExprNode::NaryMul { factors } => ExprNode::NaryMul {
            factors: factors
                .iter()
                .map(|id| old_to_new[id.index()].expect("child was compacted first"))
                .collect(),
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
        | ExprNode::EventScalar(_)
        | ExprNode::EventP4Component { .. } => Vec::new(),
        ExprNode::Unary { input, .. }
        | ExprNode::Component { input, .. }
        | ExprNode::MatrixElement { input, .. } => vec![*input],
        ExprNode::Complex { re, im } => vec![*re, *im],
        ExprNode::NaryAdd { terms } => terms.clone(),
        ExprNode::NaryMul { factors } => factors.clone(),
        ExprNode::Binary { lhs, rhs, .. }
        | ExprNode::MatMul { lhs, rhs }
        | ExprNode::Dot { lhs, rhs } => vec![*lhs, *rhs],
        ExprNode::MatVec { matrix, vector } => vec![*matrix, *vector],
        ExprNode::Solve { matrix, rhs } => vec![*matrix, *rhs],
        ExprNode::Vector { elements } | ExprNode::Matrix { elements, .. } => elements.clone(),
    }
}
