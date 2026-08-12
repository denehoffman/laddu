use laddu_expr::{ExprGraph, ExprId, ExprMetadata, ExprNode};

use crate::{CompileResult, facts::NodeFacts};

mod canonical;
mod pipeline;
mod rewrite;
mod rules;

/// Ordered optimization passes repeatedly applied until convergence.
pub struct OptimizationPipeline {
    passes: Vec<Box<dyn OptimizationPass>>,
    max_iterations: usize,
}

/// Result of one optimization pass with explicit convergence state.
pub struct OptimizationPassOutcome {
    /// Transformed graph.
    pub graph: ExprGraph,
    /// Whether the graph's executable structure changed.
    pub changed: bool,
}

/// One whole-graph transformation in an [`OptimizationPipeline`].
pub trait OptimizationPass: Send + Sync {
    /// Returns a stable diagnostic name.
    fn name(&self) -> &'static str;

    /// Transforms `graph`.
    ///
    /// # Errors
    ///
    /// Returns [`CompileError`](crate::CompileError) when the graph cannot be transformed.
    fn run(&self, graph: ExprGraph) -> CompileResult<ExprGraph>;

    /// Runs this pass and reports structural change without cloning the graph.
    ///
    /// Custom passes receive fingerprint-based change detection by default and
    /// may override this method when they can report change directly.
    ///
    /// # Errors
    ///
    /// Returns [`CompileError`](crate::CompileError) when this pass cannot
    /// transform the graph.
    fn run_with_change(&self, graph: ExprGraph) -> CompileResult<OptimizationPassOutcome> {
        let before = pipeline::graph_fingerprint(&graph);
        let graph = self.run(graph)?;
        let changed = before != pipeline::graph_fingerprint(&graph);
        Ok(OptimizationPassOutcome { graph, changed })
    }
}

/// Runs a candidate pipeline only when it strictly improves static cost.
pub struct CostGatePass {
    name: &'static str,
    candidate: OptimizationPipeline,
}

/// Canonical common-subexpression-elimination pass.
#[derive(Copy, Clone, Debug, Default)]
pub struct CanonicalCsePass;

/// Whole-graph pass applying local rewrite rules in order.
pub struct RewritePass {
    name: &'static str,
    rules: Vec<Box<dyn RewriteRule>>,
}

/// Local transformation of one expression node.
pub trait RewriteRule: Send + Sync {
    /// Returns a stable diagnostic name.
    fn name(&self) -> &'static str;

    /// Chooses how to emit the current node.
    ///
    /// # Errors
    ///
    /// Returns [`CompileError`](crate::CompileError) when the node cannot be
    /// rewritten into a valid expression.
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

/// Folds scalar operations whose operands are constants.
#[derive(Copy, Clone, Debug, Default)]
pub struct ConstantFoldScalarRule;

/// Simplifies scalar and linear-algebra identities.
#[derive(Copy, Clone, Debug, Default)]
pub struct AlgebraicIdentityRule;

/// Reduces squared norms when facts make a cheaper equivalent available.
#[derive(Copy, Clone, Debug, Default)]
pub struct NormSqrReductionRule;

/// Simplifies trigonometric identities.
#[derive(Copy, Clone, Debug, Default)]
pub struct TrigIdentityRule;

/// Combines additive terms with matching products.
#[derive(Copy, Clone, Debug, Default)]
pub struct CombineLikeTermsRule;

/// Factors products shared by additive terms.
#[derive(Copy, Clone, Debug, Default)]
pub struct FactorCommonProductRule;

/// Rewrites compatible exponential expressions.
#[derive(Copy, Clone, Debug, Default)]
pub struct ExponentialRule;

/// Expands squared norms into forms that may expose common subexpressions.
#[derive(Copy, Clone, Debug, Default)]
pub struct NormSqrExpansionRule;

/// Pushes conjugation through supported scalar expression forms.
#[derive(Copy, Clone, Debug, Default)]
pub struct ConjugationRule;

/// Simplifies matrix and vector construction/extraction operations.
#[derive(Copy, Clone, Debug, Default)]
pub struct MatrixVectorRule;

/// Propagates known real, imaginary, and complex facts through expressions.
#[derive(Copy, Clone, Debug, Default)]
pub struct ComplexFactRule;
