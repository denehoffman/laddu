use std::{
    fmt,
    hash::{Hash, Hasher},
};

use laddu_expr::ExprGraph;

use crate::{CompileResult, cost::OptimizationCost};

use super::{CanonicalCsePass, CostGatePass, OptimizationPass, OptimizationPipeline, RewritePass};

const DEFAULT_MAX_ITERATIONS: usize = 16;

impl OptimizationPipeline {
    pub(crate) fn normalization_analysis_passes() -> Self {
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
            .with_max_iterations(DEFAULT_MAX_ITERATIONS)
    }

    pub(crate) fn normalization_target_lowering_passes() -> Self {
        Self::new().with_pass(CostGatePass::new(Self::norm_sqr_expansion_candidate()))
    }

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
    ///
    /// # Errors
    ///
    /// Returns [`CompileError`](crate::CompileError) when any configured pass
    /// cannot transform the graph.
    pub fn run(&self, mut graph: ExprGraph) -> CompileResult<ExprGraph> {
        for _ in 0..self.max_iterations {
            let previous = graph_fingerprint(&graph);
            for pass in &self.passes {
                graph = pass.run(graph)?;
            }
            if previous == graph_fingerprint(&graph) {
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

pub(super) fn graph_fingerprint(graph: &ExprGraph) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    graph.root().index().hash(&mut hasher);
    graph.nodes().len().hash(&mut hasher);
    for node in graph.nodes() {
        node.structural_key().hash(&mut hasher);
    }
    hasher.finish()
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

#[cfg(test)]
mod tests {
    use super::*;

    fn pass_names(pipeline: &OptimizationPipeline) -> Vec<&'static str> {
        pipeline.passes.iter().map(|pass| pass.name()).collect()
    }

    #[test]
    fn built_in_pipeline_order_is_stable() {
        assert_eq!(
            pass_names(&OptimizationPipeline::with_default_passes()),
            [
                "simplify",
                "canonical-cse",
                "normalize-add-mul",
                "canonical-cse",
                "combine-like-terms",
                "canonical-cse",
                "factor-common-products",
                "normalize-add-mul",
                "canonical-cse",
                "exponential",
                "simplify",
                "cost-gate",
            ]
        );
        assert_eq!(
            pass_names(&OptimizationPipeline::normalization_analysis_passes()),
            [
                "simplify",
                "canonical-cse",
                "normalize-add-mul",
                "canonical-cse",
                "combine-like-terms",
                "canonical-cse",
                "factor-common-products",
                "normalize-add-mul",
                "canonical-cse",
                "exponential",
                "simplify",
            ]
        );
        assert_eq!(
            pass_names(&OptimizationPipeline::norm_sqr_expansion_candidate()),
            [
                "norm-sqr-expansion",
                "canonical-cse",
                "conjugation",
                "canonical-cse",
                "normalize-add-mul",
                "canonical-cse",
                "combine-like-terms",
                "canonical-cse",
                "factor-common-products",
                "normalize-add-mul",
                "canonical-cse",
                "exponential",
                "simplify",
            ]
        );
    }
}
