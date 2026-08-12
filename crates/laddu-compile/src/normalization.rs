use laddu_expr::{
    BinaryOp, ExprGraph, ExprId, ExprMetadata, ExprNode, ExprSourceKind, UnaryOp, ValueKind,
};
use num::complex::Complex64;

use crate::{CompileError, CompileResult, CompiledModel, GraphFacts, graph_utils::compact_to_root};

const DEFAULT_EXPANSION_BUDGET: usize = 4_096;

/// Compiler-selected family of accepted-normalization implementation.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum NormalizationStrategy {
    /// Coherent groups represented by packed Hermitian statistics.
    Hermitian,
    /// A general exact sum of parameter coefficients times event bases.
    LinearStatistics,
    /// Exact sufficient statistics plus a nonseparable additive residual.
    Hybrid,
    /// Ordinary event reduction.
    General,
}

/// Stable reason that compiler-native normalization was not selected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NormalizationFallbackReason {
    /// The model root is not a real scalar.
    NonScalarIntensity,
    /// An operation mixed parameter and event dependence without an exact rule.
    UnsupportedMixedOperation {
        /// Offending optimized graph node.
        node: ExprId,
        /// Operation category.
        operation: &'static str,
    },
    /// Exact symbolic distribution exceeded the compiler expansion budget.
    ExpansionBudgetExceeded {
        /// Maximum permitted number of sufficient-statistic terms.
        budget: usize,
    },
}

/// Stable summary of compiler-native normalization analysis.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormalizationDiagnostics {
    strategy: NormalizationStrategy,
    basis_count: usize,
    coherent_group_count: usize,
    has_residual: bool,
    fallback_reason: Option<NormalizationFallbackReason>,
}

impl NormalizationDiagnostics {
    /// Returns the compiler-selected candidate family.
    pub fn strategy(&self) -> NormalizationStrategy {
        self.strategy
    }

    /// Returns the number of event-basis statistics in the candidate.
    pub fn basis_count(&self) -> usize {
        self.basis_count
    }

    /// Returns the number of recognized coherent squared-norm groups.
    pub fn coherent_group_count(&self) -> usize {
        self.coherent_group_count
    }

    /// Returns whether evaluation retains a general additive residual.
    pub fn has_residual(&self) -> bool {
        self.has_residual
    }

    /// Returns the structured general-path reason, when present.
    pub fn fallback_reason(&self) -> Option<&NormalizationFallbackReason> {
        self.fallback_reason.as_ref()
    }
}

#[derive(Copy, Clone, Debug)]
struct SeparableTerm {
    coefficient: ExprId,
    basis: ExprId,
}

/// Exact compiler artifact consumed by execution backends.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct NormalizationPlan {
    graph: ExprGraph,
    terms: Vec<SeparableTerm>,
    residual: Option<ExprId>,
    diagnostics: NormalizationDiagnostics,
    proven_nonnegative: bool,
}

impl NormalizationPlan {
    pub(crate) fn analyze_disabled(graph: &ExprGraph) -> Self {
        Self::general(
            graph,
            NormalizationFallbackReason::UnsupportedMixedOperation {
                node: graph.root(),
                operation: "normalization analysis disabled",
            },
        )
    }

    pub(crate) fn analyze(graph: &ExprGraph, facts: &GraphFacts) -> Self {
        if !matches!(
            facts.get(graph.root()).map(|facts| facts.value_kind),
            Some(ValueKind::Real | ValueKind::Complex)
        ) {
            return Self::general(graph, NormalizationFallbackReason::NonScalarIntensity);
        }

        let mut analyzer = Analyzer::new(graph, facts, DEFAULT_EXPANSION_BUDGET);
        let mut terms = Vec::new();
        let mut residuals = Vec::new();
        for root in analyzer.additive_roots(graph.root()) {
            match analyzer.decompose(root) {
                Ok(mut extracted) => terms.append(&mut extracted),
                Err(reason) => {
                    analyzer.last_reason = Some(reason);
                    residuals.push(root);
                }
            }
        }

        if terms.is_empty() {
            return Self::general(
                graph,
                analyzer.last_reason.unwrap_or(
                    NormalizationFallbackReason::UnsupportedMixedOperation {
                        node: graph.root(),
                        operation: "root",
                    },
                ),
            );
        }

        let residual = analyzer.sum_roots(&residuals);
        let strategy = if residual.is_some() {
            NormalizationStrategy::Hybrid
        } else if analyzer.coherent_groups > 0 {
            NormalizationStrategy::Hermitian
        } else {
            NormalizationStrategy::LinearStatistics
        };
        let diagnostics = NormalizationDiagnostics {
            strategy,
            basis_count: terms.len(),
            coherent_group_count: analyzer.coherent_groups,
            has_residual: residual.is_some(),
            fallback_reason: analyzer.last_reason.clone(),
        };
        Self {
            graph: analyzer.finish(),
            terms,
            residual,
            diagnostics,
            proven_nonnegative: proves_nonnegative(graph, facts, graph.root()),
        }
    }

    fn general(graph: &ExprGraph, reason: NormalizationFallbackReason) -> Self {
        Self {
            graph: graph.clone(),
            terms: Vec::new(),
            residual: None,
            diagnostics: NormalizationDiagnostics {
                strategy: NormalizationStrategy::General,
                basis_count: 0,
                coherent_group_count: 0,
                has_residual: false,
                fallback_reason: Some(reason),
            },
            proven_nonnegative: false,
        }
    }

    /// Returns stable compiler diagnostics.
    pub fn diagnostics(&self) -> &NormalizationDiagnostics {
        &self.diagnostics
    }

    /// Returns whether the source graph proves every event value nonnegative.
    pub fn proven_nonnegative(&self) -> bool {
        self.proven_nonnegative
    }

    /// Builds one event-only compiled model per statistic.
    ///
    /// # Errors
    ///
    /// Returns a compiler error if an extracted basis graph cannot be lowered.
    pub fn basis_models(&self) -> CompileResult<Vec<CompiledModel>> {
        self.terms
            .iter()
            .map(|term| {
                CompiledModel::from_graph_without_normalization(compact_to_root(
                    &self.graph,
                    term.basis,
                )?)
            })
            .collect()
    }

    /// Builds the parameter-only scalar contraction for accumulated statistics.
    ///
    /// # Errors
    ///
    /// Returns a compiler error if the statistic count is incompatible with
    /// this plan or the coefficient graph cannot be lowered.
    pub fn evaluator_model(&self, statistics: &[Complex64]) -> CompileResult<CompiledModel> {
        if statistics.len() != self.terms.len() {
            return Err(CompileError::InvalidExecutablePlan(format!(
                "normalization expected {} statistics, got {}",
                self.terms.len(),
                statistics.len()
            )));
        }
        let mut nodes = self.graph.nodes().to_vec();
        let mut metadata = graph_metadata(&self.graph);
        let mut products = Vec::with_capacity(self.terms.len());
        for (term, statistic) in self.terms.iter().zip(statistics) {
            let constant = push_node(
                &mut nodes,
                &mut metadata,
                ExprNode::from_folded_const(*statistic),
                ExprSourceKind::Const,
            );
            products.push(push_node(
                &mut nodes,
                &mut metadata,
                ExprNode::NaryMul {
                    factors: vec![constant, term.coefficient],
                },
                ExprSourceKind::Binary,
            ));
        }
        let sum = push_node(
            &mut nodes,
            &mut metadata,
            ExprNode::NaryAdd { terms: products },
            ExprSourceKind::Binary,
        );
        let root = push_node(
            &mut nodes,
            &mut metadata,
            ExprNode::Unary {
                op: UnaryOp::Real,
                input: sum,
            },
            ExprSourceKind::Unary,
        );
        let graph = ExprGraph::from_parts(root, nodes, metadata)?;
        CompiledModel::from_graph_without_normalization(compact_to_root(&graph, root)?)
    }

    /// Builds the nonseparable residual model, when present.
    ///
    /// # Errors
    ///
    /// Returns a compiler error if the residual graph cannot be lowered.
    pub fn residual_model(&self) -> CompileResult<Option<CompiledModel>> {
        self.residual
            .map(|root| {
                CompiledModel::from_graph_without_normalization(compact_to_root(&self.graph, root)?)
            })
            .transpose()
    }
}

fn proves_nonnegative(graph: &ExprGraph, facts: &GraphFacts, id: ExprId) -> bool {
    match graph.node(id).expect("normalization node exists") {
        ExprNode::RealConst(value) => value.is_finite() && *value >= 0.0,
        ExprNode::ComplexConst(value) => value.im == 0.0 && value.re.is_finite() && value.re >= 0.0,
        ExprNode::Unary {
            op: UnaryOp::NormSqr,
            ..
        } => true,
        ExprNode::Unary {
            op: UnaryOp::Real,
            input,
        } => proves_nonnegative(graph, facts, *input),
        ExprNode::Unary {
            op: UnaryOp::PowI(power),
            input,
        } => {
            *power >= 0
                && power % 2 == 0
                && facts
                    .get(*input)
                    .is_some_and(|facts| facts.value_kind == ValueKind::Real)
        }
        ExprNode::Binary {
            op: BinaryOp::Add,
            lhs,
            rhs,
        } => proves_nonnegative(graph, facts, *lhs) && proves_nonnegative(graph, facts, *rhs),
        ExprNode::Binary {
            op: BinaryOp::Mul,
            lhs,
            rhs,
        } => proves_nonnegative(graph, facts, *lhs) && proves_nonnegative(graph, facts, *rhs),
        ExprNode::NaryAdd { terms } => terms
            .iter()
            .all(|term| proves_nonnegative(graph, facts, *term)),
        ExprNode::NaryMul { factors } => factors
            .iter()
            .all(|factor| proves_nonnegative(graph, facts, *factor)),
        _ => false,
    }
}

struct Analyzer<'a> {
    facts: &'a GraphFacts,
    nodes: Vec<ExprNode>,
    metadata: Vec<ExprMetadata>,
    one: ExprId,
    budget: usize,
    coherent_groups: usize,
    last_reason: Option<NormalizationFallbackReason>,
}

impl<'a> Analyzer<'a> {
    fn new(graph: &ExprGraph, facts: &'a GraphFacts, budget: usize) -> Self {
        let mut nodes = graph.nodes().to_vec();
        let mut metadata = graph_metadata(graph);
        let one = push_node(
            &mut nodes,
            &mut metadata,
            ExprNode::RealConst(1.0),
            ExprSourceKind::Const,
        );
        Self {
            facts,
            nodes,
            metadata,
            one,
            budget,
            coherent_groups: 0,
            last_reason: None,
        }
    }

    fn finish(&self) -> ExprGraph {
        ExprGraph::from_parts(
            ExprId::from_index(0),
            self.nodes.clone(),
            self.metadata.clone(),
        )
        .expect("augmented normalization graph is valid")
    }

    fn dependency(&self, id: ExprId) -> crate::DependencyFacts {
        if id.index() < self.facts.nodes().len() {
            return self.facts.get(id).expect("facts are complete").dependency;
        }
        self.nodes[id.index()].children().fold(
            crate::DependencyFacts::per_compile(),
            |dependency, child| dependency.union(self.dependency(child)),
        )
    }

    fn additive_roots(&self, root: ExprId) -> Vec<ExprId> {
        match &self.nodes[root.index()] {
            ExprNode::NaryAdd { terms } => terms.clone(),
            ExprNode::Binary {
                op: BinaryOp::Add,
                lhs,
                rhs,
            } => {
                let mut roots = self.additive_roots(*lhs);
                roots.extend(self.additive_roots(*rhs));
                roots
            }
            _ => vec![root],
        }
    }

    fn decompose(&mut self, id: ExprId) -> Result<Vec<SeparableTerm>, NormalizationFallbackReason> {
        let dependency = self.dependency(id);
        if !dependency.depends_on_event {
            return Ok(vec![SeparableTerm {
                coefficient: id,
                basis: self.one,
            }]);
        }
        if !dependency.depends_on_free_params {
            return Ok(vec![SeparableTerm {
                coefficient: self.one,
                basis: id,
            }]);
        }

        match self.nodes[id.index()].clone() {
            ExprNode::Binary { op, lhs, rhs } => self.decompose_binary(id, op, lhs, rhs),
            ExprNode::NaryAdd { terms } => {
                let mut result = Vec::new();
                for term in terms {
                    result.extend(self.decompose(term)?);
                    self.ensure_budget(&result)?;
                }
                Ok(result)
            }
            ExprNode::NaryMul { factors } => {
                let mut result = vec![SeparableTerm {
                    coefficient: self.one,
                    basis: self.one,
                }];
                for factor in factors {
                    let factor_terms = self.decompose(factor)?;
                    result = self.multiply_terms(&result, &factor_terms)?;
                }
                Ok(result)
            }
            ExprNode::Unary { op, input } => self.decompose_unary(id, op, input),
            _ => Err(self.unsupported(id, "structured mixed operation")),
        }
    }

    fn decompose_binary(
        &mut self,
        id: ExprId,
        op: BinaryOp,
        lhs: ExprId,
        rhs: ExprId,
    ) -> Result<Vec<SeparableTerm>, NormalizationFallbackReason> {
        match op {
            BinaryOp::Add => {
                let mut terms = self.decompose(lhs)?;
                terms.extend(self.decompose(rhs)?);
                self.ensure_budget(&terms)?;
                Ok(terms)
            }
            BinaryOp::Sub => {
                let mut terms = self.decompose(lhs)?;
                for mut term in self.decompose(rhs)? {
                    term.coefficient = self.unary(UnaryOp::Neg, term.coefficient);
                    terms.push(term);
                }
                self.ensure_budget(&terms)?;
                Ok(terms)
            }
            BinaryOp::Mul => {
                let left = self.decompose(lhs)?;
                let right = self.decompose(rhs)?;
                self.multiply_terms(&left, &right)
            }
            BinaryOp::Div => {
                let denominator = self.dependency(rhs);
                let mut terms = self.decompose(lhs)?;
                if !denominator.depends_on_event {
                    for term in &mut terms {
                        term.coefficient = self.binary(BinaryOp::Div, term.coefficient, rhs);
                    }
                    Ok(terms)
                } else if !denominator.depends_on_free_params {
                    for term in &mut terms {
                        term.basis = self.binary(BinaryOp::Div, term.basis, rhs);
                    }
                    Ok(terms)
                } else {
                    Err(self.unsupported(id, "mixed division"))
                }
            }
            BinaryOp::Atan2 => Err(self.unsupported(id, "atan2")),
        }
    }

    fn decompose_unary(
        &mut self,
        id: ExprId,
        op: UnaryOp,
        input: ExprId,
    ) -> Result<Vec<SeparableTerm>, NormalizationFallbackReason> {
        match op {
            UnaryOp::Neg => {
                let mut terms = self.decompose(input)?;
                for term in &mut terms {
                    term.coefficient = self.unary(UnaryOp::Neg, term.coefficient);
                }
                Ok(terms)
            }
            UnaryOp::Conj => {
                let mut terms = self.decompose(input)?;
                for term in &mut terms {
                    term.coefficient = self.unary(UnaryOp::Conj, term.coefficient);
                    term.basis = self.unary(UnaryOp::Conj, term.basis);
                }
                Ok(terms)
            }
            UnaryOp::NormSqr => {
                self.coherent_groups += 1;
                let terms = self.decompose(input)?;
                let packed_len = terms.len().saturating_mul(terms.len().saturating_add(1)) / 2;
                if packed_len > self.budget {
                    return Err(NormalizationFallbackReason::ExpansionBudgetExceeded {
                        budget: self.budget,
                    });
                }
                let two = self.constant(Complex64::new(2.0, 0.0));
                let mut packed = Vec::with_capacity(packed_len);
                for (row, left) in terms.iter().enumerate() {
                    for (column, right) in terms.iter().enumerate().skip(row) {
                        let right_coefficient = self.unary(UnaryOp::Conj, right.coefficient);
                        let right_basis = self.unary(UnaryOp::Conj, right.basis);
                        let mut coefficient = self.product(&[left.coefficient, right_coefficient]);
                        if column != row {
                            coefficient = self.product(&[two, coefficient]);
                        }
                        packed.push(SeparableTerm {
                            coefficient,
                            basis: self.product(&[left.basis, right_basis]),
                        });
                    }
                }
                Ok(packed)
            }
            UnaryOp::PowI(power) if power >= 0 => {
                let base = self.decompose(input)?;
                let mut result = vec![SeparableTerm {
                    coefficient: self.one,
                    basis: self.one,
                }];
                for _ in 0..power {
                    result = self.multiply_terms(&result, &base)?;
                }
                Ok(result)
            }
            UnaryOp::Real | UnaryOp::Imag => self.decompose_projection(op, input),
            UnaryOp::Sqrt
            | UnaryOp::Exp
            | UnaryOp::Sin
            | UnaryOp::Cos
            | UnaryOp::Log
            | UnaryOp::PowI(_) => Err(self.unsupported(id, "nonlinear unary operation")),
        }
    }

    fn decompose_projection(
        &mut self,
        op: UnaryOp,
        input: ExprId,
    ) -> Result<Vec<SeparableTerm>, NormalizationFallbackReason> {
        let terms = self.decompose(input)?;
        let mut result = Vec::with_capacity(terms.len() * 2);
        let factor = if op == UnaryOp::Real {
            Complex64::new(0.5, 0.0)
        } else {
            Complex64::new(0.0, -0.5)
        };
        let conjugate_factor = if op == UnaryOp::Real { factor } else { -factor };
        let factor = self.constant(factor);
        let conjugate_factor = self.constant(conjugate_factor);
        for term in terms {
            let coefficient = self.product(&[factor, term.coefficient]);
            result.push(SeparableTerm {
                coefficient,
                basis: term.basis,
            });
            let conjugated_coefficient = self.unary(UnaryOp::Conj, term.coefficient);
            let conjugated_basis = self.unary(UnaryOp::Conj, term.basis);
            let coefficient = self.product(&[conjugate_factor, conjugated_coefficient]);
            result.push(SeparableTerm {
                coefficient,
                basis: conjugated_basis,
            });
        }
        self.ensure_budget(&result)?;
        Ok(result)
    }

    fn multiply_terms(
        &mut self,
        lhs: &[SeparableTerm],
        rhs: &[SeparableTerm],
    ) -> Result<Vec<SeparableTerm>, NormalizationFallbackReason> {
        let count = lhs.len().saturating_mul(rhs.len());
        if count > self.budget {
            return Err(NormalizationFallbackReason::ExpansionBudgetExceeded {
                budget: self.budget,
            });
        }
        let mut result = Vec::with_capacity(count);
        for lhs in lhs {
            for rhs in rhs {
                let coefficient = self.product(&[lhs.coefficient, rhs.coefficient]);
                let basis = self.product(&[lhs.basis, rhs.basis]);
                result.push(SeparableTerm { coefficient, basis });
            }
        }
        Ok(result)
    }

    fn ensure_budget(&self, terms: &[SeparableTerm]) -> Result<(), NormalizationFallbackReason> {
        if terms.len() > self.budget {
            Err(NormalizationFallbackReason::ExpansionBudgetExceeded {
                budget: self.budget,
            })
        } else {
            Ok(())
        }
    }

    fn unsupported(&self, node: ExprId, operation: &'static str) -> NormalizationFallbackReason {
        NormalizationFallbackReason::UnsupportedMixedOperation { node, operation }
    }

    fn constant(&mut self, value: Complex64) -> ExprId {
        self.push(ExprNode::from_folded_const(value), ExprSourceKind::Const)
    }

    fn unary(&mut self, op: UnaryOp, input: ExprId) -> ExprId {
        self.push(ExprNode::Unary { op, input }, ExprSourceKind::Unary)
    }

    fn binary(&mut self, op: BinaryOp, lhs: ExprId, rhs: ExprId) -> ExprId {
        self.push(ExprNode::Binary { op, lhs, rhs }, ExprSourceKind::Binary)
    }

    fn product(&mut self, factors: &[ExprId]) -> ExprId {
        match factors {
            [] => self.one,
            [only] => *only,
            _ => self.push(
                ExprNode::NaryMul {
                    factors: factors.to_vec(),
                },
                ExprSourceKind::Binary,
            ),
        }
    }

    fn sum_roots(&mut self, roots: &[ExprId]) -> Option<ExprId> {
        match roots {
            [] => None,
            [only] => Some(*only),
            _ => Some(self.push(
                ExprNode::NaryAdd {
                    terms: roots.to_vec(),
                },
                ExprSourceKind::Binary,
            )),
        }
    }

    fn push(&mut self, node: ExprNode, source: ExprSourceKind) -> ExprId {
        push_node(&mut self.nodes, &mut self.metadata, node, source)
    }
}

fn graph_metadata(graph: &ExprGraph) -> Vec<ExprMetadata> {
    (0..graph.nodes().len())
        .map(|index| {
            graph
                .metadata(ExprId::from_index(index))
                .expect("normalization graph metadata is complete")
                .clone()
        })
        .collect()
}

fn push_node(
    nodes: &mut Vec<ExprNode>,
    metadata: &mut Vec<ExprMetadata>,
    node: ExprNode,
    source: ExprSourceKind,
) -> ExprId {
    let id = ExprId::from_index(nodes.len());
    nodes.push(node);
    metadata.push(ExprMetadata::new(source));
    id
}

#[cfg(test)]
mod tests {
    use laddu_expr::{Expr, complex, event_scalar, parameter, polar_complex};

    use super::*;

    fn diagnostics(expression: &Expr) -> NormalizationDiagnostics {
        CompiledModel::from_expr(expression)
            .unwrap()
            .normalization_diagnostics()
            .clone()
    }

    #[test]
    fn extracts_rectangular_and_polar_coherent_models() {
        let basis = complex(event_scalar("x"), event_scalar("y"));
        let rectangular = (complex(parameter!("re"), parameter!("im")) * basis.clone()).norm_sqr();
        let polar = (polar_complex(parameter!("mag"), parameter!("phase")) * basis).norm_sqr();
        for model in [&rectangular, &polar] {
            let diagnostics = diagnostics(model);
            assert_eq!(
                diagnostics.strategy(),
                NormalizationStrategy::Hermitian,
                "{diagnostics:?}"
            );
            assert!(!diagnostics.has_residual());
            assert!(diagnostics.basis_count() >= 1);
        }
    }

    #[test]
    fn decomposes_separable_and_nonseparable_additive_parts() {
        let x = event_scalar("x");
        let scale = Expr::from(parameter!("scale"));
        let expression = scale.clone() * x.clone() + (scale * x).sin();
        let diagnostics = diagnostics(&expression);
        assert_eq!(diagnostics.strategy(), NormalizationStrategy::Hybrid);
        assert!(diagnostics.has_residual());
    }
}
