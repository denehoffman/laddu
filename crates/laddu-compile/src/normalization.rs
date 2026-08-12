use laddu_expr::{
    BinaryOp, ExprGraph, ExprGraphRebuilder, ExprId, ExprMetadata, ExprNode, ExprSourceKind,
    UnaryOp, ValueKind,
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

        let decomposition = NormalizationAnalyzer::new(graph, facts, DEFAULT_EXPANSION_BUDGET)
            .analyze(graph.root());

        if decomposition.terms.is_empty() {
            return Self::general(
                graph,
                decomposition.last_failure().cloned().unwrap_or(
                    NormalizationFallbackReason::UnsupportedMixedOperation {
                        node: graph.root(),
                        operation: "root",
                    },
                ),
            );
        }

        let built = decomposition
            .build(graph)
            .expect("normalization decomposition emits a valid graph");
        let terms = built.terms;
        let residual = built.residual;
        let strategy = if residual.is_some() {
            NormalizationStrategy::Hybrid
        } else if built.coherent_groups > 0 {
            NormalizationStrategy::Hermitian
        } else {
            NormalizationStrategy::LinearStatistics
        };
        let diagnostics = NormalizationDiagnostics {
            strategy,
            basis_count: terms.len(),
            coherent_group_count: built.coherent_groups,
            has_residual: residual.is_some(),
            fallback_reason: built.last_failure,
        };
        Self {
            graph: built.graph,
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
        let mut builder = NormalizationGraphBuilder::new(&self.graph);
        let mut products = Vec::with_capacity(self.terms.len());
        for (term, statistic) in self.terms.iter().zip(statistics) {
            let constant = builder.constant(*statistic);
            let coefficient = builder.source(term.coefficient);
            products.push(builder.product(&[constant, coefficient]));
        }
        let sum = builder.sum(&products);
        let root = builder.unary(UnaryOp::Real, sum);
        let graph = builder.finish(root)?;
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

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum DecompositionNode {
    Source(ExprId),
    Generated(usize),
}

#[derive(Clone, Debug)]
enum GraphOperation {
    Constant(Complex64),
    Unary {
        op: UnaryOp,
        input: DecompositionNode,
    },
    Binary {
        op: BinaryOp,
        lhs: DecompositionNode,
        rhs: DecompositionNode,
    },
    Product(Vec<DecompositionNode>),
    Sum(Vec<DecompositionNode>),
}

#[derive(Clone, Debug)]
struct AnalyzedTerm {
    coefficient: DecompositionNode,
    basis: DecompositionNode,
}

#[derive(Clone, Debug)]
struct Decomposition {
    operations: Vec<GraphOperation>,
    terms: Vec<AnalyzedTerm>,
    residual: Option<DecompositionNode>,
    coherent_groups: usize,
    failures: Vec<NormalizationFallbackReason>,
}

impl Decomposition {
    fn last_failure(&self) -> Option<&NormalizationFallbackReason> {
        self.failures.last()
    }

    fn build(self, graph: &ExprGraph) -> CompileResult<BuiltDecomposition> {
        let mut builder = NormalizationGraphBuilder::new(graph);
        let mut generated = Vec::with_capacity(self.operations.len());
        for operation in self.operations {
            let resolve = |node: DecompositionNode| match node {
                DecompositionNode::Source(id) => builder.source(id),
                DecompositionNode::Generated(index) => generated[index],
            };
            let id = match operation {
                GraphOperation::Constant(value) => builder.constant(value),
                GraphOperation::Unary { op, input } => {
                    let input = resolve(input);
                    builder.unary(op, input)
                }
                GraphOperation::Binary { op, lhs, rhs } => {
                    let lhs = resolve(lhs);
                    let rhs = resolve(rhs);
                    builder.binary(op, lhs, rhs)
                }
                GraphOperation::Product(factors) => {
                    let factors = factors.into_iter().map(resolve).collect::<Vec<_>>();
                    builder.product(&factors)
                }
                GraphOperation::Sum(terms) => {
                    let terms = terms.into_iter().map(resolve).collect::<Vec<_>>();
                    builder.sum(&terms)
                }
            };
            generated.push(id);
        }
        let resolve = |node: DecompositionNode| match node {
            DecompositionNode::Source(id) => builder.source(id),
            DecompositionNode::Generated(index) => generated[index],
        };
        let terms = self
            .terms
            .into_iter()
            .map(|term| SeparableTerm {
                coefficient: resolve(term.coefficient),
                basis: resolve(term.basis),
            })
            .collect();
        let residual = self.residual.map(resolve);
        let root = builder.source(ExprId::from_index(0));
        Ok(BuiltDecomposition {
            graph: builder.finish(root)?,
            terms,
            residual,
            coherent_groups: self.coherent_groups,
            last_failure: self.failures.last().cloned(),
        })
    }
}

struct BuiltDecomposition {
    graph: ExprGraph,
    terms: Vec<SeparableTerm>,
    residual: Option<ExprId>,
    coherent_groups: usize,
    last_failure: Option<NormalizationFallbackReason>,
}

#[derive(Copy, Clone, Debug)]
struct ExpansionBudget(usize);

impl ExpansionBudget {
    fn ensure(self, count: usize) -> Result<(), NormalizationFallbackReason> {
        if count <= self.0 {
            Ok(())
        } else {
            Err(self.exceeded())
        }
    }

    fn product_count(self, lhs: usize, rhs: usize) -> Result<usize, NormalizationFallbackReason> {
        let count = lhs.checked_mul(rhs).ok_or_else(|| self.exceeded())?;
        self.ensure(count)?;
        Ok(count)
    }

    fn packed_triangle_count(self, count: usize) -> Result<usize, NormalizationFallbackReason> {
        let next = count.checked_add(1).ok_or_else(|| self.exceeded())?;
        let packed = count.checked_mul(next).ok_or_else(|| self.exceeded())? / 2;
        self.ensure(packed)?;
        Ok(packed)
    }

    fn exceeded(self) -> NormalizationFallbackReason {
        NormalizationFallbackReason::ExpansionBudgetExceeded { budget: self.0 }
    }
}

struct NormalizationAnalyzer<'a> {
    graph: &'a ExprGraph,
    facts: &'a GraphFacts,
    operations: Vec<GraphOperation>,
    one: DecompositionNode,
    budget: ExpansionBudget,
    coherent_groups: usize,
}

impl<'a> NormalizationAnalyzer<'a> {
    fn new(graph: &'a ExprGraph, facts: &'a GraphFacts, budget: usize) -> Self {
        let mut analyzer = Self {
            graph,
            facts,
            operations: Vec::new(),
            one: DecompositionNode::Source(graph.root()),
            budget: ExpansionBudget(budget),
            coherent_groups: 0,
        };
        analyzer.one = analyzer.constant(Complex64::new(1.0, 0.0));
        analyzer
    }

    fn analyze(mut self, root: ExprId) -> Decomposition {
        let mut terms = Vec::new();
        let mut residuals = Vec::new();
        let mut failures = Vec::new();
        for root in self.additive_roots(root) {
            match self.decompose(root) {
                Ok(mut extracted) => terms.append(&mut extracted),
                Err(reason) => {
                    failures.push(reason);
                    residuals.push(DecompositionNode::Source(root));
                }
            }
        }
        let residual = self.sum(&residuals);
        Decomposition {
            operations: self.operations,
            terms,
            residual,
            coherent_groups: self.coherent_groups,
            failures,
        }
    }

    fn dependency(&self, id: ExprId) -> crate::DependencyFacts {
        self.facts.get(id).expect("facts are complete").dependency
    }

    fn additive_roots(&self, root: ExprId) -> Vec<ExprId> {
        match self.graph.node(root).expect("normalization node exists") {
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

    fn decompose(&mut self, id: ExprId) -> Result<Vec<AnalyzedTerm>, NormalizationFallbackReason> {
        let dependency = self.dependency(id);
        if !dependency.depends_on_event {
            return Ok(vec![AnalyzedTerm {
                coefficient: DecompositionNode::Source(id),
                basis: self.one,
            }]);
        }
        if !dependency.depends_on_free_params {
            return Ok(vec![AnalyzedTerm {
                coefficient: self.one,
                basis: DecompositionNode::Source(id),
            }]);
        }

        match self
            .graph
            .node(id)
            .expect("normalization node exists")
            .clone()
        {
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
                let mut result = vec![AnalyzedTerm {
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
    ) -> Result<Vec<AnalyzedTerm>, NormalizationFallbackReason> {
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
                        term.coefficient = self.binary(
                            BinaryOp::Div,
                            term.coefficient,
                            DecompositionNode::Source(rhs),
                        );
                    }
                    Ok(terms)
                } else if !denominator.depends_on_free_params {
                    for term in &mut terms {
                        term.basis =
                            self.binary(BinaryOp::Div, term.basis, DecompositionNode::Source(rhs));
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
    ) -> Result<Vec<AnalyzedTerm>, NormalizationFallbackReason> {
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
                let packed_len = self.budget.packed_triangle_count(terms.len())?;
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
                        packed.push(AnalyzedTerm {
                            coefficient,
                            basis: self.product(&[left.basis, right_basis]),
                        });
                    }
                }
                Ok(packed)
            }
            UnaryOp::PowI(power) if power >= 0 => {
                let base = self.decompose(input)?;
                let mut result = vec![AnalyzedTerm {
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
    ) -> Result<Vec<AnalyzedTerm>, NormalizationFallbackReason> {
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
            result.push(AnalyzedTerm {
                coefficient,
                basis: term.basis,
            });
            let conjugated_coefficient = self.unary(UnaryOp::Conj, term.coefficient);
            let conjugated_basis = self.unary(UnaryOp::Conj, term.basis);
            let coefficient = self.product(&[conjugate_factor, conjugated_coefficient]);
            result.push(AnalyzedTerm {
                coefficient,
                basis: conjugated_basis,
            });
        }
        self.ensure_budget(&result)?;
        Ok(result)
    }

    fn multiply_terms(
        &mut self,
        lhs: &[AnalyzedTerm],
        rhs: &[AnalyzedTerm],
    ) -> Result<Vec<AnalyzedTerm>, NormalizationFallbackReason> {
        let count = self.budget.product_count(lhs.len(), rhs.len())?;
        let mut result = Vec::with_capacity(count);
        for lhs in lhs {
            for rhs in rhs {
                let coefficient = self.product(&[lhs.coefficient, rhs.coefficient]);
                let basis = self.product(&[lhs.basis, rhs.basis]);
                result.push(AnalyzedTerm { coefficient, basis });
            }
        }
        Ok(result)
    }

    fn ensure_budget(&self, terms: &[AnalyzedTerm]) -> Result<(), NormalizationFallbackReason> {
        self.budget.ensure(terms.len())
    }

    fn unsupported(&self, node: ExprId, operation: &'static str) -> NormalizationFallbackReason {
        NormalizationFallbackReason::UnsupportedMixedOperation { node, operation }
    }

    fn constant(&mut self, value: Complex64) -> DecompositionNode {
        self.push(GraphOperation::Constant(value))
    }

    fn unary(&mut self, op: UnaryOp, input: DecompositionNode) -> DecompositionNode {
        self.push(GraphOperation::Unary { op, input })
    }

    fn binary(
        &mut self,
        op: BinaryOp,
        lhs: DecompositionNode,
        rhs: DecompositionNode,
    ) -> DecompositionNode {
        self.push(GraphOperation::Binary { op, lhs, rhs })
    }

    fn product(&mut self, factors: &[DecompositionNode]) -> DecompositionNode {
        match factors {
            [] => self.one,
            [only] => *only,
            _ => self.push(GraphOperation::Product(factors.to_vec())),
        }
    }

    fn sum(&mut self, roots: &[DecompositionNode]) -> Option<DecompositionNode> {
        match roots {
            [] => None,
            [only] => Some(*only),
            _ => Some(self.push(GraphOperation::Sum(roots.to_vec()))),
        }
    }

    fn push(&mut self, operation: GraphOperation) -> DecompositionNode {
        let id = DecompositionNode::Generated(self.operations.len());
        self.operations.push(operation);
        id
    }
}

struct NormalizationGraphBuilder {
    rebuild: ExprGraphRebuilder<ExprId>,
}

impl NormalizationGraphBuilder {
    fn new(graph: &ExprGraph) -> Self {
        let mut rebuild = ExprGraphRebuilder::with_capacity(graph.nodes().len());
        for index in 0..graph.nodes().len() {
            let old_id = ExprId::from_index(index);
            let node = graph
                .node(old_id)
                .expect("normalization graph node exists")
                .map_children(|child| {
                    rebuild
                        .remapped(&child)
                        .expect("validated expression graphs emit children before parents")
                });
            let metadata = graph
                .metadata(old_id)
                .expect("normalization graph metadata is complete")
                .clone();
            rebuild.emit(old_id, node, metadata);
        }
        Self { rebuild }
    }

    fn source(&self, id: ExprId) -> ExprId {
        self.rebuild
            .remapped(&id)
            .expect("normalization source node was copied")
    }

    fn constant(&mut self, value: Complex64) -> ExprId {
        self.emit(ExprNode::from_folded_const(value), ExprSourceKind::Const)
    }

    fn unary(&mut self, op: UnaryOp, input: ExprId) -> ExprId {
        self.emit(ExprNode::Unary { op, input }, ExprSourceKind::Unary)
    }

    fn binary(&mut self, op: BinaryOp, lhs: ExprId, rhs: ExprId) -> ExprId {
        self.emit(ExprNode::Binary { op, lhs, rhs }, ExprSourceKind::Binary)
    }

    fn product(&mut self, factors: &[ExprId]) -> ExprId {
        match factors {
            [] => self.constant(Complex64::new(1.0, 0.0)),
            [only] => *only,
            _ => self.emit(
                ExprNode::NaryMul {
                    factors: factors.to_vec(),
                },
                ExprSourceKind::Binary,
            ),
        }
    }

    fn sum(&mut self, terms: &[ExprId]) -> ExprId {
        match terms {
            [] => self.constant(Complex64::new(0.0, 0.0)),
            [only] => *only,
            _ => self.emit(
                ExprNode::NaryAdd {
                    terms: terms.to_vec(),
                },
                ExprSourceKind::Binary,
            ),
        }
    }

    fn emit(&mut self, node: ExprNode, source: ExprSourceKind) -> ExprId {
        self.rebuild.emit_anonymous(node, ExprMetadata::new(source))
    }

    fn finish(self, root: ExprId) -> CompileResult<ExprGraph> {
        Ok(self.rebuild.finish(root)?)
    }
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

    fn decomposition(expression: &Expr, budget: usize) -> Decomposition {
        let graph = expression.to_graph();
        let facts = GraphFacts::analyze(&graph);
        NormalizationAnalyzer::new(&graph, &facts, budget).analyze(graph.root())
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

    #[test]
    fn classifies_binary_operations_before_building_a_graph() {
        let parameter = Expr::from(parameter!("scale"));
        let event = event_scalar("x");
        let mixed = parameter.clone() * event.clone();
        let cases = [
            ("add", parameter.clone() + event.clone(), 2, false),
            ("sub", parameter.clone() - event.clone(), 2, false),
            ("mul", mixed.clone() * mixed.clone(), 1, false),
            (
                "parameter divisor",
                mixed.clone() / parameter.clone(),
                1,
                false,
            ),
            ("event divisor", mixed.clone() / event.clone(), 1, false),
            (
                "mixed divisor",
                mixed.clone() / (parameter + event),
                0,
                true,
            ),
        ];

        for (name, expression, expected_terms, has_failure) in cases {
            let decomposition = decomposition(&expression, DEFAULT_EXPANSION_BUDGET);
            assert_eq!(decomposition.terms.len(), expected_terms, "{name}");
            assert_eq!(!decomposition.failures.is_empty(), has_failure, "{name}");
        }
    }

    #[test]
    fn classifies_unary_operations_before_building_a_graph() {
        let mixed = Expr::from(parameter!("scale")) * event_scalar("x");
        let cases = [
            ("neg", -mixed.clone(), 1, 0),
            ("conj", mixed.clone().conj(), 1, 0),
            ("norm_sqr", mixed.clone().norm_sqr(), 1, 1),
            ("powi", mixed.clone().powi(2), 1, 0),
            ("real", mixed.clone().real(), 2, 0),
            ("imag", mixed.clone().imag(), 2, 0),
            ("sin", mixed.sin(), 0, 0),
        ];

        for (name, expression, expected_terms, coherent_groups) in cases {
            let decomposition = decomposition(&expression, DEFAULT_EXPANSION_BUDGET);
            assert_eq!(decomposition.terms.len(), expected_terms, "{name}");
            assert_eq!(decomposition.coherent_groups, coherent_groups, "{name}");
            assert_eq!(
                decomposition.failures.is_empty(),
                expected_terms > 0,
                "{name}"
            );
        }
    }

    #[test]
    fn expansion_budget_checks_exact_boundaries_and_overflow() {
        let budget = ExpansionBudget(6);
        assert_eq!(budget.product_count(2, 3).unwrap(), 6);
        assert_eq!(budget.packed_triangle_count(3).unwrap(), 6);
        assert_eq!(budget.ensure(6), Ok(()));

        for result in [
            budget.product_count(2, 4),
            budget.packed_triangle_count(4),
            budget.product_count(usize::MAX, 2),
            budget.packed_triangle_count(usize::MAX),
        ] {
            assert_eq!(
                result,
                Err(NormalizationFallbackReason::ExpansionBudgetExceeded { budget: 6 })
            );
        }
    }

    #[test]
    fn fallback_diagnostics_preserve_the_last_unsupported_root() {
        let mixed = Expr::from(parameter!("scale")) * event_scalar("x");
        let expression = mixed.clone().sin() + mixed.cos();
        let graph = expression.to_graph();
        let facts = GraphFacts::analyze(&graph);
        let roots = NormalizationAnalyzer::new(&graph, &facts, DEFAULT_EXPANSION_BUDGET)
            .additive_roots(graph.root());
        let decomposition = NormalizationAnalyzer::new(&graph, &facts, DEFAULT_EXPANSION_BUDGET)
            .analyze(graph.root());

        assert_eq!(decomposition.failures.len(), 2);
        assert_eq!(
            decomposition.last_failure(),
            Some(&NormalizationFallbackReason::UnsupportedMixedOperation {
                node: roots[1],
                operation: "nonlinear unary operation",
            })
        );
    }
}
