//! Typed equivalence search for compile-time algebra.
//!
//! The public expression graph is an interchange format. Rules operate on
//! e-classes through [`Theory`], and extraction is the only place where a
//! chosen expression becomes a runtime graph again.

use std::{collections::HashMap, fmt, time::Instant};

use egg::{Analysis, DidMerge, EGraph, Id, Language};
use laddu_expr::{
    BinaryOp, ExprGraph, ExprId, ExprNode, ExprNodeSemantics, ExprSourceKind, NumberClass, UnaryOp,
    ValueKind,
};
use num::complex::Complex64;

use crate::{CompileResult, DependencyFacts, GraphFacts, NodeFacts};

mod algebra;
mod builder;
mod extract;
mod pattern;
mod rules;
mod tensor;
mod trig;

pub(crate) use extract::ExtractionDiagnostics;

#[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
enum Head {
    Source(usize),
    Real(u64),
    Complex(u64, u64),
    Unary(UnaryOp),
    Binary(BinaryOp),
    Sum,
    Product,
    ComplexParts,
    Vector,
    Matrix { rows: usize, cols: usize },
    Component(usize),
    MatrixElement(usize, usize),
    MatMul,
    MatVec,
    Dot,
    Solve,
}

impl Head {
    fn of(node: &ExprNode, source: usize) -> Self {
        match node {
            ExprNode::RealConst(value) => Self::Real(value.to_bits()),
            ExprNode::ComplexConst(value) => Self::Complex(value.re.to_bits(), value.im.to_bits()),
            ExprNode::ScalarParam(_)
            | ExprNode::EventScalar(_)
            | ExprNode::EventP4Component { .. } => Self::Source(source),
            ExprNode::Unary { op, .. } => Self::Unary(*op),
            ExprNode::Binary {
                op: BinaryOp::Add, ..
            }
            | ExprNode::NaryAdd { .. } => Self::Sum,
            ExprNode::Binary {
                op: BinaryOp::Mul, ..
            }
            | ExprNode::NaryMul { .. } => Self::Product,
            ExprNode::Binary { op, .. } => Self::Binary(*op),
            ExprNode::Complex { .. } => Self::ComplexParts,
            ExprNode::Vector { .. } => Self::Vector,
            ExprNode::Matrix { rows, cols, .. } => Self::Matrix {
                rows: *rows,
                cols: *cols,
            },
            ExprNode::Component { index, .. } => Self::Component(*index),
            ExprNode::MatrixElement { row, col, .. } => Self::MatrixElement(*row, *col),
            ExprNode::MatMul { .. } => Self::MatMul,
            ExprNode::MatVec { .. } => Self::MatVec,
            ExprNode::Dot { .. } => Self::Dot,
            ExprNode::Solve { .. } => Self::Solve,
        }
    }

    fn node(&self, source: &[ExprNode], children: &[ExprId]) -> ExprNode {
        let child = |i| children[i];
        match self {
            Self::Source(index) => source[*index].clone(),
            Self::Real(value) => ExprNode::RealConst(f64::from_bits(*value)),
            Self::Complex(re, im) => {
                ExprNode::ComplexConst(Complex64::new(f64::from_bits(*re), f64::from_bits(*im)))
            }
            Self::Unary(op) => ExprNode::Unary {
                op: *op,
                input: child(0),
            },
            Self::Binary(op) => ExprNode::Binary {
                op: *op,
                lhs: child(0),
                rhs: child(1),
            },
            Self::Sum => ExprNode::NaryAdd {
                terms: children.to_vec(),
            },
            Self::Product => ExprNode::NaryMul {
                factors: children.to_vec(),
            },
            Self::ComplexParts => ExprNode::Complex {
                re: child(0),
                im: child(1),
            },
            Self::Vector => ExprNode::Vector {
                elements: children.to_vec(),
            },
            Self::Matrix { rows, cols } => ExprNode::Matrix {
                rows: *rows,
                cols: *cols,
                elements: children.to_vec(),
            },
            Self::Component(index) => ExprNode::Component {
                input: child(0),
                index: *index,
            },
            Self::MatrixElement(row, col) => ExprNode::MatrixElement {
                input: child(0),
                row: *row,
                col: *col,
            },
            Self::MatMul => ExprNode::MatMul {
                lhs: child(0),
                rhs: child(1),
            },
            Self::MatVec => ExprNode::MatVec {
                matrix: child(0),
                vector: child(1),
            },
            Self::Dot => ExprNode::Dot {
                lhs: child(0),
                rhs: child(1),
            },
            Self::Solve => ExprNode::Solve {
                matrix: child(0),
                rhs: child(1),
            },
        }
    }

    fn source_kind(&self) -> ExprSourceKind {
        match self {
            Self::Source(_) => ExprSourceKind::Event,
            Self::Real(_) | Self::Complex(_, _) => ExprSourceKind::Const,
            Self::Unary(_) => ExprSourceKind::Unary,
            Self::Binary(_) | Self::Sum | Self::Product => ExprSourceKind::Binary,
            Self::ComplexParts => ExprSourceKind::Complex,
            Self::Vector | Self::Component(_) => ExprSourceKind::Vector,
            Self::Matrix { .. } | Self::MatrixElement(_, _) => ExprSourceKind::Matrix,
            Self::MatMul | Self::MatVec | Self::Dot | Self::Solve => ExprSourceKind::LinearAlgebra,
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
struct Term {
    head: Head,
    children: Box<[Id]>,
}

impl Term {
    fn new(head: Head, children: impl Into<Box<[Id]>>) -> Self {
        Self {
            head,
            children: children.into(),
        }
    }
}

impl fmt::Display for Term {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.head)
    }
}

impl Language for Term {
    type Discriminant = (Head, usize);

    fn discriminant(&self) -> Self::Discriminant {
        (self.head.clone(), self.children.len())
    }

    fn matches(&self, other: &Self) -> bool {
        self.head == other.head && self.children.len() == other.children.len()
    }

    fn children(&self) -> &[Id] {
        &self.children
    }

    fn children_mut(&mut self) -> &mut [Id] {
        &mut self.children
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Facts {
    kind: ValueKind,
    number: NumberClass,
    dependency: DependencyFacts,
}

#[derive(Clone, Debug, Default)]
struct CasAnalysis {
    leaves: Vec<ExprNode>,
    leaf_facts: Vec<NodeFacts>,
}

impl Analysis<Term> for CasAnalysis {
    type Data = Facts;

    fn make(egraph: &mut EGraph<Term, Self>, enode: &Term, _id: Id) -> Self::Data {
        if let Head::Source(index) = enode.head {
            let facts = egraph.analysis.leaf_facts[index];
            return Facts {
                kind: facts.value_kind,
                number: facts.number_class,
                dependency: facts.dependency,
            };
        }
        let children = enode
            .children()
            .iter()
            .map(|id| &egraph[*id].data)
            .collect::<Vec<_>>();
        let semantics = children
            .iter()
            .map(|facts| ExprNodeSemantics {
                value_kind: facts.kind,
                number_class: facts.number,
            })
            .collect::<Vec<_>>();
        let fake_ids = (0..children.len())
            .map(ExprId::from_index)
            .collect::<Vec<_>>();
        let node = enode.head.node(&egraph.analysis.leaves, &fake_ids);
        let inferred = node.semantics(&semantics);
        let dependency = children
            .iter()
            .fold(DependencyFacts::per_compile(), |acc, facts| {
                acc.union(facts.dependency)
            });
        Facts {
            kind: inferred.value_kind,
            number: inferred.number_class,
            dependency,
        }
    }

    fn merge(&mut self, into: &mut Self::Data, from: Self::Data) -> DidMerge {
        assert!(
            same_shape(into.kind, from.kind),
            "CAS merged incompatible shapes"
        );
        let old = into.clone();
        if into.kind == ValueKind::Real && from.kind == ValueKind::Complex {
            into.kind = ValueKind::Complex;
        }
        if into.number != from.number {
            into.number = NumberClass::Unknown;
        }
        into.dependency = into.dependency.union(from.dependency);
        DidMerge(*into != old, false)
    }
}

fn same_shape(a: ValueKind, b: ValueKind) -> bool {
    match (a, b) {
        (ValueKind::Real | ValueKind::Complex, ValueKind::Real | ValueKind::Complex) => true,
        (ValueKind::Vector { len: x }, ValueKind::Vector { len: y }) => x == y,
        (ValueKind::Matrix { rows: ar, cols: ac }, ValueKind::Matrix { rows: br, cols: bc }) => {
            ar == br && ac == bc
        }
        _ => false,
    }
}

/// All mathematically equivalent forms retained during bounded search.
pub(crate) struct Cas {
    source: ExprGraph,
    source_classes: Vec<Id>,
    graph: EGraph<Term, CasAnalysis>,
    root: Id,
    budget: OptimizationBudget,
    diagnostics: SearchDiagnostics,
}

fn input_children(source: &ExprGraph, node: &ExprNode, classes: &[Id], head: &Head) -> Vec<Id> {
    let mut children = Vec::new();
    fn collect(source: &ExprGraph, id: ExprId, classes: &[Id], head: &Head, out: &mut Vec<Id>) {
        let node = &source.nodes()[id.index()];
        if matches!(head, Head::Sum | Head::Product) && Head::of(node, id.index()) == *head {
            for child in node.children() {
                collect(source, child, classes, head, out);
            }
        } else {
            out.push(classes[id.index()]);
        }
    }
    for child in node.children() {
        collect(source, child, classes, head, &mut children);
    }
    if matches!(head, Head::Sum | Head::Product) {
        children.sort_unstable();
    }
    children
}

/// Limits the one-time algebraic search.
#[derive(Copy, Clone, Debug)]
pub struct OptimizationBudget {
    /// Maximum saturation rounds.
    pub rounds: usize,
    /// Maximum e-nodes retained during search.
    pub nodes: usize,
    /// Approximate memory ceiling for e-nodes and their children.
    pub memory_bytes: usize,
    /// Maximum wall time allowed for equality search.
    pub search_seconds: f64,
    /// Maximum time allowed for exact DAG extraction.
    pub solver_seconds: f64,
}

impl Default for OptimizationBudget {
    fn default() -> Self {
        Self {
            rounds: 8,
            nodes: 10_000,
            memory_bytes: 64 * 1024 * 1024,
            search_seconds: 30.0,
            solver_seconds: 5.0,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct SearchDiagnostics {
    pub rounds: usize,
    pub nodes: usize,
    pub peak_memory_bytes: usize,
    pub stop: Option<&'static str>,
}

/// Outcome of bounded equivalence search and graph extraction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OptimizationDiagnostics {
    rounds: usize,
    explored_nodes: usize,
    peak_memory_bytes: usize,
    stop_reason: &'static str,
    execution_exact: bool,
    execution_fallback: Option<&'static str>,
    normalization_exact: bool,
    normalization_fallback: Option<&'static str>,
}

impl OptimizationDiagnostics {
    /// Completed saturation rounds.
    pub fn rounds(&self) -> usize {
        self.rounds
    }
    /// Number of e-nodes retained when search stopped.
    pub fn explored_nodes(&self) -> usize {
        self.explored_nodes
    }
    /// Estimated peak bytes retained by the equivalence graph during search.
    pub fn peak_memory_bytes(&self) -> usize {
        self.peak_memory_bytes
    }
    /// Saturation or budget reason that stopped candidate generation.
    pub fn stop_reason(&self) -> &'static str {
        self.stop_reason
    }
    /// Whether execution extraction proved its cost optimum in the explored graph.
    pub fn execution_exact(&self) -> bool {
        self.execution_exact
    }
    /// Why execution extraction used a bounded fallback.
    pub fn execution_fallback(&self) -> Option<&'static str> {
        self.execution_fallback
    }
    /// Whether normalization extraction proved its cost optimum in the explored graph.
    pub fn normalization_exact(&self) -> bool {
        self.normalization_exact
    }
    /// Why normalization extraction used a bounded fallback.
    pub fn normalization_fallback(&self) -> Option<&'static str> {
        self.normalization_fallback
    }
}

impl Cas {
    pub(crate) fn diagnostics(
        &self,
        execution: &ExtractionDiagnostics,
        normalization: &ExtractionDiagnostics,
    ) -> OptimizationDiagnostics {
        OptimizationDiagnostics {
            rounds: self.diagnostics.rounds,
            explored_nodes: self.diagnostics.nodes,
            peak_memory_bytes: self.diagnostics.peak_memory_bytes,
            stop_reason: self.diagnostics.stop.unwrap_or("not started"),
            execution_exact: execution.exact,
            execution_fallback: execution.reason,
            normalization_exact: normalization.exact,
            normalization_fallback: normalization.reason,
        }
    }
}

impl Cas {
    pub(crate) fn import(source: ExprGraph, budget: OptimizationBudget) -> Self {
        let facts = GraphFacts::analyze(&source);
        let analysis = CasAnalysis {
            leaves: source.nodes().to_vec(),
            leaf_facts: facts.nodes().to_vec(),
        };
        let mut graph = EGraph::new(analysis);
        let mut source_classes = Vec::with_capacity(source.nodes().len());
        let mut canonical_leaves = HashMap::new();
        for (index, node) in source.nodes().iter().enumerate() {
            let mut head = Head::of(node, index);
            if matches!(head, Head::Source(_)) {
                let canonical = *canonical_leaves
                    .entry(node.structural_key())
                    .or_insert(index);
                head = Head::Source(canonical);
            }
            let children = input_children(&source, node, &source_classes, &head);
            let id = graph.add(Term::new(head, children));
            source_classes.push(id);
        }
        let root = source_classes[source.root().index()];
        Self {
            source,
            source_classes,
            graph,
            root,
            budget,
            diagnostics: SearchDiagnostics::default(),
        }
    }

    pub(crate) fn search(mut self) -> Self {
        let started = Instant::now();
        let mut theory = rules::standard(&self);
        const FAMILIES: usize = 4;
        self.diagnostics.nodes = self.graph.total_size();
        self.diagnostics.peak_memory_bytes = self.estimated_memory_bytes();
        if let Some(reason) = self.budget_reason() {
            self.diagnostics.stop = Some(reason);
            return self;
        }
        if !self.budget.search_seconds.is_finite() || self.budget.search_seconds <= 0.0 {
            self.diagnostics.stop = Some("time budget");
            return self;
        }
        for round in 0..self.budget.rounds {
            let before = self.graph.total_size();
            let before_classes = self.graph.number_of_classes();
            for offset in 0..FAMILIES {
                match (round + offset) % FAMILIES {
                    0 => theory.apply(&mut self),
                    1 => algebra::generate(&mut self),
                    2 => tensor::generate(&mut self),
                    _ => trig::generate(&mut self),
                }
                self.graph.rebuild();
                self.diagnostics.nodes = self.graph.total_size();
                self.diagnostics.peak_memory_bytes = self
                    .diagnostics
                    .peak_memory_bytes
                    .max(self.estimated_memory_bytes());
                if let Some(reason) = self.budget_reason() {
                    self.diagnostics.stop = Some(reason);
                    break;
                }
                if started.elapsed().as_secs_f64() >= self.budget.search_seconds {
                    self.diagnostics.stop = Some("time budget");
                    break;
                }
            }
            self.diagnostics.rounds = round + 1;
            if self.diagnostics.stop.is_some() {
                break;
            }
            if self.graph.total_size() == before && self.graph.number_of_classes() == before_classes
            {
                self.diagnostics.stop = Some("saturated");
                break;
            }
        }
        if self.diagnostics.stop.is_none() {
            self.diagnostics.stop = Some("round budget");
        }
        self
    }

    fn estimated_memory_bytes(&self) -> usize {
        let terms = self.graph.classes().flat_map(|class| class.nodes.iter());
        let terms_bytes = terms
            .map(|term| {
                std::mem::size_of::<Term>() + term.children.len() * std::mem::size_of::<Id>()
            })
            .sum::<usize>();
        // Hash-consing, class vectors, and analysis facts retain additional
        // storage beyond the language nodes themselves.
        terms_bytes
            .saturating_mul(3)
            .saturating_add(self.graph.number_of_classes() * (std::mem::size_of::<Facts>() + 64))
    }

    fn budget_reason(&self) -> Option<&'static str> {
        if self.graph.total_size() >= self.budget.nodes {
            Some("node budget")
        } else if self.estimated_memory_bytes() >= self.budget.memory_bytes {
            Some("memory budget")
        } else {
            None
        }
    }

    pub(crate) fn extract_execution(&self) -> CompileResult<(ExprGraph, ExtractionDiagnostics)> {
        extract::extract(self, self.root)
    }

    pub(crate) fn extract_normalization(
        &self,
    ) -> CompileResult<(ExprGraph, ExtractionDiagnostics)> {
        extract::extract_normalization(self, self.root)
    }
}

#[cfg(test)]
mod tests {
    use laddu_expr::{Expr, ExprNode, UnaryOp, dot, matmul, matrix, parameter, vector};

    use super::*;

    #[test]
    fn equation_search_extracts_scalar_identity() {
        let source = (Expr::from(parameter!("x")) + 0.0).to_graph();
        let cas = Cas::import(source, OptimizationBudget::default()).search();
        let (graph, diagnostics) = cas.extract_execution().unwrap();
        assert!(diagnostics.exact);
        assert!(matches!(graph.nodes(), [ExprNode::ScalarParam(_)]));
    }

    #[test]
    fn ac_match_finds_trig_identity_in_larger_sum() {
        let x = Expr::from(parameter!("x"));
        let source = (x.clone().sin().powi(2) + x.clone().cos().powi(2) + 7.0).to_graph();
        let cas = Cas::import(source, OptimizationBudget::default()).search();
        let (graph, _) = cas.extract_execution().unwrap();
        assert!(!graph.nodes().iter().any(|node| {
            matches!(
                node,
                ExprNode::Unary {
                    op: UnaryOp::Sin | UnaryOp::Cos,
                    ..
                }
            )
        }));
    }

    #[test]
    fn euler_rule_exposes_exponential_candidate() {
        use num::complex::Complex64;
        let x = Expr::from(parameter!("x"));
        let source = (x.clone().cos() + Complex64::I * x.sin()).to_graph();
        let cas = Cas::import(source, OptimizationBudget::default()).search();
        let root = cas.graph.find(cas.root);
        assert!(
            cas.graph[root]
                .nodes
                .iter()
                .any(|node| node.head == Head::Unary(UnaryOp::Exp)),
            "root forms: {:?}",
            cas.graph[root].nodes
        );
        let (graph, _) = cas.extract_execution().unwrap();
        assert!(
            graph.nodes().iter().any(|node| matches!(
                node,
                ExprNode::Unary {
                    op: UnaryOp::Exp,
                    ..
                }
            )),
            "selected: {graph}"
        );
    }

    #[test]
    fn signed_product_collects_constant_coefficient() {
        let phi = Expr::from(parameter!("phi"));
        let source = (Expr::from(-2.0) * (0.0 - phi)).to_graph();
        let cas = Cas::import(source, OptimizationBudget::default()).search();
        let (graph, _) = cas.extract_execution().unwrap();
        assert_eq!(format!("{graph}"), "2 * phi");
        assert!(
            matches!(graph.node(graph.root()), Some(ExprNode::NaryMul { factors }) if factors.len() == 2),
            "{graph:?}"
        );
        assert!(
            graph
                .nodes()
                .iter()
                .any(|node| matches!(node, ExprNode::RealConst(2.0))),
            "{graph:?}"
        );
    }

    #[test]
    fn factorization_retains_numeric_and_power_common_factor() {
        let c = Expr::from(parameter!("c"));
        let d = Expr::from(parameter!("d"));
        let lhs =
            Expr::from(-1.0) * -3.0 * 5.0 * 7.0 * c.clone() * c.clone() * d.clone() * d.clone();
        let rhs = Expr::from(-1.0) * -3.0 * 5.0 * d.clone() * d.clone();
        let cas = Cas::import((lhs - rhs).to_graph(), OptimizationBudget::default()).search();
        let root = cas.graph.find(cas.root);
        assert!(
            cas.graph[root]
                .nodes
                .iter()
                .any(|node| node.head == Head::Product),
            "root forms: {:?}, diagnostics: {:?}",
            cas.graph[root].nodes,
            cas.diagnostics
        );
        let (graph, _) = cas.extract_execution().unwrap();
        assert!(
            crate::OptimizationCost::analyze(&graph)
                .is_no_worse_than(&crate::OptimizationCost::analyze(&cas.source))
        );
    }

    #[test]
    fn search_retains_even_half_angle_and_parity_candidates() {
        let x = Expr::from(parameter!("x"));
        let power = Cas::import(
            (0.5 * x.clone()).sin().powi(6).to_graph(),
            OptimizationBudget::default(),
        )
        .search();
        let root = power.graph.find(power.root);
        assert!(
            power.graph[root]
                .nodes
                .iter()
                .any(|term| term.head == Head::Unary(UnaryOp::PowI(3)))
        );

        let parity = Cas::import((-x).sin().to_graph(), OptimizationBudget::default()).search();
        let root = parity.graph.find(parity.root);
        assert!(
            parity.graph[root]
                .nodes
                .iter()
                .any(|term| term.head == Head::Unary(UnaryOp::Neg))
        );
    }

    #[test]
    fn difference_of_squares_retains_both_factored_and_expanded_forms() {
        let x = Expr::from(parameter!("x"));
        let y = Expr::from(parameter!("y"));
        let cas = Cas::import(
            ((x.clone() - y.clone()) * (x + y)).to_graph(),
            OptimizationBudget::default(),
        )
        .search();
        let root = cas.graph.find(cas.root);
        assert!(
            cas.graph[root]
                .nodes
                .iter()
                .any(|term| term.head == Head::Product)
        );
        assert!(
            cas.graph[root]
                .nodes
                .iter()
                .any(|term| term.head == Head::Binary(BinaryOp::Sub))
        );
    }

    #[test]
    fn shared_factor_rule_retains_distributed_form() {
        let x = Expr::from(parameter!("x"));
        let a = Expr::from(parameter!("a"));
        let b = Expr::from(parameter!("b"));
        let cas = Cas::import((x * (a + b)).to_graph(), OptimizationBudget::default()).search();
        let root = cas.graph.find(cas.root);
        assert!(
            cas.graph[root]
                .nodes
                .iter()
                .any(|term| term.head == Head::Product)
        );
        assert!(
            cas.graph[root]
                .nodes
                .iter()
                .any(|term| term.head == Head::Sum)
        );
    }

    #[test]
    fn complex_conjugation_retains_elementwise_candidate() {
        let z = laddu_expr::complex(parameter!("re"), parameter!("im"));
        let cas = Cas::import(z.conj().to_graph(), OptimizationBudget::default()).search();
        let root = cas.graph.find(cas.root);
        assert!(
            cas.graph[root]
                .nodes
                .iter()
                .any(|term| term.head == Head::ComplexParts)
        );
    }

    #[test]
    fn tensor_contractions_retain_lifted_scalar_factors() {
        let scale = Expr::from(parameter!("scale"));
        let x = Expr::from(parameter!("x"));
        let y = Expr::from(parameter!("y"));
        let vector_expr = dot(vector([&scale * &x, &scale * &y]), vector([1.0, 2.0]));
        let matrix_expr = matmul(matrix([[&scale * &x, &scale * &y]]), matrix([[1.0], [2.0]]))
            .matrix_element(0, 0);
        for source in [vector_expr.to_graph(), matrix_expr.to_graph()] {
            let cas = Cas::import(source, OptimizationBudget::default()).search();
            let root = cas.graph.find(cas.root);
            assert!(
                cas.graph[root]
                    .nodes
                    .iter()
                    .any(|node| node.head == Head::Product)
            );
        }
    }
}
