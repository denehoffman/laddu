use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    time::Instant,
};

use egg::{CostFunction, Extractor, Id, Language, RecExpr};
use good_lp::{
    Expression, Solution, SolutionStatus, SolverModel, WithTimeLimit, microlp, variable, variables,
};
use laddu_expr::{ExprGraph, ExprGraphRebuilder, ExprId, ExprMetadata, ExprNode, UnaryOp};

use crate::{CompileResult, OptimizationCost};

use super::{Cas, Head, Term};

const EXACT_NODE_LIMIT: usize = 2_000;

#[derive(Clone, Debug)]
pub(crate) struct ExtractionDiagnostics {
    pub exact: bool,
    pub reason: Option<&'static str>,
}

pub(super) fn extract(cas: &Cas, root: Id) -> CompileResult<(ExprGraph, ExtractionDiagnostics)> {
    extract_with(cas, root, false)
}

pub(super) fn extract_normalization(
    cas: &Cas,
    root: Id,
) -> CompileResult<(ExprGraph, ExtractionDiagnostics)> {
    extract_with(cas, root, true)
}

fn extract_with(
    cas: &Cas,
    root: Id,
    normalization: bool,
) -> CompileResult<(ExprGraph, ExtractionDiagnostics)> {
    let reachable = reachable_classes(cas, root);
    let (selected, mut exact, mut reason) = if reachable.len() <= EXACT_NODE_LIMIT {
        match mip(cas, root, &reachable, normalization) {
            Some(expr) => (expr, true, None),
            None => (
                greedy(cas, root, normalization),
                false,
                Some("solver limit"),
            ),
        }
    } else {
        (
            greedy(cas, root, normalization),
            false,
            Some("exact extraction size limit"),
        )
    };
    let extracted = lower(cas, &selected)?;
    let chosen = if normalization {
        // The source form is often deliberately recognizable as a coherent
        // norm. Keep it if extraction has no better normalization candidate.
        let source_facts = crate::GraphFacts::analyze(&cas.source);
        let extracted_facts = crate::GraphFacts::analyze(&extracted);
        let original = crate::NormalizationPlan::analyze(&cas.source, &source_facts);
        let candidate = crate::NormalizationPlan::analyze(&extracted, &extracted_facts);
        if normalization_rank(&original) < normalization_rank(&candidate) {
            exact = false;
            reason = Some("normalization strategy correction");
            cas.source.clone()
        } else {
            extracted
        }
    } else if OptimizationCost::analyze(&cas.source)
        .is_better_than(&OptimizationCost::analyze(&extracted))
    {
        exact = false;
        reason = Some("source cost correction");
        cas.source.clone()
    } else {
        extracted
    };
    Ok((chosen, ExtractionDiagnostics { exact, reason }))
}

fn normalization_rank(plan: &crate::NormalizationPlan) -> (u8, usize) {
    use crate::NormalizationStrategy;
    let family = match plan.diagnostics().strategy() {
        NormalizationStrategy::Hermitian => 0,
        NormalizationStrategy::LinearStatistics => 1,
        NormalizationStrategy::Hybrid => 2,
        NormalizationStrategy::General => 3,
    };
    (family, plan.diagnostics().basis_count())
}

fn reachable_classes(cas: &Cas, root: Id) -> Vec<Id> {
    let mut seen = BTreeSet::new();
    let mut stack = vec![cas.graph.find(root)];
    while let Some(id) = stack.pop() {
        let id = cas.graph.find(id);
        if seen.insert(id) {
            for node in &cas.graph[id].nodes {
                stack.extend(node.children().iter().copied());
            }
        }
    }
    seen.into_iter().collect()
}

struct ClassVars {
    active: good_lp::Variable,
    rank: good_lp::Variable,
    choices: Vec<good_lp::Variable>,
}

fn mip(cas: &Cas, root: Id, classes: &[Id], normalization: bool) -> Option<RecExpr<Term>> {
    if !cas.budget.solver_seconds.is_finite() || cas.budget.solver_seconds <= 0.0 {
        return None;
    }
    let started = Instant::now();
    let stages = if normalization { 1 } else { 5 };
    let mut locked = Vec::new();
    let mut selected = None;
    for stage in 0..stages {
        let remaining = cas.budget.solver_seconds - started.elapsed().as_secs_f64();
        if remaining <= 0.0 {
            return None;
        }
        let (choices, score) =
            mip_stage(cas, root, classes, normalization, stage, &locked, remaining)?;
        locked.push((stage, score));
        selected = Some(choices);
    }
    build_recexpr(cas, root, &selected?)
}

fn mip_stage(
    cas: &Cas,
    root: Id,
    classes: &[Id],
    normalization: bool,
    stage: usize,
    locked: &[(usize, f64)],
    seconds: f64,
) -> Option<(HashMap<Id, usize>, f64)> {
    let count = classes.len() as f64;
    let mut variables = variables!();
    let mut by_class = BTreeMap::new();
    for &id in classes {
        let class = &cas.graph[id];
        by_class.insert(
            id,
            ClassVars {
                active: variables.add(variable().binary()),
                rank: variables.add(variable().min(0.0).max(count)),
                choices: class
                    .nodes
                    .iter()
                    .map(|_| variables.add(variable().binary()))
                    .collect(),
            },
        );
    }

    let objective = mip_objective(cas, classes, &by_class, normalization, stage);
    let mut problem = variables
        .minimise(objective)
        .using(microlp)
        .with_time_limit(seconds);
    for &(locked_stage, optimum) in locked {
        problem.add_constraint(
            (mip_objective(cas, classes, &by_class, normalization, locked_stage) - optimum).eq(0.0),
        );
    }
    for &id in classes {
        let vars = &by_class[&id];
        let chosen = vars
            .choices
            .iter()
            .copied()
            .fold(Expression::from(0.0), |sum, choice| sum + choice);
        problem.add_constraint((chosen - vars.active).eq(0.0));
        for (node, &choice) in cas.graph[id].nodes.iter().zip(&vars.choices) {
            for &child in node.children() {
                let child = cas.graph.find(child);
                let child_vars = &by_class[&child];
                problem.add_constraint((choice - child_vars.active).leq(0.0));
                problem.add_constraint(
                    (vars.rank - child_vars.rank - count * choice).geq(1.0 - count),
                );
            }
        }
    }
    problem.add_constraint(Expression::from(by_class[&cas.graph.find(root)].active).eq(1.0));
    let solution = problem.solve().ok()?;
    if !matches!(solution.status(), SolutionStatus::Optimal) {
        return None;
    }
    let mut selected = HashMap::new();
    for &id in classes {
        if solution.value(by_class[&id].active) > 0.5 {
            let index = by_class[&id]
                .choices
                .iter()
                .position(|choice| solution.value(*choice) > 0.5)?;
            selected.insert(id, index);
        }
    }
    let score = selected
        .iter()
        .map(|(&id, &index)| stage_cost(cas, &cas.graph[id].nodes[index], normalization, stage))
        .sum();
    Some((selected, score))
}

fn mip_objective(
    cas: &Cas,
    classes: &[Id],
    by_class: &BTreeMap<Id, ClassVars>,
    normalization: bool,
    stage: usize,
) -> Expression {
    let mut objective: Expression = 0.0.into();
    for &id in classes {
        for (node, choice) in cas.graph[id].nodes.iter().zip(&by_class[&id].choices) {
            objective += stage_cost(cas, node, normalization, stage) * *choice;
        }
    }
    objective
}

fn build_recexpr(cas: &Cas, root: Id, selected: &HashMap<Id, usize>) -> Option<RecExpr<Term>> {
    let mut emitted = HashMap::new();
    let mut active = BTreeSet::new();
    let mut out = RecExpr::default();
    fn visit(
        cas: &Cas,
        id: Id,
        selected: &HashMap<Id, usize>,
        emitted: &mut HashMap<Id, Id>,
        active: &mut BTreeSet<Id>,
        out: &mut RecExpr<Term>,
    ) -> Option<Id> {
        let id = cas.graph.find(id);
        if let Some(&done) = emitted.get(&id) {
            return Some(done);
        }
        if !active.insert(id) {
            return None;
        }
        let node = cas.graph[id].nodes.get(*selected.get(&id)?)?.clone();
        let mut children = Vec::with_capacity(node.children.len());
        for &child in &node.children {
            children.push(visit(cas, child, selected, emitted, active, out)?);
        }
        let result = out.add(Term::new(node.head, children));
        active.remove(&id);
        emitted.insert(id, result);
        Some(result)
    }
    visit(cas, root, selected, &mut emitted, &mut active, &mut out)?;
    Some(out)
}

struct TreeCost<'a> {
    cas: &'a Cas,
    normalization: bool,
}

impl CostFunction<Term> for TreeCost<'_> {
    type Cost = u64;

    fn cost<C>(&mut self, node: &Term, mut child_cost: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        let local = match node.head {
            Head::Source(_) | Head::Real(_) | Head::Complex(_, _) => 1,
            Head::Unary(UnaryOp::Exp | UnaryOp::Sin | UnaryOp::Cos | UnaryOp::Log) => 20,
            Head::MatMul | Head::MatVec | Head::Dot | Head::Solve => 50,
            _ => 2,
        };
        let preferred = if self.normalization && matches!(node.head, Head::Unary(UnaryOp::NormSqr))
        {
            0
        } else {
            local
        };
        let _ = self.cas;
        node.children()
            .iter()
            .copied()
            .fold(preferred, |acc, id| acc.saturating_add(child_cost(id)))
    }
}

fn greedy(cas: &Cas, root: Id, normalization: bool) -> RecExpr<Term> {
    Extractor::new(&cas.graph, TreeCost { cas, normalization })
        .find_best(root)
        .1
}

fn node_operation(node: &Term) -> f64 {
    match node.head {
        Head::Source(_) | Head::Real(_) | Head::Complex(_, _) => 0.0,
        Head::Unary(UnaryOp::Exp | UnaryOp::Sin | UnaryOp::Cos | UnaryOp::Log) => 20.0,
        Head::Unary(UnaryOp::Sqrt) => 8.0,
        Head::Unary(UnaryOp::NormSqr) => 4.0,
        Head::Unary(UnaryOp::PowI(power)) => match power.unsigned_abs() {
            0 | 1 => 0.0,
            2 | 3 => 3.0,
            _ => 4.0,
        },
        Head::Binary(laddu_expr::BinaryOp::Mul) => 2.0,
        Head::Binary(laddu_expr::BinaryOp::Div) => 6.0,
        Head::Binary(laddu_expr::BinaryOp::Atan2) => 20.0,
        Head::Sum => node.children.len().saturating_sub(1) as f64,
        Head::Product => (2 * node.children.len().saturating_sub(1)) as f64,
        Head::MatMul | Head::MatVec | Head::Dot | Head::Solve => 50.0,
        _ => 1.0,
    }
}

fn node_dependency(cas: &Cas, node: &Term) -> crate::DependencyFacts {
    match node.head {
        Head::Source(index) => cas.graph.analysis.leaf_facts[index].dependency,
        _ => node
            .children()
            .iter()
            .fold(crate::DependencyFacts::per_compile(), |acc, child| {
                acc.union(cas.graph[cas.graph.find(*child)].data.dependency)
            }),
    }
}

fn stage_cost(cas: &Cas, node: &Term, normalization: bool, stage: usize) -> f64 {
    if !normalization {
        if stage == 4 {
            return 1.0;
        }
        let dependency = node_dependency(cas, node);
        let node_stage = match (
            dependency.depends_on_free_params,
            dependency.depends_on_event,
        ) {
            (true, true) => 0,
            (true, false) => 1,
            (false, true) => 2,
            (false, false) => 3,
        };
        return if stage == node_stage {
            node_operation(node)
        } else {
            0.0
        };
    }
    let operation = node_operation(node);
    let dependency = node_dependency(cas, node);
    let tier = match (
        dependency.depends_on_free_params,
        dependency.depends_on_event,
    ) {
        (true, true) => 1.0e10,
        (true, false) => 1.0e6,
        (false, true) => 100.0,
        (false, false) => 1.0,
    };
    if matches!(node.head, Head::Unary(UnaryOp::NormSqr)) {
        0.1 * tier
    } else {
        (operation + 0.001) * tier
    }
}

fn lower(cas: &Cas, expr: &RecExpr<Term>) -> CompileResult<ExprGraph> {
    let mut builder = ExprGraphRebuilder::<usize>::with_capacity(expr.as_ref().len());
    let mut mapped: Vec<ExprId> = Vec::with_capacity(expr.as_ref().len());
    let mut classes = Vec::with_capacity(expr.as_ref().len());
    let mut provenance: HashMap<Id, Vec<ExprId>> = HashMap::new();
    for (index, &class) in cas.source_classes.iter().enumerate() {
        provenance
            .entry(cas.graph.find(class))
            .or_default()
            .push(ExprId::from_index(index));
    }
    for (index, term) in expr.as_ref().iter().enumerate() {
        let egraph_term = Term::new(
            term.head.clone(),
            term.children()
                .iter()
                .map(|child| classes[usize::from(*child)])
                .collect::<Vec<_>>(),
        );
        let class = cas.graph.lookup(egraph_term).map(|id| cas.graph.find(id));
        classes.push(class.expect("extracted CAS term exists in equivalence graph"));
        let mut children = term
            .children()
            .iter()
            .map(|id| mapped[usize::from(*id)])
            .collect::<Vec<_>>();
        if matches!(term.head, Head::Sum | Head::Product) {
            children.sort_by_key(|id| {
                (
                    !matches!(
                        builder.nodes()[id.index()],
                        ExprNode::RealConst(_) | ExprNode::ComplexConst(_)
                    ),
                    id.index(),
                )
            });
        }
        let node = term.head.node(cas.source.nodes(), &children);
        let metadata = if index + 1 == expr.as_ref().len() {
            cas.source
                .metadata(cas.source.root())
                .expect("source root has metadata")
                .clone()
        } else if let Some(source) = class
            .and_then(|id| provenance.get(&id))
            .and_then(|sources| {
                sources
                    .iter()
                    .copied()
                    .find(|&source| {
                        let metadata = cas
                            .source
                            .metadata(source)
                            .expect("source node has metadata");
                        metadata.name().is_some() || !metadata.tags().is_empty()
                    })
                    .or_else(|| sources.first().copied())
            })
        {
            cas.source
                .metadata(source)
                .expect("source node has metadata")
                .clone()
        } else {
            ExprMetadata::new(term.head.source_kind())
        };
        mapped.push(builder.emit(index, node, metadata));
    }
    Ok(builder.finish(*mapped.last().expect("CAS expression has a root"))?)
}
