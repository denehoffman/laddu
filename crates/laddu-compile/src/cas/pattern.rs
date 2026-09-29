//! Shared matching, guards, and typed candidate emission for CAS laws.

use super::{Cas, Head, Term, same_shape};
use egg::{Id, Language};
use laddu_expr::{BinaryOp, NumberClass, UnaryOp, ValueKind};
use std::collections::BTreeMap;

#[derive(Clone, Debug)]
pub(super) enum Pattern {
    Capture(&'static str),
    Real(f64),
    ImaginaryUnit,
    Unary(UnaryOp, Box<Self>),
    Binary(BinaryOp, Box<Self>, Box<Self>),
    Sum(Vec<Self>),
    Product(Vec<Self>),
    Call2(Call2, Box<Self>, Box<Self>),
}

#[derive(Copy, Clone, Debug)]
pub(super) enum Call2 {
    Complex,
    MatMul,
    MatVec,
    Dot,
    Solve,
}

#[derive(Clone, Debug)]
#[allow(dead_code)] // DSL guards supported even when the standard theory does not use them.
pub(super) enum Guard {
    Always,
    Real(&'static str),
    Nonzero(&'static str),
    Scalar(&'static str),
    Zero(&'static str),
    Identity(&'static str),
    And(Box<Self>, Box<Self>),
    Or(Box<Self>, Box<Self>),
}

#[derive(Clone, Debug)]
pub(super) struct Rule {
    pub(super) name: &'static str,
    pub(super) left: Pattern,
    pub(super) right: Pattern,
    pub(super) bidirectional: bool,
    pub(super) guard: Guard,
}

impl Rule {
    pub(super) fn new(
        name: &'static str,
        left: Pattern,
        right: Pattern,
        bidirectional: bool,
        guard: Guard,
    ) -> Self {
        Self {
            name,
            left: left.flatten(),
            right: right.flatten(),
            bidirectional,
            guard,
        }
    }
}

pub(super) struct RuleSet {
    rules: Vec<PreparedRule>,
}

struct PreparedRule {
    rule: Rule,
    forward: DirectionScope,
    reverse: DirectionScope,
}

enum DirectionScope {
    All,
    SourceOnce { eligible: Vec<Id>, used: Vec<Id> },
}

impl DirectionScope {
    fn for_direction(cas: &Cas, from: &Pattern, to: &Pattern) -> Self {
        if to.size() <= from.size() {
            return Self::All;
        }
        let eligible = cas
            .source_classes
            .iter()
            .copied()
            .filter(|&id| matches_root(cas, id, from) && source_growth_safe(cas, id, from))
            .collect();
        Self::SourceOnce {
            eligible,
            used: Vec::new(),
        }
    }

    fn allows(&self, cas: &Cas, id: Id) -> bool {
        match self {
            Self::All => true,
            Self::SourceOnce { eligible, used } => {
                let id = cas.graph.find(id);
                eligible.iter().any(|&source| cas.graph.find(source) == id)
                    && !used.iter().any(|&source| cas.graph.find(source) == id)
            }
        }
    }

    fn mark(&mut self, id: Id) {
        if let Self::SourceOnce { used, .. } = self {
            used.push(id);
        }
    }
}

fn source_growth_safe(cas: &Cas, id: Id, from: &Pattern) -> bool {
    if !matches!(from, Pattern::Product(_)) {
        return true;
    }
    cas.graph[cas.graph.find(id)]
        .nodes
        .iter()
        .filter(|node| node.head == Head::Product)
        .all(|node| {
            node.children()
                .iter()
                .filter(|&&child| {
                    cas.graph[cas.graph.find(child)]
                        .nodes
                        .iter()
                        .any(|node| matches!(node.head, Head::Sum | Head::Binary(BinaryOp::Sub)))
                })
                .count()
                <= 1
        })
}

impl RuleSet {
    pub(super) fn new(cas: &Cas, rules: Vec<Rule>) -> Self {
        Self {
            rules: rules
                .into_iter()
                .map(|rule| PreparedRule {
                    forward: DirectionScope::for_direction(cas, &rule.left, &rule.right),
                    reverse: DirectionScope::for_direction(cas, &rule.right, &rule.left),
                    rule,
                })
                .collect(),
        }
    }

    pub(super) fn apply(&mut self, cas: &mut Cas) {
        for prepared in &mut self.rules {
            if cas.graph.total_size() >= cas.budget.nodes {
                break;
            }
            let classes = cas
                .graph
                .classes()
                .map(|class| class.id)
                .collect::<Vec<_>>();
            for id in classes {
                if cas.graph.total_size() >= cas.budget.nodes {
                    break;
                }
                let rule = &prepared.rule;
                if prepared.forward.allows(cas, id) {
                    apply_direction(cas, id, &rule.left, &rule.right, &rule.guard);
                    prepared.forward.mark(id);
                }
                if rule.bidirectional && prepared.reverse.allows(cas, id) {
                    apply_direction(cas, id, &rule.right, &rule.left, &rule.guard);
                    prepared.reverse.mark(id);
                }
            }
            let _ = prepared.rule.name;
        }
    }
}

impl Pattern {
    pub(super) fn size(&self) -> usize {
        match self {
            Self::Capture(_) => 0,
            Self::Real(_) | Self::ImaginaryUnit => 1,
            Self::Unary(_, child) => 1 + child.size(),
            Self::Binary(_, lhs, rhs) | Self::Call2(_, lhs, rhs) => 1 + lhs.size() + rhs.size(),
            Self::Sum(children) | Self::Product(children) => {
                1 + children.iter().map(Self::size).sum::<usize>()
            }
        }
    }

    fn flatten(self) -> Self {
        match self {
            Self::Sum(children) => {
                let mut flat = Vec::new();
                for child in children.into_iter().map(Self::flatten) {
                    if let Self::Sum(nested) = child {
                        flat.extend(nested);
                    } else {
                        flat.push(child);
                    }
                }
                Self::Sum(flat)
            }
            Self::Product(children) => {
                let mut flat = Vec::new();
                for child in children.into_iter().map(Self::flatten) {
                    if let Self::Product(nested) = child {
                        flat.extend(nested);
                    } else {
                        flat.push(child);
                    }
                }
                Self::Product(flat)
            }
            Self::Unary(op, child) => Self::Unary(op, Box::new(child.flatten())),
            Self::Binary(op, lhs, rhs) => {
                Self::Binary(op, Box::new(lhs.flatten()), Box::new(rhs.flatten()))
            }
            Self::Call2(op, lhs, rhs) => {
                Self::Call2(op, Box::new(lhs.flatten()), Box::new(rhs.flatten()))
            }
            other => other,
        }
    }
}

#[derive(Clone, Debug, Default)]
struct Bindings(BTreeMap<&'static str, Id>);

impl Bindings {
    fn bind(&self, name: &'static str, id: Id, cas: &Cas) -> Option<Self> {
        let id = cas.graph.find(id);
        if self
            .0
            .get(name)
            .is_some_and(|bound| cas.graph.find(*bound) != id)
        {
            return None;
        }
        let mut next = self.clone();
        next.0.insert(name, id);
        Some(next)
    }

    fn get(&self, name: &'static str, cas: &Cas) -> Option<Id> {
        self.0.get(name).map(|id| cas.graph.find(*id))
    }
}

impl Guard {
    fn accepts(&self, bindings: &Bindings, cas: &Cas) -> bool {
        let data = |name| bindings.get(name, cas).map(|id| &cas.graph[id].data);
        match self {
            Self::Always => true,
            Self::Real(name) => data(name).is_some_and(|facts| facts.number == NumberClass::Real),
            Self::Scalar(name) => data(name).is_some_and(|facts| is_scalar(facts.kind)),
            Self::Zero(name) => bindings
                .get(name, cas)
                .is_some_and(|id| super::tensor::is_zero(cas, id)),
            Self::Identity(name) => bindings
                .get(name, cas)
                .is_some_and(|id| super::tensor::is_identity(cas, id)),
            Self::Nonzero(name) => bindings.get(name, cas).is_some_and(|id| {
                cas.graph[id].nodes.iter().any(|node| match node.head {
                    Head::Real(value) => f64::from_bits(value) != 0.0,
                    Head::Complex(re, im) => f64::from_bits(re) != 0.0 || f64::from_bits(im) != 0.0,
                    _ => false,
                })
            }),
            Self::And(lhs, rhs) => lhs.accepts(bindings, cas) && rhs.accepts(bindings, cas),
            Self::Or(lhs, rhs) => lhs.accepts(bindings, cas) || rhs.accepts(bindings, cas),
        }
    }
}

fn is_scalar(kind: ValueKind) -> bool {
    matches!(kind, ValueKind::Real | ValueKind::Complex)
}

pub(super) fn apply_direction(cas: &mut Cas, id: Id, from: &Pattern, to: &Pattern, guard: &Guard) {
    let id = cas.graph.find(id);
    let matches = match_root(cas, id, from);
    for matched in matches.into_iter().take(32) {
        if !guard.accepts(&matched.bindings, cas) {
            continue;
        }
        let Some(mut candidate) = emit(cas, to, &matched.bindings) else {
            continue;
        };
        if !matched.rest.is_empty() {
            let mut terms = matched.rest;
            terms.push(candidate);
            if !terms.iter().all(|id| is_scalar(cas.graph[*id].data.kind)) {
                continue;
            }
            terms.sort_unstable();
            candidate = cas.graph.add(Term::new(
                matched.op.expect("AC remainder has an operator"),
                terms,
            ));
        }
        if same_shape(cas.graph[id].data.kind, cas.graph[candidate].data.kind) {
            cas.graph.union(id, candidate);
        }
    }
}

struct RootMatch {
    bindings: Bindings,
    rest: Vec<Id>,
    op: Option<Head>,
}

fn match_root(cas: &Cas, id: Id, pattern: &Pattern) -> Vec<RootMatch> {
    if let Pattern::Sum(patterns) | Pattern::Product(patterns) = pattern {
        let head = if matches!(pattern, Pattern::Sum(_)) {
            Head::Sum
        } else {
            Head::Product
        };
        let mut out = Vec::new();
        for node in &cas.graph[id].nodes {
            if node.head == head && node.children.len() >= patterns.len() {
                match_ac(
                    cas,
                    patterns,
                    node.children(),
                    Bindings::default(),
                    true,
                    &mut out,
                    Some(head.clone()),
                );
            }
        }
        out
    } else {
        match_class(cas, id, pattern, Bindings::default())
            .into_iter()
            .map(|bindings| RootMatch {
                bindings,
                rest: Vec::new(),
                op: None,
            })
            .collect()
    }
}

pub(super) fn matches_root(cas: &Cas, id: Id, pattern: &Pattern) -> bool {
    !match_root(cas, id, pattern).is_empty()
}

fn match_ac(
    cas: &Cas,
    patterns: &[Pattern],
    actual: &[Id],
    start: Bindings,
    allow_rest: bool,
    out: &mut Vec<RootMatch>,
    op: Option<Head>,
) {
    if out.len() >= 32 {
        return;
    }
    if patterns.is_empty() {
        if allow_rest || actual.is_empty() {
            out.push(RootMatch {
                bindings: start,
                rest: actual.to_vec(),
                op,
            });
        }
        return;
    }
    for (index, &id) in actual.iter().enumerate() {
        let remaining = actual
            .iter()
            .enumerate()
            .filter_map(|(other, &id)| (index != other).then_some(id))
            .collect::<Vec<_>>();
        for bindings in match_class(cas, id, &patterns[0], start.clone()) {
            match_ac(
                cas,
                &patterns[1..],
                &remaining,
                bindings,
                allow_rest,
                out,
                op.clone(),
            );
        }
    }
}

fn match_class(cas: &Cas, id: Id, pattern: &Pattern, bindings: Bindings) -> Vec<Bindings> {
    let id = cas.graph.find(id);
    match pattern {
        Pattern::Capture(name) => bindings.bind(name, id, cas).into_iter().collect(),
        Pattern::Real(expected) => cas.graph[id]
            .nodes
            .iter()
            .any(|node| matches!(node.head, Head::Real(bits) if f64::from_bits(bits) == *expected))
            .then_some(bindings)
            .into_iter()
            .collect(),
        Pattern::ImaginaryUnit => cas.graph[id]
            .nodes
            .iter()
            .any(|node| {
                matches!(node.head, Head::Complex(re, im)
                if f64::from_bits(re) == 0.0 && f64::from_bits(im) == 1.0)
            })
            .then_some(bindings)
            .into_iter()
            .collect(),
        Pattern::Unary(expected, child) => cas.graph[id]
            .nodes
            .iter()
            .filter(|node| node.head == Head::Unary(*expected))
            .flat_map(|node| match_class(cas, node.children[0], child, bindings.clone()))
            .take(32)
            .collect(),
        Pattern::Binary(expected, lhs, rhs) => cas.graph[id]
            .nodes
            .iter()
            .filter(|node| node.head == Head::Binary(*expected))
            .flat_map(|node| {
                match_class(cas, node.children[0], lhs, bindings.clone())
                    .into_iter()
                    .flat_map(|state| match_class(cas, node.children[1], rhs, state))
                    .collect::<Vec<_>>()
            })
            .take(32)
            .collect(),
        Pattern::Call2(call, lhs, rhs) => {
            let head = match call {
                Call2::Complex => Head::ComplexParts,
                Call2::MatMul => Head::MatMul,
                Call2::MatVec => Head::MatVec,
                Call2::Dot => Head::Dot,
                Call2::Solve => Head::Solve,
            };
            cas.graph[id]
                .nodes
                .iter()
                .filter(|node| node.head == head)
                .flat_map(|node| {
                    match_class(cas, node.children[0], lhs, bindings.clone())
                        .into_iter()
                        .flat_map(|state| match_class(cas, node.children[1], rhs, state))
                        .collect::<Vec<_>>()
                })
                .take(32)
                .collect()
        }
        Pattern::Sum(patterns) | Pattern::Product(patterns) => {
            let head = if matches!(pattern, Pattern::Sum(_)) {
                Head::Sum
            } else {
                Head::Product
            };
            let mut out = Vec::new();
            for node in &cas.graph[id].nodes {
                if node.head == head && node.children.len() == patterns.len() {
                    match_ac(
                        cas,
                        patterns,
                        node.children(),
                        bindings.clone(),
                        false,
                        &mut out,
                        None,
                    );
                }
            }
            out.into_iter().map(|item| item.bindings).collect()
        }
    }
}

fn emit(cas: &mut Cas, pattern: &Pattern, bindings: &Bindings) -> Option<Id> {
    match pattern {
        Pattern::Capture(name) => bindings.get(name, cas),
        Pattern::Real(value) => Some(cas.graph.add(Term::new(Head::Real(value.to_bits()), []))),
        Pattern::ImaginaryUnit => Some(cas.graph.add(Term::new(
            Head::Complex(0.0f64.to_bits(), 1.0f64.to_bits()),
            [],
        ))),
        Pattern::Unary(op, child) => {
            let child = emit(cas, child, bindings)?;
            is_scalar(cas.graph[child].data.kind)
                .then(|| cas.graph.add(Term::new(Head::Unary(*op), [child])))
        }
        Pattern::Binary(op, lhs, rhs) => {
            let lhs = emit(cas, lhs, bindings)?;
            let rhs = emit(cas, rhs, bindings)?;
            (is_scalar(cas.graph[lhs].data.kind) && is_scalar(cas.graph[rhs].data.kind))
                .then(|| cas.graph.add(Term::new(Head::Binary(*op), [lhs, rhs])))
        }
        Pattern::Sum(terms) | Pattern::Product(terms) => {
            let mut children = terms
                .iter()
                .map(|term| emit(cas, term, bindings))
                .collect::<Option<Vec<_>>>()?;
            if !children
                .iter()
                .all(|id| is_scalar(cas.graph[*id].data.kind))
            {
                return None;
            }
            children.sort_unstable();
            let head = if matches!(pattern, Pattern::Sum(_)) {
                Head::Sum
            } else {
                Head::Product
            };
            Some(cas.graph.add(Term::new(head, children)))
        }
        Pattern::Call2(call, lhs, rhs) => {
            let lhs = emit(cas, lhs, bindings)?;
            let rhs = emit(cas, rhs, bindings)?;
            let (head, valid) = match call {
                Call2::Complex => (
                    Head::ComplexParts,
                    is_scalar(cas.graph[lhs].data.kind) && is_scalar(cas.graph[rhs].data.kind),
                ),
                Call2::MatMul => (
                    Head::MatMul,
                    matches!((cas.graph[lhs].data.kind, cas.graph[rhs].data.kind),
                    (ValueKind::Matrix { cols: a, .. }, ValueKind::Matrix { rows: b, .. }) if a == b),
                ),
                Call2::MatVec => (
                    Head::MatVec,
                    matches!((cas.graph[lhs].data.kind, cas.graph[rhs].data.kind),
                    (ValueKind::Matrix { cols: a, .. }, ValueKind::Vector { len: b }) if a == b),
                ),
                Call2::Dot => (
                    Head::Dot,
                    matches!((cas.graph[lhs].data.kind, cas.graph[rhs].data.kind),
                    (ValueKind::Vector { len: a }, ValueKind::Vector { len: b }) if a == b),
                ),
                Call2::Solve => (
                    Head::Solve,
                    matches!(cas.graph[lhs].data.kind, ValueKind::Matrix { .. }),
                ),
            };
            valid.then(|| cas.graph.add(Term::new(head, [lhs, rhs])))
        }
    }
}
