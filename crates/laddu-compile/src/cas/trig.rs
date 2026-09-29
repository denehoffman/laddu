//! Coefficient-aware trigonometric candidate generation.

use egg::Id;
use laddu_expr::UnaryOp;
use num::complex::Complex64;

use super::{
    Cas, Head,
    builder::{CandidateBuilder, ProductView},
};

pub(super) fn generate(cas: &mut Cas) {
    let sums = cas
        .graph
        .classes()
        .flat_map(|class| {
            class
                .nodes
                .iter()
                .filter(|node| node.head == Head::Sum)
                .cloned()
                .map(move |node| (class.id, node))
        })
        .collect::<Vec<_>>();
    for (id, sum) in sums {
        if cas.graph.total_size() >= cas.budget.nodes {
            break;
        }
        for lhs in 0..sum.children.len() {
            for rhs in lhs + 1..sum.children.len() {
                let Some(left) = trig_term(cas, sum.children[lhs]) else {
                    continue;
                };
                let Some(right) = trig_term(cas, sum.children[rhs]) else {
                    continue;
                };
                let (cos, sin) = match (left.op, right.op) {
                    (UnaryOp::Cos, UnaryOp::Sin) => (left, right),
                    (UnaryOp::Sin, UnaryOp::Cos) => (right, left),
                    _ => continue,
                };
                if cos.angle != sin.angle || cos.other != sin.other {
                    continue;
                }
                let positive = sin.coefficient == Complex64::I * cos.coefficient;
                let negative = sin.coefficient == -Complex64::I * cos.coefficient;
                if !positive && !negative {
                    continue;
                }
                let mut build = CandidateBuilder(cas);
                let imaginary = build.constant(if positive {
                    Complex64::I
                } else {
                    -Complex64::I
                });
                let Some(phase) = build.product(&[imaginary, cos.angle]) else {
                    continue;
                };
                let Some(exponential) = build.unary(UnaryOp::Exp, phase) else {
                    continue;
                };
                let prefactor = ProductView {
                    coefficient: cos.coefficient,
                    real_coefficient: cos.coefficient.im == 0.0,
                    factors: cos.other,
                };
                if prefactor.coefficient == Complex64::ZERO {
                    continue;
                }
                let Some(scale) = prefactor.emit(&mut build) else {
                    continue;
                };
                let Some(replacement) = build.product(&[scale, exponential]) else {
                    continue;
                };
                let mut terms = sum
                    .children
                    .iter()
                    .enumerate()
                    .filter_map(|(index, &term)| (index != lhs && index != rhs).then_some(term))
                    .collect::<Vec<_>>();
                terms.push(replacement);
                if let Some(candidate) = build.sum(&terms) {
                    build.emit_equivalent(id, candidate);
                }
            }
        }
    }
    let powers = cas
        .graph
        .classes()
        .flat_map(|class| {
            class.nodes.iter().filter_map(move |node| match node.head {
                Head::Unary(UnaryOp::PowI(power)) if power > 2 && power % 2 == 0 => {
                    Some((class.id, node.children[0], power))
                }
                _ => None,
            })
        })
        .collect::<Vec<_>>();
    for (id, input, power) in powers {
        if cas.graph.total_size() >= cas.budget.nodes {
            break;
        }
        let input = cas.graph.find(input);
        let half_angle = cas.graph[input]
            .nodes
            .iter()
            .find_map(|node| match node.head {
                Head::Unary(op @ (UnaryOp::Sin | UnaryOp::Cos)) => Some((op, node.children[0])),
                _ => None,
            });
        let Some((op, angle)) = half_angle else {
            continue;
        };
        let view = ProductView::of(cas, angle);
        if view.coefficient != Complex64::from(0.5)
            || view.factors.len() != 1
            || !view.factors.values().all(|exponent| *exponent == 1)
        {
            continue;
        }
        let full_angle = *view.factors.keys().next().expect("one angle factor");
        let mut build = CandidateBuilder(cas);
        let Some(cosine) = build.unary(UnaryOp::Cos, full_angle) else {
            continue;
        };
        let one = build.constant(Complex64::ONE);
        let half = build.constant(Complex64::from(0.5));
        let base = if op == UnaryOp::Sin {
            build.binary(laddu_expr::BinaryOp::Sub, one, cosine)
        } else {
            build.sum(&[one, cosine])
        };
        let Some(base) = base.and_then(|base| build.product(&[half, base])) else {
            continue;
        };
        if let Some(candidate) = build.unary(UnaryOp::PowI(power / 2), base) {
            build.emit_equivalent(id, candidate);
        }
    }
}

struct TrigTerm {
    op: UnaryOp,
    angle: Id,
    coefficient: Complex64,
    other: std::collections::BTreeMap<Id, i32>,
}

fn trig_term(cas: &Cas, id: Id) -> Option<TrigTerm> {
    let product = ProductView::of(cas, id);
    for (&factor, &power) in &product.factors {
        if power != 1 {
            continue;
        }
        for node in &cas.graph[factor].nodes {
            if let Head::Unary(op @ (UnaryOp::Sin | UnaryOp::Cos)) = node.head {
                let angle = cas.graph.find(node.children[0]);
                let mut other = product.factors.clone();
                other.remove(&factor);
                return Some(TrigTerm {
                    op,
                    angle,
                    coefficient: product.coefficient,
                    other,
                });
            }
        }
    }
    None
}
