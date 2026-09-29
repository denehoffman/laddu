//! Generic arithmetic analyses over the typed CAS views.

use egg::{Id, Language};
use laddu_expr::{BinaryOp, NumberClass, UnaryOp};
use num::complex::Complex64;

use super::{
    Cas, Head, Term,
    builder::{CandidateBuilder, ProductView, constant},
};

pub(super) fn generate(cas: &mut Cas) {
    let snapshot = cas
        .graph
        .classes()
        .flat_map(|class| {
            class
                .nodes
                .iter()
                .cloned()
                .map(move |node| (class.id, node))
        })
        .collect::<Vec<_>>();
    for (id, node) in snapshot {
        if cas.graph.total_size() >= cas.budget.nodes {
            break;
        }
        fold_constants(cas, id, &node);
        normalize(cas, id, &node);
        if node.head == Head::Sum {
            collect_sum(cas, id, &node);
        }
    }
}

fn fold_constants(cas: &mut Cas, id: Id, node: &Term) {
    let folded = match node.head {
        Head::Unary(op) => constant(cas, node.children[0]).map(|input| op.evaluate(input)),
        Head::Binary(op) => constant(cas, node.children[0])
            .zip(constant(cas, node.children[1]))
            .map(|(lhs, rhs)| op.evaluate(lhs, rhs)),
        Head::Sum => node
            .children
            .iter()
            .copied()
            .map(|child| constant(cas, child))
            .try_fold(Complex64::ZERO, |sum, value| Some(sum + value?)),
        Head::Product => node
            .children
            .iter()
            .copied()
            .map(|child| constant(cas, child))
            .try_fold(Complex64::ONE, |product, value| Some(product * value?)),
        _ => None,
    };
    if let Some(value) = folded {
        let mut build = CandidateBuilder(cas);
        let candidate = build.constant(value);
        build.emit_equivalent(id, candidate);
    }
}

fn normalize(cas: &mut Cas, id: Id, node: &Term) {
    match node.head {
        Head::Product => {
            flatten_associative(cas, id, node);
            let view = ProductView::of(cas, id);
            let mut build = CandidateBuilder(cas);
            if let Some(candidate) = view.emit(&mut build) {
                build.emit_equivalent(id, candidate);
            }
        }
        Head::Sum => flatten_associative(cas, id, node),
        Head::Binary(BinaryOp::Sub) => {
            let mut build = CandidateBuilder(cas);
            if let Some(negated) = build.unary(UnaryOp::Neg, node.children[1])
                && let Some(candidate) = build.sum(&[node.children[0], negated])
            {
                build.emit_equivalent(id, candidate);
            }
        }
        Head::Binary(BinaryOp::Div) => {
            let mut build = CandidateBuilder(cas);
            if let Some(inverse) = build.unary(UnaryOp::PowI(-1), node.children[1])
                && let Some(candidate) = build.product(&[node.children[0], inverse])
            {
                build.emit_equivalent(id, candidate);
            }
        }
        Head::Unary(UnaryOp::Neg) => {
            let mut build = CandidateBuilder(cas);
            let minus_one = build.constant(Complex64::from(-1.0));
            if let Some(candidate) = build.product(&[minus_one, node.children[0]]) {
                build.emit_equivalent(id, candidate);
            }
        }
        Head::Unary(UnaryOp::PowI(outer)) => {
            let input = cas.graph.find(node.children[0]);
            let inner = cas.graph[input]
                .nodes
                .iter()
                .find_map(|term| match term.head {
                    Head::Unary(UnaryOp::PowI(inner)) => Some((term.children[0], inner)),
                    _ => None,
                });
            if let Some((base, inner)) = inner
                && let Some(combined) = inner.checked_mul(outer)
            {
                let mut build = CandidateBuilder(cas);
                if let Some(candidate) = build.unary(UnaryOp::PowI(combined), base) {
                    build.emit_equivalent(id, candidate);
                }
            }
        }
        Head::Unary(UnaryOp::NormSqr) => {
            let input = cas.graph.find(node.children[0]);
            let parts = cas.graph[input].nodes.iter().find_map(|term| {
                (term.head == Head::ComplexParts).then(|| (term.children[0], term.children[1]))
            });
            if let Some((re, im)) = parts
                && cas.graph[re].data.number == NumberClass::Real
                && cas.graph[im].data.number == NumberClass::Real
            {
                let mut build = CandidateBuilder(cas);
                if let (Some(re2), Some(im2)) = (
                    build.unary(UnaryOp::PowI(2), re),
                    build.unary(UnaryOp::PowI(2), im),
                ) && let Some(candidate) = build.sum(&[re2, im2])
                {
                    build.emit_equivalent(id, candidate);
                }
            }
        }
        _ => (),
    }
}

fn flatten_associative(cas: &mut Cas, id: Id, node: &Term) {
    for (index, &child) in node.children().iter().enumerate() {
        let child = cas.graph.find(child);
        let nested = cas.graph[child]
            .nodes
            .iter()
            .find(|other| {
                other.head == node.head
                    && other
                        .children()
                        .iter()
                        .all(|id| cas.graph.find(*id) != child)
            })
            .cloned();
        let Some(nested) = nested else { continue };
        let mut children = node
            .children()
            .iter()
            .enumerate()
            .filter_map(|(at, &child)| (at != index).then_some(child))
            .collect::<Vec<_>>();
        children.extend_from_slice(nested.children());
        let mut build = CandidateBuilder(cas);
        let candidate = if node.head == Head::Sum {
            build.sum(&children)
        } else {
            build.product(&children)
        };
        if let Some(candidate) = candidate {
            build.emit_equivalent(id, candidate);
        }
    }
}

fn collect_sum(cas: &mut Cas, id: Id, node: &Term) {
    let mut constants = Complex64::ZERO;
    let mut constant_count = 0;
    let mut others = Vec::new();
    for &term in node.children() {
        if let Some(value) = constant(cas, term) {
            constants += value;
            constant_count += 1;
        } else {
            others.push(term);
        }
    }
    if constant_count > 0 && (constant_count > 1 || constants == Complex64::ZERO) {
        let mut build = CandidateBuilder(cas);
        if constants != Complex64::ZERO || others.is_empty() {
            others.push(build.constant(constants));
        }
        if let Some(candidate) = build.sum(&others) {
            build.emit_equivalent(id, candidate);
        }
    }

    let terms = node.children();
    for lhs_index in 0..terms.len() {
        for rhs_index in lhs_index + 1..terms.len() {
            let lhs = ProductView::of(cas, terms[lhs_index]);
            let rhs = ProductView::of(cas, terms[rhs_index]);
            let shared = lhs
                .factors
                .iter()
                .filter_map(|(&factor, &left_power)| {
                    rhs.factors
                        .get(&factor)
                        .map(|&right_power| (factor, left_power.min(right_power)))
                })
                .filter(|(_, power)| *power > 0)
                .collect::<Vec<_>>();
            let numeric = common_numeric(lhs.coefficient, rhs.coefficient);
            if shared.is_empty() && numeric == Complex64::ONE {
                continue;
            }
            let mut left_remainder = lhs;
            let mut right_remainder = rhs;
            let mut common = ProductView {
                coefficient: numeric,
                real_coefficient: numeric.im == 0.0,
                factors: Default::default(),
            };
            left_remainder.coefficient /= numeric;
            right_remainder.coefficient /= numeric;
            if left_remainder.real_coefficient && numeric.im == 0.0 {
                left_remainder.coefficient.im = 0.0;
            }
            if right_remainder.real_coefficient && numeric.im == 0.0 {
                right_remainder.coefficient.im = 0.0;
            }
            for (factor, power) in shared {
                *left_remainder
                    .factors
                    .get_mut(&factor)
                    .expect("left factor exists") -= power;
                *right_remainder
                    .factors
                    .get_mut(&factor)
                    .expect("right factor exists") -= power;
                common.factors.insert(factor, power);
            }
            let mut build = CandidateBuilder(cas);
            let (Some(left), Some(right), Some(common)) = (
                left_remainder.emit(&mut build),
                right_remainder.emit(&mut build),
                common.emit(&mut build),
            ) else {
                continue;
            };
            let Some(remainders) = build.sum(&[left, right]) else {
                continue;
            };
            let Some(factored) = build.product(&[common, remainders]) else {
                continue;
            };
            let mut replacement = terms
                .iter()
                .enumerate()
                .filter_map(|(index, &term)| {
                    (index != lhs_index && index != rhs_index).then_some(term)
                })
                .collect::<Vec<_>>();
            replacement.push(factored);
            if let Some(candidate) = build.sum(&replacement) {
                build.emit_equivalent(id, candidate);
            }
        }
    }
}

fn common_numeric(lhs: Complex64, rhs: Complex64) -> Complex64 {
    if lhs == Complex64::ZERO || rhs == Complex64::ZERO {
        return Complex64::ONE;
    }
    fn integer(value: f64) -> Option<i64> {
        (value.is_finite() && value.fract() == 0.0 && value.abs() < 1.0e9).then_some(value as i64)
    }
    let (left, right, imaginary) = if lhs.im == 0.0 && rhs.im == 0.0 {
        (lhs.re, rhs.re, false)
    } else if lhs.re == 0.0 && rhs.re == 0.0 {
        (lhs.im, rhs.im, true)
    } else {
        return Complex64::ONE;
    };
    let Some(mut left) = integer(left).map(i64::unsigned_abs) else {
        return Complex64::ONE;
    };
    let Some(mut right) = integer(right).map(i64::unsigned_abs) else {
        return Complex64::ONE;
    };
    while right != 0 {
        (left, right) = (right, left % right);
    }
    if left <= 1 && !imaginary {
        Complex64::ONE
    } else if imaginary {
        Complex64::new(0.0, left as f64)
    } else {
        Complex64::from(left as f64)
    }
}
