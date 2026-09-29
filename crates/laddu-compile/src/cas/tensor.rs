//! Indexed tensor semantics. Operators share element projection and contraction
//! construction, so selection and full expansion use the same equations.

use egg::{Id, Language};
use laddu_expr::ValueKind;
use num::complex::Complex64;

use super::{
    Cas, Head,
    builder::{CandidateBuilder, constant},
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
        let mut build = CandidateBuilder(cas);
        let candidate = match node.head {
            Head::Component(index) => component(&mut build, node.children[0], index),
            Head::MatrixElement(row, col) => matrix_element(&mut build, node.children[0], row, col),
            Head::Dot => dot(&mut build, node.children[0], node.children[1]),
            Head::MatVec => matvec(&mut build, node.children[0], node.children[1]),
            Head::MatMul => matmul(&mut build, node.children[0], node.children[1]),
            _ => None,
        };
        if let Some(candidate) = candidate {
            build.emit_equivalent(id, candidate);
        }
        if matches!(node.head, Head::Dot | Head::MatVec | Head::MatMul)
            && let Some(candidate) =
                factor_contraction(&mut build, &node.head, node.children[0], node.children[1])
        {
            build.emit_equivalent(id, candidate);
        }
    }
}

/// If every element contains one shared scalar factor, retain the compact
/// contraction with that factor outside it.
fn factor_contraction(
    build: &mut CandidateBuilder<'_>,
    head: &Head,
    lhs: Id,
    rhs: Id,
) -> Option<Id> {
    for left in [true, false] {
        let input = if left { lhs } else { rhs };
        let Some((factor, base)) = uniform_factor(build, input) else {
            continue;
        };
        let (lhs, rhs) = if left { (base, rhs) } else { (lhs, base) };
        let contraction = match head {
            Head::Dot => build.dot(lhs, rhs),
            Head::MatVec => build.matvec(lhs, rhs),
            Head::MatMul => build.matmul(lhs, rhs),
            _ => None,
        }?;
        return build.scale(contraction, factor);
    }
    None
}

fn uniform_factor(build: &mut CandidateBuilder<'_>, input: Id) -> Option<(Id, Id)> {
    let input = build.0.graph.find(input);
    let kind = build.0.graph[input].data.kind;
    let head = match kind {
        ValueKind::Vector { .. } => Head::Vector,
        ValueKind::Matrix { rows, cols } => Head::Matrix { rows, cols },
        _ => return None,
    };
    let elements = selected_constructor(build.0, input, &head)?;
    let first = *elements.first()?;
    let mut factors = build.0.graph[build.0.graph.find(first)]
        .nodes
        .iter()
        .find(|node| node.head == Head::Product)?
        .children()
        .to_vec();
    factors.sort_by_key(|id| {
        (
            build.0.graph[build.0.graph.find(*id)]
                .data
                .dependency
                .depends_on_event,
            *id,
        )
    });
    for factor in factors {
        let factor = build.0.graph.find(factor);
        let residuals = elements
            .iter()
            .map(|element| {
                let element = build.0.graph.find(*element);
                build.0.graph[element].nodes.iter().find_map(|node| {
                    if node.head != Head::Product {
                        return None;
                    }
                    let position = node
                        .children()
                        .iter()
                        .position(|child| build.0.graph.find(*child) == factor)?;
                    let mut remaining = node.children().to_vec();
                    remaining.remove(position);
                    Some(remaining)
                })
            })
            .collect::<Option<Vec<_>>>();
        let Some(residuals) = residuals else {
            continue;
        };
        let bases = residuals
            .iter()
            .map(|factors| build.product(factors))
            .collect::<Option<Vec<_>>>()?;
        let base = match kind {
            ValueKind::Vector { .. } => build.vector(&bases),
            ValueKind::Matrix { rows, cols } => build.matrix(rows, cols, &bases),
            _ => None,
        }?;
        return Some((factor, base));
    }
    None
}

fn selected_constructor(cas: &Cas, input: Id, head: &Head) -> Option<Vec<Id>> {
    let input = cas.graph.find(input);
    cas.graph[input]
        .nodes
        .iter()
        .find(|node| {
            &node.head == head
                && node
                    .children()
                    .iter()
                    .all(|id| cas.graph.find(*id) != input)
        })
        .map(|node| node.children().to_vec())
}

fn component(build: &mut CandidateBuilder<'_>, input: Id, index: usize) -> Option<Id> {
    if let Some(elements) = selected_constructor(build.0, input, &Head::Vector) {
        return elements.get(index).copied();
    }
    let input = build.0.graph.find(input);
    let nodes = build.0.graph[input].nodes.clone();
    for node in nodes {
        if node.head == Head::MatVec {
            return matvec_at(build, node.children[0], node.children[1], index);
        }
    }
    None
}

fn matrix_element(
    build: &mut CandidateBuilder<'_>,
    input: Id,
    row: usize,
    col: usize,
) -> Option<Id> {
    let ValueKind::Matrix { cols, .. } = build.0.graph[build.0.graph.find(input)].data.kind else {
        return None;
    };
    if let Some(elements) = selected_constructor(
        build.0,
        input,
        &Head::Matrix {
            rows: match build.0.graph[build.0.graph.find(input)].data.kind {
                ValueKind::Matrix { rows, .. } => rows,
                _ => unreachable!(),
            },
            cols,
        },
    ) {
        return elements.get(row * cols + col).copied();
    }
    let input = build.0.graph.find(input);
    let nodes = build.0.graph[input].nodes.clone();
    for node in nodes {
        if node.head == Head::MatMul {
            return matmul_at(build, node.children[0], node.children[1], row, col);
        }
    }
    None
}

fn dot(build: &mut CandidateBuilder<'_>, lhs: Id, rhs: Id) -> Option<Id> {
    let ValueKind::Vector { len } = build.0.graph[build.0.graph.find(lhs)].data.kind else {
        return None;
    };
    if zero_vector(build.0, lhs) || zero_vector(build.0, rhs) {
        return Some(build.constant(Complex64::ZERO));
    }
    if len > 16 {
        return None;
    }
    contraction(
        build,
        len,
        |build, index| build.component(lhs, index),
        |build, index| build.component(rhs, index),
    )
}

fn matvec(build: &mut CandidateBuilder<'_>, matrix: Id, vector: Id) -> Option<Id> {
    let ValueKind::Matrix { rows, cols } = build.0.graph[build.0.graph.find(matrix)].data.kind
    else {
        return None;
    };
    if identity_matrix(build.0, matrix, rows, cols) {
        return Some(vector);
    }
    if zero_matrix(build.0, matrix) || zero_vector(build.0, vector) {
        let zero = build.constant(Complex64::ZERO);
        return build.vector(&vec![zero; rows]);
    }
    if rows.checked_mul(cols)? > 16 {
        return None;
    }
    let elements = (0..rows)
        .map(|row| matvec_at(build, matrix, vector, row))
        .collect::<Option<Vec<_>>>()?;
    build.vector(&elements)
}

fn matvec_at(build: &mut CandidateBuilder<'_>, matrix: Id, vector: Id, row: usize) -> Option<Id> {
    let ValueKind::Matrix { cols, .. } = build.0.graph[build.0.graph.find(matrix)].data.kind else {
        return None;
    };
    if cols > 64 {
        return None;
    }
    contraction(
        build,
        cols,
        |build, col| build.matrix_element(matrix, row, col),
        |build, col| build.component(vector, col),
    )
}

fn matmul(build: &mut CandidateBuilder<'_>, lhs: Id, rhs: Id) -> Option<Id> {
    let ValueKind::Matrix { rows, cols: inner } = build.0.graph[build.0.graph.find(lhs)].data.kind
    else {
        return None;
    };
    let ValueKind::Matrix { cols, .. } = build.0.graph[build.0.graph.find(rhs)].data.kind else {
        return None;
    };
    if identity_matrix(build.0, lhs, rows, inner) {
        return Some(rhs);
    }
    if identity_matrix(build.0, rhs, inner, cols) {
        return Some(lhs);
    }
    if zero_matrix(build.0, lhs) || zero_matrix(build.0, rhs) {
        let zero = build.constant(Complex64::ZERO);
        return build.matrix(rows, cols, &vec![zero; rows * cols]);
    }
    if rows.checked_mul(cols)?.checked_mul(inner)? > 16 {
        return None;
    }
    let mut elements = Vec::with_capacity(rows * cols);
    for row in 0..rows {
        for col in 0..cols {
            elements.push(matmul_at(build, lhs, rhs, row, col)?);
        }
    }
    build.matrix(rows, cols, &elements)
}

fn matmul_at(
    build: &mut CandidateBuilder<'_>,
    lhs: Id,
    rhs: Id,
    row: usize,
    col: usize,
) -> Option<Id> {
    let ValueKind::Matrix { cols: inner, .. } = build.0.graph[build.0.graph.find(lhs)].data.kind
    else {
        return None;
    };
    if inner > 64 {
        return None;
    }
    contraction(
        build,
        inner,
        |build, mid| build.matrix_element(lhs, row, mid),
        |build, mid| build.matrix_element(rhs, mid, col),
    )
}

/// Indexed equation shared by dot products, matrix-vector products, and
/// matrix products: sum_k left(k) * right(k).
fn contraction(
    build: &mut CandidateBuilder<'_>,
    len: usize,
    mut left: impl FnMut(&mut CandidateBuilder<'_>, usize) -> Option<Id>,
    mut right: impl FnMut(&mut CandidateBuilder<'_>, usize) -> Option<Id>,
) -> Option<Id> {
    let mut terms = Vec::with_capacity(len);
    for index in 0..len {
        let lhs = left(build, index)?;
        let rhs = right(build, index)?;
        terms.push(build.product(&[lhs, rhs])?);
    }
    build.sum(&terms)
}

fn zero_vector(cas: &Cas, id: Id) -> bool {
    selected_constructor(cas, id, &Head::Vector).is_some_and(|elements| {
        elements
            .into_iter()
            .all(|id| constant(cas, id) == Some(Complex64::ZERO))
    })
}

fn zero_matrix(cas: &Cas, id: Id) -> bool {
    let id = cas.graph.find(id);
    cas.graph[id].nodes.iter().any(|node| {
        matches!(node.head, Head::Matrix { .. })
            && node
                .children()
                .iter()
                .all(|&id| constant(cas, id) == Some(Complex64::ZERO))
    })
}

fn identity_matrix(cas: &Cas, id: Id, rows: usize, cols: usize) -> bool {
    if rows != cols {
        return false;
    }
    let id = cas.graph.find(id);
    cas.graph[id].nodes.iter().any(|node| {
        node.head == (Head::Matrix { rows, cols })
            && node.children().iter().enumerate().all(|(index, &id)| {
                constant(cas, id)
                    == Some(if index / cols == index % cols {
                        Complex64::ONE
                    } else {
                        Complex64::ZERO
                    })
            })
    })
}

pub(super) fn is_identity(cas: &Cas, id: Id) -> bool {
    match cas.graph[cas.graph.find(id)].data.kind {
        ValueKind::Matrix { rows, cols } => identity_matrix(cas, id, rows, cols),
        _ => false,
    }
}

pub(super) fn is_zero(cas: &Cas, id: Id) -> bool {
    if constant(cas, id) == Some(Complex64::ZERO) {
        return true;
    }
    match cas.graph[cas.graph.find(id)].data.kind {
        ValueKind::Vector { .. } => zero_vector(cas, id),
        ValueKind::Matrix { .. } => zero_matrix(cas, id),
        _ => false,
    }
}
