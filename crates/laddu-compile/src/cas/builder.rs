//! Typed candidate construction and sum/product views.

use std::collections::{BTreeMap, BTreeSet};

use egg::{Id, Language};
use laddu_expr::{BinaryOp, UnaryOp, ValueKind};
use num::complex::Complex64;

use super::{Cas, Head, Term, same_shape};

pub(super) struct CandidateBuilder<'a>(pub(super) &'a mut Cas);

impl CandidateBuilder<'_> {
    pub(super) fn constant(&mut self, value: Complex64) -> Id {
        let head = if value.im == 0.0 && value.im.is_sign_positive() {
            Head::Real(value.re.to_bits())
        } else {
            Head::Complex(value.re.to_bits(), value.im.to_bits())
        };
        self.0.graph.add(Term::new(head, []))
    }

    pub(super) fn unary(&mut self, op: UnaryOp, input: Id) -> Option<Id> {
        let input = self.0.graph.find(input);
        scalar(self.0.graph[input].data.kind)
            .then(|| self.0.graph.add(Term::new(Head::Unary(op), [input])))
    }

    pub(super) fn binary(&mut self, op: BinaryOp, lhs: Id, rhs: Id) -> Option<Id> {
        let lhs = self.0.graph.find(lhs);
        let rhs = self.0.graph.find(rhs);
        (scalar(self.0.graph[lhs].data.kind) && scalar(self.0.graph[rhs].data.kind))
            .then(|| self.0.graph.add(Term::new(Head::Binary(op), [lhs, rhs])))
    }

    pub(super) fn component(&mut self, vector: Id, index: usize) -> Option<Id> {
        let vector = self.0.graph.find(vector);
        matches!(self.0.graph[vector].data.kind, ValueKind::Vector { len } if index < len).then(
            || {
                self.0
                    .graph
                    .add(Term::new(Head::Component(index), [vector]))
            },
        )
    }

    pub(super) fn matrix_element(&mut self, matrix: Id, row: usize, col: usize) -> Option<Id> {
        let matrix = self.0.graph.find(matrix);
        matches!(self.0.graph[matrix].data.kind,
            ValueKind::Matrix { rows, cols } if row < rows && col < cols)
        .then(|| {
            self.0
                .graph
                .add(Term::new(Head::MatrixElement(row, col), [matrix]))
        })
    }

    pub(super) fn vector(&mut self, elements: &[Id]) -> Option<Id> {
        let elements = elements
            .iter()
            .map(|id| self.0.graph.find(*id))
            .collect::<Vec<_>>();
        elements
            .iter()
            .all(|id| scalar(self.0.graph[*id].data.kind))
            .then(|| self.0.graph.add(Term::new(Head::Vector, elements)))
    }

    pub(super) fn matrix(&mut self, rows: usize, cols: usize, elements: &[Id]) -> Option<Id> {
        if elements.len() != rows.checked_mul(cols)? {
            return None;
        }
        let elements = elements
            .iter()
            .map(|id| self.0.graph.find(*id))
            .collect::<Vec<_>>();
        elements
            .iter()
            .all(|id| scalar(self.0.graph[*id].data.kind))
            .then(|| {
                self.0
                    .graph
                    .add(Term::new(Head::Matrix { rows, cols }, elements))
            })
    }

    pub(super) fn dot(&mut self, lhs: Id, rhs: Id) -> Option<Id> {
        let lhs = self.0.graph.find(lhs);
        let rhs = self.0.graph.find(rhs);
        matches!(
            (self.0.graph[lhs].data.kind, self.0.graph[rhs].data.kind),
            (ValueKind::Vector { len: a }, ValueKind::Vector { len: b }) if a == b
        )
        .then(|| self.0.graph.add(Term::new(Head::Dot, [lhs, rhs])))
    }

    pub(super) fn matvec(&mut self, matrix: Id, vector: Id) -> Option<Id> {
        let matrix = self.0.graph.find(matrix);
        let vector = self.0.graph.find(vector);
        matches!(
            (self.0.graph[matrix].data.kind, self.0.graph[vector].data.kind),
            (ValueKind::Matrix { cols, .. }, ValueKind::Vector { len }) if cols == len
        )
        .then(|| self.0.graph.add(Term::new(Head::MatVec, [matrix, vector])))
    }

    pub(super) fn matmul(&mut self, lhs: Id, rhs: Id) -> Option<Id> {
        let lhs = self.0.graph.find(lhs);
        let rhs = self.0.graph.find(rhs);
        matches!(
            (self.0.graph[lhs].data.kind, self.0.graph[rhs].data.kind),
            (ValueKind::Matrix { cols, .. }, ValueKind::Matrix { rows, .. }) if cols == rows
        )
        .then(|| self.0.graph.add(Term::new(Head::MatMul, [lhs, rhs])))
    }

    /// Broadcasts one scalar over a scalar, vector, or matrix value.
    pub(super) fn scale(&mut self, value: Id, factor: Id) -> Option<Id> {
        let value = self.0.graph.find(value);
        let factor = self.0.graph.find(factor);
        if !scalar(self.0.graph[factor].data.kind) {
            return None;
        }
        match self.0.graph[value].data.kind {
            kind if scalar(kind) => self.product(&[factor, value]),
            ValueKind::Vector { len } if len <= 16 => {
                let elements = (0..len)
                    .map(|index| {
                        let element = self.component(value, index)?;
                        self.product(&[factor, element])
                    })
                    .collect::<Option<Vec<_>>>()?;
                self.vector(&elements)
            }
            ValueKind::Matrix { rows, cols } if rows.checked_mul(cols)? <= 16 => {
                let elements = (0..rows)
                    .flat_map(|row| (0..cols).map(move |col| (row, col)))
                    .map(|(row, col)| {
                        let element = self.matrix_element(value, row, col)?;
                        self.product(&[factor, element])
                    })
                    .collect::<Option<Vec<_>>>()?;
                self.matrix(rows, cols, &elements)
            }
            _ => None,
        }
    }

    pub(super) fn sum(&mut self, terms: &[Id]) -> Option<Id> {
        self.associative(Head::Sum, terms)
    }

    pub(super) fn product(&mut self, factors: &[Id]) -> Option<Id> {
        self.associative(Head::Product, factors)
    }

    fn associative(&mut self, head: Head, children: &[Id]) -> Option<Id> {
        let mut canonical = children
            .iter()
            .map(|id| self.0.graph.find(*id))
            .collect::<Vec<_>>();
        if !canonical
            .iter()
            .all(|id| scalar(self.0.graph[*id].data.kind))
        {
            return None;
        }
        match canonical.as_slice() {
            [] => {
                return Some(self.constant(if head == Head::Sum {
                    Complex64::ZERO
                } else {
                    Complex64::ONE
                }));
            }
            [only] => return Some(*only),
            _ => (),
        }
        canonical.sort_unstable();
        Some(self.0.graph.add(Term::new(head, canonical)))
    }

    pub(super) fn emit_equivalent(&mut self, original: Id, candidate: Id) -> bool {
        let original = self.0.graph.find(original);
        let candidate = self.0.graph.find(candidate);
        if !same_shape(
            self.0.graph[original].data.kind,
            self.0.graph[candidate].data.kind,
        ) {
            return false;
        }
        self.0.graph.union(original, candidate)
    }
}

pub(super) fn scalar(kind: ValueKind) -> bool {
    matches!(kind, ValueKind::Real | ValueKind::Complex)
}

pub(super) fn constant(cas: &Cas, id: Id) -> Option<Complex64> {
    let id = cas.graph.find(id);
    cas.graph[id].nodes.iter().find_map(|node| match node.head {
        Head::Real(bits) => Some(Complex64::from(f64::from_bits(bits))),
        Head::Complex(re, im) => Some(Complex64::new(f64::from_bits(re), f64::from_bits(im))),
        _ => None,
    })
}

/// One scalar product as a coefficient and a multiset of integer powers.
/// This view is shared by collection, factorization, and phase analysis.
pub(super) struct ProductView {
    pub(super) coefficient: Complex64,
    pub(super) real_coefficient: bool,
    pub(super) factors: BTreeMap<Id, i32>,
}

impl ProductView {
    pub(super) fn of(cas: &Cas, id: Id) -> Self {
        let id = cas.graph.find(id);
        let mut view = Self {
            coefficient: Complex64::ONE,
            real_coefficient: true,
            factors: BTreeMap::new(),
        };
        if let Some(product) = cas.graph[id]
            .nodes
            .iter()
            .find(|node| node.head == Head::Product)
        {
            for &factor in product.children() {
                view.push(cas, factor, &mut BTreeSet::new());
            }
        } else {
            view.push(cas, id, &mut BTreeSet::new());
        }
        view
    }

    fn push(&mut self, cas: &Cas, id: Id, seen: &mut BTreeSet<Id>) {
        let id = cas.graph.find(id);
        if !seen.insert(id) {
            *self.factors.entry(id).or_default() += 1;
            return;
        }
        if let Some(value) = constant(cas, id) {
            if self.real_coefficient
                && cas.graph[id]
                    .nodes
                    .iter()
                    .any(|node| matches!(node.head, Head::Real(_)))
            {
                self.coefficient = Complex64::from(self.coefficient.re * value.re);
            } else {
                self.real_coefficient = false;
                self.coefficient *= value;
            }
            return;
        }
        if !cas.graph[id].nodes.iter().any(|node| {
            matches!(
                node.head,
                Head::Source(_) | Head::Sum | Head::Binary(BinaryOp::Sub | BinaryOp::Add)
            )
        }) && let Some(nested) = cas.graph[id].nodes.iter().find(|node| {
            node.head == Head::Product
                && node
                    .children()
                    .iter()
                    .all(|child| cas.graph.find(*child) != id)
        }) {
            for &factor in nested.children() {
                self.push(cas, factor, seen);
            }
            return;
        }
        if let Some(power) = cas.graph[id].nodes.iter().find_map(|node| match node.head {
            Head::Unary(UnaryOp::PowI(power)) => Some((node.children[0], power)),
            _ => None,
        }) {
            *self.factors.entry(cas.graph.find(power.0)).or_default() += power.1;
        } else if !cas.graph[id]
            .nodes
            .iter()
            .any(|node| matches!(node.head, Head::Source(_)))
            && let Some(input) = cas.graph[id].nodes.iter().find_map(|node| match node.head {
                Head::Unary(UnaryOp::Neg) => Some(node.children[0]),
                _ => None,
            })
        {
            self.coefficient = if self.real_coefficient {
                Complex64::from(-self.coefficient.re)
            } else {
                -self.coefficient
            };
            self.push(cas, input, seen);
        } else {
            *self.factors.entry(id).or_default() += 1;
        }
    }

    pub(super) fn emit(&self, builder: &mut CandidateBuilder<'_>) -> Option<Id> {
        if self.coefficient == Complex64::ONE
            && self.factors.len() > 1
            && let Some(&common_power) = self.factors.values().next()
            && common_power > 1
            && self.factors.values().all(|power| *power == common_power)
        {
            let bases = self.factors.keys().copied().collect::<Vec<_>>();
            let base = builder.product(&bases)?;
            return builder.unary(UnaryOp::PowI(common_power), base);
        }
        let mut factors = Vec::new();
        if self.coefficient != Complex64::ONE || self.factors.is_empty() {
            factors.push(builder.constant(self.coefficient));
        }
        for (&id, &power) in &self.factors {
            if power == 0 {
                continue;
            }
            factors.push(if power == 1 {
                id
            } else {
                builder.unary(UnaryOp::PowI(power), id)?
            });
        }
        builder.product(&factors)
    }
}
