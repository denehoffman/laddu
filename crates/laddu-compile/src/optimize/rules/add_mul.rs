use laddu_expr::{BinaryOp, ExprId, ExprMetadata, ExprNode, ExprSourceKind, UnaryOp};
use num::complex::Complex64;

use crate::CompileResult;

use super::super::rewrite::alias_or_preserve;
use super::super::{
    CombineLikeTermsRule, FactorCommonProductRule, Rewrite, RewriteContext, RewriteRule,
};
use super::is_scalar_value;

#[derive(Copy, Clone, Debug, Default)]
pub(crate) struct NormalizeAddMulRule;

impl RewriteRule for NormalizeAddMulRule {
    fn name(&self) -> &'static str {
        "normalize-add-mul"
    }

    fn rewrite(
        &self,
        node: &ExprNode,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite> {
        match node {
            ExprNode::Binary {
                op: BinaryOp::Add,
                lhs,
                rhs,
            } => Ok(Rewrite::Replace {
                node: ExprNode::NaryAdd {
                    terms: vec![*lhs, *rhs],
                },
                metadata: metadata.clone(),
            }),
            ExprNode::Binary {
                op: BinaryOp::Sub,
                lhs,
                rhs,
            } => self.normalize_subtraction(*lhs, *rhs, metadata, context),
            ExprNode::Binary {
                op: BinaryOp::Mul,
                lhs,
                rhs,
            } => self.normalize_product(&[*lhs, *rhs], metadata, context),
            ExprNode::Binary {
                op: BinaryOp::Div,
                lhs,
                rhs,
            } if is_scalar_value(context, *lhs) && is_scalar_value(context, *rhs) => {
                self.normalize_division(*lhs, *rhs, metadata, context)
            }
            ExprNode::NaryAdd { terms } => self.normalize_sum(terms, metadata, context),
            ExprNode::NaryMul { factors } => self.normalize_product(factors, metadata, context),
            _ => Ok(Rewrite::Keep),
        }
    }
}

impl NormalizeAddMulRule {
    fn normalize_subtraction(
        &self,
        lhs: ExprId,
        rhs: ExprId,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite> {
        let mut builder = ReplacementFragment::new(context);
        let rhs = builder.negated_term(rhs);
        builder.push(
            ExprNode::NaryAdd {
                terms: vec![lhs, rhs],
            },
            metadata.clone(),
        );
        Ok(builder.into_rewrite())
    }

    fn normalize_division(
        &self,
        lhs: ExprId,
        rhs: ExprId,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite> {
        let mut builder = ReplacementFragment::new(context);
        let reciprocal = builder.push(
            ExprNode::Unary {
                op: UnaryOp::PowI(-1),
                input: rhs,
            },
            ExprMetadata::new(ExprSourceKind::Unary),
        );
        builder.push(
            ExprNode::NaryMul {
                factors: vec![lhs, reciprocal],
            },
            metadata.clone(),
        );
        Ok(builder.into_rewrite())
    }

    fn normalize_sum(
        &self,
        terms: &[ExprId],
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite> {
        let mut builder = ReplacementFragment::new(context);
        let mut normalized = Vec::new();
        for term in terms {
            match context.node(*term) {
                Some(ExprNode::NaryAdd { terms }) => normalized.extend(terms.iter().copied()),
                Some(node) if ExprNode::is_zero(node) => {}
                _ if let Some(scaled_terms) =
                    self.scaled_sum_terms(*term, context, &mut builder) =>
                {
                    normalized.extend(scaled_terms);
                }
                _ => normalized.push(*term),
            }
        }

        Ok(match normalized.as_slice() {
            [] => Rewrite::Replace {
                node: ExprNode::RealConst(0.0),
                metadata: metadata.clone(),
            },
            [term] if builder.is_empty() => alias_or_preserve(*term, metadata, context),
            [_] => builder.into_rewrite(),
            _ if normalized == terms && builder.is_empty() => Rewrite::Keep,
            _ if builder.is_empty() => Rewrite::Replace {
                node: ExprNode::NaryAdd { terms: normalized },
                metadata: metadata.clone(),
            },
            _ => {
                builder.push(ExprNode::NaryAdd { terms: normalized }, metadata.clone());
                builder.into_rewrite()
            }
        })
    }

    fn scaled_sum_terms(
        &self,
        id: ExprId,
        context: &RewriteContext<'_>,
        builder: &mut ReplacementFragment<'_>,
    ) -> Option<Vec<ExprId>> {
        let term = ProductTerm::from_id(id, context);
        let coefficient = term.signed_coefficient();
        let factors = term.sorted_factor_key();
        let [PowerFactor { base, exponent: 1 }] = factors.as_slice() else {
            return None;
        };
        let Some(ExprNode::NaryAdd { terms }) = context.node(*base) else {
            return None;
        };
        Some(
            terms
                .iter()
                .map(|term| {
                    if ProductTerm::approx_eq(coefficient, 1.0) {
                        *term
                    } else {
                        ProductTerm::push_parts(
                            coefficient,
                            &[PowerFactor {
                                base: *term,
                                exponent: 1,
                            }],
                            builder,
                        )
                    }
                })
                .collect(),
        )
    }

    fn normalize_product(
        &self,
        factors: &[ExprId],
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite> {
        if let Some(rewrite) =
            self.absorb_additive_factor_coefficient(factors, metadata, context)?
        {
            return Ok(rewrite);
        }
        if let Some(rewrite) = self.combine_same_power_factors(factors, metadata, context)? {
            return Ok(rewrite);
        }
        let mut collector = ProductCollector::new(context);
        collector.collect_all(factors);
        Ok(collector.into_rewrite(factors, metadata))
    }

    fn combine_same_power_factors(
        &self,
        factors: &[ExprId],
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Option<Rewrite>> {
        let Some((power, selected)) = self.largest_same_power_group(factors, context) else {
            return Ok(None);
        };
        let mut builder = ReplacementFragment::new(context);
        let inner_factors = selected.iter().map(|index| {
            let Some(ExprNode::Unary {
                op: UnaryOp::PowI(_),
                input,
            }) = context.node(factors[*index])
            else {
                unreachable!("selected factors are powers")
            };
            *input
        });
        let inner = builder.push(
            ExprNode::NaryMul {
                factors: inner_factors.collect(),
            },
            ExprMetadata::new(ExprSourceKind::Binary),
        );
        let combined = builder.push(
            ExprNode::Unary {
                op: UnaryOp::PowI(power),
                input: inner,
            },
            ExprMetadata::new(ExprSourceKind::Unary),
        );
        let first_selected = selected[0];
        let mut output_factors = Vec::with_capacity(factors.len() - selected.len() + 1);
        for (index, factor) in factors.iter().enumerate() {
            if index == first_selected {
                output_factors.push(combined);
            } else if !selected.contains(&index) {
                output_factors.push(*factor);
            }
        }
        builder.push(
            ExprNode::NaryMul {
                factors: output_factors,
            },
            metadata.clone(),
        );
        let Rewrite::ReplaceMany { nodes } = builder.into_rewrite() else {
            unreachable!("replacement builder always produces fragments")
        };
        let original = ExprNode::NaryMul {
            factors: factors.to_vec(),
        };
        let original_cost = context.local_node_cost(original, metadata.clone())?;
        let candidate_cost = context.local_fragment_cost(&nodes)?;
        if candidate_cost.is_better_than(&original_cost) {
            Ok(Some(Rewrite::ReplaceMany { nodes }))
        } else {
            Ok(None)
        }
    }

    fn largest_same_power_group(
        &self,
        factors: &[ExprId],
        context: &RewriteContext<'_>,
    ) -> Option<(i32, Vec<usize>)> {
        let mut best = None;
        for (index, factor) in factors.iter().enumerate() {
            let Some(ExprNode::Unary {
                op: UnaryOp::PowI(power),
                input,
            }) = context.node(*factor)
            else {
                continue;
            };
            if *power <= 1 || !is_scalar_value(context, *input) {
                continue;
            }
            let selected: Vec<_> = factors
                .iter()
                .enumerate()
                .skip(index)
                .filter_map(|(candidate_index, candidate)| {
                    matches!(
                        context.node(*candidate),
                        Some(ExprNode::Unary {
                            op: UnaryOp::PowI(candidate_power),
                            input,
                        }) if candidate_power == power && is_scalar_value(context, *input)
                    )
                    .then_some(candidate_index)
                })
                .collect();
            if selected.len() >= 2
                && best
                    .as_ref()
                    .is_none_or(|(_, best_selected): &(i32, Vec<usize>)| {
                        selected.len() > best_selected.len()
                    })
            {
                best = Some((*power, selected));
            }
        }
        best
    }

    fn absorb_additive_factor_coefficient(
        &self,
        factors: &[ExprId],
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Option<Rewrite>> {
        let Some((scalar_index, scalar)) =
            factors
                .iter()
                .enumerate()
                .find_map(|(index, factor)| match context.node(*factor) {
                    Some(ExprNode::RealConst(value)) if value.is_finite() => Some((index, *value)),
                    _ => None,
                })
        else {
            return Ok(None);
        };

        for (add_index, factor) in factors.iter().enumerate() {
            let Some(ExprNode::NaryAdd { terms }) = context.node(*factor) else {
                continue;
            };
            let mut product_terms: Vec<_> = terms
                .iter()
                .map(|term| ProductTerm::from_id(*term, context))
                .collect();
            let common_coefficient =
                ProductTerm::take_common_coefficient_from_all(&mut product_terms);
            if ProductTerm::approx_eq(common_coefficient, 1.0) {
                continue;
            }

            let mut builder = ReplacementFragment::new(context);
            let normalized_terms = product_terms
                .iter()
                .map(|term| term.push_remainder(&mut builder))
                .collect();
            let normalized_add = builder.push(
                ExprNode::NaryAdd {
                    terms: normalized_terms,
                },
                ExprMetadata::new(ExprSourceKind::Binary),
            );
            let absorbed_scalar = builder.push(
                ExprNode::RealConst(scalar * common_coefficient),
                ExprMetadata::new(ExprSourceKind::Const),
            );
            let normalized_factors = factors
                .iter()
                .enumerate()
                .map(|(index, factor)| {
                    if index == scalar_index {
                        absorbed_scalar
                    } else if index == add_index {
                        normalized_add
                    } else {
                        *factor
                    }
                })
                .collect();
            builder.push(
                ExprNode::NaryMul {
                    factors: normalized_factors,
                },
                metadata.clone(),
            );
            let Rewrite::ReplaceMany { nodes } = builder.into_rewrite() else {
                unreachable!("replacement builder always produces fragments")
            };
            let original = ExprNode::NaryMul {
                factors: factors.to_vec(),
            };
            let original_cost = context.local_node_cost(original, metadata.clone())?;
            let candidate_cost = context.local_fragment_cost(&nodes)?;
            if candidate_cost.is_no_worse_than(&original_cost) {
                return Ok(Some(Rewrite::ReplaceMany { nodes }));
            }
        }

        Ok(None)
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum ProductPiece {
    Power { base: ExprId, exponent: i32 },
    Const(Complex64),
}

pub(super) struct ProductCollector<'a> {
    context: &'a RewriteContext<'a>,
    coefficient: Complex64,
    pieces: Vec<ProductPiece>,
    zero: bool,
    changed: bool,
}

impl<'a> ProductCollector<'a> {
    pub(super) fn new(context: &'a RewriteContext<'a>) -> Self {
        Self {
            context,
            coefficient: Complex64::ONE,
            pieces: Vec::new(),
            zero: false,
            changed: false,
        }
    }

    pub(super) fn collect_all(&mut self, factors: &[ExprId]) {
        for factor in factors {
            self.collect_factor(*factor);
        }
    }

    fn collect_factor(&mut self, id: ExprId) {
        if self.zero {
            return;
        }
        match self.context.node(id) {
            Some(ExprNode::NaryMul { factors }) => {
                self.changed = true;
                for factor in factors {
                    self.collect_factor(*factor);
                }
            }
            Some(ExprNode::Binary {
                op: BinaryOp::Mul,
                lhs,
                rhs,
            }) => {
                self.changed = true;
                self.collect_factor(*lhs);
                self.collect_factor(*rhs);
            }
            Some(ExprNode::Unary {
                op: UnaryOp::Neg,
                input,
            }) => {
                self.changed = true;
                self.coefficient = -self.coefficient;
                self.collect_factor(*input);
            }
            Some(ExprNode::Unary {
                op: UnaryOp::PowI(power),
                input,
            }) if self.is_scalar(*input) => {
                self.changed = true;
                self.collect_power(*input, *power);
            }
            Some(ExprNode::RealConst(value)) => self.collect_real_const(*value),
            Some(ExprNode::ComplexConst(value)) => {
                self.changed = true;
                self.coefficient *= *value;
                if self.coefficient == Complex64::ZERO {
                    self.zero = true;
                }
            }
            _ if self.is_scalar(id) => self.collect_power(id, 1),
            _ => self.pieces.push(ProductPiece::Power {
                base: id,
                exponent: 1,
            }),
        }
    }

    fn collect_power(&mut self, base: ExprId, exponent: i32) {
        if exponent == 0 {
            self.changed = true;
            return;
        }

        if let Some(ProductPiece::Power {
            exponent: existing,
            ..
        }) = self
            .pieces
            .iter_mut()
            .find(|piece| matches!(piece, ProductPiece::Power { base: candidate, .. } if *candidate == base))
        {
            *existing += exponent;
            self.changed = true;
        } else {
            self.pieces.push(ProductPiece::Power { base, exponent });
        }
    }

    fn collect_real_const(&mut self, value: f64) {
        if value == 0.0 {
            self.changed = true;
            self.zero = true;
            return;
        }

        self.changed = true;
        self.coefficient *= value;
    }

    pub(super) fn into_rewrite(mut self, original: &[ExprId], metadata: &ExprMetadata) -> Rewrite {
        if self.zero || self.coefficient == Complex64::ZERO {
            return Rewrite::Replace {
                node: ExprNode::RealConst(0.0),
                metadata: metadata.clone(),
            };
        }

        self.remove_cancelled_powers();

        if self.coefficient != Complex64::ONE || self.pieces.is_empty() {
            self.pieces.insert(0, ProductPiece::Const(self.coefficient));
        }

        if self.has_original_shape(original) {
            return Rewrite::Keep;
        }

        match self.pieces.as_slice() {
            [] => {
                return Rewrite::Replace {
                    node: ExprNode::RealConst(1.0),
                    metadata: metadata.clone(),
                };
            }
            [ProductPiece::Power { base, exponent: 1 }] => {
                return alias_or_preserve(*base, metadata, self.context);
            }
            [ProductPiece::Const(value)] => {
                return Rewrite::Replace {
                    node: ExprNode::from(*value),
                    metadata: metadata.clone(),
                };
            }
            _ => {}
        }

        let mut builder = ReplacementFragment::new(self.context);
        let mut factors = Vec::with_capacity(self.pieces.len());
        for piece in self.pieces {
            match piece {
                ProductPiece::Power { base, exponent: 1 } => factors.push(base),
                ProductPiece::Power { base, exponent } => factors.push(builder.push(
                    ExprNode::Unary {
                        op: UnaryOp::PowI(exponent),
                        input: base,
                    },
                    ExprMetadata::new(ExprSourceKind::Unary),
                )),
                ProductPiece::Const(value) => factors.push(builder.push(
                    ExprNode::from(value),
                    ExprMetadata::new(ExprSourceKind::Const),
                )),
            }
        }
        builder.push(ExprNode::NaryMul { factors }, metadata.clone());
        builder.into_rewrite()
    }

    fn remove_cancelled_powers(&mut self) {
        self.pieces.retain(|piece| match piece {
            ProductPiece::Power { exponent, .. } if *exponent == 0 => {
                self.changed = true;
                false
            }
            _ => true,
        });
    }

    fn has_original_shape(&self, original: &[ExprId]) -> bool {
        !self.changed
            && self.pieces.len() == original.len()
            && self
                .pieces
                .iter()
                .zip(original)
                .all(|(piece, original)| piece.as_identity_factor() == Some(*original))
    }

    fn is_scalar(&self, id: ExprId) -> bool {
        is_scalar_value(self.context, id)
    }
}

impl ProductPiece {
    fn as_identity_factor(&self) -> Option<ExprId> {
        match self {
            Self::Power { base, exponent: 1 } => Some(*base),
            Self::Power { .. } | Self::Const(_) => None,
        }
    }
}

pub(super) struct ReplacementFragment<'a> {
    pub(super) context: &'a RewriteContext<'a>,
    nodes: Vec<(ExprNode, ExprMetadata)>,
}

impl<'a> ReplacementFragment<'a> {
    pub(super) fn new(context: &'a RewriteContext<'a>) -> Self {
        Self {
            context,
            nodes: Vec::new(),
        }
    }

    pub(super) fn push(&mut self, node: ExprNode, metadata: ExprMetadata) -> ExprId {
        let id = self.next_id();
        self.nodes.push((node, metadata));
        id
    }

    pub(super) fn negated_term(&mut self, id: ExprId) -> ExprId {
        let minus_one = self.push(
            ExprNode::RealConst(-1.0),
            ExprMetadata::new(ExprSourceKind::Const),
        );
        self.push(
            ExprNode::NaryMul {
                factors: vec![minus_one, id],
            },
            ExprMetadata::new(ExprSourceKind::Binary),
        )
    }

    pub(super) fn conjugated_term(&mut self, id: ExprId) -> ExprId {
        self.push(
            ExprNode::Unary {
                op: UnaryOp::Conj,
                input: id,
            },
            ExprMetadata::new(ExprSourceKind::Unary),
        )
    }

    fn next_id(&self) -> ExprId {
        ExprId::from_index(self.context.next_id().index() + self.nodes.len())
    }

    pub(super) fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub(super) fn into_rewrite(self) -> Rewrite {
        Rewrite::ReplaceMany { nodes: self.nodes }
    }
}

impl RewriteRule for CombineLikeTermsRule {
    fn name(&self) -> &'static str {
        "combine-like-terms"
    }

    fn rewrite(
        &self,
        node: &ExprNode,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite> {
        match node {
            ExprNode::NaryAdd { terms } => Ok(self.combine_terms(terms, metadata, context)),
            _ => Ok(Rewrite::Keep),
        }
    }
}

impl CombineLikeTermsRule {
    fn combine_terms(
        &self,
        terms: &[ExprId],
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> Rewrite {
        let mut groups: Vec<LikeTermGroup> = Vec::new();
        let mut changed = false;

        for term in terms {
            let product = ProductTerm::from_id(*term, context);
            let coefficient = product.signed_coefficient();
            if coefficient == 0.0 {
                changed = true;
                continue;
            }

            let key = product.sorted_factor_key();
            if let Some(group) = groups.iter_mut().find(|group| group.factors == key) {
                group.coefficient += coefficient;
                changed = true;
            } else {
                groups.push(LikeTermGroup {
                    coefficient,
                    factors: key,
                });
            }
        }

        let mut builder = ReplacementFragment::new(context);
        let mut combined = Vec::new();
        for group in groups {
            if ProductTerm::approx_eq(group.coefficient, 0.0) {
                changed = true;
                continue;
            }
            combined.push(ProductTerm::push_parts(
                group.coefficient,
                &group.factors,
                &mut builder,
            ));
        }

        if !changed {
            return Rewrite::Keep;
        }

        match combined.as_slice() {
            [] => Rewrite::Replace {
                node: ExprNode::RealConst(0.0),
                metadata: metadata.clone(),
            },
            [term] if builder.is_empty() => alias_or_preserve(*term, metadata, context),
            [_] => builder.into_rewrite(),
            _ => {
                builder.push(ExprNode::NaryAdd { terms: combined }, metadata.clone());
                builder.into_rewrite()
            }
        }
    }
}

#[derive(Clone, Debug)]
struct LikeTermGroup {
    coefficient: f64,
    factors: Vec<PowerFactor>,
}

#[derive(Clone, Debug)]
struct FactorCandidate {
    nodes: Vec<(ExprNode, ExprMetadata)>,
}

impl RewriteRule for FactorCommonProductRule {
    fn name(&self) -> &'static str {
        "factor-common-product"
    }

    fn rewrite(
        &self,
        node: &ExprNode,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite> {
        match node {
            ExprNode::Binary {
                op: BinaryOp::Add,
                lhs,
                rhs,
            } => self.factor_common_products(node, &[*lhs, *rhs], metadata, context),
            ExprNode::NaryAdd { terms } if terms.len() >= 2 => {
                self.factor_common_products(node, terms, metadata, context)
            }
            _ => Ok(Rewrite::Keep),
        }
    }
}

impl FactorCommonProductRule {
    fn factor_common_products(
        &self,
        original: &ExprNode,
        terms: &[ExprId],
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite> {
        let Some(nodes) =
            self.best_common_product_factor_nodes(original, terms, metadata, context)?
        else {
            return Ok(Rewrite::Keep);
        };
        Ok(Rewrite::ReplaceMany { nodes })
    }

    fn best_common_product_factor_nodes(
        &self,
        original: &ExprNode,
        term_ids: &[ExprId],
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Option<Vec<(ExprNode, ExprMetadata)>>> {
        let candidates = [
            self.common_product_factor_candidate(term_ids, metadata, context),
            self.partial_common_product_factor_candidate(term_ids, metadata, context),
        ];
        let original_cost = context.local_node_cost(original.clone(), metadata.clone())?;
        let mut best_cost = None;
        let mut best_nodes = None;

        for candidate in candidates.into_iter().flatten() {
            let candidate_cost = context.local_fragment_cost(&candidate.nodes)?;
            if best_cost
                .as_ref()
                .is_none_or(|best_cost| candidate_cost.is_better_than(best_cost))
            {
                best_cost = Some(candidate_cost);
                best_nodes = Some(candidate.nodes);
            }
        }

        let Some(best_cost) = best_cost else {
            return Ok(None);
        };
        if !best_cost.is_no_worse_than(&original_cost) && context.ids_are_all_constants(term_ids) {
            return Ok(None);
        }
        Ok(best_nodes)
    }

    fn common_product_factor_candidate(
        &self,
        term_ids: &[ExprId],
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> Option<FactorCandidate> {
        let mut terms: Vec<_> = term_ids
            .iter()
            .map(|term| ProductTerm::from_id(*term, context))
            .collect();
        let common = ProductTerm::take_common_factors_from_all(&mut terms);
        let common_coefficient = ProductTerm::take_common_coefficient_from_all(&mut terms);
        if common.is_empty() && ProductTerm::approx_eq(common_coefficient, 1.0) {
            return None;
        }

        let mut builder = ReplacementFragment::new(context);
        let remainder_terms = terms
            .iter()
            .map(|term| term.push_remainder(&mut builder))
            .collect();
        let sum = builder.push(
            ExprNode::NaryAdd {
                terms: remainder_terms,
            },
            ExprMetadata::new(ExprSourceKind::Binary),
        );
        let mut factors = Vec::new();
        if !ProductTerm::approx_eq(common_coefficient, 1.0) {
            factors.push(builder.push(
                ExprNode::RealConst(common_coefficient),
                ExprMetadata::new(ExprSourceKind::Const),
            ));
        }
        factors.extend(common.iter().map(|factor| factor.emit(&mut builder)));
        factors.push(sum);
        builder.push(ExprNode::NaryMul { factors }, metadata.clone());
        let Rewrite::ReplaceMany { nodes } = builder.into_rewrite() else {
            unreachable!("replacement builder always produces fragments")
        };
        Some(FactorCandidate { nodes })
    }

    fn partial_common_product_factor_candidate(
        &self,
        term_ids: &[ExprId],
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> Option<FactorCandidate> {
        let terms: Vec<_> = term_ids
            .iter()
            .map(|term| ProductTerm::from_id(*term, context))
            .collect();
        let common = Self::best_partial_common_factors(&terms)?;
        let selected_indices: Vec<_> = terms
            .iter()
            .enumerate()
            .filter_map(|(index, term)| term.has_factors(&common).then_some(index))
            .collect();
        if selected_indices.len() < 2 || selected_indices.len() == terms.len() {
            return None;
        }

        let mut remainders = terms.clone();
        for index in &selected_indices {
            for factor in &common {
                remainders[*index].remove_factor(*factor);
            }
        }

        let mut builder = ReplacementFragment::new(context);
        let subset_terms = selected_indices
            .iter()
            .map(|index| remainders[*index].push_remainder(&mut builder))
            .collect();
        let subset_sum = builder.push(
            ExprNode::NaryAdd {
                terms: subset_terms,
            },
            ExprMetadata::new(ExprSourceKind::Binary),
        );
        let mut factored_factors: Vec<_> = common
            .iter()
            .map(|factor| factor.emit(&mut builder))
            .collect();
        factored_factors.push(subset_sum);
        let factored_term = builder.push(
            ExprNode::NaryMul {
                factors: factored_factors,
            },
            ExprMetadata::new(ExprSourceKind::Binary),
        );

        let mut output_terms = Vec::with_capacity(term_ids.len() - selected_indices.len() + 1);
        for (index, term_id) in term_ids.iter().enumerate() {
            if !selected_indices.contains(&index) {
                output_terms.push(*term_id);
            }
        }
        output_terms.push(factored_term);
        builder.push(
            ExprNode::NaryAdd {
                terms: output_terms,
            },
            metadata.clone(),
        );
        let Rewrite::ReplaceMany { nodes } = builder.into_rewrite() else {
            unreachable!("replacement builder always produces fragments")
        };
        Some(FactorCandidate { nodes })
    }

    fn best_partial_common_factors(terms: &[ProductTerm]) -> Option<Vec<PowerFactor>> {
        let mut best = Vec::new();
        for lhs_index in 0..terms.len() {
            for rhs_index in (lhs_index + 1)..terms.len() {
                let mut pair = vec![terms[lhs_index].clone(), terms[rhs_index].clone()];
                let common = ProductTerm::take_common_factors_from_all(&mut pair);
                if common.len() > best.len() {
                    best = common;
                }
            }
        }
        (!best.is_empty()).then_some(best)
    }
}

#[derive(Clone, Debug)]
pub(super) struct ProductTerm {
    pub(super) sign: f64,
    pub(super) coefficient: f64,
    pub(super) factors: Vec<PowerFactor>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) struct PowerFactor {
    pub(super) base: ExprId,
    pub(super) exponent: i32,
}

impl ProductTerm {
    pub(super) fn from_id(id: ExprId, context: &RewriteContext<'_>) -> Self {
        let mut term = Self {
            sign: 1.0,
            coefficient: 1.0,
            factors: Vec::new(),
        };
        term.collect(id, context);
        term
    }

    fn collect(&mut self, id: ExprId, context: &RewriteContext<'_>) {
        match context.node(id) {
            Some(ExprNode::NaryMul { factors }) => {
                for factor in factors {
                    self.collect(*factor, context);
                }
            }
            Some(ExprNode::Binary {
                op: BinaryOp::Mul,
                lhs,
                rhs,
            }) => {
                self.collect(*lhs, context);
                self.collect(*rhs, context);
            }
            Some(ExprNode::Unary {
                op: UnaryOp::Neg,
                input,
            }) => {
                self.sign *= -1.0;
                self.collect(*input, context);
            }
            Some(ExprNode::Unary {
                op: UnaryOp::PowI(power),
                input,
            }) if is_scalar_value(context, *input) => {
                if !self.collect_powered_product(*input, *power, context) {
                    self.collect_power(*input, *power);
                }
            }
            Some(ExprNode::RealConst(-1.0)) => {
                self.sign *= -1.0;
            }
            Some(ExprNode::RealConst(value)) if *value < 0.0 => {
                self.sign *= -1.0;
                self.coefficient *= -*value;
            }
            Some(ExprNode::RealConst(value)) => {
                self.coefficient *= *value;
            }
            _ => self.collect_power(id, 1),
        }
    }

    fn collect_power(&mut self, base: ExprId, exponent: i32) {
        if exponent == 0 {
            return;
        }

        if let Some(existing) = self.factors.iter_mut().find(|factor| factor.base == base)
            && let Some(exponent) = existing.exponent.checked_add(exponent)
        {
            existing.exponent = exponent;
            return;
        }

        self.factors.push(PowerFactor { base, exponent });
    }

    fn collect_powered_product(
        &mut self,
        input: ExprId,
        exponent: i32,
        context: &RewriteContext<'_>,
    ) -> bool {
        if !matches!(context.node(input), Some(ExprNode::NaryMul { .. })) {
            return false;
        }
        let inner = Self::from_id(input, context);
        let coefficient = inner.signed_coefficient().powi(exponent);
        if coefficient < 0.0 {
            self.sign *= -1.0;
            self.coefficient *= -coefficient;
        } else {
            self.coefficient *= coefficient;
        }
        for factor in inner.factors {
            let Some(exponent) = factor.exponent.checked_mul(exponent) else {
                return false;
            };
            self.collect_power(factor.base, exponent);
        }
        true
    }

    pub(super) fn signed_coefficient(&self) -> f64 {
        self.sign * self.coefficient
    }

    pub(super) fn sorted_factor_key(&self) -> Vec<PowerFactor> {
        let mut factors: Vec<_> = self
            .factors
            .iter()
            .copied()
            .filter(|factor| factor.exponent != 0)
            .collect();
        factors.sort_by_key(|factor| (factor.base.index(), factor.exponent));
        factors
    }

    fn push_remainder(&self, builder: &mut ReplacementFragment<'_>) -> ExprId {
        Self::push_parts(self.signed_coefficient(), &self.factors, builder)
    }

    pub(super) fn push_parts(
        coefficient: f64,
        source_factors: &[PowerFactor],
        builder: &mut ReplacementFragment<'_>,
    ) -> ExprId {
        let mut factors = Vec::new();
        for factor in source_factors {
            if factor.exponent == 0 {
                continue;
            }
            factors.push(factor.emit(builder));
        }

        if !Self::approx_eq(coefficient, 1.0) || factors.is_empty() {
            let coefficient = builder.push(
                ExprNode::RealConst(coefficient),
                ExprMetadata::new(ExprSourceKind::Const),
            );
            factors.insert(0, coefficient);
        }

        match factors.as_slice() {
            [factor] => *factor,
            _ => builder.push(
                ExprNode::NaryMul { factors },
                ExprMetadata::new(ExprSourceKind::Binary),
            ),
        }
    }

    fn take_common_factors_from_all(terms: &mut [Self]) -> Vec<PowerFactor> {
        let Some((first, rest)) = terms.split_first() else {
            return Vec::new();
        };

        let mut common = first.sorted_factor_key();
        for term in rest {
            common = common
                .into_iter()
                .filter_map(|factor| {
                    term.factors
                        .iter()
                        .find_map(|candidate| factor.common_with(candidate))
                })
                .collect();
            if common.is_empty() {
                return common;
            }
        }

        for term in terms {
            for factor in &common {
                term.remove_factor(*factor);
            }
        }

        common
    }

    fn take_common_coefficient_from_all(terms: &mut [Self]) -> f64 {
        let Some((first, rest)) = terms.split_first() else {
            return 1.0;
        };

        let mut common = first.coefficient;
        for term in rest {
            common = Self::common_real_coefficient(common, term.coefficient);
            if Self::approx_eq(common, 1.0) {
                return 1.0;
            }
        }

        if !Self::approx_eq(common, 1.0) {
            for term in terms {
                term.coefficient /= common;
            }
        }
        common
    }

    fn remove_factor(&mut self, factor: PowerFactor) {
        let Some(index) = self.factors.iter().position(|candidate| {
            candidate.base == factor.base
                && candidate.exponent.signum() == factor.exponent.signum()
                && candidate.exponent.abs() >= factor.exponent.abs()
        }) else {
            return;
        };

        self.factors[index].exponent -= factor.exponent;
        if self.factors[index].exponent == 0 {
            self.factors.remove(index);
        }
    }

    fn has_factors(&self, factors: &[PowerFactor]) -> bool {
        factors.iter().all(|factor| {
            self.factors.iter().any(|candidate| {
                candidate.base == factor.base
                    && candidate.exponent.signum() == factor.exponent.signum()
                    && candidate.exponent.abs() >= factor.exponent.abs()
            })
        })
    }

    fn common_real_coefficient(lhs: f64, rhs: f64) -> f64 {
        if lhs == 0.0 || rhs == 0.0 || !lhs.is_finite() || !rhs.is_finite() {
            return 1.0;
        }

        let lhs = lhs.abs();
        let rhs = rhs.abs();
        let lhs_integer = lhs.round();
        let rhs_integer = rhs.round();
        if Self::approx_eq(lhs, lhs_integer)
            && Self::approx_eq(rhs, rhs_integer)
            && lhs_integer > 0.0
            && rhs_integer > 0.0
            && lhs_integer <= u64::MAX as f64
            && rhs_integer <= u64::MAX as f64
        {
            return Self::gcd_u64(lhs_integer as u64, rhs_integer as u64) as f64;
        }

        let smaller = lhs.min(rhs);
        let larger = lhs.max(rhs);
        let ratio = larger / smaller;
        if Self::approx_eq(ratio, ratio.round()) {
            smaller
        } else {
            1.0
        }
    }

    pub(super) fn approx_eq(lhs: f64, rhs: f64) -> bool {
        (lhs - rhs).abs() <= f64::EPSILON * lhs.abs().max(rhs.abs()).max(1.0) * 16.0
    }

    fn gcd_u64(mut lhs: u64, mut rhs: u64) -> u64 {
        while rhs != 0 {
            let remainder = lhs % rhs;
            lhs = rhs;
            rhs = remainder;
        }
        lhs
    }
}

impl PowerFactor {
    fn common_with(&self, other: &Self) -> Option<Self> {
        if self.base != other.base || self.exponent.signum() != other.exponent.signum() {
            return None;
        }

        Some(Self {
            base: self.base,
            exponent: self.exponent.abs().min(other.exponent.abs()) * self.exponent.signum(),
        })
    }

    fn emit(&self, builder: &mut ReplacementFragment<'_>) -> ExprId {
        match self.exponent {
            1 => self.base,
            exponent => builder.push(
                ExprNode::Unary {
                    op: UnaryOp::PowI(exponent),
                    input: self.base,
                },
                ExprMetadata::new(ExprSourceKind::Unary),
            ),
        }
    }
}
