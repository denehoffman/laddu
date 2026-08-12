use laddu_expr::{BinaryOp, ExprId, ExprMetadata, ExprNode, ExprSourceKind, UnaryOp};

use crate::CompileResult;

use super::super::{Rewrite, RewriteContext, RewriteRule, TrigIdentityRule};

use super::add_mul::{PowerFactor, ProductTerm, ReplacementFragment};

impl RewriteRule for TrigIdentityRule {
    fn name(&self) -> &'static str {
        "trig-identity"
    }

    fn rewrite(
        &self,
        node: &ExprNode,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite> {
        match node {
            ExprNode::Unary {
                op: op @ (UnaryOp::Sin | UnaryOp::Cos),
                input,
            } => Ok(self.normalize_parity(*op, *input, metadata, context)),
            ExprNode::Unary {
                op: UnaryOp::PowI(power),
                input,
            } if *power > 0 && *power % 2 == 0 => {
                Ok(self.simplify_even_trig_power(*power, *input, metadata, context))
            }
            ExprNode::Binary {
                op: BinaryOp::Add,
                lhs,
                rhs,
            } => Ok(self.simplify_sum(&[*lhs, *rhs], metadata, context)),
            ExprNode::Binary {
                op: BinaryOp::Sub,
                lhs,
                rhs,
            } if context.node(*lhs).is_some_and(ExprNode::is_one) => {
                Ok(self.simplify_one_minus(*rhs, metadata, context))
            }
            ExprNode::NaryAdd { terms } => Ok(self.simplify_sum(terms, metadata, context)),
            _ => Ok(Rewrite::Keep),
        }
    }
}

impl TrigIdentityRule {
    fn simplify_sum(
        &self,
        terms: &[ExprId],
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> Rewrite {
        if let Some(rewrite) = self.simplify_affine_cos_pair(terms, metadata, context) {
            return rewrite;
        }
        if let Some(rewrite) = self.simplify_half_angle_pair(terms, metadata, context) {
            return rewrite;
        }
        if let Some(rewrite) = self.simplify_sin_cos_pair(terms, metadata, context) {
            return rewrite;
        }
        self.simplify_one_minus_trig_square(terms, metadata, context)
    }

    fn normalize_parity(
        &self,
        op: UnaryOp,
        input: ExprId,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> Rewrite {
        let Some(positive_input) = self.negated_input(input, context) else {
            return Rewrite::Keep;
        };
        match op {
            UnaryOp::Cos => Rewrite::Replace {
                node: ExprNode::Unary {
                    op: UnaryOp::Cos,
                    input: positive_input,
                },
                metadata: metadata.clone(),
            },
            UnaryOp::Sin => {
                let mut builder = ReplacementFragment::new(context);
                let sin = builder.push(
                    ExprNode::Unary {
                        op: UnaryOp::Sin,
                        input: positive_input,
                    },
                    ExprMetadata::new(ExprSourceKind::Unary),
                );
                builder.push(
                    ExprNode::Unary {
                        op: UnaryOp::Neg,
                        input: sin,
                    },
                    metadata.clone(),
                );
                builder.into_rewrite()
            }
            _ => Rewrite::Keep,
        }
    }

    fn simplify_affine_cos_pair(
        &self,
        terms: &[ExprId],
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> Option<Rewrite> {
        for (lhs_index, lhs) in terms.iter().enumerate() {
            let Some(lhs) = self.affine_cos_term(*lhs, context) else {
                continue;
            };
            for (rhs_index, rhs) in terms.iter().enumerate().skip(lhs_index + 1) {
                let Some(rhs) = self.affine_cos_term(*rhs, context) else {
                    continue;
                };
                if lhs.input == rhs.input {
                    return Some(self.replace_affine_cos_pair(
                        terms, lhs_index, lhs, rhs_index, rhs, metadata, context,
                    ));
                }
            }
        }
        None
    }

    #[allow(clippy::too_many_arguments)]
    fn replace_affine_cos_pair(
        &self,
        terms: &[ExprId],
        lhs_index: usize,
        lhs: AffineCosTerm,
        rhs_index: usize,
        rhs: AffineCosTerm,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> Rewrite {
        let constant = lhs.constant + rhs.constant;
        let cos_coefficient = lhs.cos_coefficient + rhs.cos_coefficient;

        let mut builder = ReplacementFragment::new(context);
        let mut replacement_terms = Vec::new();
        if !ProductTerm::approx_eq(constant, 0.0) {
            replacement_terms.push(builder.push(
                ExprNode::RealConst(constant),
                ExprMetadata::new(ExprSourceKind::Const),
            ));
        }
        if !ProductTerm::approx_eq(cos_coefficient, 0.0) {
            let cos = builder.push(
                ExprNode::Unary {
                    op: UnaryOp::Cos,
                    input: lhs.input,
                },
                ExprMetadata::new(ExprSourceKind::Unary),
            );
            replacement_terms.push(ProductTerm::push_parts(
                cos_coefficient,
                &[PowerFactor {
                    base: cos,
                    exponent: 1,
                }],
                &mut builder,
            ));
        }

        let replacement = match replacement_terms.as_slice() {
            [] => builder.push(
                ExprNode::RealConst(0.0),
                ExprMetadata::new(ExprSourceKind::Const),
            ),
            [term] => *term,
            _ => builder.push(
                ExprNode::NaryAdd {
                    terms: replacement_terms,
                },
                ExprMetadata::new(ExprSourceKind::Binary),
            ),
        };

        let mut new_terms: Vec<_> = terms
            .iter()
            .enumerate()
            .filter_map(|(index, term)| {
                if index == lhs_index || index == rhs_index {
                    None
                } else {
                    Some(*term)
                }
            })
            .collect();
        if new_terms.is_empty() {
            builder.into_rewrite()
        } else {
            new_terms.push(replacement);
            builder.push(ExprNode::NaryAdd { terms: new_terms }, metadata.clone());
            builder.into_rewrite()
        }
    }

    fn simplify_even_trig_power(
        &self,
        power: i32,
        input: ExprId,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> Rewrite {
        let Some((op, half_input)) = self
            .trig_call(input, context)
            .and_then(|(op, input)| self.half_angle_input(input, context).map(|half| (op, half)))
        else {
            return Rewrite::Keep;
        };
        self.replace_half_angle_power(op, half_input, power, metadata, context)
    }

    fn replace_half_angle_power(
        &self,
        op: TrigSquareOp,
        input: ExprId,
        power: i32,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> Rewrite {
        let mut builder = ReplacementFragment::new(context);
        let one = builder.push(
            ExprNode::RealConst(1.0),
            ExprMetadata::new(ExprSourceKind::Const),
        );
        let cos = builder.push(
            ExprNode::Unary {
                op: UnaryOp::Cos,
                input,
            },
            ExprMetadata::new(ExprSourceKind::Unary),
        );
        let signed_cos = match op {
            TrigSquareOp::Sin => builder.negated_term(cos),
            TrigSquareOp::Cos => cos,
        };
        let sum = builder.push(
            ExprNode::NaryAdd {
                terms: vec![one, signed_cos],
            },
            ExprMetadata::new(ExprSourceKind::Binary),
        );
        let half = builder.push(
            ExprNode::RealConst(0.5),
            ExprMetadata::new(ExprSourceKind::Const),
        );
        let square = builder.push(
            ExprNode::NaryMul {
                factors: vec![half, sum],
            },
            if power == 2 {
                metadata.clone()
            } else {
                ExprMetadata::new(ExprSourceKind::Binary)
            },
        );
        if power == 2 {
            return builder.into_rewrite();
        }
        builder.push(
            ExprNode::Unary {
                op: UnaryOp::PowI(power / 2),
                input: square,
            },
            metadata.clone(),
        );
        builder.into_rewrite()
    }

    fn simplify_half_angle_pair(
        &self,
        terms: &[ExprId],
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> Option<Rewrite> {
        for (lhs_index, lhs) in terms.iter().enumerate() {
            let Some(lhs) = self.half_angle_square_term(*lhs, context) else {
                continue;
            };
            for (rhs_index, rhs) in terms.iter().enumerate().skip(lhs_index + 1) {
                let Some(rhs) = self.half_angle_square_term(*rhs, context) else {
                    continue;
                };
                if lhs.input == rhs.input && lhs.op.is_complement(rhs.op) {
                    return Some(self.replace_half_angle_pair(
                        terms, lhs_index, lhs, rhs_index, rhs, metadata, context,
                    ));
                }
            }
        }
        None
    }

    #[allow(clippy::too_many_arguments)]
    fn replace_half_angle_pair(
        &self,
        terms: &[ExprId],
        lhs_index: usize,
        lhs: TrigSquareTerm,
        rhs_index: usize,
        rhs: TrigSquareTerm,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> Rewrite {
        let (sin_coefficient, cos_coefficient) = match (lhs.op, rhs.op) {
            (TrigSquareOp::Sin, TrigSquareOp::Cos) => (lhs.coefficient, rhs.coefficient),
            (TrigSquareOp::Cos, TrigSquareOp::Sin) => (rhs.coefficient, lhs.coefficient),
            _ => return Rewrite::Keep,
        };
        let constant = 0.5 * (sin_coefficient + cos_coefficient);
        let cosine_coefficient = 0.5 * (cos_coefficient - sin_coefficient);

        let mut builder = ReplacementFragment::new(context);
        let mut replacement_terms = Vec::new();
        if !ProductTerm::approx_eq(constant, 0.0) {
            replacement_terms.push(builder.push(
                ExprNode::RealConst(constant),
                ExprMetadata::new(ExprSourceKind::Const),
            ));
        }
        if !ProductTerm::approx_eq(cosine_coefficient, 0.0) {
            let cos = builder.push(
                ExprNode::Unary {
                    op: UnaryOp::Cos,
                    input: lhs.input,
                },
                ExprMetadata::new(ExprSourceKind::Unary),
            );
            replacement_terms.push(ProductTerm::push_parts(
                cosine_coefficient,
                &[PowerFactor {
                    base: cos,
                    exponent: 1,
                }],
                &mut builder,
            ));
        }

        let replacement = match replacement_terms.as_slice() {
            [] => builder.push(
                ExprNode::RealConst(0.0),
                ExprMetadata::new(ExprSourceKind::Const),
            ),
            [term] => *term,
            _ => builder.push(
                ExprNode::NaryAdd {
                    terms: replacement_terms,
                },
                ExprMetadata::new(ExprSourceKind::Binary),
            ),
        };

        let mut new_terms: Vec<_> = terms
            .iter()
            .enumerate()
            .filter_map(|(index, term)| {
                if index == lhs_index || index == rhs_index {
                    None
                } else {
                    Some(*term)
                }
            })
            .collect();
        if new_terms.is_empty() {
            builder.into_rewrite()
        } else {
            new_terms.push(replacement);
            builder.push(ExprNode::NaryAdd { terms: new_terms }, metadata.clone());
            builder.into_rewrite()
        }
    }

    fn simplify_sin_cos_pair(
        &self,
        terms: &[ExprId],
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> Option<Rewrite> {
        for (lhs_index, lhs) in terms.iter().enumerate() {
            let Some((lhs_op, lhs_input)) = self.trig_square(*lhs, context) else {
                continue;
            };
            for (rhs_index, rhs) in terms.iter().enumerate().skip(lhs_index + 1) {
                let Some((rhs_op, rhs_input)) = self.trig_square(*rhs, context) else {
                    continue;
                };
                if lhs_input == rhs_input && lhs_op.is_complement(rhs_op) {
                    return Some(
                        self.replace_pair_with_one(terms, lhs_index, rhs_index, metadata, context),
                    );
                }
            }
        }
        None
    }

    fn replace_pair_with_one(
        &self,
        terms: &[ExprId],
        lhs_index: usize,
        rhs_index: usize,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> Rewrite {
        let remaining: Vec<_> = terms
            .iter()
            .enumerate()
            .filter_map(|(index, term)| {
                if index == lhs_index || index == rhs_index {
                    None
                } else {
                    Some(*term)
                }
            })
            .collect();

        if remaining.is_empty() {
            return Rewrite::Replace {
                node: ExprNode::RealConst(1.0),
                metadata: metadata.clone(),
            };
        }

        let mut builder = ReplacementFragment::new(context);
        let one = builder.push(
            ExprNode::RealConst(1.0),
            ExprMetadata::new(ExprSourceKind::Const),
        );
        let mut terms = remaining;
        terms.push(one);
        builder.push(ExprNode::NaryAdd { terms }, metadata.clone());
        builder.into_rewrite()
    }

    fn simplify_one_minus_trig_square(
        &self,
        terms: &[ExprId],
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> Rewrite {
        let Some(one_index) = terms
            .iter()
            .position(|term| context.node(*term).is_some_and(ExprNode::is_one))
        else {
            return Rewrite::Keep;
        };

        for (index, term) in terms.iter().enumerate() {
            if index == one_index {
                continue;
            }
            let Some((op, input)) = self.negative_trig_square(*term, context) else {
                continue;
            };
            let remaining: Vec<_> = terms
                .iter()
                .enumerate()
                .filter_map(|(term_index, term)| {
                    if term_index == one_index || term_index == index {
                        None
                    } else {
                        Some(*term)
                    }
                })
                .collect();
            return self.replace_one_minus_term(remaining, op, input, metadata, context);
        }

        Rewrite::Keep
    }

    fn simplify_one_minus(
        &self,
        rhs: ExprId,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> Rewrite {
        let Some((op, input)) = self.trig_square(rhs, context) else {
            return Rewrite::Keep;
        };
        let mut builder = ReplacementFragment::new(context);
        self.push_complement_square(op, input, metadata.clone(), &mut builder);
        builder.into_rewrite()
    }

    fn replace_one_minus_term(
        &self,
        remaining: Vec<ExprId>,
        op: TrigSquareOp,
        input: ExprId,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> Rewrite {
        let mut builder = ReplacementFragment::new(context);
        if remaining.is_empty() {
            self.push_complement_square(op, input, metadata.clone(), &mut builder);
            return builder.into_rewrite();
        }

        let mut terms = remaining;
        let replacement = self.push_complement_square(
            op,
            input,
            ExprMetadata::new(ExprSourceKind::Unary),
            &mut builder,
        );
        terms.push(replacement);
        builder.push(ExprNode::NaryAdd { terms }, metadata.clone());
        builder.into_rewrite()
    }

    fn push_complement_square(
        &self,
        op: TrigSquareOp,
        input: ExprId,
        metadata: ExprMetadata,
        builder: &mut ReplacementFragment<'_>,
    ) -> ExprId {
        let trig = builder.push(
            ExprNode::Unary {
                op: op.complement().into_unary_op(),
                input,
            },
            ExprMetadata::new(ExprSourceKind::Unary),
        );
        builder.push(
            ExprNode::Unary {
                op: UnaryOp::PowI(2),
                input: trig,
            },
            metadata,
        )
    }

    fn trig_square(
        &self,
        id: ExprId,
        context: &RewriteContext<'_>,
    ) -> Option<(TrigSquareOp, ExprId)> {
        let ExprNode::Unary {
            op: UnaryOp::PowI(2),
            input,
        } = context.node(id)?
        else {
            return None;
        };
        self.trig_call(*input, context)
    }

    fn negative_trig_square(
        &self,
        id: ExprId,
        context: &RewriteContext<'_>,
    ) -> Option<(TrigSquareOp, ExprId)> {
        match context.node(id)? {
            ExprNode::Unary {
                op: UnaryOp::Neg,
                input,
            } => self.trig_square(*input, context),
            ExprNode::NaryMul { factors } => {
                let mut has_minus_one = false;
                let mut square = None;
                for factor in factors {
                    match context.node(*factor) {
                        Some(ExprNode::RealConst(-1.0)) if !has_minus_one => {
                            has_minus_one = true;
                        }
                        _ if square.is_none() => {
                            square = self.trig_square(*factor, context);
                        }
                        _ => return None,
                    }
                }
                has_minus_one.then_some(square).flatten()
            }
            _ => None,
        }
    }

    fn trig_call(
        &self,
        id: ExprId,
        context: &RewriteContext<'_>,
    ) -> Option<(TrigSquareOp, ExprId)> {
        match context.node(id)? {
            ExprNode::Unary {
                op: UnaryOp::Sin,
                input,
            } => Some((TrigSquareOp::Sin, *input)),
            ExprNode::Unary {
                op: UnaryOp::Cos,
                input,
            } => Some((TrigSquareOp::Cos, *input)),
            _ => None,
        }
    }

    fn half_angle_input(&self, id: ExprId, context: &RewriteContext<'_>) -> Option<ExprId> {
        let term = ProductTerm::from_id(id, context);
        if !ProductTerm::approx_eq(term.signed_coefficient(), 0.5) {
            return None;
        }
        match term.sorted_factor_key().as_slice() {
            [PowerFactor { base, exponent: 1 }] => Some(*base),
            _ => None,
        }
    }

    fn negated_input(&self, id: ExprId, context: &RewriteContext<'_>) -> Option<ExprId> {
        let term = ProductTerm::from_id(id, context);
        if !ProductTerm::approx_eq(term.signed_coefficient(), -1.0) {
            return None;
        }
        match term.sorted_factor_key().as_slice() {
            [PowerFactor { base, exponent: 1 }] => Some(*base),
            _ => None,
        }
    }

    fn half_angle_square_term(
        &self,
        id: ExprId,
        context: &RewriteContext<'_>,
    ) -> Option<TrigSquareTerm> {
        let term = ProductTerm::from_id(id, context);
        let factors = term.sorted_factor_key();
        let mut square = None;
        let mut remaining = Vec::new();
        for factor in factors {
            if square.is_none()
                && factor.exponent == 1
                && let Some((op, input)) =
                    self.trig_square(factor.base, context)
                        .and_then(|(op, input)| {
                            self.half_angle_input(input, context).map(|half| (op, half))
                        })
            {
                square = Some((op, input));
            } else {
                remaining.push(factor);
            }
        }
        if !remaining.is_empty() {
            return None;
        }
        let (op, input) = square?;
        Some(TrigSquareTerm {
            op,
            input,
            coefficient: term.signed_coefficient(),
        })
    }

    fn affine_cos_term(&self, id: ExprId, context: &RewriteContext<'_>) -> Option<AffineCosTerm> {
        let term = ProductTerm::from_id(id, context);
        let mut scale = term.signed_coefficient();
        let mut affine = None;
        for factor in term.sorted_factor_key() {
            if factor.exponent == 1
                && affine.is_none()
                && let Some(parsed) = self.affine_cos_sum(factor.base, context)
            {
                affine = Some(parsed);
            } else {
                return None;
            }
        }
        let mut affine = affine?;
        scale *= affine.scale;
        affine.scale = 1.0;
        Some(AffineCosTerm {
            input: affine.input,
            scale: 1.0,
            constant: scale * affine.constant,
            cos_coefficient: scale * affine.cos_coefficient,
        })
    }

    fn affine_cos_sum(&self, id: ExprId, context: &RewriteContext<'_>) -> Option<AffineCosTerm> {
        let Some(ExprNode::NaryAdd { terms }) = context.node(id) else {
            return None;
        };
        let mut constant = 0.0;
        let mut cos_coefficient = 0.0;
        let mut input = None;
        for term_id in terms {
            let term = ProductTerm::from_id(*term_id, context);
            let factors = term.sorted_factor_key();
            match factors.as_slice() {
                [] => constant += term.signed_coefficient(),
                [PowerFactor { base, exponent: 1 }] => {
                    let Some((TrigSquareOp::Cos, cos_input)) = self.trig_call(*base, context)
                    else {
                        return None;
                    };
                    if let Some(input) = input {
                        if input != cos_input {
                            return None;
                        }
                    } else {
                        input = Some(cos_input);
                    }
                    cos_coefficient += term.signed_coefficient();
                }
                _ => return None,
            }
        }
        Some(AffineCosTerm {
            input: input?,
            scale: 1.0,
            constant,
            cos_coefficient,
        })
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) enum TrigSquareOp {
    Sin,
    Cos,
}

impl TrigSquareOp {
    fn complement(self) -> Self {
        match self {
            Self::Sin => Self::Cos,
            Self::Cos => Self::Sin,
        }
    }

    fn is_complement(self, other: Self) -> bool {
        self.complement() == other
    }

    fn into_unary_op(self) -> UnaryOp {
        match self {
            Self::Sin => UnaryOp::Sin,
            Self::Cos => UnaryOp::Cos,
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct TrigSquareTerm {
    op: TrigSquareOp,
    input: ExprId,
    coefficient: f64,
}

#[derive(Copy, Clone, Debug)]
struct AffineCosTerm {
    input: ExprId,
    scale: f64,
    constant: f64,
    cos_coefficient: f64,
}
