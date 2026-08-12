use laddu_expr::{BinaryOp, ExprId, ExprMetadata, ExprNode, ExprSourceKind, UnaryOp};
use num::complex::Complex64;

use crate::{CompileResult, facts::NumberClass};

use super::super::rewrite::alias_or_preserve;
use super::super::{
    AlgebraicIdentityRule, ConstantFoldScalarRule, NormSqrReductionRule, Rewrite, RewriteContext,
    RewriteRule,
};
use super::{add_mul::ReplacementFragment, is_scalar_value};

impl RewriteRule for ConstantFoldScalarRule {
    fn name(&self) -> &'static str {
        "constant-fold-scalar"
    }

    fn rewrite(
        &self,
        node: &ExprNode,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite> {
        match node {
            ExprNode::Unary { op, input } => {
                let Some(input) = context.node(*input).and_then(ExprNode::const_value) else {
                    return Ok(Rewrite::Keep);
                };
                Ok(Rewrite::Replace {
                    node: ExprNode::from_folded_const(op.evaluate(input)),
                    metadata: metadata.clone(),
                })
            }
            ExprNode::Binary { op, lhs, rhs } => {
                let (Some(lhs), Some(rhs)) = (
                    context.node(*lhs).and_then(ExprNode::const_value),
                    context.node(*rhs).and_then(ExprNode::const_value),
                ) else {
                    return Ok(Rewrite::Keep);
                };
                Ok(Rewrite::Replace {
                    node: ExprNode::from_folded_const(op.evaluate(lhs, rhs)),
                    metadata: metadata.clone(),
                })
            }
            ExprNode::NaryAdd { terms } => self.fold_nary_add_constants(terms, metadata, context),
            ExprNode::NaryMul { factors } => {
                let Some(product) = factors
                    .iter()
                    .map(|id| context.node(*id).and_then(ExprNode::const_value))
                    .try_fold(Complex64::ONE, |product, value| {
                        value.map(|value| product * value)
                    })
                else {
                    return Ok(Rewrite::Keep);
                };
                Ok(Rewrite::Replace {
                    node: ExprNode::from_folded_const(product),
                    metadata: metadata.clone(),
                })
            }
            _ => Ok(Rewrite::Keep),
        }
    }
}

impl ConstantFoldScalarRule {
    fn fold_nary_add_constants(
        &self,
        terms: &[ExprId],
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite> {
        let mut constant_sum = Complex64::ZERO;
        let mut nonconstant_terms = Vec::new();
        let mut constant_count = 0;

        for term in terms {
            if let Some(value) = context.node(*term).and_then(ExprNode::const_value) {
                constant_sum += value;
                constant_count += 1;
            } else {
                nonconstant_terms.push(*term);
            }
        }

        if constant_count == 0 {
            return Ok(Rewrite::Keep);
        }

        if nonconstant_terms.is_empty() {
            return Ok(Rewrite::Replace {
                node: ExprNode::from_folded_const(constant_sum),
                metadata: metadata.clone(),
            });
        }

        if constant_sum != Complex64::ZERO {
            if constant_count == 1 {
                return Ok(Rewrite::Keep);
            }
            let mut builder = ReplacementFragment::new(context);
            let constant = builder.push(
                ExprNode::from_folded_const(constant_sum),
                ExprMetadata::new(ExprSourceKind::Const),
            );
            nonconstant_terms.push(constant);
            builder.push(
                ExprNode::NaryAdd {
                    terms: nonconstant_terms,
                },
                metadata.clone(),
            );
            return Ok(builder.into_rewrite());
        }

        Ok(match nonconstant_terms.as_slice() {
            [term] => alias_or_preserve(*term, metadata, context),
            _ => Rewrite::Replace {
                node: ExprNode::NaryAdd {
                    terms: nonconstant_terms,
                },
                metadata: metadata.clone(),
            },
        })
    }
}

impl RewriteRule for AlgebraicIdentityRule {
    fn name(&self) -> &'static str {
        "algebraic-identity"
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
            } if context.node(*lhs).is_some_and(ExprNode::is_zero) => {
                Ok(alias_or_preserve(*rhs, metadata, context))
            }
            ExprNode::Binary {
                op: BinaryOp::Add,
                lhs,
                rhs,
            } if context.node(*rhs).is_some_and(ExprNode::is_zero) => {
                Ok(alias_or_preserve(*lhs, metadata, context))
            }
            ExprNode::Binary {
                op: BinaryOp::Sub,
                lhs,
                rhs,
            } if lhs == rhs => Ok(Rewrite::Replace {
                node: ExprNode::RealConst(0.0),
                metadata: metadata.clone(),
            }),
            ExprNode::Binary {
                op: BinaryOp::Sub,
                lhs,
                rhs,
            } if context.node(*lhs).is_some_and(ExprNode::is_zero) => Ok(Rewrite::Replace {
                node: ExprNode::Unary {
                    op: UnaryOp::Neg,
                    input: *rhs,
                },
                metadata: metadata.clone(),
            }),
            ExprNode::Binary {
                op: BinaryOp::Sub,
                lhs,
                rhs,
            } if context.node(*rhs).is_some_and(ExprNode::is_zero) => {
                Ok(alias_or_preserve(*lhs, metadata, context))
            }
            ExprNode::Binary {
                op: BinaryOp::Mul,
                lhs,
                rhs,
            } if context.node(*lhs).is_some_and(ExprNode::is_one) => {
                Ok(alias_or_preserve(*rhs, metadata, context))
            }
            ExprNode::Binary {
                op: BinaryOp::Mul,
                lhs,
                rhs,
            } if context.node(*rhs).is_some_and(ExprNode::is_one) => {
                Ok(alias_or_preserve(*lhs, metadata, context))
            }
            ExprNode::Binary {
                op: BinaryOp::Mul,
                lhs,
                ..
            } if context.node(*lhs).is_some_and(ExprNode::is_zero) => {
                Ok(alias_or_preserve(*lhs, metadata, context))
            }
            ExprNode::Binary {
                op: BinaryOp::Mul,
                rhs,
                ..
            } if context.node(*rhs).is_some_and(ExprNode::is_zero) => {
                Ok(alias_or_preserve(*rhs, metadata, context))
            }
            ExprNode::Binary {
                op: BinaryOp::Div,
                lhs,
                rhs,
            } if lhs == rhs => Ok(Rewrite::Replace {
                node: ExprNode::RealConst(1.0),
                metadata: metadata.clone(),
            }),
            ExprNode::Binary {
                op: BinaryOp::Div,
                lhs,
                rhs,
            } if context.node(*rhs).is_some_and(ExprNode::is_one) => {
                Ok(alias_or_preserve(*lhs, metadata, context))
            }
            ExprNode::NaryAdd { terms } => self.simplify_nary_add(terms, metadata, context),
            ExprNode::NaryMul { factors } => self.simplify_nary_mul(factors, metadata, context),
            ExprNode::Unary {
                op: UnaryOp::Neg,
                input,
            } => {
                let Some(ExprNode::Unary {
                    op: UnaryOp::Neg,
                    input,
                }) = context.node(*input)
                else {
                    return Ok(Rewrite::Keep);
                };
                Ok(alias_or_preserve(*input, metadata, context))
            }
            ExprNode::Unary {
                op: UnaryOp::PowI(0),
                ..
            } => Ok(Rewrite::Replace {
                node: ExprNode::RealConst(1.0),
                metadata: metadata.clone(),
            }),
            ExprNode::Unary {
                op: UnaryOp::PowI(1),
                input,
            } => Ok(alias_or_preserve(*input, metadata, context)),
            ExprNode::Unary {
                op: UnaryOp::PowI(outer_power),
                input,
            } if *outer_power == 2 => {
                let Some(ExprNode::Unary {
                    op: UnaryOp::Sqrt,
                    input,
                }) = context.node(*input)
                else {
                    return Ok(Rewrite::Keep);
                };
                Ok(alias_or_preserve(*input, metadata, context))
            }
            ExprNode::Unary {
                op: UnaryOp::PowI(outer_power),
                input,
            } => {
                let Some(ExprNode::Unary {
                    op: UnaryOp::PowI(inner_power),
                    input,
                }) = context.node(*input)
                else {
                    return Ok(Rewrite::Keep);
                };
                let Some(power) = inner_power.checked_mul(*outer_power) else {
                    return Ok(Rewrite::Keep);
                };
                Ok(Rewrite::Replace {
                    node: ExprNode::Unary {
                        op: UnaryOp::PowI(power),
                        input: *input,
                    },
                    metadata: metadata.clone(),
                })
            }
            ExprNode::Unary {
                op: UnaryOp::Conj,
                input,
            } => match context.node(*input) {
                Some(ExprNode::Unary {
                    op: UnaryOp::Conj,
                    input,
                }) => Ok(alias_or_preserve(*input, metadata, context)),
                Some(ExprNode::Complex { re, im }) => {
                    let neg_id = context.next_id();
                    Ok(Rewrite::ReplaceMany {
                        nodes: vec![
                            (
                                ExprNode::Unary {
                                    op: UnaryOp::Neg,
                                    input: *im,
                                },
                                ExprMetadata::new(ExprSourceKind::Unary),
                            ),
                            (
                                ExprNode::Complex {
                                    re: *re,
                                    im: neg_id,
                                },
                                metadata.clone(),
                            ),
                        ],
                    })
                }
                _ => Ok(Rewrite::Keep),
            },
            ExprNode::Unary {
                op: UnaryOp::Real,
                input,
            } => match context.node(*input) {
                Some(ExprNode::Unary {
                    op: UnaryOp::Real, ..
                })
                | Some(ExprNode::Unary {
                    op: UnaryOp::Imag, ..
                }) => Ok(alias_or_preserve(*input, metadata, context)),
                Some(ExprNode::Complex { re, .. }) => Ok(alias_or_preserve(*re, metadata, context)),
                _ => Ok(Rewrite::Keep),
            },
            ExprNode::Unary {
                op: UnaryOp::Imag,
                input,
            } => match context.node(*input) {
                Some(ExprNode::Unary {
                    op: UnaryOp::Real, ..
                }) => Ok(Rewrite::Replace {
                    node: ExprNode::RealConst(0.0),
                    metadata: metadata.clone(),
                }),
                Some(ExprNode::Complex { im, .. }) => Ok(alias_or_preserve(*im, metadata, context)),
                _ => Ok(Rewrite::Keep),
            },
            ExprNode::Component { input, index } => {
                let Some(ExprNode::Vector { elements }) = context.node(*input) else {
                    return Ok(Rewrite::Keep);
                };
                if !elements.iter().all(|id| is_scalar_value(context, *id)) {
                    return Ok(Rewrite::Keep);
                }
                let Some(element) = elements.get(*index).copied() else {
                    return Ok(Rewrite::Keep);
                };
                Ok(alias_or_preserve(element, metadata, context))
            }
            ExprNode::MatrixElement { input, row, col } => {
                let Some(ExprNode::Matrix { cols, elements, .. }) = context.node(*input) else {
                    return Ok(Rewrite::Keep);
                };
                if !elements.iter().all(|id| is_scalar_value(context, *id)) {
                    return Ok(Rewrite::Keep);
                }
                let Some(index) = row
                    .checked_mul(*cols)
                    .and_then(|base| base.checked_add(*col))
                else {
                    return Ok(Rewrite::Keep);
                };
                let Some(element) = elements.get(index).copied() else {
                    return Ok(Rewrite::Keep);
                };
                Ok(alias_or_preserve(element, metadata, context))
            }
            _ => Ok(Rewrite::Keep),
        }
    }
}

impl RewriteRule for NormSqrReductionRule {
    fn name(&self) -> &'static str {
        "norm-sqr-reduction"
    }

    fn rewrite(
        &self,
        node: &ExprNode,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite> {
        let ExprNode::Unary {
            op: UnaryOp::NormSqr,
            input,
        } = node
        else {
            return Ok(Rewrite::Keep);
        };

        if context
            .facts(*input)
            .is_some_and(|facts| facts.number_class == NumberClass::Real)
        {
            return Ok(Rewrite::Replace {
                node: ExprNode::Unary {
                    op: UnaryOp::PowI(2),
                    input: *input,
                },
                metadata: metadata.clone(),
            });
        }

        match context.node(*input) {
            Some(ExprNode::Unary {
                op: UnaryOp::Conj | UnaryOp::Neg,
                input,
            }) => Ok(Rewrite::Replace {
                node: ExprNode::Unary {
                    op: UnaryOp::NormSqr,
                    input: *input,
                },
                metadata: metadata.clone(),
            }),
            Some(ExprNode::Complex { re, im })
                if [*re, *im].into_iter().all(|id| {
                    context
                        .facts(id)
                        .is_some_and(|facts| facts.number_class == NumberClass::Real)
                }) =>
            {
                let mut builder = ReplacementFragment::new(context);
                let re_square = builder.push(
                    ExprNode::Unary {
                        op: UnaryOp::PowI(2),
                        input: *re,
                    },
                    ExprMetadata::new(ExprSourceKind::Unary),
                );
                let im_square = builder.push(
                    ExprNode::Unary {
                        op: UnaryOp::PowI(2),
                        input: *im,
                    },
                    ExprMetadata::new(ExprSourceKind::Unary),
                );
                builder.push(
                    ExprNode::NaryAdd {
                        terms: vec![re_square, im_square],
                    },
                    metadata.clone(),
                );
                Ok(builder.into_rewrite())
            }
            _ => Ok(Rewrite::Keep),
        }
    }
}

impl AlgebraicIdentityRule {
    fn simplify_nary_add(
        &self,
        terms: &[ExprId],
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite> {
        let nonzero_terms: Vec<_> = terms
            .iter()
            .copied()
            .filter(|id| !context.node(*id).is_some_and(ExprNode::is_zero))
            .collect();

        if nonzero_terms.len() == terms.len() {
            return Ok(Rewrite::Keep);
        }

        Ok(match nonzero_terms.as_slice() {
            [] => Rewrite::Replace {
                node: ExprNode::RealConst(0.0),
                metadata: metadata.clone(),
            },
            [term] => alias_or_preserve(*term, metadata, context),
            _ => Rewrite::Replace {
                node: ExprNode::NaryAdd {
                    terms: nonzero_terms,
                },
                metadata: metadata.clone(),
            },
        })
    }

    fn simplify_nary_mul(
        &self,
        factors: &[ExprId],
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite> {
        if let Some(zero) = factors
            .iter()
            .copied()
            .find(|id| context.node(*id).is_some_and(ExprNode::is_zero))
        {
            return Ok(alias_or_preserve(zero, metadata, context));
        }

        let nonone_factors: Vec<_> = factors
            .iter()
            .copied()
            .filter(|id| !context.node(*id).is_some_and(ExprNode::is_one))
            .collect();

        if nonone_factors.len() == factors.len() {
            return Ok(Rewrite::Keep);
        }

        Ok(match nonone_factors.as_slice() {
            [] => Rewrite::Replace {
                node: ExprNode::RealConst(1.0),
                metadata: metadata.clone(),
            },
            [factor] => alias_or_preserve(*factor, metadata, context),
            _ => Rewrite::Replace {
                node: ExprNode::NaryMul {
                    factors: nonone_factors,
                },
                metadata: metadata.clone(),
            },
        })
    }
}
