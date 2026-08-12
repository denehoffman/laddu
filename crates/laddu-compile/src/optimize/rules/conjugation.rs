use laddu_expr::{BinaryOp, ExprId, ExprMetadata, ExprNode, ExprSourceKind, UnaryOp};

use crate::CompileResult;

use super::super::{ConjugationRule, NormSqrExpansionRule, Rewrite, RewriteContext, RewriteRule};

use super::add_mul::ReplacementFragment;

impl RewriteRule for NormSqrExpansionRule {
    fn name(&self) -> &'static str {
        "norm-sqr-expansion"
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
        let mut builder = ReplacementFragment::new(context);
        let conj = builder.push(
            ExprNode::Unary {
                op: UnaryOp::Conj,
                input: *input,
            },
            ExprMetadata::new(ExprSourceKind::Unary),
        );
        builder.push(
            ExprNode::NaryMul {
                factors: vec![*input, conj],
            },
            metadata.clone(),
        );
        Ok(builder.into_rewrite())
    }
}

impl RewriteRule for ConjugationRule {
    fn name(&self) -> &'static str {
        "conjugation"
    }

    fn rewrite(
        &self,
        node: &ExprNode,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite> {
        let ExprNode::Unary {
            op: UnaryOp::Conj,
            input,
        } = node
        else {
            return Ok(Rewrite::Keep);
        };
        Ok(match context.node(*input) {
            Some(ExprNode::NaryMul { factors }) => {
                self.push_conjugated_product(factors, metadata, context)
            }
            Some(ExprNode::Binary {
                op: BinaryOp::Mul,
                lhs,
                rhs,
            }) => self.push_conjugated_product(&[*lhs, *rhs], metadata, context),
            Some(ExprNode::NaryAdd { terms }) => self.push_conjugated_sum(terms, metadata, context),
            Some(ExprNode::Binary {
                op: BinaryOp::Add,
                lhs,
                rhs,
            }) => self.push_conjugated_sum(&[*lhs, *rhs], metadata, context),
            Some(ExprNode::Unary {
                op: UnaryOp::Exp,
                input,
            }) => self.push_conjugated_exp(*input, metadata, context),
            Some(ExprNode::Unary {
                op: UnaryOp::Neg,
                input,
            }) => self.push_conjugated_neg(*input, metadata, context),
            _ => Rewrite::Keep,
        })
    }
}

impl ConjugationRule {
    fn push_conjugated_product(
        &self,
        factors: &[ExprId],
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> Rewrite {
        let mut builder = ReplacementFragment::new(context);
        let factors = factors
            .iter()
            .map(|factor| builder.conjugated_term(*factor))
            .collect();
        builder.push(ExprNode::NaryMul { factors }, metadata.clone());
        builder.into_rewrite()
    }

    fn push_conjugated_sum(
        &self,
        terms: &[ExprId],
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> Rewrite {
        let mut builder = ReplacementFragment::new(context);
        let terms = terms
            .iter()
            .map(|term| builder.conjugated_term(*term))
            .collect();
        builder.push(ExprNode::NaryAdd { terms }, metadata.clone());
        builder.into_rewrite()
    }

    fn push_conjugated_exp(
        &self,
        input: ExprId,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> Rewrite {
        let mut builder = ReplacementFragment::new(context);
        let input = builder.conjugated_term(input);
        builder.push(
            ExprNode::Unary {
                op: UnaryOp::Exp,
                input,
            },
            metadata.clone(),
        );
        builder.into_rewrite()
    }

    fn push_conjugated_neg(
        &self,
        input: ExprId,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> Rewrite {
        let mut builder = ReplacementFragment::new(context);
        let input = builder.conjugated_term(input);
        builder.push(
            ExprNode::Unary {
                op: UnaryOp::Neg,
                input,
            },
            metadata.clone(),
        );
        builder.into_rewrite()
    }
}
