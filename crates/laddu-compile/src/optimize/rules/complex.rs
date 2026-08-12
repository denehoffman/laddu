use laddu_expr::{ExprMetadata, ExprNode, UnaryOp};

use crate::{CompileResult, facts::NumberClass};

use super::super::rewrite::alias_or_preserve;
use super::super::{ComplexFactRule, Rewrite, RewriteContext, RewriteRule};

impl RewriteRule for ComplexFactRule {
    fn name(&self) -> &'static str {
        "complex-fact"
    }

    fn rewrite(
        &self,
        node: &ExprNode,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite> {
        match node {
            ExprNode::Unary {
                op: UnaryOp::Real,
                input,
            } if context
                .facts(*input)
                .is_some_and(|facts| facts.number_class == NumberClass::Real) =>
            {
                Ok(alias_or_preserve(*input, metadata, context))
            }
            ExprNode::Unary {
                op: UnaryOp::Imag,
                input,
            } if context
                .facts(*input)
                .is_some_and(|facts| facts.number_class == NumberClass::Real) =>
            {
                Ok(Rewrite::Replace {
                    node: ExprNode::RealConst(0.0),
                    metadata: metadata.clone(),
                })
            }
            ExprNode::Unary {
                op: UnaryOp::Conj,
                input,
            } if context
                .facts(*input)
                .is_some_and(|facts| facts.number_class == NumberClass::Real) =>
            {
                Ok(alias_or_preserve(*input, metadata, context))
            }
            _ => Ok(Rewrite::Keep),
        }
    }
}
