mod add_mul;
mod complex;
mod conjugation;
mod exponential;
mod matrix;
mod scalar;
mod trig;

use laddu_expr::{ExprId, ValueKind};

use super::RewriteContext;

fn is_scalar_value(context: &RewriteContext<'_>, id: ExprId) -> bool {
    context
        .facts(id)
        .is_some_and(|facts| matches!(facts.value_kind, ValueKind::Real | ValueKind::Complex))
}

pub(super) use add_mul::NormalizeAddMulRule;
