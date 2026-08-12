use laddu_expr::{BinaryOp, ExprId, ExprMetadata, ExprNode, ExprSourceKind, UnaryOp};
use num::complex::Complex64;

use crate::{CompileResult, facts::NumberClass};

use super::super::{ExponentialRule, Rewrite, RewriteContext, RewriteRule};

use super::{
    add_mul::{PowerFactor, ProductTerm, ReplacementFragment},
    is_scalar_value,
    trig::TrigSquareOp,
};

impl RewriteRule for ExponentialRule {
    fn name(&self) -> &'static str {
        "exponential"
    }

    fn rewrite(
        &self,
        node: &ExprNode,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite> {
        match node {
            ExprNode::Unary {
                op: UnaryOp::Exp,
                input,
            } if context.node(*input).is_some_and(ExprNode::is_zero) => Ok(Rewrite::Replace {
                node: ExprNode::RealConst(1.0),
                metadata: metadata.clone(),
            }),
            ExprNode::Unary {
                op: UnaryOp::Exp, ..
            } => Ok(Rewrite::Keep),
            ExprNode::Binary {
                op: BinaryOp::Mul,
                lhs,
                rhs,
            } => {
                let (Some(lhs_input), Some(rhs_input)) =
                    (self.exp_input(context, *lhs), self.exp_input(context, *rhs))
                else {
                    return Ok(Rewrite::Keep);
                };
                self.merge_product(vec![lhs_input, rhs_input], Vec::new(), metadata, context)
            }
            ExprNode::NaryMul { factors } => {
                let mut exp_inputs = Vec::new();
                let mut other_factors = Vec::new();
                for factor in factors {
                    if let Some(input) = self.exp_input(context, *factor) {
                        exp_inputs.push(input);
                    } else {
                        other_factors.push(*factor);
                    }
                }
                if exp_inputs.len() < 2 {
                    return Ok(Rewrite::Keep);
                }
                self.merge_product(exp_inputs, other_factors, metadata, context)
            }
            ExprNode::NaryAdd { terms } if terms.len() >= 2 => {
                Ok(self.rewrite_euler_sum(terms, metadata, context))
            }
            _ => Ok(Rewrite::Keep),
        }
    }
}

impl ExponentialRule {
    fn merge_product(
        &self,
        exp_inputs: Vec<ExprId>,
        mut other_factors: Vec<ExprId>,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite> {
        if let Some(rewrite) =
            self.merge_imaginary_phase_product(&exp_inputs, &other_factors, metadata, context)
        {
            return Ok(rewrite);
        }

        let mut builder = ReplacementFragment::new(context);
        let add_id = builder.push(
            ExprNode::NaryAdd { terms: exp_inputs },
            ExprMetadata::new(ExprSourceKind::Binary),
        );
        let exp_id = builder.push(
            ExprNode::Unary {
                op: UnaryOp::Exp,
                input: add_id,
            },
            if other_factors.is_empty() {
                metadata.clone()
            } else {
                ExprMetadata::new(ExprSourceKind::Unary)
            },
        );

        if !other_factors.is_empty() {
            other_factors.push(exp_id);
            builder.push(
                ExprNode::NaryMul {
                    factors: other_factors,
                },
                metadata.clone(),
            );
        }
        Ok(builder.into_rewrite())
    }

    fn rewrite_euler_sum(
        &self,
        terms: &[ExprId],
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> Rewrite {
        for (lhs_index, lhs) in terms.iter().enumerate() {
            let Some(lhs) = self.trig_product(*lhs, context) else {
                continue;
            };
            for (rhs_index, rhs) in terms.iter().enumerate().skip(lhs_index + 1) {
                let Some(rhs) = self.trig_product(*rhs, context) else {
                    continue;
                };
                let Some(euler) = lhs.euler_with(&rhs) else {
                    continue;
                };
                return self
                    .replace_euler_pair(terms, lhs_index, rhs_index, euler, metadata, context);
            }
        }
        Rewrite::Keep
    }

    fn replace_euler_pair(
        &self,
        terms: &[ExprId],
        lhs_index: usize,
        rhs_index: usize,
        euler: EulerProduct,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> Rewrite {
        let mut builder = ReplacementFragment::new(context);
        let phase = self.emit_imaginary_phase(euler.phase_sign, euler.input, &mut builder);
        let exp = builder.push(
            ExprNode::Unary {
                op: UnaryOp::Exp,
                input: phase,
            },
            ExprMetadata::new(ExprSourceKind::Unary),
        );
        let euler_term = euler.emit_with_exp(exp, &mut builder);
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
            new_terms.push(euler_term);
            builder.push(ExprNode::NaryAdd { terms: new_terms }, metadata.clone());
            builder.into_rewrite()
        }
    }

    fn merge_imaginary_phase_product(
        &self,
        exp_inputs: &[ExprId],
        other_factors: &[ExprId],
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> Option<Rewrite> {
        let phases: Vec<_> = exp_inputs
            .iter()
            .map(|input| self.imaginary_phase(*input, context))
            .collect::<Option<_>>()?;
        let mut builder = ReplacementFragment::new(context);
        let phase_terms = phases
            .into_iter()
            .map(|phase| phase.emit(&mut builder))
            .collect();
        let phase_sum = builder.push(
            ExprNode::NaryAdd { terms: phase_terms },
            ExprMetadata::new(ExprSourceKind::Binary),
        );
        let exp_input = self.emit_imaginary_phase(1.0, phase_sum, &mut builder);
        let exp = builder.push(
            ExprNode::Unary {
                op: UnaryOp::Exp,
                input: exp_input,
            },
            if other_factors.is_empty() {
                metadata.clone()
            } else {
                ExprMetadata::new(ExprSourceKind::Unary)
            },
        );

        if other_factors.is_empty() {
            return Some(builder.into_rewrite());
        }

        let mut factors = other_factors.to_vec();
        factors.push(exp);
        builder.push(ExprNode::NaryMul { factors }, metadata.clone());
        Some(builder.into_rewrite())
    }

    fn emit_imaginary_phase(
        &self,
        sign: f64,
        input: ExprId,
        builder: &mut ReplacementFragment<'_>,
    ) -> ExprId {
        let i = builder.push(
            ExprNode::ComplexConst(Complex64::I),
            ExprMetadata::new(ExprSourceKind::Const),
        );
        let phase = if ProductTerm::approx_eq(sign, 1.0) {
            input
        } else {
            builder.negated_term(input)
        };
        builder.push(
            ExprNode::NaryMul {
                factors: vec![i, phase],
            },
            ExprMetadata::new(ExprSourceKind::Binary),
        )
    }

    fn trig_product(&self, id: ExprId, context: &RewriteContext<'_>) -> Option<TrigProduct> {
        let mut product = TrigProduct::new();
        product.collect(id, context)?;
        product.into_normalized()
    }

    fn imaginary_phase(&self, id: ExprId, context: &RewriteContext<'_>) -> Option<PhaseProduct> {
        let mut product = PhaseProduct::new();
        product.collect(id, context)?;
        product.into_normalized(context)
    }

    fn exp_input(&self, context: &RewriteContext<'_>, id: ExprId) -> Option<ExprId> {
        match context.node(id) {
            Some(ExprNode::Unary {
                op: UnaryOp::Exp,
                input,
            }) => Some(*input),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
struct TrigProduct {
    coefficient: Complex64,
    factors: Vec<PowerFactor>,
    trig: Option<(TrigSquareOp, ExprId)>,
}

impl TrigProduct {
    fn new() -> Self {
        Self {
            coefficient: Complex64::ONE,
            factors: Vec::new(),
            trig: None,
        }
    }

    fn collect(&mut self, id: ExprId, context: &RewriteContext<'_>) -> Option<()> {
        match context.node(id)? {
            ExprNode::NaryMul { factors } => {
                for factor in factors {
                    self.collect(*factor, context)?;
                }
            }
            ExprNode::Binary {
                op: BinaryOp::Mul,
                lhs,
                rhs,
            } => {
                self.collect(*lhs, context)?;
                self.collect(*rhs, context)?;
            }
            ExprNode::Unary {
                op: UnaryOp::Neg,
                input,
            } => {
                self.coefficient = -self.coefficient;
                self.collect(*input, context)?;
            }
            ExprNode::RealConst(value) => {
                self.coefficient *= *value;
            }
            ExprNode::ComplexConst(value) => {
                self.coefficient *= *value;
            }
            ExprNode::Unary { op, input } if matches!(op, UnaryOp::Sin | UnaryOp::Cos) => {
                if self.trig.is_some() {
                    return None;
                }
                let op = match op {
                    UnaryOp::Sin => TrigSquareOp::Sin,
                    UnaryOp::Cos => TrigSquareOp::Cos,
                    _ => unreachable!("matched trig ops"),
                };
                self.trig = Some((op, *input));
            }
            ExprNode::Unary {
                op: UnaryOp::PowI(power),
                input,
            } if is_scalar_value(context, *input) => {
                self.factors.push(PowerFactor {
                    base: *input,
                    exponent: *power,
                });
            }
            _ if is_scalar_value(context, id) => {
                self.factors.push(PowerFactor {
                    base: id,
                    exponent: 1,
                });
            }
            _ => return None,
        }
        Some(())
    }

    fn into_normalized(mut self) -> Option<Self> {
        let _ = self.trig?;
        self.factors
            .sort_by_key(|factor| (factor.base.index(), factor.exponent));
        Some(self)
    }

    fn euler_with(&self, other: &Self) -> Option<EulerProduct> {
        let (self_op, self_input) = self.trig?;
        let (other_op, other_input) = other.trig?;
        if self_input != other_input || self_op == other_op || self.factors != other.factors {
            return None;
        }

        let (cos, sin) = match (self_op, other_op) {
            (TrigSquareOp::Cos, TrigSquareOp::Sin) => (self, other),
            (TrigSquareOp::Sin, TrigSquareOp::Cos) => (other, self),
            _ => return None,
        };
        if !ProductTerm::approx_eq(cos.coefficient.im, 0.0)
            || !ProductTerm::approx_eq(sin.coefficient.re, 0.0)
            || !ProductTerm::approx_eq(cos.coefficient.re.abs(), sin.coefficient.im.abs())
        {
            return None;
        }
        let phase_sign = if (sin.coefficient.im / cos.coefficient.re).is_sign_positive() {
            1.0
        } else {
            -1.0
        };
        Some(EulerProduct {
            coefficient: cos.coefficient.re,
            factors: cos.factors.clone(),
            phase_sign,
            input: self_input,
        })
    }
}

#[derive(Clone, Debug)]
struct EulerProduct {
    coefficient: f64,
    factors: Vec<PowerFactor>,
    phase_sign: f64,
    input: ExprId,
}

impl EulerProduct {
    fn emit_with_exp(&self, exp: ExprId, builder: &mut ReplacementFragment<'_>) -> ExprId {
        let mut factors = self.factors.clone();
        factors.push(PowerFactor {
            base: exp,
            exponent: 1,
        });
        ProductTerm::push_parts(self.coefficient, &factors, builder)
    }
}

#[derive(Clone, Debug)]
struct PhaseProduct {
    coefficient: Complex64,
    factors: Vec<PowerFactor>,
}

impl PhaseProduct {
    fn new() -> Self {
        Self {
            coefficient: Complex64::ONE,
            factors: Vec::new(),
        }
    }

    fn collect(&mut self, id: ExprId, context: &RewriteContext<'_>) -> Option<()> {
        match context.node(id)? {
            ExprNode::NaryMul { factors } => {
                for factor in factors {
                    self.collect(*factor, context)?;
                }
            }
            ExprNode::Binary {
                op: BinaryOp::Mul,
                lhs,
                rhs,
            } => {
                self.collect(*lhs, context)?;
                self.collect(*rhs, context)?;
            }
            ExprNode::Unary {
                op: UnaryOp::Neg,
                input,
            } => {
                self.coefficient = -self.coefficient;
                self.collect(*input, context)?;
            }
            ExprNode::RealConst(value) => self.coefficient *= *value,
            ExprNode::ComplexConst(value) => self.coefficient *= *value,
            ExprNode::Unary {
                op: UnaryOp::PowI(power),
                input,
            } if is_scalar_value(context, *input) => {
                self.factors.push(PowerFactor {
                    base: *input,
                    exponent: *power,
                });
            }
            _ if context
                .facts(id)
                .is_some_and(|facts| facts.number_class == NumberClass::Real) =>
            {
                self.factors.push(PowerFactor {
                    base: id,
                    exponent: 1,
                });
            }
            _ => return None,
        }
        Some(())
    }

    fn into_normalized(mut self, _context: &RewriteContext<'_>) -> Option<Self> {
        if !ProductTerm::approx_eq(self.coefficient.re, 0.0)
            || ProductTerm::approx_eq(self.coefficient.im, 0.0)
        {
            return None;
        }
        self.factors
            .sort_by_key(|factor| (factor.base.index(), factor.exponent));
        Some(Self {
            coefficient: Complex64::from(self.coefficient.im),
            factors: self.factors,
        })
    }

    fn emit(&self, builder: &mut ReplacementFragment<'_>) -> ExprId {
        ProductTerm::push_parts(self.coefficient.re, &self.factors, builder)
    }
}
