use laddu_expr::{BinaryOp, UnaryOp};
use laddu_kernel::ir::KernelValueId;

use super::super::ReverseState;
use crate::AutodiffResult;

impl ReverseState<'_> {
    pub(in crate::gradient) fn unary_pullback(
        &mut self,
        op: UnaryOp,
        input: KernelValueId,
        output: KernelValueId,
        adjoint: KernelValueId,
    ) -> AutodiffResult<KernelValueId> {
        Ok(match op {
            UnaryOp::Neg => self.unary(UnaryOp::Neg, adjoint)?,
            UnaryOp::Real => self.unary(UnaryOp::Real, adjoint)?,
            UnaryOp::Imag => {
                let zero = self.real(0.0)?;
                let imaginary = self.unary(UnaryOp::Real, adjoint)?;
                self.push(laddu_kernel::ir::KernelInstruction::Complex {
                    re: zero,
                    im: imaginary,
                })?
            }
            UnaryOp::Conj => self.unary(UnaryOp::Conj, adjoint)?,
            UnaryOp::NormSqr => {
                let two = self.real(2.0)?;
                let derivative = self.mul(two, input)?;
                self.mul(adjoint, derivative)?
            }
            UnaryOp::Sqrt => {
                let two = self.real(2.0)?;
                let denominator = self.mul(two, output)?;
                let one = self.real(1.0)?;
                let derivative = self.binary(BinaryOp::Div, one, denominator)?;
                self.mul_conj(adjoint, derivative)?
            }
            UnaryOp::Exp => self.mul_conj(adjoint, output)?,
            UnaryOp::Sin => {
                let derivative = self.unary(UnaryOp::Cos, input)?;
                self.mul_conj(adjoint, derivative)?
            }
            UnaryOp::Cos => {
                let sine = self.unary(UnaryOp::Sin, input)?;
                let derivative = self.unary(UnaryOp::Neg, sine)?;
                self.mul_conj(adjoint, derivative)?
            }
            UnaryOp::Log => {
                let one = self.real(1.0)?;
                let derivative = self.binary(BinaryOp::Div, one, input)?;
                self.mul_conj(adjoint, derivative)?
            }
            UnaryOp::PowI(0) => self.real(0.0)?,
            UnaryOp::PowI(power) if power == i32::MIN => {
                let scale = self.real(power as f64)?;
                let numerator = self.mul(scale, output)?;
                let derivative = self.binary(BinaryOp::Div, numerator, input)?;
                self.mul_conj(adjoint, derivative)?
            }
            UnaryOp::PowI(power) => {
                let previous = self.unary(UnaryOp::PowI(power - 1), input)?;
                let scale = self.real(power as f64)?;
                let derivative = self.mul(scale, previous)?;
                self.mul_conj(adjoint, derivative)?
            }
        })
    }

    pub(in crate::gradient) fn binary_pullback(
        &mut self,
        op: BinaryOp,
        lhs: KernelValueId,
        rhs: KernelValueId,
        adjoint: KernelValueId,
    ) -> AutodiffResult<(KernelValueId, KernelValueId)> {
        Ok(match op {
            BinaryOp::Add => (adjoint, adjoint),
            BinaryOp::Sub => (adjoint, self.unary(UnaryOp::Neg, adjoint)?),
            BinaryOp::Mul => (self.mul_conj(adjoint, rhs)?, self.mul_conj(adjoint, lhs)?),
            BinaryOp::Div => {
                let one = self.real(1.0)?;
                let lhs_derivative = self.binary(BinaryOp::Div, one, rhs)?;
                let rhs_squared = self.mul(rhs, rhs)?;
                let quotient = self.binary(BinaryOp::Div, lhs, rhs_squared)?;
                let rhs_derivative = self.unary(UnaryOp::Neg, quotient)?;
                (
                    self.mul_conj(adjoint, lhs_derivative)?,
                    self.mul_conj(adjoint, rhs_derivative)?,
                )
            }
            BinaryOp::Atan2 => {
                let lhs_squared = self.mul(lhs, lhs)?;
                let rhs_squared = self.mul(rhs, rhs)?;
                let denominator = self.add(lhs_squared, rhs_squared)?;
                let lhs_derivative = self.binary(BinaryOp::Div, rhs, denominator)?;
                let negative_lhs = self.unary(UnaryOp::Neg, lhs)?;
                let rhs_derivative = self.binary(BinaryOp::Div, negative_lhs, denominator)?;
                (
                    self.mul(adjoint, lhs_derivative)?,
                    self.mul(adjoint, rhs_derivative)?,
                )
            }
        })
    }

    pub(in crate::gradient) fn product_pullback(
        &mut self,
        factors: &[KernelValueId],
        adjoint: KernelValueId,
    ) -> AutodiffResult<()> {
        let one = self.real(1.0)?;
        let mut prefixes = Vec::with_capacity(factors.len() + 1);
        prefixes.push(one);
        for factor in factors {
            prefixes.push(self.mul(*prefixes.last().unwrap(), *factor)?);
        }
        let mut suffix = one;
        for (index, factor) in factors.iter().enumerate().rev() {
            let derivative = self.mul(prefixes[index], suffix)?;
            let contribution = self.mul_conj(adjoint, derivative)?;
            self.accumulate(*factor, &[contribution])?;
            suffix = self.mul(*factor, suffix)?;
        }
        Ok(())
    }
}
