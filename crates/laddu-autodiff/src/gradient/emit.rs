use laddu_expr::{BinaryOp, UnaryOp};
use laddu_kernel::ir::{KernelInstruction, KernelValueId, KernelValueKind};
use num::complex::Complex64;

use super::ReverseState;
use crate::AutodiffResult;

impl ReverseState<'_> {
    pub(super) fn kind(&self, id: KernelValueId) -> KernelValueKind {
        self.primal.values()[id.index()].kind
    }

    pub(super) fn push(&mut self, instruction: KernelInstruction) -> AutodiffResult<KernelValueId> {
        self.builder.push(instruction).map_err(Into::into)
    }

    pub(super) fn real(&mut self, value: f64) -> AutodiffResult<KernelValueId> {
        self.push(KernelInstruction::RealConstant(value))
    }

    pub(super) fn complex(&mut self, re: f64, im: f64) -> AutodiffResult<KernelValueId> {
        self.push(KernelInstruction::ComplexConstant(Complex64::new(re, im)))
    }

    pub(super) fn unary(
        &mut self,
        op: UnaryOp,
        input: KernelValueId,
    ) -> AutodiffResult<KernelValueId> {
        self.push(KernelInstruction::Unary { op, input })
    }

    pub(super) fn binary(
        &mut self,
        op: BinaryOp,
        lhs: KernelValueId,
        rhs: KernelValueId,
    ) -> AutodiffResult<KernelValueId> {
        self.push(KernelInstruction::Binary { op, lhs, rhs })
    }

    pub(super) fn add(
        &mut self,
        lhs: KernelValueId,
        rhs: KernelValueId,
    ) -> AutodiffResult<KernelValueId> {
        self.push(KernelInstruction::Add(vec![lhs, rhs]))
    }

    pub(super) fn mul(
        &mut self,
        lhs: KernelValueId,
        rhs: KernelValueId,
    ) -> AutodiffResult<KernelValueId> {
        self.push(KernelInstruction::Mul(vec![lhs, rhs]))
    }

    pub(super) fn mul_conj(
        &mut self,
        lhs: KernelValueId,
        rhs: KernelValueId,
    ) -> AutodiffResult<KernelValueId> {
        let rhs = self.unary(UnaryOp::Conj, rhs)?;
        self.mul(lhs, rhs)
    }

    pub(super) fn component(
        &mut self,
        input: KernelValueId,
        index: usize,
    ) -> AutodiffResult<KernelValueId> {
        self.push(KernelInstruction::Component { input, index })
    }

    pub(super) fn matrix_element(
        &mut self,
        input: KernelValueId,
        row: usize,
        col: usize,
    ) -> AutodiffResult<KernelValueId> {
        self.push(KernelInstruction::MatrixElement { input, row, col })
    }

    pub(super) fn sum_or_zero(
        &mut self,
        terms: Vec<KernelValueId>,
    ) -> AutodiffResult<KernelValueId> {
        match terms.len() {
            0 => self.real(0.0),
            1 => Ok(terms[0]),
            _ => self.push(KernelInstruction::Add(terms)),
        }
    }
}
