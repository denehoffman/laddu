use laddu_expr::{BinaryOp, UnaryOp};
use laddu_kernel::ir::{KernelInstruction, KernelValueId};
use num::complex::Complex64;

use super::ReverseState;
use crate::AutodiffResult;

impl ReverseState<'_> {
    fn emit_instruction(
        &mut self,
        instruction: KernelInstruction,
    ) -> AutodiffResult<KernelValueId> {
        self.builder.push(instruction).map_err(Into::into)
    }

    pub(super) fn real(&mut self, value: f64) -> AutodiffResult<KernelValueId> {
        self.emit_instruction(KernelInstruction::RealConstant(value))
    }

    pub(super) fn complex(&mut self, re: f64, im: f64) -> AutodiffResult<KernelValueId> {
        self.emit_instruction(KernelInstruction::ComplexConstant(Complex64::new(re, im)))
    }

    pub(super) fn complex_value(
        &mut self,
        re: KernelValueId,
        im: KernelValueId,
    ) -> AutodiffResult<KernelValueId> {
        self.emit_instruction(KernelInstruction::Complex { re, im })
    }

    pub(super) fn unary(
        &mut self,
        op: UnaryOp,
        input: KernelValueId,
    ) -> AutodiffResult<KernelValueId> {
        self.emit_instruction(KernelInstruction::Unary { op, input })
    }

    pub(super) fn binary(
        &mut self,
        op: BinaryOp,
        lhs: KernelValueId,
        rhs: KernelValueId,
    ) -> AutodiffResult<KernelValueId> {
        self.emit_instruction(KernelInstruction::Binary { op, lhs, rhs })
    }

    pub(super) fn add(
        &mut self,
        lhs: KernelValueId,
        rhs: KernelValueId,
    ) -> AutodiffResult<KernelValueId> {
        self.emit_instruction(KernelInstruction::Add(vec![lhs, rhs]))
    }

    pub(super) fn mul(
        &mut self,
        lhs: KernelValueId,
        rhs: KernelValueId,
    ) -> AutodiffResult<KernelValueId> {
        self.emit_instruction(KernelInstruction::Mul(vec![lhs, rhs]))
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
        self.emit_instruction(KernelInstruction::Component { input, index })
    }

    pub(super) fn matrix_element(
        &mut self,
        input: KernelValueId,
        row: usize,
        col: usize,
    ) -> AutodiffResult<KernelValueId> {
        self.emit_instruction(KernelInstruction::MatrixElement { input, row, col })
    }

    pub(super) fn vector(&mut self, elements: Vec<KernelValueId>) -> AutodiffResult<KernelValueId> {
        self.emit_instruction(KernelInstruction::Vector(elements))
    }

    pub(super) fn matrix(
        &mut self,
        rows: usize,
        cols: usize,
        elements: Vec<KernelValueId>,
    ) -> AutodiffResult<KernelValueId> {
        self.emit_instruction(KernelInstruction::Matrix {
            rows,
            cols,
            elements,
        })
    }

    pub(super) fn solve(
        &mut self,
        matrix: KernelValueId,
        rhs: KernelValueId,
    ) -> AutodiffResult<KernelValueId> {
        self.emit_instruction(KernelInstruction::Solve { matrix, rhs })
    }

    pub(super) fn solve_row_adjoint_element(
        &mut self,
        row_slot: usize,
        index: usize,
        len: usize,
        adjoint: KernelValueId,
    ) -> AutodiffResult<KernelValueId> {
        self.emit_instruction(KernelInstruction::SolveRowAdjointElement {
            row_slot,
            index,
            len,
            adjoint,
        })
    }

    pub(super) fn sum_or_zero(
        &mut self,
        terms: Vec<KernelValueId>,
    ) -> AutodiffResult<KernelValueId> {
        match terms.len() {
            0 => self.real(0.0),
            1 => Ok(terms[0]),
            _ => self.emit_instruction(KernelInstruction::Add(terms)),
        }
    }
}
