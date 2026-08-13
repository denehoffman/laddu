use laddu_expr::UnaryOp;
use laddu_kernel::ir::{KernelInstruction, KernelValueId, KernelValueKind};

use super::super::ReverseState;
use crate::AutodiffResult;

impl ReverseState<'_> {
    pub(in crate::gradient) fn matmul_pullback(
        &mut self,
        lhs: KernelValueId,
        rhs: KernelValueId,
        adjoint: &[KernelValueId],
    ) -> AutodiffResult<()> {
        let (KernelValueKind::Matrix { rows, cols: inner }, KernelValueKind::Matrix { cols, .. }) =
            (self.kind(lhs), self.kind(rhs))
        else {
            unreachable!()
        };
        for row in 0..rows {
            for col in 0..cols {
                let output_adjoint = adjoint[row * cols + col];
                for k in 0..inner {
                    let rhs_value = self.matrix_element(rhs, k, col)?;
                    let lhs_contribution = self.mul_conj(output_adjoint, rhs_value)?;
                    self.accumulate_element(lhs, row * inner + k, lhs_contribution)?;
                    let lhs_value = self.matrix_element(lhs, row, k)?;
                    let rhs_contribution = self.mul_conj(output_adjoint, lhs_value)?;
                    self.accumulate_element(rhs, k * cols + col, rhs_contribution)?;
                }
            }
        }
        Ok(())
    }

    pub(in crate::gradient) fn matvec_pullback(
        &mut self,
        matrix: KernelValueId,
        vector: KernelValueId,
        adjoint: &[KernelValueId],
    ) -> AutodiffResult<()> {
        let KernelValueKind::Matrix { rows, cols } = self.kind(matrix) else {
            unreachable!()
        };
        for (row, &row_adjoint) in adjoint.iter().enumerate().take(rows) {
            for col in 0..cols {
                let vector_value = self.component(vector, col)?;
                let matrix_contribution = self.mul_conj(row_adjoint, vector_value)?;
                self.accumulate_element(matrix, row * cols + col, matrix_contribution)?;
                let matrix_value = self.matrix_element(matrix, row, col)?;
                let vector_contribution = self.mul_conj(row_adjoint, matrix_value)?;
                self.accumulate_element(vector, col, vector_contribution)?;
            }
        }
        Ok(())
    }

    pub(in crate::gradient) fn dot_pullback(
        &mut self,
        lhs: KernelValueId,
        rhs: KernelValueId,
        adjoint: KernelValueId,
    ) -> AutodiffResult<()> {
        let KernelValueKind::Vector { len } = self.kind(lhs) else {
            unreachable!()
        };
        for index in 0..len {
            let rhs_value = self.component(rhs, index)?;
            let lhs_contribution = self.mul_conj(adjoint, rhs_value)?;
            self.accumulate_element(lhs, index, lhs_contribution)?;
            let lhs_value = self.component(lhs, index)?;
            let rhs_contribution = self.mul_conj(adjoint, lhs_value)?;
            self.accumulate_element(rhs, index, rhs_contribution)?;
        }
        Ok(())
    }

    pub(in crate::gradient) fn solve_pullback(
        &mut self,
        matrix: KernelValueId,
        rhs: KernelValueId,
        solution: KernelValueId,
        adjoint: &[KernelValueId],
    ) -> AutodiffResult<()> {
        let KernelValueKind::Matrix { rows, cols } = self.kind(matrix) else {
            unreachable!()
        };
        let mut transpose = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            for col in 0..cols {
                let value = self.matrix_element(matrix, col, row)?;
                transpose.push(self.unary(UnaryOp::Conj, value)?);
            }
        }
        let transpose = self.push(KernelInstruction::Matrix {
            rows,
            cols,
            elements: transpose,
        })?;
        let adjoint_vector = self.push(KernelInstruction::Vector(adjoint.to_vec()))?;
        let lambda = self.push(KernelInstruction::Solve {
            matrix: transpose,
            rhs: adjoint_vector,
        })?;
        for row in 0..rows {
            let lambda_value = self.component(lambda, row)?;
            self.accumulate_element(rhs, row, lambda_value)?;
            for col in 0..cols {
                let solution_value = self.component(solution, col)?;
                let product = self.mul_conj(lambda_value, solution_value)?;
                let contribution = self.unary(UnaryOp::Neg, product)?;
                self.accumulate_element(matrix, row * cols + col, contribution)?;
            }
        }
        Ok(())
    }

    pub(in crate::gradient) fn solve_row_pullback(
        &mut self,
        row_slot: usize,
        rhs: &[KernelValueId],
        adjoint: KernelValueId,
    ) -> AutodiffResult<()> {
        let len = rhs.len();
        for (index, rhs) in rhs.iter().enumerate() {
            let contribution = self.push(KernelInstruction::SolveRowAdjointElement {
                row_slot,
                index,
                len,
                adjoint,
            })?;
            self.accumulate(*rhs, &[contribution])?;
        }
        Ok(())
    }
}
