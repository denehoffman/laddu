use laddu_expr::{BinaryOp, ExprId, ExprNode, UnaryOp};
use nalgebra::{DMatrix, DVector};
use num::complex::Complex64;

use super::super::{
    CpuBatchCache, CpuPlan, RuntimeError, RuntimeResult, Value, matrix_at, scalar_at, vector_at,
};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct ScalarAdjoint {
    dz: Complex64,
    dz_conj: Complex64,
}

impl ScalarAdjoint {
    fn seed() -> Self {
        Self {
            dz: Complex64::ONE,
            dz_conj: Complex64::ZERO,
        }
    }

    fn gradient(self) -> Complex64 {
        self.dz + self.dz_conj
    }
}

#[derive(Clone, Debug, PartialEq)]
enum ReverseAdjoint {
    Scalar(ScalarAdjoint),
    Vector(Vec<ScalarAdjoint>),
    Matrix {
        rows: usize,
        cols: usize,
        values: Vec<ScalarAdjoint>,
    },
}

impl ReverseAdjoint {
    fn kind(&self) -> &'static str {
        match self {
            Self::Scalar(_) => "scalar adjoint",
            Self::Vector(_) => "vector adjoint",
            Self::Matrix { .. } => "matrix adjoint",
        }
    }
}

pub(in crate::cpu) struct ReverseDerivativeWorkspace<'a> {
    plan: &'a CpuPlan,
    primals: &'a [Value],
    adjoints: Vec<Option<ReverseAdjoint>>,
    cached_factors: Option<(&'a CpuBatchCache, usize)>,
}

impl<'a> ReverseDerivativeWorkspace<'a> {
    pub(in crate::cpu) fn new(
        plan: &'a CpuPlan,
        primals: &'a [Value],
        cached_factors: Option<(&'a CpuBatchCache, usize)>,
    ) -> Self {
        Self {
            plan,
            primals,
            adjoints: vec![None; plan.graph.nodes().len()],
            cached_factors,
        }
    }

    pub(in crate::cpu) fn gradient(&mut self) -> RuntimeResult<Vec<Complex64>> {
        self.accumulate_scalar(self.plan.graph.root(), ScalarAdjoint::seed())?;
        if self.cached_factors.is_some() {
            for id in self.plan.cached_evaluation_nodes.iter().rev().copied() {
                self.propagate_node(id)?;
            }
        } else {
            for index in (0..self.plan.graph.nodes().len()).rev() {
                let id = ExprId::from_index(index);
                self.propagate_node(id)?;
            }
        }

        let mut gradient = vec![Complex64::ZERO; self.plan.autodiff.parameter_count()];
        for (index, parameter) in self.plan.parameter_slots.iter().enumerate() {
            let Some(parameter) = parameter else {
                continue;
            };
            let Ok(Some(free_id)) = self.plan.params.free_id(*parameter) else {
                continue;
            };
            if let Some(adjoint) = self.scalar_adjoint_at(index)? {
                gradient[free_id.index()] += adjoint.gradient();
            }
        }
        Ok(gradient)
    }

    fn propagate_node(&mut self, id: ExprId) -> RuntimeResult<()> {
        let index = id.index();
        if matches!(self.plan.graph.nodes()[index], ExprNode::ScalarParam(_)) {
            return Ok(());
        }
        let Some(adjoint) = self.adjoints[index].take() else {
            return Ok(());
        };
        if self.cached_factors.is_some() && self.plan.cache_slots[index].is_some() {
            return Ok(());
        }
        let node = self.plan.graph.nodes()[index].clone();
        match node {
            ExprNode::Unary { op, input } => {
                let adjoint = Self::expect_scalar_adjoint(index, adjoint)?;
                let input_value = self.primal_scalar(input)?;
                let output_value = self.primal_scalar(id)?;
                self.propagate_unary(op, input, input_value, output_value, adjoint)?;
            }
            ExprNode::Binary { op, lhs, rhs } => {
                let adjoint = Self::expect_scalar_adjoint(index, adjoint)?;
                let lhs_value = self.primal_scalar(lhs)?;
                let rhs_value = self.primal_scalar(rhs)?;
                self.propagate_binary(op, lhs, rhs, lhs_value, rhs_value, adjoint)?;
            }
            ExprNode::NaryAdd { terms } => {
                let adjoint = Self::expect_scalar_adjoint(index, adjoint)?;
                for term in terms {
                    self.accumulate_scalar(term, adjoint)?;
                }
            }
            ExprNode::NaryMul { factors } => {
                let adjoint = Self::expect_scalar_adjoint(index, adjoint)?;
                let values = factors
                    .iter()
                    .map(|factor| self.primal_scalar(*factor))
                    .collect::<RuntimeResult<Vec<_>>>()?;
                for (target, _) in factors.iter().enumerate() {
                    let mut derivative = Complex64::ONE;
                    for (source, value) in values.iter().copied().enumerate() {
                        if source != target {
                            derivative *= value;
                        }
                    }
                    self.accumulate_analytic_scalar(factors[target], adjoint, derivative)?;
                }
            }
            ExprNode::Complex { re, im } => {
                let adjoint = Self::expect_scalar_adjoint(index, adjoint)?;
                let re_part = (adjoint.dz + adjoint.dz_conj) * 0.5;
                let im_part = Complex64::I * (adjoint.dz - adjoint.dz_conj) * 0.5;
                self.accumulate_scalar(
                    re,
                    ScalarAdjoint {
                        dz: re_part,
                        dz_conj: re_part,
                    },
                )?;
                self.accumulate_scalar(
                    im,
                    ScalarAdjoint {
                        dz: im_part,
                        dz_conj: im_part,
                    },
                )?;
            }
            ExprNode::Vector { elements } => {
                let adjoint = Self::expect_vector_adjoint(index, adjoint)?;
                if elements.len() != adjoint.len() {
                    return Err(RuntimeError::InvalidShape {
                        index,
                        message: format!(
                            "vector adjoint has len {}, expected {}",
                            adjoint.len(),
                            elements.len()
                        ),
                    });
                }
                for (element, contribution) in elements.into_iter().zip(adjoint) {
                    self.accumulate_scalar(element, contribution)?;
                }
            }
            ExprNode::Matrix {
                rows,
                cols,
                elements,
            } => {
                let adjoint = Self::expect_matrix_adjoint(index, adjoint)?;
                if adjoint.0 != rows || adjoint.1 != cols || elements.len() != adjoint.2.len() {
                    return Err(RuntimeError::InvalidShape {
                        index,
                        message: format!(
                            "matrix adjoint has shape {}x{}, expected {rows}x{cols}",
                            adjoint.0, adjoint.1
                        ),
                    });
                }
                for (element, contribution) in elements.into_iter().zip(adjoint.2) {
                    self.accumulate_scalar(element, contribution)?;
                }
            }
            ExprNode::Component { input, index: i } => {
                let adjoint = Self::expect_scalar_adjoint(index, adjoint)?;
                if let (Some(plan), Some((cache, row))) =
                    (self.plan.solve_components[index], self.cached_factors)
                {
                    let inverse_row = cache.solve_row(plan.row_slot(), row)?;
                    if inverse_row.len() != plan.dimension() {
                        return Err(RuntimeError::InvalidShape {
                            index,
                            message: format!(
                                "specialized solve expected row len {}, got {}",
                                plan.dimension(),
                                inverse_row.len()
                            ),
                        });
                    }
                    let rhs_contributions = inverse_row
                        .iter()
                        .map(|value| ScalarAdjoint {
                            dz: adjoint.dz * value,
                            dz_conj: adjoint.dz_conj * value.conj(),
                        })
                        .collect::<Vec<_>>();
                    self.accumulate_solve_rhs_adjoint(
                        plan.rhs(),
                        plan.dimension(),
                        rhs_contributions,
                    )?;
                } else {
                    self.accumulate_vector_element(input, i, adjoint)?;
                }
            }
            ExprNode::MatrixElement { input, row, col } => {
                let adjoint = Self::expect_scalar_adjoint(index, adjoint)?;
                let (rows, cols, _) = self.primal_matrix(input)?;
                if row >= rows || col >= cols {
                    return Err(RuntimeError::InvalidShape {
                        index,
                        message: format!(
                            "matrix element ({row}, {col}) out of bounds for shape {rows}x{cols}"
                        ),
                    });
                }
                self.accumulate_matrix_element(input, rows, cols, row, col, adjoint)?;
            }
            ExprNode::MatMul { lhs, rhs } => {
                let (out_rows, out_cols, adjoint) = Self::expect_matrix_adjoint(index, adjoint)?;
                self.propagate_matmul(index, lhs, rhs, out_rows, out_cols, &adjoint)?;
            }
            ExprNode::MatVec { matrix, vector } => {
                let adjoint = Self::expect_vector_adjoint(index, adjoint)?;
                self.propagate_matvec(index, matrix, vector, &adjoint)?;
            }
            ExprNode::Dot { lhs, rhs } => {
                let adjoint = Self::expect_scalar_adjoint(index, adjoint)?;
                self.propagate_dot(index, lhs, rhs, adjoint)?;
            }
            ExprNode::Solve { matrix, rhs } => {
                let adjoint = Self::expect_vector_adjoint(index, adjoint)?;
                self.propagate_solve(index, matrix, rhs, &adjoint)?;
            }
            ExprNode::RealConst(_)
            | ExprNode::ComplexConst(_)
            | ExprNode::ScalarParam(_)
            | ExprNode::EventScalar(_)
            | ExprNode::EventP4Component { .. } => {}
        }
        Ok(())
    }

    fn propagate_unary(
        &mut self,
        op: UnaryOp,
        input: ExprId,
        input_value: Complex64,
        output_value: Complex64,
        adjoint: ScalarAdjoint,
    ) -> RuntimeResult<()> {
        match op {
            UnaryOp::Neg => self.accumulate_scalar(
                input,
                ScalarAdjoint {
                    dz: -adjoint.dz,
                    dz_conj: -adjoint.dz_conj,
                },
            ),
            UnaryOp::Real => {
                let contribution = (adjoint.dz + adjoint.dz_conj) * 0.5;
                self.accumulate_scalar(
                    input,
                    ScalarAdjoint {
                        dz: contribution,
                        dz_conj: contribution,
                    },
                )
            }
            UnaryOp::Imag => {
                let contribution = -Complex64::I * (adjoint.dz + adjoint.dz_conj) * 0.5;
                self.accumulate_scalar(
                    input,
                    ScalarAdjoint {
                        dz: contribution,
                        dz_conj: -contribution,
                    },
                )
            }
            UnaryOp::Conj => self.accumulate_scalar(
                input,
                ScalarAdjoint {
                    dz: adjoint.dz_conj,
                    dz_conj: adjoint.dz,
                },
            ),
            UnaryOp::NormSqr => {
                let sum = adjoint.dz + adjoint.dz_conj;
                self.accumulate_scalar(
                    input,
                    ScalarAdjoint {
                        dz: sum * input_value.conj(),
                        dz_conj: sum * input_value,
                    },
                )
            }
            UnaryOp::Sqrt => {
                self.accumulate_analytic_scalar(input, adjoint, 1.0 / (2.0 * output_value))
            }
            UnaryOp::Exp => self.accumulate_analytic_scalar(input, adjoint, output_value),
            UnaryOp::Sin => self.accumulate_analytic_scalar(input, adjoint, input_value.cos()),
            UnaryOp::Cos => self.accumulate_analytic_scalar(input, adjoint, -input_value.sin()),
            UnaryOp::Log => self.accumulate_analytic_scalar(input, adjoint, 1.0 / input_value),
            UnaryOp::PowI(power) => {
                let derivative = if power == 0 {
                    Complex64::ZERO
                } else if power == i32::MIN {
                    power as f64 * output_value / input_value
                } else {
                    power as f64 * input_value.powi(power - 1)
                };
                self.accumulate_analytic_scalar(input, adjoint, derivative)
            }
        }
    }

    fn propagate_binary(
        &mut self,
        op: BinaryOp,
        lhs: ExprId,
        rhs: ExprId,
        lhs_value: Complex64,
        rhs_value: Complex64,
        adjoint: ScalarAdjoint,
    ) -> RuntimeResult<()> {
        match op {
            BinaryOp::Add => {
                self.accumulate_scalar(lhs, adjoint)?;
                self.accumulate_scalar(rhs, adjoint)
            }
            BinaryOp::Sub => {
                self.accumulate_scalar(lhs, adjoint)?;
                self.accumulate_scalar(
                    rhs,
                    ScalarAdjoint {
                        dz: -adjoint.dz,
                        dz_conj: -adjoint.dz_conj,
                    },
                )
            }
            BinaryOp::Mul => {
                self.accumulate_analytic_scalar(lhs, adjoint, rhs_value)?;
                self.accumulate_analytic_scalar(rhs, adjoint, lhs_value)
            }
            BinaryOp::Div => {
                self.accumulate_analytic_scalar(lhs, adjoint, 1.0 / rhs_value)?;
                self.accumulate_analytic_scalar(rhs, adjoint, -lhs_value / rhs_value.powi(2))
            }
            BinaryOp::Atan2 => {
                let denominator = lhs_value.re.powi(2) + rhs_value.re.powi(2);
                let sum = adjoint.dz + adjoint.dz_conj;
                self.accumulate_real_linear_scalar(lhs, sum * rhs_value.re / denominator)?;
                self.accumulate_real_linear_scalar(rhs, -sum * lhs_value.re / denominator)
            }
        }
    }

    fn propagate_matmul(
        &mut self,
        index: usize,
        lhs: ExprId,
        rhs: ExprId,
        out_rows: usize,
        out_cols: usize,
        adjoint: &[ScalarAdjoint],
    ) -> RuntimeResult<()> {
        let (lhs_rows, lhs_cols, lhs_value) = self.primal_matrix(lhs)?;
        let (rhs_rows, rhs_cols, rhs_value) = self.primal_matrix(rhs)?;
        if lhs_cols != rhs_rows || lhs_rows != out_rows || rhs_cols != out_cols {
            return Err(RuntimeError::InvalidShape {
                index,
                message: format!(
                    "matmul adjoint has shape {out_rows}x{out_cols} for {lhs_rows}x{lhs_cols} by {rhs_rows}x{rhs_cols}"
                ),
            });
        }
        let mut lhs_adjoint = vec![ScalarAdjoint::default(); lhs_rows * lhs_cols];
        let mut rhs_adjoint = vec![ScalarAdjoint::default(); rhs_rows * rhs_cols];
        for row in 0..lhs_rows {
            for col in 0..rhs_cols {
                let output_adjoint = adjoint[row * rhs_cols + col];
                for mid in 0..lhs_cols {
                    let rhs_entry = rhs_value[mid * rhs_cols + col];
                    let lhs_entry = lhs_value[row * lhs_cols + mid];
                    let lhs_target = &mut lhs_adjoint[row * lhs_cols + mid];
                    lhs_target.dz += output_adjoint.dz * rhs_entry;
                    lhs_target.dz_conj += output_adjoint.dz_conj * rhs_entry.conj();
                    let rhs_target = &mut rhs_adjoint[mid * rhs_cols + col];
                    rhs_target.dz += output_adjoint.dz * lhs_entry;
                    rhs_target.dz_conj += output_adjoint.dz_conj * lhs_entry.conj();
                }
            }
        }
        self.accumulate_matrix(lhs, lhs_rows, lhs_cols, lhs_adjoint)?;
        self.accumulate_matrix(rhs, rhs_rows, rhs_cols, rhs_adjoint)
    }

    fn propagate_matvec(
        &mut self,
        index: usize,
        matrix: ExprId,
        vector: ExprId,
        adjoint: &[ScalarAdjoint],
    ) -> RuntimeResult<()> {
        let (rows, cols, matrix_value) = self.primal_matrix(matrix)?;
        let vector_value = self.primal_vector(vector)?;
        if cols != vector_value.len() || rows != adjoint.len() {
            return Err(RuntimeError::InvalidShape {
                index,
                message: format!(
                    "matvec adjoint has len {}, expected {rows} for matrix {rows}x{cols} and vector len {}",
                    adjoint.len(),
                    vector_value.len()
                ),
            });
        }
        let mut matrix_adjoint = vec![ScalarAdjoint::default(); rows * cols];
        let mut vector_adjoint = vec![ScalarAdjoint::default(); cols];
        for row in 0..rows {
            let output_adjoint = adjoint[row];
            for col in 0..cols {
                let vector_entry = vector_value[col];
                let matrix_entry = matrix_value[row * cols + col];
                let matrix_target = &mut matrix_adjoint[row * cols + col];
                matrix_target.dz += output_adjoint.dz * vector_entry;
                matrix_target.dz_conj += output_adjoint.dz_conj * vector_entry.conj();
                vector_adjoint[col].dz += output_adjoint.dz * matrix_entry;
                vector_adjoint[col].dz_conj += output_adjoint.dz_conj * matrix_entry.conj();
            }
        }
        self.accumulate_matrix(matrix, rows, cols, matrix_adjoint)?;
        self.accumulate_vector(vector, vector_adjoint)
    }

    fn propagate_dot(
        &mut self,
        index: usize,
        lhs: ExprId,
        rhs: ExprId,
        adjoint: ScalarAdjoint,
    ) -> RuntimeResult<()> {
        let lhs_value = self.primal_vector(lhs)?;
        let rhs_value = self.primal_vector(rhs)?;
        if lhs_value.len() != rhs_value.len() {
            return Err(RuntimeError::InvalidShape {
                index,
                message: format!(
                    "cannot dot len {} vector with len {} vector",
                    lhs_value.len(),
                    rhs_value.len()
                ),
            });
        }
        let lhs_adjoint = rhs_value
            .iter()
            .map(|value| ScalarAdjoint {
                dz: adjoint.dz * value,
                dz_conj: adjoint.dz_conj * value.conj(),
            })
            .collect();
        let rhs_adjoint = lhs_value
            .iter()
            .map(|value| ScalarAdjoint {
                dz: adjoint.dz * value,
                dz_conj: adjoint.dz_conj * value.conj(),
            })
            .collect();
        self.accumulate_vector(lhs, lhs_adjoint)?;
        self.accumulate_vector(rhs, rhs_adjoint)
    }

    fn propagate_solve(
        &mut self,
        index: usize,
        matrix: ExprId,
        rhs: ExprId,
        adjoint: &[ScalarAdjoint],
    ) -> RuntimeResult<()> {
        let (rows, cols, matrix_value) = self.primal_matrix(matrix)?;
        let rhs_value = self.primal_vector(rhs)?;
        let solution = self.primal_vector(ExprId::from_index(index))?;
        if rows != cols || rows != rhs_value.len() || rows != adjoint.len() {
            return Err(RuntimeError::InvalidShape {
                index,
                message: format!(
                    "solve adjoint has len {}, expected {rows} for {rows}x{cols} solve",
                    adjoint.len()
                ),
            });
        }
        let matrix_value = DMatrix::from_row_slice(rows, cols, matrix_value);
        let transposed = matrix_value.transpose();
        let conjugate_transposed = matrix_value.map(|value| value.conj()).transpose();
        let alpha = DVector::from_iterator(rows, adjoint.iter().map(|adjoint| adjoint.dz));
        let beta = DVector::from_iterator(rows, adjoint.iter().map(|adjoint| adjoint.dz_conj));
        let lambda = transposed
            .lu()
            .solve(&alpha)
            .ok_or(RuntimeError::SingularMatrix(index))?;
        let lambda_conj = conjugate_transposed
            .lu()
            .solve(&beta)
            .ok_or(RuntimeError::SingularMatrix(index))?;
        let solution = DVector::from_row_slice(solution);
        let mut matrix_adjoint = vec![ScalarAdjoint::default(); rows * cols];
        for row in 0..rows {
            for col in 0..cols {
                matrix_adjoint[row * cols + col].dz -= lambda[row] * solution[col];
                matrix_adjoint[row * cols + col].dz_conj -= lambda_conj[row] * solution[col].conj();
            }
        }
        let rhs_adjoint = (0..rows)
            .map(|row| ScalarAdjoint {
                dz: lambda[row],
                dz_conj: lambda_conj[row],
            })
            .collect();
        self.accumulate_matrix(matrix, rows, cols, matrix_adjoint)?;
        self.accumulate_vector(rhs, rhs_adjoint)
    }

    fn accumulate_analytic_scalar(
        &mut self,
        id: ExprId,
        adjoint: ScalarAdjoint,
        derivative: Complex64,
    ) -> RuntimeResult<()> {
        self.accumulate_scalar(
            id,
            ScalarAdjoint {
                dz: adjoint.dz * derivative,
                dz_conj: adjoint.dz_conj * derivative.conj(),
            },
        )
    }

    fn accumulate_real_linear_scalar(
        &mut self,
        id: ExprId,
        contribution: Complex64,
    ) -> RuntimeResult<()> {
        self.accumulate_scalar(
            id,
            ScalarAdjoint {
                dz: contribution * 0.5,
                dz_conj: contribution * 0.5,
            },
        )
    }

    fn accumulate_solve_rhs_adjoint(
        &mut self,
        rhs: ExprId,
        len: usize,
        contributions: Vec<ScalarAdjoint>,
    ) -> RuntimeResult<()> {
        if let Some(elements) = &self.plan.solve_rhs_elements[rhs.index()] {
            if elements.len() != len || elements.len() != contributions.len() {
                return Err(RuntimeError::InvalidShape {
                    index: rhs.index(),
                    message: format!(
                        "specialized solve expected {len} RHS elements, got {}",
                        elements.len()
                    ),
                });
            }
            for (element, contribution) in elements.iter().copied().zip(contributions) {
                self.accumulate_scalar(element, contribution)?;
            }
            Ok(())
        } else {
            self.accumulate_vector(rhs, contributions)
        }
    }

    fn accumulate_scalar(&mut self, id: ExprId, contribution: ScalarAdjoint) -> RuntimeResult<()> {
        match &mut self.adjoints[id.index()] {
            Some(ReverseAdjoint::Scalar(adjoint)) => {
                adjoint.dz += contribution.dz;
                adjoint.dz_conj += contribution.dz_conj;
            }
            Some(value) => {
                return Err(RuntimeError::TypeMismatch {
                    index: id.index(),
                    expected: "scalar adjoint",
                    actual: value.kind(),
                });
            }
            None => {
                self.adjoints[id.index()] = Some(ReverseAdjoint::Scalar(contribution));
            }
        }
        Ok(())
    }

    fn accumulate_vector(
        &mut self,
        id: ExprId,
        contributions: Vec<ScalarAdjoint>,
    ) -> RuntimeResult<()> {
        match &mut self.adjoints[id.index()] {
            Some(ReverseAdjoint::Vector(adjoint)) if adjoint.len() == contributions.len() => {
                for (target, source) in adjoint.iter_mut().zip(contributions) {
                    target.dz += source.dz;
                    target.dz_conj += source.dz_conj;
                }
            }
            Some(ReverseAdjoint::Vector(adjoint)) => {
                return Err(RuntimeError::InvalidShape {
                    index: id.index(),
                    message: format!(
                        "vector adjoint has len {}, expected {}",
                        adjoint.len(),
                        contributions.len()
                    ),
                });
            }
            Some(value) => {
                return Err(RuntimeError::TypeMismatch {
                    index: id.index(),
                    expected: "vector adjoint",
                    actual: value.kind(),
                });
            }
            None => {
                self.adjoints[id.index()] = Some(ReverseAdjoint::Vector(contributions));
            }
        }
        Ok(())
    }

    fn accumulate_vector_element(
        &mut self,
        id: ExprId,
        element: usize,
        contribution: ScalarAdjoint,
    ) -> RuntimeResult<()> {
        let len = self.primal_vector(id)?.len();
        if element >= len {
            return Err(RuntimeError::InvalidShape {
                index: id.index(),
                message: format!("component index {element} out of bounds for len {len}"),
            });
        }
        let mut contributions = vec![ScalarAdjoint::default(); len];
        contributions[element] = contribution;
        self.accumulate_vector(id, contributions)
    }

    fn accumulate_matrix(
        &mut self,
        id: ExprId,
        rows: usize,
        cols: usize,
        contributions: Vec<ScalarAdjoint>,
    ) -> RuntimeResult<()> {
        match &mut self.adjoints[id.index()] {
            Some(ReverseAdjoint::Matrix {
                rows: actual_rows,
                cols: actual_cols,
                values,
            }) if *actual_rows == rows
                && *actual_cols == cols
                && values.len() == contributions.len() =>
            {
                for (target, source) in values.iter_mut().zip(contributions) {
                    target.dz += source.dz;
                    target.dz_conj += source.dz_conj;
                }
            }
            Some(ReverseAdjoint::Matrix {
                rows: actual_rows,
                cols: actual_cols,
                ..
            }) => {
                return Err(RuntimeError::InvalidShape {
                    index: id.index(),
                    message: format!(
                        "matrix adjoint has shape {actual_rows}x{actual_cols}, expected {rows}x{cols}"
                    ),
                });
            }
            Some(value) => {
                return Err(RuntimeError::TypeMismatch {
                    index: id.index(),
                    expected: "matrix adjoint",
                    actual: value.kind(),
                });
            }
            None => {
                self.adjoints[id.index()] = Some(ReverseAdjoint::Matrix {
                    rows,
                    cols,
                    values: contributions,
                });
            }
        }
        Ok(())
    }

    fn accumulate_matrix_element(
        &mut self,
        id: ExprId,
        rows: usize,
        cols: usize,
        row: usize,
        col: usize,
        contribution: ScalarAdjoint,
    ) -> RuntimeResult<()> {
        let mut contributions = vec![ScalarAdjoint::default(); rows * cols];
        contributions[row * cols + col] = contribution;
        self.accumulate_matrix(id, rows, cols, contributions)
    }

    fn scalar_adjoint_at(&self, index: usize) -> RuntimeResult<Option<ScalarAdjoint>> {
        match &self.adjoints[index] {
            Some(ReverseAdjoint::Scalar(adjoint)) => Ok(Some(*adjoint)),
            Some(value) => Err(RuntimeError::TypeMismatch {
                index,
                expected: "scalar adjoint",
                actual: value.kind(),
            }),
            None => Ok(None),
        }
    }

    fn expect_scalar_adjoint(
        index: usize,
        adjoint: ReverseAdjoint,
    ) -> RuntimeResult<ScalarAdjoint> {
        match adjoint {
            ReverseAdjoint::Scalar(adjoint) => Ok(adjoint),
            value => Err(RuntimeError::TypeMismatch {
                index,
                expected: "scalar adjoint",
                actual: value.kind(),
            }),
        }
    }

    fn expect_vector_adjoint(
        index: usize,
        adjoint: ReverseAdjoint,
    ) -> RuntimeResult<Vec<ScalarAdjoint>> {
        match adjoint {
            ReverseAdjoint::Vector(adjoint) => Ok(adjoint),
            value => Err(RuntimeError::TypeMismatch {
                index,
                expected: "vector adjoint",
                actual: value.kind(),
            }),
        }
    }

    fn expect_matrix_adjoint(
        index: usize,
        adjoint: ReverseAdjoint,
    ) -> RuntimeResult<(usize, usize, Vec<ScalarAdjoint>)> {
        match adjoint {
            ReverseAdjoint::Matrix { rows, cols, values } => Ok((rows, cols, values)),
            value => Err(RuntimeError::TypeMismatch {
                index,
                expected: "matrix adjoint",
                actual: value.kind(),
            }),
        }
    }

    fn primal_scalar(&self, id: ExprId) -> RuntimeResult<Complex64> {
        if self.cached_factors.is_some() {
            self.plan.cached_scalar_at(self.primals, id)
        } else {
            scalar_at(self.primals, id.index())
        }
    }

    fn primal_vector(&self, id: ExprId) -> RuntimeResult<&[Complex64]> {
        if self.cached_factors.is_some() {
            self.plan.cached_vector_at(self.primals, id)
        } else {
            vector_at(self.primals, id.index())
        }
    }

    fn primal_matrix(&self, id: ExprId) -> RuntimeResult<(usize, usize, &[Complex64])> {
        if self.cached_factors.is_some() {
            self.plan.cached_matrix_at(self.primals, id)
        } else {
            matrix_at(self.primals, id.index())
        }
    }
}
