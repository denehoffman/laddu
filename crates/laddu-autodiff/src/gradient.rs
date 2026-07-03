use laddu_expr::{BinaryOp, UnaryOp, parameters::ParamId};
use laddu_kernel::ir::{
    GradientKernelIr, KernelInstruction, KernelIrBuilder, KernelValueId, KernelValueKind,
    OutputComponent, ScalarKernelIr,
};
use num::complex::Complex64;

use crate::{AutodiffError, AutodiffResult};

pub fn gradient_ir(
    primal: &ScalarKernelIr,
    free_params: &[ParamId],
    component: OutputComponent,
) -> AutodiffResult<GradientKernelIr> {
    GradientBuilder::new(primal, component).build(free_params)
}

struct GradientBuilder<'a> {
    primal: &'a ScalarKernelIr,
    builder: KernelIrBuilder,
    contributions: Vec<Vec<Vec<KernelValueId>>>,
    adjoints: Vec<Option<Vec<KernelValueId>>>,
    component: OutputComponent,
}

impl<'a> GradientBuilder<'a> {
    fn new(primal: &'a ScalarKernelIr, component: OutputComponent) -> Self {
        let contributions = primal
            .values()
            .iter()
            .map(|value| vec![Vec::new(); value.kind.width()])
            .collect();
        Self {
            primal,
            builder: KernelIrBuilder::from_scalar(primal),
            contributions,
            adjoints: vec![None; primal.values().len()],
            component,
        }
    }

    fn build(mut self, free_params: &[ParamId]) -> AutodiffResult<GradientKernelIr> {
        let root = self.primal.root();
        let root_kind = self.primal.values()[root.index()].kind;
        let seed = match (root_kind, self.component) {
            (KernelValueKind::Real, OutputComponent::Real) => self.real(1.0)?,
            (KernelValueKind::Real, OutputComponent::Imag) => self.real(0.0)?,
            (KernelValueKind::Complex, OutputComponent::Real) => self.real(1.0)?,
            (KernelValueKind::Complex, OutputComponent::Imag) => self.complex(0.0, 1.0)?,
            _ => {
                return Err(AutodiffError::InvalidKernel(
                    "gradient root must be scalar".into(),
                ));
            }
        };
        self.contributions[root.index()][0].push(seed);

        for index in (0..self.primal.values().len()).rev() {
            let Some(adjoint) = self.resolve_adjoint(index)? else {
                continue;
            };
            self.propagate(index, &adjoint)?;
            self.adjoints[index] = Some(adjoint);
        }

        let mut outputs = Vec::with_capacity(free_params.len());
        for parameter in free_params {
            let mut terms = Vec::new();
            for (index, value) in self.primal.values().iter().enumerate() {
                if matches!(value.instruction, KernelInstruction::Parameter(id) if id == *parameter)
                    && let Some(adjoint) = &self.adjoints[index]
                {
                    terms.push(self.unary(UnaryOp::Real, adjoint[0])?);
                }
            }
            outputs.push(self.sum_or_zero(terms)?);
        }
        self.builder
            .finish_gradient(root, outputs, self.component)
            .map_err(Into::into)
    }

    fn resolve_adjoint(&mut self, index: usize) -> AutodiffResult<Option<Vec<KernelValueId>>> {
        if self.contributions[index].iter().all(Vec::is_empty) {
            return Ok(None);
        }
        let mut values = Vec::with_capacity(self.contributions[index].len());
        for element in 0..self.contributions[index].len() {
            let terms = std::mem::take(&mut self.contributions[index][element]);
            values.push(self.sum_or_zero(terms)?);
        }
        Ok(Some(values))
    }

    fn propagate(&mut self, index: usize, adjoint: &[KernelValueId]) -> AutodiffResult<()> {
        let value = &self.primal.values()[index];
        match &value.instruction {
            KernelInstruction::Cached(_)
            | KernelInstruction::RealConstant(_)
            | KernelInstruction::ComplexConstant(_)
            | KernelInstruction::Parameter(_) => {}
            KernelInstruction::Unary { op, input } => {
                let contribution =
                    self.unary_pullback(*op, *input, KernelValueId::from_index(index), adjoint[0])?;
                self.accumulate(*input, &[contribution])?;
            }
            KernelInstruction::Binary { op, lhs, rhs } => {
                let (lhs_contribution, rhs_contribution) =
                    self.binary_pullback(*op, *lhs, *rhs, adjoint[0])?;
                self.accumulate(*lhs, &[lhs_contribution])?;
                self.accumulate(*rhs, &[rhs_contribution])?;
            }
            KernelInstruction::Add(terms) => {
                for term in terms {
                    self.accumulate(*term, adjoint)?;
                }
            }
            KernelInstruction::Mul(factors) => self.product_pullback(factors, adjoint[0])?,
            KernelInstruction::Complex { re, im } => {
                let re_contribution = self.unary(UnaryOp::Real, adjoint[0])?;
                let im_contribution = self.unary(UnaryOp::Imag, adjoint[0])?;
                self.accumulate(*re, &[re_contribution])?;
                self.accumulate(*im, &[im_contribution])?;
            }
            KernelInstruction::Vector(entries)
            | KernelInstruction::Matrix {
                elements: entries, ..
            } => {
                for (entry, contribution) in entries.iter().zip(adjoint) {
                    self.accumulate(*entry, &[*contribution])?;
                }
            }
            KernelInstruction::Component { input, index } => {
                self.accumulate_element(*input, *index, adjoint[0])?;
            }
            KernelInstruction::MatrixElement { input, row, col } => {
                let KernelValueKind::Matrix { cols, .. } = self.kind(*input) else {
                    unreachable!()
                };
                self.accumulate_element(*input, row * cols + col, adjoint[0])?;
            }
            KernelInstruction::MatMul { lhs, rhs } => {
                self.matmul_pullback(*lhs, *rhs, adjoint)?;
            }
            KernelInstruction::MatVec { matrix, vector } => {
                self.matvec_pullback(*matrix, *vector, adjoint)?;
            }
            KernelInstruction::Dot { lhs, rhs } => {
                self.dot_pullback(*lhs, *rhs, adjoint[0])?;
            }
            KernelInstruction::Solve { matrix, rhs } => {
                self.solve_pullback(*matrix, *rhs, KernelValueId::from_index(index), adjoint)?;
            }
            KernelInstruction::SolveRow { row_slot, rhs } => {
                self.solve_row_pullback(*row_slot, rhs, adjoint[0])?;
            }
        }
        Ok(())
    }

    fn unary_pullback(
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
                self.push(KernelInstruction::Complex {
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
            UnaryOp::PowI(power) if power == 0 => self.real(0.0)?,
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

    fn binary_pullback(
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

    fn product_pullback(
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

    fn matmul_pullback(
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

    fn matvec_pullback(
        &mut self,
        matrix: KernelValueId,
        vector: KernelValueId,
        adjoint: &[KernelValueId],
    ) -> AutodiffResult<()> {
        let KernelValueKind::Matrix { rows, cols } = self.kind(matrix) else {
            unreachable!()
        };
        for row in 0..rows {
            for col in 0..cols {
                let vector_value = self.component(vector, col)?;
                let matrix_contribution = self.mul_conj(adjoint[row], vector_value)?;
                self.accumulate_element(matrix, row * cols + col, matrix_contribution)?;
                let matrix_value = self.matrix_element(matrix, row, col)?;
                let vector_contribution = self.mul_conj(adjoint[row], matrix_value)?;
                self.accumulate_element(vector, col, vector_contribution)?;
            }
        }
        Ok(())
    }

    fn dot_pullback(
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

    fn solve_pullback(
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

    fn solve_row_pullback(
        &mut self,
        row_slot: usize,
        rhs: &[KernelValueId],
        adjoint: KernelValueId,
    ) -> AutodiffResult<()> {
        let conjugate_adjoint = self.unary(UnaryOp::Conj, adjoint)?;
        let zero = self.real(0.0)?;
        for index in 0..rhs.len() {
            let mut basis = vec![zero; rhs.len()];
            basis[index] = conjugate_adjoint;
            let product = self.push(KernelInstruction::SolveRow {
                row_slot,
                rhs: basis,
            })?;
            let contribution = self.unary(UnaryOp::Conj, product)?;
            self.accumulate(rhs[index], &[contribution])?;
        }
        Ok(())
    }

    fn accumulate(
        &mut self,
        target: KernelValueId,
        contribution: &[KernelValueId],
    ) -> AutodiffResult<()> {
        if self.contributions[target.index()].len() != contribution.len() {
            return Err(AutodiffError::InvalidKernel(format!(
                "gradient contribution width {} does not match value {} width {}",
                contribution.len(),
                target.index(),
                self.contributions[target.index()].len(),
            )));
        }
        for (target, contribution) in self.contributions[target.index()]
            .iter_mut()
            .zip(contribution)
        {
            target.push(*contribution);
        }
        Ok(())
    }

    fn accumulate_element(
        &mut self,
        target: KernelValueId,
        element: usize,
        contribution: KernelValueId,
    ) -> AutodiffResult<()> {
        let width = self.contributions[target.index()].len();
        let Some(target) = self.contributions[target.index()].get_mut(element) else {
            return Err(AutodiffError::InvalidKernel(format!(
                "gradient contribution element {element} exceeds value {} width {width}",
                target.index(),
            )));
        };
        target.push(contribution);
        Ok(())
    }

    fn kind(&self, id: KernelValueId) -> KernelValueKind {
        self.primal.values()[id.index()].kind
    }

    fn push(&mut self, instruction: KernelInstruction) -> AutodiffResult<KernelValueId> {
        self.builder.push(instruction).map_err(Into::into)
    }

    fn real(&mut self, value: f64) -> AutodiffResult<KernelValueId> {
        self.push(KernelInstruction::RealConstant(value))
    }

    fn complex(&mut self, re: f64, im: f64) -> AutodiffResult<KernelValueId> {
        self.push(KernelInstruction::ComplexConstant(Complex64::new(re, im)))
    }

    fn unary(&mut self, op: UnaryOp, input: KernelValueId) -> AutodiffResult<KernelValueId> {
        self.push(KernelInstruction::Unary { op, input })
    }

    fn binary(
        &mut self,
        op: BinaryOp,
        lhs: KernelValueId,
        rhs: KernelValueId,
    ) -> AutodiffResult<KernelValueId> {
        self.push(KernelInstruction::Binary { op, lhs, rhs })
    }

    fn add(&mut self, lhs: KernelValueId, rhs: KernelValueId) -> AutodiffResult<KernelValueId> {
        self.push(KernelInstruction::Add(vec![lhs, rhs]))
    }

    fn mul(&mut self, lhs: KernelValueId, rhs: KernelValueId) -> AutodiffResult<KernelValueId> {
        self.push(KernelInstruction::Mul(vec![lhs, rhs]))
    }

    fn mul_conj(
        &mut self,
        lhs: KernelValueId,
        rhs: KernelValueId,
    ) -> AutodiffResult<KernelValueId> {
        let rhs = self.unary(UnaryOp::Conj, rhs)?;
        self.mul(lhs, rhs)
    }

    fn component(&mut self, input: KernelValueId, index: usize) -> AutodiffResult<KernelValueId> {
        self.push(KernelInstruction::Component { input, index })
    }

    fn matrix_element(
        &mut self,
        input: KernelValueId,
        row: usize,
        col: usize,
    ) -> AutodiffResult<KernelValueId> {
        self.push(KernelInstruction::MatrixElement { input, row, col })
    }

    fn sum_or_zero(&mut self, mut terms: Vec<KernelValueId>) -> AutodiffResult<KernelValueId> {
        match terms.len() {
            0 => self.real(0.0),
            1 => Ok(terms[0]),
            _ => self.push(KernelInstruction::Add(std::mem::take(&mut terms))),
        }
    }
}

#[cfg(test)]
mod tests {
    use laddu_expr::parameters::{ParamRegistry, Parameter};
    use laddu_kernel::ir::{KernelValue, KernelValueClass};

    use super::*;

    fn value(kind: KernelValueKind, instruction: KernelInstruction) -> KernelValue {
        KernelValue {
            kind,
            class: KernelValueClass::Invariant,
            instruction,
        }
    }

    #[test]
    fn scalar_gradient_program_has_one_real_output_per_parameter() {
        let mut registry = ParamRegistry::new();
        let x = registry.register(Parameter::free("x")).unwrap();
        let primal = ScalarKernelIr::new(
            vec![
                value(KernelValueKind::Real, KernelInstruction::Parameter(x)),
                value(
                    KernelValueKind::Real,
                    KernelInstruction::Mul(vec![
                        KernelValueId::from_index(0),
                        KernelValueId::from_index(0),
                    ]),
                ),
            ],
            KernelValueId::from_index(1),
        )
        .unwrap();

        let gradient = gradient_ir(&primal, &[x], OutputComponent::Real).unwrap();

        assert_eq!(gradient.outputs().len(), 1);
        assert_eq!(
            gradient.values()[gradient.outputs()[0].index()].kind,
            KernelValueKind::Real
        );
        assert!(gradient.values().len() > primal.values().len());
    }

    #[test]
    fn solve_gradient_program_contains_adjoint_solve() {
        let mut registry = ParamRegistry::new();
        let x = registry.register(Parameter::free("x")).unwrap();
        let primal = ScalarKernelIr::new(
            vec![
                value(KernelValueKind::Real, KernelInstruction::Parameter(x)),
                value(
                    KernelValueKind::Complex,
                    KernelInstruction::ComplexConstant(Complex64::new(2.0, 0.5)),
                ),
                value(
                    KernelValueKind::Matrix { rows: 2, cols: 2 },
                    KernelInstruction::Matrix {
                        rows: 2,
                        cols: 2,
                        elements: vec![
                            KernelValueId::from_index(0),
                            KernelValueId::from_index(1),
                            KernelValueId::from_index(1),
                            KernelValueId::from_index(0),
                        ],
                    },
                ),
                value(
                    KernelValueKind::Vector { len: 2 },
                    KernelInstruction::Vector(vec![
                        KernelValueId::from_index(0),
                        KernelValueId::from_index(1),
                    ]),
                ),
                value(
                    KernelValueKind::Vector { len: 2 },
                    KernelInstruction::Solve {
                        matrix: KernelValueId::from_index(2),
                        rhs: KernelValueId::from_index(3),
                    },
                ),
                value(
                    KernelValueKind::Complex,
                    KernelInstruction::Component {
                        input: KernelValueId::from_index(4),
                        index: 0,
                    },
                ),
            ],
            KernelValueId::from_index(5),
        )
        .unwrap();

        let gradient = gradient_ir(&primal, &[x], OutputComponent::Imag).unwrap();
        let solves = gradient
            .values()
            .iter()
            .filter(|value| matches!(value.instruction, KernelInstruction::Solve { .. }))
            .count();

        assert_eq!(solves, 2);
        assert_eq!(gradient.component(), OutputComponent::Imag);
    }
}
