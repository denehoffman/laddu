use std::collections::HashMap;

use laddu_compile::CompiledModel;
use laddu_expr::{
    BinaryOp, ExprGraph, ExprNode, UnaryOp,
    parameters::{ParamLayout, ParamValues},
};
use nalgebra::{DMatrix, DVector};
use num::complex::Complex64;
use thiserror::Error;

pub type RuntimeResult<T> = Result<T, RuntimeError>;

#[derive(Clone, Debug, Error, PartialEq)]
pub enum RuntimeError {
    #[error("event scalar `{0}` was requested, but no event lookup was provided")]
    MissingEventScalar(String),
    #[error("node #{index} expected {expected}, got {actual}")]
    TypeMismatch {
        index: usize,
        expected: &'static str,
        actual: &'static str,
    },
    #[error("node #{index} has invalid shape: {message}")]
    InvalidShape { index: usize, message: String },
    #[error("matrix solve failed at node #{0}")]
    SingularMatrix(usize),
    #[error("parameter error: {0}")]
    Parameter(String),
}

pub trait EventLookup {
    fn scalar(&self, name: &str) -> Option<Complex64>;
}

impl<F> EventLookup for F
where
    F: for<'a> Fn(&'a str) -> Option<Complex64>,
{
    fn scalar(&self, name: &str) -> Option<Complex64> {
        self(name)
    }
}

impl EventLookup for HashMap<String, Complex64> {
    fn scalar(&self, name: &str) -> Option<Complex64> {
        self.get(name).copied()
    }
}

impl EventLookup for HashMap<String, f64> {
    fn scalar(&self, name: &str) -> Option<Complex64> {
        self.get(name).copied().map(Complex64::from)
    }
}

#[derive(Clone, Debug, Default)]
pub struct CpuBackend;

#[derive(Clone, Debug)]
pub struct CpuPlan {
    graph: ExprGraph,
    params: ParamLayout,
}

impl CpuBackend {
    pub fn prepare(&self, model: &CompiledModel) -> CpuPlan {
        CpuPlan {
            graph: model.graph().clone(),
            params: model.params().clone(),
        }
    }
}

impl CpuPlan {
    pub fn parameter_count(&self) -> usize {
        self.params.len()
    }

    pub fn free_parameter_count(&self) -> usize {
        self.params.n_free()
    }

    pub fn evaluate(&self, params: &ParamValues) -> RuntimeResult<Complex64> {
        self.evaluate_inner(params, None)
    }

    pub fn evaluate_with_event(
        &self,
        params: &ParamValues,
        event: &impl EventLookup,
    ) -> RuntimeResult<Complex64> {
        self.evaluate_inner(params, Some(event))
    }

    fn evaluate_inner(
        &self,
        params: &ParamValues,
        event: Option<&dyn EventLookup>,
    ) -> RuntimeResult<Complex64> {
        let values = self.evaluate_values(params, event)?;
        scalar_at(&values, self.graph.root().index())
    }

    fn evaluate_values(
        &self,
        params: &ParamValues,
        event: Option<&dyn EventLookup>,
    ) -> RuntimeResult<Vec<Value>> {
        let mut values = Vec::with_capacity(self.graph.nodes().len());

        for (index, node) in self.graph.nodes().iter().enumerate() {
            let value = match node {
                ExprNode::RealConst(value) => Value::Scalar(Complex64::from(*value)),
                ExprNode::ComplexConst(value) => Value::Scalar(*value),
                ExprNode::ScalarParam(parameter) => Value::Scalar(Complex64::from(param_value(
                    params,
                    &self.params,
                    parameter.name(),
                )?)),
                ExprNode::ComplexScalarParam { re, im } => Value::Scalar(Complex64::new(
                    param_value(params, &self.params, re.name())?,
                    param_value(params, &self.params, im.name())?,
                )),
                ExprNode::PolarComplexScalarParam { mag, phase } => {
                    let mag = param_value(params, &self.params, mag.name())?;
                    let phase = param_value(params, &self.params, phase.name())?;
                    Value::Scalar(Complex64::cis(phase) * mag)
                }
                ExprNode::EventScalar(name) => {
                    let Some(event) = event else {
                        return Err(RuntimeError::MissingEventScalar(name.to_string()));
                    };
                    Value::Scalar(
                        event
                            .scalar(name)
                            .ok_or_else(|| RuntimeError::MissingEventScalar(name.to_string()))?,
                    )
                }
                ExprNode::Unary { op, input } => {
                    let input = scalar_at(&values, input.index())?;
                    Value::Scalar(eval_unary(*op, input))
                }
                ExprNode::Binary { op, lhs, rhs } => {
                    let lhs = scalar_at(&values, lhs.index())?;
                    let rhs = scalar_at(&values, rhs.index())?;
                    Value::Scalar(eval_binary(*op, lhs, rhs))
                }
                ExprNode::Vector { elements } => Value::Vector(
                    elements
                        .iter()
                        .map(|id| scalar_at(&values, id.index()))
                        .collect::<RuntimeResult<_>>()?,
                ),
                ExprNode::Matrix {
                    rows,
                    cols,
                    elements,
                } => {
                    if elements.len() != rows * cols {
                        return Err(RuntimeError::InvalidShape {
                            index,
                            message: format!(
                                "matrix has {} elements for shape {rows}x{cols}",
                                elements.len()
                            ),
                        });
                    }
                    Value::Matrix {
                        rows: *rows,
                        cols: *cols,
                        values: elements
                            .iter()
                            .map(|id| scalar_at(&values, id.index()))
                            .collect::<RuntimeResult<_>>()?,
                    }
                }
                ExprNode::Component { input, index: i } => {
                    let vector = vector_at(&values, input.index())?;
                    Value::Scalar(*vector.get(*i).ok_or_else(|| RuntimeError::InvalidShape {
                        index,
                        message: format!(
                            "component index {i} out of bounds for len {}",
                            vector.len()
                        ),
                    })?)
                }
                ExprNode::MatrixElement { input, row, col } => {
                    let (rows, cols, matrix) = matrix_at(&values, input.index())?;
                    if *row >= rows || *col >= cols {
                        return Err(RuntimeError::InvalidShape {
                            index,
                            message: format!(
                                "matrix element ({row}, {col}) out of bounds for shape {rows}x{cols}"
                            ),
                        });
                    }
                    Value::Scalar(matrix[row * cols + col])
                }
                ExprNode::MatMul { lhs, rhs } => {
                    let (lhs_rows, lhs_cols, lhs) = matrix_at(&values, lhs.index())?;
                    let (rhs_rows, rhs_cols, rhs) = matrix_at(&values, rhs.index())?;
                    if lhs_cols != rhs_rows {
                        return Err(RuntimeError::InvalidShape {
                            index,
                            message: format!(
                                "cannot multiply {lhs_rows}x{lhs_cols} by {rhs_rows}x{rhs_cols}"
                            ),
                        });
                    }
                    let lhs = DMatrix::from_row_slice(lhs_rows, lhs_cols, lhs);
                    let rhs = DMatrix::from_row_slice(rhs_rows, rhs_cols, rhs);
                    let out = lhs * rhs;
                    Value::Matrix {
                        rows: out.nrows(),
                        cols: out.ncols(),
                        values: matrix_values_row_major(&out),
                    }
                }
                ExprNode::MatVec { matrix, vector } => {
                    let (rows, cols, matrix) = matrix_at(&values, matrix.index())?;
                    let vector = vector_at(&values, vector.index())?;
                    if cols != vector.len() {
                        return Err(RuntimeError::InvalidShape {
                            index,
                            message: format!(
                                "cannot multiply {rows}x{cols} matrix by len {} vector",
                                vector.len()
                            ),
                        });
                    }
                    let matrix = DMatrix::from_row_slice(rows, cols, matrix);
                    let vector = DVector::from_row_slice(vector);
                    Value::Vector((matrix * vector).iter().copied().collect())
                }
                ExprNode::Dot { lhs, rhs } => {
                    let lhs = vector_at(&values, lhs.index())?;
                    let rhs = vector_at(&values, rhs.index())?;
                    if lhs.len() != rhs.len() {
                        return Err(RuntimeError::InvalidShape {
                            index,
                            message: format!(
                                "cannot dot len {} vector with len {} vector",
                                lhs.len(),
                                rhs.len()
                            ),
                        });
                    }
                    Value::Scalar(lhs.iter().zip(rhs).map(|(lhs, rhs)| lhs * rhs).sum())
                }
                ExprNode::Solve { matrix, rhs } => {
                    let (rows, cols, matrix) = matrix_at(&values, matrix.index())?;
                    let rhs = vector_at(&values, rhs.index())?;
                    if rows != cols || rows != rhs.len() {
                        return Err(RuntimeError::InvalidShape {
                            index,
                            message: format!(
                                "cannot solve {rows}x{cols} matrix against len {} vector",
                                rhs.len()
                            ),
                        });
                    }
                    let matrix = DMatrix::from_row_slice(rows, cols, matrix);
                    let rhs = DVector::from_row_slice(rhs);
                    let solution = matrix
                        .lu()
                        .solve(&rhs)
                        .ok_or(RuntimeError::SingularMatrix(index))?;
                    Value::Vector(solution.iter().copied().collect())
                }
            };
            values.push(value);
        }

        Ok(values)
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Value {
    Scalar(Complex64),
    Vector(Vec<Complex64>),
    Matrix {
        rows: usize,
        cols: usize,
        values: Vec<Complex64>,
    },
}

impl Value {
    fn kind(&self) -> &'static str {
        match self {
            Self::Scalar(_) => "scalar",
            Self::Vector(_) => "vector",
            Self::Matrix { .. } => "matrix",
        }
    }
}

fn param_value(params: &ParamValues, layout: &ParamLayout, name: &str) -> RuntimeResult<f64> {
    let id = layout
        .id(name)
        .ok_or_else(|| RuntimeError::Parameter(format!("unknown parameter `{name}`")))?;
    params
        .get(id)
        .map_err(|err| RuntimeError::Parameter(err.to_string()))
}

fn scalar_at(values: &[Value], index: usize) -> RuntimeResult<Complex64> {
    match &values[index] {
        Value::Scalar(value) => Ok(*value),
        value => Err(RuntimeError::TypeMismatch {
            index,
            expected: "scalar",
            actual: value.kind(),
        }),
    }
}

fn vector_at(values: &[Value], index: usize) -> RuntimeResult<&[Complex64]> {
    match &values[index] {
        Value::Vector(value) => Ok(value),
        value => Err(RuntimeError::TypeMismatch {
            index,
            expected: "vector",
            actual: value.kind(),
        }),
    }
}

fn matrix_at(values: &[Value], index: usize) -> RuntimeResult<(usize, usize, &[Complex64])> {
    match &values[index] {
        Value::Matrix { rows, cols, values } => Ok((*rows, *cols, values)),
        value => Err(RuntimeError::TypeMismatch {
            index,
            expected: "matrix",
            actual: value.kind(),
        }),
    }
}

fn matrix_values_row_major(matrix: &DMatrix<Complex64>) -> Vec<Complex64> {
    let mut values = Vec::with_capacity(matrix.nrows() * matrix.ncols());
    for row in 0..matrix.nrows() {
        for col in 0..matrix.ncols() {
            values.push(matrix[(row, col)]);
        }
    }
    values
}

fn eval_unary(op: UnaryOp, input: Complex64) -> Complex64 {
    match op {
        UnaryOp::Neg => -input,
        UnaryOp::Real => Complex64::from(input.re),
        UnaryOp::Imag => Complex64::from(input.im),
        UnaryOp::Conj => input.conj(),
        UnaryOp::NormSqr => Complex64::from(input.norm_sqr()),
        UnaryOp::Sqrt => input.sqrt(),
        UnaryOp::Exp => input.exp(),
        UnaryOp::Sin => input.sin(),
        UnaryOp::Cos => input.cos(),
        UnaryOp::Log => input.ln(),
        UnaryOp::PowI(power) => input.powi(power),
    }
}

fn eval_binary(op: BinaryOp, lhs: Complex64, rhs: Complex64) -> Complex64 {
    match op {
        BinaryOp::Add => lhs + rhs,
        BinaryOp::Sub => lhs - rhs,
        BinaryOp::Mul => lhs * rhs,
        BinaryOp::Div => lhs / rhs,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use laddu_compile::CompiledModel;
    use laddu_expr::{complex, dot, matrix, parameter, solve, vector};

    use super::*;

    fn evaluate(expr: &laddu_expr::Expr) -> Complex64 {
        let model = CompiledModel::from_expr(expr).unwrap();
        let params = Arc::new(model.params().clone()).default_values();
        CpuBackend.prepare(&model).evaluate(&params).unwrap()
    }

    #[test]
    fn evaluates_scalar_expression_with_parameters() {
        let expr = (2.0 * parameter!("x", initial: 3.0)
            + complex(
                parameter!("re", initial: 1.0),
                parameter!("im", initial: 2.0),
            ))
        .norm_sqr();

        assert_eq!(evaluate(&expr), Complex64::from(53.0));
    }

    #[test]
    fn evaluates_event_scalars() {
        let expr = laddu_expr::event_scalar("x") * 2.0;
        let model = CompiledModel::from_expr(&expr).unwrap();
        let params = Arc::new(model.params().clone()).default_values();
        let plan = CpuBackend.prepare(&model);
        let event = HashMap::from([("x".to_owned(), Complex64::from(3.0))]);

        assert_eq!(
            plan.evaluate_with_event(&params, &event).unwrap(),
            Complex64::from(6.0)
        );
    }

    #[test]
    fn evaluates_linear_algebra_nodes() {
        let a = matrix([[2.0, 0.0], [0.0, 4.0]]);
        let b = vector([8.0, 12.0]);
        let x = solve(a, b);
        let expr = dot(&x, vector([1.0, 1.0]));

        assert_eq!(evaluate(&expr), Complex64::from(7.0));
    }
}
