use std::{
    collections::HashMap,
    ops::{Add, Div, Mul, Neg, Sub},
    sync::Arc,
};

use laddu_params::{ParamId, ParamLayout, ParamValues};
use num::complex::Complex64;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub type ExprResult<T> = Result<T, ExprError>;

#[derive(Clone, Debug, Error)]
pub enum ExprError {
    #[error("unknown parameter id #{0}")]
    UnknownParameter(usize),
    #[error("parameter error: {0}")]
    Parameter(#[from] laddu_params::ParamError),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ExprId(u32);

impl ExprId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UnaryOp {
    Neg,
    Real,
    Imag,
    Conj,
    NormSqr,
    Sqrt,
    Exp,
    Sin,
    Cos,
    Log,
    PowI(i32),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ExprNode {
    Const(Complex64),
    Param(ParamId),
    Unary {
        op: UnaryOp,
        input: ExprId,
    },
    Binary {
        op: BinaryOp,
        lhs: ExprId,
        rhs: ExprId,
    },
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ExprGraph {
    nodes: Vec<ExprNode>,
}

impl ExprGraph {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    pub fn constant(&mut self, value: impl Into<Complex64>) -> Expr {
        self.push(ExprNode::Const(value.into()))
    }

    pub fn real_constant(&mut self, value: f64) -> Expr {
        self.constant(Complex64::new(value, 0.0))
    }

    pub fn param(&mut self, id: ParamId) -> Expr {
        self.push(ExprNode::Param(id))
    }

    pub fn node(&self, id: ExprId) -> Option<&ExprNode> {
        self.nodes.get(id.index())
    }

    pub fn nodes(&self) -> &[ExprNode] {
        &self.nodes
    }

    pub fn evaluate(&self, root: Expr, params: &ParamValues) -> ExprResult<Complex64> {
        let mut values = vec![Complex64::ZERO; self.nodes.len()];
        for (index, node) in self.nodes.iter().enumerate() {
            values[index] = match *node {
                ExprNode::Const(value) => value,
                ExprNode::Param(id) => params.get(id)?.into(),
                ExprNode::Unary { op, input } => eval_unary(op, values[input.index()]),
                ExprNode::Binary { op, lhs, rhs } => {
                    eval_binary(op, values[lhs.index()], values[rhs.index()])
                }
            };
        }
        Ok(values[root.id.index()])
    }

    pub fn evaluate_with_gradient(
        &self,
        root: Expr,
        params: &ParamValues,
    ) -> ExprResult<(Complex64, Vec<Complex64>)> {
        let layout = params.layout();
        let mut values = vec![Complex64::ZERO; self.nodes.len()];
        let mut gradients = vec![vec![Complex64::ZERO; layout.n_free()]; self.nodes.len()];

        for (index, node) in self.nodes.iter().enumerate() {
            match *node {
                ExprNode::Const(value) => values[index] = value,
                ExprNode::Param(id) => {
                    values[index] = params.get(id)?.into();
                    if let Some(free_id) = layout.free_id(id)? {
                        gradients[index][free_id.index()] = Complex64::ONE;
                    }
                }
                ExprNode::Unary { op, input } => {
                    let input_index = input.index();
                    values[index] = eval_unary(op, values[input_index]);
                    let factor = unary_derivative(op, values[input_index], values[index]);
                    for free in 0..layout.n_free() {
                        gradients[index][free] = match op {
                            UnaryOp::Real => Complex64::new(gradients[input_index][free].re, 0.0),
                            UnaryOp::Imag => Complex64::new(gradients[input_index][free].im, 0.0),
                            UnaryOp::Conj => gradients[input_index][free].conj(),
                            UnaryOp::NormSqr => Complex64::new(
                                2.0 * (gradients[input_index][free] * values[input_index].conj())
                                    .re,
                                0.0,
                            ),
                            _ => gradients[input_index][free] * factor,
                        };
                    }
                }
                ExprNode::Binary { op, lhs, rhs } => {
                    let lhs_index = lhs.index();
                    let rhs_index = rhs.index();
                    let lhs_value = values[lhs_index];
                    let rhs_value = values[rhs_index];
                    values[index] = eval_binary(op, lhs_value, rhs_value);
                    for free in 0..layout.n_free() {
                        let dl = gradients[lhs_index][free];
                        let dr = gradients[rhs_index][free];
                        gradients[index][free] = match op {
                            BinaryOp::Add => dl + dr,
                            BinaryOp::Sub => dl - dr,
                            BinaryOp::Mul => dl * rhs_value + dr * lhs_value,
                            BinaryOp::Div => (dl * rhs_value - dr * lhs_value) / rhs_value.powi(2),
                        };
                    }
                }
            }
        }

        Ok((values[root.id.index()], gradients[root.id.index()].clone()))
    }

    fn unary(&mut self, op: UnaryOp, input: Expr) -> Expr {
        match op {
            UnaryOp::Neg => {
                if let Some(ExprNode::Const(value)) = self.node(input.id) {
                    return self.constant(-*value);
                }
            }
            UnaryOp::Real => {
                if let Some(ExprNode::Const(value)) = self.node(input.id) {
                    return self.constant(Complex64::new(value.re, 0.0));
                }
            }
            UnaryOp::Imag => {
                if let Some(ExprNode::Const(value)) = self.node(input.id) {
                    return self.constant(Complex64::new(value.im, 0.0));
                }
            }
            UnaryOp::Conj => {
                if let Some(ExprNode::Const(value)) = self.node(input.id) {
                    return self.constant(value.conj());
                }
            }
            UnaryOp::NormSqr => {
                if let Some(ExprNode::Const(value)) = self.node(input.id) {
                    return self.constant(Complex64::new(value.norm_sqr(), 0.0));
                }
            }
            _ => {}
        }
        self.push(ExprNode::Unary {
            op,
            input: input.id,
        })
    }

    fn binary(&mut self, op: BinaryOp, lhs: Expr, rhs: Expr) -> Expr {
        if let (Some(ExprNode::Const(l)), Some(ExprNode::Const(r))) =
            (self.node(lhs.id), self.node(rhs.id))
        {
            return self.constant(eval_binary(op, *l, *r));
        }

        match op {
            BinaryOp::Add if self.is_zero(lhs) => return rhs,
            BinaryOp::Add if self.is_zero(rhs) => return lhs,
            BinaryOp::Sub if self.is_zero(rhs) => return lhs,
            BinaryOp::Mul if self.is_one(lhs) => return rhs,
            BinaryOp::Mul if self.is_one(rhs) => return lhs,
            BinaryOp::Mul if self.is_zero(lhs) || self.is_zero(rhs) => {
                return self.constant(Complex64::ZERO);
            }
            BinaryOp::Div if self.is_one(rhs) => return lhs,
            _ => {}
        }

        self.push(ExprNode::Binary {
            op,
            lhs: lhs.id,
            rhs: rhs.id,
        })
    }

    fn push(&mut self, node: ExprNode) -> Expr {
        let id = ExprId(self.nodes.len() as u32);
        self.nodes.push(node);
        Expr { id }
    }

    fn is_zero(&self, expr: Expr) -> bool {
        matches!(self.node(expr.id), Some(ExprNode::Const(value)) if *value == Complex64::ZERO)
    }

    fn is_one(&self, expr: Expr) -> bool {
        matches!(self.node(expr.id), Some(ExprNode::Const(value)) if *value == Complex64::ONE)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Expr {
    id: ExprId,
}

impl Expr {
    pub fn id(self) -> ExprId {
        self.id
    }

    pub fn real(self, graph: &mut ExprGraph) -> Self {
        graph.unary(UnaryOp::Real, self)
    }

    pub fn imag(self, graph: &mut ExprGraph) -> Self {
        graph.unary(UnaryOp::Imag, self)
    }

    pub fn conj(self, graph: &mut ExprGraph) -> Self {
        graph.unary(UnaryOp::Conj, self)
    }

    pub fn norm_sqr(self, graph: &mut ExprGraph) -> Self {
        graph.unary(UnaryOp::NormSqr, self)
    }

    pub fn sqrt(self, graph: &mut ExprGraph) -> Self {
        graph.unary(UnaryOp::Sqrt, self)
    }

    pub fn exp(self, graph: &mut ExprGraph) -> Self {
        graph.unary(UnaryOp::Exp, self)
    }

    pub fn sin(self, graph: &mut ExprGraph) -> Self {
        graph.unary(UnaryOp::Sin, self)
    }

    pub fn cos(self, graph: &mut ExprGraph) -> Self {
        graph.unary(UnaryOp::Cos, self)
    }

    pub fn log(self, graph: &mut ExprGraph) -> Self {
        graph.unary(UnaryOp::Log, self)
    }

    pub fn powi(self, graph: &mut ExprGraph, power: i32) -> Self {
        graph.unary(UnaryOp::PowI(power), self)
    }
}

#[derive(Clone, Debug, Default)]
pub struct ExprBuilder {
    graph: ExprGraph,
    param_names: HashMap<Arc<str>, ParamId>,
}

impl ExprBuilder {
    pub fn new(layout: &ParamLayout) -> Self {
        let param_names = layout
            .specs()
            .iter()
            .filter_map(|spec| {
                layout
                    .id(spec.name())
                    .map(|id| (Arc::from(spec.name()), id))
            })
            .collect();
        Self {
            graph: ExprGraph::new(),
            param_names,
        }
    }

    pub fn param(&mut self, name: &str) -> ExprResult<Expr> {
        let id = self
            .param_names
            .get(name)
            .copied()
            .ok_or(ExprError::UnknownParameter(usize::MAX))?;
        Ok(self.graph.param(id))
    }

    pub fn real(&mut self, value: f64) -> Expr {
        self.graph.real_constant(value)
    }

    pub fn complex(&mut self, re: f64, im: f64) -> Expr {
        self.graph.constant(Complex64::new(re, im))
    }

    pub fn finish(self) -> ExprGraph {
        self.graph
    }

    pub fn graph_mut(&mut self) -> &mut ExprGraph {
        &mut self.graph
    }
}

impl Add for Expr {
    type Output = BinaryExpr;

    fn add(self, rhs: Self) -> Self::Output {
        BinaryExpr {
            op: BinaryOp::Add,
            lhs: self,
            rhs,
        }
    }
}

impl Sub for Expr {
    type Output = BinaryExpr;

    fn sub(self, rhs: Self) -> Self::Output {
        BinaryExpr {
            op: BinaryOp::Sub,
            lhs: self,
            rhs,
        }
    }
}

impl Mul for Expr {
    type Output = BinaryExpr;

    fn mul(self, rhs: Self) -> Self::Output {
        BinaryExpr {
            op: BinaryOp::Mul,
            lhs: self,
            rhs,
        }
    }
}

impl Div for Expr {
    type Output = BinaryExpr;

    fn div(self, rhs: Self) -> Self::Output {
        BinaryExpr {
            op: BinaryOp::Div,
            lhs: self,
            rhs,
        }
    }
}

impl Neg for Expr {
    type Output = UnaryExpr;

    fn neg(self) -> Self::Output {
        UnaryExpr {
            op: UnaryOp::Neg,
            input: self,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct UnaryExpr {
    op: UnaryOp,
    input: Expr,
}

impl UnaryExpr {
    pub fn build(self, graph: &mut ExprGraph) -> Expr {
        graph.unary(self.op, self.input)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct BinaryExpr {
    op: BinaryOp,
    lhs: Expr,
    rhs: Expr,
}

impl BinaryExpr {
    pub fn build(self, graph: &mut ExprGraph) -> Expr {
        graph.binary(self.op, self.lhs, self.rhs)
    }
}

fn eval_unary(op: UnaryOp, value: Complex64) -> Complex64 {
    match op {
        UnaryOp::Neg => -value,
        UnaryOp::Real => Complex64::new(value.re, 0.0),
        UnaryOp::Imag => Complex64::new(value.im, 0.0),
        UnaryOp::Conj => value.conj(),
        UnaryOp::NormSqr => Complex64::new(value.norm_sqr(), 0.0),
        UnaryOp::Sqrt => value.sqrt(),
        UnaryOp::Exp => value.exp(),
        UnaryOp::Sin => value.sin(),
        UnaryOp::Cos => value.cos(),
        UnaryOp::Log => value.ln(),
        UnaryOp::PowI(power) => value.powi(power),
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

fn unary_derivative(op: UnaryOp, input: Complex64, output: Complex64) -> Complex64 {
    match op {
        UnaryOp::Neg => -Complex64::ONE,
        UnaryOp::Real | UnaryOp::Imag | UnaryOp::Conj | UnaryOp::NormSqr => Complex64::ZERO,
        UnaryOp::Sqrt => Complex64::new(0.5, 0.0) / output,
        UnaryOp::Exp => output,
        UnaryOp::Sin => input.cos(),
        UnaryOp::Cos => -input.sin(),
        UnaryOp::Log => Complex64::ONE / input,
        UnaryOp::PowI(power) => match power {
            0 => Complex64::ZERO,
            1 => Complex64::ONE,
            _ => Complex64::new(power as f64, 0.0) * input.powi(power - 1),
        },
    }
}

#[cfg(test)]
mod tests {
    use laddu_params::{ParamLayout, param};

    use super::*;

    #[test]
    fn evaluates_expression_and_gradient() {
        let layout = ParamLayout::new([param("x").initial(2.0), param("y").initial(3.0)]).unwrap();
        let x_id = layout.id("x").unwrap();
        let y_id = layout.id("y").unwrap();
        let params = layout.expand_free_values(&[2.0, 3.0]).unwrap();

        let mut graph = ExprGraph::new();
        let x = graph.param(x_id);
        let y = graph.param(y_id);
        let xy = (x * y).build(&mut graph);
        let root = (xy + x).build(&mut graph);

        let (value, grad) = graph.evaluate_with_gradient(root, &params).unwrap();

        assert_eq!(value, Complex64::new(8.0, 0.0));
        assert_eq!(
            grad,
            vec![Complex64::new(4.0, 0.0), Complex64::new(2.0, 0.0)]
        );
    }

    #[test]
    fn differentiates_norm_sqr() {
        let layout = ParamLayout::new([param("x").initial(3.0), param("y").initial(4.0)]).unwrap();
        let params = layout.expand_free_values(&[3.0, 4.0]).unwrap();

        let mut graph = ExprGraph::new();
        let x = graph.param(layout.id("x").unwrap());
        let y = graph.param(layout.id("y").unwrap());
        let i = graph.constant(Complex64::new(0.0, 1.0));
        let iy = (i * y).build(&mut graph);
        let z = (x + iy).build(&mut graph);
        let root = z.norm_sqr(&mut graph);

        let (value, grad) = graph.evaluate_with_gradient(root, &params).unwrap();

        assert_eq!(value, Complex64::new(25.0, 0.0));
        assert_eq!(
            grad,
            vec![Complex64::new(6.0, 0.0), Complex64::new(8.0, 0.0)]
        );
    }

    #[test]
    fn folds_simple_constants() {
        let mut graph = ExprGraph::new();
        let two = graph.real_constant(2.0);
        let three = graph.real_constant(3.0);
        let root = (two + three).build(&mut graph);

        assert_eq!(
            graph.node(root.id()),
            Some(&ExprNode::Const(Complex64::new(5.0, 0.0)))
        );
    }
}
