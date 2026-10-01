use laddu_expr::{
    BinaryOp, Expr, UnaryOp, complex, dot, event_scalar, matmul, matrix, matvec, parameter,
    parameters::Parameter, polar_complex, vector,
};
use num::complex::Complex64;

use super::*;

fn exact_options() -> CompileOptions {
    CompileOptions::default().with_optimization_budget(crate::OptimizationBudget {
        solver_seconds: 5.0,
        ..Default::default()
    })
}

fn count_nary_add(compiled: &CompiledModel) -> usize {
    compiled
        .graph()
        .nodes()
        .iter()
        .filter(|node| matches!(node, ExprNode::NaryAdd { .. }))
        .count()
}

fn count_nary_mul(compiled: &CompiledModel) -> usize {
    compiled
        .graph()
        .nodes()
        .iter()
        .filter(|node| matches!(node, ExprNode::NaryMul { .. }))
        .count()
}

fn count_unary_op(compiled: &CompiledModel, op: UnaryOp) -> usize {
    compiled
        .graph()
        .nodes()
        .iter()
        .filter(|node| matches!(node, ExprNode::Unary { op: node_op, .. } if *node_op == op))
        .count()
}

fn has_real_const(compiled: &CompiledModel, expected: f64) -> bool {
    compiled.graph().nodes().iter().any(|node| {
            matches!(node, ExprNode::RealConst(value) if (*value - expected).abs() <= f64::EPSILON * expected.abs().max(1.0) * 16.0)
        })
}

#[path = "tests/algebra.rs"]
mod algebra;
#[path = "tests/cse.rs"]
mod cse;
#[path = "tests/matrix.rs"]
mod matrix;
#[path = "tests/pipeline.rs"]
mod pipeline;
#[path = "tests/private.rs"]
mod private;
#[path = "tests/trig.rs"]
mod trig;
