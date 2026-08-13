use laddu_expr::{
    BinaryOp, Expr, ExprId, ExprMetadata, UnaryOp, complex, dot, event_scalar, matmul, matrix,
    matvec, parameter, parameters::Parameter, polar_complex, vector,
};
use num::complex::Complex64;

use super::*;

#[derive(Copy, Clone, Debug)]
struct WrapRootInExp;

impl OptimizationPass for WrapRootInExp {
    fn name(&self) -> &'static str {
        "wrap-root-in-exp"
    }

    fn run(&self, graph: ExprGraph) -> CompileResult<ExprGraph> {
        let mut nodes = graph.nodes().to_vec();
        let mut metadata = graph
            .nodes()
            .iter()
            .enumerate()
            .map(|(index, _)| {
                graph
                    .metadata(ExprId::from_index(index))
                    .expect("graph metadata length is validated")
                    .clone()
            })
            .collect::<Vec<_>>();
        let root = ExprId::from_index(nodes.len());
        nodes.push(ExprNode::Unary {
            op: UnaryOp::Exp,
            input: graph.root(),
        });
        metadata.push(ExprMetadata::new(laddu_expr::ExprSourceKind::Unary));
        Ok(ExprGraph::from_parts(root, nodes, metadata)?)
    }
}

fn count_binary_op(compiled: &CompiledModel, op: BinaryOp) -> usize {
    compiled
        .graph()
        .nodes()
        .iter()
        .filter(|node| matches!(node, ExprNode::Binary { op: node_op, .. } if *node_op == op))
        .count()
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
