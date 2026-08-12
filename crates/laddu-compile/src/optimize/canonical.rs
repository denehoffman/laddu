use std::collections::HashMap;

use laddu_expr::{
    BinaryOp, ExprGraph, ExprId, ExprMetadata, ExprNode, ExprNodeStructuralKey, UnaryOp,
};

use crate::{CompileResult, graph_utils::compact_to_root};

use super::{CanonicalCsePass, OptimizationPass};

impl OptimizationPass for CanonicalCsePass {
    fn name(&self) -> &'static str {
        "canonical-cse"
    }

    fn run(&self, graph: ExprGraph) -> CompileResult<ExprGraph> {
        let mut old_to_new = Vec::with_capacity(graph.nodes().len());
        let mut nodes = Vec::with_capacity(graph.nodes().len());
        let mut metadata = Vec::with_capacity(graph.nodes().len());
        let mut keys = HashMap::with_capacity(graph.nodes().len());

        for (old_index, node) in graph.nodes().iter().enumerate() {
            let old_id = ExprId::from_index(old_index);
            let node_metadata = graph
                .metadata(old_id)
                .expect("graph metadata length is validated")
                .clone();
            let new_id = emit_canonical_node(
                node.map_children(|id| old_to_new[id.index()]),
                node_metadata,
                &mut nodes,
                &mut metadata,
                &mut keys,
            );
            old_to_new.push(new_id);
        }

        let root = old_to_new[graph.root().index()];
        let graph = ExprGraph::from_parts(root, nodes, metadata)?;
        compact_to_root(&graph, root)
    }
}

fn emit_canonical_node(
    node: ExprNode,
    node_metadata: ExprMetadata,
    nodes: &mut Vec<ExprNode>,
    metadata: &mut Vec<ExprMetadata>,
    keys: &mut HashMap<ExprNodeStructuralKey, ExprId>,
) -> ExprId {
    match node {
        ExprNode::Binary {
            op: op @ (BinaryOp::Add | BinaryOp::Mul),
            lhs,
            rhs,
        } => {
            let mut operands = Vec::new();
            collect_associative_operands(op, lhs, nodes, &mut operands);
            collect_associative_operands(op, rhs, nodes, &mut operands);
            emit_canonical_associative(op, operands, node_metadata, nodes, metadata, keys)
        }
        ExprNode::NaryAdd { terms } => {
            emit_canonical_associative(BinaryOp::Add, terms, node_metadata, nodes, metadata, keys)
        }
        ExprNode::NaryMul { factors } => {
            emit_canonical_associative(BinaryOp::Mul, factors, node_metadata, nodes, metadata, keys)
        }
        node => intern_canonical_node(node, node_metadata, nodes, metadata, keys),
    }
}

fn emit_canonical_associative(
    op: BinaryOp,
    operands: Vec<ExprId>,
    node_metadata: ExprMetadata,
    nodes: &mut Vec<ExprNode>,
    metadata: &mut Vec<ExprMetadata>,
    keys: &mut HashMap<ExprNodeStructuralKey, ExprId>,
) -> ExprId {
    let mut flattened = Vec::new();
    for operand in operands {
        collect_associative_operands(op, operand, nodes, &mut flattened);
    }
    flattened.sort_by(|lhs, rhs| {
        operand_sort_key(op, *lhs, nodes).cmp(&operand_sort_key(op, *rhs, nodes))
    });

    match flattened.as_slice() {
        [] => intern_canonical_node(
            identity_for_associative_op(op),
            node_metadata,
            nodes,
            metadata,
            keys,
        ),
        [operand] => *operand,
        _ => {
            let node = match op {
                BinaryOp::Add => ExprNode::NaryAdd { terms: flattened },
                BinaryOp::Mul => ExprNode::NaryMul { factors: flattened },
                BinaryOp::Sub | BinaryOp::Div | BinaryOp::Atan2 => {
                    unreachable!("only associative ops are canonicalized")
                }
            };
            intern_canonical_node(node, node_metadata, nodes, metadata, keys)
        }
    }
}

fn identity_for_associative_op(op: BinaryOp) -> ExprNode {
    match op {
        BinaryOp::Add => ExprNode::RealConst(0.0),
        BinaryOp::Mul => ExprNode::RealConst(1.0),
        BinaryOp::Sub | BinaryOp::Div | BinaryOp::Atan2 => {
            unreachable!("only associative ops have identities")
        }
    }
}

fn intern_canonical_node(
    node: ExprNode,
    node_metadata: ExprMetadata,
    nodes: &mut Vec<ExprNode>,
    metadata: &mut Vec<ExprMetadata>,
    keys: &mut HashMap<ExprNodeStructuralKey, ExprId>,
) -> ExprId {
    let canonical = canonicalize_node(node);
    let key = canonical.structural_key();
    if let Some(id) = keys.get(&key).copied() {
        return id;
    }

    let id = ExprId::from_index(nodes.len());
    keys.insert(key, id);
    nodes.push(canonical);
    metadata.push(node_metadata);
    id
}

fn collect_associative_operands(
    op: BinaryOp,
    id: ExprId,
    nodes: &[ExprNode],
    operands: &mut Vec<ExprId>,
) {
    match nodes.get(id.index()) {
        Some(ExprNode::Binary {
            op: child_op,
            lhs,
            rhs,
        }) if *child_op == op => {
            collect_associative_operands(op, *lhs, nodes, operands);
            collect_associative_operands(op, *rhs, nodes, operands);
        }
        Some(ExprNode::NaryAdd { terms }) if op == BinaryOp::Add => {
            for term in terms {
                collect_associative_operands(op, *term, nodes, operands);
            }
        }
        Some(ExprNode::NaryMul { factors }) if op == BinaryOp::Mul => {
            for factor in factors {
                collect_associative_operands(op, *factor, nodes, operands);
            }
        }
        _ => operands.push(id),
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct OperandSortKey {
    category: u8,
    structural: ExprNodeStructuralKey,
    id: usize,
}

fn operand_sort_key(op: BinaryOp, id: ExprId, nodes: &[ExprNode]) -> OperandSortKey {
    let node = nodes
        .get(id.index())
        .expect("associative operands are emitted nodes");
    let is_negative_add_operand = || match node {
        ExprNode::RealConst(value) => *value < 0.0,
        ExprNode::NaryMul { factors } => factors.iter().any(|factor| {
            matches!(
                nodes.get(factor.index()),
                Some(ExprNode::RealConst(value)) if *value < 0.0
            )
        }),
        _ => false,
    };
    let category = match (op, node) {
        (BinaryOp::Add, ExprNode::RealConst(value)) if *value >= 0.0 => 0,
        (BinaryOp::Add, _) if !is_negative_add_operand() => 1,
        (BinaryOp::Add, _) => 2,
        (
            BinaryOp::Mul,
            ExprNode::Unary {
                op: UnaryOp::Exp, ..
            },
        ) => 0,
        _ => 1,
    };
    OperandSortKey {
        category,
        structural: node.structural_key(),
        id: id.index(),
    }
}

fn canonicalize_node(node: ExprNode) -> ExprNode {
    match node {
        ExprNode::Binary {
            op: op @ (BinaryOp::Add | BinaryOp::Mul),
            lhs,
            rhs,
        } if rhs.index() < lhs.index() => ExprNode::Binary {
            op,
            lhs: rhs,
            rhs: lhs,
        },
        node => node,
    }
}
