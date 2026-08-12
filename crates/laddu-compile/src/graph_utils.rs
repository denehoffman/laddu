use laddu_expr::{ExprGraph, ExprGraphRebuilder, ExprId};

use crate::CompileResult;

pub(crate) fn mark_reachable(
    graph: &ExprGraph,
    roots: impl IntoIterator<Item = ExprId>,
    required: &mut [bool],
) {
    debug_assert_eq!(required.len(), graph.nodes().len());
    for id in graph.reachable_post_order(roots) {
        required[id.index()] = true;
    }
}

pub(crate) fn compact_to_root(graph: &ExprGraph, root: ExprId) -> CompileResult<ExprGraph> {
    let order = graph.reachable_post_order([root]);
    let mut rebuild = ExprGraphRebuilder::with_capacity(order.len());
    for old_id in order {
        let node = graph
            .node(old_id)
            .expect("validated graph traversal only returns valid nodes")
            .map_children(|child| {
                rebuild
                    .remapped(&child)
                    .expect("post-order traversal emits children before parents")
            });
        let metadata = graph
            .metadata(old_id)
            .expect("validated graph metadata is aligned with its nodes")
            .clone();
        rebuild.emit(old_id, node, metadata);
    }
    let root = rebuild
        .remapped(&root)
        .expect("the compacted graph includes its requested root");
    Ok(rebuild.finish(root)?)
}

#[cfg(test)]
mod tests {
    use laddu_expr::{BinaryOp, ExprMetadata, ExprNode, ExprSourceKind, UnaryOp};

    use super::*;

    fn id(index: usize) -> ExprId {
        ExprId::from_index(index)
    }

    #[test]
    fn reachability_handles_empty_and_multiple_root_sets() {
        let graph = ExprGraph::from_parts(
            id(3),
            vec![
                ExprNode::RealConst(1.0),
                ExprNode::RealConst(2.0),
                ExprNode::Unary {
                    op: UnaryOp::Neg,
                    input: id(0),
                },
                ExprNode::Binary {
                    op: BinaryOp::Add,
                    lhs: id(2),
                    rhs: id(1),
                },
            ],
            vec![ExprMetadata::new(ExprSourceKind::Const); 4],
        )
        .unwrap();
        let mut required = vec![false; graph.nodes().len()];
        mark_reachable(&graph, [], &mut required);
        assert_eq!(required, [false; 4]);

        mark_reachable(&graph, [id(1), id(2)], &mut required);
        assert_eq!(required, [true, true, true, false]);
    }

    #[test]
    fn compaction_preserves_shared_children_order_and_metadata() {
        let metadata = [
            ExprMetadata::new(ExprSourceKind::Const),
            ExprMetadata::new(ExprSourceKind::Event),
            ExprMetadata::new(ExprSourceKind::Unary),
            ExprMetadata::new(ExprSourceKind::Binary),
        ];
        let graph = ExprGraph::from_parts(
            id(3),
            vec![
                ExprNode::RealConst(99.0),
                ExprNode::RealConst(2.0),
                ExprNode::Unary {
                    op: UnaryOp::Neg,
                    input: id(1),
                },
                ExprNode::Binary {
                    op: BinaryOp::Add,
                    lhs: id(2),
                    rhs: id(1),
                },
            ],
            metadata.to_vec(),
        )
        .unwrap();

        let compacted = compact_to_root(&graph, graph.root()).unwrap();
        assert_eq!(compacted.nodes().len(), 3);
        assert_eq!(
            compacted
                .node(compacted.root())
                .unwrap()
                .children()
                .collect::<Vec<_>>(),
            [id(1), id(0)]
        );
        assert_eq!(
            (0..3)
                .map(|index| compacted.metadata(id(index)).unwrap().source())
                .collect::<Vec<_>>(),
            [
                ExprSourceKind::Event,
                ExprSourceKind::Unary,
                ExprSourceKind::Binary,
            ]
        );
    }

    #[test]
    fn compaction_is_iterative_for_deep_graphs() {
        const DEPTH: usize = 2_048;
        let mut nodes = Vec::with_capacity(DEPTH + 2);
        nodes.push(ExprNode::RealConst(0.0));
        nodes.push(ExprNode::RealConst(1.0));
        for index in 2..=DEPTH + 1 {
            nodes.push(ExprNode::Unary {
                op: UnaryOp::Neg,
                input: id(index - 1),
            });
        }
        let graph = ExprGraph::from_parts(
            id(DEPTH + 1),
            nodes,
            vec![ExprMetadata::new(ExprSourceKind::Unary); DEPTH + 2],
        )
        .unwrap();

        let compacted = compact_to_root(&graph, graph.root()).unwrap();
        assert_eq!(compacted.nodes().len(), DEPTH + 1);
    }
}
