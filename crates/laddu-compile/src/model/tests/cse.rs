use super::*;

#[test]
fn cse_merges_duplicate_subtrees() {
    let x = Expr::from(parameter!("x"));
    let y = Expr::from(parameter!("y"));
    let sum = x + y;
    let model = sum.clone() * sum;
    let compiled = CompiledModel::from_expr(&model).unwrap();

    assert_eq!(count_nary_add(&compiled), 1);
}

#[test]
fn cse_canonicalizes_commutative_binary_operands() {
    let x = Expr::from(parameter!("x"));
    let y = Expr::from(parameter!("y"));
    let model = (x.clone() + y.clone()) * (y + x);
    let compiled = CompiledModel::from_expr(&model).unwrap();

    assert_eq!(count_nary_add(&compiled), 1);
    assert!(matches!(
        compiled.graph().node(compiled.graph().root()),
        Some(
            ExprNode::Unary {
                op: UnaryOp::PowI(2),
                ..
            } | ExprNode::NaryMul { .. }
        )
    ));
}

#[test]
fn cse_canonicalizes_associative_addition_trees() {
    let x = Expr::from(parameter!("x"));
    let y = Expr::from(parameter!("y"));
    let z = Expr::from(parameter!("z"));
    let lhs = (x.clone() + y.clone()) + z.clone();
    let rhs = x + (z + y);
    let compiled = CompiledModel::from_expr(&(lhs * rhs)).unwrap();

    assert!(matches!(
        compiled.graph().node(compiled.graph().root()),
        Some(
            ExprNode::Unary {
                op: UnaryOp::PowI(2),
                ..
            } | ExprNode::NaryMul { .. }
        )
    ));
    assert_eq!(count_nary_add(&compiled), 1);
}

#[test]
fn cse_canonicalizes_associative_multiplication_trees() {
    let x = Expr::from(parameter!("x"));
    let y = Expr::from(parameter!("y"));
    let z = Expr::from(parameter!("z"));
    let lhs = (x.clone() * y.clone()) * z.clone();
    let rhs = z * (y * x);
    let compiled = CompiledModel::from_expr(&(lhs + rhs)).unwrap();
    assert!(compiled.cost().weighted_ops() <= 6);
}

#[test]
fn cse_ignores_metadata_when_merging_duplicate_subtrees() {
    let x = Expr::from(parameter!("x"));
    let y = Expr::from(parameter!("y"));
    let lhs = (x.clone() + y.clone()).named("lhs");
    let rhs = (x + y).tagged("rhs");
    let compiled = CompiledModel::from_expr(&(lhs * rhs)).unwrap();

    assert_eq!(count_nary_add(&compiled), 1);
}

#[test]
fn rewritten_subexpression_keeps_source_annotation() {
    let x = Expr::from(parameter!("x"));
    let marked = (x + 0.0).named("inner").tagged("retain");
    let compiled = CompiledModel::from_expr(&marked.sin()).unwrap();
    let parameter = compiled
        .graph()
        .nodes()
        .iter()
        .position(|node| matches!(node, ExprNode::ScalarParam(_)))
        .unwrap();
    let metadata = compiled
        .graph()
        .metadata(laddu_expr::ExprId::from_index(parameter))
        .unwrap();
    assert_eq!(metadata.name(), Some("inner"));
    assert!(metadata.has_tag("retain"));
}
