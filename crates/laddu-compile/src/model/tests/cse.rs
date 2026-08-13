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
        Some(ExprNode::Unary {
            op: UnaryOp::PowI(2),
            input,
        }) if matches!(compiled.graph().node(*input), Some(ExprNode::NaryAdd { .. }))
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
        Some(ExprNode::Unary {
            op: UnaryOp::PowI(2),
            input,
        }) if matches!(compiled.graph().node(*input), Some(ExprNode::NaryAdd { .. }))
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
    let options =
        CompileOptions::with_pipeline(OptimizationPipeline::new().with_pass(CanonicalCsePass));
    let compiled = CompiledModel::from_expr_with_options(&(lhs + rhs), &options).unwrap();

    assert!(matches!(
        compiled.graph().node(compiled.graph().root()),
        Some(ExprNode::NaryAdd { terms }) if terms.len() == 2 && terms[0] == terms[1]
    ));
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
