use super::*;

#[test]
fn vector_and_matrix_extraction_alias_selected_scalar() {
    let x = Expr::from(parameter!("x"));
    let y = Expr::from(parameter!("y"));
    let component = CompiledModel::from_expr(&vector([x.clone(), y.clone()]).component(1)).unwrap();
    let element = CompiledModel::from_expr(
        &matrix([[x, y.clone()], [3.0.into(), 4.0.into()]]).matrix_element(0, 1),
    )
    .unwrap();

    for compiled in [component, element] {
        assert_eq!(compiled.graph().nodes().len(), 1);
        assert!(matches!(
            compiled.graph().node(ExprId::from_index(0)),
            Some(ExprNode::ScalarParam(parameter)) if parameter.name() == "y"
        ));
    }
}

#[test]
fn matrix_vector_identities_and_zeroes_simplify() {
    let x = event_scalar("x");
    let y = event_scalar("y");
    let identity = matrix([[1.0, 0.0], [0.0, 1.0]]);
    let zero_matrix = matrix([[0.0, 0.0], [0.0, 0.0]]);
    let vector = vector([x, y]);

    let identity_product = CompiledModel::from_expr(&matvec(identity, vector.clone())).unwrap();
    assert!(matches!(
        identity_product.graph().node(identity_product.graph().root()),
        Some(ExprNode::Vector { elements }) if elements.len() == 2
    ));

    let zero_product = CompiledModel::from_expr(&matvec(zero_matrix, vector)).unwrap();
    assert!(matches!(
        zero_product.graph().node(zero_product.graph().root()),
        Some(ExprNode::Vector { elements }) if elements.len() == 2
            && elements.iter().all(|id| matches!(zero_product.graph().node(*id), Some(ExprNode::RealConst(0.0))))
    ));
}

#[test]
fn dot_and_matvec_lower_to_scalar_arithmetic_when_cheaper() {
    let x = event_scalar("x");
    let y = event_scalar("y");
    let dot_product =
        CompiledModel::from_expr(&dot(vector([x.clone(), y.clone()]), vector([2.0, 3.0]))).unwrap();
    assert_eq!(
        dot_product
            .graph()
            .nodes()
            .iter()
            .filter(|node| matches!(node, ExprNode::Dot { .. }))
            .count(),
        0
    );
    assert!(matches!(
        dot_product.graph().node(dot_product.graph().root()),
        Some(ExprNode::NaryAdd { terms }) if terms.len() == 2
    ));

    let matrix_product =
        CompiledModel::from_expr(&matvec(matrix([[1.0, 2.0], [3.0, 4.0]]), vector([x, y])))
            .unwrap();
    assert_eq!(
        matrix_product
            .graph()
            .nodes()
            .iter()
            .filter(|node| matches!(node, ExprNode::MatVec { .. }))
            .count(),
        0
    );
    assert!(matches!(
        matrix_product.graph().node(matrix_product.graph().root()),
        Some(ExprNode::Vector { elements }) if elements.len() == 2
    ));
}

#[test]
fn selected_aggregate_outputs_only_lower_required_contractions() {
    const N: usize = 8;
    let matrix_values = matrix::<N, N, Expr>(std::array::from_fn(|row| {
        std::array::from_fn(|col| event_scalar(format!("m{row}_{col}")))
    }));
    let vector_values = vector(std::array::from_fn::<Expr, N, _>(|index| {
        event_scalar(format!("v{index}"))
    }));
    let selected_row =
        CompiledModel::from_expr(&matvec(matrix_values, vector_values).component(3)).unwrap();

    assert!(matches!(
        selected_row.graph().node(selected_row.graph().root()),
        Some(ExprNode::NaryAdd { terms }) if terms.len() == N
    ));
    assert!(
        !selected_row
            .graph()
            .nodes()
            .iter()
            .any(|node| matches!(node, ExprNode::MatVec { .. } | ExprNode::Component { .. }))
    );
    let selected_names = selected_row
        .graph()
        .nodes()
        .iter()
        .filter_map(|node| match node {
            ExprNode::EventScalar(name) => Some(name.as_ref()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(selected_names.len(), 2 * N);
    assert!(
        selected_names
            .iter()
            .all(|name| { name.starts_with("v") || name.starts_with("m3_") })
    );

    let lhs = matrix::<N, N, Expr>(std::array::from_fn(|row| {
        std::array::from_fn(|col| event_scalar(format!("a{row}_{col}")))
    }));
    let rhs = matrix::<N, N, Expr>(std::array::from_fn(|row| {
        std::array::from_fn(|col| event_scalar(format!("b{row}_{col}")))
    }));
    let selected_element =
        CompiledModel::from_expr(&matmul(lhs, rhs).matrix_element(2, 5)).unwrap();

    assert!(matches!(
        selected_element.graph().node(selected_element.graph().root()),
        Some(ExprNode::NaryAdd { terms }) if terms.len() == N
    ));
    assert!(
        !selected_element.graph().nodes().iter().any(|node| matches!(
            node,
            ExprNode::MatMul { .. } | ExprNode::MatrixElement { .. }
        ))
    );
    let selected_names = selected_element
        .graph()
        .nodes()
        .iter()
        .filter_map(|node| match node {
            ExprNode::EventScalar(name) => Some(name.as_ref()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(selected_names.len(), 2 * N);
    assert!(
        selected_names
            .iter()
            .all(|name| { name.starts_with("a2_") || name.ends_with("_5") })
    );
}

#[test]
fn matrix_multiplication_identity_and_zero_simplify() {
    let x = event_scalar("x");
    let matrix_value = matrix([[x, 2.0.into()], [3.0.into(), 4.0.into()]]);
    let identity = matrix([[1.0, 0.0], [0.0, 1.0]]);
    let identity_product = CompiledModel::from_expr(&matmul(identity, matrix_value)).unwrap();
    assert!(matches!(
        identity_product
            .graph()
            .node(identity_product.graph().root()),
        Some(ExprNode::Matrix {
            rows: 2,
            cols: 2,
            ..
        })
    ));

    let zero_product = CompiledModel::from_expr(&matmul(
        matrix([[0.0, 0.0], [0.0, 0.0]]),
        matrix([[1.0, 2.0], [3.0, 4.0]]),
    ))
    .unwrap();
    assert!(matches!(
        zero_product.graph().node(zero_product.graph().root()),
        Some(ExprNode::Matrix { rows: 2, cols: 2, elements }) if elements
            .iter()
            .all(|id| matches!(zero_product.graph().node(*id), Some(ExprNode::RealConst(0.0))))
    ));
}
