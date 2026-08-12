//! Stable serialization and public-enum contract fixtures.

use laddu_expr::{
    BinaryOp, ExprNode, UnaryOp,
    parameters::{ParamLayout, Parameter},
};
use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;

fn assert_stable_fixture<T>(fixture: &str, value: &T)
where
    T: DeserializeOwned + Serialize,
{
    let expected: Value = serde_json::from_str(fixture).expect("fixture must be valid JSON");
    let actual = serde_json::to_value(value).expect("value must serialize");
    assert_eq!(actual, expected);

    let decoded: T = serde_json::from_value(expected.clone()).expect("fixture must deserialize");
    assert_eq!(
        serde_json::to_value(decoded).expect("decoded fixture must serialize"),
        expected
    );
}

#[test]
fn expression_graph_serde_contract_is_stable() {
    let graph = ((laddu_expr::parameter!("mass", initial: 1.25) + 2.0).named("offset")
        * laddu_expr::event_scalar("weight").tagged("data"))
    .tagged("model")
    .to_graph();

    assert_stable_fixture(include_str!("fixtures/expr_graph.json"), &graph);
}

#[test]
fn parameter_layout_and_values_serde_contracts_are_stable() {
    let layout = ParamLayout::new([
        Parameter::free("mass")
            .with_initial(1.25)
            .with_bounds(0.0, 2.0)
            .with_scale(0.1)
            .with_unit("GeV")
            .with_latex("m"),
        Parameter::fixed("offset", -0.5).with_description("fixed background offset"),
    ])
    .expect("fixture parameters must be valid");
    let values = layout.values(&[1.5]).expect("fixture value must be valid");

    assert_stable_fixture(include_str!("fixtures/param_layout.json"), &layout);
    assert_stable_fixture(include_str!("fixtures/param_values.json"), &values);
}

#[test]
fn public_expression_variant_inventory_is_exhaustive() {
    fn unary_variant(op: UnaryOp) -> &'static str {
        match op {
            UnaryOp::Neg => "Neg",
            UnaryOp::Real => "Real",
            UnaryOp::Imag => "Imag",
            UnaryOp::Conj => "Conj",
            UnaryOp::NormSqr => "NormSqr",
            UnaryOp::Sqrt => "Sqrt",
            UnaryOp::Exp => "Exp",
            UnaryOp::Sin => "Sin",
            UnaryOp::Cos => "Cos",
            UnaryOp::Log => "Log",
            UnaryOp::PowI(_) => "PowI",
        }
    }

    fn binary_variant(op: BinaryOp) -> &'static str {
        match op {
            BinaryOp::Add => "Add",
            BinaryOp::Sub => "Sub",
            BinaryOp::Mul => "Mul",
            BinaryOp::Div => "Div",
            BinaryOp::Atan2 => "Atan2",
        }
    }

    fn node_variant(node: &ExprNode) -> &'static str {
        match node {
            ExprNode::RealConst(_) => "RealConst",
            ExprNode::ComplexConst(_) => "ComplexConst",
            ExprNode::ScalarParam(_) => "ScalarParam",
            ExprNode::EventScalar(_) => "EventScalar",
            ExprNode::EventP4Component { .. } => "EventP4Component",
            ExprNode::Unary { .. } => "Unary",
            ExprNode::Binary { .. } => "Binary",
            ExprNode::NaryAdd { .. } => "NaryAdd",
            ExprNode::NaryMul { .. } => "NaryMul",
            ExprNode::Complex { .. } => "Complex",
            ExprNode::Vector { .. } => "Vector",
            ExprNode::Matrix { .. } => "Matrix",
            ExprNode::Component { .. } => "Component",
            ExprNode::MatrixElement { .. } => "MatrixElement",
            ExprNode::MatMul { .. } => "MatMul",
            ExprNode::MatVec { .. } => "MatVec",
            ExprNode::Dot { .. } => "Dot",
            ExprNode::Solve { .. } => "Solve",
        }
    }

    assert_eq!(unary_variant(UnaryOp::PowI(2)), "PowI");
    assert_eq!(binary_variant(BinaryOp::Atan2), "Atan2");
    assert_eq!(node_variant(&ExprNode::RealConst(0.0)), "RealConst");
}
