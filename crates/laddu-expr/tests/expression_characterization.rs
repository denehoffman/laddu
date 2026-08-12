//! Characterization coverage for expression structure and reconstruction contracts.

use std::sync::Arc;

use laddu_expr::{
    BinaryOp, Expr, ExprDependencyKind, ExprGraph, ExprGraphRebuilder, ExprId, ExprMetadata,
    ExprNode, ExprNodeSemantics, ExprSourceKind, NumberClass, P4Component, ParamError, UnaryOp,
    ValueKind, complex, dot, event_scalar, matrix_from_flat, parameter,
    parameters::{ParamLayout, Parameter},
    vector,
};
use num::complex::Complex64;

fn id(index: usize) -> ExprId {
    ExprId::from_index(index)
}

#[test]
fn every_node_variant_reports_children_in_semantic_order() {
    let cases = [
        ("real constant", ExprNode::RealConst(1.0), vec![]),
        (
            "complex constant",
            ExprNode::ComplexConst(Complex64::new(1.0, 2.0)),
            vec![],
        ),
        (
            "scalar parameter",
            ExprNode::ScalarParam(Parameter::free("p")),
            vec![],
        ),
        (
            "event scalar",
            ExprNode::EventScalar(Arc::from("x")),
            vec![],
        ),
        (
            "event four-momentum component",
            ExprNode::EventP4Component {
                name: Arc::from("p4"),
                component: P4Component::E,
            },
            vec![],
        ),
        (
            "unary input",
            ExprNode::Unary {
                op: UnaryOp::Sin,
                input: id(3),
            },
            vec![id(3)],
        ),
        (
            "binary lhs then rhs",
            ExprNode::Binary {
                op: BinaryOp::Sub,
                lhs: id(3),
                rhs: id(1),
            },
            vec![id(3), id(1)],
        ),
        (
            "n-ary sum terms",
            ExprNode::NaryAdd {
                terms: vec![id(3), id(1), id(2)],
            },
            vec![id(3), id(1), id(2)],
        ),
        (
            "n-ary product factors",
            ExprNode::NaryMul {
                factors: vec![id(2), id(0), id(3)],
            },
            vec![id(2), id(0), id(3)],
        ),
        (
            "complex real then imaginary",
            ExprNode::Complex {
                re: id(3),
                im: id(1),
            },
            vec![id(3), id(1)],
        ),
        (
            "vector elements",
            ExprNode::Vector {
                elements: vec![id(3), id(1), id(2)],
            },
            vec![id(3), id(1), id(2)],
        ),
        (
            "row-major matrix elements",
            ExprNode::Matrix {
                rows: 1,
                cols: 3,
                elements: vec![id(3), id(1), id(2)],
            },
            vec![id(3), id(1), id(2)],
        ),
        (
            "vector component input",
            ExprNode::Component {
                input: id(3),
                index: 1,
            },
            vec![id(3)],
        ),
        (
            "matrix element input",
            ExprNode::MatrixElement {
                input: id(3),
                row: 1,
                col: 2,
            },
            vec![id(3)],
        ),
        (
            "matrix multiplication lhs then rhs",
            ExprNode::MatMul {
                lhs: id(3),
                rhs: id(1),
            },
            vec![id(3), id(1)],
        ),
        (
            "matrix then vector",
            ExprNode::MatVec {
                matrix: id(3),
                vector: id(1),
            },
            vec![id(3), id(1)],
        ),
        (
            "dot lhs then rhs",
            ExprNode::Dot {
                lhs: id(3),
                rhs: id(1),
            },
            vec![id(3), id(1)],
        ),
        (
            "solve matrix then rhs",
            ExprNode::Solve {
                matrix: id(3),
                rhs: id(1),
            },
            vec![id(3), id(1)],
        ),
    ];

    for (description, node, expected) in cases {
        assert_eq!(
            node.children().collect::<Vec<_>>(),
            expected,
            "{description}"
        );
        assert_eq!(node.child_ids(), expected, "compatibility: {description}");
        assert_eq!(node.map_children(|child| child), node, "{description}");

        let mut visited = Vec::new();
        let mapped = node.map_children(|child| {
            visited.push(child);
            id(child.index() + 10)
        });
        assert_eq!(visited, expected, "mapping order: {description}");
        assert_eq!(
            mapped.children().collect::<Vec<_>>(),
            expected
                .iter()
                .map(|child| id(child.index() + 10))
                .collect::<Vec<_>>(),
            "mapped children: {description}"
        );
    }
}

#[test]
fn graph_rebuilder_keeps_remaps_and_metadata_aligned() {
    let mut rebuild = ExprGraphRebuilder::with_capacity(3);
    let lhs = rebuild.emit(
        "lhs",
        ExprNode::RealConst(1.0),
        ExprMetadata::new(ExprSourceKind::Const),
    );
    let rhs = rebuild.emit(
        "rhs",
        ExprNode::EventScalar(Arc::from("x")),
        ExprMetadata::new(ExprSourceKind::Event),
    );
    let root = rebuild.emit(
        "root",
        ExprNode::Binary {
            op: BinaryOp::Add,
            lhs,
            rhs,
        },
        ExprMetadata::new(ExprSourceKind::Binary),
    );

    assert_eq!(rebuild.remapped(&"lhs"), Some(lhs));
    assert_eq!(rebuild.remapped(&"rhs"), Some(rhs));
    assert_eq!(rebuild.remapped(&"root"), Some(root));
    assert_eq!(rebuild.nodes().len(), 3);
    assert_eq!(rebuild.metadata().len(), rebuild.nodes().len());

    let graph = rebuild.finish(root).unwrap();
    assert_eq!(graph.root(), root);
    assert_eq!(
        graph.node(root).unwrap().children().collect::<Vec<_>>(),
        [lhs, rhs]
    );
    assert_eq!(
        [lhs, rhs, root].map(|id| graph.metadata(id).unwrap().source()),
        [
            ExprSourceKind::Const,
            ExprSourceKind::Event,
            ExprSourceKind::Binary,
        ]
    );
}

#[test]
fn graph_rebuilder_supports_aliases_and_anonymous_fragments() {
    let mut rebuild = ExprGraphRebuilder::with_capacity(2);
    let value = rebuild.emit_anonymous(
        ExprNode::RealConst(1.0),
        ExprMetadata::new(ExprSourceKind::Const),
    );
    rebuild.alias("source", value);

    assert_eq!(rebuild.remapped(&"source"), Some(value));
    let graph = rebuild.finish(value).unwrap();
    assert_eq!(graph.nodes(), [ExprNode::RealConst(1.0)]);
    assert_eq!(
        graph.metadata(value).unwrap().source(),
        ExprSourceKind::Const
    );
}

#[test]
#[should_panic(expected = "children must be emitted before their parent")]
fn graph_rebuilder_rejects_parent_before_child_emission() {
    let mut rebuild = ExprGraphRebuilder::with_capacity(1);
    rebuild.emit(
        "root",
        ExprNode::Unary {
            op: UnaryOp::Neg,
            input: id(0),
        },
        ExprMetadata::new(ExprSourceKind::Unary),
    );
}

#[test]
fn intrinsic_constant_and_source_facts_are_stable() {
    assert_eq!(
        ExprNode::RealConst(-0.0).const_value(),
        Some(Complex64::new(-0.0, 0.0))
    );
    assert!(ExprNode::is_zero(&ExprNode::ComplexConst(Complex64::ZERO)));
    assert!(ExprNode::is_one(&ExprNode::RealConst(1.0)));
    assert_eq!(
        ExprNode::from_folded_const(Complex64::new(2.0, 0.0)),
        ExprNode::RealConst(2.0)
    );
    assert_eq!(
        ExprNode::from_folded_const(Complex64::new(2.0, -0.0)),
        ExprNode::ComplexConst(Complex64::new(2.0, -0.0))
    );

    let scalar = Expr::from(parameter!("p"));
    let event = event_scalar("x");
    let matrix = matrix_from_flat(1, 2, [scalar.clone(), event.clone()]).unwrap();
    let expressions = [
        (Expr::from(1.0), ExprSourceKind::Const),
        (scalar.clone(), ExprSourceKind::Param),
        (event.clone(), ExprSourceKind::Event),
        (event.sin(), ExprSourceKind::Unary),
        (scalar.clone() + 1.0, ExprSourceKind::Binary),
        (
            complex(scalar.clone(), event.clone()),
            ExprSourceKind::Complex,
        ),
        (
            vector([scalar.clone(), event.clone()]),
            ExprSourceKind::Vector,
        ),
        (matrix, ExprSourceKind::Matrix),
        (
            dot(
                vector([scalar.clone(), event.clone()]),
                vector([scalar, event]),
            ),
            ExprSourceKind::Vector,
        ),
    ];

    for (expression, expected) in expressions {
        let graph = expression.to_graph();
        assert_eq!(graph.metadata(graph.root()).unwrap().source(), expected);
    }
}

#[test]
fn node_semantics_are_context_free_and_exhaustive() {
    let no_children: &[ExprNodeSemantics] = &[];
    let real = ExprNode::RealConst(1.0);
    assert_eq!(
        real.semantics(no_children),
        ExprNodeSemantics {
            value_kind: ValueKind::Real,
            number_class: NumberClass::Real,
        }
    );
    assert_eq!(real.dependency_kind(), ExprDependencyKind::Constant);

    let imaginary = ExprNode::ComplexConst(Complex64::new(0.0, 1.0));
    assert_eq!(
        imaginary.semantics(no_children),
        ExprNodeSemantics {
            value_kind: ValueKind::Complex,
            number_class: NumberClass::Imaginary,
        }
    );

    let real_projection = ExprNode::Unary {
        op: UnaryOp::NormSqr,
        input: id(0),
    };
    assert_eq!(
        real_projection.semantics(no_children),
        ExprNodeSemantics {
            value_kind: ValueKind::Real,
            number_class: NumberClass::Real,
        }
    );
    assert_eq!(
        real_projection.dependency_kind(),
        ExprDependencyKind::Children
    );

    let inherited = ExprNode::Unary {
        op: UnaryOp::Neg,
        input: id(0),
    };
    assert_eq!(
        inherited.semantics(&[ExprNodeSemantics {
            value_kind: ValueKind::Complex,
            number_class: NumberClass::Imaginary,
        }]),
        ExprNodeSemantics {
            value_kind: ValueKind::Complex,
            number_class: NumberClass::Imaginary,
        }
    );
    assert_eq!(
        ExprNode::ScalarParam(Parameter::free("p")).dependency_kind(),
        ExprDependencyKind::Parameter
    );
    assert_eq!(
        ExprNode::EventScalar(Arc::from("x")).dependency_kind(),
        ExprDependencyKind::Event
    );
}

#[test]
fn structural_identity_is_bit_exact_and_includes_parameter_semantics() {
    assert_ne!(
        ExprNode::RealConst(0.0).structural_key(),
        ExprNode::RealConst(-0.0).structural_key()
    );

    let nan_a = f64::from_bits(0x7ff8_0000_0000_0001);
    let nan_b = f64::from_bits(0x7ff8_0000_0000_0002);
    assert_eq!(
        ExprNode::RealConst(nan_a).structural_key(),
        ExprNode::RealConst(nan_a).structural_key()
    );
    assert_ne!(
        ExprNode::RealConst(nan_a).structural_key(),
        ExprNode::RealConst(nan_b).structural_key()
    );

    let base = Parameter::free("p");
    let decorated = base
        .clone()
        .with_bounds(-1.0, 1.0)
        .with_periodic()
        .with_scale(0.25)
        .with_unit("GeV")
        .with_latex("p")
        .with_description("fit parameter");
    assert_ne!(
        ExprNode::ScalarParam(base).structural_key(),
        ExprNode::ScalarParam(decorated).structural_key()
    );
}

#[test]
fn equation_and_latex_formatting_preserve_operator_grammar() {
    let a = Expr::from(parameter!("a"));
    let b = Expr::from(parameter!("b"));
    let c = Expr::from(parameter!("c"));
    let cases = [
        (
            a.clone() + b.clone() * c.clone(),
            "a + b * c",
            "a + b \\cdot c",
        ),
        (
            a.clone() * (b.clone() + c.clone()),
            "a * (b + c)",
            "a \\cdot \\left(b + c\\right)",
        ),
        (
            a.clone() - (b.clone() - c.clone()),
            "a - (b - c)",
            "a - \\left(b - c\\right)",
        ),
        ((a / b).powi(-2), "(a / b)^(-2)", "\\frac{a}{b}^{-2}"),
    ];

    for (expression, equation, latex) in cases {
        let graph = expression.to_graph();
        assert_eq!(graph.display_equation().to_string(), equation);
        assert_eq!(graph.display_latex().to_string(), latex);
    }
}

#[test]
fn parameter_validation_error_classification_is_exhaustive() {
    assert!(matches!(
        ParamLayout::new([Parameter::free("")]),
        Err(ParamError::EmptyName)
    ));
    assert!(matches!(
        ParamLayout::new([Parameter::free("x"), Parameter::fixed("x", 0.0)]),
        Err(ParamError::DuplicateName(name)) if name == "x"
    ));
    assert!(matches!(
        ParamLayout::new([Parameter::free("x").with_bounds(2.0, 1.0)]),
        Err(ParamError::InvalidBounds { .. })
    ));
    assert!(matches!(
        ParamLayout::new([Parameter::free("x").with_initial((2.0, 1.0))]),
        Err(ParamError::InvalidInitialRange { .. })
    ));
    assert!(matches!(
        ParamLayout::new([Parameter::free("x").with_bounds(0.0, 1.0).with_initial(2.0)]),
        Err(ParamError::InitialOutOfBounds { .. })
    ));
    assert!(matches!(
        ParamLayout::new([Parameter::free("x")
            .with_bounds(0.0, 1.0)
            .with_initial((0.5, 1.5))]),
        Err(ParamError::InitialRangeOutOfBounds { .. })
    ));
    assert!(matches!(
        ParamLayout::new([Parameter::fixed("x", 2.0).with_bounds(0.0, 1.0)]),
        Err(ParamError::FixedValueOutOfBounds { .. })
    ));
    assert!(matches!(
        ParamLayout::new([Parameter::free("x").with_periodic()]),
        Err(ParamError::PeriodicRequiresFiniteBounds { .. })
    ));
    assert!(matches!(
        ParamLayout::new([Parameter::free("x").with_scale(0.0)]),
        Err(ParamError::InvalidScale { .. })
    ));
    assert!(matches!(
        ParamLayout::new([Parameter::free("x")
            .with_bounds(0.0, 1.0)
            .with_periodic()
            .with_initial(1.0),]),
        Err(ParamError::ValueOutsidePeriodicDomain { .. })
    ));

    let layout = ParamLayout::new([Parameter::free("x").with_bounds(0.0, 1.0)]).unwrap();
    assert!(matches!(
        layout.validate_free_values(&[2.0]),
        Err(ParamError::ValueOutOfBounds { .. })
    ));
    assert!(matches!(
        layout.validate_free_values(&[]),
        Err(ParamError::FreeLengthMismatch {
            expected: 1,
            actual: 0
        })
    ));
}

#[test]
fn graph_reconstruction_preserves_all_non_nary_node_variants_and_metadata() {
    let nodes = vec![
        ExprNode::RealConst(1.0),
        ExprNode::ComplexConst(Complex64::new(2.0, 3.0)),
        ExprNode::ScalarParam(Parameter::free("p")),
        ExprNode::EventScalar(Arc::from("x")),
        ExprNode::EventP4Component {
            name: Arc::from("p4"),
            component: P4Component::Pz,
        },
        ExprNode::Unary {
            op: UnaryOp::Cos,
            input: id(3),
        },
        ExprNode::Binary {
            op: BinaryOp::Atan2,
            lhs: id(2),
            rhs: id(3),
        },
        ExprNode::Complex {
            re: id(0),
            im: id(2),
        },
        ExprNode::Vector {
            elements: vec![id(0), id(2)],
        },
        ExprNode::Matrix {
            rows: 2,
            cols: 2,
            elements: vec![id(0), id(2), id(3), id(4)],
        },
        ExprNode::Component {
            input: id(8),
            index: 1,
        },
        ExprNode::MatrixElement {
            input: id(9),
            row: 1,
            col: 0,
        },
        ExprNode::MatMul {
            lhs: id(9),
            rhs: id(9),
        },
        ExprNode::MatVec {
            matrix: id(9),
            vector: id(8),
        },
        ExprNode::Dot {
            lhs: id(8),
            rhs: id(8),
        },
        ExprNode::Solve {
            matrix: id(9),
            rhs: id(8),
        },
        ExprNode::Vector {
            elements: (0..16).map(id).collect(),
        },
    ];
    let metadata: Vec<_> = (0..nodes.len())
        .map(|index| {
            let source = if index < 2 {
                ExprSourceKind::Const
            } else {
                ExprSourceKind::LinearAlgebra
            };
            ExprMetadata::new(source)
        })
        .collect();
    let graph = ExprGraph::from_parts(id(nodes.len() - 1), nodes, metadata).unwrap();
    let rebuilt = Expr::from_graph(graph.clone()).unwrap().to_graph();

    assert_eq!(rebuilt.nodes(), graph.nodes());
    for index in 0..graph.nodes().len() {
        assert_eq!(rebuilt.metadata(id(index)), graph.metadata(id(index)));
    }
}

#[test]
fn nary_reconstruction_is_left_associative_and_deep_graphs_are_accepted() {
    let metadata = ExprMetadata::new(ExprSourceKind::Const);
    let nary = ExprGraph::from_parts(
        id(3),
        vec![
            ExprNode::RealConst(1.0),
            ExprNode::RealConst(2.0),
            ExprNode::RealConst(3.0),
            ExprNode::NaryAdd {
                terms: vec![id(0), id(1), id(2)],
            },
        ],
        vec![metadata.clone(); 4],
    )
    .unwrap();
    let rebuilt = Expr::from_graph(nary).unwrap().to_graph();
    assert_eq!(rebuilt.to_string(), "1 + 2 + 3");
    assert_eq!(rebuilt.nodes().len(), 5);
    assert!(
        rebuilt
            .nodes()
            .iter()
            .all(|node| !matches!(node, ExprNode::NaryAdd { .. }))
    );

    const DEPTH: usize = 1_024;
    let mut nodes = Vec::with_capacity(DEPTH + 1);
    nodes.push(ExprNode::RealConst(1.0));
    for index in 1..=DEPTH {
        nodes.push(ExprNode::Unary {
            op: UnaryOp::Neg,
            input: id(index - 1),
        });
    }
    let deep = ExprGraph::from_parts(
        id(DEPTH),
        nodes,
        vec![ExprMetadata::new(ExprSourceKind::Unary); DEPTH + 1],
    )
    .unwrap();
    let rebuilt =
        Expr::from_graph(deep).expect("deep topological reconstruction must be iterative");
    assert_eq!(rebuilt.to_graph().nodes().len(), DEPTH + 1);
}
