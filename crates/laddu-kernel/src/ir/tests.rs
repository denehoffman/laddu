use super::*;
use laddu_expr::parameters::{ParamLayout, Parameter};

fn id(index: usize) -> KernelValueId {
    KernelValueId::from_index(index)
}

fn value(
    kind: KernelValueKind,
    class: KernelValueClass,
    instruction: KernelInstruction,
) -> KernelValue {
    KernelValue {
        kind,
        class,
        instruction,
    }
}

fn inference_inputs() -> Vec<KernelValue> {
    vec![
        value(
            KernelValueKind::Real,
            KernelValueClass::Invariant,
            KernelInstruction::RealConstant(1.0),
        ),
        value(
            KernelValueKind::Complex,
            KernelValueClass::Invariant,
            KernelInstruction::ComplexConstant(Complex64::new(1.0, 1.0)),
        ),
        value(
            KernelValueKind::Real,
            KernelValueClass::Event,
            KernelInstruction::Cached(0),
        ),
        value(
            KernelValueKind::Vector { len: 2 },
            KernelValueClass::Invariant,
            KernelInstruction::Vector(vec![id(0), id(1)]),
        ),
        value(
            KernelValueKind::Matrix { rows: 2, cols: 2 },
            KernelValueClass::Invariant,
            KernelInstruction::Matrix {
                rows: 2,
                cols: 2,
                elements: vec![id(0), id(1), id(1), id(0)],
            },
        ),
        value(
            KernelValueKind::Vector { len: 2 },
            KernelValueClass::Event,
            KernelInstruction::Vector(vec![id(2), id(1)]),
        ),
        value(
            KernelValueKind::Matrix { rows: 2, cols: 2 },
            KernelValueClass::Event,
            KernelInstruction::Matrix {
                rows: 2,
                cols: 2,
                elements: vec![id(2), id(1), id(1), id(0)],
            },
        ),
    ]
}

fn assert_inference(
    instruction: KernelInstruction,
    expected_kind: Option<KernelValueKind>,
    expected_class: KernelValueClass,
) {
    let values = inference_inputs();
    assert_eq!(
        instruction.expected_kind(&values, values.len()).unwrap(),
        expected_kind
    );
    assert_eq!(instruction.expected_class(&values), expected_class);
}

fn assert_shape_error(instruction: KernelInstruction, operation: &'static str) {
    let values = inference_inputs();
    assert!(matches!(
        instruction.expected_kind(&values, values.len()),
        Err(KernelIrError::InvalidShape {
            value: 7,
            operation: actual,
            ..
        }) if actual == operation
    ));
}

#[test]
fn instruction_metadata_covers_intrinsic_and_operand_event_dependence() {
    assert_eq!(
        KernelInstruction::RealConstant(1.0).event_dependence(),
        KernelEventDependence::Invariant
    );
    assert_eq!(
        KernelInstruction::Cached(0).event_dependence(),
        KernelEventDependence::Event
    );
    assert_eq!(
        KernelInstruction::Add(vec![id(0)]).event_dependence(),
        KernelEventDependence::Operands
    );
    assert_eq!(
        KernelInstruction::SolveRowAdjointElement {
            row_slot: 0,
            index: 0,
            len: 1,
            adjoint: id(0),
        }
        .diagnostic_name(),
        "SolveRowAdjointElement"
    );
}

#[test]
fn matrix_shape_arithmetic_is_checked() {
    assert_eq!(checked_matrix_width(3, 4), Some(12));
    assert_eq!(checked_matrix_width(usize::MAX, 2), None);
    assert_eq!(checked_row_major_index(3, 4, 2, 3), Some(11));
    assert_eq!(checked_row_major_index(3, 4, 3, 0), None);
    assert_eq!(checked_row_major_index(usize::MAX, usize::MAX, 0, 0), None);
    assert_eq!(
        checked_row_major_index(usize::MAX, usize::MAX, usize::MAX - 1, 0),
        None
    );

    let error = ScalarKernelIr::new(
        vec![value(
            KernelValueKind::Matrix {
                rows: usize::MAX,
                cols: 2,
            },
            KernelValueClass::Event,
            KernelInstruction::Cached(0),
        )],
        id(0),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        KernelIrError::InvalidShape {
            value: 0,
            operation: "matrix shape",
            ..
        }
    ));

    assert_shape_error(
        KernelInstruction::Matrix {
            rows: usize::MAX,
            cols: 2,
            elements: Vec::new(),
        },
        "matrix construction",
    );
}

#[test]
fn operand_discovery_is_complete_and_ordered() {
    let parameter = ParamLayout::new([Parameter::free("p")])
        .unwrap()
        .id("p")
        .unwrap();
    let cases = [
        (KernelInstruction::Cached(0), vec![]),
        (KernelInstruction::RealConstant(1.0), vec![]),
        (
            KernelInstruction::ComplexConstant(Complex64::new(1.0, 2.0)),
            vec![],
        ),
        (KernelInstruction::Parameter(parameter), vec![]),
        (
            KernelInstruction::Unary {
                op: UnaryOp::Neg,
                input: id(0),
            },
            vec![id(0)],
        ),
        (
            KernelInstruction::Binary {
                op: BinaryOp::Add,
                lhs: id(0),
                rhs: id(1),
            },
            vec![id(0), id(1)],
        ),
        (
            KernelInstruction::Add(vec![id(2), id(0)]),
            vec![id(2), id(0)],
        ),
        (
            KernelInstruction::Mul(vec![id(1), id(2)]),
            vec![id(1), id(2)],
        ),
        (
            KernelInstruction::Complex {
                re: id(0),
                im: id(1),
            },
            vec![id(0), id(1)],
        ),
        (
            KernelInstruction::Vector(vec![id(1), id(0)]),
            vec![id(1), id(0)],
        ),
        (
            KernelInstruction::Matrix {
                rows: 1,
                cols: 2,
                elements: vec![id(0), id(1)],
            },
            vec![id(0), id(1)],
        ),
        (
            KernelInstruction::Component {
                input: id(3),
                index: 0,
            },
            vec![id(3)],
        ),
        (
            KernelInstruction::MatrixElement {
                input: id(4),
                row: 0,
                col: 1,
            },
            vec![id(4)],
        ),
        (
            KernelInstruction::MatMul {
                lhs: id(4),
                rhs: id(6),
            },
            vec![id(4), id(6)],
        ),
        (
            KernelInstruction::MatVec {
                matrix: id(4),
                vector: id(5),
            },
            vec![id(4), id(5)],
        ),
        (
            KernelInstruction::Dot {
                lhs: id(3),
                rhs: id(5),
            },
            vec![id(3), id(5)],
        ),
        (
            KernelInstruction::Solve {
                matrix: id(4),
                rhs: id(5),
            },
            vec![id(4), id(5)],
        ),
        (
            KernelInstruction::SolveRow {
                row_slot: 0,
                rhs: vec![id(1), id(0)],
            },
            vec![id(1), id(0)],
        ),
        (
            KernelInstruction::SolveRowAdjointElement {
                row_slot: 0,
                index: 0,
                len: 2,
                adjoint: id(1),
            },
            vec![id(1)],
        ),
    ];

    for (instruction, expected) in cases {
        assert_eq!(instruction.operands(), expected);
    }
}

#[test]
fn scalar_kind_inference_and_promotion_are_characterized() {
    let invariant = KernelValueClass::Invariant;
    for (instruction, expected) in [
        (
            KernelInstruction::Unary {
                op: UnaryOp::Neg,
                input: id(0),
            },
            KernelValueKind::Real,
        ),
        (
            KernelInstruction::Unary {
                op: UnaryOp::Neg,
                input: id(1),
            },
            KernelValueKind::Complex,
        ),
        (
            KernelInstruction::Unary {
                op: UnaryOp::NormSqr,
                input: id(1),
            },
            KernelValueKind::Real,
        ),
        (
            KernelInstruction::Binary {
                op: BinaryOp::Add,
                lhs: id(0),
                rhs: id(0),
            },
            KernelValueKind::Real,
        ),
        (
            KernelInstruction::Binary {
                op: BinaryOp::Mul,
                lhs: id(0),
                rhs: id(1),
            },
            KernelValueKind::Complex,
        ),
        (
            KernelInstruction::Binary {
                op: BinaryOp::Atan2,
                lhs: id(0),
                rhs: id(0),
            },
            KernelValueKind::Real,
        ),
        (
            KernelInstruction::Add(vec![id(0), id(1)]),
            KernelValueKind::Complex,
        ),
        (
            KernelInstruction::Mul(vec![id(0), id(0)]),
            KernelValueKind::Real,
        ),
    ] {
        assert_inference(instruction, Some(expected), invariant);
    }

    assert_shape_error(
        KernelInstruction::Unary {
            op: UnaryOp::Neg,
            input: id(3),
        },
        "unary operation",
    );
    assert_shape_error(
        KernelInstruction::Binary {
            op: BinaryOp::Atan2,
            lhs: id(0),
            rhs: id(1),
        },
        "atan2",
    );
    assert_shape_error(KernelInstruction::Add(vec![id(0), id(3)]), "addition");

    let values = inference_inputs();
    for (instruction, operation) in [
        (KernelInstruction::Add(vec![]), "addition"),
        (KernelInstruction::Mul(vec![]), "multiplication"),
    ] {
        assert_eq!(
            instruction.expected_kind(&values, values.len()),
            Err(KernelIrError::EmptyOperands {
                value: 7,
                operation,
            })
        );
    }
}

#[test]
fn constants_construction_and_event_class_are_characterized() {
    assert_inference(
        KernelInstruction::ComplexConstant(Complex64::new(2.0, 0.0)),
        Some(KernelValueKind::Real),
        KernelValueClass::Invariant,
    );
    assert_inference(
        KernelInstruction::ComplexConstant(Complex64::new(2.0, -0.5)),
        Some(KernelValueKind::Complex),
        KernelValueClass::Invariant,
    );
    assert_inference(KernelInstruction::Cached(3), None, KernelValueClass::Event);
    assert_inference(
        KernelInstruction::Complex {
            re: id(0),
            im: id(0),
        },
        Some(KernelValueKind::Complex),
        KernelValueClass::Invariant,
    );
    assert_shape_error(
        KernelInstruction::Complex {
            re: id(1),
            im: id(0),
        },
        "complex construction",
    );

    for instruction in [
        KernelInstruction::Unary {
            op: UnaryOp::Sin,
            input: id(2),
        },
        KernelInstruction::Add(vec![id(0), id(2)]),
        KernelInstruction::Vector(vec![id(0), id(2)]),
    ] {
        assert_eq!(
            instruction.expected_class(&inference_inputs()),
            KernelValueClass::Event
        );
    }
    assert_inference(
        KernelInstruction::SolveRow {
            row_slot: 0,
            rhs: vec![id(0)],
        },
        Some(KernelValueKind::Complex),
        KernelValueClass::Event,
    );
    assert_inference(
        KernelInstruction::SolveRowAdjointElement {
            row_slot: 0,
            index: 0,
            len: 1,
            adjoint: id(0),
        },
        Some(KernelValueKind::Complex),
        KernelValueClass::Event,
    );
}

#[test]
fn vector_matrix_shapes_and_indices_are_characterized() {
    assert_inference(
        KernelInstruction::Vector(vec![]),
        Some(KernelValueKind::Vector { len: 0 }),
        KernelValueClass::Invariant,
    );
    assert_inference(
        KernelInstruction::Matrix {
            rows: 1,
            cols: 2,
            elements: vec![id(0), id(1)],
        },
        Some(KernelValueKind::Matrix { rows: 1, cols: 2 }),
        KernelValueClass::Invariant,
    );
    assert_inference(
        KernelInstruction::Component {
            input: id(3),
            index: 1,
        },
        Some(KernelValueKind::Complex),
        KernelValueClass::Invariant,
    );
    assert_inference(
        KernelInstruction::MatrixElement {
            input: id(4),
            row: 1,
            col: 1,
        },
        Some(KernelValueKind::Complex),
        KernelValueClass::Invariant,
    );

    assert_shape_error(
        KernelInstruction::Vector(vec![id(3)]),
        "vector construction",
    );
    assert_shape_error(
        KernelInstruction::Matrix {
            rows: 2,
            cols: 2,
            elements: vec![id(0)],
        },
        "matrix construction",
    );
    assert_shape_error(
        KernelInstruction::Component {
            input: id(3),
            index: 2,
        },
        "component",
    );
    assert_shape_error(
        KernelInstruction::MatrixElement {
            input: id(4),
            row: 2,
            col: 0,
        },
        "matrix element",
    );
}

#[test]
fn linear_algebra_compatibility_is_characterized() {
    let invariant = KernelValueClass::Invariant;
    assert_inference(
        KernelInstruction::MatMul {
            lhs: id(4),
            rhs: id(4),
        },
        Some(KernelValueKind::Matrix { rows: 2, cols: 2 }),
        invariant,
    );
    assert_inference(
        KernelInstruction::MatVec {
            matrix: id(4),
            vector: id(3),
        },
        Some(KernelValueKind::Vector { len: 2 }),
        invariant,
    );
    assert_inference(
        KernelInstruction::Dot {
            lhs: id(3),
            rhs: id(3),
        },
        Some(KernelValueKind::Complex),
        invariant,
    );
    assert_inference(
        KernelInstruction::Solve {
            matrix: id(4),
            rhs: id(3),
        },
        Some(KernelValueKind::Vector { len: 2 }),
        invariant,
    );

    for (instruction, operation) in [
        (
            KernelInstruction::MatMul {
                lhs: id(4),
                rhs: id(3),
            },
            "matrix multiplication",
        ),
        (
            KernelInstruction::MatVec {
                matrix: id(4),
                vector: id(0),
            },
            "matrix-vector multiplication",
        ),
        (
            KernelInstruction::Dot {
                lhs: id(3),
                rhs: id(0),
            },
            "dot product",
        ),
        (
            KernelInstruction::Solve {
                matrix: id(3),
                rhs: id(3),
            },
            "linear solve",
        ),
    ] {
        assert_shape_error(instruction, operation);
    }
}

#[test]
fn specialized_solve_shape_errors_are_structured() {
    assert_shape_error(
        KernelInstruction::SolveRow {
            row_slot: 0,
            rhs: vec![],
        },
        "specialized solve row",
    );
    assert_shape_error(
        KernelInstruction::SolveRow {
            row_slot: 0,
            rhs: vec![id(3)],
        },
        "specialized solve row",
    );
    for (index, len, adjoint) in [(0, 0, id(0)), (2, 2, id(0)), (0, 1, id(3))] {
        assert_shape_error(
            KernelInstruction::SolveRowAdjointElement {
                row_slot: 0,
                index,
                len,
                adjoint,
            },
            "specialized solve-row adjoint",
        );
    }
}

#[test]
fn validates_aggregate_operations() {
    let values = vec![
        KernelValue {
            kind: KernelValueKind::Complex,
            class: KernelValueClass::Invariant,
            instruction: KernelInstruction::ComplexConstant(Complex64::new(1.0, 2.0)),
        },
        KernelValue {
            kind: KernelValueKind::Vector { len: 1 },
            class: KernelValueClass::Invariant,
            instruction: KernelInstruction::Vector(vec![KernelValueId::from_index(0)]),
        },
        KernelValue {
            kind: KernelValueKind::Complex,
            class: KernelValueClass::Invariant,
            instruction: KernelInstruction::Dot {
                lhs: KernelValueId::from_index(1),
                rhs: KernelValueId::from_index(1),
            },
        },
    ];
    ScalarKernelIr::new(values, KernelValueId::from_index(2)).unwrap();
}

#[test]
fn rejects_forward_references() {
    let error = ScalarKernelIr::new(
        vec![
            KernelValue {
                kind: KernelValueKind::Real,
                class: KernelValueClass::Invariant,
                instruction: KernelInstruction::Unary {
                    op: UnaryOp::Neg,
                    input: KernelValueId::from_index(1),
                },
            },
            KernelValue {
                kind: KernelValueKind::Real,
                class: KernelValueClass::Invariant,
                instruction: KernelInstruction::RealConstant(1.0),
            },
        ],
        KernelValueId::from_index(0),
    )
    .unwrap_err();
    assert_eq!(
        error,
        KernelIrError::InvalidOperand {
            value: 0,
            operand: 1
        }
    );
}

#[test]
fn rejects_invalid_matrix_shapes() {
    let error = ScalarKernelIr::new(
        vec![
            KernelValue {
                kind: KernelValueKind::Real,
                class: KernelValueClass::Invariant,
                instruction: KernelInstruction::RealConstant(1.0),
            },
            KernelValue {
                kind: KernelValueKind::Matrix { rows: 2, cols: 2 },
                class: KernelValueClass::Invariant,
                instruction: KernelInstruction::Matrix {
                    rows: 2,
                    cols: 2,
                    elements: vec![KernelValueId::from_index(0)],
                },
            },
        ],
        KernelValueId::from_index(0),
    )
    .unwrap_err();
    assert!(matches!(error, KernelIrError::InvalidShape { .. }));
}

#[test]
fn gradient_builder_appends_valid_real_outputs() {
    let primal = ScalarKernelIr::new(
        vec![KernelValue {
            kind: KernelValueKind::Complex,
            class: KernelValueClass::Invariant,
            instruction: KernelInstruction::ComplexConstant(Complex64::new(1.0, 2.0)),
        }],
        KernelValueId::from_index(0),
    )
    .unwrap();
    let mut builder = KernelIrBuilder::from_scalar(&primal);
    let output = builder
        .push(KernelInstruction::Unary {
            op: UnaryOp::Real,
            input: primal.root(),
        })
        .unwrap();
    let gradient = builder
        .finish_gradient(primal.root(), vec![output], OutputComponent::Real)
        .unwrap();

    assert_eq!(gradient.primal_root(), primal.root());
    assert_eq!(gradient.outputs(), &[output]);
    assert_eq!(gradient.component(), OutputComponent::Real);
    assert_eq!(gradient.values().len(), 2);
}

#[test]
fn cache_kernel_preserves_multiple_typed_outputs() {
    let values = vec![
        KernelValue {
            kind: KernelValueKind::Real,
            class: KernelValueClass::Event,
            instruction: KernelInstruction::Cached(0),
        },
        KernelValue {
            kind: KernelValueKind::Real,
            class: KernelValueClass::Event,
            instruction: KernelInstruction::Unary {
                op: UnaryOp::Sin,
                input: KernelValueId::from_index(0),
            },
        },
    ];
    let kernel = CacheKernelIr::new(
        values,
        vec![KernelValueId::from_index(0), KernelValueId::from_index(1)],
    )
    .unwrap();

    assert_eq!(kernel.outputs().len(), 2);
    assert_eq!(
        kernel.values()[kernel.outputs()[1].index()].kind,
        KernelValueKind::Real
    );
}

#[test]
fn wrapper_policies_reject_empty_value_graphs() {
    assert_eq!(
        ScalarKernelIr::new(vec![], id(0)).unwrap_err(),
        KernelIrError::Empty
    );
    assert_eq!(
        CacheKernelIr::new(vec![], vec![id(0)]).unwrap_err(),
        KernelIrError::Empty
    );
    assert_eq!(
        GradientKernelIr::new(vec![], id(0), vec![], OutputComponent::Real).unwrap_err(),
        KernelIrError::Empty
    );
}

#[test]
fn cache_output_bounds_errors_do_not_depend_on_output_position() {
    let values = vec![value(
        KernelValueKind::Real,
        KernelValueClass::Invariant,
        KernelInstruction::RealConstant(1.0),
    )];

    for outputs in [vec![id(1)], vec![id(0), id(1)]] {
        assert_eq!(
            CacheKernelIr::new(values.clone(), outputs).unwrap_err(),
            KernelIrError::CacheOutputOutOfBounds { output: 1, len: 1 }
        );
    }
}

#[test]
fn scalar_and_gradient_roots_must_be_scalar() {
    let values = vec![value(
        KernelValueKind::Vector { len: 0 },
        KernelValueClass::Invariant,
        KernelInstruction::Vector(vec![]),
    )];

    assert_eq!(
        ScalarKernelIr::new(values.clone(), id(0)).unwrap_err(),
        KernelIrError::InvalidShape {
            value: 0,
            operation: "kernel root",
            message: "root must be scalar".into(),
        }
    );
    assert_eq!(
        GradientKernelIr::new(values, id(0), vec![], OutputComponent::Real).unwrap_err(),
        KernelIrError::InvalidShape {
            value: 0,
            operation: "gradient primal root",
            message: "primal root must be scalar".into(),
        }
    );
}

#[test]
fn gradient_output_bounds_errors_are_specific() {
    let values = vec![value(
        KernelValueKind::Real,
        KernelValueClass::Invariant,
        KernelInstruction::RealConstant(1.0),
    )];

    assert_eq!(
        GradientKernelIr::new(values, id(0), vec![id(1)], OutputComponent::Real).unwrap_err(),
        KernelIrError::GradientOutOfBounds { output: 1, len: 1 }
    );
}

#[test]
fn gradient_outputs_must_be_real() {
    let primal = ScalarKernelIr::new(
        vec![KernelValue {
            kind: KernelValueKind::Complex,
            class: KernelValueClass::Invariant,
            instruction: KernelInstruction::ComplexConstant(Complex64::new(1.0, 2.0)),
        }],
        KernelValueId::from_index(0),
    )
    .unwrap();
    let error = GradientKernelIr::new(
        primal.values().to_vec(),
        primal.root(),
        vec![primal.root()],
        OutputComponent::Real,
    )
    .unwrap_err();

    assert_eq!(
        error,
        KernelIrError::GradientKindMismatch {
            output: 0,
            actual: KernelValueKind::Complex,
        }
    );
}
