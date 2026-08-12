//! Auditable support matrix for every kernel instruction variant.
//!
//! `Supported` means the layer has an explicit semantic path for the variant.
//! `Rejected` means it has an explicit, intentional unsupported path rather
//! than silently accepting the instruction. The exhaustive match makes a new
//! `KernelInstruction` variant fail this test at compile time until its support
//! status is reviewed across every layer.

use laddu_expr::{
    BinaryOp, UnaryOp,
    parameters::{ParamLayout, Parameter},
};
use laddu_kernel::ir::{KernelInstruction, KernelValueId};
use num::complex::Complex64;
use std::collections::HashSet;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Support {
    Supported,
    Rejected,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct SupportRow {
    name: &'static str,
    validation: Support,
    autodiff: Support,
    cpu_interpreter: Support,
    jit: Support,
    wgpu: Support,
}

fn support(instruction: &KernelInstruction) -> SupportRow {
    use Support::{Rejected, Supported};

    let (autodiff, wgpu) = match instruction {
        KernelInstruction::Cached(_)
        | KernelInstruction::RealConstant(_)
        | KernelInstruction::ComplexConstant(_)
        | KernelInstruction::Parameter(_)
        | KernelInstruction::Unary { .. }
        | KernelInstruction::Binary { .. }
        | KernelInstruction::Add(_)
        | KernelInstruction::Mul(_)
        | KernelInstruction::Complex { .. }
        | KernelInstruction::Vector(_)
        | KernelInstruction::Matrix { .. }
        | KernelInstruction::Component { .. }
        | KernelInstruction::MatrixElement { .. }
        | KernelInstruction::MatMul { .. }
        | KernelInstruction::MatVec { .. }
        | KernelInstruction::Dot { .. }
        | KernelInstruction::Solve { .. } => (Supported, Supported),
        KernelInstruction::SolveRow { .. } => (Supported, Rejected),
        KernelInstruction::SolveRowAdjointElement { .. } => (Rejected, Rejected),
    };

    SupportRow {
        name: instruction.diagnostic_name(),
        validation: Supported,
        autodiff,
        cpu_interpreter: Supported,
        jit: Supported,
        wgpu,
    }
}

fn id(index: usize) -> KernelValueId {
    KernelValueId::from_index(index)
}

fn representatives() -> Vec<KernelInstruction> {
    let parameter = ParamLayout::new([Parameter::free("p")])
        .unwrap()
        .id("p")
        .unwrap();
    vec![
        KernelInstruction::Cached(0),
        KernelInstruction::RealConstant(1.0),
        KernelInstruction::ComplexConstant(Complex64::new(1.0, 1.0)),
        KernelInstruction::Parameter(parameter),
        KernelInstruction::Unary {
            op: UnaryOp::Neg,
            input: id(0),
        },
        KernelInstruction::Binary {
            op: BinaryOp::Add,
            lhs: id(0),
            rhs: id(1),
        },
        KernelInstruction::Add(vec![id(0)]),
        KernelInstruction::Mul(vec![id(0)]),
        KernelInstruction::Complex {
            re: id(0),
            im: id(1),
        },
        KernelInstruction::Vector(vec![id(0)]),
        KernelInstruction::Matrix {
            rows: 1,
            cols: 1,
            elements: vec![id(0)],
        },
        KernelInstruction::Component {
            input: id(0),
            index: 0,
        },
        KernelInstruction::MatrixElement {
            input: id(0),
            row: 0,
            col: 0,
        },
        KernelInstruction::MatMul {
            lhs: id(0),
            rhs: id(1),
        },
        KernelInstruction::MatVec {
            matrix: id(0),
            vector: id(1),
        },
        KernelInstruction::Dot {
            lhs: id(0),
            rhs: id(1),
        },
        KernelInstruction::Solve {
            matrix: id(0),
            rhs: id(1),
        },
        KernelInstruction::SolveRow {
            row_slot: 0,
            rhs: vec![id(0)],
        },
        KernelInstruction::SolveRowAdjointElement {
            row_slot: 0,
            index: 0,
            len: 1,
            adjoint: id(0),
        },
    ]
}

#[test]
fn every_instruction_variant_has_one_support_row() {
    let rows: Vec<_> = representatives().iter().map(support).collect();
    let names: HashSet<_> = rows.iter().map(|row| row.name).collect();

    assert_eq!(rows.len(), 19);
    assert_eq!(names.len(), rows.len());
    assert!(rows.iter().all(|row| row.validation == Support::Supported));
    assert!(
        rows.iter()
            .all(|row| row.cpu_interpreter == Support::Supported)
    );
    assert!(rows.iter().all(|row| row.jit == Support::Supported));
}

#[test]
fn intentional_support_gaps_are_explicit() {
    let rows: Vec<_> = representatives().iter().map(support).collect();
    let rejected_by_autodiff: Vec<_> = rows
        .iter()
        .filter(|row| row.autodiff == Support::Rejected)
        .map(|row| row.name)
        .collect();
    let rejected_by_wgpu: Vec<_> = rows
        .iter()
        .filter(|row| row.wgpu == Support::Rejected)
        .map(|row| row.name)
        .collect();

    assert_eq!(rejected_by_autodiff, ["SolveRowAdjointElement"]);
    assert_eq!(rejected_by_wgpu, ["SolveRow", "SolveRowAdjointElement"]);
}
