use laddu_expr::{BinaryOp, UnaryOp, parameters::ParamId};
use num::complex::Complex64;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct KernelValueId(usize);

impl KernelValueId {
    pub fn from_index(index: usize) -> Self {
        Self(index)
    }

    pub fn index(self) -> usize {
        self.0
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum KernelScalarKind {
    Real,
    Complex,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum KernelValueClass {
    Invariant,
    Event,
}

#[derive(Clone, Debug)]
pub enum KernelInstruction {
    Cached(usize),
    RealConstant(f64),
    ComplexConstant(Complex64),
    Parameter(ParamId),
    Unary {
        op: UnaryOp,
        input: KernelValueId,
    },
    Binary {
        op: BinaryOp,
        lhs: KernelValueId,
        rhs: KernelValueId,
    },
    Add(Vec<KernelValueId>),
    Mul(Vec<KernelValueId>),
    Complex {
        re: KernelValueId,
        im: KernelValueId,
    },
    SolveRow {
        row_slot: usize,
        rhs: Vec<KernelValueId>,
    },
}

#[derive(Clone, Debug)]
pub struct KernelValue {
    pub kind: KernelScalarKind,
    pub class: KernelValueClass,
    pub instruction: KernelInstruction,
}

#[derive(Clone, Debug)]
pub struct ScalarKernelIr {
    values: Vec<KernelValue>,
    root: KernelValueId,
}

impl ScalarKernelIr {
    pub fn new(values: Vec<KernelValue>, root: KernelValueId) -> Self {
        Self { values, root }
    }

    pub fn values(&self) -> &[KernelValue] {
        &self.values
    }

    pub fn root(&self) -> KernelValueId {
        self.root
    }
}
