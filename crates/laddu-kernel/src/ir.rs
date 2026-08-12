pub use crate::KernelIrError;
use laddu_expr::{BinaryOp, UnaryOp, parameters::ParamId};
use num::complex::Complex64;

/// Stable identifier for a value in kernel IR.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct KernelValueId(usize);

impl KernelValueId {
    /// Creates an identifier from a zero-based value index.
    pub fn from_index(index: usize) -> Self {
        Self(index)
    }

    /// Returns the zero-based value index.
    pub fn index(self) -> usize {
        self.0
    }
}

/// Runtime shape and scalar representation of a kernel value.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum KernelValueKind {
    /// A real scalar.
    Real,
    /// A complex scalar.
    Complex,
    /// A complex vector.
    Vector {
        /// Number of elements.
        len: usize,
    },
    /// A complex matrix.
    Matrix {
        /// Number of rows.
        rows: usize,
        /// Number of columns.
        cols: usize,
    },
}

impl KernelValueKind {
    /// Returns the number of logical scalar elements.
    ///
    /// # Panics
    ///
    /// Panics when matrix dimensions exceed the addressable `usize` width.
    /// Validated kernel IR rejects such dimensions during construction.
    pub fn width(self) -> usize {
        match self {
            Self::Real | Self::Complex => 1,
            Self::Vector { len } => len,
            Self::Matrix { rows, cols } => checked_matrix_width(rows, cols)
                .expect("kernel matrix dimensions exceed addressable width"),
        }
    }

    fn scalar_combine(self, rhs: Self) -> Option<Self> {
        match (self, rhs) {
            (Self::Real, Self::Real) => Some(Self::Real),
            (Self::Real | Self::Complex, Self::Real | Self::Complex) => Some(Self::Complex),
            _ => None,
        }
    }

    fn is_scalar(self) -> bool {
        matches!(self, Self::Real | Self::Complex)
    }
}

fn checked_matrix_width(rows: usize, cols: usize) -> Option<usize> {
    rows.checked_mul(cols)
}

fn checked_row_major_index(rows: usize, cols: usize, row: usize, col: usize) -> Option<usize> {
    checked_matrix_width(rows, cols)?;
    if row >= rows || col >= cols {
        return None;
    }
    row.checked_mul(cols)?.checked_add(col)
}

/// Whether a kernel value is constant across events or event-dependent.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum KernelValueClass {
    /// The value depends only on constants and parameters.
    Invariant,
    /// The value depends on event data or cache inputs.
    Event,
}

/// How an instruction's event dependence is determined.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum KernelEventDependence {
    /// The instruction is invariant regardless of its operands.
    Invariant,
    /// The instruction is event-dependent regardless of its operands.
    Event,
    /// The instruction is event-dependent when any direct operand is event-dependent.
    Operands,
}

/// Operation that produces one value in kernel IR.
#[derive(Clone, Debug)]
pub enum KernelInstruction {
    /// Reads a precomputed cache slot.
    Cached(usize),
    /// Emits a real constant.
    RealConstant(f64),
    /// Emits a complex constant.
    ComplexConstant(Complex64),
    /// Reads a scalar parameter.
    Parameter(ParamId),
    /// Applies a unary scalar operation.
    Unary {
        /// Operation to apply.
        op: UnaryOp,
        /// Input value.
        input: KernelValueId,
    },
    /// Applies a binary scalar operation.
    Binary {
        /// Operation to apply.
        op: BinaryOp,
        /// Left operand.
        lhs: KernelValueId,
        /// Right operand.
        rhs: KernelValueId,
    },
    /// Adds scalar operands.
    Add(Vec<KernelValueId>),
    /// Multiplies scalar operands.
    Mul(Vec<KernelValueId>),
    /// Constructs a complex scalar from real components.
    Complex {
        /// Real component.
        re: KernelValueId,
        /// Imaginary component.
        im: KernelValueId,
    },
    /// Constructs a vector from scalar elements.
    Vector(Vec<KernelValueId>),
    /// Constructs a row-major matrix.
    Matrix {
        /// Number of rows.
        rows: usize,
        /// Number of columns.
        cols: usize,
        /// Row-major scalar elements.
        elements: Vec<KernelValueId>,
    },
    /// Selects a vector component.
    Component {
        /// Vector input.
        input: KernelValueId,
        /// Zero-based component index.
        index: usize,
    },
    /// Selects a matrix element.
    MatrixElement {
        /// Matrix input.
        input: KernelValueId,
        /// Zero-based row index.
        row: usize,
        /// Zero-based column index.
        col: usize,
    },
    /// Multiplies two matrices.
    MatMul {
        /// Left matrix.
        lhs: KernelValueId,
        /// Right matrix.
        rhs: KernelValueId,
    },
    /// Multiplies a matrix by a vector.
    MatVec {
        /// Matrix operand.
        matrix: KernelValueId,
        /// Vector operand.
        vector: KernelValueId,
    },
    /// Computes a vector dot product.
    Dot {
        /// Left vector.
        lhs: KernelValueId,
        /// Right vector.
        rhs: KernelValueId,
    },
    /// Solves a linear system.
    Solve {
        /// Coefficient matrix.
        matrix: KernelValueId,
        /// Right-hand-side vector.
        rhs: KernelValueId,
    },
    /// Evaluates one row of a specialized cached solve.
    SolveRow {
        /// Cache slot containing the decomposed matrix row data.
        row_slot: usize,
        /// Right-hand-side scalar values.
        rhs: Vec<KernelValueId>,
    },
    /// Evaluates one adjoint element for a specialized solve row.
    SolveRowAdjointElement {
        /// Cache slot containing the decomposed matrix row data.
        row_slot: usize,
        /// Element index within the row.
        index: usize,
        /// Row length.
        len: usize,
        /// Incoming scalar adjoint.
        adjoint: KernelValueId,
    },
}

/// Typed instruction and evaluation class for one kernel IR value.
#[derive(Clone, Debug)]
pub struct KernelValue {
    /// Value shape and scalar representation.
    pub kind: KernelValueKind,
    /// Event-dependency class.
    pub class: KernelValueClass,
    /// Instruction that produces the value.
    pub instruction: KernelInstruction,
}

/// Validated IR for a kernel with one scalar output.
#[derive(Clone, Debug)]
pub struct ScalarKernelIr {
    values: Vec<KernelValue>,
    root: KernelValueId,
}

/// Validated IR for a kernel that populates multiple cache outputs.
#[derive(Clone, Debug)]
pub struct CacheKernelIr {
    values: Vec<KernelValue>,
    outputs: Vec<KernelValueId>,
}

/// Scalar component of a complex primal output to differentiate.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum OutputComponent {
    /// Differentiate the real component.
    Real,
    /// Differentiate the imaginary component.
    Imag,
}

/// Validated IR containing a primal computation and real gradient outputs.
#[derive(Clone, Debug)]
pub struct GradientKernelIr {
    values: Vec<KernelValue>,
    primal_root: KernelValueId,
    outputs: Vec<KernelValueId>,
    component: OutputComponent,
}

/// Builder for appending type-checked instructions to existing scalar IR.
#[derive(Clone, Debug)]
pub struct KernelIrBuilder {
    values: Vec<KernelValue>,
}

mod builder;
mod instruction;
mod validate;
mod wrappers;

use validate::validate_graph;

#[cfg(test)]
mod tests;
