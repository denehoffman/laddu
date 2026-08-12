use std::{
    collections::HashMap,
    fmt,
    sync::{Arc, OnceLock},
};

use num::complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::{
    ExprGraphError, ExprShapeError, ParamError, ParamResult,
    parameters::{InitialSpec, ParamState, Parameter},
};

/// Stable identifier for a node in a serialized [`ExprGraph`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ExprId(u64);

impl ExprId {
    /// Creates an identifier from a zero-based node index.
    pub fn from_index(index: usize) -> Self {
        Self(index as u64)
    }

    /// Returns the zero-based node index.
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// Runtime value category produced by an expression node.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ValueKind {
    /// A real scalar.
    Real,
    /// A complex scalar.
    Complex,
    /// A vector with a fixed number of elements.
    Vector {
        /// Number of vector elements.
        len: usize,
    },
    /// A matrix with fixed dimensions.
    Matrix {
        /// Number of rows.
        rows: usize,
        /// Number of columns.
        cols: usize,
    },
}

/// Statically known scalar number category.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum NumberClass {
    /// No narrower category is known.
    Unknown,
    /// The value is real.
    Real,
    /// The value is purely imaginary.
    Imaginary,
    /// The value may have real and imaginary components.
    Complex,
}

/// Context-free value semantics inferred for one expression node.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ExprNodeSemantics {
    /// Runtime value kind.
    pub value_kind: ValueKind,
    /// Known relationship between real and imaginary components.
    pub number_class: NumberClass,
}

fn add_number_class(lhs: NumberClass, rhs: NumberClass) -> NumberClass {
    use NumberClass::{Complex, Imaginary, Real, Unknown};
    match (lhs, rhs) {
        (Real, Real) => Real,
        (Imaginary, Imaginary) => Imaginary,
        (Complex, _) | (_, Complex) => Complex,
        (Unknown, _) | (_, Unknown) => Unknown,
        _ => Complex,
    }
}

fn mul_number_class(lhs: NumberClass, rhs: NumberClass) -> NumberClass {
    use NumberClass::{Complex, Imaginary, Real, Unknown};
    match (lhs, rhs) {
        (Real, Real) | (Imaginary, Imaginary) => Real,
        (Real, Imaginary) | (Imaginary, Real) => Imaginary,
        (Complex, _) | (_, Complex) => Complex,
        (Unknown, _) | (_, Unknown) => Unknown,
    }
}

/// Intrinsic source of a node's evaluation dependencies.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ExprDependencyKind {
    /// The node is a compile-time constant.
    Constant,
    /// The node directly reads a parameter definition.
    Parameter,
    /// The node directly reads event data.
    Event,
    /// The node inherits the union of its children's dependencies.
    Children,
}

/// Structural shape of an expression.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExprShape {
    /// A scalar expression.
    Scalar,
    /// A vector expression.
    Vector {
        /// Number of vector elements.
        len: usize,
    },
    /// A matrix expression.
    Matrix {
        /// Number of rows.
        rows: usize,
        /// Number of columns.
        cols: usize,
    },
}

impl fmt::Display for ExprShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scalar => write!(f, "scalar"),
            Self::Vector { len } => write!(f, "vector[{len}]"),
            Self::Matrix { rows, cols } => write!(f, "matrix[{rows}x{cols}]"),
        }
    }
}

/// Converts a component selector into a zero-based index.
pub trait ComponentIndex {
    /// Returns the selected zero-based component index.
    fn component_index(self) -> usize;
}

impl ComponentIndex for usize {
    fn component_index(self) -> usize {
        self
    }
}

impl ComponentIndex for i32 {
    fn component_index(self) -> usize {
        usize::try_from(self).expect("component index must be nonnegative")
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
/// A named component of a four-momentum in `(E, px, py, pz)` order.
pub enum P4Component {
    /// Energy.
    E,
    /// Momentum in the x direction.
    Px,
    /// Momentum in the y direction.
    Py,
    /// Momentum in the z direction.
    Pz,
}

impl P4Component {
    /// Return the lowercase event-column suffix for this component.
    pub fn label(self) -> &'static str {
        match self {
            Self::E => "e",
            Self::Px => "px",
            Self::Py => "py",
            Self::Pz => "pz",
        }
    }

    /// Return the component's position in `(E, px, py, pz)` order.
    pub fn index(self) -> usize {
        match self {
            Self::E => 0,
            Self::Px => 1,
            Self::Py => 2,
            Self::Pz => 3,
        }
    }
}

/// Unary operation in an expression graph.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum UnaryOp {
    /// Arithmetic negation.
    Neg,
    /// Real part.
    Real,
    /// Imaginary part.
    Imag,
    /// Complex conjugate.
    Conj,
    /// Squared complex norm.
    NormSqr,
    /// Principal square root.
    Sqrt,
    /// Exponential.
    Exp,
    /// Sine.
    Sin,
    /// Cosine.
    Cos,
    /// Natural logarithm.
    Log,
    /// Integer power.
    PowI(i32),
}

impl UnaryOp {
    /// Applies this operation to a scalar complex value.
    pub fn evaluate(&self, value: Complex64) -> Complex64 {
        match self {
            Self::Neg => -value,
            Self::Real => Complex64::from(value.re),
            Self::Imag => Complex64::from(value.im),
            Self::Conj => value.conj(),
            Self::NormSqr => Complex64::from(value.norm_sqr()),
            Self::Sqrt => value.sqrt(),
            Self::Exp => value.exp(),
            Self::Sin => value.sin(),
            Self::Cos => value.cos(),
            Self::Log => value.ln(),
            Self::PowI(power) => value.powi(*power),
        }
    }
}

/// Binary operation in an expression graph.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum BinaryOp {
    /// Addition.
    Add,
    /// Subtraction.
    Sub,
    /// Multiplication.
    Mul,
    /// Division.
    Div,
    /// Two-argument arctangent of the real parts.
    Atan2,
}

impl BinaryOp {
    /// Applies this operation to two scalar complex values.
    pub fn evaluate(&self, a: Complex64, b: Complex64) -> Complex64 {
        match self {
            Self::Add => a + b,
            Self::Sub => a - b,
            Self::Mul => a * b,
            Self::Div => a / b,
            Self::Atan2 => Complex64::from(a.re.atan2(b.re)),
        }
    }
}

/// Serialized node in a topologically ordered [`ExprGraph`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ExprNode {
    /// A real constant.
    RealConst(f64),
    /// A complex constant.
    ComplexConst(Complex64),
    /// A scalar fit parameter.
    ScalarParam(Parameter),
    /// A named scalar event column.
    EventScalar(Arc<str>),
    /// One component of a named event four-momentum.
    EventP4Component {
        /// Event column base name.
        name: Arc<str>,
        /// Requested four-momentum component.
        component: P4Component,
    },
    /// A unary operation.
    Unary {
        /// Operation to apply.
        op: UnaryOp,
        /// Input node.
        input: ExprId,
    },
    /// A binary operation.
    Binary {
        /// Operation to apply.
        op: BinaryOp,
        /// Left operand.
        lhs: ExprId,
        /// Right operand.
        rhs: ExprId,
    },
    /// A sum of zero or more terms.
    NaryAdd {
        /// Term nodes.
        terms: Vec<ExprId>,
    },
    /// A product of zero or more factors.
    NaryMul {
        /// Factor nodes.
        factors: Vec<ExprId>,
    },
    /// A complex scalar assembled from real and imaginary expressions.
    Complex {
        /// Real component.
        re: ExprId,
        /// Imaginary component.
        im: ExprId,
    },
    /// A vector assembled from scalar elements.
    Vector {
        /// Scalar element nodes.
        elements: Vec<ExprId>,
    },
    /// A row-major matrix assembled from scalar elements.
    Matrix {
        /// Number of rows.
        rows: usize,
        /// Number of columns.
        cols: usize,
        /// Row-major scalar elements.
        elements: Vec<ExprId>,
    },
    /// A vector component selection.
    Component {
        /// Vector input.
        input: ExprId,
        /// Zero-based component index.
        index: usize,
    },
    /// A matrix element selection.
    MatrixElement {
        /// Matrix input.
        input: ExprId,
        /// Zero-based row index.
        row: usize,
        /// Zero-based column index.
        col: usize,
    },
    /// Matrix-matrix multiplication.
    MatMul {
        /// Left matrix.
        lhs: ExprId,
        /// Right matrix.
        rhs: ExprId,
    },
    /// Matrix-vector multiplication.
    MatVec {
        /// Matrix operand.
        matrix: ExprId,
        /// Vector operand.
        vector: ExprId,
    },
    /// Vector dot product.
    Dot {
        /// Left vector.
        lhs: ExprId,
        /// Right vector.
        rhs: ExprId,
    },
    /// Solution of a linear system.
    Solve {
        /// Coefficient matrix.
        matrix: ExprId,
        /// Right-hand-side vector or matrix.
        rhs: ExprId,
    },
}

/// Bit-exact structural identity for a scalar parameter definition.
///
/// Equality includes state, initial-value policy, bounds, periodicity, scale,
/// and user-facing labels. Floating-point values are compared by their bit
/// patterns, so signed zero and distinct NaN payloads remain distinct.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParameterStructuralKey {
    name: Arc<str>,
    state: ParameterStateStructuralKey,
    initial: InitialStructuralKey,
    bounds: (Option<u64>, Option<u64>),
    periodic: bool,
    scale: Option<u64>,
    unit: Option<Arc<str>>,
    latex: Option<Arc<str>>,
    description: Option<Arc<str>>,
}

#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ParameterStateStructuralKey {
    Free,
    Fixed(u64),
}

#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum InitialStructuralKey {
    Default,
    Value(u64),
    Uniform { min: u64, max: u64 },
}

/// Bit-exact, metadata-free structural identity for an expression node.
///
/// The key includes the node variant, semantic payload, child identifiers, and
/// complete parameter definitions. It deliberately excludes [`ExprMetadata`].
/// Its ordering is deterministic but its representation and hash values are an
/// internal workspace contract, not a stable serialized or persisted format.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ExprNodeStructuralKey {
    RealConst(u64),
    ComplexConst {
        re: u64,
        im: u64,
    },
    ScalarParam(ParameterStructuralKey),
    EventScalar(Arc<str>),
    EventP4Component {
        name: Arc<str>,
        component: P4Component,
    },
    Unary {
        op: UnaryOp,
        input: ExprId,
    },
    Binary {
        op: BinaryOp,
        lhs: ExprId,
        rhs: ExprId,
    },
    NaryAdd {
        terms: Vec<ExprId>,
    },
    NaryMul {
        factors: Vec<ExprId>,
    },
    Complex {
        re: ExprId,
        im: ExprId,
    },
    Vector {
        elements: Vec<ExprId>,
    },
    Matrix {
        rows: usize,
        cols: usize,
        elements: Vec<ExprId>,
    },
    Component {
        input: ExprId,
        index: usize,
    },
    MatrixElement {
        input: ExprId,
        row: usize,
        col: usize,
    },
    MatMul {
        lhs: ExprId,
        rhs: ExprId,
    },
    MatVec {
        matrix: ExprId,
        vector: ExprId,
    },
    Dot {
        lhs: ExprId,
        rhs: ExprId,
    },
    Solve {
        matrix: ExprId,
        rhs: ExprId,
    },
}

impl From<&Parameter> for ParameterStructuralKey {
    fn from(parameter: &Parameter) -> Self {
        let state = match parameter.state() {
            ParamState::Free => ParameterStateStructuralKey::Free,
            ParamState::Fixed(value) => ParameterStateStructuralKey::Fixed(value.to_bits()),
        };
        let initial = match parameter.initial_spec() {
            InitialSpec::Default => InitialStructuralKey::Default,
            InitialSpec::Value(value) => InitialStructuralKey::Value(value.to_bits()),
            InitialSpec::Uniform { min, max } => InitialStructuralKey::Uniform {
                min: min.to_bits(),
                max: max.to_bits(),
            },
        };
        Self {
            name: Arc::from(parameter.name()),
            state,
            initial,
            bounds: (
                parameter.bounds_spec().min.map(f64::to_bits),
                parameter.bounds_spec().max.map(f64::to_bits),
            ),
            periodic: parameter.is_periodic(),
            scale: parameter.scale().map(f64::to_bits),
            unit: parameter.unit_label().map(Arc::from),
            latex: parameter.latex_label().map(Arc::from),
            description: parameter.description_text().map(Arc::from),
        }
    }
}

impl From<Complex64> for ExprNode {
    fn from(value: Complex64) -> Self {
        if value.im == 0.0 {
            Self::RealConst(value.re)
        } else {
            Self::ComplexConst(value)
        }
    }
}

impl ExprNode {
    /// Infers this node's context-free value semantics from the already
    /// computed semantics of earlier nodes in the expression graph.
    pub fn semantics(&self, children: &[ExprNodeSemantics]) -> ExprNodeSemantics {
        ExprNodeSemantics {
            value_kind: self.infer_value_kind(children),
            number_class: self.infer_number_class(children),
        }
    }

    fn infer_value_kind(&self, children: &[ExprNodeSemantics]) -> ValueKind {
        match self {
            Self::RealConst(_) | Self::ScalarParam(_) => ValueKind::Real,
            Self::ComplexConst(value) => {
                if value.im == 0.0 {
                    ValueKind::Real
                } else {
                    ValueKind::Complex
                }
            }
            Self::EventScalar(_) | Self::EventP4Component { .. } => ValueKind::Real,
            Self::Unary { op, input } => match op {
                UnaryOp::Real | UnaryOp::Imag | UnaryOp::NormSqr => ValueKind::Real,
                UnaryOp::Neg
                | UnaryOp::Conj
                | UnaryOp::Sqrt
                | UnaryOp::Exp
                | UnaryOp::Sin
                | UnaryOp::Cos
                | UnaryOp::Log
                | UnaryOp::PowI(_) => children[input.index()].value_kind,
            },
            Self::Binary { op, lhs, rhs } => {
                if *op == BinaryOp::Atan2 {
                    return ValueKind::Real;
                }
                if children[lhs.index()].value_kind == ValueKind::Real
                    && children[rhs.index()].value_kind == ValueKind::Real
                {
                    ValueKind::Real
                } else {
                    ValueKind::Complex
                }
            }
            Self::NaryAdd { terms } => {
                if terms
                    .iter()
                    .all(|id| children[id.index()].value_kind == ValueKind::Real)
                {
                    ValueKind::Real
                } else {
                    ValueKind::Complex
                }
            }
            Self::NaryMul { factors } => {
                if factors
                    .iter()
                    .all(|id| children[id.index()].value_kind == ValueKind::Real)
                {
                    ValueKind::Real
                } else {
                    ValueKind::Complex
                }
            }
            Self::Complex { .. } => ValueKind::Complex,
            Self::Vector { elements } => ValueKind::Vector {
                len: elements.len(),
            },
            Self::Matrix { rows, cols, .. } => ValueKind::Matrix {
                rows: *rows,
                cols: *cols,
            },
            Self::Component { input, .. } => match children[input.index()].value_kind {
                ValueKind::Vector { .. } => ValueKind::Complex,
                kind => kind,
            },
            Self::MatrixElement { .. } | Self::Dot { .. } => ValueKind::Complex,
            Self::MatMul { lhs, rhs } => {
                let ValueKind::Matrix { rows, .. } = children[lhs.index()].value_kind else {
                    return ValueKind::Complex;
                };
                let ValueKind::Matrix { cols, .. } = children[rhs.index()].value_kind else {
                    return ValueKind::Complex;
                };
                ValueKind::Matrix { rows, cols }
            }
            Self::MatVec { matrix, .. } => {
                let ValueKind::Matrix { rows, .. } = children[matrix.index()].value_kind else {
                    return ValueKind::Complex;
                };
                ValueKind::Vector { len: rows }
            }
            Self::Solve { rhs, .. } => children[rhs.index()].value_kind,
        }
    }

    fn infer_number_class(&self, children: &[ExprNodeSemantics]) -> NumberClass {
        match self {
            Self::RealConst(_) | Self::ScalarParam(_) => NumberClass::Real,
            Self::ComplexConst(value) => match (value.re == 0.0, value.im == 0.0) {
                (_, true) => NumberClass::Real,
                (true, false) => NumberClass::Imaginary,
                (false, false) => NumberClass::Complex,
            },
            Self::EventScalar(_) | Self::EventP4Component { .. } => NumberClass::Real,
            Self::Unary { op, input } => match op {
                UnaryOp::Neg | UnaryOp::Conj => children[input.index()].number_class,
                UnaryOp::Real | UnaryOp::Imag | UnaryOp::NormSqr => NumberClass::Real,
                UnaryOp::Exp | UnaryOp::Sin | UnaryOp::Cos | UnaryOp::PowI(_) => {
                    let input = children[input.index()].number_class;
                    if input == NumberClass::Real {
                        NumberClass::Real
                    } else {
                        NumberClass::Unknown
                    }
                }
                UnaryOp::Sqrt | UnaryOp::Log => NumberClass::Unknown,
            },
            Self::Binary { op, lhs, rhs } => {
                let lhs = children[lhs.index()].number_class;
                let rhs = children[rhs.index()].number_class;
                match op {
                    BinaryOp::Add | BinaryOp::Sub => add_number_class(lhs, rhs),
                    BinaryOp::Mul | BinaryOp::Div => mul_number_class(lhs, rhs),
                    BinaryOp::Atan2 => NumberClass::Real,
                }
            }
            Self::NaryAdd { terms } => {
                let mut classes = terms.iter().map(|id| children[id.index()].number_class);
                let Some(first) = classes.next() else {
                    return NumberClass::Real;
                };
                classes.fold(first, add_number_class)
            }
            Self::NaryMul { factors } => {
                let mut classes = factors.iter().map(|id| children[id.index()].number_class);
                let Some(first) = classes.next() else {
                    return NumberClass::Real;
                };
                classes.fold(first, mul_number_class)
            }
            Self::Complex { .. } => NumberClass::Complex,
            Self::Vector { .. }
            | Self::Matrix { .. }
            | Self::Component { .. }
            | Self::MatrixElement { .. }
            | Self::MatMul { .. }
            | Self::MatVec { .. }
            | Self::Dot { .. }
            | Self::Solve { .. } => NumberClass::Unknown,
        }
    }

    /// Returns the intrinsic source of this node's evaluation dependencies.
    pub fn dependency_kind(&self) -> ExprDependencyKind {
        match self {
            Self::RealConst(_) | Self::ComplexConst(_) => ExprDependencyKind::Constant,
            Self::ScalarParam(_) => ExprDependencyKind::Parameter,
            Self::EventScalar(_) | Self::EventP4Component { .. } => ExprDependencyKind::Event,
            _ => ExprDependencyKind::Children,
        }
    }

    /// Returns this node's bit-exact, metadata-free structural identity.
    #[doc(hidden)]
    pub fn structural_key(&self) -> ExprNodeStructuralKey {
        match self {
            Self::RealConst(value) => ExprNodeStructuralKey::RealConst(value.to_bits()),
            Self::ComplexConst(value) => ExprNodeStructuralKey::ComplexConst {
                re: value.re.to_bits(),
                im: value.im.to_bits(),
            },
            Self::ScalarParam(parameter) => {
                ExprNodeStructuralKey::ScalarParam(ParameterStructuralKey::from(parameter))
            }
            Self::EventScalar(name) => ExprNodeStructuralKey::EventScalar(Arc::clone(name)),
            Self::EventP4Component { name, component } => ExprNodeStructuralKey::EventP4Component {
                name: Arc::clone(name),
                component: *component,
            },
            Self::Unary { op, input } => ExprNodeStructuralKey::Unary {
                op: *op,
                input: *input,
            },
            Self::Binary { op, lhs, rhs } => ExprNodeStructuralKey::Binary {
                op: *op,
                lhs: *lhs,
                rhs: *rhs,
            },
            Self::NaryAdd { terms } => ExprNodeStructuralKey::NaryAdd {
                terms: terms.clone(),
            },
            Self::NaryMul { factors } => ExprNodeStructuralKey::NaryMul {
                factors: factors.clone(),
            },
            Self::Complex { re, im } => ExprNodeStructuralKey::Complex { re: *re, im: *im },
            Self::Vector { elements } => ExprNodeStructuralKey::Vector {
                elements: elements.clone(),
            },
            Self::Matrix {
                rows,
                cols,
                elements,
            } => ExprNodeStructuralKey::Matrix {
                rows: *rows,
                cols: *cols,
                elements: elements.clone(),
            },
            Self::Component { input, index } => ExprNodeStructuralKey::Component {
                input: *input,
                index: *index,
            },
            Self::MatrixElement { input, row, col } => ExprNodeStructuralKey::MatrixElement {
                input: *input,
                row: *row,
                col: *col,
            },
            Self::MatMul { lhs, rhs } => ExprNodeStructuralKey::MatMul {
                lhs: *lhs,
                rhs: *rhs,
            },
            Self::MatVec { matrix, vector } => ExprNodeStructuralKey::MatVec {
                matrix: *matrix,
                vector: *vector,
            },
            Self::Dot { lhs, rhs } => ExprNodeStructuralKey::Dot {
                lhs: *lhs,
                rhs: *rhs,
            },
            Self::Solve { matrix, rhs } => ExprNodeStructuralKey::Solve {
                matrix: *matrix,
                rhs: *rhs,
            },
        }
    }

    /// Creates the most compact constant-node representation for `value`.
    pub fn from_folded_const(value: Complex64) -> Self {
        if value.im == 0.0 && value.im.is_sign_positive() {
            Self::RealConst(value.re)
        } else {
            Self::ComplexConst(value)
        }
    }

    /// Returns the node's scalar constant value, if it is a constant.
    pub fn const_value(&self) -> Option<Complex64> {
        match self {
            ExprNode::RealConst(value) => Some(Complex64::from(*value)),
            ExprNode::ComplexConst(value) => Some(*value),
            _ => None,
        }
    }

    /// Returns whether `node` is the scalar constant zero.
    pub fn is_zero(node: &ExprNode) -> bool {
        node.const_value()
            .is_some_and(|value| value == Complex64::ZERO)
    }

    /// Returns whether `node` is the scalar constant one.
    pub fn is_one(node: &ExprNode) -> bool {
        node.const_value()
            .is_some_and(|value| value == Complex64::ONE)
    }

    /// Iterates over this node's direct dependencies in semantic operand order.
    ///
    /// The iterator borrows the node and does not allocate. Binary operands are
    /// returned left-to-right, and vector, matrix, sum, and product children
    /// retain their stored order.
    pub fn children(&self) -> impl ExactSizeIterator<Item = ExprId> + DoubleEndedIterator + '_ {
        (0..self.child_count()).map(|index| self.child_at(index))
    }

    /// Returns the identifiers of this node's direct dependencies.
    ///
    /// This compatibility helper collects [`Self::children`]. Prefer the
    /// borrowed iterator when an owned vector is not required.
    pub fn child_ids(&self) -> Vec<ExprId> {
        self.children().collect()
    }

    /// Returns a copy of this node with each direct dependency transformed.
    ///
    /// Children are passed to `map` in the same semantic order as
    /// [`Self::children`]. Non-child fields are preserved exactly.
    pub fn map_children(&self, mut map: impl FnMut(ExprId) -> ExprId) -> Self {
        match self {
            Self::RealConst(_)
            | Self::ComplexConst(_)
            | Self::ScalarParam(_)
            | Self::EventScalar(_)
            | Self::EventP4Component { .. } => self.clone(),
            Self::Unary { op, input } => Self::Unary {
                op: *op,
                input: map(*input),
            },
            Self::Binary { op, lhs, rhs } => Self::Binary {
                op: *op,
                lhs: map(*lhs),
                rhs: map(*rhs),
            },
            Self::NaryAdd { terms } => Self::NaryAdd {
                terms: terms.iter().copied().map(&mut map).collect(),
            },
            Self::NaryMul { factors } => Self::NaryMul {
                factors: factors.iter().copied().map(&mut map).collect(),
            },
            Self::Complex { re, im } => Self::Complex {
                re: map(*re),
                im: map(*im),
            },
            Self::Vector { elements } => Self::Vector {
                elements: elements.iter().copied().map(&mut map).collect(),
            },
            Self::Matrix {
                rows,
                cols,
                elements,
            } => Self::Matrix {
                rows: *rows,
                cols: *cols,
                elements: elements.iter().copied().map(&mut map).collect(),
            },
            Self::Component { input, index } => Self::Component {
                input: map(*input),
                index: *index,
            },
            Self::MatrixElement { input, row, col } => Self::MatrixElement {
                input: map(*input),
                row: *row,
                col: *col,
            },
            Self::MatMul { lhs, rhs } => Self::MatMul {
                lhs: map(*lhs),
                rhs: map(*rhs),
            },
            Self::MatVec { matrix, vector } => Self::MatVec {
                matrix: map(*matrix),
                vector: map(*vector),
            },
            Self::Dot { lhs, rhs } => Self::Dot {
                lhs: map(*lhs),
                rhs: map(*rhs),
            },
            Self::Solve { matrix, rhs } => Self::Solve {
                matrix: map(*matrix),
                rhs: map(*rhs),
            },
        }
    }

    fn child_count(&self) -> usize {
        match self {
            Self::RealConst(_)
            | Self::ComplexConst(_)
            | Self::ScalarParam(_)
            | Self::EventScalar(_)
            | Self::EventP4Component { .. } => 0,
            Self::Unary { .. } | Self::Component { .. } | Self::MatrixElement { .. } => 1,
            Self::Binary { .. }
            | Self::Complex { .. }
            | Self::MatMul { .. }
            | Self::MatVec { .. }
            | Self::Dot { .. }
            | Self::Solve { .. } => 2,
            Self::NaryAdd { terms } => terms.len(),
            Self::NaryMul { factors } => factors.len(),
            Self::Vector { elements } | Self::Matrix { elements, .. } => elements.len(),
        }
    }

    fn child_at(&self, index: usize) -> ExprId {
        match self {
            Self::Unary { input, .. }
            | Self::Component { input, .. }
            | Self::MatrixElement { input, .. } => *input,
            Self::Binary { lhs, rhs, .. }
            | Self::Complex { re: lhs, im: rhs }
            | Self::MatMul { lhs, rhs }
            | Self::Dot { lhs, rhs } => [*lhs, *rhs][index],
            Self::MatVec { matrix, vector } => [*matrix, *vector][index],
            Self::Solve { matrix, rhs } => [*matrix, *rhs][index],
            Self::NaryAdd { terms } => terms[index],
            Self::NaryMul { factors } => factors[index],
            Self::Vector { elements } | Self::Matrix { elements, .. } => elements[index],
            Self::RealConst(_)
            | Self::ComplexConst(_)
            | Self::ScalarParam(_)
            | Self::EventScalar(_)
            | Self::EventP4Component { .. } => unreachable!("leaf node has no children"),
        }
    }
}

/// Broad origin category recorded in [`ExprMetadata`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExprSourceKind {
    /// Constant literal.
    Const,
    /// Fit parameter.
    Param,
    /// Event data.
    Event,
    /// Unary operation.
    Unary,
    /// Binary or n-ary operation.
    Binary,
    /// Complex-number construction.
    Complex,
    /// Vector construction or selection.
    Vector,
    /// Matrix construction or selection.
    Matrix,
    /// Linear-algebra operation.
    LinearAlgebra,
}

/// User-facing annotations and origin information for an expression node.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExprMetadata {
    source: ExprSourceKind,
    name: Option<Arc<str>>,
    tags: Vec<Arc<str>>,
}

impl ExprMetadata {
    /// Creates metadata for the given source category.
    pub fn new(source: ExprSourceKind) -> Self {
        Self {
            source,
            name: None,
            tags: Vec::new(),
        }
    }

    /// Returns the node's source category.
    pub fn source(&self) -> ExprSourceKind {
        self.source
    }

    /// Returns the optional user-assigned name.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Returns the user-assigned tags.
    pub fn tags(&self) -> &[Arc<str>] {
        &self.tags
    }

    /// Returns whether the metadata contains `tag`.
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.iter().any(|candidate| candidate.as_ref() == tag)
    }
}

/// Shareable symbolic expression represented internally as a directed acyclic graph.
#[derive(Clone, Debug)]
pub struct Expr {
    node: Arc<DagNode>,
}

#[derive(Clone, Debug)]
struct DagNode {
    kind: DagNodeKind,
    metadata: ExprMetadata,
    shape: OnceLock<Result<ExprShape, ExprShapeError>>,
}

#[derive(Clone, Debug)]
enum DagNodeKind {
    RealConst(f64),
    ComplexConst(Complex64),
    ScalarParam(Parameter),
    EventScalar(Arc<str>),
    EventP4Component {
        name: Arc<str>,
        component: P4Component,
    },
    Unary {
        op: UnaryOp,
        input: Expr,
    },
    Binary {
        op: BinaryOp,
        lhs: Expr,
        rhs: Expr,
    },
    Complex {
        re: Expr,
        im: Expr,
    },
    Vector {
        elements: Vec<Expr>,
    },
    Matrix {
        rows: usize,
        cols: usize,
        elements: Vec<Expr>,
    },
    Component {
        input: Expr,
        index: usize,
    },
    MatrixElement {
        input: Expr,
        row: usize,
        col: usize,
    },
    MatMul {
        lhs: Expr,
        rhs: Expr,
    },
    MatVec {
        matrix: Expr,
        vector: Expr,
    },
    Dot {
        lhs: Expr,
        rhs: Expr,
    },
    Solve {
        matrix: Expr,
        rhs: Expr,
    },
}

impl Expr {
    fn new(kind: DagNodeKind) -> Self {
        let source = source_kind(&kind);
        Self {
            node: Arc::new(DagNode {
                kind,
                metadata: ExprMetadata::new(source),
                shape: OnceLock::new(),
            }),
        }
    }

    /// Assigns a display name to the expression root.
    pub fn named(self, name: impl Into<Arc<str>>) -> Self {
        self.with_metadata(|metadata| metadata.name = Some(name.into()))
    }

    /// Adds a tag to the expression root.
    pub fn tagged(self, tag: impl Into<Arc<str>>) -> Self {
        let tag = tag.into();
        self.with_metadata(|metadata| {
            if !metadata.tags.iter().any(|existing| existing == &tag) {
                metadata.tags.push(tag);
            }
        })
    }

    /// Adds each supplied tag to the expression root.
    pub fn tagged_with(self, tags: impl IntoIterator<Item = impl Into<Arc<str>>>) -> Self {
        tags.into_iter().fold(self, Self::tagged)
    }

    /// Replace tagged components that do not match any requested tag with zero.
    ///
    /// Untagged nodes remain active, while a matching tagged node retains its complete subtree.
    pub fn project_tags<'a>(&self, tags: impl IntoIterator<Item = &'a str>) -> Self {
        let tags: Vec<_> = tags.into_iter().collect();
        self.project_tags_inner(&tags)
    }

    fn project_tags_inner(&self, tags: &[&str]) -> Self {
        if !self.node.metadata.tags.is_empty() {
            return if self
                .node
                .metadata
                .tags
                .iter()
                .any(|candidate| tags.contains(&candidate.as_ref()))
            {
                self.clone()
            } else {
                self.zero_like()
            };
        }

        let kind = match &self.node.kind {
            DagNodeKind::RealConst(_)
            | DagNodeKind::ComplexConst(_)
            | DagNodeKind::ScalarParam(_)
            | DagNodeKind::EventScalar(_)
            | DagNodeKind::EventP4Component { .. } => return self.clone(),
            DagNodeKind::Unary { op, input } => DagNodeKind::Unary {
                op: *op,
                input: input.project_tags_inner(tags),
            },
            DagNodeKind::Binary { op, lhs, rhs } => DagNodeKind::Binary {
                op: *op,
                lhs: lhs.project_tags_inner(tags),
                rhs: rhs.project_tags_inner(tags),
            },
            DagNodeKind::Complex { re, im } => DagNodeKind::Complex {
                re: re.project_tags_inner(tags),
                im: im.project_tags_inner(tags),
            },
            DagNodeKind::Vector { elements } => DagNodeKind::Vector {
                elements: elements
                    .iter()
                    .map(|value| value.project_tags_inner(tags))
                    .collect(),
            },
            DagNodeKind::Matrix {
                rows,
                cols,
                elements,
            } => DagNodeKind::Matrix {
                rows: *rows,
                cols: *cols,
                elements: elements
                    .iter()
                    .map(|value| value.project_tags_inner(tags))
                    .collect(),
            },
            DagNodeKind::Component { input, index } => DagNodeKind::Component {
                input: input.project_tags_inner(tags),
                index: *index,
            },
            DagNodeKind::MatrixElement { input, row, col } => DagNodeKind::MatrixElement {
                input: input.project_tags_inner(tags),
                row: *row,
                col: *col,
            },
            DagNodeKind::MatMul { lhs, rhs } => DagNodeKind::MatMul {
                lhs: lhs.project_tags_inner(tags),
                rhs: rhs.project_tags_inner(tags),
            },
            DagNodeKind::MatVec { matrix, vector } => DagNodeKind::MatVec {
                matrix: matrix.project_tags_inner(tags),
                vector: vector.project_tags_inner(tags),
            },
            DagNodeKind::Dot { lhs, rhs } => DagNodeKind::Dot {
                lhs: lhs.project_tags_inner(tags),
                rhs: rhs.project_tags_inner(tags),
            },
            DagNodeKind::Solve { matrix, rhs } => DagNodeKind::Solve {
                matrix: matrix.project_tags_inner(tags),
                rhs: rhs.project_tags_inner(tags),
            },
        };
        Expr::new(kind).with_metadata(|metadata| *metadata = self.node.metadata.clone())
    }

    fn zero_like(&self) -> Self {
        match self
            .shape()
            .expect("valid expression shapes are cached eagerly")
        {
            ExprShape::Scalar => Expr::from(0.0),
            ExprShape::Vector { len } => vector((0..len).map(|_| Expr::from(0.0))),
            ExprShape::Matrix { rows, cols } => {
                matrix_from_flat(rows, cols, (0..rows * cols).map(|_| Expr::from(0.0)))
                    .expect("zero matrix dimensions match")
            }
        }
    }

    /// Returns an expression for the real part.
    pub fn real(&self) -> Self {
        unary(UnaryOp::Real, self)
    }

    /// Returns an expression for the imaginary part.
    pub fn imag(&self) -> Self {
        unary(UnaryOp::Imag, self)
    }

    /// Returns an expression for the complex conjugate.
    pub fn conj(&self) -> Self {
        unary(UnaryOp::Conj, self)
    }

    /// Returns an expression for the squared complex norm.
    pub fn norm_sqr(&self) -> Self {
        unary(UnaryOp::NormSqr, self)
    }

    /// Returns an expression for the principal square root.
    pub fn sqrt(&self) -> Self {
        unary(UnaryOp::Sqrt, self)
    }

    /// Returns an expression for the exponential.
    pub fn exp(&self) -> Self {
        unary(UnaryOp::Exp, self)
    }

    /// Returns an expression for the sine.
    pub fn sin(&self) -> Self {
        unary(UnaryOp::Sin, self)
    }

    /// Returns an expression for the cosine.
    pub fn cos(&self) -> Self {
        unary(UnaryOp::Cos, self)
    }

    /// Returns an expression for the principal arccosine.
    pub fn acos(&self) -> Self {
        atan2((Expr::from(1.0) - self.powi(2)).sqrt(), self)
    }

    /// Returns an expression for the natural logarithm.
    pub fn log(&self) -> Self {
        unary(UnaryOp::Log, self)
    }

    /// Returns an expression raised to an integer power.
    pub fn powi(&self, power: i32) -> Self {
        unary(UnaryOp::PowI(power), self)
    }

    /// Selects a component from a vector-valued expression.
    pub fn component(&self, index: impl ComponentIndex) -> Self {
        Expr::new(DagNodeKind::Component {
            input: self.clone(),
            index: index.component_index(),
        })
    }

    /// Selects an element from a matrix-valued expression.
    pub fn matrix_element(&self, row: usize, col: usize) -> Self {
        Expr::new(DagNodeKind::MatrixElement {
            input: self.clone(),
            row,
            col,
        })
    }

    /// Serializes the shareable expression DAG into a topologically ordered graph.
    pub fn to_graph(&self) -> ExprGraph {
        GraphBuilder::new().build(self)
    }

    /// Rebuilds a shareable expression DAG from its serialized graph form.
    ///
    /// # Errors
    ///
    /// Returns [`ExprGraphError`] when the graph is empty, its root or a child
    /// identifier is invalid, its metadata length does not match its node
    /// count, or its nodes are not topologically ordered.
    pub fn from_graph(graph: ExprGraph) -> Result<Self, ExprGraphError> {
        let ExprGraph {
            root,
            nodes,
            metadata,
        } = graph;
        let graph = ExprGraph::from_parts(root, nodes, metadata)?;
        let mut expressions: Vec<Expr> = Vec::with_capacity(graph.nodes.len());
        for (index, node) in graph.nodes.iter().enumerate() {
            let child = |id: ExprId| expressions[id.index()].clone();
            let expression = match node {
                ExprNode::RealConst(value) => Expr::new(DagNodeKind::RealConst(*value)),
                ExprNode::ComplexConst(value) => Expr::new(DagNodeKind::ComplexConst(*value)),
                ExprNode::ScalarParam(parameter) => {
                    Expr::new(DagNodeKind::ScalarParam(parameter.clone()))
                }
                ExprNode::EventScalar(name) => {
                    Expr::new(DagNodeKind::EventScalar(Arc::clone(name)))
                }
                ExprNode::EventP4Component { name, component } => {
                    Expr::new(DagNodeKind::EventP4Component {
                        name: Arc::clone(name),
                        component: *component,
                    })
                }
                ExprNode::Unary { op, input } => Expr::new(DagNodeKind::Unary {
                    op: *op,
                    input: child(*input),
                }),
                ExprNode::Binary { op, lhs, rhs } => Expr::new(DagNodeKind::Binary {
                    op: *op,
                    lhs: child(*lhs),
                    rhs: child(*rhs),
                }),
                ExprNode::NaryAdd { terms } => terms
                    .iter()
                    .map(|id| child(*id))
                    .reduce(|lhs, rhs| binary(BinaryOp::Add, &lhs, &rhs))
                    .unwrap_or_else(|| Expr::from(0.0)),
                ExprNode::NaryMul { factors } => factors
                    .iter()
                    .map(|id| child(*id))
                    .reduce(|lhs, rhs| binary(BinaryOp::Mul, &lhs, &rhs))
                    .unwrap_or_else(|| Expr::from(1.0)),
                ExprNode::Complex { re, im } => Expr::new(DagNodeKind::Complex {
                    re: child(*re),
                    im: child(*im),
                }),
                ExprNode::Vector { elements } => Expr::new(DagNodeKind::Vector {
                    elements: elements.iter().map(|id| child(*id)).collect(),
                }),
                ExprNode::Matrix {
                    rows,
                    cols,
                    elements,
                } => Expr::new(DagNodeKind::Matrix {
                    rows: *rows,
                    cols: *cols,
                    elements: elements.iter().map(|id| child(*id)).collect(),
                }),
                ExprNode::Component { input, index } => Expr::new(DagNodeKind::Component {
                    input: child(*input),
                    index: *index,
                }),
                ExprNode::MatrixElement { input, row, col } => {
                    Expr::new(DagNodeKind::MatrixElement {
                        input: child(*input),
                        row: *row,
                        col: *col,
                    })
                }
                ExprNode::MatMul { lhs, rhs } => Expr::new(DagNodeKind::MatMul {
                    lhs: child(*lhs),
                    rhs: child(*rhs),
                }),
                ExprNode::MatVec { matrix, vector } => Expr::new(DagNodeKind::MatVec {
                    matrix: child(*matrix),
                    vector: child(*vector),
                }),
                ExprNode::Dot { lhs, rhs } => Expr::new(DagNodeKind::Dot {
                    lhs: child(*lhs),
                    rhs: child(*rhs),
                }),
                ExprNode::Solve { matrix, rhs } => Expr::new(DagNodeKind::Solve {
                    matrix: child(*matrix),
                    rhs: child(*rhs),
                }),
            };
            let mut dag = (*expression.node).clone();
            dag.metadata = graph.metadata[index].clone();
            expressions.push(Expr {
                node: Arc::new(dag),
            });
        }
        Ok(expressions[graph.root.index()].clone())
    }

    /// Determines and validates the expression's structural shape.
    ///
    /// # Errors
    ///
    /// Returns [`ExprShapeError`] when this expression contains an operation
    /// whose operand shapes are incompatible.
    pub fn shape(&self) -> Result<ExprShape, ExprShapeError> {
        self.node
            .shape
            .get_or_init(|| self.node.kind.shape())
            .clone()
    }

    fn with_metadata(self, f: impl FnOnce(&mut ExprMetadata)) -> Self {
        let mut node = (*self.node).clone();
        f(&mut node.metadata);
        Self {
            node: Arc::new(node),
        }
    }
}

impl Serialize for Expr {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.to_graph().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Expr {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Expr::from_graph(ExprGraph::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}

impl DagNodeKind {
    fn shape(&self) -> Result<ExprShape, ExprShapeError> {
        match self {
            Self::RealConst(_)
            | Self::ComplexConst(_)
            | Self::ScalarParam(_)
            | Self::EventScalar(_)
            | Self::EventP4Component { .. } => Ok(ExprShape::Scalar),
            Self::Unary { input, .. } => {
                input.expect_shape("unary operation", ExprShape::Scalar)?;
                Ok(ExprShape::Scalar)
            }
            Self::Binary { lhs, rhs, .. } => {
                lhs.expect_shape("binary operation", ExprShape::Scalar)?;
                rhs.expect_shape("binary operation", ExprShape::Scalar)?;
                Ok(ExprShape::Scalar)
            }
            Self::Complex { re, im } => {
                re.expect_shape("complex constructor", ExprShape::Scalar)?;
                im.expect_shape("complex constructor", ExprShape::Scalar)?;
                Ok(ExprShape::Scalar)
            }
            Self::Vector { elements } => {
                for element in elements {
                    element.expect_shape("vector constructor", ExprShape::Scalar)?;
                }
                Ok(ExprShape::Vector {
                    len: elements.len(),
                })
            }
            Self::Matrix {
                rows,
                cols,
                elements,
            } => {
                let expected = rows.checked_mul(*cols).ok_or_else(|| {
                    ExprShapeError::new("matrix constructor", "row/column product overflowed")
                })?;
                if elements.len() != expected {
                    return Err(ExprShapeError::new(
                        "matrix constructor",
                        format!(
                            "shape {rows}x{cols} requires {expected} elements, got {}",
                            elements.len()
                        ),
                    ));
                }
                for element in elements {
                    element.expect_shape("matrix constructor", ExprShape::Scalar)?;
                }
                Ok(ExprShape::Matrix {
                    rows: *rows,
                    cols: *cols,
                })
            }
            Self::Component { input, index } => {
                let ExprShape::Vector { len } = input.shape()? else {
                    return Err(ExprShapeError::new(
                        "component",
                        format!("expected vector, got {}", input.shape()?),
                    ));
                };
                if *index >= len {
                    return Err(ExprShapeError::new(
                        "component",
                        format!("index {index} is out of bounds for vector[{len}]"),
                    ));
                }
                Ok(ExprShape::Scalar)
            }
            Self::MatrixElement { input, row, col } => {
                let ExprShape::Matrix { rows, cols } = input.shape()? else {
                    return Err(ExprShapeError::new(
                        "matrix element",
                        format!("expected matrix, got {}", input.shape()?),
                    ));
                };
                if *row >= rows || *col >= cols {
                    return Err(ExprShapeError::new(
                        "matrix element",
                        format!("index ({row}, {col}) is out of bounds for matrix[{rows}x{cols}]"),
                    ));
                }
                Ok(ExprShape::Scalar)
            }
            Self::MatMul { lhs, rhs } => {
                let ExprShape::Matrix {
                    rows: lhs_rows,
                    cols: lhs_cols,
                } = lhs.shape()?
                else {
                    return Err(ExprShapeError::new(
                        "matrix multiplication",
                        format!("left input must be a matrix, got {}", lhs.shape()?),
                    ));
                };
                let ExprShape::Matrix {
                    rows: rhs_rows,
                    cols: rhs_cols,
                } = rhs.shape()?
                else {
                    return Err(ExprShapeError::new(
                        "matrix multiplication",
                        format!("right input must be a matrix, got {}", rhs.shape()?),
                    ));
                };
                if lhs_cols != rhs_rows {
                    return Err(ExprShapeError::new(
                        "matrix multiplication",
                        format!("cannot multiply {lhs_rows}x{lhs_cols} by {rhs_rows}x{rhs_cols}"),
                    ));
                }
                Ok(ExprShape::Matrix {
                    rows: lhs_rows,
                    cols: rhs_cols,
                })
            }
            Self::MatVec { matrix, vector } => {
                let ExprShape::Matrix { rows, cols } = matrix.shape()? else {
                    return Err(ExprShapeError::new(
                        "matrix-vector multiplication",
                        format!("left input must be a matrix, got {}", matrix.shape()?),
                    ));
                };
                let ExprShape::Vector { len } = vector.shape()? else {
                    return Err(ExprShapeError::new(
                        "matrix-vector multiplication",
                        format!("right input must be a vector, got {}", vector.shape()?),
                    ));
                };
                if cols != len {
                    return Err(ExprShapeError::new(
                        "matrix-vector multiplication",
                        format!("cannot multiply {rows}x{cols} matrix by vector[{len}]"),
                    ));
                }
                Ok(ExprShape::Vector { len: rows })
            }
            Self::Dot { lhs, rhs } => {
                let ExprShape::Vector { len: lhs_len } = lhs.shape()? else {
                    return Err(ExprShapeError::new(
                        "dot product",
                        format!("left input must be a vector, got {}", lhs.shape()?),
                    ));
                };
                let ExprShape::Vector { len: rhs_len } = rhs.shape()? else {
                    return Err(ExprShapeError::new(
                        "dot product",
                        format!("right input must be a vector, got {}", rhs.shape()?),
                    ));
                };
                if lhs_len != rhs_len {
                    return Err(ExprShapeError::new(
                        "dot product",
                        format!("vector lengths differ: {lhs_len} and {rhs_len}"),
                    ));
                }
                Ok(ExprShape::Scalar)
            }
            Self::Solve { matrix, rhs } => {
                let ExprShape::Matrix { rows, cols } = matrix.shape()? else {
                    return Err(ExprShapeError::new(
                        "linear solve",
                        format!("left input must be a matrix, got {}", matrix.shape()?),
                    ));
                };
                let ExprShape::Vector { len } = rhs.shape()? else {
                    return Err(ExprShapeError::new(
                        "linear solve",
                        format!("right input must be a vector, got {}", rhs.shape()?),
                    ));
                };
                if rows != cols || rows != len {
                    return Err(ExprShapeError::new(
                        "linear solve",
                        format!("cannot solve matrix[{rows}x{cols}] against vector[{len}]"),
                    ));
                }
                Ok(ExprShape::Vector { len })
            }
        }
    }
}

impl Expr {
    fn expect_shape(
        &self,
        operation: &'static str,
        expected: ExprShape,
    ) -> Result<(), ExprShapeError> {
        let actual = self.shape()?;
        if actual != expected {
            return Err(ExprShapeError::new(
                operation,
                format!("expected {expected}, got {actual}"),
            ));
        }
        Ok(())
    }
}

auto_ops::impl_op_ex!(+ |a: &Expr, b: &Expr| -> Expr { binary(BinaryOp::Add, a, b) });

auto_ops::impl_op_ex!(+ |a: &Expr, b: &f64| -> Expr { binary(BinaryOp::Add, a, b) });
auto_ops::impl_op_ex!(+ |a: &f64, b: &Expr| -> Expr { binary(BinaryOp::Add, a, b) });

auto_ops::impl_op_ex!(+ |a: &Expr, b: &Complex64| -> Expr { binary(BinaryOp::Add, a, b) });
auto_ops::impl_op_ex!(+ |a: &Complex64, b: &Expr| -> Expr { binary(BinaryOp::Add, a, b) });

auto_ops::impl_op_ex!(+ |a: &Expr, b: &Parameter| -> Expr { binary(BinaryOp::Add, a, b) });
auto_ops::impl_op_ex!(+ |a: &Parameter, b: &Expr| -> Expr { binary(BinaryOp::Add, a, b) });

auto_ops::impl_op_ex!(+ |a: &Parameter, b: &f64| -> Expr { binary(BinaryOp::Add, a, b) });
auto_ops::impl_op_ex!(+ |a: &f64, b: &Parameter| -> Expr { binary(BinaryOp::Add, a, b) });

auto_ops::impl_op_ex!(+ |a: &Parameter, b: &Complex64| -> Expr { binary(BinaryOp::Add, a, b) });
auto_ops::impl_op_ex!(+ |a: &Complex64, b: &Parameter| -> Expr { binary(BinaryOp::Add, a, b) });

auto_ops::impl_op_ex!(+ |a: &Parameter, b: &Parameter| -> Expr { binary(BinaryOp::Add, a, b) });

auto_ops::impl_op_ex!(-|a: &Expr, b: &Expr| -> Expr { binary(BinaryOp::Sub, a, b) });

auto_ops::impl_op_ex!(-|a: &Expr, b: &f64| -> Expr { binary(BinaryOp::Sub, a, b) });

auto_ops::impl_op_ex!(-|a: &f64, b: &Expr| -> Expr { binary(BinaryOp::Sub, a, b) });

auto_ops::impl_op_ex!(-|a: &Expr, b: &Complex64| -> Expr { binary(BinaryOp::Sub, a, b) });

auto_ops::impl_op_ex!(-|a: &Complex64, b: &Expr| -> Expr { binary(BinaryOp::Sub, a, b) });

auto_ops::impl_op_ex!(-|a: &Expr, b: &Parameter| -> Expr { binary(BinaryOp::Sub, a, b) });

auto_ops::impl_op_ex!(-|a: &Parameter, b: &Expr| -> Expr { binary(BinaryOp::Sub, a, b) });

auto_ops::impl_op_ex!(-|a: &f64, b: &Parameter| -> Expr { binary(BinaryOp::Sub, a, b) });

auto_ops::impl_op_ex!(-|a: &Parameter, b: &f64| -> Expr { binary(BinaryOp::Sub, a, b) });

auto_ops::impl_op_ex!(-|a: &Complex64, b: &Parameter| -> Expr { binary(BinaryOp::Sub, a, b) });

auto_ops::impl_op_ex!(-|a: &Parameter, b: &Complex64| -> Expr { binary(BinaryOp::Sub, a, b) });

auto_ops::impl_op_ex!(-|a: &Parameter, b: &Parameter| -> Expr { binary(BinaryOp::Sub, a, b) });

auto_ops::impl_op_ex!(*|a: &Expr, b: &Expr| -> Expr { binary(BinaryOp::Mul, a, b) });

auto_ops::impl_op_ex!(*|a: &Expr, b: &f64| -> Expr { binary(BinaryOp::Mul, a, b) });
auto_ops::impl_op_ex!(*|a: &f64, b: &Expr| -> Expr { binary(BinaryOp::Mul, a, b) });

auto_ops::impl_op_ex!(*|a: &Expr, b: &Complex64| -> Expr { binary(BinaryOp::Mul, a, b) });
auto_ops::impl_op_ex!(*|a: &Complex64, b: &Expr| -> Expr { binary(BinaryOp::Mul, a, b) });

auto_ops::impl_op_ex!(*|a: &Expr, b: &Parameter| -> Expr { binary(BinaryOp::Mul, a, b) });
auto_ops::impl_op_ex!(*|a: &Parameter, b: &Expr| -> Expr { binary(BinaryOp::Mul, a, b) });

auto_ops::impl_op_ex!(*|a: &f64, b: &Parameter| -> Expr { binary(BinaryOp::Mul, a, b) });
auto_ops::impl_op_ex!(*|a: &Parameter, b: &f64| -> Expr { binary(BinaryOp::Mul, a, b) });

auto_ops::impl_op_ex!(*|a: &Complex64, b: &Parameter| -> Expr { binary(BinaryOp::Mul, a, b) });
auto_ops::impl_op_ex!(*|a: &Parameter, b: &Complex64| -> Expr { binary(BinaryOp::Mul, a, b) });

auto_ops::impl_op_ex!(*|a: &Parameter, b: &Parameter| -> Expr { binary(BinaryOp::Mul, a, b) });

auto_ops::impl_op_ex!(/ |a: &Expr, b: &Expr| -> Expr {
    binary(BinaryOp::Div, a, b)
});

auto_ops::impl_op_ex!(/ |a: &Expr, b: &Complex64| -> Expr { binary(BinaryOp::Div, a, b) });
auto_ops::impl_op_ex!(/ |a: &Complex64, b: &Expr| -> Expr { binary(BinaryOp::Div, a, b) });

auto_ops::impl_op_ex!(/ |a: &Expr, b: &f64| -> Expr { binary(BinaryOp::Div, a, b) });
auto_ops::impl_op_ex!(/ |a: &f64, b: &Expr| -> Expr { binary(BinaryOp::Div, a, b) });

auto_ops::impl_op_ex!(/|a: &Expr, b: &Parameter| -> Expr {
    binary(BinaryOp::Div, a, b)
});
auto_ops::impl_op_ex!(/|a: &Parameter, b: &Expr| -> Expr {
    binary(BinaryOp::Div, a, b)
});

auto_ops::impl_op_ex!(/|a: &f64, b: &Parameter| -> Expr {
    binary(BinaryOp::Div, a, b)
});
auto_ops::impl_op_ex!(/|a: &Parameter, b: &f64| -> Expr {
    binary(BinaryOp::Div, a, b)
});

auto_ops::impl_op_ex!(/|a: &Complex64, b: &Parameter| -> Expr {
    binary(BinaryOp::Div, a, b)
});
auto_ops::impl_op_ex!(/|a: &Parameter, b: &Complex64| -> Expr {
    binary(BinaryOp::Div, a, b)
});

auto_ops::impl_op_ex!(/|a: &Parameter, b: &Parameter| -> Expr {
    binary(BinaryOp::Div, a, b)
});

auto_ops::impl_op_ex!(-|a: &Expr| -> Expr { unary(UnaryOp::Neg, a) });
auto_ops::impl_op_ex!(-|a: &Parameter| -> Expr { unary(UnaryOp::Neg, a) });

auto_ops::impl_op_ex!(+= |a: &mut Expr, b: &Expr| {
    *a = binary(BinaryOp::Add, &*a, b);
});
auto_ops::impl_op_ex!(+= |a: &mut Expr, b: &f64| {
    *a = binary(BinaryOp::Add, &*a, b);
});
auto_ops::impl_op_ex!(+= |a: &mut Expr, b: &Complex64| {
    *a = binary(BinaryOp::Add, &*a, b);
});
auto_ops::impl_op_ex!(+= |a: &mut Expr, b: &Parameter| {
    *a = binary(BinaryOp::Add, &*a, b);
});

auto_ops::impl_op_ex!(-= |a: &mut Expr, b: &Expr| {
    *a = binary(BinaryOp::Sub, &*a, b);
});
auto_ops::impl_op_ex!(-= |a: &mut Expr, b: &f64| {
    *a = binary(BinaryOp::Sub, &*a, b);
});
auto_ops::impl_op_ex!(-= |a: &mut Expr, b: &Complex64| {
    *a = binary(BinaryOp::Sub, &*a, b);
});
auto_ops::impl_op_ex!(-= |a: &mut Expr, b: &Parameter| {
    *a = binary(BinaryOp::Sub, &*a, b);
});

auto_ops::impl_op_ex!(*= |a: &mut Expr, b: &Expr| {
    *a = binary(BinaryOp::Mul, &*a, b);
});
auto_ops::impl_op_ex!(*= |a: &mut Expr, b: &f64| {
    *a = binary(BinaryOp::Mul, &*a, b);
});
auto_ops::impl_op_ex!(*= |a: &mut Expr, b: &Complex64| {
    *a = binary(BinaryOp::Mul, &*a, b);
});
auto_ops::impl_op_ex!(*= |a: &mut Expr, b: &Parameter| {
    *a = binary(BinaryOp::Mul, &*a, b);
});

auto_ops::impl_op_ex!(/= |a: &mut Expr, b: &Expr| {
    *a = binary(BinaryOp::Div, &*a, b);
});
auto_ops::impl_op_ex!(/= |a: &mut Expr, b: &f64| {
    *a = binary(BinaryOp::Div, &*a, b);
});
auto_ops::impl_op_ex!(/= |a: &mut Expr, b: &Complex64| {
    *a = binary(BinaryOp::Div, &*a, b);
});
auto_ops::impl_op_ex!(/= |a: &mut Expr, b: &Parameter| {
    *a = binary(BinaryOp::Div, &*a, b);
});

impl From<f64> for Expr {
    fn from(value: f64) -> Self {
        Self::new(DagNodeKind::RealConst(value))
    }
}

impl From<&f64> for Expr {
    fn from(value: &f64) -> Self {
        Self::new(DagNodeKind::RealConst(*value))
    }
}

impl From<Complex64> for Expr {
    fn from(value: Complex64) -> Self {
        Self::new(DagNodeKind::ComplexConst(value))
    }
}

impl From<&Complex64> for Expr {
    fn from(value: &Complex64) -> Self {
        Self::new(DagNodeKind::ComplexConst(*value))
    }
}

impl From<&Expr> for Expr {
    fn from(value: &Expr) -> Self {
        value.clone()
    }
}

impl From<Parameter> for Expr {
    fn from(parameter: Parameter) -> Self {
        Expr::new(DagNodeKind::ScalarParam(parameter))
    }
}

impl From<&Parameter> for Expr {
    fn from(parameter: &Parameter) -> Self {
        parameter.clone().into()
    }
}

/// Constructs `cos(phase) + i sin(phase)`.
pub fn cis(phase: Expr) -> Expr {
    phase.cos() + Complex64::I * phase.sin()
}

/// Constructs a complex scalar from real and imaginary expressions.
pub fn complex(re: impl Into<Expr>, im: impl Into<Expr>) -> Expr {
    Expr::new(DagNodeKind::Complex {
        re: re.into(),
        im: im.into(),
    })
}

/// Constructs a complex scalar from magnitude and phase expressions.
pub fn polar_complex(mag: impl Into<Expr>, phase: impl Into<Expr>) -> Expr {
    mag.into() * (Complex64::I * phase.into()).exp()
}

/// References a named scalar column in each event.
pub fn event_scalar(name: impl Into<Arc<str>>) -> Expr {
    Expr::new(DagNodeKind::EventScalar(name.into()))
}

/// References one component of a named event four-momentum.
pub fn event_p4_component(name: impl Into<Arc<str>>, component: P4Component) -> Expr {
    Expr::new(DagNodeKind::EventP4Component {
        name: name.into(),
        component,
    })
}

/// Constructs the two-argument arctangent `atan2(y, x)`.
pub fn atan2(y: impl Into<Expr>, x: impl Into<Expr>) -> Expr {
    binary(BinaryOp::Atan2, y, x)
}

/// Constructs the principal arccosine of `value`.
pub fn acos(value: impl Into<Expr>) -> Expr {
    value.into().acos()
}

/// Constructs a vector expression from scalar elements.
pub fn vector<E>(elements: impl IntoIterator<Item = E>) -> Expr
where
    E: Into<Expr>,
    Expr: From<E>,
{
    Expr::new(DagNodeKind::Vector {
        elements: elements.into_iter().map(Expr::from).collect(),
    })
}

/// Constructs a row-major matrix expression from a nested array.
pub fn matrix<const R: usize, const C: usize, E>(elements: [[E; C]; R]) -> Expr
where
    E: Into<Expr>,
    Expr: From<E>,
{
    Expr::new(DagNodeKind::Matrix {
        rows: R,
        cols: C,
        elements: elements.into_iter().flatten().map(Expr::from).collect(),
    })
}

/// Constructs a row-major matrix from a flat sequence.
///
/// # Errors
///
/// Returns [`ExprShapeError`] when either dimension is zero, the dimension
/// product overflows, the element count differs from `rows * cols`, or an
/// element is not scalar-valued.
pub fn matrix_from_flat<E>(
    rows: usize,
    cols: usize,
    elements: impl IntoIterator<Item = E>,
) -> Result<Expr, ExprShapeError>
where
    E: Into<Expr>,
    Expr: From<E>,
{
    if rows == 0 || cols == 0 {
        return Err(ExprShapeError::new(
            "matrix constructor",
            format!("matrix dimensions must be nonzero, got {rows}x{cols}"),
        ));
    }
    let expected = rows.checked_mul(cols).ok_or_else(|| {
        ExprShapeError::new("matrix constructor", "row/column product overflowed")
    })?;
    let elements = elements.into_iter().map(Expr::from).collect::<Vec<_>>();
    if elements.len() != expected {
        return Err(ExprShapeError::new(
            "matrix constructor",
            format!(
                "shape {rows}x{cols} requires {expected} elements, got {}",
                elements.len()
            ),
        ));
    }
    for element in &elements {
        element.expect_shape("matrix constructor", ExprShape::Scalar)?;
    }
    Ok(Expr::new(DagNodeKind::Matrix {
        rows,
        cols,
        elements,
    }))
}

/// Constructs a matrix-matrix multiplication expression.
pub fn matmul(lhs: impl Into<Expr>, rhs: impl Into<Expr>) -> Expr {
    Expr::new(DagNodeKind::MatMul {
        lhs: lhs.into(),
        rhs: rhs.into(),
    })
}

/// Constructs a matrix-vector multiplication expression.
pub fn matvec(matrix: impl Into<Expr>, vector: impl Into<Expr>) -> Expr {
    Expr::new(DagNodeKind::MatVec {
        matrix: matrix.into(),
        vector: vector.into(),
    })
}

/// Constructs a vector dot-product expression.
pub fn dot(lhs: impl Into<Expr>, rhs: impl Into<Expr>) -> Expr {
    Expr::new(DagNodeKind::Dot {
        lhs: lhs.into(),
        rhs: rhs.into(),
    })
}

/// Constructs an expression that solves a linear system.
pub fn solve(matrix: impl Into<Expr>, rhs: impl Into<Expr>) -> Expr {
    Expr::new(DagNodeKind::Solve {
        matrix: matrix.into(),
        rhs: rhs.into(),
    })
}

fn unary(op: UnaryOp, expr: impl Into<Expr>) -> Expr {
    Expr::new(DagNodeKind::Unary {
        op,
        input: expr.into(),
    })
}

fn binary(op: BinaryOp, lhs: impl Into<Expr>, rhs: impl Into<Expr>) -> Expr {
    Expr::new(DagNodeKind::Binary {
        op,
        lhs: lhs.into(),
        rhs: rhs.into(),
    })
}

/// Topologically ordered, serializable representation of an [`Expr`] DAG.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExprGraph {
    root: ExprId,
    nodes: Vec<ExprNode>,
    metadata: Vec<ExprMetadata>,
}

impl ExprGraph {
    /// Return a copy of this graph with the named scalar parameter fixed.
    ///
    /// # Errors
    ///
    /// Returns [`ParamError::UnknownName`] when the graph has no scalar
    /// parameter named `name`, or [`ParamError::FixedValueOutOfBounds`] when
    /// `value` is outside that parameter's bounds.
    pub fn fix_parameter(&self, name: &str, value: f64) -> ParamResult<Self> {
        self.map_parameter(name, |parameter| {
            if !parameter.bounds_spec().contains(value) {
                return Err(ParamError::FixedValueOutOfBounds {
                    name: name.to_owned(),
                    value,
                });
            }
            Ok(parameter.clone().with_fixed_value(value))
        })
    }

    /// Return a copy of this graph with the named scalar parameter free.
    ///
    /// # Errors
    ///
    /// Returns [`ParamError::UnknownName`] when the graph has no scalar
    /// parameter named `name`.
    pub fn free_parameter(&self, name: &str) -> ParamResult<Self> {
        self.map_parameter(name, |parameter| Ok(parameter.clone().with_free()))
    }

    fn map_parameter(
        &self,
        name: &str,
        mut map: impl FnMut(&Parameter) -> ParamResult<Parameter>,
    ) -> ParamResult<Self> {
        let mut found = false;
        let mut graph = self.clone();
        for node in &mut graph.nodes {
            if let ExprNode::ScalarParam(parameter) = node
                && parameter.name() == name
            {
                *parameter = map(parameter)?;
                found = true;
            }
        }
        if !found {
            return Err(ParamError::UnknownName(name.to_owned()));
        }
        Ok(graph)
    }

    /// Replaces tagged components that match none of `tags` with zero.
    ///
    /// Untagged nodes remain active, and a matching tagged node retains its
    /// entire subtree.
    pub fn project_tags<'a>(&self, tags: impl IntoIterator<Item = &'a str>) -> Self {
        let tags: Vec<_> = tags.into_iter().collect();
        let mut nodes = Vec::new();
        let mut metadata = Vec::new();
        let mut remapped = HashMap::new();
        let root = self.project_node(
            self.root,
            &tags,
            false,
            &mut nodes,
            &mut metadata,
            &mut remapped,
        );
        Self {
            root,
            nodes,
            metadata,
        }
    }

    fn project_node(
        &self,
        old: ExprId,
        tags: &[&str],
        retain_all: bool,
        nodes: &mut Vec<ExprNode>,
        metadata: &mut Vec<ExprMetadata>,
        remapped: &mut HashMap<(ExprId, bool), ExprId>,
    ) -> ExprId {
        if let Some(id) = remapped.get(&(old, retain_all)) {
            return *id;
        }
        let old_metadata = &self.metadata[old.index()];
        let matches = old_metadata
            .tags
            .iter()
            .any(|tag| tags.contains(&tag.as_ref()));
        let node = if !retain_all && !old_metadata.tags.is_empty() && !matches {
            ExprNode::RealConst(0.0)
        } else {
            let retain_children = retain_all || matches;
            self.nodes[old.index()].map_children(|child| {
                self.project_node(child, tags, retain_children, nodes, metadata, remapped)
            })
        };
        let id = ExprId::from_index(nodes.len());
        nodes.push(node);
        metadata.push(if matches || retain_all {
            old_metadata.clone()
        } else {
            ExprMetadata::new(old_metadata.source)
        });
        remapped.insert((old, retain_all), id);
        id
    }

    /// Validates and constructs a graph from its serialized parts.
    ///
    /// Child nodes must precede their parents, metadata must have one entry per
    /// node, and `root` must identify an existing node.
    ///
    /// # Errors
    ///
    /// Returns [`ExprGraphError`] when the graph is empty, the metadata and
    /// node lengths differ, `root` is invalid, or a child identifier is
    /// invalid or does not precede its parent.
    pub fn from_parts(
        root: ExprId,
        nodes: Vec<ExprNode>,
        metadata: Vec<ExprMetadata>,
    ) -> Result<Self, ExprGraphError> {
        if nodes.is_empty() {
            return Err(ExprGraphError::Empty);
        }
        if nodes.len() != metadata.len() {
            return Err(ExprGraphError::MetadataLength {
                node_len: nodes.len(),
                metadata_len: metadata.len(),
            });
        }
        if root.index() >= nodes.len() {
            return Err(ExprGraphError::InvalidRoot {
                root: root.index(),
                node_len: nodes.len(),
            });
        }
        for (index, node) in nodes.iter().enumerate() {
            for child in node.children() {
                if child.index() >= nodes.len() {
                    return Err(ExprGraphError::InvalidChild {
                        node: index,
                        child: child.index(),
                    });
                }
                if child.index() >= index {
                    return Err(ExprGraphError::InvalidChildOrder {
                        node: index,
                        child: child.index(),
                    });
                }
            }
        }
        Ok(Self {
            root,
            nodes,
            metadata,
        })
    }

    /// Returns the root node identifier.
    pub fn root(&self) -> ExprId {
        self.root
    }

    /// Returns the node identified by `id`, if it exists.
    pub fn node(&self, id: ExprId) -> Option<&ExprNode> {
        self.nodes.get(id.index())
    }

    /// Returns all nodes in topological order.
    pub fn nodes(&self) -> &[ExprNode] {
        &self.nodes
    }

    /// Returns the metadata associated with `id`, if it exists.
    pub fn metadata(&self, id: ExprId) -> Option<&ExprMetadata> {
        self.metadata.get(id.index())
    }
}

pub(crate) fn node_children(node: &ExprNode) -> Vec<(String, ExprId)> {
    node.children()
        .enumerate()
        .map(|(index, child)| (node_child_label(node, index), child))
        .collect()
}

fn node_child_label(node: &ExprNode, index: usize) -> String {
    match node {
        ExprNode::Unary { .. } | ExprNode::Component { .. } | ExprNode::MatrixElement { .. } => {
            "input".into()
        }
        ExprNode::Binary { .. } | ExprNode::MatMul { .. } | ExprNode::Dot { .. } => {
            if index == 0 { "lhs" } else { "rhs" }.into()
        }
        ExprNode::NaryAdd { .. } => format!("term[{index}]"),
        ExprNode::NaryMul { .. } => format!("factor[{index}]"),
        ExprNode::Complex { .. } => if index == 0 { "re" } else { "im" }.into(),
        ExprNode::Vector { .. } => format!("element[{index}]"),
        ExprNode::Matrix { cols, .. } => {
            format!("element[{},{}]", index / cols, index % cols)
        }
        ExprNode::MatVec { .. } => if index == 0 { "matrix" } else { "vector" }.into(),
        ExprNode::Solve { .. } => if index == 0 { "matrix" } else { "rhs" }.into(),
        ExprNode::RealConst(_)
        | ExprNode::ComplexConst(_)
        | ExprNode::ScalarParam(_)
        | ExprNode::EventScalar(_)
        | ExprNode::EventP4Component { .. } => unreachable!("leaf nodes have no child labels"),
    }
}

#[derive(Default)]
struct GraphBuilder {
    nodes: Vec<ExprNode>,
    metadata: Vec<ExprMetadata>,
    ids: HashMap<usize, ExprId>,
}

impl GraphBuilder {
    fn new() -> Self {
        Self::default()
    }

    fn build(mut self, expr: &Expr) -> ExprGraph {
        let root = self.visit(expr);
        ExprGraph {
            root,
            nodes: self.nodes,
            metadata: self.metadata,
        }
    }

    fn visit(&mut self, expr: &Expr) -> ExprId {
        let key = Arc::as_ptr(&expr.node) as usize;
        if let Some(id) = self.ids.get(&key) {
            return *id;
        }
        let node = match &expr.node.kind {
            DagNodeKind::RealConst(value) => ExprNode::RealConst(*value),
            DagNodeKind::ComplexConst(value) => ExprNode::ComplexConst(*value),
            DagNodeKind::ScalarParam(parameter) => ExprNode::ScalarParam(parameter.clone()),
            DagNodeKind::EventScalar(name) => ExprNode::EventScalar(Arc::clone(name)),
            DagNodeKind::EventP4Component { name, component } => ExprNode::EventP4Component {
                name: Arc::clone(name),
                component: *component,
            },
            DagNodeKind::Unary { op, input } => {
                let input = self.visit(input);
                ExprNode::Unary { op: *op, input }
            }
            DagNodeKind::Binary { op, lhs, rhs } => {
                let lhs = self.visit(lhs);
                let rhs = self.visit(rhs);
                ExprNode::Binary { op: *op, lhs, rhs }
            }
            DagNodeKind::Complex { re, im } => {
                let re = self.visit(re);
                let im = self.visit(im);
                ExprNode::Complex { re, im }
            }
            DagNodeKind::Vector { elements } => ExprNode::Vector {
                elements: elements.iter().map(|expr| self.visit(expr)).collect(),
            },
            DagNodeKind::Matrix {
                rows,
                cols,
                elements,
            } => ExprNode::Matrix {
                rows: *rows,
                cols: *cols,
                elements: elements.iter().map(|expr| self.visit(expr)).collect(),
            },
            DagNodeKind::Component { input, index } => {
                let input = self.visit(input);
                ExprNode::Component {
                    input,
                    index: *index,
                }
            }
            DagNodeKind::MatrixElement { input, row, col } => {
                let input = self.visit(input);
                ExprNode::MatrixElement {
                    input,
                    row: *row,
                    col: *col,
                }
            }
            DagNodeKind::MatMul { lhs, rhs } => {
                let lhs = self.visit(lhs);
                let rhs = self.visit(rhs);
                ExprNode::MatMul { lhs, rhs }
            }
            DagNodeKind::MatVec { matrix, vector } => {
                let matrix = self.visit(matrix);
                let vector = self.visit(vector);
                ExprNode::MatVec { matrix, vector }
            }
            DagNodeKind::Dot { lhs, rhs } => {
                let lhs = self.visit(lhs);
                let rhs = self.visit(rhs);
                ExprNode::Dot { lhs, rhs }
            }
            DagNodeKind::Solve { matrix, rhs } => {
                let matrix = self.visit(matrix);
                let rhs = self.visit(rhs);
                ExprNode::Solve { matrix, rhs }
            }
        };

        let id = ExprId::from_index(self.nodes.len());
        self.nodes.push(node);
        self.metadata.push(expr.node.metadata.clone());
        self.ids.insert(key, id);
        id
    }
}

fn source_kind(kind: &DagNodeKind) -> ExprSourceKind {
    match kind {
        DagNodeKind::RealConst(_) | DagNodeKind::ComplexConst(_) => ExprSourceKind::Const,
        DagNodeKind::ScalarParam(_) => ExprSourceKind::Param,
        DagNodeKind::EventScalar(_) | DagNodeKind::EventP4Component { .. } => ExprSourceKind::Event,
        DagNodeKind::Unary { .. } => ExprSourceKind::Unary,
        DagNodeKind::Binary { .. } => ExprSourceKind::Binary,
        DagNodeKind::Complex { .. } => ExprSourceKind::Complex,
        DagNodeKind::Vector { .. } | DagNodeKind::Component { .. } | DagNodeKind::Dot { .. } => {
            ExprSourceKind::Vector
        }
        DagNodeKind::Matrix { .. } | DagNodeKind::MatrixElement { .. } => ExprSourceKind::Matrix,
        DagNodeKind::MatMul { .. } | DagNodeKind::MatVec { .. } | DagNodeKind::Solve { .. } => {
            ExprSourceKind::LinearAlgebra
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parameter;

    #[test]
    fn builds_target_syntax_without_layout_or_context() {
        let model = (Complex64::I * parameter!("y", initial : 1.0, bounds : (0.0, 2.0))
            + parameter!("x"))
        .norm_sqr();

        let graph = model.to_graph();
        assert!(matches!(
            graph.node(graph.root()),
            Some(ExprNode::Unary {
                op: UnaryOp::NormSqr,
                ..
            })
        ));
    }

    #[test]
    fn parameter_nodes_store_specs_but_do_not_make_layouts() {
        let graph = Expr::from(parameter!("x", initial: 1.0)).to_graph();
        assert!(matches!(
            graph.node(graph.root()),
            Some(ExprNode::ScalarParam(spec)) if spec.name() == "x"
        ));
    }

    #[test]
    fn complex_constructor_builds_expression_node() {
        let graph = complex(parameter!("re"), parameter!("im")).to_graph();

        assert!(matches!(
            graph.node(graph.root()),
            Some(ExprNode::Complex { .. })
        ));
    }

    #[test]
    fn polar_complex_lowers_to_expression_graph() {
        let graph = polar_complex(parameter!("mag"), parameter!("phase")).to_graph();

        assert!(graph.nodes().iter().any(|node| matches!(
            node,
            ExprNode::Unary {
                op: UnaryOp::Exp,
                ..
            }
        )));
    }

    #[test]
    fn metadata_survives_graph_construction() {
        let graph = event_scalar("mass")
            .named("event mass")
            .tagged("data")
            .tagged("data")
            .to_graph();
        let metadata = graph.metadata(graph.root()).unwrap();
        assert_eq!(metadata.name(), Some("event mass"));
        assert_eq!(metadata.tags(), &[Arc::from("data")]);
        assert!(metadata.has_tag("data"));
    }

    #[test]
    fn expressions_round_trip_through_serde_with_metadata() {
        let expression = ((parameter!("x", initial: 1.0) + 2.0).named("offset")
            * event_scalar("mass").tagged("data"))
        .tagged("model");
        let encoded = serde_json::to_string(&expression).unwrap();
        let decoded: Expr = serde_json::from_str(&encoded).unwrap();

        assert_eq!(
            serde_json::to_value(expression.to_graph()).unwrap(),
            serde_json::to_value(decoded.to_graph()).unwrap()
        );
    }

    #[test]
    fn display_formats_graph_as_labeled_tree() {
        let graph = ((parameter!("x") + 1.0).named("offset") * event_scalar("mass").tagged("data"))
            .to_graph();
        let display = graph.display_tree().to_string();

        assert!(display.starts_with("ExprGraph(root=#"));
        assert!(display.contains("Binary(Mul)"));
        assert!(display.contains("┣ lhs:"));
        assert!(display.contains("┗ rhs:"));
        assert!(display.contains("Binary(Add) name=\"offset\""));
        assert!(display.contains("ScalarParam(x)"));
        assert!(display.contains("RealConst(1)"));
        assert!(display.contains("EventScalar(mass) tags=[data]"));
    }

    #[test]
    fn display_formats_graph_as_expression() {
        let costheta = Expr::from(parameter!("costheta"));
        let phi = event_scalar("phi");
        let p = Expr::from(parameter!("p"));
        let phase = Expr::from(7.0) * Complex64::I;
        let graph =
            (((costheta.powi(2) * phi.sin()) - 5.2).norm_sqr() * p.conj() - phase.exp()).to_graph();

        assert_eq!(
            graph.to_string(),
            "|costheta^2 * sin(phi) - 5.2|^2 * conj(p) - exp(7 * i)"
        );
    }

    #[test]
    fn display_parenthesizes_when_precedence_requires_it() {
        let a = Expr::from(parameter!("a"));
        let b = Expr::from(parameter!("b"));
        let c = Expr::from(parameter!("c"));

        assert_eq!(
            (a.clone() * (b.clone() + c.clone())).to_graph().to_string(),
            "a * (b + c)"
        );
        assert_eq!(
            (a.clone() - (b.clone() - c.clone())).to_graph().to_string(),
            "a - (b - c)"
        );
        assert_eq!(((a / b) / c).to_graph().to_string(), "a / b / c");
    }

    #[test]
    fn display_rounds_tiny_float_representation_noise() {
        let metadata = ExprMetadata::new(ExprSourceKind::Const);
        let graph = ExprGraph::from_parts(
            ExprId::from_index(2),
            vec![
                ExprNode::RealConst(2.9999999999999996),
                ExprNode::ComplexConst(Complex64::new(0.30000000000000004, 1.9999999999999998)),
                ExprNode::Binary {
                    op: BinaryOp::Add,
                    lhs: ExprId::from_index(0),
                    rhs: ExprId::from_index(1),
                },
            ],
            vec![metadata.clone(), metadata.clone(), metadata],
        )
        .unwrap();

        assert_eq!(graph.to_string(), "3 + 0.3 + 2i");
        assert!(graph.display_tree().to_string().contains("RealConst(3)"));
        assert!(
            graph
                .display_tree()
                .to_string()
                .contains("ComplexConst(0.3 + 2i)")
        );
    }

    #[test]
    fn display_formats_p4_components_and_atan2() {
        let expr = atan2(
            event_p4_component("ks1", P4Component::Py),
            event_p4_component("ks1", P4Component::Px),
        );

        assert_eq!(expr.to_graph().to_string(), "atan2(ks1.py, ks1.px)");
    }

    #[test]
    fn graph_from_parts_validates_structure() {
        let metadata = ExprMetadata::new(ExprSourceKind::Const);
        let graph = ExprGraph::from_parts(
            ExprId::from_index(1),
            vec![
                ExprNode::RealConst(1.0),
                ExprNode::Unary {
                    op: UnaryOp::Neg,
                    input: ExprId::from_index(0),
                },
            ],
            vec![metadata.clone(), metadata.clone()],
        )
        .unwrap();
        assert!(matches!(
            graph.node(graph.root()),
            Some(ExprNode::Unary {
                op: UnaryOp::Neg,
                ..
            })
        ));

        let err = ExprGraph::from_parts(
            ExprId::from_index(0),
            vec![ExprNode::RealConst(1.0)],
            Vec::new(),
        )
        .unwrap_err();
        assert!(matches!(err, ExprGraphError::MetadataLength { .. }));

        let err = ExprGraph::from_parts(
            ExprId::from_index(0),
            vec![ExprNode::Unary {
                op: UnaryOp::Neg,
                input: ExprId::from_index(0),
            }],
            vec![metadata],
        )
        .unwrap_err();
        assert!(matches!(err, ExprGraphError::InvalidChildOrder { .. }));
    }

    #[test]
    fn graph_preserves_unsimplified_expression_shape() {
        let graph = (parameter!("x") + 0.0).to_graph();
        assert!(matches!(
            graph.node(graph.root()),
            Some(ExprNode::Binary {
                op: BinaryOp::Add,
                ..
            })
        ));
    }

    #[test]
    fn graph_preserves_written_operand_order_for_commutative_ops() {
        let left_param = (parameter!("x") + 1.0).to_graph();
        assert!(matches!(
            left_param.node(left_param.root()),
            Some(ExprNode::Binary {
                op: BinaryOp::Add,
                lhs,
                rhs
            }) if matches!(left_param.node(*lhs), Some(ExprNode::ScalarParam(parameter)) if parameter.name() == "x")
                && matches!(left_param.node(*rhs), Some(ExprNode::RealConst(1.0)))
        ));

        let right_param = (1.0 + parameter!("x")).to_graph();
        assert!(matches!(
            right_param.node(right_param.root()),
            Some(ExprNode::Binary {
                op: BinaryOp::Add,
                lhs,
                rhs
            }) if matches!(right_param.node(*lhs), Some(ExprNode::RealConst(1.0)))
                && matches!(right_param.node(*rhs), Some(ExprNode::ScalarParam(parameter)) if parameter.name() == "x")
        ));
    }

    #[test]
    fn represents_kmatrix_style_solve_graph() {
        let beta = vector([
            complex(parameter!("b0_re"), parameter!("b0_im")),
            complex(parameter!("b1_re"), parameter!("b1_im")),
        ]);
        let a = matrix([
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)],
            [Complex64::new(0.0, -1.0), Complex64::new(1.0, 0.0)],
        ]);
        let graph = solve(a, beta).component(0).to_graph();

        assert!(
            graph
                .nodes()
                .iter()
                .any(|node| matches!(node, ExprNode::Solve { .. }))
        );
    }

    #[test]
    fn graph_builder_preserves_shared_dag_nodes() {
        let shared = event_scalar("x").sin();
        let expression = vector((0..1_000).map(|_| shared.clone()));
        let graph = expression.to_graph();

        assert_eq!(graph.nodes().len(), 3);
        let ExprNode::Vector { elements } = graph.node(graph.root()).unwrap() else {
            panic!("root should be a vector");
        };
        assert!(elements.windows(2).all(|pair| pair[0] == pair[1]));
    }

    #[test]
    fn dynamic_matrices_and_shapes_are_checked_eagerly() {
        let dynamic = matrix_from_flat(2, 2, [1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(
            dynamic.shape().unwrap(),
            ExprShape::Matrix { rows: 2, cols: 2 }
        );
        assert!(matrix_from_flat(2, 2, [1.0, 2.0, 3.0]).is_err());
        assert!(matmul(dynamic, matrix([[1.0, 2.0, 3.0]])).shape().is_err());
    }

    #[test]
    fn assignment_operators_build_binary_expression_nodes() {
        let mut expr = Expr::from(parameter!("x"));
        expr += parameter!("y");
        expr -= 1.0;
        expr *= Complex64::I;
        expr /= Expr::from(parameter!("z"));

        let graph = expr.to_graph();
        assert!(matches!(
            graph.node(graph.root()),
            Some(ExprNode::Binary {
                op: BinaryOp::Div,
                ..
            })
        ));
        assert_eq!(
            graph
                .nodes()
                .iter()
                .filter(|node| matches!(node, ExprNode::Binary { .. }))
                .count(),
            4
        );
    }

    #[test]
    fn assignment_operators_accept_borrowed_rhs_values() {
        let y = parameter!("y");
        let one = 1.0;
        let i = Complex64::I;
        let z = Expr::from(parameter!("z"));

        let mut expr = Expr::from(parameter!("x"));
        expr += &y;
        expr -= &one;
        expr *= &i;
        expr /= &z;

        let graph = expr.to_graph();
        assert!(matches!(
            graph.node(graph.root()),
            Some(ExprNode::Binary {
                op: BinaryOp::Div,
                ..
            })
        ));
    }
}
