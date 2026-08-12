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
    pub fn width(self) -> usize {
        match self {
            Self::Real | Self::Complex => 1,
            Self::Vector { len } => len,
            Self::Matrix { rows, cols } => rows * cols,
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

/// Whether a kernel value is constant across events or event-dependent.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum KernelValueClass {
    /// The value depends only on constants and parameters.
    Invariant,
    /// The value depends on event data or cache inputs.
    Event,
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

impl KernelInstruction {
    /// Returns the direct input value identifiers.
    pub fn operands(&self) -> Vec<KernelValueId> {
        match self {
            Self::Cached(_)
            | Self::RealConstant(_)
            | Self::ComplexConstant(_)
            | Self::Parameter(_) => Vec::new(),
            Self::Unary { input, .. }
            | Self::Component { input, .. }
            | Self::MatrixElement { input, .. }
            | Self::SolveRowAdjointElement { adjoint: input, .. } => vec![*input],
            Self::Binary { lhs, rhs, .. } | Self::MatMul { lhs, rhs } | Self::Dot { lhs, rhs } => {
                vec![*lhs, *rhs]
            }
            Self::MatVec { matrix, vector }
            | Self::Solve {
                matrix,
                rhs: vector,
            } => vec![*matrix, *vector],
            Self::Add(values)
            | Self::Mul(values)
            | Self::Vector(values)
            | Self::SolveRow { rhs: values, .. } => values.clone(),
            Self::Complex { re, im } => vec![*re, *im],
            Self::Matrix { elements, .. } => elements.clone(),
        }
    }

    fn shape_error(
        value: usize,
        operation: &'static str,
        message: impl Into<String>,
    ) -> KernelIrError {
        KernelIrError::InvalidShape {
            value,
            operation,
            message: message.into(),
        }
    }

    fn expected_kind(
        &self,
        values: &[KernelValue],
        value: usize,
    ) -> Result<Option<KernelValueKind>, KernelIrError> {
        let kind = |id: KernelValueId| values[id.index()].kind;
        Ok(Some(match self {
            Self::Cached(_) => return Ok(None),
            Self::RealConstant(_) | Self::Parameter(_) => KernelValueKind::Real,
            Self::ComplexConstant(value) if value.im == 0.0 => KernelValueKind::Real,
            Self::ComplexConstant(_) => KernelValueKind::Complex,
            Self::Unary { op, input } => {
                if !kind(*input).is_scalar() {
                    return Err(Self::shape_error(
                        value,
                        "unary operation",
                        "input is not scalar",
                    ));
                }
                match op {
                    UnaryOp::Real | UnaryOp::Imag | UnaryOp::NormSqr => KernelValueKind::Real,
                    _ => kind(*input),
                }
            }
            Self::Binary { op, lhs, rhs } => {
                if *op == BinaryOp::Atan2 {
                    if kind(*lhs) != KernelValueKind::Real || kind(*rhs) != KernelValueKind::Real {
                        return Err(Self::shape_error(
                            value,
                            "atan2",
                            "both inputs must be real",
                        ));
                    }
                    KernelValueKind::Real
                } else {
                    kind(*lhs).scalar_combine(kind(*rhs)).ok_or_else(|| {
                        Self::shape_error(value, "binary operation", "both inputs must be scalar")
                    })?
                }
            }
            Self::Add(terms) | Self::Mul(terms) => {
                let operation = if matches!(self, Self::Add(_)) {
                    "addition"
                } else {
                    "multiplication"
                };
                let mut terms = terms.iter();
                let first = terms
                    .next()
                    .ok_or(KernelIrError::EmptyOperands { value, operation })?;
                terms.try_fold(kind(*first), |acc, term| {
                    acc.scalar_combine(kind(*term)).ok_or_else(|| {
                        Self::shape_error(value, operation, "all inputs must be scalar")
                    })
                })?
            }
            Self::Complex { re, im } => {
                if kind(*re) != KernelValueKind::Real || kind(*im) != KernelValueKind::Real {
                    return Err(Self::shape_error(
                        value,
                        "complex construction",
                        "components must be real",
                    ));
                }
                KernelValueKind::Complex
            }
            Self::Vector(elements) => {
                if elements.iter().any(|element| !kind(*element).is_scalar()) {
                    return Err(Self::shape_error(
                        value,
                        "vector construction",
                        "elements must be scalar",
                    ));
                }
                KernelValueKind::Vector {
                    len: elements.len(),
                }
            }
            Self::Matrix {
                rows,
                cols,
                elements,
            } => {
                if elements.len() != rows * cols
                    || elements.iter().any(|element| !kind(*element).is_scalar())
                {
                    return Err(Self::shape_error(
                        value,
                        "matrix construction",
                        format!("expected {} scalar elements", rows * cols),
                    ));
                }
                KernelValueKind::Matrix {
                    rows: *rows,
                    cols: *cols,
                }
            }
            Self::Component { input, index } => match kind(*input) {
                KernelValueKind::Vector { len } if *index < len => KernelValueKind::Complex,
                actual => {
                    return Err(Self::shape_error(
                        value,
                        "component",
                        format!("index {index} is invalid for {actual:?}"),
                    ));
                }
            },
            Self::MatrixElement { input, row, col } => match kind(*input) {
                KernelValueKind::Matrix { rows, cols } if *row < rows && *col < cols => {
                    KernelValueKind::Complex
                }
                actual => {
                    return Err(Self::shape_error(
                        value,
                        "matrix element",
                        format!("index ({row}, {col}) is invalid for {actual:?}"),
                    ));
                }
            },
            Self::MatMul { lhs, rhs } => match (kind(*lhs), kind(*rhs)) {
                (
                    KernelValueKind::Matrix { rows, cols: inner },
                    KernelValueKind::Matrix {
                        rows: rhs_rows,
                        cols,
                    },
                ) if inner == rhs_rows => KernelValueKind::Matrix { rows, cols },
                shapes => {
                    return Err(Self::shape_error(
                        value,
                        "matrix multiplication",
                        format!("incompatible operands {shapes:?}"),
                    ));
                }
            },
            Self::MatVec { matrix, vector } => match (kind(*matrix), kind(*vector)) {
                (KernelValueKind::Matrix { rows, cols }, KernelValueKind::Vector { len })
                    if cols == len =>
                {
                    KernelValueKind::Vector { len: rows }
                }
                shapes => {
                    return Err(Self::shape_error(
                        value,
                        "matrix-vector multiplication",
                        format!("incompatible operands {shapes:?}"),
                    ));
                }
            },
            Self::Dot { lhs, rhs } => match (kind(*lhs), kind(*rhs)) {
                (KernelValueKind::Vector { len }, KernelValueKind::Vector { len: rhs_len })
                    if len == rhs_len =>
                {
                    KernelValueKind::Complex
                }
                shapes => {
                    return Err(Self::shape_error(
                        value,
                        "dot product",
                        format!("incompatible operands {shapes:?}"),
                    ));
                }
            },
            Self::Solve { matrix, rhs } => match (kind(*matrix), kind(*rhs)) {
                (KernelValueKind::Matrix { rows, cols }, KernelValueKind::Vector { len })
                    if rows == cols && rows == len =>
                {
                    KernelValueKind::Vector { len }
                }
                shapes => {
                    return Err(Self::shape_error(
                        value,
                        "linear solve",
                        format!("incompatible operands {shapes:?}"),
                    ));
                }
            },
            Self::SolveRow { rhs, .. } => {
                if rhs.is_empty() || rhs.iter().any(|entry| !kind(*entry).is_scalar()) {
                    return Err(Self::shape_error(
                        value,
                        "specialized solve row",
                        "right-hand side must contain scalars",
                    ));
                }
                KernelValueKind::Complex
            }
            Self::SolveRowAdjointElement {
                index,
                len,
                adjoint,
                ..
            } => {
                if *len == 0 || *index >= *len || !kind(*adjoint).is_scalar() {
                    return Err(Self::shape_error(
                        value,
                        "specialized solve-row adjoint",
                        "adjoint must be scalar and index must be within a non-empty row",
                    ));
                }
                KernelValueKind::Complex
            }
        }))
    }

    fn expected_class(&self, values: &[KernelValue]) -> KernelValueClass {
        match self {
            Self::Cached(_) | Self::SolveRow { .. } | Self::SolveRowAdjointElement { .. } => {
                KernelValueClass::Event
            }
            Self::RealConstant(_) | Self::ComplexConstant(_) | Self::Parameter(_) => {
                KernelValueClass::Invariant
            }
            _ if self
                .operands()
                .iter()
                .any(|operand| values[operand.index()].class == KernelValueClass::Event) =>
            {
                KernelValueClass::Event
            }
            _ => KernelValueClass::Invariant,
        }
    }
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

impl ScalarKernelIr {
    /// Validates values and constructs a scalar kernel rooted at `root`.
    ///
    /// # Errors
    ///
    /// Returns [`KernelIrError`] when the value list is empty, `root` or an
    /// operand is out of bounds, values are not topologically ordered, a
    /// value's kind or class is inconsistent with its instruction, or the
    /// root is not scalar.
    pub fn new(values: Vec<KernelValue>, root: KernelValueId) -> Result<Self, KernelIrError> {
        let ir = Self { values, root };
        ir.validate()?;
        if !ir.values[ir.root.index()].kind.is_scalar() {
            return Err(KernelIrError::InvalidShape {
                value: ir.root.index(),
                operation: "kernel root",
                message: "root must be scalar".into(),
            });
        }
        Ok(ir)
    }

    /// Revalidates ordering, types, classes, and the scalar root.
    ///
    /// # Errors
    ///
    /// Returns [`KernelIrError`] when the IR is empty, its root or an operand
    /// is out of bounds, its values are not topologically ordered, or a
    /// value's kind, class, or shape is inconsistent with its instruction.
    pub fn validate(&self) -> Result<(), KernelIrError> {
        Self::validate_values(&self.values, self.root)
    }

    fn validate_values(values: &[KernelValue], root: KernelValueId) -> Result<(), KernelIrError> {
        if values.is_empty() {
            return Err(KernelIrError::Empty);
        }
        if root.index() >= values.len() {
            return Err(KernelIrError::RootOutOfBounds {
                root: root.index(),
                len: values.len(),
            });
        }
        for (index, value) in values.iter().enumerate() {
            for operand in value.instruction.operands() {
                if operand.index() >= index {
                    return Err(KernelIrError::InvalidOperand {
                        value: index,
                        operand: operand.index(),
                    });
                }
            }
            if let Some(expected) = value.instruction.expected_kind(values, index)?
                && value.kind != expected
            {
                return Err(KernelIrError::KindMismatch {
                    value: index,
                    expected,
                    actual: value.kind,
                });
            }
            let expected = value.instruction.expected_class(values);
            if value.class != expected {
                return Err(KernelIrError::ClassMismatch {
                    value: index,
                    expected,
                    actual: value.class,
                });
            }
        }
        Ok(())
    }

    /// Returns all IR values in topological order.
    pub fn values(&self) -> &[KernelValue] {
        &self.values
    }
    /// Returns the scalar output identifier.
    pub fn root(&self) -> KernelValueId {
        self.root
    }
}

impl CacheKernelIr {
    /// Validates values and constructs a cache kernel with the given outputs.
    ///
    /// # Errors
    ///
    /// Returns [`KernelIrError`] when `outputs` is empty, an output or operand
    /// is out of bounds, values are not topologically ordered, or a value's
    /// kind, class, or shape is inconsistent with its instruction.
    pub fn new(
        values: Vec<KernelValue>,
        outputs: Vec<KernelValueId>,
    ) -> Result<Self, KernelIrError> {
        let Some(first) = outputs.first().copied() else {
            return Err(KernelIrError::EmptyCacheOutputs);
        };
        ScalarKernelIr::validate_values(&values, first)?;
        for output in &outputs {
            if output.index() >= values.len() {
                return Err(KernelIrError::CacheOutputOutOfBounds {
                    output: output.index(),
                    len: values.len(),
                });
            }
        }
        Ok(Self { values, outputs })
    }

    /// Returns all IR values in topological order.
    pub fn values(&self) -> &[KernelValue] {
        &self.values
    }

    /// Returns cache output identifiers in storage order.
    pub fn outputs(&self) -> &[KernelValueId] {
        &self.outputs
    }
}

impl GradientKernelIr {
    /// Validates and constructs a gradient kernel.
    ///
    /// # Errors
    ///
    /// Returns [`KernelIrError`] when the primal IR is invalid, the primal
    /// root is not scalar, a gradient output is out of bounds, or a gradient
    /// output is not real-valued.
    pub fn new(
        values: Vec<KernelValue>,
        primal_root: KernelValueId,
        outputs: Vec<KernelValueId>,
        component: OutputComponent,
    ) -> Result<Self, KernelIrError> {
        let ir = Self {
            values,
            primal_root,
            outputs,
            component,
        };
        ir.validate()?;
        Ok(ir)
    }

    /// Revalidates the primal root and real gradient outputs.
    ///
    /// # Errors
    ///
    /// Returns [`KernelIrError`] when the primal IR is invalid, the primal
    /// root is not scalar, a gradient output is out of bounds, or a gradient
    /// output is not real-valued.
    pub fn validate(&self) -> Result<(), KernelIrError> {
        ScalarKernelIr::validate_values(&self.values, self.primal_root)?;
        if !self.values[self.primal_root.index()].kind.is_scalar() {
            return Err(KernelIrError::InvalidShape {
                value: self.primal_root.index(),
                operation: "gradient primal root",
                message: "primal root must be scalar".into(),
            });
        }
        for output in &self.outputs {
            let Some(value) = self.values.get(output.index()) else {
                return Err(KernelIrError::GradientOutOfBounds {
                    output: output.index(),
                    len: self.values.len(),
                });
            };
            if value.kind != KernelValueKind::Real {
                return Err(KernelIrError::GradientKindMismatch {
                    output: output.index(),
                    actual: value.kind,
                });
            }
        }
        Ok(())
    }

    /// Returns all primal and derivative IR values in topological order.
    pub fn values(&self) -> &[KernelValue] {
        &self.values
    }

    /// Returns the primal scalar output identifier.
    pub fn primal_root(&self) -> KernelValueId {
        self.primal_root
    }

    /// Returns derivative output identifiers.
    pub fn outputs(&self) -> &[KernelValueId] {
        &self.outputs
    }

    /// Returns the differentiated component of the complex primal.
    pub fn component(&self) -> OutputComponent {
        self.component
    }
}

impl KernelIrBuilder {
    /// Creates a builder initialized with a scalar kernel's values.
    pub fn from_scalar(ir: &ScalarKernelIr) -> Self {
        Self {
            values: ir.values.clone(),
        }
    }

    /// Appends an instruction after validating its operands and inferred type.
    ///
    /// # Errors
    ///
    /// Returns [`KernelIrError`] when an operand does not precede the new
    /// instruction, the operand shapes are incompatible, or the instruction's
    /// value kind cannot be inferred.
    pub fn push(&mut self, instruction: KernelInstruction) -> Result<KernelValueId, KernelIrError> {
        let index = self.values.len();
        for operand in instruction.operands() {
            if operand.index() >= index {
                return Err(KernelIrError::InvalidOperand {
                    value: index,
                    operand: operand.index(),
                });
            }
        }
        let kind = instruction
            .expected_kind(&self.values, index)?
            .ok_or_else(|| KernelIrError::InvalidShape {
                value: index,
                operation: "derived instruction",
                message: "instruction requires an explicitly supplied value kind".into(),
            })?;
        let class = instruction.expected_class(&self.values);
        let id = KernelValueId::from_index(index);
        self.values.push(KernelValue {
            kind,
            class,
            instruction,
        });
        Ok(id)
    }

    /// Finishes the builder as a validated gradient kernel.
    ///
    /// # Errors
    ///
    /// Returns [`KernelIrError`] when the accumulated primal IR is invalid,
    /// `primal_root` is not scalar, a gradient output is out of bounds, or a
    /// gradient output is not real-valued.
    pub fn finish_gradient(
        self,
        primal_root: KernelValueId,
        outputs: Vec<KernelValueId>,
        component: OutputComponent,
    ) -> Result<GradientKernelIr, KernelIrError> {
        GradientKernelIr::new(self.values, primal_root, outputs, component)
    }

    /// Returns the values accumulated so far.
    pub fn values(&self) -> &[KernelValue] {
        &self.values
    }
}

#[cfg(test)]
mod tests {
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
}
