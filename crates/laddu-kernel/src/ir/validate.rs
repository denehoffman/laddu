use super::*;

struct InferenceContext<'a> {
    values: &'a [KernelValue],
    output: usize,
}

impl InferenceContext<'_> {
    fn kind(&self, id: KernelValueId) -> KernelValueKind {
        self.values[id.index()].kind
    }

    fn invalid_shape(&self, operation: &'static str, message: impl Into<String>) -> KernelIrError {
        KernelInstruction::shape_error(self.output, operation, message)
    }

    fn infer_unary(
        &self,
        op: UnaryOp,
        input: KernelValueId,
    ) -> Result<KernelValueKind, KernelIrError> {
        let input_kind = self.kind(input);
        if !input_kind.is_scalar() {
            return Err(self.invalid_shape("unary operation", "input is not scalar"));
        }
        Ok(match op {
            UnaryOp::Real | UnaryOp::Imag | UnaryOp::NormSqr => KernelValueKind::Real,
            _ => input_kind,
        })
    }

    fn infer_binary(
        &self,
        op: BinaryOp,
        lhs: KernelValueId,
        rhs: KernelValueId,
    ) -> Result<KernelValueKind, KernelIrError> {
        if op == BinaryOp::Atan2 {
            if self.kind(lhs) != KernelValueKind::Real || self.kind(rhs) != KernelValueKind::Real {
                return Err(self.invalid_shape("atan2", "both inputs must be real"));
            }
            Ok(KernelValueKind::Real)
        } else {
            self.kind(lhs)
                .scalar_combine(self.kind(rhs))
                .ok_or_else(|| self.invalid_shape("binary operation", "both inputs must be scalar"))
        }
    }

    fn infer_variadic_scalar(
        &self,
        terms: &[KernelValueId],
        operation: &'static str,
    ) -> Result<KernelValueKind, KernelIrError> {
        let mut terms = terms.iter();
        let first = terms.next().ok_or(KernelIrError::EmptyOperands {
            value: self.output,
            operation,
        })?;
        terms.try_fold(self.kind(*first), |acc, term| {
            acc.scalar_combine(self.kind(*term))
                .ok_or_else(|| self.invalid_shape(operation, "all inputs must be scalar"))
        })
    }

    fn infer_complex(
        &self,
        re: KernelValueId,
        im: KernelValueId,
    ) -> Result<KernelValueKind, KernelIrError> {
        if self.kind(re) != KernelValueKind::Real || self.kind(im) != KernelValueKind::Real {
            return Err(self.invalid_shape("complex construction", "components must be real"));
        }
        Ok(KernelValueKind::Complex)
    }

    fn infer_vector(&self, elements: &[KernelValueId]) -> Result<KernelValueKind, KernelIrError> {
        if elements
            .iter()
            .any(|element| !self.kind(*element).is_scalar())
        {
            return Err(self.invalid_shape("vector construction", "elements must be scalar"));
        }
        Ok(KernelValueKind::Vector {
            len: elements.len(),
        })
    }

    fn infer_matrix(
        &self,
        rows: usize,
        cols: usize,
        elements: &[KernelValueId],
    ) -> Result<KernelValueKind, KernelIrError> {
        let width = checked_matrix_width(rows, cols).ok_or_else(|| {
            self.invalid_shape(
                "matrix construction",
                format!("shape {rows}x{cols} exceeds addressable width"),
            )
        })?;
        if elements.len() != width
            || elements
                .iter()
                .any(|element| !self.kind(*element).is_scalar())
        {
            return Err(self.invalid_shape(
                "matrix construction",
                format!("expected {width} scalar elements"),
            ));
        }
        Ok(KernelValueKind::Matrix { rows, cols })
    }

    fn infer_component(
        &self,
        input: KernelValueId,
        index: usize,
    ) -> Result<KernelValueKind, KernelIrError> {
        match self.kind(input) {
            KernelValueKind::Vector { len } if index < len => Ok(KernelValueKind::Complex),
            actual => Err(self.invalid_shape(
                "component",
                format!("index {index} is invalid for {actual:?}"),
            )),
        }
    }

    fn infer_matrix_element(
        &self,
        input: KernelValueId,
        row: usize,
        col: usize,
    ) -> Result<KernelValueKind, KernelIrError> {
        match self.kind(input) {
            KernelValueKind::Matrix { rows, cols }
                if checked_row_major_index(rows, cols, row, col).is_some() =>
            {
                Ok(KernelValueKind::Complex)
            }
            actual => Err(self.invalid_shape(
                "matrix element",
                format!("index ({row}, {col}) is invalid for {actual:?}"),
            )),
        }
    }

    fn infer_mat_mul(
        &self,
        lhs: KernelValueId,
        rhs: KernelValueId,
    ) -> Result<KernelValueKind, KernelIrError> {
        match (self.kind(lhs), self.kind(rhs)) {
            (
                KernelValueKind::Matrix { rows, cols: inner },
                KernelValueKind::Matrix {
                    rows: rhs_rows,
                    cols,
                },
            ) if inner == rhs_rows => Ok(KernelValueKind::Matrix { rows, cols }),
            shapes => Err(self.invalid_shape(
                "matrix multiplication",
                format!("incompatible operands {shapes:?}"),
            )),
        }
    }

    fn infer_mat_vec(
        &self,
        matrix: KernelValueId,
        vector: KernelValueId,
    ) -> Result<KernelValueKind, KernelIrError> {
        match (self.kind(matrix), self.kind(vector)) {
            (KernelValueKind::Matrix { rows, cols }, KernelValueKind::Vector { len })
                if cols == len =>
            {
                Ok(KernelValueKind::Vector { len: rows })
            }
            shapes => Err(self.invalid_shape(
                "matrix-vector multiplication",
                format!("incompatible operands {shapes:?}"),
            )),
        }
    }

    fn infer_dot(
        &self,
        lhs: KernelValueId,
        rhs: KernelValueId,
    ) -> Result<KernelValueKind, KernelIrError> {
        match (self.kind(lhs), self.kind(rhs)) {
            (KernelValueKind::Vector { len }, KernelValueKind::Vector { len: rhs_len })
                if len == rhs_len =>
            {
                Ok(KernelValueKind::Complex)
            }
            shapes => {
                Err(self.invalid_shape("dot product", format!("incompatible operands {shapes:?}")))
            }
        }
    }

    fn infer_solve(
        &self,
        matrix: KernelValueId,
        rhs: KernelValueId,
    ) -> Result<KernelValueKind, KernelIrError> {
        match (self.kind(matrix), self.kind(rhs)) {
            (KernelValueKind::Matrix { rows, cols }, KernelValueKind::Vector { len })
                if rows == cols && rows == len =>
            {
                Ok(KernelValueKind::Vector { len })
            }
            shapes => {
                Err(self.invalid_shape("linear solve", format!("incompatible operands {shapes:?}")))
            }
        }
    }

    fn infer_solve_row(&self, rhs: &[KernelValueId]) -> Result<KernelValueKind, KernelIrError> {
        if rhs.is_empty() || rhs.iter().any(|entry| !self.kind(*entry).is_scalar()) {
            return Err(self.invalid_shape(
                "specialized solve row",
                "right-hand side must contain scalars",
            ));
        }
        Ok(KernelValueKind::Complex)
    }

    fn infer_solve_row_adjoint(
        &self,
        index: usize,
        len: usize,
        adjoint: KernelValueId,
    ) -> Result<KernelValueKind, KernelIrError> {
        if len == 0 || index >= len || !self.kind(adjoint).is_scalar() {
            return Err(self.invalid_shape(
                "specialized solve-row adjoint",
                "adjoint must be scalar and index must be within a non-empty row",
            ));
        }
        Ok(KernelValueKind::Complex)
    }
}

impl KernelInstruction {
    pub(super) fn shape_error(
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

    pub(super) fn expected_kind(
        &self,
        values: &[KernelValue],
        value: usize,
    ) -> Result<Option<KernelValueKind>, KernelIrError> {
        let context = InferenceContext {
            values,
            output: value,
        };
        let kind = match self {
            Self::Cached(_) => return Ok(None),
            Self::RealConstant(_) | Self::Parameter(_) => KernelValueKind::Real,
            Self::ComplexConstant(value) if value.im == 0.0 => KernelValueKind::Real,
            Self::ComplexConstant(_) => KernelValueKind::Complex,
            Self::Unary { op, input } => context.infer_unary(*op, *input)?,
            Self::Binary { op, lhs, rhs } => context.infer_binary(*op, *lhs, *rhs)?,
            Self::Add(terms) => context.infer_variadic_scalar(terms, "addition")?,
            Self::Mul(terms) => context.infer_variadic_scalar(terms, "multiplication")?,
            Self::Complex { re, im } => context.infer_complex(*re, *im)?,
            Self::Vector(elements) => context.infer_vector(elements)?,
            Self::Matrix {
                rows,
                cols,
                elements,
            } => context.infer_matrix(*rows, *cols, elements)?,
            Self::Component { input, index } => context.infer_component(*input, *index)?,
            Self::MatrixElement { input, row, col } => {
                context.infer_matrix_element(*input, *row, *col)?
            }
            Self::MatMul { lhs, rhs } => context.infer_mat_mul(*lhs, *rhs)?,
            Self::MatVec { matrix, vector } => context.infer_mat_vec(*matrix, *vector)?,
            Self::Dot { lhs, rhs } => context.infer_dot(*lhs, *rhs)?,
            Self::Solve { matrix, rhs } => context.infer_solve(*matrix, *rhs)?,
            Self::SolveRow { rhs, .. } => context.infer_solve_row(rhs)?,
            Self::SolveRowAdjointElement {
                index,
                len,
                adjoint,
                ..
            } => context.infer_solve_row_adjoint(*index, *len, *adjoint)?,
        };
        Ok(Some(kind))
    }

    pub(super) fn expected_class(&self, values: &[KernelValue]) -> KernelValueClass {
        match self.event_dependence() {
            KernelEventDependence::Invariant => KernelValueClass::Invariant,
            KernelEventDependence::Event => KernelValueClass::Event,
            KernelEventDependence::Operands => {
                let mut class = KernelValueClass::Invariant;
                self.visit_operands(|operand| {
                    if values[operand.index()].class == KernelValueClass::Event {
                        class = KernelValueClass::Event;
                    }
                });
                class
            }
        }
    }
}

pub(super) fn validate_graph(values: &[KernelValue]) -> Result<(), KernelIrError> {
    if values.is_empty() {
        return Err(KernelIrError::Empty);
    }
    for (index, value) in values.iter().enumerate() {
        if let KernelValueKind::Matrix { rows, cols } = value.kind
            && checked_matrix_width(rows, cols).is_none()
        {
            return Err(KernelInstruction::shape_error(
                index,
                "matrix shape",
                format!("shape {rows}x{cols} exceeds addressable width"),
            ));
        }
        value.instruction.validate_operand_order(index)?;
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
