use std::path::PathBuf;

use num::complex::Complex64;
use pyo3::{
    class::basic::CompareOp,
    exceptions::PyTypeError,
    prelude::*,
    types::{PyAny, PyTuple},
};

use laddu_expr::parameters::Parameter;
use laddu_expr::{
    Expr, ExprShape, acos as expr_acos, atan2 as expr_atan2, cis as expr_cis,
    complex as expr_complex, dot as expr_dot, event_scalar, matmul as expr_matmul,
    matrix_from_flat, matvec as expr_matvec, polar_complex as expr_polar_complex,
    solve as expr_solve, vector as expr_vector,
};

use super::query::PyPredicate;
use super::{
    error::to_py_err,
    visualization::{
        PyNodeStyleRule, expression_dot, expression_equation, expression_latex, expression_svg,
        expression_tree,
    },
};

#[pyclass(name = "Expr", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone)]
/// A symbolic expression evaluated once per event.
///
/// Expressions form an immutable computation graph. Arithmetic with Python
/// complex values or other expressions builds a new graph; it does not
/// immediately compute a value. Compile the expression as part of a model or
/// likelihood to evaluate it efficiently over a dataset.
///
/// Parameters
/// ----------
/// value : Expr or complex
///     Value from which to construct the expression.
///
/// Examples
/// --------
/// >>> import laddu as ld
/// >>> mass = ld.parameter("mass", initial=1.0, bounds=(0.5, 2.0))
/// >>> intensity = (mass * mass + 1.0).named("intensity")
pub struct PyExpr {
    pub(crate) inner: Expr,
}

impl From<Expr> for PyExpr {
    fn from(inner: Expr) -> Self {
        Self { inner }
    }
}

/// Convert a Python expression-like value to its Rust expression graph.
///
/// # Errors
///
/// Returns [`PyTypeError`] unless `value` is an [`Expr`][PyExpr] or a complex value.
pub fn extract_expr(value: &Bound<'_, PyAny>) -> PyResult<Expr> {
    if let Ok(expression) = value.extract::<PyRef<'_, PyExpr>>() {
        return Ok(expression.inner.clone());
    }
    if let Ok(value) = value.extract::<f64>() {
        return Ok(Expr::from(value));
    }
    if let Ok(value) = value.extract::<Complex64>() {
        return Ok(Expr::from(value));
    }
    Err(PyTypeError::new_err("expected an Expr, float, or complex"))
}

fn binary(
    lhs: &Expr,
    rhs: &Bound<'_, PyAny>,
    operation: impl FnOnce(&Expr, &Expr) -> Expr,
) -> PyResult<PyExpr> {
    let rhs = extract_expr(rhs)?;
    Ok(operation(lhs, &rhs).into())
}

fn reflected(
    lhs: &Bound<'_, PyAny>,
    rhs: &Expr,
    operation: impl FnOnce(&Expr, &Expr) -> Expr,
) -> PyResult<PyExpr> {
    let lhs = extract_expr(lhs)?;
    Ok(operation(&lhs, rhs).into())
}

fn matrix_product(lhs: Expr, rhs: Expr) -> PyResult<PyExpr> {
    let product = match (
        lhs.shape().map_err(to_py_err)?,
        rhs.shape().map_err(to_py_err)?,
    ) {
        (ExprShape::Matrix { .. }, ExprShape::Matrix { .. }) => expr_matmul(lhs, rhs),
        (ExprShape::Matrix { .. }, ExprShape::Vector { .. }) => expr_matvec(lhs, rhs),
        (ExprShape::Vector { .. }, ExprShape::Vector { .. }) => expr_dot(lhs, rhs),
        (lhs, rhs) => {
            return Err(PyTypeError::new_err(format!(
                "the @ operator requires matrix @ matrix, matrix @ vector, or vector @ vector operands, got {lhs:?} @ {rhs:?}"
            )));
        }
    };
    product.shape().map_err(to_py_err)?;
    Ok(product.into())
}

#[pymethods]
impl PyExpr {
    /// Create a symbolic expression from an expression or numeric constant.
    ///
    /// Parameters
    /// ----------
    /// value : Expr or complex
    ///     Symbolic expression or constant to wrap.
    ///
    /// Returns
    /// -------
    /// Expr
    ///     An immutable symbolic expression.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If `value` cannot be converted to an expression.
    #[new]
    #[pyo3(signature = (value: "Expr | complex"))]
    fn new(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(extract_expr(value)?.into())
    }

    fn __repr__(&self) -> String {
        format!("Expr({})", self.inner.to_graph())
    }

    fn __str__(&self) -> String {
        self.inner.to_graph().to_string()
    }

    /// Return the expression as a compact mathematical equation.
    ///
    /// Parameters
    /// ----------
    /// colors : {'light', 'dark', 'none'}, optional
    ///     Built-in ANSI color palette. The default emits no color.
    /// style_rules : sequence of NodeStyleRule, optional
    ///     Custom rules applied after the preset; later rules take precedence.
    #[pyo3(signature = (*, colors=None, style_rules=None))]
    fn equation(
        &self,
        colors: Option<&Bound<'_, PyAny>>,
        style_rules: Option<Vec<PyNodeStyleRule>>,
    ) -> PyResult<String> {
        expression_equation(&self.inner.to_graph(), colors, style_rules)
    }

    /// Return the expression as a LaTeX math-mode fragment.
    ///
    /// Parameters
    /// ----------
    /// colors : {'light', 'dark', 'none'}, optional
    ///     Built-in ``\\color[RGB]`` palette. Colored output requires the
    ///     LaTeX ``xcolor`` package. The default emits no color commands.
    /// style_rules : sequence of NodeStyleRule, optional
    ///     Custom foreground-color rules applied after the preset.
    #[pyo3(signature = (*, colors=None, style_rules=None))]
    fn latex(
        &self,
        colors: Option<&Bound<'_, PyAny>>,
        style_rules: Option<Vec<PyNodeStyleRule>>,
    ) -> PyResult<String> {
        expression_latex(&self.inner.to_graph(), colors, style_rules)
    }

    /// Return an indented tree representation of the expression graph.
    ///
    /// Parameters
    /// ----------
    /// colors : {'light', 'dark', 'none'}, optional
    ///     Color palette for ANSI terminal output. The default emits no color.
    /// expand_repeated : bool, default=True
    ///     Expand shared subtrees at every occurrence. If false, later
    ///     occurrences are printed as references.
    /// style_rules : sequence of NodeStyleRule, optional
    ///     Custom rules applied after the preset; later rules take precedence.
    #[pyo3(signature = (*, colors=None, expand_repeated=true, style_rules=None))]
    fn tree(
        &self,
        colors: Option<&Bound<'_, PyAny>>,
        expand_repeated: bool,
        style_rules: Option<Vec<PyNodeStyleRule>>,
    ) -> PyResult<String> {
        expression_tree(&self.inner.to_graph(), colors, expand_repeated, style_rules)
    }

    /// Return the expression graph as Graphviz DOT source.
    ///
    /// Parameters
    /// ----------
    /// colors : {'light', 'dark', 'none'}, optional
    ///     Color palette for graph text, fills, and borders.
    /// expand_repeated : bool, default=True
    ///     Duplicate shared subtrees at every occurrence. If false, emit a
    ///     shared directed acyclic graph.
    /// style_rules : sequence of NodeStyleRule, optional
    ///     Custom rules applied after the preset; later rules take precedence.
    #[pyo3(signature = (*, colors=None, expand_repeated=true, style_rules=None))]
    fn dot(
        &self,
        colors: Option<&Bound<'_, PyAny>>,
        expand_repeated: bool,
        style_rules: Option<Vec<PyNodeStyleRule>>,
    ) -> PyResult<String> {
        expression_dot(&self.inner.to_graph(), colors, expand_repeated, style_rules)
    }

    /// Render the expression graph to an SVG file.
    ///
    /// Parameters
    /// ----------
    /// path : str or os.PathLike
    ///     Destination SVG path. An existing file is replaced.
    /// colors : {'light', 'dark', 'none'}, optional
    ///     Color palette for graph text, fills, and borders.
    /// expand_repeated : bool, default=True
    ///     Duplicate shared subtrees at every occurrence. If false, render a
    ///     shared directed acyclic graph.
    /// style_rules : sequence of NodeStyleRule, optional
    ///     Custom rules applied after the preset; later rules take precedence.
    ///
    /// Returns
    /// -------
    /// None
    #[pyo3(signature = (path, *, colors=None, expand_repeated=true, style_rules=None))]
    fn svg(
        &self,
        path: PathBuf,
        colors: Option<&Bound<'_, PyAny>>,
        expand_repeated: bool,
        style_rules: Option<Vec<PyNodeStyleRule>>,
    ) -> PyResult<()> {
        expression_svg(
            &self.inner.to_graph(),
            &path,
            colors,
            expand_repeated,
            style_rules,
        )
    }

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        binary(&self.inner, other, |lhs, rhs| lhs + rhs)
    }

    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        reflected(other, &self.inner, |lhs, rhs| lhs + rhs)
    }

    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        binary(&self.inner, other, |lhs, rhs| lhs - rhs)
    }

    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        reflected(other, &self.inner, |lhs, rhs| lhs - rhs)
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        binary(&self.inner, other, |lhs, rhs| lhs * rhs)
    }

    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        reflected(other, &self.inner, |lhs, rhs| lhs * rhs)
    }

    /// Multiply matrices or take a vector dot product.
    ///
    /// ``matrix @ matrix`` constructs a matrix product, ``matrix @ vector``
    /// constructs a matrix-vector product, and ``vector @ vector`` constructs
    /// a dot product. Incompatible shapes are rejected immediately.
    fn __matmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        matrix_product(self.inner.clone(), extract_expr(other)?)
    }

    fn __rmatmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        matrix_product(extract_expr(other)?, self.inner.clone())
    }

    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        binary(&self.inner, other, |lhs, rhs| lhs / rhs)
    }

    fn __rtruediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        reflected(other, &self.inner, |lhs, rhs| lhs / rhs)
    }

    fn __neg__(&self) -> Self {
        Self::from(-&self.inner)
    }

    fn __pow__(&self, power: i32, modulo: Option<i32>) -> PyResult<Self> {
        if modulo.is_some() {
            return Err(PyTypeError::new_err(
                "modular exponentiation is not defined for expressions",
            ));
        }
        Ok(self.inner.powi(power).into())
    }

    fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: CompareOp) -> PyResult<PyPredicate> {
        let rhs = extract_expr(other)?;
        let inner = match op {
            CompareOp::Lt => laddu_runtime::Predicate::lt(self.inner.clone(), rhs),
            CompareOp::Le => laddu_runtime::Predicate::le(self.inner.clone(), rhs),
            CompareOp::Eq => laddu_runtime::Predicate::eq(self.inner.clone(), rhs),
            CompareOp::Ne => laddu_runtime::Predicate::ne(self.inner.clone(), rhs),
            CompareOp::Gt => laddu_runtime::Predicate::gt(self.inner.clone(), rhs),
            CompareOp::Ge => laddu_runtime::Predicate::ge(self.inner.clone(), rhs),
        };
        Ok(PyPredicate { inner })
    }

    fn __getitem__(&self, index: usize) -> Self {
        self.inner.component(index).into()
    }

    #[pyo3(signature = (row, col))]
    /// Select an element from a matrix-valued expression.
    ///
    /// Parameters
    /// ----------
    /// row : int
    ///     Zero-based row index.
    /// col : int
    ///     Zero-based column index.
    ///
    /// Returns
    /// -------
    /// Expr
    ///     Scalar expression representing the selected element.
    fn at(&self, row: usize, col: usize) -> Self {
        self.inner.matrix_element(row, col).into()
    }

    /// Return the real part of this expression.
    fn real(&self) -> Self {
        self.inner.real().into()
    }

    /// Return the imaginary part of this expression.
    fn imag(&self) -> Self {
        self.inner.imag().into()
    }

    /// Return the complex conjugate of this expression.
    fn conj(&self) -> Self {
        self.inner.conj().into()
    }

    /// Return the squared complex norm of this expression.
    fn norm_sqr(&self) -> Self {
        self.inner.norm_sqr().into()
    }

    /// Apply the principal square-root function.
    fn sqrt(&self) -> Self {
        self.inner.sqrt().into()
    }

    /// Apply the exponential function.
    fn exp(&self) -> Self {
        self.inner.exp().into()
    }

    /// Apply the natural logarithm.
    fn log(&self) -> Self {
        self.inner.log().into()
    }

    /// Apply the sine function.
    fn sin(&self) -> Self {
        self.inner.sin().into()
    }

    /// Apply the cosine function.
    fn cos(&self) -> Self {
        self.inner.cos().into()
    }

    /// Apply the inverse cosine function.
    fn acos(&self) -> Self {
        self.inner.acos().into()
    }

    /// Attach a display name to the root node of this expression.
    ///
    /// Names make generated graphs and diagnostics easier to read but do not
    /// change the numerical result.
    fn named(&self, name: String) -> Self {
        self.inner.clone().named(name).into()
    }

    /// Attach a projection tag to the root node of this expression.
    ///
    /// Tags can later be selected with :meth:`project` to isolate matching
    /// contributions from a larger expression graph.
    fn tagged(&self, tag: String) -> Self {
        self.inner.clone().tagged(tag).into()
    }

    /// Project an expression onto contributions carrying any selected tag.
    ///
    /// Parameters
    /// ----------
    /// tags : sequence of str
    ///     Tags retained by the projection.
    ///
    /// Returns
    /// -------
    /// Expr
    ///     Projected symbolic expression.
    ///
    /// Examples
    /// --------
    /// >>> import laddu as ld
    /// >>> signal = ld.parameter("signal").tagged("signal")
    /// >>> background = ld.parameter("background").tagged("background")
    /// >>> projected = (signal + background).project(["signal"])
    fn project(&self, tags: Vec<String>) -> Self {
        self.inner
            .project_tags(tags.iter().map(String::as_str))
            .into()
    }

    #[getter]
    /// tuple of int: The expression dimensions; scalars have shape ``()``.
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let dimensions = match self.inner.shape().map_err(to_py_err)? {
            ExprShape::Scalar => vec![],
            ExprShape::Vector { len } => vec![len],
            ExprShape::Matrix { rows, cols } => vec![rows, cols],
        };
        PyTuple::new(py, dimensions)
    }
}

#[pyfunction]
#[pyo3(signature = (
    name,
    *,
    initial: "float | tuple[float, float] | None" = None,
    bounds=None,
    fixed=None,
    periodic=false,
    scale=None,
    unit=None,
    latex=None,
    description=None
))]
#[allow(clippy::too_many_arguments)]
/// Create a named fit parameter expression.
///
/// Parameters
/// ----------
/// name : str
///     Unique parameter name used when compiling models and supplying values.
/// initial : float or tuple of float, optional
///     Initial value, or a ``(minimum, maximum)`` range from which an optimizer
///     may choose an initial value.
/// bounds : tuple of float or None, optional
///     Inclusive ``(minimum, maximum)`` bounds. Either endpoint may be ``None``.
/// fixed : float, optional
///     Fix the parameter at this value instead of fitting it.
/// periodic : bool, default=False
///     Wrap values into the finite interval given by `bounds`.
/// scale : float, optional
///     Positive characteristic scale supplied to optimizers.
/// unit : str, optional
///     Human-readable unit label.
/// latex : str, optional
///     LaTeX label for plots and reports.
/// description : str, optional
///     Longer human-readable description.
///
/// Returns
/// -------
/// Expr
///     Scalar expression referring to the parameter.
///
/// Raises
/// ------
/// TypeError
///     If `initial` is neither a float nor a pair of floats.
/// ValueError
///     If the name, bounds, initial value, periodicity, or scale is invalid.
///
/// Examples
/// --------
/// >>> import laddu as ld
/// >>> width = ld.parameter(
/// ...     "width", initial=0.15, bounds=(0.0, 1.0), unit="GeV"
/// ... )
pub fn parameter(
    name: String,
    initial: Option<&Bound<'_, PyAny>>,
    bounds: Option<(Option<f64>, Option<f64>)>,
    fixed: Option<f64>,
    periodic: bool,
    scale: Option<f64>,
    unit: Option<String>,
    latex: Option<String>,
    description: Option<String>,
) -> PyResult<PyExpr> {
    let mut parameter = match fixed {
        Some(value) => Parameter::fixed(name, value),
        None => Parameter::free(name),
    };
    if let Some(initial) = initial {
        if let Ok(initial) = initial.extract::<f64>() {
            parameter = parameter.with_initial(initial);
        } else if let Ok(initial) = initial.extract::<(f64, f64)>() {
            parameter = parameter.with_initial(initial);
        } else {
            return Err(PyTypeError::new_err(
                "initial must be a float or a (minimum, maximum) tuple",
            ));
        }
    }
    if let Some((minimum, maximum)) = bounds {
        parameter = parameter.with_bounds(minimum, maximum);
    }
    parameter = parameter.with_periodicity(periodic);
    if let Some(scale) = scale {
        parameter = parameter.with_scale(scale);
    }
    if let Some(unit) = unit {
        parameter = parameter.with_unit(unit);
    }
    if let Some(latex) = latex {
        parameter = parameter.with_latex(latex);
    }
    if let Some(description) = description {
        parameter = parameter.with_description(description);
    }
    // ParamLayout performs the same validation used during compilation, so fail early.
    laddu_expr::parameters::ParamLayout::new([parameter.clone()]).map_err(to_py_err)?;
    Ok(Expr::from(parameter).into())
}

#[pyfunction]
/// Create an expression that reads a named scalar from each event.
///
/// Parameters
/// ----------
/// name : str
///     Name of the scalar column in the event dataset.
///
/// Returns
/// -------
/// Expr
///     Event-dependent scalar expression.
pub fn scalar(name: String) -> PyExpr {
    event_scalar(name).into()
}

#[pyfunction]
#[pyo3(signature = (
    re: "Expr | float",
    im: "Expr | float"
))]
/// Construct a complex expression from Cartesian components.
///
/// Parameters
/// ----------
/// re, im : Expr or float
///     Real and imaginary components.
///
/// Returns
/// -------
/// Expr
///     Complex-valued expression ``re + 1j * im``.
///
/// Raises
/// ------
/// TypeError
///     If either component cannot be converted to an expression.
pub fn complex(re: &Bound<'_, PyAny>, im: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
    Ok(expr_complex(extract_expr(re)?, extract_expr(im)?).into())
}

#[pyfunction]
#[pyo3(signature = (
    magnitude: "Expr | float",
    phase: "Expr | float"
))]
/// Construct a complex expression from polar components.
///
/// Parameters
/// ----------
/// magnitude, phase : Expr or float
///     Magnitude and phase in radians.
///
/// Returns
/// -------
/// Expr
///     Complex expression ``magnitude * exp(1j * phase)``.
///
/// Raises
/// ------
/// TypeError
///     If either argument cannot be converted to an expression.
pub fn polar_complex(magnitude: &Bound<'_, PyAny>, phase: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
    Ok(expr_polar_complex(extract_expr(magnitude)?, extract_expr(phase)?).into())
}

#[pyfunction]
#[pyo3(signature = (phase: "Expr | float"))]
/// Construct the unit complex expression ``exp(1j * phase)``.
///
/// Parameters
/// ----------
/// phase : Expr or float
///     Phase in radians.
///
/// Returns
/// -------
/// Expr
///     Unit-magnitude complex expression.
///
/// Raises
/// ------
/// TypeError
///     If `phase` cannot be converted to an expression.
pub fn cis(phase: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
    Ok(expr_cis(extract_expr(phase)?).into())
}

#[pyfunction]
#[pyo3(signature = (
    y: "Expr | float",
    x: "Expr | float"
))]
/// Construct the quadrant-aware angle ``atan2(y, x)``.
///
/// Parameters
/// ----------
/// y, x : Expr or float
///     Cartesian components of the angle.
///
/// Returns
/// -------
/// Expr
///     Angle in radians.
///
/// Raises
/// ------
/// TypeError
///     If either argument cannot be converted to an expression.
pub fn atan2(y: &Bound<'_, PyAny>, x: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
    Ok(expr_atan2(extract_expr(y)?, extract_expr(x)?).into())
}

#[pyfunction]
#[pyo3(signature = (value: "Expr | float"))]
/// Apply inverse cosine to an expression-like value.
///
/// Parameters
/// ----------
/// value : Expr or float
///     Input expression.
///
/// Returns
/// -------
/// Expr
///     Inverse cosine of `value`.
///
/// Raises
/// ------
/// TypeError
///     If `value` cannot be converted to an expression.
pub fn acos(value: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
    Ok(expr_acos(extract_expr(value)?).into())
}

#[pyfunction]
#[pyo3(signature = (elements: "Sequence[Expr | complex]"))]
/// Construct a vector-valued expression.
///
/// Parameters
/// ----------
/// elements : sequence of Expr or complex
///     Vector elements in order.
///
/// Returns
/// -------
/// Expr
///     Vector-valued symbolic expression.
///
/// Raises
/// ------
/// TypeError
///     If an element cannot be converted to an expression.
pub fn vector(elements: Vec<Bound<'_, PyAny>>) -> PyResult<PyExpr> {
    let elements = elements
        .iter()
        .map(extract_expr)
        .collect::<PyResult<Vec<_>>>()?;
    Ok(expr_vector(elements).into())
}

#[pyfunction]
#[pyo3(signature = (
    elements: "Sequence[Sequence[Expr | complex]]"
))]
/// Construct a matrix-valued expression from rows.
///
/// Parameters
/// ----------
/// elements : sequence of sequence of Expr or complex
///     Rectangular matrix in row-major order.
///
/// Returns
/// -------
/// Expr
///     Matrix-valued symbolic expression.
///
/// Raises
/// ------
/// TypeError
///     If rows have unequal lengths or an element cannot be converted.
pub fn matrix(elements: Vec<Vec<Bound<'_, PyAny>>>) -> PyResult<PyExpr> {
    let rows = elements.len();
    let cols = elements.first().map_or(0, Vec::len);
    if elements.iter().any(|row| row.len() != cols) {
        return Err(PyTypeError::new_err("matrix rows must have equal length"));
    }
    let flat = elements
        .iter()
        .flatten()
        .map(extract_expr)
        .collect::<PyResult<Vec<_>>>()?;
    matrix_from_flat(rows, cols, flat)
        .map(PyExpr::from)
        .map_err(to_py_err)
}

macro_rules! binary_function {
    ($name:ident, $function:ident, $doc:literal) => {
        #[pyfunction]
        #[pyo3(signature = (lhs: "Expr | complex", rhs: "Expr | complex"))]
        #[doc = $doc]
        pub fn $name(lhs: &Bound<'_, PyAny>, rhs: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
            Ok($function(extract_expr(lhs)?, extract_expr(rhs)?).into())
        }
    };
}

binary_function!(
    dot,
    expr_dot,
    "Compute the symbolic dot product of two vector expressions.\n\nParameters\n----------\nlhs, rhs : Expr\n    Vector-valued expressions with matching lengths.\n\nReturns\n-------\nExpr\n    Scalar dot-product expression.\n\nRaises\n------\nTypeError\n    If either operand cannot be converted to an expression."
);
binary_function!(
    matmul,
    expr_matmul,
    "Compute symbolic matrix multiplication.\n\nParameters\n----------\nlhs, rhs : Expr\n    Matrix-valued expressions with compatible dimensions.\n\nReturns\n-------\nExpr\n    Matrix product expression.\n\nRaises\n------\nTypeError\n    If either operand cannot be converted to an expression."
);
binary_function!(
    matvec,
    expr_matvec,
    "Multiply a symbolic matrix by a symbolic vector.\n\nParameters\n----------\nlhs : Expr\n    Matrix-valued expression.\nrhs : Expr\n    Vector-valued expression with a compatible length.\n\nReturns\n-------\nExpr\n    Vector product expression.\n\nRaises\n------\nTypeError\n    If either operand cannot be converted to an expression."
);
binary_function!(
    solve,
    expr_solve,
    "Solve a symbolic linear system.\n\nParameters\n----------\nlhs : Expr\n    Square coefficient-matrix expression.\nrhs : Expr\n    Right-hand-side vector or matrix expression.\n\nReturns\n-------\nExpr\n    Symbolic solution of the system.\n\nRaises\n------\nTypeError\n    If either operand cannot be converted to an expression."
);

impl_json_methods!(PyExpr);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visualization_methods_render_all_formats() {
        let shared = event_scalar("x");
        let expression = PyExpr::from((shared.clone() + 1.0) * (shared + 2.0));

        assert!(expression.equation(None, None).unwrap().contains('x'));
        assert!(expression.latex(None, None).unwrap().contains('x'));

        let expanded = expression.tree(None, true, None).unwrap();
        assert_eq!(expanded.matches("EventScalar(x)").count(), 2);

        let referenced = expression.tree(None, false, None).unwrap();
        assert_eq!(referenced.matches("EventScalar(x)").count(), 1);
        assert!(referenced.contains("<reference to #"));

        let dot = expression.dot(None, false, None).unwrap();
        assert!(dot.starts_with("digraph ExprGraph"));
        assert_eq!(dot.matches("EventScalar(x)").count(), 1);

        let path = std::env::temp_dir().join(format!(
            "laddu-expression-visualization-{}.svg",
            std::process::id()
        ));
        expression.svg(path.clone(), None, false, None).unwrap();
        let svg = std::fs::read_to_string(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
    }
}
