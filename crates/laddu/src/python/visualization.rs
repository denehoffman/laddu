use laddu_expr::{
    ColorPreset, DisplayColor, ExprGraph, ExprNodeKind, NodeSelector, NodeStyle, NodeStyleRule,
};
use pyo3::{
    exceptions::{PyOSError, PyRuntimeError, PyTypeError, PyValueError},
    prelude::*,
    types::PyAny,
};

#[pyclass(
    name = "ExprNodeKind",
    module = "laddu",
    frozen,
    eq,
    eq_int,
    from_py_object,
    rename_all = "SCREAMING_SNAKE_CASE"
)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// A category of node in an expression graph.
pub enum PyExprNodeKind {
    /// Real constant node.
    RealConst,
    /// Complex constant node.
    ComplexConst,
    /// Scalar parameter node.
    ScalarParam,
    /// Scalar event-data node.
    EventScalar,
    /// Four-momentum component event-data node.
    EventP4Component,
    /// Unary-operation node.
    Unary,
    /// Binary-operation node.
    Binary,
    /// N-ary addition node.
    NaryAdd,
    /// N-ary multiplication node.
    NaryMul,
    /// Complex-construction node.
    Complex,
    /// Vector-construction node.
    Vector,
    /// Matrix-construction node.
    Matrix,
    /// Vector-component node.
    Component,
    /// Matrix-element node.
    MatrixElement,
    /// Matrix-matrix multiplication node.
    MatMul,
    /// Matrix-vector multiplication node.
    MatVec,
    /// Dot-product node.
    Dot,
    /// Linear-system solution node.
    Solve,
}

impl From<PyExprNodeKind> for ExprNodeKind {
    fn from(value: PyExprNodeKind) -> Self {
        match value {
            PyExprNodeKind::RealConst => Self::RealConst,
            PyExprNodeKind::ComplexConst => Self::ComplexConst,
            PyExprNodeKind::ScalarParam => Self::ScalarParam,
            PyExprNodeKind::EventScalar => Self::EventScalar,
            PyExprNodeKind::EventP4Component => Self::EventP4Component,
            PyExprNodeKind::Unary => Self::Unary,
            PyExprNodeKind::Binary => Self::Binary,
            PyExprNodeKind::NaryAdd => Self::NaryAdd,
            PyExprNodeKind::NaryMul => Self::NaryMul,
            PyExprNodeKind::Complex => Self::Complex,
            PyExprNodeKind::Vector => Self::Vector,
            PyExprNodeKind::Matrix => Self::Matrix,
            PyExprNodeKind::Component => Self::Component,
            PyExprNodeKind::MatrixElement => Self::MatrixElement,
            PyExprNodeKind::MatMul => Self::MatMul,
            PyExprNodeKind::MatVec => Self::MatVec,
            PyExprNodeKind::Dot => Self::Dot,
            PyExprNodeKind::Solve => Self::Solve,
        }
    }
}

#[pyclass(
    name = "DisplayColor",
    module = "laddu",
    frozen,
    get_all,
    eq,
    from_py_object
)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// An RGB color used by expression displays.
pub struct PyDisplayColor {
    /// Red channel.
    pub red: u8,
    /// Green channel.
    pub green: u8,
    /// Blue channel.
    pub blue: u8,
}

#[pymethods]
impl PyDisplayColor {
    #[new]
    fn new(red: u8, green: u8, blue: u8) -> Self {
        Self { red, green, blue }
    }

    fn __repr__(&self) -> String {
        format!("DisplayColor({}, {}, {})", self.red, self.green, self.blue)
    }
}

impl From<PyDisplayColor> for DisplayColor {
    fn from(value: PyDisplayColor) -> Self {
        Self::rgb(value.red, value.green, value.blue)
    }
}

#[pyclass(
    name = "NodeStyle",
    module = "laddu",
    frozen,
    get_all,
    eq,
    from_py_object
)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Optional text, background, and outline colors for matching nodes.
pub struct PyNodeStyle {
    /// Text color.
    pub foreground: Option<PyDisplayColor>,
    /// Terminal background or Graphviz fill color.
    pub fill: Option<PyDisplayColor>,
    /// Graphviz outline color.
    pub border: Option<PyDisplayColor>,
}

#[pymethods]
impl PyNodeStyle {
    #[new]
    #[pyo3(signature = (*, foreground=None, fill=None, border=None))]
    fn new(
        foreground: Option<PyDisplayColor>,
        fill: Option<PyDisplayColor>,
        border: Option<PyDisplayColor>,
    ) -> Self {
        Self {
            foreground,
            fill,
            border,
        }
    }
}

impl From<PyNodeStyle> for NodeStyle {
    fn from(value: PyNodeStyle) -> Self {
        Self {
            foreground: value.foreground.map(Into::into),
            fill: value.fill.map(Into::into),
            border: value.border.map(Into::into),
        }
    }
}

#[pyclass(name = "NodeSelector", module = "laddu", frozen, eq, from_py_object)]
#[derive(Clone, Debug, PartialEq, Eq)]
/// A predicate selecting expression nodes for a style rule.
pub struct PyNodeSelector {
    pub(crate) inner: NodeSelector,
}

#[pymethods]
impl PyNodeSelector {
    /// Select every node.
    #[staticmethod]
    #[pyo3(name = "any")]
    fn any_selector() -> Self {
        Self {
            inner: NodeSelector::Any,
        }
    }

    /// Select nodes of a category.
    #[staticmethod]
    fn kind(kind: PyExprNodeKind) -> Self {
        Self {
            inner: NodeSelector::Kind(kind.into()),
        }
    }

    /// Select nodes with a matching metadata or source name.
    #[staticmethod]
    fn name(name: String) -> Self {
        Self {
            inner: NodeSelector::Name(name),
        }
    }

    /// Select nodes carrying a metadata tag.
    #[staticmethod]
    fn tag(tag: String) -> Self {
        Self {
            inner: NodeSelector::Tag(tag),
        }
    }
}

#[pyclass(
    name = "NodeStyleRule",
    module = "laddu",
    frozen,
    get_all,
    eq,
    from_py_object
)]
#[derive(Clone, Debug, PartialEq, Eq)]
/// A selector and style to overlay on matching expression nodes.
pub struct PyNodeStyleRule {
    /// Predicate used to select nodes.
    pub selector: PyNodeSelector,
    /// Style overlaid on selected nodes.
    pub style: PyNodeStyle,
}

#[pymethods]
impl PyNodeStyleRule {
    #[new]
    fn new(selector: PyNodeSelector, style: PyNodeStyle) -> Self {
        Self { selector, style }
    }
}

impl From<PyNodeStyleRule> for NodeStyleRule {
    fn from(value: PyNodeStyleRule) -> Self {
        Self::new(value.selector.inner, value.style.into())
    }
}

fn color_preset(colors: Option<&Bound<'_, PyAny>>) -> PyResult<Option<ColorPreset>> {
    let Some(colors) = colors else {
        return Ok(None);
    };
    if let Ok(value) = colors.extract::<String>() {
        return match value.trim().to_ascii_lowercase().as_str() {
            "none" => Ok(None),
            "light" => Ok(Some(ColorPreset::Light)),
            "dark" => Ok(Some(ColorPreset::Dark)),
            _ => Err(PyValueError::new_err(
                "colors must be 'light', 'dark', 'none', or None",
            )),
        };
    }
    Err(PyTypeError::new_err("colors must be a string or None"))
}

pub(crate) fn expression_equation(
    graph: &ExprGraph,
    colors: Option<&Bound<'_, PyAny>>,
    style_rules: Option<Vec<PyNodeStyleRule>>,
) -> PyResult<String> {
    let mut display = graph.display_equation();
    if let Some(preset) = color_preset(colors)? {
        display = display.with_preset(preset);
    }
    for rule in style_rules.unwrap_or_default() {
        display = display.with_style_rule(rule.into());
    }
    Ok(display.to_string())
}

pub(crate) fn expression_latex(
    graph: &ExprGraph,
    colors: Option<&Bound<'_, PyAny>>,
    style_rules: Option<Vec<PyNodeStyleRule>>,
) -> PyResult<String> {
    let mut display = graph.display_latex();
    if let Some(preset) = color_preset(colors)? {
        display = display.with_preset(preset);
    }
    for rule in style_rules.unwrap_or_default() {
        display = display.with_style_rule(rule.into());
    }
    Ok(display.to_string())
}

pub(crate) fn expression_tree(
    graph: &ExprGraph,
    colors: Option<&Bound<'_, PyAny>>,
    expand_repeated: bool,
    style_rules: Option<Vec<PyNodeStyleRule>>,
) -> PyResult<String> {
    let mut display = graph.display_tree().expand_repeated(expand_repeated);
    if let Some(preset) = color_preset(colors)? {
        display = display.with_preset(preset);
    }
    for rule in style_rules.unwrap_or_default() {
        display = display.with_style_rule(rule.into());
    }
    Ok(display.to_string())
}

pub(crate) fn expression_dot(
    graph: &ExprGraph,
    colors: Option<&Bound<'_, PyAny>>,
    expand_repeated: bool,
    style_rules: Option<Vec<PyNodeStyleRule>>,
) -> PyResult<String> {
    let mut display = graph.display_dot().expand_repeated(expand_repeated);
    if let Some(preset) = color_preset(colors)? {
        display = display.with_preset(preset);
    }
    for rule in style_rules.unwrap_or_default() {
        display = display.with_style_rule(rule.into());
    }
    Ok(display.to_string())
}

pub(crate) fn expression_svg(
    graph: &ExprGraph,
    path: &Path,
    colors: Option<&Bound<'_, PyAny>>,
    expand_repeated: bool,
    style_rules: Option<Vec<PyNodeStyleRule>>,
) -> PyResult<()> {
    let mut display = graph.display_dot().expand_repeated(expand_repeated);
    if let Some(preset) = color_preset(colors)? {
        display = display.with_preset(preset);
    }
    for rule in style_rules.unwrap_or_default() {
        display = display.with_style_rule(rule.into());
    }
    let svg = display.render_svg().map_err(|error| {
        PyRuntimeError::new_err(format!("failed to render expression graph as SVG: {error}"))
    })?;
    fs::write(path, svg).map_err(|error| {
        PyOSError::new_err(format!(
            "failed to write SVG to '{}': {error}",
            path.display()
        ))
    })
}
use std::{fs, path::Path};
