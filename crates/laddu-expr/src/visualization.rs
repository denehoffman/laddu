use std::{collections::HashSet, fmt};

use num::complex::Complex64;

use crate::{
    BinaryOp, ExprGraph, ExprId, ExprMetadata, ExprNode, UnaryOp, expression::node_children,
};

/// Node categories available to visualization style selectors.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ExprNodeKind {
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

impl ExprNodeKind {
    /// Returns the category corresponding to `node`.
    pub fn of(node: &ExprNode) -> Self {
        match node {
            ExprNode::RealConst(_) => Self::RealConst,
            ExprNode::ComplexConst(_) => Self::ComplexConst,
            ExprNode::ScalarParam(_) => Self::ScalarParam,
            ExprNode::EventScalar(_) => Self::EventScalar,
            ExprNode::EventP4Component { .. } => Self::EventP4Component,
            ExprNode::Unary { .. } => Self::Unary,
            ExprNode::Binary { .. } => Self::Binary,
            ExprNode::NaryAdd { .. } => Self::NaryAdd,
            ExprNode::NaryMul { .. } => Self::NaryMul,
            ExprNode::Complex { .. } => Self::Complex,
            ExprNode::Vector { .. } => Self::Vector,
            ExprNode::Matrix { .. } => Self::Matrix,
            ExprNode::Component { .. } => Self::Component,
            ExprNode::MatrixElement { .. } => Self::MatrixElement,
            ExprNode::MatMul { .. } => Self::MatMul,
            ExprNode::MatVec { .. } => Self::MatVec,
            ExprNode::Dot { .. } => Self::Dot,
            ExprNode::Solve { .. } => Self::Solve,
        }
    }
}

/// An RGB color used by tree and Graphviz displays.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DisplayColor {
    red: u8,
    green: u8,
    blue: u8,
}

impl DisplayColor {
    /// Creates a color from red, green, and blue channels.
    pub const fn rgb(red: u8, green: u8, blue: u8) -> Self {
        Self { red, green, blue }
    }

    fn dot(self) -> String {
        format!("#{:02x}{:02x}{:02x}", self.red, self.green, self.blue)
    }

    fn ansi_foreground(self) -> String {
        format!("\x1b[38;2;{};{};{}m", self.red, self.green, self.blue)
    }

    fn ansi_background(self) -> String {
        format!("\x1b[48;2;{};{};{}m", self.red, self.green, self.blue)
    }
}

/// Optional foreground, fill, and border colors for a displayed node.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct NodeStyle {
    /// Text color.
    pub foreground: Option<DisplayColor>,
    /// Background or fill color.
    pub fill: Option<DisplayColor>,
    /// Outline color.
    pub border: Option<DisplayColor>,
}

impl NodeStyle {
    /// Creates a style with no color overrides.
    pub const fn new() -> Self {
        Self {
            foreground: None,
            fill: None,
            border: None,
        }
    }

    /// Sets the text color.
    pub const fn with_foreground(mut self, color: DisplayColor) -> Self {
        self.foreground = Some(color);
        self
    }

    /// Sets the background or fill color.
    pub const fn with_fill(mut self, color: DisplayColor) -> Self {
        self.fill = Some(color);
        self
    }

    /// Sets the outline color.
    pub const fn with_border(mut self, color: DisplayColor) -> Self {
        self.border = Some(color);
        self
    }

    fn overlay(&mut self, other: Self) {
        if other.foreground.is_some() {
            self.foreground = other.foreground;
        }
        if other.fill.is_some() {
            self.fill = other.fill;
        }
        if other.border.is_some() {
            self.border = other.border;
        }
    }

    fn ansi(self, text: String) -> String {
        if self.foreground.is_none() && self.fill.is_none() {
            return text;
        }
        let mut prefix = String::new();
        if let Some(color) = self.foreground {
            prefix.push_str(&color.ansi_foreground());
        }
        if let Some(color) = self.fill {
            prefix.push_str(&color.ansi_background());
        }
        format!("{prefix}{text}\x1b[0m")
    }

    fn latex(self, text: String) -> String {
        self.foreground.map_or(text.clone(), |color| {
            format!(
                "{{\\color[RGB]{{{},{},{}}}{text}}}",
                color.red, color.green, color.blue
            )
        })
    }
}

/// Predicate selecting expression nodes for a [`NodeStyleRule`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NodeSelector {
    /// Select every node.
    Any,
    /// Select nodes in a category.
    Kind(ExprNodeKind),
    /// Select nodes with a matching metadata or source name.
    Name(String),
    /// Select nodes carrying a metadata tag.
    Tag(String),
}

impl NodeSelector {
    fn matches(&self, node: &ExprNode, metadata: Option<&ExprMetadata>) -> bool {
        match self {
            Self::Any => true,
            Self::Kind(kind) => *kind == ExprNodeKind::of(node),
            Self::Name(name) => {
                metadata.and_then(ExprMetadata::name) == Some(name.as_str())
                    || match node {
                        ExprNode::ScalarParam(parameter) => parameter.name() == name,
                        ExprNode::EventScalar(node_name)
                        | ExprNode::EventP4Component {
                            name: node_name, ..
                        } => node_name.as_ref() == name,
                        _ => false,
                    }
            }
            Self::Tag(tag) => metadata.is_some_and(|metadata| metadata.has_tag(tag)),
        }
    }
}

/// A selector and the style to overlay on matching nodes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NodeStyleRule {
    /// Predicate used to select nodes.
    pub selector: NodeSelector,
    /// Style overlaid on selected nodes.
    pub style: NodeStyle,
}

impl NodeStyleRule {
    /// Creates a style rule from a selector and style.
    pub fn new(selector: NodeSelector, style: NodeStyle) -> Self {
        Self { selector, style }
    }
}

/// Built-in color palette for expression graphs.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ColorPreset {
    /// Colors selected for light backgrounds.
    Light,
    /// Colors selected for dark backgrounds.
    Dark,
}

#[derive(Clone, Debug)]
struct DisplayOptions {
    expand_repeated: bool,
    rules: Vec<NodeStyleRule>,
}

impl Default for DisplayOptions {
    fn default() -> Self {
        Self {
            expand_repeated: true,
            rules: Vec::new(),
        }
    }
}

impl DisplayOptions {
    fn with_preset(&mut self, preset: ColorPreset) {
        let (constant, parameter, event, operation, linear_algebra) = match preset {
            ColorPreset::Light => (
                DisplayColor::rgb(88, 96, 105),
                DisplayColor::rgb(0, 92, 197),
                DisplayColor::rgb(3, 102, 214),
                DisplayColor::rgb(130, 80, 223),
                DisplayColor::rgb(207, 34, 46),
            ),
            ColorPreset::Dark => (
                DisplayColor::rgb(139, 148, 158),
                DisplayColor::rgb(88, 166, 255),
                DisplayColor::rgb(121, 192, 255),
                DisplayColor::rgb(210, 168, 255),
                DisplayColor::rgb(255, 123, 114),
            ),
        };
        let style = |color| NodeStyle::new().with_foreground(color).with_border(color);
        for kind in [ExprNodeKind::RealConst, ExprNodeKind::ComplexConst] {
            self.rules.push(NodeStyleRule::new(
                NodeSelector::Kind(kind),
                style(constant),
            ));
        }
        self.rules.push(NodeStyleRule::new(
            NodeSelector::Kind(ExprNodeKind::ScalarParam),
            style(parameter),
        ));
        for kind in [ExprNodeKind::EventScalar, ExprNodeKind::EventP4Component] {
            self.rules
                .push(NodeStyleRule::new(NodeSelector::Kind(kind), style(event)));
        }
        for kind in [
            ExprNodeKind::Unary,
            ExprNodeKind::Binary,
            ExprNodeKind::NaryAdd,
            ExprNodeKind::NaryMul,
            ExprNodeKind::Complex,
            ExprNodeKind::Vector,
            ExprNodeKind::Matrix,
            ExprNodeKind::Component,
            ExprNodeKind::MatrixElement,
        ] {
            self.rules.push(NodeStyleRule::new(
                NodeSelector::Kind(kind),
                style(operation),
            ));
        }
        for kind in [
            ExprNodeKind::MatMul,
            ExprNodeKind::MatVec,
            ExprNodeKind::Dot,
            ExprNodeKind::Solve,
        ] {
            self.rules.push(NodeStyleRule::new(
                NodeSelector::Kind(kind),
                style(linear_algebra),
            ));
        }
    }

    fn resolve(&self, graph: &ExprGraph, id: ExprId, node: &ExprNode) -> NodeStyle {
        let mut style = NodeStyle::default();
        let metadata = graph.metadata(id);
        for rule in &self.rules {
            if rule.selector.matches(node, metadata) {
                style.overlay(rule.style);
            }
        }
        style
    }
}

impl ExprGraph {
    /// Creates a configurable indented-tree display.
    pub fn display_tree(&self) -> crate::ExprGraphTreeDisplay<'_> {
        crate::ExprGraphTreeDisplay::new(self)
    }

    /// Creates a configurable compact-equation display.
    pub fn display_equation(&self) -> crate::ExprGraphEquationDisplay<'_> {
        crate::ExprGraphEquationDisplay::new(self)
    }

    /// Creates a configurable LaTeX equation display.
    pub fn display_latex(&self) -> crate::ExprGraphLatexDisplay<'_> {
        crate::ExprGraphLatexDisplay::new(self)
    }

    /// Creates a configurable Graphviz DOT display.
    pub fn display_dot(&self) -> crate::ExprGraphDotDisplay<'_> {
        crate::ExprGraphDotDisplay::new(self)
    }

    fn format_expression(&self, id: ExprId) -> String {
        self.format_expression_with(id, &|_, _, text| text)
    }

    pub(crate) fn format_expression_with(
        &self,
        id: ExprId,
        decorate: &dyn Fn(ExprId, &ExprNode, String) -> String,
    ) -> String {
        self.format_child(id, ExprPrecedence::Lowest, false, decorate)
    }

    fn format_child(
        &self,
        id: ExprId,
        parent_precedence: ExprPrecedence,
        parenthesize_equal: bool,
        decorate: &dyn Fn(ExprId, &ExprNode, String) -> String,
    ) -> String {
        let (text, precedence) = self.format_node_expression(id, decorate);
        if precedence < parent_precedence || (parenthesize_equal && precedence == parent_precedence)
        {
            format!("({text})")
        } else {
            text
        }
    }

    fn format_node_expression(
        &self,
        id: ExprId,
        decorate: &dyn Fn(ExprId, &ExprNode, String) -> String,
    ) -> (String, ExprPrecedence) {
        let Some(node) = self.node(id) else {
            return (format!("<missing #{}>", id.index()), ExprPrecedence::Atom);
        };

        let (text, precedence) = match node {
            ExprNode::RealConst(value) => (Self::format_real_number(*value), ExprPrecedence::Atom),
            ExprNode::ComplexConst(value) => self.format_complex_const(*value),
            ExprNode::ScalarParam(parameter) => (parameter.name().to_owned(), ExprPrecedence::Atom),
            ExprNode::EventScalar(name) => (name.to_string(), ExprPrecedence::Atom),
            ExprNode::EventP4Component { name, component } => (
                format!("{name}.{}", component.label()),
                ExprPrecedence::Atom,
            ),
            ExprNode::Unary { op, input } => self.format_unary_expression(*op, *input, decorate),
            ExprNode::Binary { op, lhs, rhs } => {
                self.format_binary_expression(*op, *lhs, *rhs, decorate)
            }
            ExprNode::NaryAdd { terms } => self.format_sum_expression(terms, decorate),
            ExprNode::NaryMul { factors } => self.format_product_expression(factors, decorate),
            ExprNode::Complex { re, im } => (
                format!(
                    "complex({}, {})",
                    self.format_expression_with(*re, decorate),
                    self.format_expression_with(*im, decorate)
                ),
                ExprPrecedence::Atom,
            ),
            ExprNode::Vector { elements } => (
                format!(
                    "[{}]",
                    elements
                        .iter()
                        .map(|id| self.format_expression_with(*id, decorate))
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
                ExprPrecedence::Atom,
            ),
            ExprNode::Matrix {
                rows,
                cols,
                elements,
            } => {
                let rows = (0..*rows)
                    .map(|row| {
                        let start = row * *cols;
                        let end = start + *cols;
                        format!(
                            "[{}]",
                            elements[start..end]
                                .iter()
                                .map(|id| self.format_expression_with(*id, decorate))
                                .collect::<Vec<_>>()
                                .join(", ")
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                (format!("[{rows}]"), ExprPrecedence::Atom)
            }
            ExprNode::Component { input, index } => (
                format!(
                    "{}[{index}]",
                    self.format_child(*input, ExprPrecedence::Postfix, false, decorate)
                ),
                ExprPrecedence::Postfix,
            ),
            ExprNode::MatrixElement { input, row, col } => (
                format!(
                    "{}[{row}, {col}]",
                    self.format_child(*input, ExprPrecedence::Postfix, false, decorate)
                ),
                ExprPrecedence::Postfix,
            ),
            ExprNode::MatMul { lhs, rhs } => (
                format!(
                    "matmul({}, {})",
                    self.format_expression_with(*lhs, decorate),
                    self.format_expression_with(*rhs, decorate)
                ),
                ExprPrecedence::Atom,
            ),
            ExprNode::MatVec { matrix, vector } => (
                format!(
                    "matvec({}, {})",
                    self.format_expression_with(*matrix, decorate),
                    self.format_expression_with(*vector, decorate)
                ),
                ExprPrecedence::Atom,
            ),
            ExprNode::Dot { lhs, rhs } => (
                format!(
                    "dot({}, {})",
                    self.format_expression_with(*lhs, decorate),
                    self.format_expression_with(*rhs, decorate)
                ),
                ExprPrecedence::Atom,
            ),
            ExprNode::Solve { matrix, rhs } => (
                format!(
                    "solve({}, {})",
                    self.format_expression_with(*matrix, decorate),
                    self.format_expression_with(*rhs, decorate)
                ),
                ExprPrecedence::Atom,
            ),
        };
        (decorate(id, node, text), precedence)
    }

    fn format_unary_expression(
        &self,
        op: UnaryOp,
        input: ExprId,
        decorate: &dyn Fn(ExprId, &ExprNode, String) -> String,
    ) -> (String, ExprPrecedence) {
        match op {
            UnaryOp::Neg => (
                format!(
                    "-{}",
                    self.format_child(input, ExprPrecedence::Unary, true, decorate)
                ),
                ExprPrecedence::Unary,
            ),
            UnaryOp::Real => (
                self.format_call_expression("real", input, decorate),
                ExprPrecedence::Atom,
            ),
            UnaryOp::Imag => (
                self.format_call_expression("imag", input, decorate),
                ExprPrecedence::Atom,
            ),
            UnaryOp::Conj => (
                self.format_call_expression("conj", input, decorate),
                ExprPrecedence::Atom,
            ),
            UnaryOp::NormSqr => (
                format!("|{}|^2", self.format_expression_with(input, decorate)),
                ExprPrecedence::Pow,
            ),
            UnaryOp::Sqrt => (
                self.format_call_expression("sqrt", input, decorate),
                ExprPrecedence::Atom,
            ),
            UnaryOp::Exp => (
                self.format_call_expression("exp", input, decorate),
                ExprPrecedence::Atom,
            ),
            UnaryOp::Sin => (
                self.format_call_expression("sin", input, decorate),
                ExprPrecedence::Atom,
            ),
            UnaryOp::Cos => (
                self.format_call_expression("cos", input, decorate),
                ExprPrecedence::Atom,
            ),
            UnaryOp::Log => (
                self.format_call_expression("log", input, decorate),
                ExprPrecedence::Atom,
            ),
            UnaryOp::PowI(power) => {
                let exponent = if power < 0 {
                    format!("({power})")
                } else {
                    power.to_string()
                };
                (
                    format!(
                        "{}^{exponent}",
                        self.format_child(input, ExprPrecedence::Pow, true, decorate)
                    ),
                    ExprPrecedence::Pow,
                )
            }
        }
    }

    fn format_call_expression(
        &self,
        name: &str,
        input: ExprId,
        decorate: &dyn Fn(ExprId, &ExprNode, String) -> String,
    ) -> String {
        format!("{name}({})", self.format_expression_with(input, decorate))
    }

    fn format_binary_expression(
        &self,
        op: BinaryOp,
        lhs: ExprId,
        rhs: ExprId,
        decorate: &dyn Fn(ExprId, &ExprNode, String) -> String,
    ) -> (String, ExprPrecedence) {
        match op {
            BinaryOp::Add => self.format_sum_expression(&[lhs, rhs], decorate),
            BinaryOp::Sub => {
                let lhs = self.format_child(lhs, ExprPrecedence::Add, false, decorate);
                let rhs = self.format_child(rhs, ExprPrecedence::Add, true, decorate);
                (format!("{lhs} - {rhs}"), ExprPrecedence::Add)
            }
            BinaryOp::Mul => self.format_product_expression(&[lhs, rhs], decorate),
            BinaryOp::Div => {
                let lhs = self.format_child(lhs, ExprPrecedence::Mul, false, decorate);
                let rhs = self.format_child(rhs, ExprPrecedence::Mul, true, decorate);
                (format!("{lhs} / {rhs}"), ExprPrecedence::Mul)
            }
            BinaryOp::Atan2 => (
                format!(
                    "atan2({}, {})",
                    self.format_expression_with(lhs, decorate),
                    self.format_expression_with(rhs, decorate)
                ),
                ExprPrecedence::Atom,
            ),
        }
    }

    fn format_sum_expression(
        &self,
        terms: &[ExprId],
        decorate: &dyn Fn(ExprId, &ExprNode, String) -> String,
    ) -> (String, ExprPrecedence) {
        let mut formatted = String::new();
        for term in terms {
            let (negative, term) = self.format_signed_term(*term, decorate);
            if formatted.is_empty() {
                if negative {
                    formatted.push('-');
                }
                formatted.push_str(&term);
            } else if negative {
                formatted.push_str(" - ");
                formatted.push_str(&term);
            } else {
                formatted.push_str(" + ");
                formatted.push_str(&term);
            }
        }
        if formatted.is_empty() {
            formatted.push('0');
        }
        (formatted, ExprPrecedence::Add)
    }

    fn format_signed_term(
        &self,
        id: ExprId,
        decorate: &dyn Fn(ExprId, &ExprNode, String) -> String,
    ) -> (bool, String) {
        match self.node(id) {
            Some(ExprNode::RealConst(value)) if *value < 0.0 => (
                true,
                decorate(
                    id,
                    self.node(id).unwrap(),
                    Self::format_real_number(-*value),
                ),
            ),
            Some(ExprNode::Unary {
                op: UnaryOp::Neg,
                input,
            }) => (
                true,
                self.format_child(*input, ExprPrecedence::Add, false, decorate),
            ),
            Some(ExprNode::NaryMul { factors }) => {
                let (negative, product) = self.format_product_parts(factors, decorate);
                (negative, product)
            }
            _ => (
                false,
                self.format_child(id, ExprPrecedence::Add, false, decorate),
            ),
        }
    }

    fn format_product_expression(
        &self,
        factors: &[ExprId],
        decorate: &dyn Fn(ExprId, &ExprNode, String) -> String,
    ) -> (String, ExprPrecedence) {
        let (negative, product) = self.format_product_parts(factors, decorate);
        if negative {
            (format!("-{product}"), ExprPrecedence::Unary)
        } else {
            (product, ExprPrecedence::Mul)
        }
    }

    fn format_product_parts(
        &self,
        factors: &[ExprId],
        decorate: &dyn Fn(ExprId, &ExprNode, String) -> String,
    ) -> (bool, String) {
        let mut negative = false;
        let mut pieces = Vec::new();

        for factor in factors {
            match self.node(*factor) {
                Some(ExprNode::RealConst(value)) if *value < 0.0 => {
                    negative = !negative;
                    if *value != -1.0 || factors.len() == 1 {
                        pieces.push(decorate(
                            *factor,
                            self.node(*factor).unwrap(),
                            Self::format_real_number(-*value),
                        ));
                    }
                }
                Some(ExprNode::Unary {
                    op: UnaryOp::Neg,
                    input,
                }) => {
                    negative = !negative;
                    pieces.push(self.format_child(*input, ExprPrecedence::Mul, false, decorate));
                }
                _ => pieces.push(self.format_child(*factor, ExprPrecedence::Mul, false, decorate)),
            }
        }

        if pieces.is_empty() {
            pieces.push("1".to_owned());
        }

        (negative, pieces.join(" * "))
    }

    fn format_complex_const(&self, value: Complex64) -> (String, ExprPrecedence) {
        match (value.re, value.im) {
            (re, 0.0) => (Self::format_real_number(re), ExprPrecedence::Atom),
            (0.0, im) => (Self::format_imaginary_unit(im), ExprPrecedence::Atom),
            (re, im) if im < 0.0 => (
                format!(
                    "{} - {}",
                    Self::format_real_number(re),
                    Self::format_imaginary_unit(-im)
                ),
                ExprPrecedence::Add,
            ),
            (re, im) => (
                format!(
                    "{} + {}",
                    Self::format_real_number(re),
                    Self::format_imaginary_unit(im)
                ),
                ExprPrecedence::Add,
            ),
        }
    }

    fn format_real_number(value: f64) -> String {
        let Some((value, decimals)) = Self::nearby_simple_decimal(value) else {
            return value.to_string();
        };

        if decimals == 0 {
            return value.to_string();
        }

        let mut formatted = format!("{value:.decimals$}");
        while formatted.contains('.') && formatted.ends_with('0') {
            formatted.pop();
        }
        if formatted.ends_with('.') {
            formatted.pop();
        }
        formatted
    }

    fn nearby_simple_decimal(value: f64) -> Option<(f64, usize)> {
        if !value.is_finite() {
            return None;
        }

        for decimals in 0..=12 {
            let scale = 10_f64.powi(decimals as i32);
            let rounded = (value * scale).round() / scale;
            if Self::nearly_equal(value, rounded) {
                return Some((rounded, decimals));
            }
        }

        None
    }

    fn nearly_equal(lhs: f64, rhs: f64) -> bool {
        (lhs - rhs).abs() <= f64::EPSILON * lhs.abs().max(rhs.abs()).max(1.0) * 16.0
    }

    fn format_imaginary_unit(value: f64) -> String {
        match value {
            1.0 => "i".to_owned(),
            -1.0 => "-i".to_owned(),
            value => format!("{}i", Self::format_real_number(value)),
        }
    }

    pub(crate) fn node_label(&self, id: ExprId, node: &ExprNode) -> String {
        let mut label = match node {
            ExprNode::RealConst(value) => {
                format!(
                    "#{} RealConst({})",
                    id.index(),
                    Self::format_real_number(*value)
                )
            }
            ExprNode::ComplexConst(value) => {
                let (value, _) = self.format_complex_const(*value);
                format!("#{} ComplexConst({value})", id.index())
            }
            ExprNode::ScalarParam(parameter) => {
                format!("#{} ScalarParam({})", id.index(), parameter.name())
            }
            ExprNode::EventScalar(name) => format!("#{} EventScalar({name})", id.index()),
            ExprNode::EventP4Component { name, component } => {
                format!(
                    "#{} EventP4Component({name}.{})",
                    id.index(),
                    component.label()
                )
            }
            ExprNode::Unary { op, .. } => format!("#{} Unary({op:?})", id.index()),
            ExprNode::Binary { op, .. } => format!("#{} Binary({op:?})", id.index()),
            ExprNode::NaryAdd { terms } => {
                format!("#{} NaryAdd(len={})", id.index(), terms.len())
            }
            ExprNode::NaryMul { factors } => {
                format!("#{} NaryMul(len={})", id.index(), factors.len())
            }
            ExprNode::Complex { .. } => format!("#{} Complex", id.index()),
            ExprNode::Vector { elements } => {
                format!("#{} Vector(len={})", id.index(), elements.len())
            }
            ExprNode::Matrix { rows, cols, .. } => {
                format!("#{} Matrix({rows}x{cols})", id.index())
            }
            ExprNode::Component { index, .. } => {
                format!("#{} Component(index={index})", id.index())
            }
            ExprNode::MatrixElement { row, col, .. } => {
                format!("#{} MatrixElement(row={row}, col={col})", id.index())
            }
            ExprNode::MatMul { .. } => format!("#{} MatMul", id.index()),
            ExprNode::MatVec { .. } => format!("#{} MatVec", id.index()),
            ExprNode::Dot { .. } => format!("#{} Dot", id.index()),
            ExprNode::Solve { .. } => format!("#{} Solve", id.index()),
        };

        if let Some(metadata) = self.metadata(id) {
            if let Some(name) = metadata.name() {
                label.push_str(&format!(" name=\"{name}\""));
            }
            if !metadata.tags().is_empty() {
                label.push_str(" tags=[");
                for (index, tag) in metadata.tags().iter().enumerate() {
                    if index != 0 {
                        label.push_str(", ");
                    }
                    label.push_str(tag);
                }
                label.push(']');
            }
        }

        label
    }
}

impl fmt::Display for ExprGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.format_expression(self.root()))
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum ExprPrecedence {
    Lowest,
    Add,
    Mul,
    Unary,
    Pow,
    Postfix,
    Atom,
}

/// Configurable compact-equation display for an [`ExprGraph`].
pub struct ExprGraphEquationDisplay<'a> {
    graph: &'a ExprGraph,
    options: DisplayOptions,
}

impl<'a> ExprGraphEquationDisplay<'a> {
    pub(crate) fn new(graph: &'a ExprGraph) -> Self {
        Self {
            graph,
            options: DisplayOptions::default(),
        }
    }

    /// Adds the style rules from a built-in color palette.
    pub fn with_preset(mut self, preset: ColorPreset) -> Self {
        self.options.with_preset(preset);
        self
    }

    /// Appends a node style rule.
    ///
    /// Later matching rules override fields set by earlier rules.
    pub fn with_style_rule(mut self, rule: NodeStyleRule) -> Self {
        self.options.rules.push(rule);
        self
    }
}

impl fmt::Display for ExprGraphEquationDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(
            &self
                .graph
                .format_expression_with(self.graph.root(), &|id, node, text| {
                    self.options.resolve(self.graph, id, node).ansi(text)
                }),
        )
    }
}

/// Configurable LaTeX-equation display for an [`ExprGraph`].
///
/// The output is a math-mode fragment. It does not include `$` delimiters or
/// a document preamble. Vector and matrix nodes use `bmatrix` from `amsmath`.
/// Color rules emit `\\color[RGB]` declarations and require `xcolor`.
pub struct ExprGraphLatexDisplay<'a> {
    graph: &'a ExprGraph,
    options: DisplayOptions,
}

impl<'a> ExprGraphLatexDisplay<'a> {
    pub(crate) fn new(graph: &'a ExprGraph) -> Self {
        Self {
            graph,
            options: DisplayOptions::default(),
        }
    }

    /// Adds the style rules from a built-in color palette.
    pub fn with_preset(mut self, preset: ColorPreset) -> Self {
        self.options.with_preset(preset);
        self
    }

    /// Appends a node style rule.
    ///
    /// Later matching rules override fields set by earlier rules. Only the
    /// foreground color is meaningful for LaTeX output.
    pub fn with_style_rule(mut self, rule: NodeStyleRule) -> Self {
        self.options.rules.push(rule);
        self
    }
}

impl fmt::Display for ExprGraphLatexDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let decorate = |id: ExprId, node: &ExprNode, text: String| {
            self.options.resolve(self.graph, id, node).latex(text)
        };
        f.write_str(&LatexFormatter::new(self.graph, &decorate).format())
    }
}

struct LatexFormatter<'a> {
    graph: &'a ExprGraph,
    decorate: &'a dyn Fn(ExprId, &ExprNode, String) -> String,
}

impl<'a> LatexFormatter<'a> {
    fn new(
        graph: &'a ExprGraph,
        decorate: &'a dyn Fn(ExprId, &ExprNode, String) -> String,
    ) -> Self {
        Self { graph, decorate }
    }

    fn format(&self) -> String {
        self.format_expression(self.graph.root())
    }

    fn format_expression(&self, id: ExprId) -> String {
        self.format_child(id, ExprPrecedence::Lowest, false)
    }

    fn format_child(
        &self,
        id: ExprId,
        parent_precedence: ExprPrecedence,
        parenthesize_equal: bool,
    ) -> String {
        let (text, precedence) = self.format_node(id);
        if precedence < parent_precedence || (parenthesize_equal && precedence == parent_precedence)
        {
            format!("\\left({text}\\right)")
        } else {
            text
        }
    }

    fn format_node(&self, id: ExprId) -> (String, ExprPrecedence) {
        let Some(node) = self.graph.node(id) else {
            return (
                format!("\\text{{missing node \\#{}}}", id.index()),
                ExprPrecedence::Atom,
            );
        };

        let (text, precedence) = match node {
            ExprNode::RealConst(value) => (format_latex_number(*value), ExprPrecedence::Atom),
            ExprNode::ComplexConst(value) => format_latex_complex(*value),
            ExprNode::ScalarParam(parameter) => (
                parameter
                    .latex_label()
                    .map(str::to_owned)
                    .unwrap_or_else(|| escape_latex(parameter.name())),
                ExprPrecedence::Atom,
            ),
            ExprNode::EventScalar(name) => (escape_latex(name), ExprPrecedence::Atom),
            ExprNode::EventP4Component { name, component } => (
                format!(
                    "{}_{{\\mathrm{{{}}}}}",
                    escape_latex(name),
                    escape_latex(component.label())
                ),
                ExprPrecedence::Atom,
            ),
            ExprNode::Unary { op, input } => self.format_unary(*op, *input),
            ExprNode::Binary { op, lhs, rhs } => self.format_binary(*op, *lhs, *rhs),
            ExprNode::NaryAdd { terms } => self.format_sum(terms),
            ExprNode::NaryMul { factors } => self.format_product(factors),
            ExprNode::Complex { re, im } => (
                self.format_operator("complex", &[*re, *im]),
                ExprPrecedence::Atom,
            ),
            ExprNode::Vector { elements } => (
                format!(
                    "\\begin{{bmatrix}}{}\\end{{bmatrix}}",
                    elements
                        .iter()
                        .map(|id| self.format_expression(*id))
                        .collect::<Vec<_>>()
                        .join(" \\\\ ")
                ),
                ExprPrecedence::Atom,
            ),
            ExprNode::Matrix {
                rows,
                cols,
                elements,
            } => {
                let rows = (0..*rows)
                    .map(|row| {
                        let start = row * *cols;
                        let end = start + *cols;
                        elements[start..end]
                            .iter()
                            .map(|id| self.format_expression(*id))
                            .collect::<Vec<_>>()
                            .join(" & ")
                    })
                    .collect::<Vec<_>>()
                    .join(" \\\\ ");
                (
                    format!("\\begin{{bmatrix}}{rows}\\end{{bmatrix}}"),
                    ExprPrecedence::Atom,
                )
            }
            ExprNode::Component { input, index } => (
                format!(
                    "{}_{{{index}}}",
                    self.format_child(*input, ExprPrecedence::Postfix, false)
                ),
                ExprPrecedence::Postfix,
            ),
            ExprNode::MatrixElement { input, row, col } => (
                format!(
                    "{}_{{{row},{col}}}",
                    self.format_child(*input, ExprPrecedence::Postfix, false)
                ),
                ExprPrecedence::Postfix,
            ),
            ExprNode::MatMul { lhs, rhs } => (
                self.format_operator("matmul", &[*lhs, *rhs]),
                ExprPrecedence::Atom,
            ),
            ExprNode::MatVec { matrix, vector } => (
                self.format_operator("matvec", &[*matrix, *vector]),
                ExprPrecedence::Atom,
            ),
            ExprNode::Dot { lhs, rhs } => (
                self.format_operator("dot", &[*lhs, *rhs]),
                ExprPrecedence::Atom,
            ),
            ExprNode::Solve { matrix, rhs } => (
                self.format_operator("solve", &[*matrix, *rhs]),
                ExprPrecedence::Atom,
            ),
        };
        ((self.decorate)(id, node, text), precedence)
    }

    fn format_unary(&self, op: UnaryOp, input: ExprId) -> (String, ExprPrecedence) {
        match op {
            UnaryOp::Neg => (
                format!("-{}", self.format_child(input, ExprPrecedence::Unary, true)),
                ExprPrecedence::Unary,
            ),
            UnaryOp::Real => (self.format_operator("Re", &[input]), ExprPrecedence::Atom),
            UnaryOp::Imag => (self.format_operator("Im", &[input]), ExprPrecedence::Atom),
            UnaryOp::Conj => (
                format!("\\overline{{{}}}", self.format_expression(input)),
                ExprPrecedence::Atom,
            ),
            UnaryOp::NormSqr => (
                format!("\\left|{}\\right|^{{2}}", self.format_expression(input)),
                ExprPrecedence::Pow,
            ),
            UnaryOp::Sqrt => (
                format!("\\sqrt{{{}}}", self.format_expression(input)),
                ExprPrecedence::Atom,
            ),
            UnaryOp::Exp => (self.format_function("exp", input), ExprPrecedence::Atom),
            UnaryOp::Sin => (self.format_function("sin", input), ExprPrecedence::Atom),
            UnaryOp::Cos => (self.format_function("cos", input), ExprPrecedence::Atom),
            UnaryOp::Log => (self.format_function("log", input), ExprPrecedence::Atom),
            UnaryOp::PowI(power) => (
                format!(
                    "{}^{{{power}}}",
                    self.format_child(input, ExprPrecedence::Pow, true)
                ),
                ExprPrecedence::Pow,
            ),
        }
    }

    fn format_function(&self, name: &str, input: ExprId) -> String {
        format!("\\{name}\\left({}\\right)", self.format_expression(input))
    }

    fn format_operator(&self, name: &str, inputs: &[ExprId]) -> String {
        format!(
            "\\operatorname{{{name}}}\\left({}\\right)",
            inputs
                .iter()
                .map(|id| self.format_expression(*id))
                .collect::<Vec<_>>()
                .join(", ")
        )
    }

    fn format_binary(&self, op: BinaryOp, lhs: ExprId, rhs: ExprId) -> (String, ExprPrecedence) {
        match op {
            BinaryOp::Add => self.format_sum(&[lhs, rhs]),
            BinaryOp::Sub => {
                let lhs = self.format_child(lhs, ExprPrecedence::Add, false);
                let rhs = self.format_child(rhs, ExprPrecedence::Add, true);
                (format!("{lhs} - {rhs}"), ExprPrecedence::Add)
            }
            BinaryOp::Mul => self.format_product(&[lhs, rhs]),
            BinaryOp::Div => (
                format!(
                    "\\frac{{{}}}{{{}}}",
                    self.format_expression(lhs),
                    self.format_expression(rhs)
                ),
                ExprPrecedence::Atom,
            ),
            BinaryOp::Atan2 => (
                self.format_operator("atan2", &[lhs, rhs]),
                ExprPrecedence::Atom,
            ),
        }
    }

    fn format_sum(&self, terms: &[ExprId]) -> (String, ExprPrecedence) {
        let mut formatted = String::new();
        for term in terms {
            let (negative, term) = self.format_signed_term(*term);
            if formatted.is_empty() {
                if negative {
                    formatted.push('-');
                }
                formatted.push_str(&term);
            } else if negative {
                formatted.push_str(" - ");
                formatted.push_str(&term);
            } else {
                formatted.push_str(" + ");
                formatted.push_str(&term);
            }
        }
        if formatted.is_empty() {
            formatted.push('0');
        }
        (formatted, ExprPrecedence::Add)
    }

    fn format_signed_term(&self, id: ExprId) -> (bool, String) {
        match self.graph.node(id) {
            Some(node @ ExprNode::RealConst(value)) if *value < 0.0 => (
                true,
                (self.decorate)(id, node, format_latex_number(-*value)),
            ),
            Some(ExprNode::Unary {
                op: UnaryOp::Neg,
                input,
            }) => (true, self.format_child(*input, ExprPrecedence::Add, false)),
            Some(ExprNode::NaryMul { factors }) => self.format_product_parts(factors),
            _ => (false, self.format_child(id, ExprPrecedence::Add, false)),
        }
    }

    fn format_product(&self, factors: &[ExprId]) -> (String, ExprPrecedence) {
        let (negative, product) = self.format_product_parts(factors);
        if negative {
            (format!("-{product}"), ExprPrecedence::Unary)
        } else {
            (product, ExprPrecedence::Mul)
        }
    }

    fn format_product_parts(&self, factors: &[ExprId]) -> (bool, String) {
        let mut negative = false;
        let mut pieces = Vec::new();
        for factor in factors {
            match self.graph.node(*factor) {
                Some(node @ ExprNode::RealConst(value)) if *value < 0.0 => {
                    negative = !negative;
                    if *value != -1.0 || factors.len() == 1 {
                        pieces.push((self.decorate)(*factor, node, format_latex_number(-*value)));
                    }
                }
                Some(ExprNode::Unary {
                    op: UnaryOp::Neg,
                    input,
                }) => {
                    negative = !negative;
                    pieces.push(self.format_child(*input, ExprPrecedence::Mul, false));
                }
                _ => pieces.push(self.format_child(*factor, ExprPrecedence::Mul, false)),
            }
        }
        if pieces.is_empty() {
            pieces.push("1".to_owned());
        }
        (negative, pieces.join(" \\cdot "))
    }
}

fn format_latex_complex(value: Complex64) -> (String, ExprPrecedence) {
    match (value.re, value.im) {
        (re, 0.0) => (format_latex_number(re), ExprPrecedence::Atom),
        (0.0, im) => (format_latex_imaginary(im), ExprPrecedence::Atom),
        (re, im) if im < 0.0 => (
            format!(
                "{} - {}",
                format_latex_number(re),
                format_latex_imaginary(-im)
            ),
            ExprPrecedence::Add,
        ),
        (re, im) => (
            format!(
                "{} + {}",
                format_latex_number(re),
                format_latex_imaginary(im)
            ),
            ExprPrecedence::Add,
        ),
    }
}

fn format_latex_number(value: f64) -> String {
    if value == f64::INFINITY {
        "\\infty".to_owned()
    } else if value == f64::NEG_INFINITY {
        "-\\infty".to_owned()
    } else if value.is_nan() {
        "\\mathrm{NaN}".to_owned()
    } else {
        ExprGraph::format_real_number(value)
    }
}

fn format_latex_imaginary(value: f64) -> String {
    match value {
        1.0 => "\\mathrm{i}".to_owned(),
        -1.0 => "-\\mathrm{i}".to_owned(),
        value => format!("{}\\mathrm{{i}}", format_latex_number(value)),
    }
}

fn escape_latex(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '\\' => escaped.push_str("\\backslash "),
            '{' => escaped.push_str("\\{"),
            '}' => escaped.push_str("\\}"),
            '_' => escaped.push_str("\\_"),
            '^' => escaped.push_str("\\^{}"),
            '#' => escaped.push_str("\\#"),
            '$' => escaped.push_str("\\$"),
            '%' => escaped.push_str("\\%"),
            '&' => escaped.push_str("\\&"),
            '~' => escaped.push_str("\\~{}"),
            _ => escaped.push(character),
        }
    }
    escaped
}

/// Configurable indented-tree display for an [`ExprGraph`].
pub struct ExprGraphTreeDisplay<'a> {
    graph: &'a ExprGraph,
    options: DisplayOptions,
}

impl<'a> ExprGraphTreeDisplay<'a> {
    pub(crate) fn new(graph: &'a ExprGraph) -> Self {
        Self {
            graph,
            options: DisplayOptions::default(),
        }
    }

    /// Sets whether nodes reached through multiple paths are fully expanded.
    pub fn expand_repeated(mut self, expand: bool) -> Self {
        self.options.expand_repeated = expand;
        self
    }

    /// Adds the style rules from a built-in color palette.
    pub fn with_preset(mut self, preset: ColorPreset) -> Self {
        self.options.with_preset(preset);
        self
    }

    /// Appends a node style rule.
    ///
    /// Later matching rules override fields set by earlier rules.
    pub fn with_style_rule(mut self, rule: NodeStyleRule) -> Self {
        self.options.rules.push(rule);
        self
    }

    fn fmt_node(
        &self,
        f: &mut fmt::Formatter<'_>,
        id: ExprId,
        prefix: &str,
        edge: Option<(&str, bool)>,
        visited: &mut HashSet<ExprId>,
    ) -> fmt::Result {
        let Some(node) = self.graph.node(id) else {
            return write_tree_line(f, prefix, edge, &format!("#{} <missing node>", id.index()));
        };
        let repeated = !visited.insert(id);
        let mut line = if repeated && !self.options.expand_repeated {
            format!("#{0} <reference to #{0}>", id.index())
        } else {
            self.graph.node_label(id, node)
        };
        line = self.options.resolve(self.graph, id, node).ansi(line);
        write_tree_line(f, prefix, edge, &line)?;
        if repeated && !self.options.expand_repeated {
            return Ok(());
        }

        let children = node_children(node);
        let child_prefix = match edge {
            Some((_, true)) => format!("{prefix}   "),
            Some((_, false)) => format!("{prefix}┃  "),
            None => prefix.to_owned(),
        };
        for (index, (label, child)) in children.iter().enumerate() {
            self.fmt_node(
                f,
                *child,
                &child_prefix,
                Some((label, index + 1 == children.len())),
                visited,
            )?;
        }
        Ok(())
    }
}

impl fmt::Display for ExprGraphTreeDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "ExprGraph(root=#{})", self.graph.root().index())?;
        self.fmt_node(f, self.graph.root(), "", None, &mut HashSet::new())
    }
}

/// Configurable Graphviz DOT display for an [`ExprGraph`].
pub struct ExprGraphDotDisplay<'a> {
    graph: &'a ExprGraph,
    options: DisplayOptions,
}

impl<'a> ExprGraphDotDisplay<'a> {
    pub(crate) fn new(graph: &'a ExprGraph) -> Self {
        Self {
            graph,
            options: DisplayOptions::default(),
        }
    }

    /// Sets whether nodes reached through multiple paths are fully expanded.
    pub fn expand_repeated(mut self, expand: bool) -> Self {
        self.options.expand_repeated = expand;
        self
    }

    /// Adds the style rules from a built-in color palette.
    pub fn with_preset(mut self, preset: ColorPreset) -> Self {
        self.options.with_preset(preset);
        self
    }

    /// Appends a node style rule.
    ///
    /// Later matching rules override fields set by earlier rules.
    pub fn with_style_rule(mut self, rule: NodeStyleRule) -> Self {
        self.options.rules.push(rule);
        self
    }

    #[cfg(feature = "svg")]
    /// Renders the generated Graphviz graph as an SVG document.
    ///
    /// # Errors
    ///
    /// Returns [`GraphRenderError::Dot`] when the generated Graphviz DOT
    /// source cannot be parsed.
    pub fn render_svg(&self) -> Result<String, GraphRenderError> {
        use layout::{backends::svg::SVGWriter, gv};

        let dot = self.to_string();
        let mut parser = gv::DotParser::new(&dot);
        let graph = parser.process().map_err(GraphRenderError::Dot)?;
        let mut builder = gv::GraphBuilder::new();
        builder.visit_graph(&graph);
        let mut graph = builder.get();
        let mut svg = SVGWriter::new();
        graph.do_it(false, false, false, &mut svg);
        Ok(svg.finalize())
    }

    fn node_attributes(&self, id: ExprId, node: &ExprNode) -> String {
        let mut attributes = vec![format!(
            "label=\"{}\"",
            escape_dot(&self.graph.node_label(id, node))
        )];
        let style = self.options.resolve(self.graph, id, node);
        if let Some(color) = style.foreground {
            attributes.push(format!("fontcolor=\"{}\"", color.dot()));
        }
        if let Some(color) = style.border {
            attributes.push(format!("color=\"{}\"", color.dot()));
        }
        if let Some(color) = style.fill {
            attributes.push(format!("fillcolor=\"{}\"", color.dot()));
            attributes.push("style=filled".to_owned());
        }
        attributes.join(", ")
    }

    fn write_expanded(
        &self,
        f: &mut fmt::Formatter<'_>,
        id: ExprId,
        occurrence: &mut usize,
    ) -> fmt::Result {
        let current = *occurrence;
        *occurrence += 1;
        let Some(node) = self.graph.node(id) else {
            return Ok(());
        };
        writeln!(f, "  n{current} [{}];", self.node_attributes(id, node))?;
        for (label, child) in node_children(node) {
            let child_occurrence = *occurrence;
            self.write_expanded(f, child, occurrence)?;
            writeln!(
                f,
                "  n{current} -> n{child_occurrence} [label=\"{}\"];",
                escape_dot(&label)
            )?;
        }
        Ok(())
    }

    fn write_shared(
        &self,
        f: &mut fmt::Formatter<'_>,
        id: ExprId,
        visited: &mut HashSet<ExprId>,
    ) -> fmt::Result {
        if !visited.insert(id) {
            return Ok(());
        }
        let Some(node) = self.graph.node(id) else {
            return Ok(());
        };
        writeln!(f, "  n{} [{}];", id.index(), self.node_attributes(id, node))?;
        for (label, child) in node_children(node) {
            self.write_shared(f, child, visited)?;
            writeln!(
                f,
                "  n{} -> n{} [label=\"{}\"];",
                id.index(),
                child.index(),
                escape_dot(&label)
            )?;
        }
        Ok(())
    }
}

#[cfg(feature = "svg")]
/// Errors produced while rendering an expression graph.
#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum GraphRenderError {
    /// The generated Graphviz DOT source could not be parsed.
    #[error("failed to parse generated DOT: {0}")]
    Dot(String),
}

impl fmt::Display for ExprGraphDotDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "digraph ExprGraph {{")?;
        if self.options.expand_repeated {
            self.write_expanded(f, self.graph.root(), &mut 0)?;
        } else {
            self.write_shared(f, self.graph.root(), &mut HashSet::new())?;
        }
        writeln!(f, "}}")
    }
}

fn write_tree_line(
    f: &mut fmt::Formatter<'_>,
    prefix: &str,
    edge: Option<(&str, bool)>,
    text: &str,
) -> fmt::Result {
    if let Some((label, is_last)) = edge {
        let connector = if is_last { "┗" } else { "┣" };
        writeln!(f, "{prefix}{connector} {label}: {text}")
    } else {
        writeln!(f, "{text}")
    }
}

fn escape_dot(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event_scalar;

    fn shared_graph() -> ExprGraph {
        let shared = event_scalar("x").named("shared").tagged("data");
        ((shared.clone() + 1.0) * (shared + 2.0)).to_graph()
    }

    #[test]
    fn equation_latex_tree_and_dot_have_no_color_by_default() {
        let graph = shared_graph();
        let shared_id = graph
            .nodes()
            .iter()
            .position(|node| matches!(node, ExprNode::EventScalar(name) if name.as_ref() == "x"))
            .unwrap();
        let needle = format!("#{shared_id} EventScalar(x)");
        let tree = graph.display_tree().to_string();
        let dot = graph.display_dot().to_string();
        let equation = graph.display_equation().to_string();
        let latex = graph.display_latex().to_string();

        assert_eq!(tree.matches(&needle).count(), 2);
        assert_eq!(dot.matches(&needle).count(), 2);
        assert!(!tree.contains("\x1b["));
        assert!(!dot.contains("fontcolor="));
        assert!(!dot.contains("fillcolor="));
        assert!(!equation.contains("\x1b["));
        assert_eq!(equation, graph.to_string());
        assert!(!latex.contains("\\color"));
        assert!(latex.contains("\\cdot"));
    }

    #[test]
    fn reference_mode_suppresses_repeated_tree_expansion_and_emits_a_shared_dag() {
        let graph = shared_graph();
        let tree = graph.display_tree().expand_repeated(false).to_string();
        let dot = graph.display_dot().expand_repeated(false).to_string();

        assert_eq!(tree.matches("EventScalar(x)").count(), 1);
        assert_eq!(tree.matches("<reference to #").count(), 1);
        assert_eq!(dot.matches("EventScalar(x)").count(), 1);
        assert_eq!(dot.matches(" -> ").count(), 6);
    }

    #[test]
    fn later_style_rules_override_matching_preset_fields() {
        let graph = shared_graph();
        let override_color = DisplayColor::rgb(1, 2, 3);
        let rule = NodeStyleRule::new(
            NodeSelector::Tag("data".to_owned()),
            NodeStyle::new().with_foreground(override_color),
        );
        let tree = graph
            .display_tree()
            .with_preset(ColorPreset::Light)
            .with_style_rule(rule.clone())
            .to_string();
        let dot = graph
            .display_dot()
            .with_preset(ColorPreset::Light)
            .with_style_rule(rule)
            .to_string();
        let equation = graph
            .display_equation()
            .with_preset(ColorPreset::Light)
            .with_style_rule(NodeStyleRule::new(
                NodeSelector::Tag("data".to_owned()),
                NodeStyle::new().with_foreground(override_color),
            ))
            .to_string();
        let latex = graph
            .display_latex()
            .with_preset(ColorPreset::Light)
            .with_style_rule(NodeStyleRule::new(
                NodeSelector::Tag("data".to_owned()),
                NodeStyle::new().with_foreground(override_color),
            ))
            .to_string();

        assert!(tree.contains("\x1b[38;2;1;2;3m"));
        assert!(dot.contains("fontcolor=\"#010203\""));
        assert!(equation.contains("\x1b[38;2;1;2;3m"));
        assert!(latex.contains("\\color[RGB]{1,2,3}"));
    }

    #[test]
    fn latex_uses_math_constructs_and_escapes_names() {
        let numerator = crate::event_scalar("x_value");
        let denominator = crate::event_scalar("y").sqrt();
        let latex = (numerator / denominator)
            .powi(-2)
            .to_graph()
            .display_latex()
            .to_string();

        assert!(latex.contains("\\frac{"));
        assert!(latex.contains("x\\_value"));
        assert!(latex.contains("\\sqrt{y}"));
        assert!(latex.contains("^{-2}"));
    }

    #[test]
    fn latex_uses_parameter_labels_with_name_fallback() {
        let labeled = crate::parameter!("alpha_internal", latex: r"\alpha");
        let unlabeled = crate::parameter!("beta_internal");
        let latex = (labeled + unlabeled).to_graph().display_latex().to_string();

        assert!(latex.contains(r"\alpha"));
        assert!(!latex.contains("alpha_internal"));
        assert!(latex.contains(r"beta\_internal"));
    }

    #[test]
    fn dot_escapes_metadata_and_event_labels() {
        let graph = event_scalar("x\\\"y").named("quoted\"name").to_graph();
        let dot = graph.display_dot().to_string();

        assert!(dot.contains("x\\\\\\\"y"));
        assert!(dot.contains("quoted\\\"name"));
    }

    #[cfg(feature = "svg")]
    #[test]
    fn dot_display_renders_svg_in_process() {
        let svg = shared_graph()
            .display_dot()
            .with_preset(ColorPreset::Light)
            .render_svg()
            .unwrap();

        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
    }
}
