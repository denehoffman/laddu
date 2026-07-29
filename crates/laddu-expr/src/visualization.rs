use std::{collections::HashSet, fmt};

use crate::{ExprGraph, ExprId, ExprMetadata, ExprNode, expression::node_children};

/// Controls how graph displays handle nodes reached through multiple paths.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum RepeatedSubtrees {
    /// Render the complete subtree at every occurrence.
    #[default]
    Expand,
    /// Render a later occurrence as a reference to the first.
    Reference,
}

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

#[derive(Clone, Debug, Default)]
struct DisplayOptions {
    repeated_subtrees: RepeatedSubtrees,
    rules: Vec<NodeStyleRule>,
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

macro_rules! display_builder_methods {
    () => {
        /// Sets how nodes reached through multiple paths are rendered.
        pub fn repeated_subtrees(mut self, repeated_subtrees: RepeatedSubtrees) -> Self {
            self.options.repeated_subtrees = repeated_subtrees;
            self
        }

        /// Chooses between fully expanding and referencing repeated subtrees.
        pub fn expand_repeated(self, expand: bool) -> Self {
            self.repeated_subtrees(if expand {
                RepeatedSubtrees::Expand
            } else {
                RepeatedSubtrees::Reference
            })
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
    };
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

    display_builder_methods!();

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
        let mut line = if repeated && self.options.repeated_subtrees == RepeatedSubtrees::Reference
        {
            format!("#{0} <reference to #{0}>", id.index())
        } else {
            self.graph.node_label(id, node)
        };
        if let Some(color) = self.options.resolve(self.graph, id, node).foreground {
            line = format!("{}{line}\x1b[0m", color.ansi_foreground());
        }
        write_tree_line(f, prefix, edge, &line)?;
        if repeated && self.options.repeated_subtrees == RepeatedSubtrees::Reference {
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

    display_builder_methods!();

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
        match self.options.repeated_subtrees {
            RepeatedSubtrees::Expand => self.write_expanded(f, self.graph.root(), &mut 0)?,
            RepeatedSubtrees::Reference => {
                self.write_shared(f, self.graph.root(), &mut HashSet::new())?
            }
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
    fn tree_and_dot_expand_repeated_subtrees_without_color_by_default() {
        let graph = shared_graph();
        let shared_id = graph
            .nodes()
            .iter()
            .position(|node| matches!(node, ExprNode::EventScalar(name) if name.as_ref() == "x"))
            .unwrap();
        let needle = format!("#{shared_id} EventScalar(x)");
        let tree = graph.display_tree().to_string();
        let dot = graph.display_dot().to_string();

        assert_eq!(tree.matches(&needle).count(), 2);
        assert_eq!(dot.matches(&needle).count(), 2);
        assert!(!tree.contains("\x1b["));
        assert!(!dot.contains("fontcolor="));
        assert!(!dot.contains("fillcolor="));
    }

    #[test]
    fn reference_mode_suppresses_repeated_tree_expansion_and_emits_a_shared_dag() {
        let graph = shared_graph();
        let tree = graph
            .display_tree()
            .repeated_subtrees(RepeatedSubtrees::Reference)
            .to_string();
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

        assert!(tree.contains("\x1b[38;2;1;2;3m"));
        assert!(dot.contains("fontcolor=\"#010203\""));
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
