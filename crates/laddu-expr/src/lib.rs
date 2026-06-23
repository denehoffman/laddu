use std::{fmt, sync::Arc};

use num::complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::parameters::Parameter;

pub mod parameters;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ExprId(u32);

impl ExprId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ValueKind {
    Real,
    Complex,
    Vector { len: usize },
    Matrix { rows: usize, cols: usize },
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UnaryOp {
    Neg,
    Real,
    Imag,
    Conj,
    NormSqr,
    Sqrt,
    Exp,
    Sin,
    Cos,
    Log,
    PowI(i32),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ExprNode {
    RealConst(f64),
    ComplexConst(Complex64),
    ScalarParam(Parameter),
    ComplexScalarParam {
        re: Parameter,
        im: Parameter,
    },
    PolarComplexScalarParam {
        mag: Parameter,
        phase: Parameter,
    },
    EventScalar(Arc<str>),
    Unary {
        op: UnaryOp,
        input: ExprId,
    },
    Binary {
        op: BinaryOp,
        lhs: ExprId,
        rhs: ExprId,
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

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExprSourceKind {
    Const,
    Param,
    Event,
    Unary,
    Binary,
    Vector,
    Matrix,
    LinearAlgebra,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExprMetadata {
    source: ExprSourceKind,
    name: Option<Arc<str>>,
    tags: Vec<Arc<str>>,
}

impl ExprMetadata {
    pub fn new(source: ExprSourceKind) -> Self {
        Self {
            source,
            name: None,
            tags: Vec::new(),
        }
    }

    pub fn source(&self) -> ExprSourceKind {
        self.source
    }

    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    pub fn tags(&self) -> &[Arc<str>] {
        &self.tags
    }

    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.iter().any(|candidate| candidate.as_ref() == tag)
    }
}

#[derive(Clone, Debug)]
pub struct Expr {
    node: Arc<DagNode>,
}

#[derive(Clone, Debug)]
struct DagNode {
    kind: DagNodeKind,
    metadata: ExprMetadata,
}

#[derive(Clone, Debug)]
enum DagNodeKind {
    RealConst(f64),
    ComplexConst(Complex64),
    ScalarParam(Parameter),
    ComplexScalarParam {
        re: Parameter,
        im: Parameter,
    },
    PolarComplexScalarParam {
        mag: Parameter,
        phase: Parameter,
    },
    EventScalar(Arc<str>),
    Unary {
        op: UnaryOp,
        input: Expr,
    },
    Binary {
        op: BinaryOp,
        lhs: Expr,
        rhs: Expr,
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
            }),
        }
    }

    pub fn named(self, name: impl Into<Arc<str>>) -> Self {
        self.with_metadata(|metadata| metadata.name = Some(name.into()))
    }

    pub fn tagged(self, tag: impl Into<Arc<str>>) -> Self {
        let tag = tag.into();
        self.with_metadata(|metadata| {
            if !metadata.tags.iter().any(|existing| existing == &tag) {
                metadata.tags.push(tag);
            }
        })
    }

    pub fn real(&self) -> Self {
        unary(UnaryOp::Real, self)
    }

    pub fn imag(&self) -> Self {
        unary(UnaryOp::Imag, self)
    }

    pub fn conj(&self) -> Self {
        unary(UnaryOp::Conj, self)
    }

    pub fn norm_sqr(&self) -> Self {
        unary(UnaryOp::NormSqr, self)
    }

    pub fn sqrt(&self) -> Self {
        unary(UnaryOp::Sqrt, self)
    }

    pub fn exp(&self) -> Self {
        unary(UnaryOp::Exp, self)
    }

    pub fn sin(&self) -> Self {
        unary(UnaryOp::Sin, self)
    }

    pub fn cos(&self) -> Self {
        unary(UnaryOp::Cos, self)
    }

    pub fn log(&self) -> Self {
        unary(UnaryOp::Log, self)
    }

    pub fn powi(&self, power: i32) -> Self {
        unary(UnaryOp::PowI(power), self)
    }

    pub fn component(&self, index: usize) -> Self {
        Expr::new(DagNodeKind::Component {
            input: self.clone(),
            index,
        })
    }

    pub fn matrix_element(&self, row: usize, col: usize) -> Self {
        Expr::new(DagNodeKind::MatrixElement {
            input: self.clone(),
            row,
            col,
        })
    }

    pub fn to_graph(&self) -> ExprGraph {
        GraphBuilder::new().build(self)
    }

    fn with_metadata(self, f: impl FnOnce(&mut ExprMetadata)) -> Self {
        let mut node = (*self.node).clone();
        f(&mut node.metadata);
        Self {
            node: Arc::new(node),
        }
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

pub fn cis(phase: Expr) -> Expr {
    phase.cos() + Complex64::I * phase.sin()
}

pub fn complex(re: impl Into<Parameter>, im: impl Into<Parameter>) -> Expr {
    Expr::new(DagNodeKind::ComplexScalarParam {
        re: re.into(),
        im: im.into(),
    })
}

pub fn polar_complex(mag: impl Into<Parameter>, phase: impl Into<Parameter>) -> Expr {
    Expr::new(DagNodeKind::PolarComplexScalarParam {
        mag: mag.into(),
        phase: phase.into(),
    })
}

pub fn event_scalar(name: impl Into<Arc<str>>) -> Expr {
    Expr::new(DagNodeKind::EventScalar(name.into()))
}

pub fn vector<E>(elements: impl IntoIterator<Item = E>) -> Expr
where
    E: Into<Expr>,
    Expr: From<E>,
{
    Expr::new(DagNodeKind::Vector {
        elements: elements.into_iter().map(Expr::from).collect(),
    })
}

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

pub fn matmul(lhs: impl Into<Expr>, rhs: impl Into<Expr>) -> Expr {
    Expr::new(DagNodeKind::MatMul {
        lhs: lhs.into(),
        rhs: rhs.into(),
    })
}

pub fn matvec(matrix: impl Into<Expr>, vector: impl Into<Expr>) -> Expr {
    Expr::new(DagNodeKind::MatVec {
        matrix: matrix.into(),
        vector: vector.into(),
    })
}

pub fn dot(lhs: impl Into<Expr>, rhs: impl Into<Expr>) -> Expr {
    Expr::new(DagNodeKind::Dot {
        lhs: lhs.into(),
        rhs: rhs.into(),
    })
}

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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExprGraph {
    root: ExprId,
    nodes: Vec<ExprNode>,
    metadata: Vec<ExprMetadata>,
}

impl ExprGraph {
    pub fn root(&self) -> ExprId {
        self.root
    }

    pub fn node(&self, id: ExprId) -> Option<&ExprNode> {
        self.nodes.get(id.index())
    }

    pub fn nodes(&self) -> &[ExprNode] {
        &self.nodes
    }

    pub fn metadata(&self, id: ExprId) -> Option<&ExprMetadata> {
        self.metadata.get(id.index())
    }

    fn fmt_node(
        &self,
        f: &mut fmt::Formatter<'_>,
        id: ExprId,
        prefix: &str,
        edge: Option<(&str, bool)>,
    ) -> fmt::Result {
        let Some(node) = self.node(id) else {
            return write_tree_line(f, prefix, edge, &format!("#{} <missing node>", id.index()));
        };
        let line = self.node_label(id, node);
        write_tree_line(f, prefix, edge, &line)?;

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
                Some((label.as_str(), index + 1 == children.len())),
            )?;
        }

        Ok(())
    }

    fn node_label(&self, id: ExprId, node: &ExprNode) -> String {
        let mut label = match node {
            ExprNode::RealConst(value) => format!("#{} RealConst({value})", id.index()),
            ExprNode::ComplexConst(value) => format!("#{} ComplexConst({value})", id.index()),
            ExprNode::ScalarParam(parameter) => {
                format!("#{} ScalarParam({})", id.index(), parameter.name())
            }
            ExprNode::ComplexScalarParam { re, im } => format!(
                "#{} ComplexScalarParam(re={}, im={})",
                id.index(),
                re.name(),
                im.name()
            ),
            ExprNode::PolarComplexScalarParam { mag, phase } => format!(
                "#{} PolarComplexScalarParam(mag={}, phase={})",
                id.index(),
                mag.name(),
                phase.name()
            ),
            ExprNode::EventScalar(name) => format!("#{} EventScalar({name})", id.index()),
            ExprNode::Unary { op, .. } => format!("#{} Unary({op:?})", id.index()),
            ExprNode::Binary { op, .. } => format!("#{} Binary({op:?})", id.index()),
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
        writeln!(f, "ExprGraph(root=#{})", self.root.index())?;
        self.fmt_node(f, self.root, "", None)
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

fn node_children(node: &ExprNode) -> Vec<(String, ExprId)> {
    match node {
        ExprNode::RealConst(_)
        | ExprNode::ComplexConst(_)
        | ExprNode::ScalarParam(_)
        | ExprNode::ComplexScalarParam { .. }
        | ExprNode::PolarComplexScalarParam { .. }
        | ExprNode::EventScalar(_) => Vec::new(),
        ExprNode::Unary { input, .. } => vec![("input".into(), *input)],
        ExprNode::Binary { lhs, rhs, .. } => vec![("lhs".into(), *lhs), ("rhs".into(), *rhs)],
        ExprNode::Vector { elements } => elements
            .iter()
            .enumerate()
            .map(|(index, id)| (format!("element[{index}]"), *id))
            .collect(),
        ExprNode::Matrix { cols, elements, .. } => elements
            .iter()
            .enumerate()
            .map(|(index, id)| (format!("element[{},{}]", index / cols, index % cols), *id))
            .collect(),
        ExprNode::Component { input, .. } | ExprNode::MatrixElement { input, .. } => {
            vec![("input".into(), *input)]
        }
        ExprNode::MatMul { lhs, rhs } | ExprNode::Dot { lhs, rhs } => {
            vec![("lhs".into(), *lhs), ("rhs".into(), *rhs)]
        }
        ExprNode::MatVec { matrix, vector } => {
            vec![("matrix".into(), *matrix), ("vector".into(), *vector)]
        }
        ExprNode::Solve { matrix, rhs } => vec![("matrix".into(), *matrix), ("rhs".into(), *rhs)],
    }
}

#[derive(Default)]
struct GraphBuilder {
    nodes: Vec<ExprNode>,
    metadata: Vec<ExprMetadata>,
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
        let node = match &expr.node.kind {
            DagNodeKind::RealConst(value) => ExprNode::RealConst(*value),
            DagNodeKind::ComplexConst(value) => ExprNode::ComplexConst(*value),
            DagNodeKind::ScalarParam(parameter) => ExprNode::ScalarParam(parameter.clone()),
            DagNodeKind::ComplexScalarParam { re, im } => ExprNode::ComplexScalarParam {
                re: re.clone(),
                im: im.clone(),
            },
            DagNodeKind::PolarComplexScalarParam { mag, phase } => {
                ExprNode::PolarComplexScalarParam {
                    mag: mag.clone(),
                    phase: phase.clone(),
                }
            }
            DagNodeKind::EventScalar(name) => ExprNode::EventScalar(Arc::clone(name)),
            DagNodeKind::Unary { op, input } => {
                let input = self.visit(input);
                ExprNode::Unary { op: *op, input }
            }
            DagNodeKind::Binary { op, lhs, rhs } => {
                let lhs = self.visit(lhs);
                let rhs = self.visit(rhs);
                ExprNode::Binary { op: *op, lhs, rhs }
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

        let id = ExprId(self.nodes.len() as u32);
        self.nodes.push(node);
        self.metadata.push(expr.node.metadata.clone());
        id
    }
}

fn source_kind(kind: &DagNodeKind) -> ExprSourceKind {
    match kind {
        DagNodeKind::RealConst(_) | DagNodeKind::ComplexConst(_) => ExprSourceKind::Const,
        DagNodeKind::ScalarParam(_)
        | DagNodeKind::ComplexScalarParam { .. }
        | DagNodeKind::PolarComplexScalarParam { .. } => ExprSourceKind::Param,
        DagNodeKind::EventScalar(_) => ExprSourceKind::Event,
        DagNodeKind::Unary { .. } => ExprSourceKind::Unary,
        DagNodeKind::Binary { .. } => ExprSourceKind::Binary,
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
    fn display_formats_graph_as_labeled_tree() {
        let graph = ((parameter!("x") + 1.0).named("offset") * event_scalar("mass").tagged("data"))
            .to_graph();
        let display = graph.to_string();

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
