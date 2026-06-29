use std::{fmt, sync::Arc};

use num::complex::Complex64;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::parameters::Parameter;

pub mod parameters;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ExprId(u32);

impl ExprId {
    pub fn from_index(index: usize) -> Option<Self> {
        u32::try_from(index).ok().map(Self)
    }

    pub fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum ExprGraphError {
    #[error("graph metadata length {metadata_len} does not match node length {node_len}")]
    MetadataLength {
        node_len: usize,
        metadata_len: usize,
    },
    #[error("graph root node #{root} is out of bounds for graph with {node_len} nodes")]
    InvalidRoot { root: usize, node_len: usize },
    #[error("graph node #{node} references missing child #{child}")]
    InvalidChild { node: usize, child: usize },
    #[error(
        "graph node #{node} references child #{child}, but children must appear before parents"
    )]
    InvalidChildOrder { node: usize, child: usize },
    #[error("graph is empty")]
    Empty,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ValueKind {
    Real,
    Complex,
    Vector { len: usize },
    Matrix { rows: usize, cols: usize },
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum P4Component {
    Px,
    Py,
    Pz,
    E,
}

impl P4Component {
    pub fn label(self) -> &'static str {
        match self {
            Self::Px => "px",
            Self::Py => "py",
            Self::Pz => "pz",
            Self::E => "e",
        }
    }

    pub fn index(self) -> usize {
        match self {
            Self::Px => 0,
            Self::Py => 1,
            Self::Pz => 2,
            Self::E => 3,
        }
    }
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

impl UnaryOp {
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

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Atan2,
}

impl BinaryOp {
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
    pub fn const_value(&self) -> Option<Complex64> {
        match self {
            ExprNode::RealConst(value) => Some(Complex64::from(*value)),
            ExprNode::ComplexConst(value) => Some(*value),
            _ => None,
        }
    }

    pub fn is_zero(node: &ExprNode) -> bool {
        node.const_value()
            .is_some_and(|value| value == Complex64::ZERO)
    }

    pub fn is_one(node: &ExprNode) -> bool {
        node.const_value()
            .is_some_and(|value| value == Complex64::ONE)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExprSourceKind {
    Const,
    Param,
    Event,
    Unary,
    Binary,
    Complex,
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

pub fn complex(re: impl Into<Expr>, im: impl Into<Expr>) -> Expr {
    Expr::new(DagNodeKind::Complex {
        re: re.into(),
        im: im.into(),
    })
}

pub fn polar_complex(mag: impl Into<Expr>, phase: impl Into<Expr>) -> Expr {
    mag.into() * (Complex64::I * phase.into()).exp()
}

pub fn event_scalar(name: impl Into<Arc<str>>) -> Expr {
    Expr::new(DagNodeKind::EventScalar(name.into()))
}

pub fn event_p4_component(name: impl Into<Arc<str>>, component: P4Component) -> Expr {
    Expr::new(DagNodeKind::EventP4Component {
        name: name.into(),
        component,
    })
}

pub fn atan2(y: impl Into<Expr>, x: impl Into<Expr>) -> Expr {
    binary(BinaryOp::Atan2, y, x)
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
            for child in node_child_ids(node) {
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

    pub fn display_tree(&self) -> ExprGraphTreeDisplay<'_> {
        ExprGraphTreeDisplay { graph: self }
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
        let (text, precedence) = self.format_node_expression(id);
        if precedence < parent_precedence || (parenthesize_equal && precedence == parent_precedence)
        {
            format!("({text})")
        } else {
            text
        }
    }

    fn format_node_expression(&self, id: ExprId) -> (String, ExprPrecedence) {
        let Some(node) = self.node(id) else {
            return (format!("<missing #{}>", id.index()), ExprPrecedence::Atom);
        };

        match node {
            ExprNode::RealConst(value) => (Self::format_real_number(*value), ExprPrecedence::Atom),
            ExprNode::ComplexConst(value) => self.format_complex_const(*value),
            ExprNode::ScalarParam(parameter) => (parameter.name().to_owned(), ExprPrecedence::Atom),
            ExprNode::ComplexScalarParam { re, im } => (
                format!("complex({}, {})", re.name(), im.name()),
                ExprPrecedence::Atom,
            ),
            ExprNode::PolarComplexScalarParam { mag, phase } => (
                format!("polar({}, {})", mag.name(), phase.name()),
                ExprPrecedence::Atom,
            ),
            ExprNode::EventScalar(name) => (name.to_string(), ExprPrecedence::Atom),
            ExprNode::EventP4Component { name, component } => (
                format!("{name}.{}", component.label()),
                ExprPrecedence::Atom,
            ),
            ExprNode::Unary { op, input } => self.format_unary_expression(*op, *input),
            ExprNode::Binary { op, lhs, rhs } => self.format_binary_expression(*op, *lhs, *rhs),
            ExprNode::NaryAdd { terms } => self.format_sum_expression(terms),
            ExprNode::NaryMul { factors } => self.format_product_expression(factors),
            ExprNode::Complex { re, im } => (
                format!(
                    "complex({}, {})",
                    self.format_expression(*re),
                    self.format_expression(*im)
                ),
                ExprPrecedence::Atom,
            ),
            ExprNode::Vector { elements } => (
                format!(
                    "[{}]",
                    elements
                        .iter()
                        .map(|id| self.format_expression(*id))
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
                                .map(|id| self.format_expression(*id))
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
                    self.format_child(*input, ExprPrecedence::Postfix, false)
                ),
                ExprPrecedence::Postfix,
            ),
            ExprNode::MatrixElement { input, row, col } => (
                format!(
                    "{}[{row}, {col}]",
                    self.format_child(*input, ExprPrecedence::Postfix, false)
                ),
                ExprPrecedence::Postfix,
            ),
            ExprNode::MatMul { lhs, rhs } => (
                format!(
                    "matmul({}, {})",
                    self.format_expression(*lhs),
                    self.format_expression(*rhs)
                ),
                ExprPrecedence::Atom,
            ),
            ExprNode::MatVec { matrix, vector } => (
                format!(
                    "matvec({}, {})",
                    self.format_expression(*matrix),
                    self.format_expression(*vector)
                ),
                ExprPrecedence::Atom,
            ),
            ExprNode::Dot { lhs, rhs } => (
                format!(
                    "dot({}, {})",
                    self.format_expression(*lhs),
                    self.format_expression(*rhs)
                ),
                ExprPrecedence::Atom,
            ),
            ExprNode::Solve { matrix, rhs } => (
                format!(
                    "solve({}, {})",
                    self.format_expression(*matrix),
                    self.format_expression(*rhs)
                ),
                ExprPrecedence::Atom,
            ),
        }
    }

    fn format_unary_expression(&self, op: UnaryOp, input: ExprId) -> (String, ExprPrecedence) {
        match op {
            UnaryOp::Neg => (
                format!("-{}", self.format_child(input, ExprPrecedence::Unary, true)),
                ExprPrecedence::Unary,
            ),
            UnaryOp::Real => (
                self.format_call_expression("real", input),
                ExprPrecedence::Atom,
            ),
            UnaryOp::Imag => (
                self.format_call_expression("imag", input),
                ExprPrecedence::Atom,
            ),
            UnaryOp::Conj => (
                self.format_call_expression("conj", input),
                ExprPrecedence::Atom,
            ),
            UnaryOp::NormSqr => (
                format!("|{}|^2", self.format_expression(input)),
                ExprPrecedence::Pow,
            ),
            UnaryOp::Sqrt => (
                self.format_call_expression("sqrt", input),
                ExprPrecedence::Atom,
            ),
            UnaryOp::Exp => (
                self.format_call_expression("exp", input),
                ExprPrecedence::Atom,
            ),
            UnaryOp::Sin => (
                self.format_call_expression("sin", input),
                ExprPrecedence::Atom,
            ),
            UnaryOp::Cos => (
                self.format_call_expression("cos", input),
                ExprPrecedence::Atom,
            ),
            UnaryOp::Log => (
                self.format_call_expression("log", input),
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
                        self.format_child(input, ExprPrecedence::Pow, true)
                    ),
                    ExprPrecedence::Pow,
                )
            }
        }
    }

    fn format_call_expression(&self, name: &str, input: ExprId) -> String {
        format!("{name}({})", self.format_expression(input))
    }

    fn format_binary_expression(
        &self,
        op: BinaryOp,
        lhs: ExprId,
        rhs: ExprId,
    ) -> (String, ExprPrecedence) {
        match op {
            BinaryOp::Add => self.format_sum_expression(&[lhs, rhs]),
            BinaryOp::Sub => {
                let lhs = self.format_child(lhs, ExprPrecedence::Add, false);
                let rhs = self.format_child(rhs, ExprPrecedence::Add, true);
                (format!("{lhs} - {rhs}"), ExprPrecedence::Add)
            }
            BinaryOp::Mul => self.format_product_expression(&[lhs, rhs]),
            BinaryOp::Div => {
                let lhs = self.format_child(lhs, ExprPrecedence::Mul, false);
                let rhs = self.format_child(rhs, ExprPrecedence::Mul, true);
                (format!("{lhs} / {rhs}"), ExprPrecedence::Mul)
            }
            BinaryOp::Atan2 => (
                format!(
                    "atan2({}, {})",
                    self.format_expression(lhs),
                    self.format_expression(rhs)
                ),
                ExprPrecedence::Atom,
            ),
        }
    }

    fn format_sum_expression(&self, terms: &[ExprId]) -> (String, ExprPrecedence) {
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
        match self.node(id) {
            Some(ExprNode::RealConst(value)) if *value < 0.0 => {
                (true, Self::format_real_number(-*value))
            }
            Some(ExprNode::Unary {
                op: UnaryOp::Neg,
                input,
            }) => (true, self.format_child(*input, ExprPrecedence::Add, false)),
            Some(ExprNode::NaryMul { factors }) => {
                let (negative, product) = self.format_product_parts(factors);
                (negative, product)
            }
            _ => (false, self.format_child(id, ExprPrecedence::Add, false)),
        }
    }

    fn format_product_expression(&self, factors: &[ExprId]) -> (String, ExprPrecedence) {
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
            match self.node(*factor) {
                Some(ExprNode::RealConst(value)) if *value < 0.0 => {
                    negative = !negative;
                    if *value != -1.0 || factors.len() == 1 {
                        pieces.push(Self::format_real_number(-*value));
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
        f.write_str(&self.format_expression(self.root))
    }
}

pub struct ExprGraphTreeDisplay<'a> {
    graph: &'a ExprGraph,
}

impl fmt::Display for ExprGraphTreeDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "ExprGraph(root=#{})", self.graph.root.index())?;
        self.graph.fmt_node(f, self.graph.root, "", None)
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
        | ExprNode::EventScalar(_)
        | ExprNode::EventP4Component { .. } => Vec::new(),
        ExprNode::Unary { input, .. } => vec![("input".into(), *input)],
        ExprNode::Binary { lhs, rhs, .. } => vec![("lhs".into(), *lhs), ("rhs".into(), *rhs)],
        ExprNode::NaryAdd { terms } => terms
            .iter()
            .enumerate()
            .map(|(index, id)| (format!("term[{index}]"), *id))
            .collect(),
        ExprNode::NaryMul { factors } => factors
            .iter()
            .enumerate()
            .map(|(index, id)| (format!("factor[{index}]"), *id))
            .collect(),
        ExprNode::Complex { re, im } => vec![("re".into(), *re), ("im".into(), *im)],
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

fn node_child_ids(node: &ExprNode) -> Vec<ExprId> {
    node_children(node)
        .into_iter()
        .map(|(_, child)| child)
        .collect()
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

        let id = ExprId(self.nodes.len() as u32);
        self.nodes.push(node);
        self.metadata.push(expr.node.metadata.clone());
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
        assert!(
            graph
                .nodes()
                .iter()
                .all(|node| !matches!(node, ExprNode::ComplexScalarParam { .. }))
        );
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
        assert!(
            graph
                .nodes()
                .iter()
                .all(|node| !matches!(node, ExprNode::PolarComplexScalarParam { .. }))
        );
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
            ExprId::from_index(2).unwrap(),
            vec![
                ExprNode::RealConst(2.9999999999999996),
                ExprNode::ComplexConst(Complex64::new(0.30000000000000004, 1.9999999999999998)),
                ExprNode::Binary {
                    op: BinaryOp::Add,
                    lhs: ExprId::from_index(0).unwrap(),
                    rhs: ExprId::from_index(1).unwrap(),
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
            ExprId::from_index(1).unwrap(),
            vec![
                ExprNode::RealConst(1.0),
                ExprNode::Unary {
                    op: UnaryOp::Neg,
                    input: ExprId::from_index(0).unwrap(),
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
            ExprId::from_index(0).unwrap(),
            vec![ExprNode::RealConst(1.0)],
            Vec::new(),
        )
        .unwrap_err();
        assert!(matches!(err, ExprGraphError::MetadataLength { .. }));

        let err = ExprGraph::from_parts(
            ExprId::from_index(0).unwrap(),
            vec![ExprNode::Unary {
                op: UnaryOp::Neg,
                input: ExprId::from_index(0).unwrap(),
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
