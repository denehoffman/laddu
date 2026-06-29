use laddu_expr::{
    BinaryOp, ExprGraph, ExprId, ExprNode, UnaryOp, ValueKind,
    parameters::{ParamState, Parameter},
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphFacts {
    nodes: Vec<NodeFacts>,
}

impl GraphFacts {
    pub fn analyze(graph: &ExprGraph) -> Self {
        let mut nodes = Vec::with_capacity(graph.nodes().len());
        for node in graph.nodes() {
            nodes.push(NodeFacts::for_node(node, &nodes));
        }
        Self { nodes }
    }

    pub fn get(&self, id: ExprId) -> Option<&NodeFacts> {
        self.nodes.get(id.index())
    }

    pub fn nodes(&self) -> &[NodeFacts] {
        &self.nodes
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct NodeFacts {
    pub value_kind: ValueKind,
    pub number_class: NumberClass,
    pub dependency: DependencyFacts,
}

impl NodeFacts {
    pub(crate) fn for_node(node: &ExprNode, facts: &[NodeFacts]) -> Self {
        let value_kind = value_kind(node, facts);
        let number_class = number_class(node, facts);
        let dependency = dependency(node, facts);
        Self {
            value_kind,
            number_class,
            dependency,
        }
    }

    pub fn evaluation_class(self) -> EvaluationClass {
        self.dependency.evaluation_class()
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum NumberClass {
    Unknown,
    Real,
    Imaginary,
    Complex,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct DependencyFacts {
    pub depends_on_free_params: bool,
    pub depends_on_fixed_params: bool,
    pub depends_on_event: bool,
}

impl DependencyFacts {
    pub fn per_compile() -> Self {
        Self::default()
    }

    pub fn from_parameter(parameter: &Parameter) -> Self {
        match parameter.state() {
            ParamState::Free => Self {
                depends_on_free_params: true,
                ..Self::default()
            },
            ParamState::Fixed(_) => Self {
                depends_on_fixed_params: true,
                ..Self::default()
            },
        }
    }

    pub fn from_event() -> Self {
        Self {
            depends_on_event: true,
            ..Self::default()
        }
    }

    pub fn union(self, other: Self) -> Self {
        Self {
            depends_on_free_params: self.depends_on_free_params || other.depends_on_free_params,
            depends_on_fixed_params: self.depends_on_fixed_params || other.depends_on_fixed_params,
            depends_on_event: self.depends_on_event || other.depends_on_event,
        }
    }

    pub fn evaluation_class(self) -> EvaluationClass {
        if self.depends_on_free_params {
            EvaluationClass::PerEvaluation
        } else if self.depends_on_event {
            EvaluationClass::PerEvent
        } else {
            EvaluationClass::PerCompile
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum EvaluationClass {
    PerCompile,
    PerEvent,
    PerEvaluation,
}

fn value_kind(node: &ExprNode, facts: &[NodeFacts]) -> ValueKind {
    match node {
        ExprNode::RealConst(_) | ExprNode::ScalarParam(_) => ValueKind::Real,
        ExprNode::ComplexConst(value) => {
            if value.im == 0.0 {
                ValueKind::Real
            } else {
                ValueKind::Complex
            }
        }
        ExprNode::ComplexScalarParam { .. }
        | ExprNode::PolarComplexScalarParam { .. }
        | ExprNode::Complex { .. }
        | ExprNode::EventScalar(_) => ValueKind::Complex,
        ExprNode::EventP4Component { .. } => ValueKind::Real,
        ExprNode::Unary { op, input } => match op {
            UnaryOp::Real | UnaryOp::Imag | UnaryOp::NormSqr => ValueKind::Real,
            UnaryOp::Neg
            | UnaryOp::Conj
            | UnaryOp::Sqrt
            | UnaryOp::Exp
            | UnaryOp::Sin
            | UnaryOp::Cos
            | UnaryOp::Log
            | UnaryOp::PowI(_) => facts[input.index()].value_kind,
        },
        ExprNode::Binary { op, lhs, rhs } => {
            if *op == BinaryOp::Atan2 {
                return ValueKind::Real;
            }
            if facts[lhs.index()].value_kind == ValueKind::Real
                && facts[rhs.index()].value_kind == ValueKind::Real
            {
                ValueKind::Real
            } else {
                ValueKind::Complex
            }
        }
        ExprNode::NaryAdd { terms } => {
            if terms
                .iter()
                .all(|id| facts[id.index()].value_kind == ValueKind::Real)
            {
                ValueKind::Real
            } else {
                ValueKind::Complex
            }
        }
        ExprNode::NaryMul { factors } => {
            if factors
                .iter()
                .all(|id| facts[id.index()].value_kind == ValueKind::Real)
            {
                ValueKind::Real
            } else {
                ValueKind::Complex
            }
        }
        ExprNode::Vector { elements } => ValueKind::Vector {
            len: elements.len(),
        },
        ExprNode::Matrix { rows, cols, .. } => ValueKind::Matrix {
            rows: *rows,
            cols: *cols,
        },
        ExprNode::Component { input, .. } => match facts[input.index()].value_kind {
            ValueKind::Vector { .. } => ValueKind::Complex,
            kind => kind,
        },
        ExprNode::MatrixElement { .. } | ExprNode::Dot { .. } => ValueKind::Complex,
        ExprNode::MatMul { lhs, rhs } => {
            let ValueKind::Matrix { rows, .. } = facts[lhs.index()].value_kind else {
                return ValueKind::Complex;
            };
            let ValueKind::Matrix { cols, .. } = facts[rhs.index()].value_kind else {
                return ValueKind::Complex;
            };
            ValueKind::Matrix { rows, cols }
        }
        ExprNode::MatVec { matrix, .. } => {
            let ValueKind::Matrix { rows, .. } = facts[matrix.index()].value_kind else {
                return ValueKind::Complex;
            };
            ValueKind::Vector { len: rows }
        }
        ExprNode::Solve { rhs, .. } => facts[rhs.index()].value_kind,
    }
}

fn number_class(node: &ExprNode, facts: &[NodeFacts]) -> NumberClass {
    match node {
        ExprNode::RealConst(_) | ExprNode::ScalarParam(_) => NumberClass::Real,
        ExprNode::ComplexConst(value) => {
            let value = *value;
            match (value.re == 0.0, value.im == 0.0) {
                (_, true) => NumberClass::Real,
                (true, false) => NumberClass::Imaginary,
                (false, false) => NumberClass::Complex,
            }
        }
        ExprNode::ComplexScalarParam { .. }
        | ExprNode::PolarComplexScalarParam { .. }
        | ExprNode::Complex { .. } => NumberClass::Complex,
        ExprNode::EventScalar(_) => NumberClass::Unknown,
        ExprNode::EventP4Component { .. } => NumberClass::Real,
        ExprNode::Unary { op, input } => {
            let input = facts[input.index()].number_class;
            match op {
                UnaryOp::Neg | UnaryOp::Conj => input,
                UnaryOp::Real | UnaryOp::Imag | UnaryOp::NormSqr => NumberClass::Real,
                UnaryOp::Exp | UnaryOp::Sin | UnaryOp::Cos | UnaryOp::PowI(_) => {
                    if input == NumberClass::Real {
                        NumberClass::Real
                    } else {
                        NumberClass::Unknown
                    }
                }
                UnaryOp::Sqrt | UnaryOp::Log => NumberClass::Unknown,
            }
        }
        ExprNode::Binary { op, lhs, rhs } => {
            let op = *op;
            let lhs = facts[lhs.index()].number_class;
            let rhs = facts[rhs.index()].number_class;
            use NumberClass::{Complex, Imaginary, Real, Unknown};
            match op {
                BinaryOp::Add | BinaryOp::Sub => match (lhs, rhs) {
                    (Real, Real) => Real,
                    (Imaginary, Imaginary) => Imaginary,
                    (Complex, _) | (_, Complex) => Complex,
                    (Unknown, _) | (_, Unknown) => Unknown,
                    _ => Complex,
                },
                BinaryOp::Mul | BinaryOp::Div => match (lhs, rhs) {
                    (Real, Real) | (Imaginary, Imaginary) => Real,
                    (Real, Imaginary) | (Imaginary, Real) => Imaginary,
                    (Complex, _) | (_, Complex) => Complex,
                    (Unknown, _) | (_, Unknown) => Unknown,
                },
                BinaryOp::Atan2 => Real,
            }
        }
        ExprNode::NaryAdd { terms } => {
            let mut classes = terms.iter().map(|id| facts[id.index()].number_class);
            let Some(first) = classes.next() else {
                return NumberClass::Real;
            };
            classes.fold(first, add_number_class)
        }
        ExprNode::NaryMul { factors } => {
            let mut classes = factors.iter().map(|id| facts[id.index()].number_class);
            let Some(first) = classes.next() else {
                return NumberClass::Real;
            };
            classes.fold(first, mul_number_class)
        }
        ExprNode::Vector { .. }
        | ExprNode::Matrix { .. }
        | ExprNode::Component { .. }
        | ExprNode::MatrixElement { .. }
        | ExprNode::MatMul { .. }
        | ExprNode::MatVec { .. }
        | ExprNode::Dot { .. }
        | ExprNode::Solve { .. } => NumberClass::Unknown,
    }
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

fn dependency(node: &ExprNode, facts: &[NodeFacts]) -> DependencyFacts {
    match node {
        ExprNode::RealConst(_) | ExprNode::ComplexConst(_) => DependencyFacts::per_compile(),
        ExprNode::ScalarParam(parameter) => DependencyFacts::from_parameter(parameter),
        ExprNode::ComplexScalarParam { re, im } => {
            DependencyFacts::from_parameter(re).union(DependencyFacts::from_parameter(im))
        }
        ExprNode::PolarComplexScalarParam { mag, phase } => {
            DependencyFacts::from_parameter(mag).union(DependencyFacts::from_parameter(phase))
        }
        ExprNode::EventScalar(_) | ExprNode::EventP4Component { .. } => {
            DependencyFacts::from_event()
        }
        ExprNode::Unary { input, .. }
        | ExprNode::Component { input, .. }
        | ExprNode::MatrixElement { input, .. } => facts[input.index()].dependency,
        ExprNode::Complex { re, im } => facts[re.index()]
            .dependency
            .union(facts[im.index()].dependency),
        ExprNode::NaryAdd { terms } => terms
            .iter()
            .fold(DependencyFacts::per_compile(), |dependency, id| {
                dependency.union(facts[id.index()].dependency)
            }),
        ExprNode::NaryMul { factors } => factors
            .iter()
            .fold(DependencyFacts::per_compile(), |dependency, id| {
                dependency.union(facts[id.index()].dependency)
            }),
        ExprNode::Binary { lhs, rhs, .. }
        | ExprNode::MatMul { lhs, rhs }
        | ExprNode::Dot { lhs, rhs } => facts[lhs.index()]
            .dependency
            .union(facts[rhs.index()].dependency),
        ExprNode::MatVec { matrix, vector }
        | ExprNode::Solve {
            matrix,
            rhs: vector,
        } => facts[matrix.index()]
            .dependency
            .union(facts[vector.index()].dependency),
        ExprNode::Vector { elements } | ExprNode::Matrix { elements, .. } => elements
            .iter()
            .fold(DependencyFacts::per_compile(), |dependency, id| {
                dependency.union(facts[id.index()].dependency)
            }),
    }
}
