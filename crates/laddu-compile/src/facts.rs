pub use laddu_expr::NumberClass;
use laddu_expr::{
    ExprGraph, ExprId, ExprNode, ExprNodeSemantics, ValueKind,
    parameters::{ParamState, Parameter},
};

/// Static facts for every node in an expression graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphFacts {
    nodes: Vec<NodeFacts>,
}

impl GraphFacts {
    /// Analyzes all nodes in topological order.
    pub fn analyze(graph: &ExprGraph) -> Self {
        let mut nodes = Vec::with_capacity(graph.nodes().len());
        let mut semantics = Vec::with_capacity(graph.nodes().len());
        for node in graph.nodes() {
            let node_semantics = node.semantics(&semantics);
            nodes.push(NodeFacts::for_node(node, &nodes, node_semantics));
            semantics.push(node_semantics);
        }
        Self { nodes }
    }

    /// Returns facts for `id`, if it exists.
    pub fn get(&self, id: ExprId) -> Option<&NodeFacts> {
        self.nodes.get(id.index())
    }

    /// Returns facts in graph-node order.
    pub fn nodes(&self) -> &[NodeFacts] {
        &self.nodes
    }
}

/// Inferred value and dependency properties for one graph node.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct NodeFacts {
    /// Runtime value kind.
    pub value_kind: ValueKind,
    /// Known relationship between real and imaginary components.
    pub number_class: NumberClass,
    /// Parameter and event dependencies.
    pub dependency: DependencyFacts,
}

impl NodeFacts {
    pub(crate) fn for_node(
        node: &ExprNode,
        facts: &[NodeFacts],
        semantics: ExprNodeSemantics,
    ) -> Self {
        let dependency = dependency(node, facts);
        Self {
            value_kind: semantics.value_kind,
            number_class: semantics.number_class,
            dependency,
        }
    }

    /// Returns when the node must be reevaluated.
    pub fn evaluation_class(self) -> EvaluationClass {
        self.dependency.evaluation_class()
    }
}

/// Whether a node transitively depends on parameters or event data.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct DependencyFacts {
    /// Depends on at least one free parameter.
    pub depends_on_free_params: bool,
    /// Depends on at least one fixed parameter.
    pub depends_on_fixed_params: bool,
    /// Depends on event data.
    pub depends_on_event: bool,
}

impl DependencyFacts {
    /// Returns dependency facts for compile-time constants.
    pub fn per_compile() -> Self {
        Self::default()
    }

    /// Returns dependency facts for a parameter definition.
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

    /// Returns dependency facts for event data.
    pub fn from_event() -> Self {
        Self {
            depends_on_event: true,
            ..Self::default()
        }
    }

    /// Combines dependencies from two inputs.
    pub fn union(self, other: Self) -> Self {
        Self {
            depends_on_free_params: self.depends_on_free_params || other.depends_on_free_params,
            depends_on_fixed_params: self.depends_on_fixed_params || other.depends_on_fixed_params,
            depends_on_event: self.depends_on_event || other.depends_on_event,
        }
    }

    /// Returns when a value with these dependencies must be evaluated.
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

/// Frequency at which a graph node must be evaluated.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum EvaluationClass {
    /// Once while compiling the model.
    PerCompile,
    /// Once per event when preparing a dataset cache.
    PerEvent,
    /// Once per event for every free-parameter evaluation.
    PerEvaluation,
}

fn dependency(node: &ExprNode, facts: &[NodeFacts]) -> DependencyFacts {
    match node {
        ExprNode::RealConst(_) | ExprNode::ComplexConst(_) => DependencyFacts::per_compile(),
        ExprNode::ScalarParam(parameter) => DependencyFacts::from_parameter(parameter),
        ExprNode::EventScalar(_) | ExprNode::EventP4Component { .. } => {
            DependencyFacts::from_event()
        }
        _ => node
            .children()
            .fold(DependencyFacts::per_compile(), |dependency, child| {
                dependency.union(facts[child.index()].dependency)
            }),
    }
}
