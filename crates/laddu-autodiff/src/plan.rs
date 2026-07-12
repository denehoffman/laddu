use crate::AutodiffResult;
use laddu_compile::CompiledModel;
use laddu_expr::{ExprId, ExprNode};
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AutodiffMode {
    #[default]
    Forward,
    Reverse,
}

#[derive(Clone, Debug)]
pub struct AutodiffPlan {
    mode: AutodiffMode,
    active_nodes: Vec<Vec<ExprId>>,
}

impl AutodiffPlan {
    pub fn from_model(model: &CompiledModel, mode: AutodiffMode) -> AutodiffResult<Self> {
        let parameter_count = model.params().n_free();
        let mut node_dependencies = Vec::<Vec<bool>>::with_capacity(model.graph().nodes().len());

        for node in model.graph().nodes() {
            let mut dependencies = vec![false; parameter_count];
            match node {
                ExprNode::ScalarParam(parameter) => {
                    Self::mark_parameter_dependency(model, parameter.name(), &mut dependencies);
                }
                _ => {
                    for child in Self::children(node) {
                        for (target, source) in dependencies
                            .iter_mut()
                            .zip(&node_dependencies[child.index()])
                        {
                            *target |= *source;
                        }
                    }
                }
            }
            node_dependencies.push(dependencies);
        }

        let mut active_nodes = vec![Vec::new(); parameter_count];
        for (index, dependencies) in node_dependencies.iter().enumerate() {
            let id = ExprId::from_index(index);
            for (parameter, active) in dependencies.iter().copied().enumerate() {
                if active {
                    active_nodes[parameter].push(id);
                }
            }
        }

        Ok(Self { mode, active_nodes })
    }

    pub fn mode(&self) -> AutodiffMode {
        self.mode
    }

    pub fn parameter_count(&self) -> usize {
        self.active_nodes.len()
    }

    pub fn active_nodes(&self, free_parameter: usize) -> Option<&[ExprId]> {
        self.active_nodes.get(free_parameter).map(Vec::as_slice)
    }

    fn mark_parameter_dependency(model: &CompiledModel, name: &str, dependencies: &mut [bool]) {
        let Some(id) = model.params().id(name) else {
            return;
        };
        let Ok(Some(free_id)) = model.params().free_id(id) else {
            return;
        };
        dependencies[free_id.index()] = true;
    }

    fn children(node: &ExprNode) -> Vec<ExprId> {
        match node {
            ExprNode::Unary { input, .. }
            | ExprNode::Component { input, .. }
            | ExprNode::MatrixElement { input, .. } => vec![*input],
            ExprNode::Binary { lhs, rhs, .. }
            | ExprNode::MatMul { lhs, rhs }
            | ExprNode::Dot { lhs, rhs } => vec![*lhs, *rhs],
            ExprNode::NaryAdd { terms } => terms.clone(),
            ExprNode::NaryMul { factors } => factors.clone(),
            ExprNode::Complex { re, im } => vec![*re, *im],
            ExprNode::Vector { elements } | ExprNode::Matrix { elements, .. } => elements.clone(),
            ExprNode::MatVec { matrix, vector } => vec![*matrix, *vector],
            ExprNode::Solve { matrix, rhs } => vec![*matrix, *rhs],
            ExprNode::RealConst(_)
            | ExprNode::ComplexConst(_)
            | ExprNode::ScalarParam(_)
            | ExprNode::EventScalar(_)
            | ExprNode::EventP4Component { .. } => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use laddu_compile::CompiledModel;
    use laddu_expr::{complex, event_scalar, parameter};

    use super::*;

    #[test]
    fn tracks_exact_parameter_dependencies() {
        let x = parameter!("x");
        let y = parameter!("y");
        let expression = complex(x.clone(), y.clone()) * x + event_scalar("event");
        let model = CompiledModel::from_expr(&expression).unwrap();
        let plan = AutodiffPlan::from_model(&model, AutodiffMode::Forward).unwrap();

        assert_eq!(plan.parameter_count(), 2);
        assert!(
            plan.active_nodes(0)
                .unwrap()
                .contains(&model.graph().root())
        );
        assert!(
            plan.active_nodes(1)
                .unwrap()
                .contains(&model.graph().root())
        );
    }

    #[test]
    fn accepts_reverse_mode() {
        let expression = parameter!("x").into();
        let model = CompiledModel::from_expr(&expression).unwrap();
        let plan = AutodiffPlan::from_model(&model, AutodiffMode::Reverse).unwrap();

        assert_eq!(plan.mode(), AutodiffMode::Reverse);
        assert_eq!(plan.parameter_count(), 1);
    }
}
