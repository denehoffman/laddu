use laddu_compile::CompiledModel;
use laddu_expr::{ExprId, ExprNode};
use thiserror::Error;

pub type AutodiffResult<T> = Result<T, AutodiffError>;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum AutodiffMode {
    Forward,
    Reverse,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum SeedKind {
    Real,
    ComplexReal,
    ComplexImag,
    PolarMagnitude,
    PolarPhase,
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum AutodiffError {
    #[error("autodiff mode {0:?} is not implemented")]
    UnsupportedMode(AutodiffMode),
}

#[derive(Clone, Debug)]
pub struct AutodiffPlan {
    mode: AutodiffMode,
    active_nodes: Vec<Vec<ExprId>>,
    seeds: Vec<Vec<(usize, SeedKind)>>,
}

impl AutodiffPlan {
    pub fn from_model(model: &CompiledModel, mode: AutodiffMode) -> AutodiffResult<Self> {
        if mode != AutodiffMode::Forward {
            return Err(AutodiffError::UnsupportedMode(mode));
        }

        let parameter_count = model.params().n_free();
        let mut node_dependencies = Vec::<Vec<bool>>::with_capacity(model.graph().nodes().len());
        let mut seeds = Vec::with_capacity(model.graph().nodes().len());

        for node in model.graph().nodes() {
            let mut dependencies = vec![false; parameter_count];
            let mut node_seeds = Vec::new();
            match node {
                ExprNode::ScalarParam(parameter) => {
                    Self::add_seed(
                        model,
                        parameter.name(),
                        SeedKind::Real,
                        &mut dependencies,
                        &mut node_seeds,
                    );
                }
                ExprNode::ComplexScalarParam { re, im } => {
                    Self::add_seed(
                        model,
                        re.name(),
                        SeedKind::ComplexReal,
                        &mut dependencies,
                        &mut node_seeds,
                    );
                    Self::add_seed(
                        model,
                        im.name(),
                        SeedKind::ComplexImag,
                        &mut dependencies,
                        &mut node_seeds,
                    );
                }
                ExprNode::PolarComplexScalarParam { mag, phase } => {
                    Self::add_seed(
                        model,
                        mag.name(),
                        SeedKind::PolarMagnitude,
                        &mut dependencies,
                        &mut node_seeds,
                    );
                    Self::add_seed(
                        model,
                        phase.name(),
                        SeedKind::PolarPhase,
                        &mut dependencies,
                        &mut node_seeds,
                    );
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
            seeds.push(node_seeds);
        }

        let mut active_nodes = vec![Vec::new(); parameter_count];
        for (index, dependencies) in node_dependencies.iter().enumerate() {
            let id = ExprId::from_index(index).expect("expression graph exceeds ExprId capacity");
            for (parameter, active) in dependencies.iter().copied().enumerate() {
                if active {
                    active_nodes[parameter].push(id);
                }
            }
        }

        Ok(Self {
            mode,
            active_nodes,
            seeds,
        })
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

    pub fn seed_kind(&self, node: ExprId, free_parameter: usize) -> Option<SeedKind> {
        self.seeds[node.index()]
            .iter()
            .find_map(|(parameter, seed)| (*parameter == free_parameter).then_some(*seed))
    }

    fn add_seed(
        model: &CompiledModel,
        name: &str,
        kind: SeedKind,
        dependencies: &mut [bool],
        seeds: &mut Vec<(usize, SeedKind)>,
    ) {
        let Some(id) = model.params().id(name) else {
            return;
        };
        let Ok(Some(free_id)) = model.params().free_id(id) else {
            return;
        };
        dependencies[free_id.index()] = true;
        seeds.push((free_id.index(), kind));
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
            | ExprNode::ComplexScalarParam { .. }
            | ExprNode::PolarComplexScalarParam { .. }
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
    fn tracks_exact_parameter_dependencies_and_seeds() {
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
    fn rejects_unimplemented_reverse_mode() {
        let expression = parameter!("x").into();
        let model = CompiledModel::from_expr(&expression).unwrap();
        assert_eq!(
            AutodiffPlan::from_model(&model, AutodiffMode::Reverse).unwrap_err(),
            AutodiffError::UnsupportedMode(AutodiffMode::Reverse)
        );
    }
}
