use crate::AutodiffResult;
use laddu_compile::CompiledModel;
use laddu_expr::{ExprId, ExprNode, parameters::FreeParamId};
use serde::{Deserialize, Serialize};

/// Algorithm selected for automatic differentiation.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AutodiffMode {
    /// Select forward or reverse mode from compiler work estimates.
    #[default]
    Auto,
    /// Propagate one tangent per free parameter through the primal graph.
    Forward,
    /// Propagate adjoints backward from the scalar output.
    Reverse,
}

/// Per-parameter graph dependency plan for automatic differentiation.
#[derive(Clone, Debug)]
pub struct AutodiffPlan {
    parameter_count: usize,
    strategy: PlanStrategy,
}

#[derive(Clone, Debug)]
enum PlanStrategy {
    Forward { active_nodes: Vec<Vec<ExprId>> },
    Reverse,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct WorkEstimate {
    forward: usize,
    reverse: usize,
}

#[derive(Clone, Debug, Default)]
struct ParameterDependencies(Vec<FreeParamId>);

impl ParameterDependencies {
    fn one(parameter: Option<FreeParamId>) -> Self {
        Self(parameter.into_iter().collect())
    }

    fn union_assign(&mut self, other: &Self) {
        self.0 = merge_sorted_unique(&self.0, &other.0);
    }
}

impl AutodiffPlan {
    /// Analyzes a compiled model for the selected differentiation mode.
    ///
    /// # Errors
    ///
    /// This currently succeeds for every validated [`CompiledModel`]. The
    /// result type is retained so future dependency analyses can report an
    /// error without changing this API.
    pub fn from_model(model: &CompiledModel, mode: AutodiffMode) -> AutodiffResult<Self> {
        Ok(Self::analyze(model, mode))
    }

    fn analyze(model: &CompiledModel, requested: AutodiffMode) -> Self {
        let parameter_count = model.params().n_free();
        if requested == AutodiffMode::Reverse {
            return Self {
                parameter_count,
                strategy: PlanStrategy::Reverse,
            };
        }

        let node_dependencies = Self::analyze_dependencies(model);
        let estimate = Self::estimate_work(
            &node_dependencies,
            model.graph().nodes().len(),
            parameter_count,
        );
        let strategy = match Self::select_strategy(requested, estimate) {
            AutodiffMode::Forward => PlanStrategy::Forward {
                active_nodes: Self::invert_dependencies(&node_dependencies, parameter_count),
            },
            AutodiffMode::Reverse => PlanStrategy::Reverse,
            AutodiffMode::Auto => unreachable!("strategy selection always resolves auto mode"),
        };
        Self {
            parameter_count,
            strategy,
        }
    }

    fn analyze_dependencies(model: &CompiledModel) -> Vec<ParameterDependencies> {
        let mut node_dependencies = Vec::with_capacity(model.graph().nodes().len());
        for node in model.graph().nodes() {
            let dependencies = match node {
                ExprNode::ScalarParam(parameter) => {
                    ParameterDependencies::one(Self::parameter_dependency(model, parameter.name()))
                }
                _ => {
                    let mut dependencies = ParameterDependencies::default();
                    for child in node.children() {
                        dependencies.union_assign(&node_dependencies[child.index()]);
                    }
                    dependencies
                }
            };
            node_dependencies.push(dependencies);
        }
        node_dependencies
    }

    fn estimate_work(
        dependencies: &[ParameterDependencies],
        node_count: usize,
        parameter_count: usize,
    ) -> WorkEstimate {
        WorkEstimate {
            forward: dependencies
                .iter()
                .map(|dependencies| dependencies.0.len())
                .sum(),
            // Reverse mode is modeled as a primal and reverse graph pass plus
            // one output collection step per free parameter.
            reverse: node_count.saturating_mul(2).saturating_add(parameter_count),
        }
    }

    fn select_strategy(requested: AutodiffMode, estimate: WorkEstimate) -> AutodiffMode {
        match requested {
            AutodiffMode::Auto if estimate.reverse < estimate.forward => AutodiffMode::Reverse,
            AutodiffMode::Auto | AutodiffMode::Forward => AutodiffMode::Forward,
            AutodiffMode::Reverse => AutodiffMode::Reverse,
        }
    }

    fn invert_dependencies(
        node_dependencies: &[ParameterDependencies],
        parameter_count: usize,
    ) -> Vec<Vec<ExprId>> {
        let mut active_nodes = vec![Vec::new(); parameter_count];
        for (index, dependencies) in node_dependencies.iter().enumerate() {
            let id = ExprId::from_index(index);
            for parameter in &dependencies.0 {
                active_nodes[parameter.index()].push(id);
            }
        }
        active_nodes
    }

    /// Returns the selected differentiation mode.
    pub fn mode(&self) -> AutodiffMode {
        match self.strategy {
            PlanStrategy::Forward { .. } => AutodiffMode::Forward,
            PlanStrategy::Reverse => AutodiffMode::Reverse,
        }
    }

    /// Returns the number of free parameters.
    pub fn parameter_count(&self) -> usize {
        self.parameter_count
    }

    /// Returns graph nodes depending on one free parameter.
    pub fn active_nodes(&self, free_parameter: usize) -> Option<&[ExprId]> {
        match &self.strategy {
            PlanStrategy::Forward { active_nodes } => {
                active_nodes.get(free_parameter).map(Vec::as_slice)
            }
            PlanStrategy::Reverse => None,
        }
    }

    fn parameter_dependency(model: &CompiledModel, name: &str) -> Option<FreeParamId> {
        let id = model.params().id(name)?;
        // `id` came from this same validated layout, so lookup failure is an
        // internal invariant violation. A fixed parameter is the only normal
        // reason for there to be no free-parameter dependency.
        match model.params().free_id(id) {
            Ok(free_id) => free_id,
            Err(error) => {
                debug_assert!(false, "parameter registry returned an invalid id: {error}");
                None
            }
        }
    }
}

fn merge_sorted_unique(lhs: &[FreeParamId], rhs: &[FreeParamId]) -> Vec<FreeParamId> {
    let mut merged = Vec::with_capacity(lhs.len() + rhs.len());
    let (mut left, mut right) = (0, 0);
    while left < lhs.len() || right < rhs.len() {
        let next = match (lhs.get(left), rhs.get(right)) {
            (Some(lhs), Some(rhs)) if lhs.index() < rhs.index() => {
                left += 1;
                *lhs
            }
            (Some(lhs), Some(rhs)) if rhs.index() < lhs.index() => {
                right += 1;
                *rhs
            }
            (Some(value), Some(_)) => {
                left += 1;
                right += 1;
                *value
            }
            (Some(value), None) => {
                left += 1;
                *value
            }
            (None, Some(value)) => {
                right += 1;
                *value
            }
            (None, None) => break,
        };
        merged.push(next);
    }
    merged
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

    #[test]
    fn auto_selects_reverse_without_dense_forward_dependencies() {
        let sum = parameter!("a")
            + parameter!("b")
            + parameter!("c")
            + parameter!("d")
            + parameter!("e")
            + parameter!("f");
        let expression = sum.sin().exp().cos().powi(2);
        let model = CompiledModel::from_expr(&expression).unwrap();
        let plan = AutodiffPlan::from_model(&model, AutodiffMode::Auto).unwrap();

        assert_eq!(plan.mode(), AutodiffMode::Reverse);
        assert_eq!(plan.parameter_count(), 6);
        assert!(plan.active_nodes(0).is_none());
    }

    #[test]
    fn auto_selection_uses_forward_at_equality() {
        let estimate = WorkEstimate {
            forward: 10,
            reverse: 10,
        };
        assert_eq!(
            AutodiffPlan::select_strategy(AutodiffMode::Auto, estimate),
            AutodiffMode::Forward
        );
    }

    #[test]
    fn auto_selection_covers_both_sides_of_threshold() {
        assert_eq!(
            AutodiffPlan::select_strategy(
                AutodiffMode::Auto,
                WorkEstimate {
                    forward: 11,
                    reverse: 10,
                },
            ),
            AutodiffMode::Reverse
        );
        assert_eq!(
            AutodiffPlan::select_strategy(
                AutodiffMode::Auto,
                WorkEstimate {
                    forward: 9,
                    reverse: 10,
                },
            ),
            AutodiffMode::Forward
        );
    }
}
