use crate::AutodiffResult;
use laddu_compile::CompiledModel;
use laddu_expr::{ExprId, ExprNode};
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
    mode: AutodiffMode,
    parameter_count: usize,
    active_nodes: Vec<Vec<ExprId>>,
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
        let parameter_count = model.params().n_free();
        if mode == AutodiffMode::Reverse {
            return Ok(Self {
                mode,
                parameter_count,
                active_nodes: Vec::new(),
            });
        }
        let mut node_dependencies = Vec::<Vec<usize>>::with_capacity(model.graph().nodes().len());

        for node in model.graph().nodes() {
            let dependencies = match node {
                ExprNode::ScalarParam(parameter) => {
                    Self::parameter_dependency(model, parameter.name())
                        .into_iter()
                        .collect()
                }
                _ => {
                    let mut dependencies = Vec::new();
                    for child in node.children() {
                        dependencies =
                            merge_sorted_unique(&dependencies, &node_dependencies[child.index()]);
                    }
                    dependencies
                }
            };
            node_dependencies.push(dependencies);
        }

        let forward_work = node_dependencies.iter().map(Vec::len).sum::<usize>();
        let reverse_work = model
            .graph()
            .nodes()
            .len()
            .saturating_mul(2)
            .saturating_add(parameter_count);
        if mode == AutodiffMode::Auto && reverse_work < forward_work {
            return Ok(Self {
                mode: AutodiffMode::Reverse,
                parameter_count,
                active_nodes: Vec::new(),
            });
        }

        let mut active_nodes = vec![Vec::new(); parameter_count];
        for (index, dependencies) in node_dependencies.iter().enumerate() {
            let id = ExprId::from_index(index);
            for parameter in dependencies {
                active_nodes[*parameter].push(id);
            }
        }

        Ok(Self {
            mode: AutodiffMode::Forward,
            parameter_count,
            active_nodes,
        })
    }

    /// Returns the selected differentiation mode.
    pub fn mode(&self) -> AutodiffMode {
        self.mode
    }

    /// Returns the number of free parameters.
    pub fn parameter_count(&self) -> usize {
        self.parameter_count
    }

    /// Returns graph nodes depending on one free parameter.
    pub fn active_nodes(&self, free_parameter: usize) -> Option<&[ExprId]> {
        self.active_nodes.get(free_parameter).map(Vec::as_slice)
    }

    fn parameter_dependency(model: &CompiledModel, name: &str) -> Option<usize> {
        let id = model.params().id(name)?;
        let Ok(Some(free_id)) = model.params().free_id(id) else {
            return None;
        };
        Some(free_id.index())
    }
}

fn merge_sorted_unique(lhs: &[usize], rhs: &[usize]) -> Vec<usize> {
    let mut merged = Vec::with_capacity(lhs.len() + rhs.len());
    let (mut left, mut right) = (0, 0);
    while left < lhs.len() || right < rhs.len() {
        let next = match (lhs.get(left), rhs.get(right)) {
            (Some(lhs), Some(rhs)) if lhs < rhs => {
                left += 1;
                *lhs
            }
            (Some(lhs), Some(rhs)) if rhs < lhs => {
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
}
