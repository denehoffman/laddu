use std::collections::HashMap;

use laddu_expr::{
    ExprGraph, ExprId, ExprNode, ValueKind,
    parameters::{ParamId, ParamLayout},
};
use laddu_kernel::ir::{
    CacheKernelIr, KernelInstruction, KernelValue, KernelValueClass, KernelValueId,
    KernelValueKind, ScalarKernelIr,
};

use crate::{CachePlan, CompileResult, CompiledModel};

#[derive(Copy, Clone, Debug)]
pub struct SolveComponentPlan {
    rhs: ExprId,
    row_slot: usize,
    dimension: usize,
}

impl SolveComponentPlan {
    pub fn rhs(self) -> ExprId {
        self.rhs
    }

    pub fn row_slot(self) -> usize {
        self.row_slot
    }

    pub fn dimension(self) -> usize {
        self.dimension
    }
}

#[derive(Clone, Debug)]
pub struct SolveRowMatrixPlan {
    matrix: ExprId,
    dimension: usize,
    rows: Vec<(usize, usize)>,
}

impl SolveRowMatrixPlan {
    pub fn matrix(&self) -> ExprId {
        self.matrix
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn rows(&self) -> &[(usize, usize)] {
        &self.rows
    }
}

#[derive(Clone, Debug)]
pub struct ExecutablePlan {
    graph: ExprGraph,
    params: ParamLayout,
    parameter_slots: Vec<Option<ParamId>>,
    cache_plan: CachePlan,
    cache_slots: Vec<Option<usize>>,
    evaluation_nodes: Vec<ExprId>,
    value_slots: Vec<Option<usize>>,
    scalar_kernel: Option<ScalarKernelIr>,
    cache_kernel: Option<CacheKernelIr>,
    cache_input_nodes: Vec<ExprId>,
    cache_materialization_nodes: Vec<ExprId>,
    solve_components: Vec<Option<SolveComponentPlan>>,
    solve_rhs_elements: Vec<Option<Vec<ExprId>>>,
    solve_row_matrices: Vec<SolveRowMatrixPlan>,
    solve_row_keys: Vec<(ExprId, usize, usize)>,
    factor_matrix_slots: Vec<Option<usize>>,
    factor_matrices: Vec<(ExprId, usize)>,
    constant_factor_slots: Vec<Option<usize>>,
    constant_factor_matrices: Vec<(ExprId, usize)>,
}

impl ExecutablePlan {
    pub fn from_model(model: &CompiledModel) -> CompileResult<Self> {
        let graph = model.graph().clone();
        let params = model.params().clone();
        let cache_plan = model.cache_plan().clone();
        let parameter_slots = graph
            .nodes()
            .iter()
            .map(|node| match node {
                ExprNode::ScalarParam(parameter) => params.id(parameter.name()),
                _ => None,
            })
            .collect::<Vec<_>>();
        let mut cache_slots = vec![None; graph.nodes().len()];
        for (slot, entry) in cache_plan.entries().iter().enumerate() {
            cache_slots[entry.node().index()] = Some(slot);
        }
        let (solve_components, solve_rhs_elements, solve_row_matrices, solve_row_keys) =
            Self::solve_component_plans(model);
        let (evaluation_nodes, value_slots) = Self::evaluation_schedule(
            &graph,
            &cache_slots,
            &solve_components,
            &solve_rhs_elements,
        )?;
        let scalar_kernel = Self::scalar_kernel_ir(
            model,
            &evaluation_nodes,
            &cache_slots,
            &parameter_slots,
            &solve_components,
            &solve_rhs_elements,
        )?;

        let mut cache_required_nodes = vec![false; graph.nodes().len()];
        for node in cache_plan.materialization_nodes() {
            Self::mark_required(&graph, *node, &mut cache_required_nodes);
        }
        for plan in &solve_row_matrices {
            Self::mark_required(&graph, plan.matrix, &mut cache_required_nodes);
        }
        let cache_materialization_nodes = Self::ids_from_flags(cache_required_nodes)?;
        let (cache_kernel, cache_input_nodes) =
            Self::cache_kernel_ir(model, &cache_materialization_nodes)?;

        let mut factor_matrix_slots = vec![None; graph.nodes().len()];
        let mut factor_matrices = Vec::new();
        let mut constant_factor_slots = vec![None; graph.nodes().len()];
        let mut constant_factor_matrices = Vec::new();
        for (index, node) in graph.nodes().iter().enumerate() {
            let ExprNode::Solve { matrix, .. } = node else {
                continue;
            };
            if value_slots[index].is_none() {
                continue;
            }
            let Some(facts) = model.node_facts(*matrix) else {
                continue;
            };
            if facts.dependency.depends_on_free_params || facts.dependency.depends_on_fixed_params {
                continue;
            }
            let ValueKind::Matrix { rows, cols } = facts.value_kind else {
                continue;
            };
            if rows != cols {
                continue;
            }
            if facts.dependency.depends_on_event {
                if factor_matrix_slots[matrix.index()].is_none() {
                    let slot = factor_matrices.len();
                    factor_matrix_slots[matrix.index()] = Some(slot);
                    factor_matrices.push((*matrix, rows));
                }
            } else if constant_factor_slots[matrix.index()].is_none() {
                let slot = constant_factor_matrices.len();
                constant_factor_slots[matrix.index()] = Some(slot);
                constant_factor_matrices.push((*matrix, rows));
            }
        }

        Ok(Self {
            graph,
            params,
            parameter_slots,
            cache_plan,
            cache_slots,
            evaluation_nodes,
            value_slots,
            scalar_kernel,
            cache_kernel,
            cache_input_nodes,
            cache_materialization_nodes,
            solve_components,
            solve_rhs_elements,
            solve_row_matrices,
            solve_row_keys,
            factor_matrix_slots,
            factor_matrices,
            constant_factor_slots,
            constant_factor_matrices,
        })
    }

    pub fn graph(&self) -> &ExprGraph {
        &self.graph
    }

    pub fn params(&self) -> &ParamLayout {
        &self.params
    }

    pub fn parameter_slots(&self) -> &[Option<ParamId>] {
        &self.parameter_slots
    }

    pub fn cache_plan(&self) -> &CachePlan {
        &self.cache_plan
    }

    pub fn cache_slots(&self) -> &[Option<usize>] {
        &self.cache_slots
    }

    pub fn evaluation_nodes(&self) -> &[ExprId] {
        &self.evaluation_nodes
    }

    pub fn value_slots(&self) -> &[Option<usize>] {
        &self.value_slots
    }

    pub fn scalar_kernel(&self) -> Option<&ScalarKernelIr> {
        self.scalar_kernel.as_ref()
    }

    pub fn cache_kernel(&self) -> Option<&CacheKernelIr> {
        self.cache_kernel.as_ref()
    }

    pub fn cache_input_nodes(&self) -> &[ExprId] {
        &self.cache_input_nodes
    }

    pub fn cache_materialization_nodes(&self) -> &[ExprId] {
        &self.cache_materialization_nodes
    }

    pub fn solve_components(&self) -> &[Option<SolveComponentPlan>] {
        &self.solve_components
    }

    pub fn solve_rhs_elements(&self) -> &[Option<Vec<ExprId>>] {
        &self.solve_rhs_elements
    }

    pub fn solve_row_matrices(&self) -> &[SolveRowMatrixPlan] {
        &self.solve_row_matrices
    }

    pub fn solve_row_keys(&self) -> &[(ExprId, usize, usize)] {
        &self.solve_row_keys
    }

    pub fn factor_matrix_slots(&self) -> &[Option<usize>] {
        &self.factor_matrix_slots
    }

    pub fn factor_matrices(&self) -> &[(ExprId, usize)] {
        &self.factor_matrices
    }

    pub fn constant_factor_slots(&self) -> &[Option<usize>] {
        &self.constant_factor_slots
    }

    pub fn constant_factor_matrices(&self) -> &[(ExprId, usize)] {
        &self.constant_factor_matrices
    }

    fn solve_component_plans(
        model: &CompiledModel,
    ) -> (
        Vec<Option<SolveComponentPlan>>,
        Vec<Option<Vec<ExprId>>>,
        Vec<SolveRowMatrixPlan>,
        Vec<(ExprId, usize, usize)>,
    ) {
        let mut components = vec![None; model.graph().nodes().len()];
        let mut rhs_elements = vec![None; model.graph().nodes().len()];
        let mut row_slots = HashMap::<(ExprId, usize), usize>::new();
        let mut row_keys = Vec::new();
        let mut matrix_slots = HashMap::<ExprId, usize>::new();
        let mut matrices = Vec::<SolveRowMatrixPlan>::new();

        for (node_index, node) in model.graph().nodes().iter().enumerate() {
            let ExprNode::Component { input, index } = node else {
                continue;
            };
            let Some(ExprNode::Solve { matrix, rhs }) = model.graph().node(*input) else {
                continue;
            };
            let Some(matrix_facts) = model.node_facts(*matrix) else {
                continue;
            };
            let matrix_dependency = matrix_facts.dependency;
            if !matrix_dependency.depends_on_event
                || matrix_dependency.depends_on_free_params
                || matrix_dependency.depends_on_fixed_params
            {
                continue;
            }
            let Some(rhs_facts) = model.node_facts(*rhs) else {
                continue;
            };
            let rhs_dependency = rhs_facts.dependency;
            if !rhs_dependency.depends_on_free_params && !rhs_dependency.depends_on_fixed_params {
                continue;
            }
            let ValueKind::Matrix { rows, cols } = matrix_facts.value_kind else {
                continue;
            };
            let ValueKind::Vector { len } = rhs_facts.value_kind else {
                continue;
            };
            if rows == 0 || rows != cols || rows != len || *index >= rows {
                continue;
            }

            let row_slot = if let Some(slot) = row_slots.get(&(*matrix, *index)) {
                *slot
            } else {
                let slot = row_keys.len();
                row_slots.insert((*matrix, *index), slot);
                row_keys.push((*matrix, *index, rows));
                let matrix_slot = if let Some(slot) = matrix_slots.get(matrix) {
                    *slot
                } else {
                    let slot = matrices.len();
                    matrix_slots.insert(*matrix, slot);
                    matrices.push(SolveRowMatrixPlan {
                        matrix: *matrix,
                        dimension: rows,
                        rows: Vec::new(),
                    });
                    slot
                };
                matrices[matrix_slot].rows.push((slot, *index));
                slot
            };
            components[node_index] = Some(SolveComponentPlan {
                rhs: *rhs,
                row_slot,
                dimension: rows,
            });
            if let Some(ExprNode::Vector { elements }) = model.graph().node(*rhs) {
                rhs_elements[rhs.index()] = Some(elements.clone());
            }
        }

        (components, rhs_elements, matrices, row_keys)
    }

    fn scalar_kernel_ir(
        model: &CompiledModel,
        evaluation_nodes: &[ExprId],
        cache_slots: &[Option<usize>],
        parameter_slots: &[Option<ParamId>],
        solve_components: &[Option<SolveComponentPlan>],
        solve_rhs_elements: &[Option<Vec<ExprId>>],
    ) -> CompileResult<Option<ScalarKernelIr>> {
        let mut value_ids = vec![None; model.graph().nodes().len()];
        let mut values = Vec::with_capacity(evaluation_nodes.len());

        for id in evaluation_nodes {
            let index = id.index();
            let value_id = |id: ExprId| {
                value_ids[id.index()].ok_or_else(|| {
                    crate::CompileError::InvalidExecutablePlan(format!(
                        "node {index} depends on unscheduled node {}",
                        id.index()
                    ))
                })
            };
            let instruction = if let Some(cache_slot) = cache_slots[index] {
                KernelInstruction::Cached(cache_slot)
            } else {
                match model.graph().node(*id).ok_or_else(|| {
                    crate::CompileError::InvalidExecutablePlan(format!(
                        "scheduled node {index} is out of bounds"
                    ))
                })? {
                    ExprNode::RealConst(value) => KernelInstruction::RealConstant(*value),
                    ExprNode::ComplexConst(value) => KernelInstruction::ComplexConstant(*value),
                    ExprNode::ScalarParam(_) => {
                        KernelInstruction::Parameter(parameter_slots[index].ok_or_else(|| {
                            crate::CompileError::InvalidExecutablePlan(format!(
                                "parameter node {index} is not bound"
                            ))
                        })?)
                    }
                    ExprNode::Unary { op, input } => KernelInstruction::Unary {
                        op: *op,
                        input: value_id(*input)?,
                    },
                    ExprNode::Binary { op, lhs, rhs } => KernelInstruction::Binary {
                        op: *op,
                        lhs: value_id(*lhs)?,
                        rhs: value_id(*rhs)?,
                    },
                    ExprNode::NaryAdd { terms } => KernelInstruction::Add(
                        terms
                            .iter()
                            .map(|term| value_id(*term))
                            .collect::<CompileResult<_>>()?,
                    ),
                    ExprNode::NaryMul { factors } => KernelInstruction::Mul(
                        factors
                            .iter()
                            .map(|factor| value_id(*factor))
                            .collect::<CompileResult<_>>()?,
                    ),
                    ExprNode::Complex { re, im } => KernelInstruction::Complex {
                        re: value_id(*re)?,
                        im: value_id(*im)?,
                    },
                    ExprNode::Vector { elements } => KernelInstruction::Vector(
                        elements
                            .iter()
                            .map(|element| value_id(*element))
                            .collect::<CompileResult<_>>()?,
                    ),
                    ExprNode::Matrix {
                        rows,
                        cols,
                        elements,
                    } => KernelInstruction::Matrix {
                        rows: *rows,
                        cols: *cols,
                        elements: elements
                            .iter()
                            .map(|element| value_id(*element))
                            .collect::<CompileResult<_>>()?,
                    },
                    ExprNode::Component {
                        input,
                        index: component,
                    } => {
                        if let Some(solve) = solve_components[index]
                            && let Some(elements) = solve_rhs_elements[solve.rhs.index()].as_ref()
                        {
                            KernelInstruction::SolveRow {
                                row_slot: solve.row_slot,
                                rhs: elements
                                    .iter()
                                    .map(|element| value_id(*element))
                                    .collect::<CompileResult<_>>()?,
                            }
                        } else {
                            KernelInstruction::Component {
                                input: value_id(*input)?,
                                index: *component,
                            }
                        }
                    }
                    ExprNode::MatrixElement { input, row, col } => {
                        KernelInstruction::MatrixElement {
                            input: value_id(*input)?,
                            row: *row,
                            col: *col,
                        }
                    }
                    ExprNode::MatMul { lhs, rhs } => KernelInstruction::MatMul {
                        lhs: value_id(*lhs)?,
                        rhs: value_id(*rhs)?,
                    },
                    ExprNode::MatVec { matrix, vector } => KernelInstruction::MatVec {
                        matrix: value_id(*matrix)?,
                        vector: value_id(*vector)?,
                    },
                    ExprNode::Dot { lhs, rhs } => KernelInstruction::Dot {
                        lhs: value_id(*lhs)?,
                        rhs: value_id(*rhs)?,
                    },
                    ExprNode::Solve { matrix, rhs } => KernelInstruction::Solve {
                        matrix: value_id(*matrix)?,
                        rhs: value_id(*rhs)?,
                    },
                    ExprNode::EventScalar(_) | ExprNode::EventP4Component { .. } => {
                        return Ok(None);
                    }
                }
            };
            let facts = model.node_facts(*id).ok_or_else(|| {
                crate::CompileError::InvalidExecutablePlan(format!(
                    "facts for scheduled node {index} are missing"
                ))
            })?;
            let kind = match facts.value_kind {
                ValueKind::Real => KernelValueKind::Real,
                ValueKind::Complex => KernelValueKind::Complex,
                ValueKind::Vector { len } => KernelValueKind::Vector { len },
                ValueKind::Matrix { rows, cols } => KernelValueKind::Matrix { rows, cols },
            };
            let class = if facts.dependency.depends_on_event {
                KernelValueClass::Event
            } else {
                KernelValueClass::Invariant
            };
            let kernel_id = KernelValueId::from_index(values.len());
            values.push(KernelValue {
                kind,
                class,
                instruction,
            });
            value_ids[index] = Some(kernel_id);
        }

        Ok(Some(ScalarKernelIr::new(
            values,
            value_ids[model.graph().root().index()].ok_or_else(|| {
                crate::CompileError::InvalidExecutablePlan("graph root is not scheduled".into())
            })?,
        )?))
    }

    fn cache_kernel_ir(
        model: &CompiledModel,
        nodes: &[ExprId],
    ) -> CompileResult<(Option<CacheKernelIr>, Vec<ExprId>)> {
        if model.cache_plan().is_empty() {
            return Ok((None, Vec::new()));
        }
        let mut value_ids = vec![None; model.graph().nodes().len()];
        let mut values = Vec::with_capacity(nodes.len());
        let mut inputs = Vec::new();
        for id in nodes {
            let index = id.index();
            let operand = |child: ExprId| {
                value_ids[child.index()].ok_or_else(|| {
                    crate::CompileError::InvalidExecutablePlan(format!(
                        "cache node {index} depends on unscheduled node {}",
                        child.index()
                    ))
                })
            };
            let node = model.graph().node(*id).ok_or_else(|| {
                crate::CompileError::InvalidExecutablePlan(format!(
                    "cache node {index} is out of bounds"
                ))
            })?;
            let instruction = match node {
                ExprNode::EventScalar(_) | ExprNode::EventP4Component { .. } => {
                    let slot = inputs.len();
                    inputs.push(*id);
                    KernelInstruction::Cached(slot)
                }
                ExprNode::RealConst(value) => KernelInstruction::RealConstant(*value),
                ExprNode::ComplexConst(value) => KernelInstruction::ComplexConstant(*value),
                ExprNode::Unary { op, input } => KernelInstruction::Unary {
                    op: *op,
                    input: operand(*input)?,
                },
                ExprNode::Binary { op, lhs, rhs } => KernelInstruction::Binary {
                    op: *op,
                    lhs: operand(*lhs)?,
                    rhs: operand(*rhs)?,
                },
                ExprNode::NaryAdd { terms } => KernelInstruction::Add(
                    terms
                        .iter()
                        .map(|term| operand(*term))
                        .collect::<CompileResult<_>>()?,
                ),
                ExprNode::NaryMul { factors } => KernelInstruction::Mul(
                    factors
                        .iter()
                        .map(|factor| operand(*factor))
                        .collect::<CompileResult<_>>()?,
                ),
                ExprNode::Complex { re, im } => KernelInstruction::Complex {
                    re: operand(*re)?,
                    im: operand(*im)?,
                },
                ExprNode::Vector { elements } => KernelInstruction::Vector(
                    elements
                        .iter()
                        .map(|element| operand(*element))
                        .collect::<CompileResult<_>>()?,
                ),
                ExprNode::Matrix {
                    rows,
                    cols,
                    elements,
                } => KernelInstruction::Matrix {
                    rows: *rows,
                    cols: *cols,
                    elements: elements
                        .iter()
                        .map(|element| operand(*element))
                        .collect::<CompileResult<_>>()?,
                },
                ExprNode::Component { input, index } => KernelInstruction::Component {
                    input: operand(*input)?,
                    index: *index,
                },
                ExprNode::MatrixElement { input, row, col } => KernelInstruction::MatrixElement {
                    input: operand(*input)?,
                    row: *row,
                    col: *col,
                },
                ExprNode::MatMul { lhs, rhs } => KernelInstruction::MatMul {
                    lhs: operand(*lhs)?,
                    rhs: operand(*rhs)?,
                },
                ExprNode::MatVec { matrix, vector } => KernelInstruction::MatVec {
                    matrix: operand(*matrix)?,
                    vector: operand(*vector)?,
                },
                ExprNode::Dot { lhs, rhs } => KernelInstruction::Dot {
                    lhs: operand(*lhs)?,
                    rhs: operand(*rhs)?,
                },
                unsupported => {
                    let _ = unsupported;
                    return Ok((None, Vec::new()));
                }
            };
            let facts = model.node_facts(*id).ok_or_else(|| {
                crate::CompileError::InvalidExecutablePlan(format!(
                    "facts for cache node {index} are missing"
                ))
            })?;
            let kind = match facts.value_kind {
                ValueKind::Real => KernelValueKind::Real,
                ValueKind::Complex => KernelValueKind::Complex,
                ValueKind::Vector { len } => KernelValueKind::Vector { len },
                ValueKind::Matrix { rows, cols } => KernelValueKind::Matrix { rows, cols },
            };
            let kernel_id = KernelValueId::from_index(values.len());
            values.push(KernelValue {
                kind,
                class: if facts.dependency.depends_on_event {
                    KernelValueClass::Event
                } else {
                    KernelValueClass::Invariant
                },
                instruction,
            });
            value_ids[index] = Some(kernel_id);
        }
        let outputs = model
            .cache_plan()
            .entries()
            .iter()
            .map(|entry| {
                value_ids[entry.node().index()].ok_or_else(|| {
                    crate::CompileError::InvalidExecutablePlan(format!(
                        "cache output node {} is not scheduled",
                        entry.node().index()
                    ))
                })
            })
            .collect::<CompileResult<Vec<_>>>()?;
        Ok((Some(CacheKernelIr::new(values, outputs)?), inputs))
    }

    fn evaluation_schedule(
        graph: &ExprGraph,
        cache_slots: &[Option<usize>],
        solve_components: &[Option<SolveComponentPlan>],
        solve_rhs_elements: &[Option<Vec<ExprId>>],
    ) -> CompileResult<(Vec<ExprId>, Vec<Option<usize>>)> {
        let mut required = vec![false; graph.nodes().len()];
        Self::mark_evaluation_node(
            graph,
            graph.root(),
            cache_slots,
            solve_components,
            solve_rhs_elements,
            &mut required,
        );
        let nodes = Self::ids_from_flags(required)?;
        let mut value_slots = vec![None; graph.nodes().len()];
        for (slot, id) in nodes.iter().enumerate() {
            value_slots[id.index()] = Some(slot);
        }
        Ok((nodes, value_slots))
    }

    fn mark_evaluation_node(
        graph: &ExprGraph,
        id: ExprId,
        cache_slots: &[Option<usize>],
        solve_components: &[Option<SolveComponentPlan>],
        solve_rhs_elements: &[Option<Vec<ExprId>>],
        required: &mut [bool],
    ) {
        if required[id.index()] {
            return;
        }
        required[id.index()] = true;
        if cache_slots[id.index()].is_some() {
            return;
        }
        if let Some(plan) = solve_components[id.index()] {
            if let Some(elements) = &solve_rhs_elements[plan.rhs.index()] {
                for element in elements {
                    Self::mark_evaluation_node(
                        graph,
                        *element,
                        cache_slots,
                        solve_components,
                        solve_rhs_elements,
                        required,
                    );
                }
            } else {
                Self::mark_evaluation_node(
                    graph,
                    plan.rhs,
                    cache_slots,
                    solve_components,
                    solve_rhs_elements,
                    required,
                );
            }
            return;
        }
        if let Some(node) = graph.node(id) {
            for child in node.child_ids() {
                Self::mark_evaluation_node(
                    graph,
                    child,
                    cache_slots,
                    solve_components,
                    solve_rhs_elements,
                    required,
                );
            }
        }
    }

    fn mark_required(graph: &ExprGraph, id: ExprId, required: &mut [bool]) {
        if required[id.index()] {
            return;
        }
        required[id.index()] = true;
        if let Some(node) = graph.node(id) {
            for child in node.child_ids() {
                Self::mark_required(graph, child, required);
            }
        }
    }

    fn ids_from_flags(flags: Vec<bool>) -> CompileResult<Vec<ExprId>> {
        flags
            .into_iter()
            .enumerate()
            .filter_map(|(index, required)| {
                required.then(|| {
                    ExprId::from_index(index).ok_or_else(|| {
                        crate::CompileError::InvalidExecutablePlan(
                            "expression graph exceeds supported node count".into(),
                        )
                    })
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use laddu_expr::{event_scalar, matrix, matvec, parameter, solve, vector};

    use super::*;

    #[test]
    fn executable_plan_lowers_scalar_kernel_and_cache_boundary() {
        let model = CompiledModel::from_expr(
            &(event_scalar("x") * parameter!("scale") + parameter!("offset")),
        )
        .unwrap();
        let plan = ExecutablePlan::from_model(&model).unwrap();

        assert_eq!(plan.params().n_free(), 2);
        assert!(!plan.cache_plan().is_empty());
        let kernel = plan.scalar_kernel().unwrap();
        assert!(
            kernel
                .values()
                .iter()
                .any(|value| matches!(value.instruction, KernelInstruction::Cached(_)))
        );
        assert!(
            kernel
                .values()
                .iter()
                .any(|value| matches!(value.instruction, KernelInstruction::Parameter(_)))
        );
    }

    #[test]
    fn executable_plan_lowers_computed_scalar_cache_kernel() {
        let model = CompiledModel::from_expr(&event_scalar("x").sin()).unwrap();
        let plan = ExecutablePlan::from_model(&model).unwrap();
        let kernel = plan.cache_kernel().unwrap();

        assert_eq!(plan.cache_input_nodes().len(), 1);
        assert_eq!(kernel.outputs().len(), model.cache_plan().len());
        assert!(kernel.values().iter().any(|value| {
            matches!(
                value.instruction,
                KernelInstruction::Unary {
                    op: laddu_expr::UnaryOp::Sin,
                    ..
                }
            )
        }));
    }

    #[test]
    fn executable_plan_lowers_aggregate_cache_kernel() {
        let cached_matrix = matrix([
            [event_scalar("x"), event_scalar("y")],
            [event_scalar("y") + 1.0, event_scalar("x") - 1.0],
        ]);
        let expression =
            matvec(cached_matrix, vector([parameter!("a"), parameter!("b")])).component(1);
        let model = CompiledModel::from_expr_with_options(
            &expression,
            &crate::CompileOptions::without_optimizations(),
        )
        .unwrap();
        let plan = ExecutablePlan::from_model(&model).unwrap();
        let cache = plan.cache_kernel().unwrap();

        assert!(cache.values().iter().any(|value| {
            matches!(
                value.instruction,
                KernelInstruction::Matrix {
                    rows: 2,
                    cols: 2,
                    ..
                }
            ) && value.kind == KernelValueKind::Matrix { rows: 2, cols: 2 }
        }));
        assert!(plan.scalar_kernel().unwrap().values().iter().any(|value| {
            matches!(value.instruction, KernelInstruction::Cached(_))
                && value.kind == KernelValueKind::Matrix { rows: 2, cols: 2 }
        }));
        assert!(plan.scalar_kernel().unwrap().values().iter().any(|value| {
            matches!(value.instruction, KernelInstruction::MatVec { .. })
                && value.kind == KernelValueKind::Vector { len: 2 }
        }));
    }

    #[test]
    fn executable_plan_specializes_selected_event_solve_rows() {
        let expression = solve(
            matrix([[event_scalar("x") + 2.0]]),
            vector([parameter!("rhs")]),
        )
        .component(0);
        let model = CompiledModel::from_expr(&expression).unwrap();
        let plan = ExecutablePlan::from_model(&model).unwrap();

        assert_eq!(plan.solve_row_matrices().len(), 1);
        assert_eq!(plan.solve_row_matrices()[0].dimension(), 1);
        assert_eq!(plan.solve_row_keys().len(), 1);
        assert!(
            plan.scalar_kernel()
                .unwrap()
                .values()
                .iter()
                .any(|value| matches!(
                    value.instruction,
                    KernelInstruction::SolveRow { row_slot: 0, .. }
                ))
        );
    }

    #[test]
    fn executable_plan_preserves_generic_fallback_without_caches() {
        let model = CompiledModel::from_expr_with_options(
            &event_scalar("x"),
            &crate::CompileOptions::default().with_cache_policy(crate::CachePolicy::Off),
        )
        .unwrap();
        let plan = ExecutablePlan::from_model(&model).unwrap();

        assert!(plan.scalar_kernel().is_none());
    }
}
