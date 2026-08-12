use std::collections::HashMap;

use laddu_expr::{
    ExprGraph, ExprId, ExprNode, ValueKind,
    parameters::{ParamId, ParamLayout},
};
use laddu_kernel::ir::{
    CacheKernelIr, KernelInstruction, KernelValue, KernelValueClass, KernelValueId,
    KernelValueKind, ScalarKernelIr,
};

use crate::{CachePlan, CompileResult, CompiledModel, graph_utils::mark_reachable};

/// Specialized plan for reading one component of an event-dependent linear solve.
#[derive(Copy, Clone, Debug)]
pub struct SolveComponentPlan {
    rhs: ExprId,
    row_slot: usize,
    dimension: usize,
}

impl SolveComponentPlan {
    /// Returns the solve right-hand-side node.
    pub fn rhs(self) -> ExprId {
        self.rhs
    }

    /// Returns the cached factorization row slot.
    pub fn row_slot(self) -> usize {
        self.row_slot
    }

    /// Returns the square matrix dimension.
    pub fn dimension(self) -> usize {
        self.dimension
    }
}

/// Cached row layout for one event-dependent solve matrix.
#[derive(Clone, Debug)]
pub struct SolveRowMatrixPlan {
    matrix: ExprId,
    dimension: usize,
    rows: Vec<(usize, usize)>,
}

#[derive(Clone, Debug)]
struct NodeBindings {
    parameter_slots: Vec<Option<ParamId>>,
    cache_slots: Vec<Option<usize>>,
    evaluation_nodes: Vec<ExprId>,
    value_slots: Vec<Option<usize>>,
}

#[derive(Clone, Debug)]
struct SolvePlan {
    components: Vec<Option<SolveComponentPlan>>,
    rhs_elements: Vec<Option<Vec<ExprId>>>,
    row_matrices: Vec<SolveRowMatrixPlan>,
    row_keys: Vec<(ExprId, usize, usize)>,
}

#[derive(Clone, Debug)]
struct FactorizationPlan {
    event_slots: Vec<Option<usize>>,
    event_matrices: Vec<(ExprId, usize)>,
    constant_slots: Vec<Option<usize>>,
    constant_matrices: Vec<(ExprId, usize)>,
}

impl SolveRowMatrixPlan {
    /// Returns the matrix graph node.
    pub fn matrix(&self) -> ExprId {
        self.matrix
    }

    /// Returns the square matrix dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns `(row_index, cache_slot)` pairs.
    pub fn rows(&self) -> &[(usize, usize)] {
        &self.rows
    }
}

/// Backend-neutral schedule, cache layout, and lowered kernel IR for a model.
#[derive(Clone, Debug)]
pub struct ExecutablePlan {
    graph: ExprGraph,
    params: ParamLayout,
    cache_plan: CachePlan,
    bindings: NodeBindings,
    scalar_kernel: Option<ScalarKernelIr>,
    cache_kernel: Option<CacheKernelIr>,
    cache_input_nodes: Vec<ExprId>,
    cache_materialization_nodes: Vec<ExprId>,
    solve: SolvePlan,
    factorization: FactorizationPlan,
}

impl ExecutablePlan {
    /// Lowers a compiled model into an executable plan.
    ///
    /// # Errors
    ///
    /// Returns [`CompileError`](crate::CompileError) when the model cannot be
    /// lowered to valid scalar or cache kernel IR.
    pub fn from_model(model: &CompiledModel) -> CompileResult<Self> {
        Self::from_model_with_solve_rows(model, true)
    }

    /// Build an executable plan without the CPU-oriented cached solve-row specialization.
    ///
    /// Backends with inexpensive fused solves can use this form to keep the original `Solve`
    /// instruction and consume an ordinarily cached event-dependent matrix directly.
    ///
    /// # Errors
    ///
    /// Returns [`CompileError`](crate::CompileError) when the model cannot be
    /// lowered to valid scalar or cache kernel IR.
    pub fn from_model_without_solve_rows(model: &CompiledModel) -> CompileResult<Self> {
        Self::from_model_with_solve_rows(model, false)
    }

    fn from_model_with_solve_rows(
        model: &CompiledModel,
        specialize_solve_rows: bool,
    ) -> CompileResult<Self> {
        PlanBuilder::new(model, specialize_solve_rows).build()
    }

    /// Returns the optimized expression graph.
    pub fn graph(&self) -> &ExprGraph {
        &self.graph
    }

    /// Returns the model parameter layout.
    pub fn params(&self) -> &ParamLayout {
        &self.params
    }

    /// Maps graph nodes to parameter identifiers.
    pub fn parameter_slots(&self) -> &[Option<ParamId>] {
        &self.bindings.parameter_slots
    }

    /// Returns the ordinary event-cache plan.
    pub fn cache_plan(&self) -> &CachePlan {
        &self.cache_plan
    }

    /// Maps graph nodes to ordinary cache slots.
    pub fn cache_slots(&self) -> &[Option<usize>] {
        &self.bindings.cache_slots
    }

    /// Returns nodes evaluated by the scalar schedule.
    pub fn evaluation_nodes(&self) -> &[ExprId] {
        &self.bindings.evaluation_nodes
    }

    /// Maps graph nodes to scalar schedule value slots.
    pub fn value_slots(&self) -> &[Option<usize>] {
        &self.bindings.value_slots
    }

    /// Returns lowered scalar kernel IR, if the plan has scalar work.
    pub fn scalar_kernel(&self) -> Option<&ScalarKernelIr> {
        self.scalar_kernel.as_ref()
    }

    /// Returns lowered cache-materialization kernel IR, if required.
    pub fn cache_kernel(&self) -> Option<&CacheKernelIr> {
        self.cache_kernel.as_ref()
    }

    /// Returns graph nodes supplied as inputs to the cache kernel.
    pub fn cache_input_nodes(&self) -> &[ExprId] {
        &self.cache_input_nodes
    }

    /// Returns nodes evaluated while materializing all caches.
    pub fn cache_materialization_nodes(&self) -> &[ExprId] {
        &self.cache_materialization_nodes
    }

    /// Maps graph nodes to specialized solve-component plans.
    pub fn solve_components(&self) -> &[Option<SolveComponentPlan>] {
        &self.solve.components
    }

    /// Maps solve nodes to flattened right-hand-side element nodes.
    pub fn solve_rhs_elements(&self) -> &[Option<Vec<ExprId>>] {
        &self.solve.rhs_elements
    }

    /// Returns event-dependent matrices using cached solve rows.
    pub fn solve_row_matrices(&self) -> &[SolveRowMatrixPlan] {
        &self.solve.row_matrices
    }

    /// Returns keys identifying specialized matrix rows.
    pub fn solve_row_keys(&self) -> &[(ExprId, usize, usize)] {
        &self.solve.row_keys
    }

    /// Maps event-dependent matrix nodes to factorization slots.
    pub fn factor_matrix_slots(&self) -> &[Option<usize>] {
        &self.factorization.event_slots
    }

    /// Returns event-dependent matrices to factor per event.
    pub fn factor_matrices(&self) -> &[(ExprId, usize)] {
        &self.factorization.event_matrices
    }

    /// Maps invariant matrix nodes to constant factorization slots.
    pub fn constant_factor_slots(&self) -> &[Option<usize>] {
        &self.factorization.constant_slots
    }

    /// Returns invariant matrices to factor once.
    pub fn constant_factor_matrices(&self) -> &[(ExprId, usize)] {
        &self.factorization.constant_matrices
    }

    fn solve_component_plans(model: &CompiledModel) -> CompileResult<SolvePlan> {
        let node_count = model.graph().nodes().len();
        let mut components = vec![None; node_count];
        let mut rhs_elements = vec![None; node_count];
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

        SolvePlan::new(node_count, components, rhs_elements, matrices, row_keys)
    }

    fn scalar_kernel_ir(
        model: &CompiledModel,
        evaluation_nodes: &[ExprId],
        cache_slots: &[Option<usize>],
        parameter_slots: &[Option<ParamId>],
        solve_components: &[Option<SolveComponentPlan>],
        solve_rhs_elements: &[Option<Vec<ExprId>>],
    ) -> CompileResult<Option<ScalarKernelIr>> {
        let mut boundary = ScalarBoundary {
            cache_slots,
            parameter_slots,
            solve_components,
            solve_rhs_elements,
        };
        let lowered = KernelLowerer::new(model, evaluation_nodes).lower(&mut boundary)?;
        let LoweringOutcome::Lowered(lowered) = lowered else {
            return Ok(None);
        };
        let root = lowered.value_ids[model.graph().root().index()].ok_or_else(|| {
            crate::CompileError::InvalidExecutablePlan("graph root is not scheduled".into())
        })?;
        Ok(Some(ScalarKernelIr::new(lowered.values, root)?))
    }

    fn cache_kernel_ir(
        model: &CompiledModel,
        nodes: &[ExprId],
    ) -> CompileResult<(Option<CacheKernelIr>, Vec<ExprId>)> {
        if model.cache_plan().is_empty() {
            return Ok((None, Vec::new()));
        }
        let mut boundary = CacheBoundary::default();
        let lowered = KernelLowerer::new(model, nodes).lower(&mut boundary)?;
        let LoweringOutcome::Lowered(lowered) = lowered else {
            return Ok((None, Vec::new()));
        };
        let outputs = model
            .cache_plan()
            .entries()
            .iter()
            .map(|entry| {
                lowered.value_ids[entry.node().index()].ok_or_else(|| {
                    crate::CompileError::InvalidExecutablePlan(format!(
                        "cache output node {} is not scheduled",
                        entry.node().index()
                    ))
                })
            })
            .collect::<CompileResult<Vec<_>>>()?;
        Ok((
            Some(CacheKernelIr::new(lowered.values, outputs)?),
            boundary.inputs,
        ))
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
        let mut stack = vec![id];
        while let Some(id) = stack.pop() {
            if required[id.index()] {
                continue;
            }
            required[id.index()] = true;
            if cache_slots[id.index()].is_some() {
                continue;
            }
            if let Some(plan) = solve_components[id.index()] {
                if let Some(elements) = &solve_rhs_elements[plan.rhs.index()] {
                    stack.extend(elements.iter().rev().copied());
                } else {
                    stack.push(plan.rhs);
                }
                continue;
            }
            if let Some(node) = graph.node(id) {
                stack.extend(node.children().rev());
            }
        }
    }

    fn ids_from_flags(flags: Vec<bool>) -> CompileResult<Vec<ExprId>> {
        flags
            .into_iter()
            .enumerate()
            .filter(|&(_index, required)| required)
            .map(|(index, _required)| Ok(ExprId::from_index(index)))
            .collect()
    }
}

struct PlanBuilder<'a> {
    model: &'a CompiledModel,
    specialize_solve_rows: bool,
}

impl<'a> PlanBuilder<'a> {
    fn new(model: &'a CompiledModel, specialize_solve_rows: bool) -> Self {
        Self {
            model,
            specialize_solve_rows,
        }
    }

    fn build(self) -> CompileResult<ExecutablePlan> {
        let graph = self.model.graph().clone();
        let params = self.model.params().clone();
        let cache_plan = self.model.cache_plan().clone();
        let solve = if self.specialize_solve_rows {
            ExecutablePlan::solve_component_plans(self.model)?
        } else {
            SolvePlan::empty(graph.nodes().len())
        };
        let bindings = NodeBindings::build(self.model, &solve)?;
        let scalar_kernel = ExecutablePlan::scalar_kernel_ir(
            self.model,
            &bindings.evaluation_nodes,
            &bindings.cache_slots,
            &bindings.parameter_slots,
            &solve.components,
            &solve.rhs_elements,
        )?;

        let mut cache_required_nodes = vec![false; graph.nodes().len()];
        mark_reachable(
            &graph,
            cache_plan
                .materialization_nodes()
                .iter()
                .copied()
                .chain(solve.row_matrices.iter().map(|plan| plan.matrix)),
            &mut cache_required_nodes,
        );
        let cache_materialization_nodes = ExecutablePlan::ids_from_flags(cache_required_nodes)?;
        let (cache_kernel, cache_input_nodes) =
            ExecutablePlan::cache_kernel_ir(self.model, &cache_materialization_nodes)?;
        let factorization = FactorizationPlan::build(self.model, &bindings.value_slots)?;

        Ok(ExecutablePlan {
            graph,
            params,
            cache_plan,
            bindings,
            scalar_kernel,
            cache_kernel,
            cache_input_nodes,
            cache_materialization_nodes,
            solve,
            factorization,
        })
    }
}

impl NodeBindings {
    fn build(model: &CompiledModel, solve: &SolvePlan) -> CompileResult<Self> {
        let graph = model.graph();
        let node_count = graph.nodes().len();
        let parameter_slots = graph
            .nodes()
            .iter()
            .map(|node| match node {
                ExprNode::ScalarParam(parameter) => model.params().id(parameter.name()),
                _ => None,
            })
            .collect::<Vec<_>>();
        let mut cache_slots = vec![None; node_count];
        for (slot, entry) in model.cache_plan().entries().iter().enumerate() {
            let entry_slot = cache_slots.get_mut(entry.node().index()).ok_or_else(|| {
                crate::CompileError::InvalidExecutablePlan(format!(
                    "cache entry node {} is out of bounds for {node_count} nodes",
                    entry.node().index()
                ))
            })?;
            *entry_slot = Some(slot);
        }
        let (evaluation_nodes, value_slots) = ExecutablePlan::evaluation_schedule(
            graph,
            &cache_slots,
            &solve.components,
            &solve.rhs_elements,
        )?;
        let bindings = Self {
            parameter_slots,
            cache_slots,
            evaluation_nodes,
            value_slots,
        };
        bindings.validate(node_count)?;
        Ok(bindings)
    }

    fn validate(&self, node_count: usize) -> CompileResult<()> {
        for (name, len) in [
            ("parameter slots", self.parameter_slots.len()),
            ("cache slots", self.cache_slots.len()),
            ("value slots", self.value_slots.len()),
        ] {
            if len != node_count {
                return Err(crate::CompileError::InvalidExecutablePlan(format!(
                    "{name} length {len} does not match graph length {node_count}"
                )));
            }
        }
        for (expected_slot, id) in self.evaluation_nodes.iter().enumerate() {
            if self.value_slots.get(id.index()).copied().flatten() != Some(expected_slot) {
                return Err(crate::CompileError::InvalidExecutablePlan(format!(
                    "evaluation node {} is not bound to value slot {expected_slot}",
                    id.index()
                )));
            }
        }
        Ok(())
    }
}

impl SolvePlan {
    fn empty(node_count: usize) -> Self {
        Self {
            components: vec![None; node_count],
            rhs_elements: vec![None; node_count],
            row_matrices: Vec::new(),
            row_keys: Vec::new(),
        }
    }

    fn new(
        node_count: usize,
        components: Vec<Option<SolveComponentPlan>>,
        rhs_elements: Vec<Option<Vec<ExprId>>>,
        row_matrices: Vec<SolveRowMatrixPlan>,
        row_keys: Vec<(ExprId, usize, usize)>,
    ) -> CompileResult<Self> {
        if components.len() != node_count || rhs_elements.len() != node_count {
            return Err(crate::CompileError::InvalidExecutablePlan(
                "solve plan node maps do not match the graph length".into(),
            ));
        }
        for component in components.iter().flatten() {
            if component.rhs.index() >= node_count
                || component.row_slot >= row_keys.len()
                || row_keys[component.row_slot].2 != component.dimension
            {
                return Err(crate::CompileError::InvalidExecutablePlan(
                    "solve component references an invalid RHS node or row slot".into(),
                ));
            }
        }
        if rhs_elements
            .iter()
            .flatten()
            .flatten()
            .any(|id| id.index() >= node_count)
            || row_keys.iter().any(|(matrix, row, dimension)| {
                matrix.index() >= node_count || *dimension == 0 || *row >= *dimension
            })
        {
            return Err(crate::CompileError::InvalidExecutablePlan(
                "solve plan contains an invalid graph node or row key".into(),
            ));
        }
        for matrix in &row_matrices {
            if matrix.matrix.index() >= node_count
                || matrix.rows.iter().any(|(slot, row)| {
                    *slot >= row_keys.len()
                        || *row >= matrix.dimension
                        || row_keys[*slot] != (matrix.matrix, *row, matrix.dimension)
                })
            {
                return Err(crate::CompileError::InvalidExecutablePlan(
                    "solve row matrix contains an invalid node, row, or slot".into(),
                ));
            }
        }
        Ok(Self {
            components,
            rhs_elements,
            row_matrices,
            row_keys,
        })
    }
}

impl FactorizationPlan {
    fn build(model: &CompiledModel, value_slots: &[Option<usize>]) -> CompileResult<Self> {
        let node_count = model.graph().nodes().len();
        if value_slots.len() != node_count {
            return Err(crate::CompileError::InvalidExecutablePlan(
                "factorization value slots do not match the graph length".into(),
            ));
        }
        let mut plan = Self {
            event_slots: vec![None; node_count],
            event_matrices: Vec::new(),
            constant_slots: vec![None; node_count],
            constant_matrices: Vec::new(),
        };
        for (index, node) in model.graph().nodes().iter().enumerate() {
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
            let (slots, matrices) = if facts.dependency.depends_on_event {
                (&mut plan.event_slots, &mut plan.event_matrices)
            } else {
                (&mut plan.constant_slots, &mut plan.constant_matrices)
            };
            if slots[matrix.index()].is_none() {
                let slot = matrices.len();
                slots[matrix.index()] = Some(slot);
                matrices.push((*matrix, rows));
            }
        }
        plan.validate(node_count)?;
        Ok(plan)
    }

    fn validate(&self, node_count: usize) -> CompileResult<()> {
        for (slots, matrices, name) in [
            (&self.event_slots, &self.event_matrices, "event"),
            (&self.constant_slots, &self.constant_matrices, "constant"),
        ] {
            if slots.len() != node_count {
                return Err(crate::CompileError::InvalidExecutablePlan(format!(
                    "{name} factor slots do not match the graph length"
                )));
            }
            if slots.iter().flatten().any(|slot| *slot >= matrices.len()) {
                return Err(crate::CompileError::InvalidExecutablePlan(format!(
                    "{name} factor slots contain an invalid matrix slot"
                )));
            }
            for (slot, (matrix, _dimension)) in matrices.iter().enumerate() {
                if slots.get(matrix.index()).copied().flatten() != Some(slot) {
                    return Err(crate::CompileError::InvalidExecutablePlan(format!(
                        "{name} factor matrix {} is not bound to slot {slot}",
                        matrix.index()
                    )));
                }
            }
        }
        Ok(())
    }
}

#[derive(Copy, Clone, Debug)]
enum LoweringFailure {
    EventLeaf,
    ParameterLeaf,
}

enum LoweringOutcome<T> {
    Lowered(T),
    Unsupported(LoweringFailure),
}

struct LoweredKernel {
    values: Vec<KernelValue>,
    value_ids: Vec<Option<KernelValueId>>,
}

trait LoweringBoundary {
    fn context(&self) -> &'static str;

    fn scheduled_context(&self) -> &'static str {
        self.context()
    }

    fn cached(&self, _id: ExprId) -> Option<KernelInstruction> {
        None
    }

    fn parameter(&mut self, id: ExprId) -> CompileResult<LoweringOutcome<KernelInstruction>>;

    fn event(&mut self, id: ExprId) -> LoweringOutcome<KernelInstruction>;

    fn component(
        &mut self,
        id: ExprId,
        input: ExprId,
        index: usize,
        operand: &dyn Fn(ExprId) -> CompileResult<KernelValueId>,
    ) -> CompileResult<LoweringOutcome<KernelInstruction>>;
}

struct ScalarBoundary<'a> {
    cache_slots: &'a [Option<usize>],
    parameter_slots: &'a [Option<ParamId>],
    solve_components: &'a [Option<SolveComponentPlan>],
    solve_rhs_elements: &'a [Option<Vec<ExprId>>],
}

impl LoweringBoundary for ScalarBoundary<'_> {
    fn context(&self) -> &'static str {
        "node"
    }

    fn scheduled_context(&self) -> &'static str {
        "scheduled node"
    }

    fn cached(&self, id: ExprId) -> Option<KernelInstruction> {
        self.cache_slots[id.index()].map(KernelInstruction::Cached)
    }

    fn parameter(&mut self, id: ExprId) -> CompileResult<LoweringOutcome<KernelInstruction>> {
        let parameter = self.parameter_slots[id.index()].ok_or_else(|| {
            crate::CompileError::InvalidExecutablePlan(format!(
                "parameter node {} is not bound",
                id.index()
            ))
        })?;
        Ok(LoweringOutcome::Lowered(KernelInstruction::Parameter(
            parameter,
        )))
    }

    fn event(&mut self, _id: ExprId) -> LoweringOutcome<KernelInstruction> {
        LoweringOutcome::Unsupported(LoweringFailure::EventLeaf)
    }

    fn component(
        &mut self,
        id: ExprId,
        input: ExprId,
        index: usize,
        operand: &dyn Fn(ExprId) -> CompileResult<KernelValueId>,
    ) -> CompileResult<LoweringOutcome<KernelInstruction>> {
        let instruction = if let Some(solve) = self.solve_components[id.index()]
            && let Some(elements) = self.solve_rhs_elements[solve.rhs.index()].as_ref()
        {
            KernelInstruction::SolveRow {
                row_slot: solve.row_slot,
                rhs: elements
                    .iter()
                    .map(|element| operand(*element))
                    .collect::<CompileResult<_>>()?,
            }
        } else {
            KernelInstruction::Component {
                input: operand(input)?,
                index,
            }
        };
        Ok(LoweringOutcome::Lowered(instruction))
    }
}

#[derive(Default)]
struct CacheBoundary {
    inputs: Vec<ExprId>,
}

impl LoweringBoundary for CacheBoundary {
    fn context(&self) -> &'static str {
        "cache node"
    }

    fn parameter(&mut self, _id: ExprId) -> CompileResult<LoweringOutcome<KernelInstruction>> {
        Ok(LoweringOutcome::Unsupported(LoweringFailure::ParameterLeaf))
    }

    fn event(&mut self, id: ExprId) -> LoweringOutcome<KernelInstruction> {
        let slot = self.inputs.len();
        self.inputs.push(id);
        LoweringOutcome::Lowered(KernelInstruction::Cached(slot))
    }

    fn component(
        &mut self,
        _id: ExprId,
        input: ExprId,
        index: usize,
        operand: &dyn Fn(ExprId) -> CompileResult<KernelValueId>,
    ) -> CompileResult<LoweringOutcome<KernelInstruction>> {
        Ok(LoweringOutcome::Lowered(KernelInstruction::Component {
            input: operand(input)?,
            index,
        }))
    }
}

struct KernelLowerer<'a> {
    model: &'a CompiledModel,
    nodes: &'a [ExprId],
}

impl<'a> KernelLowerer<'a> {
    fn new(model: &'a CompiledModel, nodes: &'a [ExprId]) -> Self {
        Self { model, nodes }
    }

    fn lower(
        self,
        boundary: &mut impl LoweringBoundary,
    ) -> CompileResult<LoweringOutcome<LoweredKernel>> {
        let mut value_ids = vec![None; self.model.graph().nodes().len()];
        let mut values = Vec::with_capacity(self.nodes.len());
        for id in self.nodes {
            let index = id.index();
            let context = boundary.context();
            let scheduled_context = boundary.scheduled_context();
            let operand = |child: ExprId| {
                value_ids
                    .get(child.index())
                    .copied()
                    .flatten()
                    .ok_or_else(|| {
                        crate::CompileError::InvalidExecutablePlan(format!(
                            "{} {index} depends on unscheduled node {}",
                            context,
                            child.index()
                        ))
                    })
            };
            let instruction = if let Some(instruction) = boundary.cached(*id) {
                LoweringOutcome::Lowered(instruction)
            } else {
                let node = self.model.graph().node(*id).ok_or_else(|| {
                    crate::CompileError::InvalidExecutablePlan(format!(
                        "{} {index} is out of bounds",
                        scheduled_context
                    ))
                })?;
                match node {
                    ExprNode::RealConst(value) => {
                        LoweringOutcome::Lowered(KernelInstruction::RealConstant(*value))
                    }
                    ExprNode::ComplexConst(value) => {
                        LoweringOutcome::Lowered(KernelInstruction::ComplexConstant(*value))
                    }
                    ExprNode::ScalarParam(_) => boundary.parameter(*id)?,
                    ExprNode::EventScalar(_) | ExprNode::EventP4Component { .. } => {
                        boundary.event(*id)
                    }
                    ExprNode::Unary { op, input } => {
                        LoweringOutcome::Lowered(KernelInstruction::Unary {
                            op: *op,
                            input: operand(*input)?,
                        })
                    }
                    ExprNode::Binary { op, lhs, rhs } => {
                        LoweringOutcome::Lowered(KernelInstruction::Binary {
                            op: *op,
                            lhs: operand(*lhs)?,
                            rhs: operand(*rhs)?,
                        })
                    }
                    ExprNode::NaryAdd { terms } => LoweringOutcome::Lowered(
                        KernelInstruction::Add(Self::operands(terms, &operand)?),
                    ),
                    ExprNode::NaryMul { factors } => LoweringOutcome::Lowered(
                        KernelInstruction::Mul(Self::operands(factors, &operand)?),
                    ),
                    ExprNode::Complex { re, im } => {
                        LoweringOutcome::Lowered(KernelInstruction::Complex {
                            re: operand(*re)?,
                            im: operand(*im)?,
                        })
                    }
                    ExprNode::Vector { elements } => LoweringOutcome::Lowered(
                        KernelInstruction::Vector(Self::operands(elements, &operand)?),
                    ),
                    ExprNode::Matrix {
                        rows,
                        cols,
                        elements,
                    } => LoweringOutcome::Lowered(KernelInstruction::Matrix {
                        rows: *rows,
                        cols: *cols,
                        elements: Self::operands(elements, &operand)?,
                    }),
                    ExprNode::Component { input, index } => {
                        boundary.component(*id, *input, *index, &operand)?
                    }
                    ExprNode::MatrixElement { input, row, col } => {
                        LoweringOutcome::Lowered(KernelInstruction::MatrixElement {
                            input: operand(*input)?,
                            row: *row,
                            col: *col,
                        })
                    }
                    ExprNode::MatMul { lhs, rhs } => {
                        LoweringOutcome::Lowered(KernelInstruction::MatMul {
                            lhs: operand(*lhs)?,
                            rhs: operand(*rhs)?,
                        })
                    }
                    ExprNode::MatVec { matrix, vector } => {
                        LoweringOutcome::Lowered(KernelInstruction::MatVec {
                            matrix: operand(*matrix)?,
                            vector: operand(*vector)?,
                        })
                    }
                    ExprNode::Dot { lhs, rhs } => {
                        LoweringOutcome::Lowered(KernelInstruction::Dot {
                            lhs: operand(*lhs)?,
                            rhs: operand(*rhs)?,
                        })
                    }
                    ExprNode::Solve { matrix, rhs } => {
                        LoweringOutcome::Lowered(KernelInstruction::Solve {
                            matrix: operand(*matrix)?,
                            rhs: operand(*rhs)?,
                        })
                    }
                }
            };
            let instruction = match instruction {
                LoweringOutcome::Lowered(instruction) => instruction,
                LoweringOutcome::Unsupported(reason) => {
                    return Ok(LoweringOutcome::Unsupported(reason));
                }
            };
            let facts = self.model.node_facts(*id).ok_or_else(|| {
                crate::CompileError::InvalidExecutablePlan(format!(
                    "facts for {} {index} are missing",
                    scheduled_context
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
        Ok(LoweringOutcome::Lowered(LoweredKernel {
            values,
            value_ids,
        }))
    }

    fn operands(
        ids: &[ExprId],
        operand: &dyn Fn(ExprId) -> CompileResult<KernelValueId>,
    ) -> CompileResult<Vec<KernelValueId>> {
        ids.iter().map(|id| operand(*id)).collect()
    }
}

#[cfg(test)]
mod tests {
    use laddu_expr::{Expr, event_scalar, matrix, matvec, parameter, solve, vector};

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

    #[test]
    fn scalar_and_cache_boundaries_share_common_node_lowering() {
        let expression = (vector([Expr::from(1.0), Expr::from(2.0)]).component(1) + 3.0).sin();
        let model = CompiledModel::from_expr_with_options(
            &expression,
            &crate::CompileOptions::without_optimizations()
                .with_cache_policy(crate::CachePolicy::Off),
        )
        .unwrap();
        let nodes = (0..model.graph().nodes().len())
            .map(ExprId::from_index)
            .collect::<Vec<_>>();
        let cache_slots = vec![None; nodes.len()];
        let parameter_slots = vec![None; nodes.len()];
        let solve = SolvePlan::empty(nodes.len());

        let mut scalar_boundary = ScalarBoundary {
            cache_slots: &cache_slots,
            parameter_slots: &parameter_slots,
            solve_components: &solve.components,
            solve_rhs_elements: &solve.rhs_elements,
        };
        let LoweringOutcome::Lowered(scalar) = KernelLowerer::new(&model, &nodes)
            .lower(&mut scalar_boundary)
            .unwrap()
        else {
            panic!("common scalar lowering should be supported");
        };

        let mut cache_boundary = CacheBoundary::default();
        let LoweringOutcome::Lowered(cache) = KernelLowerer::new(&model, &nodes)
            .lower(&mut cache_boundary)
            .unwrap()
        else {
            panic!("common cache lowering should be supported");
        };

        assert!(cache_boundary.inputs.is_empty());
        assert_eq!(scalar.values.len(), cache.values.len());
        for (scalar, cache) in scalar.values.iter().zip(&cache.values) {
            assert_eq!(scalar.kind, cache.kind);
            assert_eq!(scalar.class, cache.class);
            assert_eq!(
                format!("{:?}", scalar.instruction),
                format!("{:?}", cache.instruction)
            );
        }
    }
}
