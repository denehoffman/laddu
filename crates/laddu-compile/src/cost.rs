use laddu_expr::{BinaryOp, ExprGraph, ExprNode, UnaryOp};

use crate::{DependencyFacts, GraphFacts};

/// Weighted work partitioned by the lifecycle that pays for it.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct LifecycleCost {
    compile: u64,
    dataset_event: u64,
    evaluation_invariant: u64,
    evaluation_event: u64,
}

impl LifecycleCost {
    /// Work performed once while compiling or preparing a parameter-only program.
    pub fn compile(&self) -> u64 {
        self.compile
    }

    /// Work performed once per event while preparing a dataset.
    pub fn dataset_event(&self) -> u64 {
        self.dataset_event
    }

    /// Parameter-only work performed once per objective evaluation.
    pub fn evaluation_invariant(&self) -> u64 {
        self.evaluation_invariant
    }

    /// Mixed parameter/event work performed for every event and evaluation.
    pub fn evaluation_event(&self) -> u64 {
        self.evaluation_event
    }

    fn add(&mut self, dependency: DependencyFacts, weight: u64) {
        match (
            dependency.depends_on_free_params,
            dependency.depends_on_event,
        ) {
            (false, false) => self.compile += weight,
            (false, true) => self.dataset_event += weight,
            (true, false) => self.evaluation_invariant += weight,
            (true, true) => self.evaluation_event += weight,
        }
    }

    fn hot_path_key(self) -> (u64, u64, u64, u64) {
        (
            self.evaluation_event,
            self.evaluation_invariant,
            self.dataset_event,
            self.compile,
        )
    }
}

/// Static counts and weighted score used to compare expression graphs.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OptimizationCost {
    node_count: usize,
    weighted_ops: u64,
    free_nodes: usize,
    scalar_adds: usize,
    scalar_muls: usize,
    scalar_divs: usize,
    cheap_unary_ops: usize,
    power_ops: usize,
    transcendental_ops: usize,
    constructors: usize,
    extractions: usize,
    linear_algebra_ops: usize,
    lifecycle: LifecycleCost,
}

impl OptimizationCost {
    /// Computes the cost metrics for `graph`.
    pub fn analyze(graph: &ExprGraph) -> Self {
        let facts = GraphFacts::analyze(graph);
        let mut cost = Self::default();
        for (node, facts) in graph.nodes().iter().zip(facts.nodes()) {
            let before = cost.weighted_ops;
            cost.add_node(node);
            cost.lifecycle
                .add(facts.dependency, cost.weighted_ops - before);
        }
        cost
    }

    /// Returns the total number of graph nodes.
    pub fn node_count(&self) -> usize {
        self.node_count
    }

    /// Returns the weighted operation score.
    pub fn weighted_ops(&self) -> u64 {
        self.weighted_ops
    }

    /// Returns whether this cost is strictly lower than `baseline`.
    pub fn is_better_than(&self, baseline: &Self) -> bool {
        self.lifecycle.hot_path_key() < baseline.lifecycle.hot_path_key()
            || (self.lifecycle == baseline.lifecycle
                && (self.weighted_ops < baseline.weighted_ops
                    || (self.weighted_ops == baseline.weighted_ops
                        && self.node_count < baseline.node_count)))
    }

    /// Returns whether this cost is lower than or equal to `baseline`.
    pub fn is_no_worse_than(&self, baseline: &Self) -> bool {
        self.lifecycle.hot_path_key() < baseline.lifecycle.hot_path_key()
            || (self.lifecycle == baseline.lifecycle
                && (self.weighted_ops < baseline.weighted_ops
                    || (self.weighted_ops == baseline.weighted_ops
                        && self.node_count <= baseline.node_count)))
    }

    /// Returns weighted work split by compilation and execution lifecycle.
    pub fn lifecycle(&self) -> LifecycleCost {
        self.lifecycle
    }

    /// Returns the number of constants, parameters, and event inputs.
    pub fn free_nodes(&self) -> usize {
        self.free_nodes
    }

    /// Returns the number of scalar additions.
    pub fn scalar_adds(&self) -> usize {
        self.scalar_adds
    }

    /// Returns the number of scalar multiplications.
    pub fn scalar_muls(&self) -> usize {
        self.scalar_muls
    }

    /// Returns the number of scalar divisions.
    pub fn scalar_divs(&self) -> usize {
        self.scalar_divs
    }

    /// Returns the number of inexpensive unary operations.
    pub fn cheap_unary_ops(&self) -> usize {
        self.cheap_unary_ops
    }

    /// Returns the number of integer-power operations.
    pub fn power_ops(&self) -> usize {
        self.power_ops
    }

    /// Returns the number of transcendental operations.
    pub fn transcendental_ops(&self) -> usize {
        self.transcendental_ops
    }

    /// Returns the number of complex, vector, and matrix constructors.
    pub fn constructors(&self) -> usize {
        self.constructors
    }

    /// Returns the number of component and element extractions.
    pub fn extractions(&self) -> usize {
        self.extractions
    }

    /// Returns the number of linear-algebra operations.
    pub fn linear_algebra_ops(&self) -> usize {
        self.linear_algebra_ops
    }

    fn add_node(&mut self, node: &ExprNode) {
        self.node_count += 1;
        match node {
            ExprNode::RealConst(_)
            | ExprNode::ComplexConst(_)
            | ExprNode::ScalarParam(_)
            | ExprNode::EventScalar(_)
            | ExprNode::EventP4Component { .. } => {
                self.free_nodes += 1;
            }
            ExprNode::Unary { op, .. } => self.add_unary(*op),
            ExprNode::Binary { op, .. } => self.add_binary(*op, 2),
            ExprNode::NaryAdd { terms } => self.add_binary(BinaryOp::Add, terms.len()),
            ExprNode::NaryMul { factors } => self.add_binary(BinaryOp::Mul, factors.len()),
            ExprNode::Complex { .. } | ExprNode::Vector { .. } | ExprNode::Matrix { .. } => {
                self.constructors += 1;
                self.weighted_ops += 1;
            }
            ExprNode::Component { .. } | ExprNode::MatrixElement { .. } => {
                self.extractions += 1;
                self.weighted_ops += 1;
            }
            ExprNode::MatMul { .. }
            | ExprNode::MatVec { .. }
            | ExprNode::Dot { .. }
            | ExprNode::Solve { .. } => {
                self.linear_algebra_ops += 1;
                self.weighted_ops += 50;
            }
        }
    }

    fn add_unary(&mut self, op: UnaryOp) {
        match op {
            UnaryOp::Neg | UnaryOp::Real | UnaryOp::Imag | UnaryOp::Conj => {
                self.cheap_unary_ops += 1;
                self.weighted_ops += 1;
            }
            UnaryOp::NormSqr => {
                self.cheap_unary_ops += 1;
                self.weighted_ops += 4;
            }
            UnaryOp::PowI(power) => {
                self.power_ops += 1;
                self.weighted_ops += powi_weight(power);
            }
            UnaryOp::Sqrt => {
                self.transcendental_ops += 1;
                self.weighted_ops += 8;
            }
            UnaryOp::Exp | UnaryOp::Sin | UnaryOp::Cos | UnaryOp::Log => {
                self.transcendental_ops += 1;
                self.weighted_ops += 20;
            }
        }
    }

    fn add_binary(&mut self, op: BinaryOp, operand_count: usize) {
        let operations = operand_count.saturating_sub(1);
        match op {
            BinaryOp::Add | BinaryOp::Sub => {
                self.scalar_adds += operations;
                self.weighted_ops += operations as u64;
            }
            BinaryOp::Mul => {
                self.scalar_muls += operations;
                self.weighted_ops += 2 * operations as u64;
            }
            BinaryOp::Div => {
                self.scalar_divs += operations;
                self.weighted_ops += 6 * operations as u64;
            }
            BinaryOp::Atan2 => {
                self.transcendental_ops += operations;
                self.weighted_ops += 20 * operations as u64;
            }
        }
    }
}

fn powi_weight(power: i32) -> u64 {
    match power.unsigned_abs() {
        0 | 1 => 0,
        2 | 3 => 3,
        _ => 4,
    }
}

#[cfg(test)]
mod tests {
    use laddu_expr::{BinaryOp, Expr, UnaryOp, event_scalar, parameter};
    use num::complex::Complex64;

    use crate::{
        CanonicalCsePass, CompileOptions, CompiledModel, OptimizationPipeline, RewritePass,
    };

    use super::*;

    fn compile_cost(expr: &Expr, pipeline: OptimizationPipeline) -> OptimizationCost {
        CompiledModel::from_expr_with_options(expr, &CompileOptions::with_pipeline(pipeline))
            .unwrap()
            .cost()
    }

    #[test]
    fn unary_operation_cost_table_is_exhaustive() {
        let cases = [
            (UnaryOp::Neg, (1, 0, 0, 1)),
            (UnaryOp::Real, (1, 0, 0, 1)),
            (UnaryOp::Imag, (1, 0, 0, 1)),
            (UnaryOp::Conj, (1, 0, 0, 1)),
            (UnaryOp::NormSqr, (1, 0, 0, 4)),
            (UnaryOp::PowI(0), (0, 1, 0, 0)),
            (UnaryOp::PowI(2), (0, 1, 0, 3)),
            (UnaryOp::PowI(-4), (0, 1, 0, 4)),
            (UnaryOp::Sqrt, (0, 0, 1, 8)),
            (UnaryOp::Exp, (0, 0, 1, 20)),
            (UnaryOp::Sin, (0, 0, 1, 20)),
            (UnaryOp::Cos, (0, 0, 1, 20)),
            (UnaryOp::Log, (0, 0, 1, 20)),
        ];

        for (op, expected) in cases {
            let mut cost = OptimizationCost::default();
            cost.add_unary(op);
            assert_eq!(
                (
                    cost.cheap_unary_ops,
                    cost.power_ops,
                    cost.transcendental_ops,
                    cost.weighted_ops,
                ),
                expected,
                "unexpected cost for {op:?}"
            );
        }
    }

    #[test]
    fn binary_and_nary_operation_costs_share_one_table() {
        let cases = [
            (BinaryOp::Add, (3, 0, 0, 0, 3)),
            (BinaryOp::Sub, (3, 0, 0, 0, 3)),
            (BinaryOp::Mul, (0, 3, 0, 0, 6)),
            (BinaryOp::Div, (0, 0, 3, 0, 18)),
            (BinaryOp::Atan2, (0, 0, 0, 3, 60)),
        ];

        for (op, expected) in cases {
            let mut cost = OptimizationCost::default();
            cost.add_binary(op, 4);
            assert_eq!(
                (
                    cost.scalar_adds,
                    cost.scalar_muls,
                    cost.scalar_divs,
                    cost.transcendental_ops,
                    cost.weighted_ops,
                ),
                expected,
                "unexpected cost for {op:?}"
            );
        }
    }

    #[test]
    fn partitions_operation_work_by_execution_lifecycle() {
        let parameter_only = parameter!("scale") + 1.0;
        let event_only = event_scalar("x") + 2.0;
        let mixed = parameter_only * event_only;
        let graph = mixed.to_graph();
        let cost = OptimizationCost::analyze(&graph).lifecycle();
        assert!(cost.evaluation_invariant() > 0);
        assert!(cost.dataset_event() > 0);
        assert!(cost.evaluation_event() > 0);
    }

    #[test]
    fn optimization_cost_reports_weighted_operation_breakdown() {
        let x = Expr::from(parameter!("x"));
        let compiled = CompiledModel::from_expr_with_options(
            &(x.sin() + x.exp() * x.powi(2)),
            &CompileOptions::without_optimizations(),
        )
        .unwrap();
        let cost = compiled.cost();

        assert_eq!(cost.transcendental_ops(), 2);
        assert_eq!(cost.power_ops(), 1);
        assert_eq!(cost.scalar_adds(), 1);
        assert_eq!(cost.scalar_muls(), 1);
        assert_eq!(cost.weighted_ops(), 46);
    }

    #[test]
    fn optimization_cost_compares_pipeline_effectiveness() {
        let phi = Expr::from(parameter!("phi"));
        let euler = phi.cos() + Complex64::I * phi.sin();
        let without_exponential = compile_cost(
            &euler,
            OptimizationPipeline::new()
                .with_pass(RewritePass::simplify())
                .with_pass(CanonicalCsePass)
                .with_pass(RewritePass::normalize_add_mul())
                .with_pass(CanonicalCsePass)
                .with_max_iterations(4),
        );
        let with_exponential = compile_cost(
            &euler,
            OptimizationPipeline::new()
                .with_pass(RewritePass::simplify())
                .with_pass(CanonicalCsePass)
                .with_pass(RewritePass::normalize_add_mul())
                .with_pass(CanonicalCsePass)
                .with_pass(RewritePass::exponential())
                .with_pass(RewritePass::simplify())
                .with_max_iterations(4),
        );

        assert!(with_exponential.weighted_ops() < without_exponential.weighted_ops());
        assert_eq!(with_exponential.transcendental_ops(), 1);
    }
}
