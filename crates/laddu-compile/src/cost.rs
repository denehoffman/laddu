use laddu_expr::{BinaryOp, ExprGraph, ExprNode, UnaryOp};

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
}

impl OptimizationCost {
    /// Computes the cost metrics for `graph`.
    pub fn analyze(graph: &ExprGraph) -> Self {
        let mut cost = Self::default();
        for node in graph.nodes() {
            cost.add_node(node);
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
        self.weighted_ops < baseline.weighted_ops
            || (self.weighted_ops == baseline.weighted_ops && self.node_count < baseline.node_count)
    }

    /// Returns whether this cost is lower than or equal to `baseline`.
    pub fn is_no_worse_than(&self, baseline: &Self) -> bool {
        self.weighted_ops < baseline.weighted_ops
            || (self.weighted_ops == baseline.weighted_ops
                && self.node_count <= baseline.node_count)
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
