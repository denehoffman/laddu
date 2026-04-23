#![allow(dead_code)]

use std::collections::HashMap;

use nalgebra::DVector;
use num::complex::Complex64;

use super::ExpressionNode;

type IrValueId = usize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum DependenceClass {
    ParameterOnly,
    CacheOnly,
    Mixed,
}

impl DependenceClass {
    pub(super) fn merge(self, other: Self) -> Self {
        match (self, other) {
            (Self::Mixed, _) | (_, Self::Mixed) => Self::Mixed,
            (Self::ParameterOnly, Self::ParameterOnly) => Self::ParameterOnly,
            (Self::CacheOnly, Self::CacheOnly) => Self::CacheOnly,
            (Self::ParameterOnly, Self::CacheOnly) | (Self::CacheOnly, Self::ParameterOnly) => {
                Self::Mixed
            }
        }
    }

    fn apply_unary(self, _op: IrUnaryOp) -> Self {
        self
    }

    fn apply_binary(self, _op: IrBinaryOp, rhs: Self) -> Self {
        self.merge(rhs)
    }
}

fn unary_output_is_real(op: IrUnaryOp, input_is_real: bool) -> bool {
    match op {
        IrUnaryOp::Neg => input_is_real,
        IrUnaryOp::Real | IrUnaryOp::Imag | IrUnaryOp::NormSqr => true,
        IrUnaryOp::Conj => input_is_real,
        IrUnaryOp::Exp | IrUnaryOp::PowI(_) | IrUnaryOp::Sin | IrUnaryOp::Cos => input_is_real,
        IrUnaryOp::Sqrt | IrUnaryOp::PowF(_) | IrUnaryOp::Log | IrUnaryOp::Cis => false,
    }
}

fn binary_output_is_real(op: IrBinaryOp, left_is_real: bool, right_is_real: bool) -> bool {
    match op {
        IrBinaryOp::Add | IrBinaryOp::Sub | IrBinaryOp::Mul | IrBinaryOp::Div => {
            left_is_real && right_is_real
        }
        IrBinaryOp::Pow => false,
    }
}

fn apply_unary_op(op: IrUnaryOp, value: Complex64) -> Complex64 {
    match op {
        IrUnaryOp::Neg => -value,
        IrUnaryOp::Real => Complex64::new(value.re, 0.0),
        IrUnaryOp::Imag => Complex64::new(value.im, 0.0),
        IrUnaryOp::Conj => value.conj(),
        IrUnaryOp::NormSqr => Complex64::new(value.norm_sqr(), 0.0),
        IrUnaryOp::Sqrt => value.sqrt(),
        IrUnaryOp::PowI(power) => value.powi(power),
        IrUnaryOp::PowF(power) => value.powc(Complex64::new(f64::from_bits(power), 0.0)),
        IrUnaryOp::Exp => value.exp(),
        IrUnaryOp::Sin => value.sin(),
        IrUnaryOp::Cos => value.cos(),
        IrUnaryOp::Log => value.ln(),
        IrUnaryOp::Cis => (Complex64::new(0.0, 1.0) * value).exp(),
    }
}

fn apply_binary_op(op: IrBinaryOp, left_value: Complex64, right_value: Complex64) -> Complex64 {
    match op {
        IrBinaryOp::Add => left_value + right_value,
        IrBinaryOp::Sub => left_value - right_value,
        IrBinaryOp::Mul => left_value * right_value,
        IrBinaryOp::Div => left_value / right_value,
        IrBinaryOp::Pow => left_value.powc(right_value),
    }
}

fn apply_unary_gradient_op(
    op: IrUnaryOp,
    value: Complex64,
    input_grad: &DVector<Complex64>,
    dst_grad: &mut DVector<Complex64>,
) {
    match op {
        IrUnaryOp::Neg => {
            for (dst_item, input_item) in dst_grad.iter_mut().zip(input_grad.iter()) {
                *dst_item = -*input_item;
            }
        }
        IrUnaryOp::Real => {
            for (dst_item, input_item) in dst_grad.iter_mut().zip(input_grad.iter()) {
                *dst_item = Complex64::new(input_item.re, 0.0);
            }
        }
        IrUnaryOp::Imag => {
            for (dst_item, input_item) in dst_grad.iter_mut().zip(input_grad.iter()) {
                *dst_item = Complex64::new(input_item.im, 0.0);
            }
        }
        IrUnaryOp::Conj => {
            for (dst_item, input_item) in dst_grad.iter_mut().zip(input_grad.iter()) {
                *dst_item = input_item.conj();
            }
        }
        IrUnaryOp::NormSqr => {
            let conj_input = value.conj();
            for (dst_item, input_item) in dst_grad.iter_mut().zip(input_grad.iter()) {
                *dst_item = Complex64::new(2.0 * (*input_item * conj_input).re, 0.0);
            }
        }
        IrUnaryOp::Sqrt => {
            let factor = Complex64::new(0.5, 0.0) / value.sqrt();
            for (dst_item, input_item) in dst_grad.iter_mut().zip(input_grad.iter()) {
                *dst_item = *input_item * factor;
            }
        }
        IrUnaryOp::PowI(power) => {
            let factor = match power {
                0 => Complex64::ZERO,
                1 => Complex64::ONE,
                _ => {
                    let multiplier = Complex64::new(power as f64, 0.0);
                    if let Some(derivative_power) = power.checked_sub(1) {
                        multiplier * value.powi(derivative_power)
                    } else {
                        multiplier * value.powi(power) / value
                    }
                }
            };
            for (dst_item, input_item) in dst_grad.iter_mut().zip(input_grad.iter()) {
                *dst_item = *input_item * factor;
            }
        }
        IrUnaryOp::PowF(power) => {
            let power = f64::from_bits(power);
            let factor = if power == 0.0 {
                Complex64::ZERO
            } else {
                Complex64::new(power, 0.0) * value.powc(Complex64::new(power - 1.0, 0.0))
            };
            for (dst_item, input_item) in dst_grad.iter_mut().zip(input_grad.iter()) {
                *dst_item = *input_item * factor;
            }
        }
        IrUnaryOp::Exp => {
            let factor = value.exp();
            for (dst_item, input_item) in dst_grad.iter_mut().zip(input_grad.iter()) {
                *dst_item = *input_item * factor;
            }
        }
        IrUnaryOp::Sin => {
            let factor = value.cos();
            for (dst_item, input_item) in dst_grad.iter_mut().zip(input_grad.iter()) {
                *dst_item = *input_item * factor;
            }
        }
        IrUnaryOp::Cis => {
            let factor = Complex64::new(0.0, 1.0) * apply_unary_op(IrUnaryOp::Cis, value);
            for (dst_item, input_item) in dst_grad.iter_mut().zip(input_grad.iter()) {
                *dst_item = *input_item * factor;
            }
        }
        IrUnaryOp::Cos => {
            let factor = -value.sin();
            for (dst_item, input_item) in dst_grad.iter_mut().zip(input_grad.iter()) {
                *dst_item = *input_item * factor;
            }
        }
        IrUnaryOp::Log => {
            let factor = Complex64::ONE / value;
            for (dst_item, input_item) in dst_grad.iter_mut().zip(input_grad.iter()) {
                *dst_item = *input_item * factor;
            }
        }
    }
}

fn apply_binary_gradient_op(
    op: IrBinaryOp,
    left_value: Complex64,
    right_value: Complex64,
    output_value: Complex64,
    left_grad: &DVector<Complex64>,
    right_grad: &DVector<Complex64>,
    dst_grad: &mut DVector<Complex64>,
) {
    match op {
        IrBinaryOp::Add => {
            for ((dst_item, left_item), right_item) in dst_grad
                .iter_mut()
                .zip(left_grad.iter())
                .zip(right_grad.iter())
            {
                *dst_item = *left_item + *right_item;
            }
        }
        IrBinaryOp::Sub => {
            for ((dst_item, left_item), right_item) in dst_grad
                .iter_mut()
                .zip(left_grad.iter())
                .zip(right_grad.iter())
            {
                *dst_item = *left_item - *right_item;
            }
        }
        IrBinaryOp::Mul => {
            for ((dst_item, left_item), right_item) in dst_grad
                .iter_mut()
                .zip(left_grad.iter())
                .zip(right_grad.iter())
            {
                *dst_item = *left_item * right_value + *right_item * left_value;
            }
        }
        IrBinaryOp::Div => {
            let denom = right_value * right_value;
            for ((dst_item, left_item), right_item) in dst_grad
                .iter_mut()
                .zip(left_grad.iter())
                .zip(right_grad.iter())
            {
                *dst_item = (*left_item * right_value - *right_item * left_value) / denom;
            }
        }
        IrBinaryOp::Pow => {
            for ((dst_item, left_item), right_item) in dst_grad
                .iter_mut()
                .zip(left_grad.iter())
                .zip(right_grad.iter())
            {
                *dst_item = output_value
                    * (*right_item * left_value.ln() + right_value * *left_item / left_value);
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum IrUnaryOp {
    Neg,
    Real,
    Imag,
    Conj,
    NormSqr,
    Sqrt,
    PowI(i32),
    PowF(u64),
    Exp,
    Sin,
    Cos,
    Log,
    Cis,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum IrBinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum IrNode {
    Constant(Complex64),
    Amp(usize),
    Unary {
        op: IrUnaryOp,
        input: IrValueId,
    },
    Binary {
        op: IrBinaryOp,
        left: IrValueId,
        right: IrValueId,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct ExpressionIR {
    nodes: Vec<IrNode>,
    root: IrValueId,
    dependence_annotations: Vec<DependenceClass>,
    dependence_warnings: Vec<String>,
    rewrite_diagnostics: Vec<String>,
    separable_mul_candidates: Vec<SeparableMulCandidate>,
    normalization_plan: NormalizationPlan,
    normalization_execution_sets: NormalizationExecutionSets,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SeparableMulCandidate {
    pub node_index: usize,
    pub left_node_index: usize,
    pub right_node_index: usize,
    pub left_dependence: DependenceClass,
    pub right_dependence: DependenceClass,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct CachedIntegralDescriptor {
    pub mul_node_index: usize,
    pub parameter_node_index: usize,
    pub cache_node_index: usize,
    pub coefficient: i32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct NormalizationPlan {
    pub cached_integral_descriptors: Vec<CachedIntegralDescriptor>,
    pub cached_separable_nodes: Vec<usize>,
    pub residual_terms: Vec<usize>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct NormalizationExecutionSets {
    pub cached_parameter_amplitudes: Vec<usize>,
    pub cached_cache_amplitudes: Vec<usize>,
    pub residual_amplitudes: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct NormalizationPlanExplain {
    pub root_dependence: DependenceClass,
    pub warnings: Vec<String>,
    pub separable_mul_candidates: Vec<SeparableMulCandidate>,
    pub cached_integral_descriptors: Vec<CachedIntegralDescriptor>,
    pub cached_separable_nodes: Vec<usize>,
    pub residual_terms: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum IrNodeKey {
    Constant {
        re_bits: u64,
        im_bits: u64,
    },
    Amp(usize),
    Unary {
        op: IrUnaryOp,
        input: IrValueId,
    },
    Binary {
        op: IrBinaryOp,
        left: IrValueId,
        right: IrValueId,
    },
}

fn key_for_ir_node(node: &IrNode) -> IrNodeKey {
    match *node {
        IrNode::Constant(value) => IrNodeKey::Constant {
            re_bits: value.re.to_bits(),
            im_bits: value.im.to_bits(),
        },
        IrNode::Amp(idx) => IrNodeKey::Amp(idx),
        IrNode::Unary { op, input } => IrNodeKey::Unary { op, input },
        IrNode::Binary { op, left, right } => IrNodeKey::Binary { op, left, right },
    }
}

fn intern_ir_node(
    node: IrNode,
    nodes: &mut Vec<IrNode>,
    interned: &mut HashMap<IrNodeKey, IrValueId>,
) -> IrValueId {
    let key = key_for_ir_node(&node);
    if let Some(&existing) = interned.get(&key) {
        return existing;
    }
    let id = nodes.len();
    nodes.push(node);
    interned.insert(key, id);
    id
}

#[derive(Clone, Copy, Debug)]
struct RewriteLimits {
    max_iterations: usize,
    max_expansions: usize,
    max_nodes_multiplier: usize,
    max_nodes_additive: usize,
}

impl Default for RewriteLimits {
    fn default() -> Self {
        Self {
            max_iterations: 4,
            max_expansions: 128,
            max_nodes_multiplier: 4,
            max_nodes_additive: 32,
        }
    }
}

impl RewriteLimits {
    fn node_cap(self, baseline_nodes: usize) -> usize {
        baseline_nodes
            .saturating_mul(self.max_nodes_multiplier)
            .max(baseline_nodes.saturating_add(self.max_nodes_additive))
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct IrPassReport {
    changed: bool,
    rewrites: usize,
    saturated: bool,
}

struct IrPassContext<'a> {
    amplitude_dependencies: &'a [DependenceClass],
    amplitude_realness: &'a [bool],
    rewrite_limits: RewriteLimits,
    diagnostics: Vec<String>,
}

trait IrPass {
    fn name(&self) -> &'static str;
    fn run(&self, ir: &mut ExpressionIR, ctx: &mut IrPassContext<'_>) -> IrPassReport;
}

impl ExpressionIR {
    fn from_expression_node(node: &ExpressionNode) -> Self {
        fn lower(node: &ExpressionNode, nodes: &mut Vec<IrNode>) -> IrValueId {
            match node {
                ExpressionNode::Zero => {
                    let id = nodes.len();
                    nodes.push(IrNode::Constant(Complex64::ZERO));
                    id
                }
                ExpressionNode::One => {
                    let id = nodes.len();
                    nodes.push(IrNode::Constant(Complex64::ONE));
                    id
                }
                ExpressionNode::Constant(value) => {
                    let id = nodes.len();
                    nodes.push(IrNode::Constant(Complex64::from(value)));
                    id
                }
                ExpressionNode::ComplexConstant(value) => {
                    let id = nodes.len();
                    nodes.push(IrNode::Constant(*value));
                    id
                }
                ExpressionNode::Amp(idx) => {
                    let id = nodes.len();
                    nodes.push(IrNode::Amp(*idx));
                    id
                }
                ExpressionNode::Add(a, b) => {
                    let left = lower(a, nodes);
                    let right = lower(b, nodes);
                    let id = nodes.len();
                    nodes.push(IrNode::Binary {
                        op: IrBinaryOp::Add,
                        left,
                        right,
                    });
                    id
                }
                ExpressionNode::Sub(a, b) => {
                    let left = lower(a, nodes);
                    let right = lower(b, nodes);
                    let id = nodes.len();
                    nodes.push(IrNode::Binary {
                        op: IrBinaryOp::Sub,
                        left,
                        right,
                    });
                    id
                }
                ExpressionNode::Mul(a, b) => {
                    let left = lower(a, nodes);
                    let right = lower(b, nodes);
                    let id = nodes.len();
                    nodes.push(IrNode::Binary {
                        op: IrBinaryOp::Mul,
                        left,
                        right,
                    });
                    id
                }
                ExpressionNode::Div(a, b) => {
                    let left = lower(a, nodes);
                    let right = lower(b, nodes);
                    let id = nodes.len();
                    nodes.push(IrNode::Binary {
                        op: IrBinaryOp::Div,
                        left,
                        right,
                    });
                    id
                }
                ExpressionNode::Neg(a) => {
                    let input = lower(a, nodes);
                    let id = nodes.len();
                    nodes.push(IrNode::Unary {
                        op: IrUnaryOp::Neg,
                        input,
                    });
                    id
                }
                ExpressionNode::Real(a) => {
                    let input = lower(a, nodes);
                    let id = nodes.len();
                    nodes.push(IrNode::Unary {
                        op: IrUnaryOp::Real,
                        input,
                    });
                    id
                }
                ExpressionNode::Imag(a) => {
                    let input = lower(a, nodes);
                    let id = nodes.len();
                    nodes.push(IrNode::Unary {
                        op: IrUnaryOp::Imag,
                        input,
                    });
                    id
                }
                ExpressionNode::Conj(a) => {
                    let input = lower(a, nodes);
                    let id = nodes.len();
                    nodes.push(IrNode::Unary {
                        op: IrUnaryOp::Conj,
                        input,
                    });
                    id
                }
                ExpressionNode::NormSqr(a) => {
                    let input = lower(a, nodes);
                    let id = nodes.len();
                    nodes.push(IrNode::Unary {
                        op: IrUnaryOp::NormSqr,
                        input,
                    });
                    id
                }
                ExpressionNode::Sqrt(a) => {
                    let input = lower(a, nodes);
                    let id = nodes.len();
                    nodes.push(IrNode::Unary {
                        op: IrUnaryOp::Sqrt,
                        input,
                    });
                    id
                }
                ExpressionNode::Pow(a, b) => {
                    let value = lower(a, nodes);
                    let power = lower(b, nodes);
                    let id = nodes.len();
                    nodes.push(IrNode::Binary {
                        op: IrBinaryOp::Pow,
                        left: value,
                        right: power,
                    });
                    id
                }
                ExpressionNode::PowI(a, power) => {
                    let input = lower(a, nodes);
                    let id = nodes.len();
                    nodes.push(IrNode::Unary {
                        op: IrUnaryOp::PowI(*power),
                        input,
                    });
                    id
                }
                ExpressionNode::PowF(a, power) => {
                    let input = lower(a, nodes);
                    let id = nodes.len();
                    nodes.push(IrNode::Unary {
                        op: IrUnaryOp::PowF(power.to_bits()),
                        input,
                    });
                    id
                }
                ExpressionNode::Exp(a) => {
                    let input = lower(a, nodes);
                    let id = nodes.len();
                    nodes.push(IrNode::Unary {
                        op: IrUnaryOp::Exp,
                        input,
                    });
                    id
                }
                ExpressionNode::Sin(a) => {
                    let input = lower(a, nodes);
                    let id = nodes.len();
                    nodes.push(IrNode::Unary {
                        op: IrUnaryOp::Sin,
                        input,
                    });
                    id
                }
                ExpressionNode::Cos(a) => {
                    let input = lower(a, nodes);
                    let id = nodes.len();
                    nodes.push(IrNode::Unary {
                        op: IrUnaryOp::Cos,
                        input,
                    });
                    id
                }
                ExpressionNode::Log(a) => {
                    let input = lower(a, nodes);
                    let id = nodes.len();
                    nodes.push(IrNode::Unary {
                        op: IrUnaryOp::Log,
                        input,
                    });
                    id
                }
                ExpressionNode::Cis(a) => {
                    let input = lower(a, nodes);
                    let id = nodes.len();
                    nodes.push(IrNode::Unary {
                        op: IrUnaryOp::Cis,
                        input,
                    });
                    id
                }
            }
        }

        let mut nodes = Vec::new();
        let root = lower(node, &mut nodes);
        Self {
            nodes,
            root,
            dependence_annotations: Vec::new(),
            dependence_warnings: Vec::new(),
            rewrite_diagnostics: Vec::new(),
            separable_mul_candidates: Vec::new(),
            normalization_plan: NormalizationPlan::default(),
            normalization_execution_sets: NormalizationExecutionSets::default(),
        }
    }

    pub(super) fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub(super) fn nodes(&self) -> &[IrNode] {
        &self.nodes
    }

    pub(super) fn root(&self) -> IrValueId {
        self.root
    }

    #[cfg(test)]
    pub(super) fn from_test_nodes(nodes: Vec<IrNode>, root: IrValueId) -> Self {
        Self {
            nodes,
            root,
            dependence_annotations: Vec::new(),
            dependence_warnings: Vec::new(),
            rewrite_diagnostics: Vec::new(),
            separable_mul_candidates: Vec::new(),
            normalization_plan: NormalizationPlan::default(),
            normalization_execution_sets: NormalizationExecutionSets::default(),
        }
    }

    pub(super) fn root_dependence(&self) -> DependenceClass {
        if self.nodes.is_empty() {
            return DependenceClass::ParameterOnly;
        }
        if self.dependence_annotations.len() == self.nodes.len() {
            return self.dependence_annotations[self.root];
        }
        DependenceAnnotatePass::compute_annotations(&self.nodes, &[])[self.root]
    }

    pub(super) fn node_dependence_annotations(&self) -> &[DependenceClass] {
        &self.dependence_annotations
    }

    pub(super) fn dependence_warnings(&self) -> &[String] {
        &self.dependence_warnings
    }

    pub(super) fn rewrite_diagnostics(&self) -> &[String] {
        &self.rewrite_diagnostics
    }

    pub(super) fn separable_mul_candidates(&self) -> &[SeparableMulCandidate] {
        &self.separable_mul_candidates
    }

    pub(super) fn normalization_plan(&self) -> &NormalizationPlan {
        &self.normalization_plan
    }

    pub(super) fn normalization_execution_sets(&self) -> &NormalizationExecutionSets {
        &self.normalization_execution_sets
    }

    pub(super) fn cached_integral_descriptors(&self) -> &[CachedIntegralDescriptor] {
        &self.normalization_plan.cached_integral_descriptors
    }

    pub(super) fn normalization_plan_explain(&self) -> NormalizationPlanExplain {
        NormalizationPlanExplain {
            root_dependence: self.root_dependence(),
            warnings: self.dependence_warnings.clone(),
            separable_mul_candidates: self.separable_mul_candidates.clone(),
            cached_integral_descriptors: self
                .normalization_plan
                .cached_integral_descriptors
                .clone(),
            cached_separable_nodes: self.normalization_plan.cached_separable_nodes.clone(),
            residual_terms: self.normalization_plan.residual_terms.clone(),
        }
    }

    fn fill_values(&self, amplitude_values: &[Complex64], slots: &mut [Complex64]) {
        debug_assert!(slots.len() >= self.nodes.len());
        for (node_index, node) in self.nodes.iter().enumerate() {
            slots[node_index] = match *node {
                IrNode::Constant(value) => value,
                IrNode::Amp(amp_idx) => amplitude_values.get(amp_idx).copied().unwrap_or_default(),
                IrNode::Unary { op, input } => {
                    let value = slots[input];
                    apply_unary_op(op, value)
                }
                IrNode::Binary { op, left, right } => {
                    let left_value = slots[left];
                    let right_value = slots[right];
                    apply_binary_op(op, left_value, right_value)
                }
            };
        }
    }

    fn fill_values_with_zeroed_nodes(
        &self,
        amplitude_values: &[Complex64],
        slots: &mut [Complex64],
        zeroed_nodes: &[bool],
    ) {
        debug_assert!(slots.len() >= self.nodes.len());
        for (node_index, node) in self.nodes.iter().enumerate() {
            if zeroed_nodes.get(node_index).copied().unwrap_or(false) {
                slots[node_index] = Complex64::ZERO;
                continue;
            }
            slots[node_index] = match *node {
                IrNode::Constant(value) => value,
                IrNode::Amp(amp_idx) => amplitude_values.get(amp_idx).copied().unwrap_or_default(),
                IrNode::Unary { op, input } => {
                    let value = slots[input];
                    apply_unary_op(op, value)
                }
                IrNode::Binary { op, left, right } => {
                    let left_value = slots[left];
                    let right_value = slots[right];
                    apply_binary_op(op, left_value, right_value)
                }
            };
        }
    }

    pub(super) fn evaluate_into(
        &self,
        amplitude_values: &[Complex64],
        value_slots: &mut [Complex64],
    ) -> Complex64 {
        if self.nodes.is_empty() {
            return Complex64::ZERO;
        }
        self.fill_values(amplitude_values, value_slots);
        value_slots[self.root]
    }

    pub(super) fn evaluate(&self, amplitude_values: &[Complex64]) -> Complex64 {
        let mut value_slots = vec![Complex64::ZERO; self.nodes.len()];
        self.evaluate_into(amplitude_values, &mut value_slots)
    }

    pub(super) fn evaluate_into_with_zeroed_nodes(
        &self,
        amplitude_values: &[Complex64],
        value_slots: &mut [Complex64],
        zeroed_nodes: &[bool],
    ) -> Complex64 {
        if self.nodes.is_empty() {
            return Complex64::ZERO;
        }
        self.fill_values_with_zeroed_nodes(amplitude_values, value_slots, zeroed_nodes);
        value_slots[self.root]
    }

    pub(super) fn evaluate_gradient_into(
        &self,
        amplitude_values: &[Complex64],
        amplitude_gradients: &[DVector<Complex64>],
        value_slots: &mut [Complex64],
        gradient_slots: &mut [DVector<Complex64>],
    ) -> DVector<Complex64> {
        if self.nodes.is_empty() {
            let grad_dim = amplitude_gradients.first().map(|g| g.len()).unwrap_or(0);
            return DVector::zeros(grad_dim);
        }
        self.fill_values(amplitude_values, value_slots);
        for (node_index, node) in self.nodes.iter().enumerate() {
            let (before, tail) = gradient_slots.split_at_mut(node_index);
            let (dst_grad, _) = tail
                .split_first_mut()
                .expect("destination gradient slot should exist");
            match *node {
                IrNode::Constant(_) => {
                    for value in dst_grad.iter_mut() {
                        *value = Complex64::ZERO;
                    }
                }
                IrNode::Amp(amp_idx) => {
                    if let Some(source) = amplitude_gradients.get(amp_idx) {
                        dst_grad.clone_from(source);
                    } else {
                        for value in dst_grad.iter_mut() {
                            *value = Complex64::ZERO;
                        }
                    }
                }
                IrNode::Unary { op, input } => {
                    let input_grad = &before[input];
                    apply_unary_gradient_op(op, value_slots[input], input_grad, dst_grad);
                }
                IrNode::Binary { op, left, right } => {
                    let left_grad = &before[left];
                    let right_grad = &before[right];
                    apply_binary_gradient_op(
                        op,
                        value_slots[left],
                        value_slots[right],
                        value_slots[node_index],
                        left_grad,
                        right_grad,
                        dst_grad,
                    );
                }
            }
        }
        gradient_slots[self.root].clone()
    }

    pub(super) fn evaluate_value_gradient_into(
        &self,
        amplitude_values: &[Complex64],
        amplitude_gradients: &[DVector<Complex64>],
        value_slots: &mut [Complex64],
        gradient_slots: &mut [DVector<Complex64>],
    ) -> (Complex64, DVector<Complex64>) {
        let gradient = self.evaluate_gradient_into(
            amplitude_values,
            amplitude_gradients,
            value_slots,
            gradient_slots,
        );
        (value_slots[self.root], gradient)
    }

    pub(super) fn evaluate_gradient(
        &self,
        amplitude_values: &[Complex64],
        amplitude_gradients: &[DVector<Complex64>],
    ) -> DVector<Complex64> {
        let grad_dim = amplitude_gradients.first().map(|g| g.len()).unwrap_or(0);
        let mut value_slots = vec![Complex64::ZERO; self.nodes.len()];
        let mut gradient_slots: Vec<DVector<Complex64>> = (0..self.nodes.len())
            .map(|_| DVector::zeros(grad_dim))
            .collect();
        self.evaluate_gradient_into(
            amplitude_values,
            amplitude_gradients,
            &mut value_slots,
            &mut gradient_slots,
        )
    }

    pub(super) fn evaluate_gradient_into_with_zeroed_nodes(
        &self,
        amplitude_values: &[Complex64],
        amplitude_gradients: &[DVector<Complex64>],
        value_slots: &mut [Complex64],
        gradient_slots: &mut [DVector<Complex64>],
        zeroed_nodes: &[bool],
    ) -> DVector<Complex64> {
        if self.nodes.is_empty() {
            let grad_dim = amplitude_gradients.first().map(|g| g.len()).unwrap_or(0);
            return DVector::zeros(grad_dim);
        }
        self.fill_values_with_zeroed_nodes(amplitude_values, value_slots, zeroed_nodes);
        for (node_index, node) in self.nodes.iter().enumerate() {
            let (before, tail) = gradient_slots.split_at_mut(node_index);
            let (dst_grad, _) = tail
                .split_first_mut()
                .expect("destination gradient slot should exist");
            if zeroed_nodes.get(node_index).copied().unwrap_or(false) {
                dst_grad.fill(Complex64::ZERO);
                continue;
            }
            match *node {
                IrNode::Constant(_) => {
                    dst_grad.fill(Complex64::ZERO);
                }
                IrNode::Amp(amp_idx) => {
                    if let Some(source) = amplitude_gradients.get(amp_idx) {
                        dst_grad.clone_from(source);
                    } else {
                        dst_grad.fill(Complex64::ZERO);
                    }
                }
                IrNode::Unary { op, input } => {
                    let input_grad = &before[input];
                    apply_unary_gradient_op(op, value_slots[input], input_grad, dst_grad);
                }
                IrNode::Binary { op, left, right } => {
                    let left_grad = &before[left];
                    let right_grad = &before[right];
                    apply_binary_gradient_op(
                        op,
                        value_slots[left],
                        value_slots[right],
                        value_slots[node_index],
                        left_grad,
                        right_grad,
                        dst_grad,
                    );
                }
            }
        }
        gradient_slots[self.root].clone()
    }
}

pub(super) fn compile_expression_ir(
    tree: &ExpressionNode,
    active_amplitudes: &[bool],
    amplitude_dependencies: &[DependenceClass],
) -> ExpressionIR {
    let amplitude_realness = vec![false; amplitude_dependencies.len()];
    compile_expression_ir_with_real_hints(
        tree,
        active_amplitudes,
        amplitude_dependencies,
        &amplitude_realness,
    )
}

pub(super) fn compile_expression_ir_with_real_hints(
    tree: &ExpressionNode,
    active_amplitudes: &[bool],
    amplitude_dependencies: &[DependenceClass],
    amplitude_realness: &[bool],
) -> ExpressionIR {
    let mut ir = ExpressionIR::from_expression_node(tree);
    ir.rewrite_diagnostics = ExpressionIrPipeline::new()
        .with_rewrite_limits(RewriteLimits::default())
        .with_amplitude_realness(amplitude_realness.to_vec())
        .cse()
        .activation_specialize(active_amplitudes.to_vec())
        .constant_fold()
        .rewrite_fixed_point(amplitude_dependencies.to_vec())
        .dependence_annotate(amplitude_dependencies.to_vec())
        .run(&mut ir);
    let expression_dependence = ir.root_dependence();
    ir.dependence_warnings = collect_dependence_warnings(
        active_amplitudes,
        amplitude_dependencies,
        expression_dependence,
    );
    ir.separable_mul_candidates =
        collect_separable_mul_candidates(&ir.nodes, &ir.dependence_annotations);
    ir.normalization_plan =
        build_normalization_plan(&ir.nodes, ir.root, &ir.separable_mul_candidates);
    ir.normalization_execution_sets =
        build_normalization_execution_sets(&ir.nodes, ir.root, &ir.normalization_plan);
    ir
}

fn collect_dependence_warnings(
    active_amplitudes: &[bool],
    amplitude_dependencies: &[DependenceClass],
    root_dependence: DependenceClass,
) -> Vec<String> {
    let mut warnings = Vec::new();
    let has_active_parameter_only = active_amplitudes.iter().enumerate().any(|(index, active)| {
        *active
            && matches!(
                amplitude_dependencies.get(index),
                Some(DependenceClass::ParameterOnly)
            )
    });
    let has_active_cache_only = active_amplitudes.iter().enumerate().any(|(index, active)| {
        *active
            && matches!(
                amplitude_dependencies.get(index),
                Some(DependenceClass::CacheOnly)
            )
    });
    let has_unknown_active_hint = active_amplitudes
        .iter()
        .enumerate()
        .any(|(index, active)| *active && amplitude_dependencies.get(index).is_none());

    if root_dependence == DependenceClass::ParameterOnly && has_active_cache_only {
        warnings.push(
            "root dependence is ParameterOnly while active CacheOnly amplitude hints exist"
                .to_string(),
        );
    }
    if root_dependence == DependenceClass::CacheOnly && has_active_parameter_only {
        warnings.push(
            "root dependence is CacheOnly while active ParameterOnly amplitude hints exist"
                .to_string(),
        );
    }
    if has_active_parameter_only && has_active_cache_only {
        warnings.push(
            "both ParameterOnly and CacheOnly amplitude hints are active; expression dependence may be Mixed"
                .to_string(),
        );
    }
    if has_unknown_active_hint {
        warnings.push(
            "one or more active amplitudes are missing dependence hints; defaulting to Mixed"
                .to_string(),
        );
    }
    warnings
}

fn collect_separable_mul_candidates(
    nodes: &[IrNode],
    dependencies: &[DependenceClass],
) -> Vec<SeparableMulCandidate> {
    let mut candidates = Vec::new();
    for (node_index, node) in nodes.iter().enumerate() {
        let IrNode::Binary {
            op: IrBinaryOp::Mul,
            left,
            right,
        } = *node
        else {
            continue;
        };
        let left_dependence = dependencies
            .get(left)
            .copied()
            .unwrap_or(DependenceClass::Mixed);
        let right_dependence = dependencies
            .get(right)
            .copied()
            .unwrap_or(DependenceClass::Mixed);
        let is_separable = matches!(
            (left_dependence, right_dependence),
            (DependenceClass::ParameterOnly, DependenceClass::CacheOnly)
                | (DependenceClass::CacheOnly, DependenceClass::ParameterOnly)
        );
        if is_separable {
            candidates.push(SeparableMulCandidate {
                node_index,
                left_node_index: left,
                right_node_index: right,
                left_dependence,
                right_dependence,
            });
        }
    }
    candidates
}

fn build_normalization_plan(
    nodes: &[IrNode],
    root: usize,
    separable_mul_candidates: &[SeparableMulCandidate],
) -> NormalizationPlan {
    #[derive(Clone, Copy)]
    enum ParentPosition {
        Left,
        Right,
        Unary,
    }

    fn build_parent_edges(nodes: &[IrNode]) -> Vec<Vec<(usize, ParentPosition)>> {
        let mut parents = vec![Vec::new(); nodes.len()];
        for (parent_index, node) in nodes.iter().enumerate() {
            match *node {
                IrNode::Unary { input, .. } => {
                    parents[input].push((parent_index, ParentPosition::Unary));
                }
                IrNode::Binary { left, right, .. } => {
                    parents[left].push((parent_index, ParentPosition::Left));
                    parents[right].push((parent_index, ParentPosition::Right));
                }
                IrNode::Constant(_) | IrNode::Amp(_) => {}
            }
        }
        parents
    }

    fn descriptor_extractable_coefficient(
        mul_node_index: usize,
        root: usize,
        nodes: &[IrNode],
        parents: &[Vec<(usize, ParentPosition)>],
    ) -> Option<i32> {
        fn coefficient_to_root(
            node_index: usize,
            root: usize,
            nodes: &[IrNode],
            parents: &[Vec<(usize, ParentPosition)>],
            memo: &mut [Option<Option<i32>>],
        ) -> Option<i32> {
            if node_index == root {
                return Some(1);
            }
            if let Some(cached) = memo[node_index] {
                return cached;
            }
            let node_parents = parents.get(node_index)?;
            if node_parents.is_empty() {
                memo[node_index] = Some(None);
                return None;
            }
            let mut coefficient_sum = 0_i32;
            for (parent_index, position) in node_parents {
                let parent_coefficient =
                    coefficient_to_root(*parent_index, root, nodes, parents, memo)?;
                let edge_sign = match nodes[*parent_index] {
                    IrNode::Binary {
                        op: IrBinaryOp::Add,
                        ..
                    } => 1,
                    IrNode::Binary {
                        op: IrBinaryOp::Sub,
                        ..
                    } => match position {
                        ParentPosition::Left => 1,
                        ParentPosition::Right => -1,
                        ParentPosition::Unary => return None,
                    },
                    IrNode::Unary {
                        op: IrUnaryOp::Neg, ..
                    } => -1,
                    _ => return None,
                };
                coefficient_sum += edge_sign * parent_coefficient;
            }
            let result = Some(coefficient_sum);
            memo[node_index] = Some(result);
            result
        }

        if mul_node_index >= nodes.len() {
            return None;
        }
        let mut memo = vec![None; nodes.len()];
        coefficient_to_root(mul_node_index, root, nodes, parents, &mut memo)
    }

    fn collect_contributing(nodes: &[IrNode], root: usize) -> Vec<usize> {
        let mut seen = vec![false; nodes.len()];
        let mut stack = vec![root];
        while let Some(node_index) = stack.pop() {
            if seen.get(node_index).copied().unwrap_or(true) {
                continue;
            }
            seen[node_index] = true;
            match nodes[node_index] {
                IrNode::Unary { input, .. } => stack.push(input),
                IrNode::Binary { left, right, .. } => {
                    stack.push(left);
                    stack.push(right);
                }
                IrNode::Constant(_) | IrNode::Amp(_) => {}
            }
        }
        seen.iter()
            .enumerate()
            .filter_map(|(index, used)| used.then_some(index))
            .collect()
    }

    let contributing = collect_contributing(nodes, root);
    let parent_edges = build_parent_edges(nodes);
    let mut cached_integral_descriptors = separable_mul_candidates
        .iter()
        .filter(|candidate| contributing.contains(&candidate.node_index))
        .filter_map(
            |candidate| match (candidate.left_dependence, candidate.right_dependence) {
                (DependenceClass::ParameterOnly, DependenceClass::CacheOnly) => {
                    let coefficient = descriptor_extractable_coefficient(
                        candidate.node_index,
                        root,
                        nodes,
                        &parent_edges,
                    )?;
                    if coefficient == 0 {
                        return None;
                    }
                    Some(CachedIntegralDescriptor {
                        mul_node_index: candidate.node_index,
                        parameter_node_index: candidate.left_node_index,
                        cache_node_index: candidate.right_node_index,
                        coefficient,
                    })
                }
                (DependenceClass::CacheOnly, DependenceClass::ParameterOnly) => {
                    let coefficient = descriptor_extractable_coefficient(
                        candidate.node_index,
                        root,
                        nodes,
                        &parent_edges,
                    )?;
                    if coefficient == 0 {
                        return None;
                    }
                    Some(CachedIntegralDescriptor {
                        mul_node_index: candidate.node_index,
                        parameter_node_index: candidate.right_node_index,
                        cache_node_index: candidate.left_node_index,
                        coefficient,
                    })
                }
                _ => None,
            },
        )
        .collect::<Vec<_>>();
    cached_integral_descriptors.sort_unstable_by_key(|descriptor| descriptor.mul_node_index);
    cached_integral_descriptors.dedup_by_key(|descriptor| descriptor.mul_node_index);
    let mut cached_separable_nodes = cached_integral_descriptors
        .iter()
        .map(|descriptor| descriptor.mul_node_index)
        .collect::<Vec<_>>();
    cached_separable_nodes.sort_unstable();
    cached_separable_nodes.dedup();

    let mut residual_terms = contributing
        .into_iter()
        .filter(|index| cached_separable_nodes.binary_search(index).is_err())
        .collect::<Vec<_>>();
    residual_terms.sort_unstable();
    NormalizationPlan {
        cached_integral_descriptors,
        cached_separable_nodes,
        residual_terms,
    }
}

fn collect_reachable_amplitudes(
    nodes: &[IrNode],
    roots: &[usize],
    cut_nodes: Option<&[bool]>,
) -> Vec<usize> {
    if nodes.is_empty() || roots.is_empty() {
        return Vec::new();
    }
    let mut visited = vec![false; nodes.len()];
    let mut stack = roots.to_vec();
    let mut amplitudes = Vec::new();
    while let Some(node_index) = stack.pop() {
        if node_index >= nodes.len() || visited[node_index] {
            continue;
        }
        visited[node_index] = true;
        if cut_nodes
            .and_then(|nodes| nodes.get(node_index))
            .copied()
            .unwrap_or(false)
        {
            continue;
        }
        match nodes[node_index] {
            IrNode::Amp(amplitude_index) => amplitudes.push(amplitude_index),
            IrNode::Unary { input, .. } => stack.push(input),
            IrNode::Binary { left, right, .. } => {
                stack.push(left);
                stack.push(right);
            }
            IrNode::Constant(_) => {}
        }
    }
    amplitudes.sort_unstable();
    amplitudes.dedup();
    amplitudes
}

fn build_normalization_execution_sets(
    nodes: &[IrNode],
    root: usize,
    normalization_plan: &NormalizationPlan,
) -> NormalizationExecutionSets {
    let parameter_roots = normalization_plan
        .cached_integral_descriptors
        .iter()
        .map(|descriptor| descriptor.parameter_node_index)
        .collect::<Vec<_>>();
    let cache_roots = normalization_plan
        .cached_integral_descriptors
        .iter()
        .map(|descriptor| descriptor.cache_node_index)
        .collect::<Vec<_>>();
    let mut zeroed_nodes = vec![false; nodes.len()];
    for &node_index in &normalization_plan.cached_separable_nodes {
        if let Some(node) = zeroed_nodes.get_mut(node_index) {
            *node = true;
        }
    }
    NormalizationExecutionSets {
        cached_parameter_amplitudes: collect_reachable_amplitudes(nodes, &parameter_roots, None),
        cached_cache_amplitudes: collect_reachable_amplitudes(nodes, &cache_roots, None),
        residual_amplitudes: collect_reachable_amplitudes(
            nodes,
            std::slice::from_ref(&root),
            Some(&zeroed_nodes),
        ),
    }
}

struct AlgebraicNormalizePass;

impl AlgebraicNormalizePass {
    fn apply(&self, ir: &mut ExpressionIR) -> bool {
        fn rewrite_node(
            node_index: usize,
            conj_context: bool,
            old_nodes: &[IrNode],
            new_nodes: &mut Vec<IrNode>,
            interned: &mut HashMap<IrNodeKey, IrValueId>,
            memo: &mut HashMap<(usize, bool), IrValueId>,
        ) -> IrValueId {
            fn rewrite_normsqr(
                input: usize,
                old_nodes: &[IrNode],
                new_nodes: &mut Vec<IrNode>,
                interned: &mut HashMap<IrNodeKey, IrValueId>,
                memo: &mut HashMap<(usize, bool), IrValueId>,
            ) -> IrValueId {
                let left = rewrite_node(input, false, old_nodes, new_nodes, interned, memo);
                let right = rewrite_node(input, true, old_nodes, new_nodes, interned, memo);
                intern_ir_node(
                    IrNode::Binary {
                        op: IrBinaryOp::Mul,
                        left,
                        right,
                    },
                    new_nodes,
                    interned,
                )
            }

            fn rewrite_real_with_conj_simplify(
                input: usize,
                old_nodes: &[IrNode],
                new_nodes: &mut Vec<IrNode>,
                interned: &mut HashMap<IrNodeKey, IrValueId>,
                memo: &mut HashMap<(usize, bool), IrValueId>,
            ) -> IrValueId {
                let normalized_input = if let IrNode::Unary {
                    op: IrUnaryOp::Conj,
                    input: inner,
                } = old_nodes[input]
                {
                    rewrite_node(inner, false, old_nodes, new_nodes, interned, memo)
                } else {
                    rewrite_node(input, false, old_nodes, new_nodes, interned, memo)
                };
                intern_ir_node(
                    IrNode::Unary {
                        op: IrUnaryOp::Real,
                        input: normalized_input,
                    },
                    new_nodes,
                    interned,
                )
            }

            if let Some(&cached) = memo.get(&(node_index, conj_context)) {
                return cached;
            }

            let rewritten = match old_nodes[node_index] {
                IrNode::Constant(value) => {
                    let folded = if conj_context { value.conj() } else { value };
                    intern_ir_node(IrNode::Constant(folded), new_nodes, interned)
                }
                IrNode::Amp(amp_idx) => {
                    if conj_context {
                        let base = intern_ir_node(IrNode::Amp(amp_idx), new_nodes, interned);
                        intern_ir_node(
                            IrNode::Unary {
                                op: IrUnaryOp::Conj,
                                input: base,
                            },
                            new_nodes,
                            interned,
                        )
                    } else {
                        intern_ir_node(IrNode::Amp(amp_idx), new_nodes, interned)
                    }
                }
                IrNode::Unary { op, input } => match (conj_context, op) {
                    (false, IrUnaryOp::Conj) => {
                        rewrite_node(input, true, old_nodes, new_nodes, interned, memo)
                    }
                    (_, IrUnaryOp::NormSqr) => {
                        rewrite_normsqr(input, old_nodes, new_nodes, interned, memo)
                    }
                    (_, IrUnaryOp::Real) => {
                        rewrite_real_with_conj_simplify(input, old_nodes, new_nodes, interned, memo)
                    }
                    (false, _) => {
                        let rewritten_input =
                            rewrite_node(input, false, old_nodes, new_nodes, interned, memo);
                        intern_ir_node(
                            IrNode::Unary {
                                op,
                                input: rewritten_input,
                            },
                            new_nodes,
                            interned,
                        )
                    }
                    (true, IrUnaryOp::Conj) => {
                        rewrite_node(input, false, old_nodes, new_nodes, interned, memo)
                    }
                    (true, IrUnaryOp::Neg) => {
                        let rewritten_input =
                            rewrite_node(input, true, old_nodes, new_nodes, interned, memo);
                        intern_ir_node(
                            IrNode::Unary {
                                op: IrUnaryOp::Neg,
                                input: rewritten_input,
                            },
                            new_nodes,
                            interned,
                        )
                    }
                    (true, _) => {
                        let base =
                            rewrite_node(node_index, false, old_nodes, new_nodes, interned, memo);
                        intern_ir_node(
                            IrNode::Unary {
                                op: IrUnaryOp::Conj,
                                input: base,
                            },
                            new_nodes,
                            interned,
                        )
                    }
                },
                IrNode::Binary { op, left, right } => {
                    let (left_conj, right_conj) = if conj_context {
                        (true, true)
                    } else {
                        (false, false)
                    };
                    let rewritten_left =
                        rewrite_node(left, left_conj, old_nodes, new_nodes, interned, memo);
                    let rewritten_right =
                        rewrite_node(right, right_conj, old_nodes, new_nodes, interned, memo);
                    intern_ir_node(
                        IrNode::Binary {
                            op,
                            left: rewritten_left,
                            right: rewritten_right,
                        },
                        new_nodes,
                        interned,
                    )
                }
            };

            memo.insert((node_index, conj_context), rewritten);
            rewritten
        }

        if ir.nodes.is_empty() {
            return false;
        }

        let old_root = ir.root;
        let old_nodes = ir.nodes.clone();
        let mut new_nodes = Vec::with_capacity(old_nodes.len());
        let mut interned = HashMap::new();
        let mut memo = HashMap::new();
        let root = rewrite_node(
            ir.root,
            false,
            &old_nodes,
            &mut new_nodes,
            &mut interned,
            &mut memo,
        );
        ir.nodes = new_nodes;
        ir.root = root;
        ir.root != old_root || ir.nodes != old_nodes
    }
}

impl IrPass for AlgebraicNormalizePass {
    fn name(&self) -> &'static str {
        "rewrite-algebraic-normalization"
    }

    fn run(&self, ir: &mut ExpressionIR, _ctx: &mut IrPassContext<'_>) -> IrPassReport {
        let changed = self.apply(ir);
        IrPassReport {
            changed,
            rewrites: usize::from(changed),
            saturated: false,
        }
    }
}

struct ControlledExpansionPass {
    max_expansions_override: Option<usize>,
    max_nodes_override: Option<usize>,
}

impl ControlledExpansionPass {
    fn new() -> Self {
        Self {
            max_expansions_override: None,
            max_nodes_override: None,
        }
    }

    #[cfg(test)]
    fn with_limits(max_expansions: usize, max_nodes: usize) -> Self {
        Self {
            max_expansions_override: Some(max_expansions),
            max_nodes_override: Some(max_nodes),
        }
    }

    fn apply(
        &self,
        ir: &mut ExpressionIR,
        amplitude_dependencies: &[DependenceClass],
        rewrite_limits: RewriteLimits,
    ) -> (bool, usize, bool) {
        fn is_separable_mul(left: DependenceClass, right: DependenceClass) -> bool {
            matches!(
                (left, right),
                (DependenceClass::ParameterOnly, DependenceClass::CacheOnly)
                    | (DependenceClass::CacheOnly, DependenceClass::ParameterOnly)
            )
        }

        fn sum_terms(node_index: usize, nodes: &[IrNode]) -> Option<(IrBinaryOp, usize, usize)> {
            match nodes.get(node_index)? {
                IrNode::Binary {
                    op: IrBinaryOp::Add,
                    left,
                    right,
                } => Some((IrBinaryOp::Add, *left, *right)),
                IrNode::Binary {
                    op: IrBinaryOp::Sub,
                    left,
                    right,
                } => Some((IrBinaryOp::Sub, *left, *right)),
                _ => None,
            }
        }

        #[derive(Clone, Copy)]
        enum ExpansionCandidate {
            Left {
                sum_op: IrBinaryOp,
                term_a: usize,
                term_b: usize,
                factor: usize,
            },
            Right {
                factor: usize,
                sum_op: IrBinaryOp,
                term_a: usize,
                term_b: usize,
            },
        }

        fn maybe_expansion_candidate(
            left: usize,
            right: usize,
            nodes: &[IrNode],
            dependencies: &[DependenceClass],
        ) -> Option<ExpansionCandidate> {
            let left_dep = dependencies
                .get(left)
                .copied()
                .unwrap_or(DependenceClass::Mixed);
            let right_dep = dependencies
                .get(right)
                .copied()
                .unwrap_or(DependenceClass::Mixed);
            let base = if is_separable_mul(left_dep, right_dep) {
                1_i32
            } else {
                0_i32
            };

            let left_gain = sum_terms(left, nodes).map(|(sum_op, term_a, term_b)| {
                let dep_a = dependencies
                    .get(term_a)
                    .copied()
                    .unwrap_or(DependenceClass::Mixed);
                let dep_b = dependencies
                    .get(term_b)
                    .copied()
                    .unwrap_or(DependenceClass::Mixed);
                let after = i32::from(is_separable_mul(dep_a, right_dep))
                    + i32::from(is_separable_mul(dep_b, right_dep));
                (
                    after - base,
                    ExpansionCandidate::Left {
                        sum_op,
                        term_a,
                        term_b,
                        factor: right,
                    },
                )
            });

            let right_gain = sum_terms(right, nodes).map(|(sum_op, term_a, term_b)| {
                let dep_a = dependencies
                    .get(term_a)
                    .copied()
                    .unwrap_or(DependenceClass::Mixed);
                let dep_b = dependencies
                    .get(term_b)
                    .copied()
                    .unwrap_or(DependenceClass::Mixed);
                let after = i32::from(is_separable_mul(left_dep, dep_a))
                    + i32::from(is_separable_mul(left_dep, dep_b));
                (
                    after - base,
                    ExpansionCandidate::Right {
                        factor: left,
                        sum_op,
                        term_a,
                        term_b,
                    },
                )
            });

            match (left_gain, right_gain) {
                (Some((left_score, left_candidate)), Some((right_score, right_candidate))) => {
                    if left_score <= 0 && right_score <= 0 {
                        None
                    } else if left_score >= right_score {
                        Some(left_candidate)
                    } else {
                        Some(right_candidate)
                    }
                }
                (Some((left_score, left_candidate)), None) if left_score > 0 => {
                    Some(left_candidate)
                }
                (None, Some((right_score, right_candidate))) if right_score > 0 => {
                    Some(right_candidate)
                }
                _ => None,
            }
        }

        #[allow(clippy::too_many_arguments)]
        fn rewrite_node(
            node_index: usize,
            old_nodes: &[IrNode],
            dependencies: &[DependenceClass],
            new_nodes: &mut Vec<IrNode>,
            interned: &mut HashMap<IrNodeKey, IrValueId>,
            memo: &mut HashMap<usize, IrValueId>,
            expansions_used: &mut usize,
            max_expansions: usize,
            max_nodes: usize,
        ) -> IrValueId {
            #[allow(clippy::too_many_arguments)]
            fn rewrite_binary_mul(
                left: usize,
                right: usize,
                old_nodes: &[IrNode],
                dependencies: &[DependenceClass],
                new_nodes: &mut Vec<IrNode>,
                interned: &mut HashMap<IrNodeKey, IrValueId>,
                memo: &mut HashMap<usize, IrValueId>,
                expansions_used: &mut usize,
                max_expansions: usize,
                max_nodes: usize,
            ) -> IrValueId {
                if *expansions_used < max_expansions
                    && new_nodes.len().saturating_add(2) <= max_nodes
                {
                    if let Some(candidate) =
                        maybe_expansion_candidate(left, right, old_nodes, dependencies)
                    {
                        *expansions_used = expansions_used.saturating_add(1);
                        match candidate {
                            ExpansionCandidate::Left {
                                sum_op,
                                term_a,
                                term_b,
                                factor,
                            } => {
                                let expanded_left = rewrite_binary_mul(
                                    term_a,
                                    factor,
                                    old_nodes,
                                    dependencies,
                                    new_nodes,
                                    interned,
                                    memo,
                                    expansions_used,
                                    max_expansions,
                                    max_nodes,
                                );
                                let expanded_right = rewrite_binary_mul(
                                    term_b,
                                    factor,
                                    old_nodes,
                                    dependencies,
                                    new_nodes,
                                    interned,
                                    memo,
                                    expansions_used,
                                    max_expansions,
                                    max_nodes,
                                );
                                return intern_ir_node(
                                    IrNode::Binary {
                                        op: sum_op,
                                        left: expanded_left,
                                        right: expanded_right,
                                    },
                                    new_nodes,
                                    interned,
                                );
                            }
                            ExpansionCandidate::Right {
                                factor,
                                sum_op,
                                term_a,
                                term_b,
                            } => {
                                let expanded_left = rewrite_binary_mul(
                                    factor,
                                    term_a,
                                    old_nodes,
                                    dependencies,
                                    new_nodes,
                                    interned,
                                    memo,
                                    expansions_used,
                                    max_expansions,
                                    max_nodes,
                                );
                                let expanded_right = rewrite_binary_mul(
                                    factor,
                                    term_b,
                                    old_nodes,
                                    dependencies,
                                    new_nodes,
                                    interned,
                                    memo,
                                    expansions_used,
                                    max_expansions,
                                    max_nodes,
                                );
                                return intern_ir_node(
                                    IrNode::Binary {
                                        op: sum_op,
                                        left: expanded_left,
                                        right: expanded_right,
                                    },
                                    new_nodes,
                                    interned,
                                );
                            }
                        }
                    }
                }

                let rewritten_left = rewrite_node(
                    left,
                    old_nodes,
                    dependencies,
                    new_nodes,
                    interned,
                    memo,
                    expansions_used,
                    max_expansions,
                    max_nodes,
                );
                let rewritten_right = rewrite_node(
                    right,
                    old_nodes,
                    dependencies,
                    new_nodes,
                    interned,
                    memo,
                    expansions_used,
                    max_expansions,
                    max_nodes,
                );
                intern_ir_node(
                    IrNode::Binary {
                        op: IrBinaryOp::Mul,
                        left: rewritten_left,
                        right: rewritten_right,
                    },
                    new_nodes,
                    interned,
                )
            }

            if let Some(&cached) = memo.get(&node_index) {
                return cached;
            }

            let rewritten = match old_nodes[node_index] {
                IrNode::Constant(value) => {
                    intern_ir_node(IrNode::Constant(value), new_nodes, interned)
                }
                IrNode::Amp(amp_idx) => intern_ir_node(IrNode::Amp(amp_idx), new_nodes, interned),
                IrNode::Unary { op, input } => {
                    let rewritten_input = rewrite_node(
                        input,
                        old_nodes,
                        dependencies,
                        new_nodes,
                        interned,
                        memo,
                        expansions_used,
                        max_expansions,
                        max_nodes,
                    );
                    intern_ir_node(
                        IrNode::Unary {
                            op,
                            input: rewritten_input,
                        },
                        new_nodes,
                        interned,
                    )
                }
                IrNode::Binary { op, left, right } => {
                    if op == IrBinaryOp::Mul {
                        rewrite_binary_mul(
                            left,
                            right,
                            old_nodes,
                            dependencies,
                            new_nodes,
                            interned,
                            memo,
                            expansions_used,
                            max_expansions,
                            max_nodes,
                        )
                    } else {
                        let rewritten_left = rewrite_node(
                            left,
                            old_nodes,
                            dependencies,
                            new_nodes,
                            interned,
                            memo,
                            expansions_used,
                            max_expansions,
                            max_nodes,
                        );
                        let rewritten_right = rewrite_node(
                            right,
                            old_nodes,
                            dependencies,
                            new_nodes,
                            interned,
                            memo,
                            expansions_used,
                            max_expansions,
                            max_nodes,
                        );
                        intern_ir_node(
                            IrNode::Binary {
                                op,
                                left: rewritten_left,
                                right: rewritten_right,
                            },
                            new_nodes,
                            interned,
                        )
                    }
                }
            };

            memo.insert(node_index, rewritten);
            rewritten
        }

        if ir.nodes.is_empty() {
            return (false, 0, false);
        }

        let old_root = ir.root;
        let old_nodes = ir.nodes.clone();
        let dependencies =
            DependenceAnnotatePass::compute_annotations(&old_nodes, amplitude_dependencies);
        let max_expansions = self
            .max_expansions_override
            .unwrap_or(rewrite_limits.max_expansions);
        let max_nodes = self
            .max_nodes_override
            .unwrap_or_else(|| rewrite_limits.node_cap(old_nodes.len()));
        let mut new_nodes = Vec::with_capacity(old_nodes.len());
        let mut interned = HashMap::new();
        let mut memo = HashMap::new();
        let mut expansions_used = 0usize;
        let root = rewrite_node(
            ir.root,
            &old_nodes,
            &dependencies,
            &mut new_nodes,
            &mut interned,
            &mut memo,
            &mut expansions_used,
            max_expansions,
            max_nodes,
        );
        ir.nodes = new_nodes;
        ir.root = root;
        let changed = ir.root != old_root || ir.nodes != old_nodes;
        let saturated = expansions_used >= max_expansions || ir.nodes.len() >= max_nodes;
        (changed, expansions_used, saturated)
    }
}

impl IrPass for ControlledExpansionPass {
    fn name(&self) -> &'static str {
        "rewrite-controlled-expansion"
    }

    fn run(&self, ir: &mut ExpressionIR, ctx: &mut IrPassContext<'_>) -> IrPassReport {
        let (changed, rewrites, saturated) =
            self.apply(ir, ctx.amplitude_dependencies, ctx.rewrite_limits);
        IrPassReport {
            changed,
            rewrites,
            saturated,
        }
    }
}

struct ConstantFoldPass;

impl ConstantFoldPass {
    fn apply(&self, ir: &mut ExpressionIR) -> bool {
        let mut changed = false;
        let mut constants: Vec<Option<Complex64>> = vec![None; ir.nodes.len()];
        for index in 0..ir.nodes.len() {
            let folded = match ir.nodes[index].clone() {
                IrNode::Constant(value) => Some(value),
                IrNode::Amp(_) => None,
                IrNode::Unary { op, input } => {
                    constants[input].map(|value| apply_unary_op(op, value))
                }
                IrNode::Binary { op, left, right } => match (constants[left], constants[right]) {
                    (Some(a), Some(b)) => Some(apply_binary_op(op, a, b)),
                    _ => None,
                },
            };
            if let Some(value) = folded {
                if !matches!(ir.nodes[index], IrNode::Constant(existing) if existing == value) {
                    ir.nodes[index] = IrNode::Constant(value);
                    changed = true;
                }
                constants[index] = Some(value);
            }
        }
        changed
    }
}

impl IrPass for ConstantFoldPass {
    fn name(&self) -> &'static str {
        "constant-fold"
    }

    fn run(&self, ir: &mut ExpressionIR, _ctx: &mut IrPassContext<'_>) -> IrPassReport {
        let changed = self.apply(ir);
        IrPassReport {
            changed,
            rewrites: usize::from(changed),
            saturated: false,
        }
    }
}

struct ActivationSpecializePass {
    active_amplitudes: Vec<bool>,
}

impl ActivationSpecializePass {
    fn apply(&self, ir: &mut ExpressionIR) -> bool {
        let mut changed = false;
        for node in &mut ir.nodes {
            if let IrNode::Amp(amp_idx) = node {
                let active = self
                    .active_amplitudes
                    .get(*amp_idx)
                    .copied()
                    .unwrap_or(false);
                if !active {
                    *node = IrNode::Constant(Complex64::ZERO);
                    changed = true;
                }
            }
        }
        changed
    }
}

impl IrPass for ActivationSpecializePass {
    fn name(&self) -> &'static str {
        "activation-specialize"
    }

    fn run(&self, ir: &mut ExpressionIR, _ctx: &mut IrPassContext<'_>) -> IrPassReport {
        let changed = self.apply(ir);
        IrPassReport {
            changed,
            rewrites: usize::from(changed),
            saturated: false,
        }
    }
}

struct CsePass;

impl CsePass {
    fn apply(&self, ir: &mut ExpressionIR) -> bool {
        fn remap_node(node: &IrNode, remap: &[IrValueId]) -> IrNode {
            match *node {
                IrNode::Constant(value) => IrNode::Constant(value),
                IrNode::Amp(idx) => IrNode::Amp(idx),
                IrNode::Unary { op, input } => IrNode::Unary {
                    op,
                    input: remap[input],
                },
                IrNode::Binary { op, left, right } => IrNode::Binary {
                    op,
                    left: remap[left],
                    right: remap[right],
                },
            }
        }

        let mut remap = vec![0usize; ir.nodes.len()];
        let mut interned: HashMap<IrNodeKey, IrValueId> = HashMap::new();
        let mut compacted: Vec<IrNode> = Vec::with_capacity(ir.nodes.len());
        let old_root = ir.root;
        let old_len = ir.nodes.len();

        for (old_id, node) in ir.nodes.iter().enumerate() {
            let remapped = remap_node(node, &remap);
            let key = key_for_ir_node(&remapped);
            if let Some(&existing) = interned.get(&key) {
                remap[old_id] = existing;
            } else {
                let new_id = compacted.len();
                compacted.push(remapped);
                interned.insert(key, new_id);
                remap[old_id] = new_id;
            }
        }

        ir.root = remap[ir.root];
        ir.nodes = compacted;
        ir.root != old_root || ir.nodes.len() != old_len
    }
}

impl IrPass for CsePass {
    fn name(&self) -> &'static str {
        "cse"
    }

    fn run(&self, ir: &mut ExpressionIR, _ctx: &mut IrPassContext<'_>) -> IrPassReport {
        let changed = self.apply(ir);
        IrPassReport {
            changed,
            rewrites: usize::from(changed),
            saturated: false,
        }
    }
}

struct FixedPointRewritePass {
    passes: Vec<Box<dyn IrPass>>,
}

impl FixedPointRewritePass {
    fn new() -> Self {
        Self {
            passes: vec![
                Box::new(AlgebraicNormalizePass),
                Box::new(ConstantFoldPass),
                Box::new(RealValueSimplifyPass),
                Box::new(CsePass),
                Box::new(ControlledExpansionPass::new()),
                Box::new(ConstantFoldPass),
                Box::new(RealValueSimplifyPass),
                Box::new(CsePass),
            ],
        }
    }
}

impl IrPass for FixedPointRewritePass {
    fn name(&self) -> &'static str {
        "rewrite-fixed-point"
    }

    fn run(&self, ir: &mut ExpressionIR, ctx: &mut IrPassContext<'_>) -> IrPassReport {
        let baseline_nodes = ir.nodes.len();
        let node_cap = ctx.rewrite_limits.node_cap(baseline_nodes);
        let max_iterations = ctx.rewrite_limits.max_iterations;
        let mut changed_any = false;
        let mut rewrites = 0usize;
        let mut saturated = false;

        for iteration in 0..max_iterations {
            let mut changed_iteration = false;
            for pass in &self.passes {
                let report = pass.run(ir, ctx);
                changed_iteration |= report.changed;
                rewrites = rewrites.saturating_add(report.rewrites);
                saturated |= report.saturated;
            }

            if ir.nodes.len() > node_cap {
                saturated = true;
                ctx.diagnostics.push(format!(
                    "{} stopped at iteration {} due to node-growth cap ({} > {})",
                    self.name(),
                    iteration + 1,
                    ir.nodes.len(),
                    node_cap
                ));
                break;
            }

            if !changed_iteration {
                return IrPassReport {
                    changed: changed_any,
                    rewrites,
                    saturated,
                };
            }
            changed_any = true;
        }

        if changed_any {
            saturated = true;
            ctx.diagnostics.push(format!(
                "{} reached iteration cap ({})",
                self.name(),
                max_iterations
            ));
        }

        IrPassReport {
            changed: changed_any,
            rewrites,
            saturated,
        }
    }
}

struct ExpressionIrPipeline {
    passes: Vec<Box<dyn IrPass>>,
    rewrite_limits: RewriteLimits,
    amplitude_dependencies: Vec<DependenceClass>,
    amplitude_realness: Vec<bool>,
}

impl ExpressionIrPipeline {
    fn new() -> Self {
        Self {
            passes: Vec::new(),
            rewrite_limits: RewriteLimits::default(),
            amplitude_dependencies: Vec::new(),
            amplitude_realness: Vec::new(),
        }
    }

    fn with_rewrite_limits(mut self, rewrite_limits: RewriteLimits) -> Self {
        self.rewrite_limits = rewrite_limits;
        self
    }

    fn with_amplitude_realness(mut self, amplitude_realness: Vec<bool>) -> Self {
        self.amplitude_realness = amplitude_realness;
        self
    }

    fn cse(mut self) -> Self {
        self.passes.push(Box::new(CsePass));
        self
    }

    fn constant_fold(mut self) -> Self {
        self.passes.push(Box::new(ConstantFoldPass));
        self
    }

    fn activation_specialize(mut self, active_amplitudes: Vec<bool>) -> Self {
        self.passes
            .push(Box::new(ActivationSpecializePass { active_amplitudes }));
        self
    }

    fn rewrite_algebraic_normalization(mut self) -> Self {
        self.passes.push(Box::new(AlgebraicNormalizePass));
        self
    }

    fn rewrite_fixed_point(mut self, amplitude_dependencies: Vec<DependenceClass>) -> Self {
        self.amplitude_dependencies = amplitude_dependencies;
        self.passes.push(Box::new(FixedPointRewritePass::new()));
        self
    }

    fn dependence_annotate(mut self, amplitude_dependencies: Vec<DependenceClass>) -> Self {
        self.amplitude_dependencies = amplitude_dependencies;
        self.passes.push(Box::new(DependenceAnnotatePass));
        self
    }

    fn run(&self, ir: &mut ExpressionIR) -> Vec<String> {
        let mut ctx = IrPassContext {
            amplitude_dependencies: &self.amplitude_dependencies,
            amplitude_realness: &self.amplitude_realness,
            rewrite_limits: self.rewrite_limits,
            diagnostics: Vec::new(),
        };
        for pass in &self.passes {
            let _ = pass.run(ir, &mut ctx);
        }
        ctx.diagnostics
    }
}

struct RealValueSimplifyPass;

impl RealValueSimplifyPass {
    fn compute_annotations(nodes: &[IrNode], amplitude_realness: &[bool]) -> Vec<bool> {
        let mut annotations = vec![false; nodes.len()];
        for (node_index, node) in nodes.iter().enumerate() {
            annotations[node_index] = match *node {
                IrNode::Constant(value) => value.im == 0.0,
                IrNode::Amp(index) => amplitude_realness.get(index).copied().unwrap_or(false),
                IrNode::Unary { op, input } => unary_output_is_real(op, annotations[input]),
                IrNode::Binary { op, left, right } => {
                    binary_output_is_real(op, annotations[left], annotations[right])
                }
            };
        }
        annotations
    }

    fn apply(&self, ir: &mut ExpressionIR, amplitude_realness: &[bool]) -> bool {
        let annotations = Self::compute_annotations(&ir.nodes, amplitude_realness);
        let mut changed = false;
        for index in 0..ir.nodes.len() {
            let rewritten = match ir.nodes[index].clone() {
                IrNode::Unary {
                    op: IrUnaryOp::Real,
                    input,
                } if annotations[input] => Some(ir.nodes[input].clone()),
                IrNode::Unary {
                    op: IrUnaryOp::Imag,
                    input,
                } if annotations[input] => Some(IrNode::Constant(Complex64::ZERO)),
                IrNode::Unary {
                    op: IrUnaryOp::Conj,
                    input,
                } if annotations[input] => Some(ir.nodes[input].clone()),
                _ => None,
            };
            if let Some(node) = rewritten {
                if ir.nodes[index] != node {
                    ir.nodes[index] = node;
                    changed = true;
                }
            }
        }
        changed
    }
}

impl IrPass for RealValueSimplifyPass {
    fn name(&self) -> &'static str {
        "rewrite-real-valued"
    }

    fn run(&self, ir: &mut ExpressionIR, ctx: &mut IrPassContext<'_>) -> IrPassReport {
        let changed = self.apply(ir, ctx.amplitude_realness);
        IrPassReport {
            changed,
            rewrites: usize::from(changed),
            saturated: false,
        }
    }
}

struct DependenceAnnotatePass;

impl DependenceAnnotatePass {
    fn compute_annotations(
        nodes: &[IrNode],
        amplitude_dependencies: &[DependenceClass],
    ) -> Vec<DependenceClass> {
        let mut annotations = vec![DependenceClass::ParameterOnly; nodes.len()];
        for (node_index, node) in nodes.iter().enumerate() {
            annotations[node_index] = match *node {
                IrNode::Constant(_) => DependenceClass::ParameterOnly,
                IrNode::Amp(index) => amplitude_dependencies
                    .get(index)
                    .copied()
                    .unwrap_or(DependenceClass::Mixed),
                IrNode::Unary { op, input } => annotations[input].apply_unary(op),
                IrNode::Binary { op, left, right } => {
                    annotations[left].apply_binary(op, annotations[right])
                }
            };
        }
        annotations
    }

    fn apply(&self, ir: &mut ExpressionIR, amplitude_dependencies: &[DependenceClass]) -> bool {
        let new_annotations = Self::compute_annotations(&ir.nodes, amplitude_dependencies);
        let changed = ir.dependence_annotations != new_annotations;
        ir.dependence_annotations = new_annotations;
        changed
    }
}

impl IrPass for DependenceAnnotatePass {
    fn name(&self) -> &'static str {
        "dependence-annotate"
    }

    fn run(&self, ir: &mut ExpressionIR, ctx: &mut IrPassContext<'_>) -> IrPassReport {
        let changed = self.apply(ir, ctx.amplitude_dependencies);
        IrPassReport {
            changed,
            rewrites: usize::from(changed),
            saturated: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use num::complex::Complex64;

    use super::{
        compile_expression_ir, compile_expression_ir_with_real_hints, ControlledExpansionPass,
        DependenceClass, ExpressionIR, ExpressionIrPipeline,
    };
    use crate::expression::ExpressionNode;

    #[test]
    fn test_expression_ir_constant_fold_pipeline() {
        let tree = ExpressionNode::Mul(
            Box::new(ExpressionNode::Add(
                Box::new(ExpressionNode::One),
                Box::new(ExpressionNode::One),
            )),
            Box::new(ExpressionNode::One),
        );
        let mut ir = ExpressionIR::from_expression_node(&tree);
        ExpressionIrPipeline::new().constant_fold().run(&mut ir);
        assert!(matches!(
            ir.nodes.get(ir.root),
            Some(super::IrNode::Constant(value)) if *value == Complex64::new(2.0, 0.0)
        ));
    }

    #[test]
    fn test_real_valued_amplitude_hint_folds_imag_projection_to_zero() {
        let ir = compile_expression_ir_with_real_hints(
            &ExpressionNode::Imag(Box::new(ExpressionNode::Amp(0))),
            &[true],
            &[DependenceClass::Mixed],
            &[true],
        );

        assert!(matches!(
            ir.nodes().get(ir.root()),
            Some(super::IrNode::Constant(value)) if *value == Complex64::ZERO
        ));
    }

    #[test]
    fn test_real_valued_amplitude_hint_simplifies_real_and_conj() {
        let real_ir = compile_expression_ir_with_real_hints(
            &ExpressionNode::Real(Box::new(ExpressionNode::Amp(0))),
            &[true],
            &[DependenceClass::Mixed],
            &[true],
        );
        assert!(matches!(
            real_ir.nodes().get(real_ir.root()),
            Some(super::IrNode::Amp(0))
        ));

        let conj_ir = compile_expression_ir_with_real_hints(
            &ExpressionNode::Conj(Box::new(ExpressionNode::Amp(0))),
            &[true],
            &[DependenceClass::Mixed],
            &[true],
        );
        assert!(matches!(
            conj_ir.nodes().get(conj_ir.root()),
            Some(super::IrNode::Amp(0))
        ));
    }

    #[test]
    fn test_algebraic_normalize_rewrites_normsqr_to_mul_conj() {
        let tree = ExpressionNode::NormSqr(Box::new(ExpressionNode::Amp(0)));
        let mut ir = ExpressionIR::from_expression_node(&tree);
        ExpressionIrPipeline::new()
            .rewrite_algebraic_normalization()
            .run(&mut ir);
        assert!(!ir.nodes.iter().any(|node| matches!(
            node,
            super::IrNode::Unary {
                op: super::IrUnaryOp::NormSqr,
                ..
            }
        )));
        assert!(ir.nodes.iter().any(|node| {
            matches!(
                node,
                super::IrNode::Binary {
                    op: super::IrBinaryOp::Mul,
                    ..
                }
            )
        }));
    }

    #[test]
    fn test_algebraic_normalize_pushes_conj_through_add() {
        let tree = ExpressionNode::Conj(Box::new(ExpressionNode::Add(
            Box::new(ExpressionNode::Amp(0)),
            Box::new(ExpressionNode::Amp(1)),
        )));
        let mut ir = ExpressionIR::from_expression_node(&tree);
        ExpressionIrPipeline::new()
            .rewrite_algebraic_normalization()
            .cse()
            .run(&mut ir);
        assert!(matches!(
            ir.nodes[ir.root],
            super::IrNode::Binary {
                op: super::IrBinaryOp::Add,
                ..
            }
        ));
        assert!(!ir.nodes.iter().any(|node| {
            matches!(
                node,
                super::IrNode::Unary {
                    op: super::IrUnaryOp::Conj,
                    input,
                } if matches!(
                    ir.nodes[*input],
                    super::IrNode::Binary {
                        op: super::IrBinaryOp::Add
                            | super::IrBinaryOp::Sub
                            | super::IrBinaryOp::Mul
                            | super::IrBinaryOp::Div,
                        ..
                    }
                )
            )
        }));
    }

    #[test]
    fn test_algebraic_normalize_simplifies_double_conj() {
        let tree = ExpressionNode::Conj(Box::new(ExpressionNode::Conj(Box::new(
            ExpressionNode::Amp(0),
        ))));
        let mut ir = ExpressionIR::from_expression_node(&tree);
        ExpressionIrPipeline::new()
            .rewrite_algebraic_normalization()
            .run(&mut ir);
        assert!(matches!(ir.nodes[ir.root], super::IrNode::Amp(0)));
    }

    #[test]
    fn test_algebraic_normalize_simplifies_real_conj_identities() {
        let tree = ExpressionNode::Real(Box::new(ExpressionNode::Conj(Box::new(
            ExpressionNode::Amp(0),
        ))));
        let mut ir = ExpressionIR::from_expression_node(&tree);
        ExpressionIrPipeline::new()
            .rewrite_algebraic_normalization()
            .run(&mut ir);
        assert!(matches!(
            ir.nodes[ir.root],
            super::IrNode::Unary {
                op: super::IrUnaryOp::Real,
                input
            } if matches!(ir.nodes[input], super::IrNode::Amp(0))
        ));

        let tree = ExpressionNode::Conj(Box::new(ExpressionNode::Real(Box::new(
            ExpressionNode::Amp(0),
        ))));
        let mut ir = ExpressionIR::from_expression_node(&tree);
        ExpressionIrPipeline::new()
            .rewrite_algebraic_normalization()
            .run(&mut ir);
        assert!(matches!(
            ir.nodes[ir.root],
            super::IrNode::Unary {
                op: super::IrUnaryOp::Real,
                input
            } if matches!(ir.nodes[input], super::IrNode::Amp(0))
        ));
    }

    #[test]
    fn test_compile_expression_ir_applies_algebraic_normalization_rules() {
        let tree = ExpressionNode::Conj(Box::new(ExpressionNode::NormSqr(Box::new(
            ExpressionNode::Amp(0),
        ))));
        let ir = compile_expression_ir(&tree, &[true], &[DependenceClass::Mixed]);
        assert!(!ir.nodes.iter().any(|node| matches!(
            node,
            super::IrNode::Unary {
                op: super::IrUnaryOp::NormSqr,
                ..
            }
        )));
        assert!(matches!(
            ir.nodes[ir.root],
            super::IrNode::Binary {
                op: super::IrBinaryOp::Mul,
                ..
            }
        ));
    }

    #[test]
    fn test_controlled_expansion_distributes_when_separable_yield_improves() {
        let tree = ExpressionNode::Mul(
            Box::new(ExpressionNode::Add(
                Box::new(ExpressionNode::Amp(0)),
                Box::new(ExpressionNode::Amp(1)),
            )),
            Box::new(ExpressionNode::Amp(2)),
        );
        let ir = compile_expression_ir(
            &tree,
            &[true, true, true],
            &[
                DependenceClass::ParameterOnly,
                DependenceClass::ParameterOnly,
                DependenceClass::CacheOnly,
            ],
        );
        assert!(matches!(
            ir.nodes[ir.root],
            super::IrNode::Binary {
                op: super::IrBinaryOp::Add,
                ..
            }
        ));
        assert_eq!(ir.cached_integral_descriptors().len(), 2);
        assert!(!ir.nodes.iter().any(|node| {
            matches!(
                node,
                super::IrNode::Binary {
                    op: super::IrBinaryOp::Mul,
                    left,
                    right
                } if matches!(
                    ir.nodes[*left],
                    super::IrNode::Binary {
                        op: super::IrBinaryOp::Add | super::IrBinaryOp::Sub,
                        ..
                    }
                ) || matches!(
                    ir.nodes[*right],
                    super::IrNode::Binary {
                        op: super::IrBinaryOp::Add | super::IrBinaryOp::Sub,
                        ..
                    }
                )
            )
        }));
    }

    #[test]
    fn test_controlled_expansion_skips_when_separable_yield_does_not_improve() {
        let tree = ExpressionNode::Mul(
            Box::new(ExpressionNode::Add(
                Box::new(ExpressionNode::Amp(0)),
                Box::new(ExpressionNode::Amp(1)),
            )),
            Box::new(ExpressionNode::Amp(2)),
        );
        let ir = compile_expression_ir(
            &tree,
            &[true, true, true],
            &[
                DependenceClass::Mixed,
                DependenceClass::Mixed,
                DependenceClass::CacheOnly,
            ],
        );
        assert!(matches!(
            ir.nodes[ir.root],
            super::IrNode::Binary {
                op: super::IrBinaryOp::Mul,
                left,
                right,
            } if matches!(
                ir.nodes[left],
                super::IrNode::Binary {
                    op: super::IrBinaryOp::Add,
                    ..
                }
            ) || matches!(
                ir.nodes[right],
                super::IrNode::Binary {
                    op: super::IrBinaryOp::Add,
                    ..
                }
            )
        ));
        assert!(ir.cached_integral_descriptors().is_empty());
    }

    #[test]
    fn test_controlled_expansion_honors_expansion_budget() {
        let tree = ExpressionNode::Mul(
            Box::new(ExpressionNode::Add(
                Box::new(ExpressionNode::Amp(0)),
                Box::new(ExpressionNode::Amp(1)),
            )),
            Box::new(ExpressionNode::Amp(2)),
        );
        let mut ir = ExpressionIR::from_expression_node(&tree);
        let pass = ControlledExpansionPass::with_limits(0, 100);
        let mut ctx = super::IrPassContext {
            amplitude_dependencies: &[
                DependenceClass::ParameterOnly,
                DependenceClass::ParameterOnly,
                DependenceClass::CacheOnly,
            ],
            amplitude_realness: &[],
            rewrite_limits: super::RewriteLimits::default(),
            diagnostics: Vec::new(),
        };
        super::IrPass::run(&pass, &mut ir, &mut ctx);
        assert!(matches!(
            ir.nodes[ir.root],
            super::IrNode::Binary {
                op: super::IrBinaryOp::Mul,
                left,
                right,
            } if matches!(
                ir.nodes[left],
                super::IrNode::Binary {
                    op: super::IrBinaryOp::Add,
                    ..
                }
            ) || matches!(
                ir.nodes[right],
                super::IrNode::Binary {
                    op: super::IrBinaryOp::Add,
                    ..
                }
            )
        ));
    }

    #[test]
    fn test_rewrite_fixed_point_reports_iteration_cap() {
        let tree = ExpressionNode::NormSqr(Box::new(ExpressionNode::Amp(0)));
        let mut ir = ExpressionIR::from_expression_node(&tree);
        let diagnostics = ExpressionIrPipeline::new()
            .with_rewrite_limits(super::RewriteLimits {
                max_iterations: 1,
                max_expansions: 128,
                max_nodes_multiplier: 4,
                max_nodes_additive: 32,
            })
            .rewrite_fixed_point(vec![DependenceClass::Mixed])
            .run(&mut ir);
        assert!(diagnostics
            .iter()
            .any(|message| message.contains("reached iteration cap")));
    }

    #[test]
    fn test_rewrite_fixed_point_reports_node_growth_cap() {
        let tree = ExpressionNode::Mul(
            Box::new(ExpressionNode::Add(
                Box::new(ExpressionNode::Amp(0)),
                Box::new(ExpressionNode::Amp(1)),
            )),
            Box::new(ExpressionNode::Add(
                Box::new(ExpressionNode::Amp(2)),
                Box::new(ExpressionNode::Amp(3)),
            )),
        );
        let mut ir = ExpressionIR::from_expression_node(&tree);
        let diagnostics = ExpressionIrPipeline::new()
            .with_rewrite_limits(super::RewriteLimits {
                max_iterations: 4,
                max_expansions: 128,
                max_nodes_multiplier: 1,
                max_nodes_additive: 0,
            })
            .rewrite_fixed_point(vec![
                DependenceClass::ParameterOnly,
                DependenceClass::ParameterOnly,
                DependenceClass::CacheOnly,
                DependenceClass::CacheOnly,
            ])
            .run(&mut ir);
        assert!(diagnostics
            .iter()
            .any(|message| message.contains("node-growth cap")));
    }

    #[test]
    fn test_expression_ir_activation_specialization_pipeline() {
        let tree = ExpressionNode::Add(
            Box::new(ExpressionNode::Amp(0)),
            Box::new(ExpressionNode::Amp(1)),
        );
        let mut ir = ExpressionIR::from_expression_node(&tree);
        ExpressionIrPipeline::new()
            .activation_specialize(vec![true, false])
            .constant_fold()
            .run(&mut ir);
        let constant_zeros = ir
            .nodes
            .iter()
            .filter(
                |node| matches!(node, super::IrNode::Constant(value) if *value == Complex64::ZERO),
            )
            .count();
        assert!(constant_zeros >= 1);
    }

    #[test]
    fn test_expression_ir_cse_pipeline_deduplicates_nodes() {
        let repeated = ExpressionNode::Add(
            Box::new(ExpressionNode::Amp(0)),
            Box::new(ExpressionNode::Amp(1)),
        );
        let tree = ExpressionNode::Add(Box::new(repeated.clone()), Box::new(repeated));
        let mut ir = ExpressionIR::from_expression_node(&tree);
        let before = ir.node_count();
        ExpressionIrPipeline::new().cse().run(&mut ir);
        let after = ir.node_count();
        assert!(after < before);
    }

    #[test]
    fn test_dependence_class_merge_rules() {
        assert_eq!(
            DependenceClass::ParameterOnly.merge(DependenceClass::ParameterOnly),
            DependenceClass::ParameterOnly
        );
        assert_eq!(
            DependenceClass::CacheOnly.merge(DependenceClass::CacheOnly),
            DependenceClass::CacheOnly
        );
        assert_eq!(
            DependenceClass::ParameterOnly.merge(DependenceClass::CacheOnly),
            DependenceClass::Mixed
        );
        assert_eq!(
            DependenceClass::CacheOnly.merge(DependenceClass::ParameterOnly),
            DependenceClass::Mixed
        );
        assert_eq!(
            DependenceClass::Mixed.merge(DependenceClass::CacheOnly),
            DependenceClass::Mixed
        );
    }

    #[test]
    fn test_root_dependence_for_constant_expression() {
        let tree =
            ExpressionNode::Add(Box::new(ExpressionNode::One), Box::new(ExpressionNode::One));
        let ir = ExpressionIR::from_expression_node(&tree);
        assert_eq!(ir.root_dependence(), DependenceClass::ParameterOnly);
    }

    #[test]
    fn test_root_dependence_for_amplitude_expression_is_mixed() {
        let tree = ExpressionNode::Mul(
            Box::new(ExpressionNode::Amp(0)),
            Box::new(ExpressionNode::One),
        );
        let ir = ExpressionIR::from_expression_node(&tree);
        assert_eq!(ir.root_dependence(), DependenceClass::Mixed);
    }

    #[test]
    fn test_dependence_annotation_pass_marks_each_node() {
        let tree = ExpressionNode::Add(
            Box::new(ExpressionNode::One),
            Box::new(ExpressionNode::Amp(0)),
        );
        let mut ir = ExpressionIR::from_expression_node(&tree);
        ExpressionIrPipeline::new()
            .dependence_annotate(vec![DependenceClass::Mixed])
            .run(&mut ir);
        assert_eq!(ir.node_dependence_annotations().len(), ir.node_count());
        assert_eq!(
            ir.node_dependence_annotations()[0],
            DependenceClass::ParameterOnly
        );
        assert_eq!(ir.node_dependence_annotations()[1], DependenceClass::Mixed);
        assert_eq!(
            ir.node_dependence_annotations()[ir.root],
            DependenceClass::Mixed
        );
    }

    #[test]
    fn test_dependence_annotation_after_specialization_and_fold() {
        let tree = ExpressionNode::Add(
            Box::new(ExpressionNode::Amp(0)),
            Box::new(ExpressionNode::One),
        );
        let mut ir = ExpressionIR::from_expression_node(&tree);
        ExpressionIrPipeline::new()
            .activation_specialize(vec![false])
            .constant_fold()
            .dependence_annotate(vec![DependenceClass::Mixed])
            .run(&mut ir);
        assert!(ir
            .node_dependence_annotations()
            .iter()
            .all(|class| *class == DependenceClass::ParameterOnly));
        assert_eq!(ir.root_dependence(), DependenceClass::ParameterOnly);
    }

    #[test]
    fn test_dependence_warning_when_parameter_and_cache_hints_are_both_active() {
        let tree = ExpressionNode::Add(
            Box::new(ExpressionNode::Amp(0)),
            Box::new(ExpressionNode::Amp(1)),
        );
        let ir = compile_expression_ir(
            &tree,
            &[true, true],
            &[DependenceClass::ParameterOnly, DependenceClass::CacheOnly],
        );
        assert!(ir
            .dependence_warnings()
            .iter()
            .any(|warning: &String| warning.contains("both ParameterOnly and CacheOnly")));
    }

    #[test]
    fn test_dependence_warning_when_active_hint_is_missing() {
        let tree = ExpressionNode::Amp(0);
        let ir = compile_expression_ir(&tree, &[true], &[]);
        assert!(ir
            .dependence_warnings()
            .iter()
            .any(|warning: &String| warning.contains("missing dependence hints")));
    }

    #[test]
    fn test_separable_mul_candidate_detects_parameter_times_cache() {
        let tree = ExpressionNode::Mul(
            Box::new(ExpressionNode::Amp(0)),
            Box::new(ExpressionNode::Amp(1)),
        );
        let ir = compile_expression_ir(
            &tree,
            &[true, true],
            &[DependenceClass::ParameterOnly, DependenceClass::CacheOnly],
        );
        assert_eq!(ir.separable_mul_candidates().len(), 1);
        let candidate = ir.separable_mul_candidates()[0];
        assert_eq!(candidate.left_dependence, DependenceClass::ParameterOnly);
        assert_eq!(candidate.right_dependence, DependenceClass::CacheOnly);
    }

    #[test]
    fn test_separable_mul_candidate_ignores_mixed_inputs() {
        let tree = ExpressionNode::Mul(
            Box::new(ExpressionNode::Amp(0)),
            Box::new(ExpressionNode::Amp(1)),
        );
        let ir = compile_expression_ir(
            &tree,
            &[true, true],
            &[DependenceClass::Mixed, DependenceClass::CacheOnly],
        );
        assert!(ir.separable_mul_candidates().is_empty());
    }

    #[test]
    fn test_separable_mul_candidate_stable_under_specialization() {
        let tree = ExpressionNode::Mul(
            Box::new(ExpressionNode::Amp(0)),
            Box::new(ExpressionNode::Amp(1)),
        );
        let ir = compile_expression_ir(
            &tree,
            &[true, false],
            &[DependenceClass::ParameterOnly, DependenceClass::CacheOnly],
        );
        assert!(ir.separable_mul_candidates().is_empty());
    }

    #[test]
    fn test_normalization_plan_partitions_separable_terms() {
        let tree = ExpressionNode::Mul(
            Box::new(ExpressionNode::Amp(0)),
            Box::new(ExpressionNode::Amp(1)),
        );
        let ir = compile_expression_ir(
            &tree,
            &[true, true],
            &[DependenceClass::ParameterOnly, DependenceClass::CacheOnly],
        );
        assert_eq!(ir.normalization_plan().cached_separable_nodes.len(), 1);
        let cached = ir.normalization_plan().cached_separable_nodes[0];
        assert_eq!(cached, ir.root);
        assert_eq!(ir.normalization_plan().cached_separable_nodes, vec![cached]);
        assert!(ir
            .normalization_plan()
            .residual_terms
            .iter()
            .all(|index| *index != cached));
    }

    #[test]
    fn test_normalization_plan_non_separable_is_residual_only() {
        let tree = ExpressionNode::Mul(
            Box::new(ExpressionNode::Amp(0)),
            Box::new(ExpressionNode::Amp(1)),
        );
        let ir = compile_expression_ir(
            &tree,
            &[true, true],
            &[DependenceClass::Mixed, DependenceClass::CacheOnly],
        );
        assert!(ir.normalization_plan().cached_separable_nodes.is_empty());
        assert!(ir.normalization_plan().residual_terms.contains(&ir.root));
    }

    #[test]
    fn test_normalization_plan_stable_under_specialization() {
        let tree = ExpressionNode::Mul(
            Box::new(ExpressionNode::Amp(0)),
            Box::new(ExpressionNode::Amp(1)),
        );
        let ir = compile_expression_ir(
            &tree,
            &[true, false],
            &[DependenceClass::ParameterOnly, DependenceClass::CacheOnly],
        );
        assert!(ir.normalization_plan().cached_separable_nodes.is_empty());
        assert!(ir.normalization_plan().residual_terms.contains(&ir.root));
    }

    #[test]
    fn test_normalization_plan_explain_for_separable_case() {
        let tree = ExpressionNode::Mul(
            Box::new(ExpressionNode::Amp(0)),
            Box::new(ExpressionNode::Amp(1)),
        );
        let ir = compile_expression_ir(
            &tree,
            &[true, true],
            &[DependenceClass::ParameterOnly, DependenceClass::CacheOnly],
        );
        let explain = ir.normalization_plan_explain();
        assert_eq!(explain.root_dependence, DependenceClass::Mixed);
        assert_eq!(explain.separable_mul_candidates.len(), 1);
        assert_eq!(explain.cached_separable_nodes, vec![ir.root]);
        assert!(explain.residual_terms.iter().all(|index| *index != ir.root));
    }

    #[test]
    fn test_normalization_plan_explain_for_non_separable_case() {
        let tree = ExpressionNode::Mul(
            Box::new(ExpressionNode::Amp(0)),
            Box::new(ExpressionNode::Amp(1)),
        );
        let ir = compile_expression_ir(
            &tree,
            &[true, true],
            &[DependenceClass::Mixed, DependenceClass::Mixed],
        );
        let explain = ir.normalization_plan_explain();
        assert!(explain.separable_mul_candidates.is_empty());
        assert!(explain.cached_separable_nodes.is_empty());
        assert!(explain.residual_terms.contains(&ir.root));
    }

    #[test]
    fn test_normalization_execution_sets_for_fully_separable_term() {
        let tree = ExpressionNode::Mul(
            Box::new(ExpressionNode::Amp(0)),
            Box::new(ExpressionNode::Amp(1)),
        );
        let ir = compile_expression_ir(
            &tree,
            &[true, true],
            &[DependenceClass::ParameterOnly, DependenceClass::CacheOnly],
        );
        let sets = ir.normalization_execution_sets();
        assert_eq!(sets.cached_parameter_amplitudes, vec![0]);
        assert_eq!(sets.cached_cache_amplitudes, vec![1]);
        assert!(sets.residual_amplitudes.is_empty());
    }

    #[test]
    fn test_normalization_execution_sets_for_partial_factorization() {
        let tree = ExpressionNode::Add(
            Box::new(ExpressionNode::Mul(
                Box::new(ExpressionNode::Amp(0)),
                Box::new(ExpressionNode::Amp(1)),
            )),
            Box::new(ExpressionNode::Amp(2)),
        );
        let ir = compile_expression_ir(
            &tree,
            &[true, true, true],
            &[
                DependenceClass::ParameterOnly,
                DependenceClass::CacheOnly,
                DependenceClass::Mixed,
            ],
        );
        let sets = ir.normalization_execution_sets();
        assert_eq!(sets.cached_parameter_amplitudes, vec![0]);
        assert_eq!(sets.cached_cache_amplitudes, vec![1]);
        assert_eq!(sets.residual_amplitudes, vec![2]);
    }

    #[test]
    fn test_normalization_execution_sets_for_non_separable_term() {
        let tree = ExpressionNode::Mul(
            Box::new(ExpressionNode::Amp(0)),
            Box::new(ExpressionNode::Amp(1)),
        );
        let ir = compile_expression_ir(
            &tree,
            &[true, true],
            &[DependenceClass::Mixed, DependenceClass::CacheOnly],
        );
        let sets = ir.normalization_execution_sets();
        assert!(sets.cached_parameter_amplitudes.is_empty());
        assert!(sets.cached_cache_amplitudes.is_empty());
        assert_eq!(sets.residual_amplitudes, vec![0, 1]);
    }

    #[test]
    fn test_cached_separable_nodes_match_candidates_for_mixed_tree() {
        let tree = ExpressionNode::Add(
            Box::new(ExpressionNode::Mul(
                Box::new(ExpressionNode::Amp(0)),
                Box::new(ExpressionNode::Amp(1)),
            )),
            Box::new(ExpressionNode::Mul(
                Box::new(ExpressionNode::Amp(2)),
                Box::new(ExpressionNode::Amp(3)),
            )),
        );
        let ir = compile_expression_ir(
            &tree,
            &[true, true, true, true],
            &[
                DependenceClass::ParameterOnly,
                DependenceClass::CacheOnly,
                DependenceClass::Mixed,
                DependenceClass::CacheOnly,
            ],
        );
        let mut candidate_nodes = ir
            .separable_mul_candidates()
            .iter()
            .map(|candidate| candidate.node_index)
            .collect::<Vec<_>>();
        candidate_nodes.sort_unstable();
        assert_eq!(
            ir.normalization_plan().cached_separable_nodes,
            candidate_nodes
        );
    }

    #[test]
    fn test_cached_integral_descriptors_include_parameter_and_cache_nodes() {
        let tree = ExpressionNode::Mul(
            Box::new(ExpressionNode::Amp(0)),
            Box::new(ExpressionNode::Amp(1)),
        );
        let ir = compile_expression_ir(
            &tree,
            &[true, true],
            &[DependenceClass::ParameterOnly, DependenceClass::CacheOnly],
        );
        assert_eq!(ir.cached_integral_descriptors().len(), 1);
        let descriptor = ir.cached_integral_descriptors()[0];
        assert_eq!(descriptor.mul_node_index, ir.root);
        assert_eq!(descriptor.parameter_node_index, 0);
        assert_eq!(descriptor.cache_node_index, 1);
        assert_eq!(descriptor.coefficient, 1);
    }

    #[test]
    fn test_cached_integral_descriptors_empty_for_non_separable_tree() {
        let tree = ExpressionNode::Mul(
            Box::new(ExpressionNode::Amp(0)),
            Box::new(ExpressionNode::Amp(1)),
        );
        let ir = compile_expression_ir(
            &tree,
            &[true, true],
            &[DependenceClass::Mixed, DependenceClass::CacheOnly],
        );
        assert!(ir.cached_integral_descriptors().is_empty());
    }

    #[test]
    fn test_cached_integral_descriptors_require_global_extractability() {
        let tree = ExpressionNode::Mul(
            Box::new(ExpressionNode::Amp(2)),
            Box::new(ExpressionNode::Mul(
                Box::new(ExpressionNode::Amp(0)),
                Box::new(ExpressionNode::Amp(1)),
            )),
        );
        let ir = compile_expression_ir(
            &tree,
            &[true, true, true],
            &[
                DependenceClass::ParameterOnly,
                DependenceClass::CacheOnly,
                DependenceClass::Mixed,
            ],
        );
        assert!(ir.cached_integral_descriptors().is_empty());
    }

    #[test]
    fn test_cached_integral_descriptors_capture_subtraction_sign() {
        let tree = ExpressionNode::Sub(
            Box::new(ExpressionNode::One),
            Box::new(ExpressionNode::Mul(
                Box::new(ExpressionNode::Amp(0)),
                Box::new(ExpressionNode::Amp(1)),
            )),
        );
        let ir = compile_expression_ir(
            &tree,
            &[true, true],
            &[DependenceClass::ParameterOnly, DependenceClass::CacheOnly],
        );
        assert_eq!(ir.cached_integral_descriptors().len(), 1);
        assert_eq!(ir.cached_integral_descriptors()[0].coefficient, -1);
    }

    #[test]
    fn test_cached_integral_descriptors_capture_negation_sign() {
        let tree = ExpressionNode::Neg(Box::new(ExpressionNode::Mul(
            Box::new(ExpressionNode::Amp(0)),
            Box::new(ExpressionNode::Amp(1)),
        )));
        let ir = compile_expression_ir(
            &tree,
            &[true, true],
            &[DependenceClass::ParameterOnly, DependenceClass::CacheOnly],
        );
        assert_eq!(ir.cached_integral_descriptors().len(), 1);
        assert_eq!(ir.cached_integral_descriptors()[0].coefficient, -1);
    }
}
