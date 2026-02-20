use super::ExpressionNode;
use nalgebra::DVector;
use num::complex::Complex64;
use std::collections::HashMap;

type IrValueId = usize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum IrUnaryOp {
    Neg,
    Real,
    Imag,
    Conj,
    NormSqr,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum IrBinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Clone, Debug, PartialEq)]
enum IrNode {
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
pub(super) struct ExpressionIR {
    nodes: Vec<IrNode>,
    root: IrValueId,
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
            }
        }

        let mut nodes = Vec::new();
        let root = lower(node, &mut nodes);
        Self { nodes, root }
    }

    pub(super) fn node_count(&self) -> usize {
        self.nodes.len()
    }

    fn fill_values(&self, amplitude_values: &[Complex64], slots: &mut [Complex64]) {
        debug_assert!(slots.len() >= self.nodes.len());
        for (node_index, node) in self.nodes.iter().enumerate() {
            slots[node_index] = match *node {
                IrNode::Constant(value) => value,
                IrNode::Amp(amp_idx) => amplitude_values.get(amp_idx).copied().unwrap_or_default(),
                IrNode::Unary { op, input } => {
                    let value = slots[input];
                    match op {
                        IrUnaryOp::Neg => -value,
                        IrUnaryOp::Real => Complex64::new(value.re, 0.0),
                        IrUnaryOp::Imag => Complex64::new(value.im, 0.0),
                        IrUnaryOp::Conj => value.conj(),
                        IrUnaryOp::NormSqr => Complex64::new(value.norm_sqr(), 0.0),
                    }
                }
                IrNode::Binary { op, left, right } => {
                    let left_value = slots[left];
                    let right_value = slots[right];
                    match op {
                        IrBinaryOp::Add => left_value + right_value,
                        IrBinaryOp::Sub => left_value - right_value,
                        IrBinaryOp::Mul => left_value * right_value,
                        IrBinaryOp::Div => left_value / right_value,
                    }
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
                    match op {
                        IrUnaryOp::Neg => {
                            for (dst_item, input_item) in dst_grad.iter_mut().zip(input_grad.iter())
                            {
                                *dst_item = -*input_item;
                            }
                        }
                        IrUnaryOp::Real => {
                            for (dst_item, input_item) in dst_grad.iter_mut().zip(input_grad.iter())
                            {
                                *dst_item = Complex64::new(input_item.re, 0.0);
                            }
                        }
                        IrUnaryOp::Imag => {
                            for (dst_item, input_item) in dst_grad.iter_mut().zip(input_grad.iter())
                            {
                                *dst_item = Complex64::new(input_item.im, 0.0);
                            }
                        }
                        IrUnaryOp::Conj => {
                            for (dst_item, input_item) in dst_grad.iter_mut().zip(input_grad.iter())
                            {
                                *dst_item = input_item.conj();
                            }
                        }
                        IrUnaryOp::NormSqr => {
                            let conj_input = value_slots[input].conj();
                            for (dst_item, input_item) in dst_grad.iter_mut().zip(input_grad.iter())
                            {
                                *dst_item =
                                    Complex64::new(2.0 * (*input_item * conj_input).re, 0.0);
                            }
                        }
                    }
                }
                IrNode::Binary { op, left, right } => {
                    let left_grad = &before[left];
                    let right_grad = &before[right];
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
                            let left_value = value_slots[left];
                            let right_value = value_slots[right];
                            for ((dst_item, left_item), right_item) in dst_grad
                                .iter_mut()
                                .zip(left_grad.iter())
                                .zip(right_grad.iter())
                            {
                                *dst_item = *left_item * right_value + *right_item * left_value;
                            }
                        }
                        IrBinaryOp::Div => {
                            let left_value = value_slots[left];
                            let right_value = value_slots[right];
                            let denom = right_value * right_value;
                            for ((dst_item, left_item), right_item) in dst_grad
                                .iter_mut()
                                .zip(left_grad.iter())
                                .zip(right_grad.iter())
                            {
                                *dst_item =
                                    (*left_item * right_value - *right_item * left_value) / denom;
                            }
                        }
                    }
                }
            }
        }
        gradient_slots[self.root].clone()
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
}

pub(super) fn compile_expression_ir(
    tree: &ExpressionNode,
    active_amplitudes: &[bool],
) -> ExpressionIR {
    let mut ir = ExpressionIR::from_expression_node(tree);
    ExpressionIrPipeline::new()
        .cse()
        .activation_specialize(active_amplitudes.to_vec())
        .constant_fold()
        .run(&mut ir);
    ir
}

struct ConstantFoldPass;

impl ConstantFoldPass {
    fn run(&self, ir: &mut ExpressionIR) {
        let mut constants: Vec<Option<Complex64>> = vec![None; ir.nodes.len()];
        for index in 0..ir.nodes.len() {
            let folded = match ir.nodes[index].clone() {
                IrNode::Constant(value) => Some(value),
                IrNode::Amp(_) => None,
                IrNode::Unary { op, input } => constants[input].map(|value| match op {
                    IrUnaryOp::Neg => -value,
                    IrUnaryOp::Real => Complex64::new(value.re, 0.0),
                    IrUnaryOp::Imag => Complex64::new(value.im, 0.0),
                    IrUnaryOp::Conj => value.conj(),
                    IrUnaryOp::NormSqr => Complex64::new(value.norm_sqr(), 0.0),
                }),
                IrNode::Binary { op, left, right } => match (constants[left], constants[right]) {
                    (Some(a), Some(b)) => Some(match op {
                        IrBinaryOp::Add => a + b,
                        IrBinaryOp::Sub => a - b,
                        IrBinaryOp::Mul => a * b,
                        IrBinaryOp::Div => a / b,
                    }),
                    _ => None,
                },
            };
            if let Some(value) = folded {
                ir.nodes[index] = IrNode::Constant(value);
                constants[index] = Some(value);
            }
        }
    }
}

struct ActivationSpecializePass {
    active_amplitudes: Vec<bool>,
}

impl ActivationSpecializePass {
    fn run(&self, ir: &mut ExpressionIR) {
        for node in &mut ir.nodes {
            if let IrNode::Amp(amp_idx) = node {
                let active = self
                    .active_amplitudes
                    .get(*amp_idx)
                    .copied()
                    .unwrap_or(false);
                if !active {
                    *node = IrNode::Constant(Complex64::ZERO);
                }
            }
        }
    }
}

struct CsePass;

impl CsePass {
    fn run(&self, ir: &mut ExpressionIR) {
        fn key_for(node: &IrNode) -> IrNodeKey {
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

        for (old_id, node) in ir.nodes.iter().enumerate() {
            let remapped = remap_node(node, &remap);
            let key = key_for(&remapped);
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
    }
}

enum IrPassKind {
    Cse,
    ConstantFold,
    ActivationSpecialize(Vec<bool>),
}

struct ExpressionIrPipeline {
    passes: Vec<IrPassKind>,
}

impl ExpressionIrPipeline {
    fn new() -> Self {
        Self { passes: Vec::new() }
    }

    fn cse(mut self) -> Self {
        self.passes.push(IrPassKind::Cse);
        self
    }

    fn constant_fold(mut self) -> Self {
        self.passes.push(IrPassKind::ConstantFold);
        self
    }

    fn activation_specialize(mut self, active_amplitudes: Vec<bool>) -> Self {
        self.passes
            .push(IrPassKind::ActivationSpecialize(active_amplitudes));
        self
    }

    fn run(&self, ir: &mut ExpressionIR) {
        for pass in &self.passes {
            match pass {
                IrPassKind::Cse => CsePass.run(ir),
                IrPassKind::ConstantFold => ConstantFoldPass.run(ir),
                IrPassKind::ActivationSpecialize(active_amplitudes) => ActivationSpecializePass {
                    active_amplitudes: active_amplitudes.clone(),
                }
                .run(ir),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use num::complex::Complex64;

    use super::{ExpressionIR, ExpressionIrPipeline};
    use crate::amplitudes::ExpressionNode;

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
}
