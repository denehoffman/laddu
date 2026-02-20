use super::ExpressionNode;
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
