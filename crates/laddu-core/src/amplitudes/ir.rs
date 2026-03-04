use super::ExpressionNode;
use nalgebra::DVector;
use num::complex::Complex64;
use std::collections::HashMap;

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
    dependence_annotations: Vec<DependenceClass>,
    dependence_warnings: Vec<String>,
    separable_mul_candidates: Vec<SeparableMulCandidate>,
    normalization_plan: NormalizationPlan,
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
        Self {
            nodes,
            root,
            dependence_annotations: Vec::new(),
            dependence_warnings: Vec::new(),
            separable_mul_candidates: Vec::new(),
            normalization_plan: NormalizationPlan::default(),
        }
    }

    pub(super) fn node_count(&self) -> usize {
        self.nodes.len()
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

    pub(super) fn separable_mul_candidates(&self) -> &[SeparableMulCandidate] {
        &self.separable_mul_candidates
    }

    pub(super) fn normalization_plan(&self) -> &NormalizationPlan {
        &self.normalization_plan
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
}

pub(super) fn compile_expression_ir(
    tree: &ExpressionNode,
    active_amplitudes: &[bool],
    amplitude_dependencies: &[DependenceClass],
) -> ExpressionIR {
    let mut ir = ExpressionIR::from_expression_node(tree);
    ExpressionIrPipeline::new()
        .cse()
        .activation_specialize(active_amplitudes.to_vec())
        .constant_fold()
        .rewrite_algebraic_normalization()
        .constant_fold()
        .cse()
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

struct AlgebraicNormalizePass;

impl AlgebraicNormalizePass {
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

        fn intern_node(
            node: IrNode,
            nodes: &mut Vec<IrNode>,
            interned: &mut HashMap<IrNodeKey, IrValueId>,
        ) -> IrValueId {
            let key = key_for(&node);
            if let Some(&existing) = interned.get(&key) {
                return existing;
            }
            let id = nodes.len();
            nodes.push(node);
            interned.insert(key, id);
            id
        }

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
                intern_node(
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
                intern_node(
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
                    intern_node(IrNode::Constant(folded), new_nodes, interned)
                }
                IrNode::Amp(amp_idx) => {
                    if conj_context {
                        let base = intern_node(IrNode::Amp(amp_idx), new_nodes, interned);
                        intern_node(
                            IrNode::Unary {
                                op: IrUnaryOp::Conj,
                                input: base,
                            },
                            new_nodes,
                            interned,
                        )
                    } else {
                        intern_node(IrNode::Amp(amp_idx), new_nodes, interned)
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
                        intern_node(
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
                        intern_node(
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
                        intern_node(
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
                    intern_node(
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
            return;
        }

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
    }
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
    RewriteAlgebraicNormalization,
    DependenceAnnotate(Vec<DependenceClass>),
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

    fn rewrite_algebraic_normalization(mut self) -> Self {
        self.passes.push(IrPassKind::RewriteAlgebraicNormalization);
        self
    }

    fn dependence_annotate(mut self, amplitude_dependencies: Vec<DependenceClass>) -> Self {
        self.passes
            .push(IrPassKind::DependenceAnnotate(amplitude_dependencies));
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
                IrPassKind::RewriteAlgebraicNormalization => AlgebraicNormalizePass.run(ir),
                IrPassKind::DependenceAnnotate(amplitude_dependencies) => DependenceAnnotatePass {
                    amplitude_dependencies: amplitude_dependencies.clone(),
                }
                .run(ir),
            }
        }
    }
}

struct DependenceAnnotatePass {
    amplitude_dependencies: Vec<DependenceClass>,
}

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

    fn run(&self, ir: &mut ExpressionIR) {
        ir.dependence_annotations =
            Self::compute_annotations(&ir.nodes, &self.amplitude_dependencies);
    }
}

#[cfg(test)]
mod tests {
    use num::complex::Complex64;

    use super::{compile_expression_ir, DependenceClass, ExpressionIR, ExpressionIrPipeline};
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
