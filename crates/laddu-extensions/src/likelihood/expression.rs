use std::{
    collections::HashMap,
    fmt::{Debug, Display},
    sync::Arc,
};

use auto_ops::*;
use laddu_core::{amplitudes::ParameterMap, LadduResult};
use nalgebra::DVector;
use parking_lot::RwLock;

use super::term::LikelihoodTerm;

#[derive(Debug)]
struct LikelihoodValues(Vec<f64>);

#[derive(Debug)]
struct LikelihoodGradients(Vec<DVector<f64>>);

#[derive(Clone, Default)]
enum LikelihoodNode {
    #[default]
    Zero,
    One,
    Term(usize),
    Add(Box<LikelihoodNode>, Box<LikelihoodNode>),
    Mul(Box<LikelihoodNode>, Box<LikelihoodNode>),
}

impl LikelihoodNode {
    fn remap(&self, mapping: &[usize]) -> Self {
        match self {
            Self::Term(idx) => Self::Term(mapping[*idx]),
            Self::Add(a, b) => Self::Add(Box::new(a.remap(mapping)), Box::new(b.remap(mapping))),
            Self::Mul(a, b) => Self::Mul(Box::new(a.remap(mapping)), Box::new(b.remap(mapping))),
            Self::Zero => Self::Zero,
            Self::One => Self::One,
        }
    }

    fn evaluate(&self, likelihood_values: &LikelihoodValues) -> f64 {
        match self {
            LikelihoodNode::Zero => 0.0,
            LikelihoodNode::One => 1.0,
            LikelihoodNode::Term(idx) => likelihood_values.0[*idx],
            LikelihoodNode::Add(a, b) => {
                a.evaluate(likelihood_values) + b.evaluate(likelihood_values)
            }
            LikelihoodNode::Mul(a, b) => {
                a.evaluate(likelihood_values) * b.evaluate(likelihood_values)
            }
        }
    }

    fn evaluate_gradient(
        &self,
        likelihood_values: &LikelihoodValues,
        likelihood_gradients: &LikelihoodGradients,
    ) -> DVector<f64> {
        match self {
            LikelihoodNode::Zero => DVector::zeros(0),
            LikelihoodNode::One => DVector::zeros(0),
            LikelihoodNode::Term(idx) => likelihood_gradients.0[*idx].clone(),
            LikelihoodNode::Add(a, b) => {
                a.evaluate_gradient(likelihood_values, likelihood_gradients)
                    + b.evaluate_gradient(likelihood_values, likelihood_gradients)
            }
            LikelihoodNode::Mul(a, b) => {
                a.evaluate_gradient(likelihood_values, likelihood_gradients)
                    * b.evaluate(likelihood_values)
                    + b.evaluate_gradient(likelihood_values, likelihood_gradients)
                        * a.evaluate(likelihood_values)
            }
        }
    }

    fn write_tree(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        parent_prefix: &str,
        immediate_prefix: &str,
        parent_suffix: &str,
    ) -> std::fmt::Result {
        let display_string = match self {
            Self::Zero => "0".to_string(),
            Self::One => "1".to_string(),
            Self::Term(idx) => format!("term({idx})"),
            Self::Add(_, _) => "+".to_string(),
            Self::Mul(_, _) => "*".to_string(),
        };
        writeln!(f, "{}{}{}", parent_prefix, immediate_prefix, display_string)?;
        match self {
            Self::Term(_) | Self::Zero | Self::One => {}
            Self::Add(a, b) | Self::Mul(a, b) => {
                let terms = [a, b];
                let mut it = terms.iter().peekable();
                let child_prefix = format!("{}{}", parent_prefix, parent_suffix);
                while let Some(child) = it.next() {
                    match it.peek() {
                        Some(_) => child.write_tree(f, &child_prefix, "├─ ", "│  ")?,
                        None => child.write_tree(f, &child_prefix, "└─ ", "   ")?,
                    }
                }
            }
        }
        Ok(())
    }
}

/// A combination of [`LikelihoodTerm`]s as well as sums and products of them.
///
/// # Notes
/// When multiple terms provide parameters with the same name, the term earliest in the expression
/// (or argument list) defines the fixed/free status and default value.
#[derive(Clone, Default)]
pub struct LikelihoodExpression {
    registry: LikelihoodRegistry,
    tree: LikelihoodNode,
}

impl LikelihoodExpression {
    /// Build a [`LikelihoodExpression`] from a single [`LikelihoodTerm`].
    pub fn from_term(term: Box<dyn LikelihoodTerm>) -> LadduResult<Self> {
        let registry = LikelihoodRegistry::singleton(term)?;
        Ok(Self {
            registry,
            tree: LikelihoodNode::Term(0),
        })
    }

    /// Create an expression representing zero, the additive identity.
    pub fn zero() -> Self {
        Self {
            registry: LikelihoodRegistry::default(),
            tree: LikelihoodNode::Zero,
        }
    }

    /// Create an expression representing one, the multiplicative identity.
    pub fn one() -> Self {
        Self {
            registry: LikelihoodRegistry::default(),
            tree: LikelihoodNode::One,
        }
    }

    fn binary_op(
        a: &LikelihoodExpression,
        b: &LikelihoodExpression,
        build: impl Fn(Box<LikelihoodNode>, Box<LikelihoodNode>) -> LikelihoodNode,
    ) -> LikelihoodExpression {
        let (registry, left_map, right_map) = a.registry.merge(&b.registry);
        let left_tree = a.tree.remap(&left_map);
        let right_tree = b.tree.remap(&right_map);
        LikelihoodExpression {
            registry,
            tree: build(Box::new(left_tree), Box::new(right_tree)),
        }
    }

    fn write_tree(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        parent_prefix: &str,
        immediate_prefix: &str,
        parent_suffix: &str,
    ) -> std::fmt::Result {
        self.tree
            .write_tree(f, parent_prefix, immediate_prefix, parent_suffix)
    }

    /// The parameter names referenced across all terms in this expression.
    pub fn parameters(&self) -> Vec<String> {
        self.registry.parameter_map.read().names()
    }

    /// The free parameter names which require user-provided values.
    pub fn free_parameters(&self) -> Vec<String> {
        self.registry.parameter_map.read().free().names()
    }

    /// The names of parameters with constant (fixed) values.
    pub fn fixed_parameters(&self) -> Vec<String> {
        self.registry.parameter_map.read().fixed().names()
    }

    /// Number of free parameters.
    pub fn n_free(&self) -> usize {
        self.registry.parameter_map.read().free().len()
    }

    /// Number of fixed parameters.
    pub fn n_fixed(&self) -> usize {
        self.registry.parameter_map.read().fixed().len()
    }

    /// Total number of parameters (free + fixed).
    pub fn n_parameters(&self) -> usize {
        self.registry.parameter_map.read().len()
    }

    fn assemble(&self, parameters: &[f64]) -> LadduResult<Vec<f64>> {
        Ok(self
            .registry
            .parameter_map
            .read()
            .assemble(parameters)?
            .values()
            .to_vec())
    }

    /// Evaluate the sum/product of all terms.
    pub fn evaluate(&self, parameters: &[f64]) -> LadduResult<f64> {
        let parameters = self.assemble(parameters)?;
        let mut param_buffers = self.registry.buffers();
        for (layout, buffer) in self
            .registry
            .param_layouts
            .iter()
            .zip(param_buffers.iter_mut())
        {
            for (buffer_idx, &param_idx) in layout.iter().enumerate() {
                buffer[buffer_idx] = parameters[param_idx];
            }
        }
        let likelihood_values = LikelihoodValues(
            self.registry
                .terms
                .iter()
                .zip(param_buffers.iter())
                .map(|(term, buffer)| term.evaluate(buffer))
                .collect::<LadduResult<Vec<_>>>()?,
        );
        Ok(self.tree.evaluate(&likelihood_values))
    }

    /// Evaluate the gradient.
    pub fn evaluate_gradient(&self, parameters: &[f64]) -> LadduResult<DVector<f64>> {
        let free_parameter_count = parameters.len();
        let free_parameter_indices = self.registry.parameter_map.read().free_parameter_indices();
        let parameters = self.assemble(parameters)?;
        let mut param_buffers = self.registry.buffers();
        for (layout, buffer) in self
            .registry
            .param_layouts
            .iter()
            .zip(param_buffers.iter_mut())
        {
            for (buffer_idx, &param_idx) in layout.iter().enumerate() {
                buffer[buffer_idx] = parameters[param_idx];
            }
        }
        let likelihood_values = LikelihoodValues(
            self.registry
                .terms
                .iter()
                .zip(param_buffers.iter())
                .map(|(term, buffer)| term.evaluate(buffer))
                .collect::<LadduResult<Vec<_>>>()?,
        );
        let mut gradient_buffers: Vec<DVector<f64>> = (0..self.registry.terms.len())
            .map(|_| DVector::zeros(parameters.len()))
            .collect();
        for (((term, param_buffer), gradient_buffer), layout) in self
            .registry
            .terms
            .iter()
            .zip(param_buffers.iter())
            .zip(gradient_buffers.iter_mut())
            .zip(self.registry.param_layouts.iter())
        {
            let term_gradient = term.evaluate_gradient(param_buffer)?; // This has a local layout
            for (term_idx, &buffer_idx) in layout.iter().enumerate() {
                gradient_buffer[buffer_idx] = term_gradient[term_idx] // This has a global layout
            }
        }
        let likelihood_gradients = LikelihoodGradients(gradient_buffers);
        let full_gradient = self
            .tree
            .evaluate_gradient(&likelihood_values, &likelihood_gradients);
        let mut reduced = DVector::zeros(free_parameter_count);
        for (out_idx, &global_idx) in free_parameter_indices.iter().enumerate() {
            reduced[out_idx] = full_gradient[global_idx];
        }
        Ok(reduced)
    }
}

impl LikelihoodTerm for LikelihoodExpression {
    fn evaluate(&self, parameters: &[f64]) -> LadduResult<f64> {
        LikelihoodExpression::evaluate(self, parameters)
    }
    fn evaluate_gradient(&self, parameters: &[f64]) -> LadduResult<DVector<f64>> {
        LikelihoodExpression::evaluate_gradient(self, parameters)
    }
    fn update(&self) {
        self.registry.terms.iter().for_each(|term| term.update())
    }
    fn fix_parameter(&self, name: &str, value: f64) -> LadduResult<()> {
        self.registry.fix_parameter(name, value)
    }

    fn free_parameter(&self, name: &str) -> LadduResult<()> {
        self.registry.free_parameter(name)
    }

    fn rename_parameter(&self, old: &str, new: &str) -> LadduResult<()> {
        self.registry.rename_parameter(old, new)
    }

    fn rename_parameters(&self, mapping: &HashMap<String, String>) -> LadduResult<()> {
        self.registry.rename_parameters(mapping)
    }

    fn parameter_map(&self) -> ParameterMap {
        self.registry.parameter_map.read().clone()
    }
}

impl Debug for LikelihoodExpression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.write_tree(f, "", "", "")
    }
}

impl Display for LikelihoodExpression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.write_tree(f, "", "", "")
    }
}

impl_op_ex!(+ |a: &LikelihoodExpression, b: &LikelihoodExpression| -> LikelihoodExpression {
    LikelihoodExpression::binary_op(a, b, LikelihoodNode::Add)
});
impl_op_ex!(
    *|a: &LikelihoodExpression, b: &LikelihoodExpression| -> LikelihoodExpression {
        LikelihoodExpression::binary_op(a, b, LikelihoodNode::Mul)
    }
);

#[derive(Clone, Default)]
struct LikelihoodRegistry {
    terms: Vec<Box<dyn LikelihoodTerm>>,
    param_layouts: Vec<Vec<usize>>,
    parameter_map: Arc<RwLock<ParameterMap>>,
}

impl LikelihoodRegistry {
    fn singleton(term: Box<dyn LikelihoodTerm>) -> LadduResult<Self> {
        let mut registry = Self::default();
        registry.push_term(term);
        Ok(registry)
    }

    fn buffers(&self) -> Vec<Vec<f64>> {
        self.param_layouts
            .iter()
            .map(|layout| vec![0.0; layout.len()])
            .collect()
    }

    fn push_term(&mut self, term: Box<dyn LikelihoodTerm>) -> usize {
        let term_idx = self.terms.len();
        let right_parameter_map = term.parameter_map();
        let (merged_parameter_map, left_indices, right_indices) = {
            let left_parameter_map = self.parameter_map.read();
            left_parameter_map.merge(&right_parameter_map)
        };
        *self.parameter_map.write() = merged_parameter_map;
        for layout in &mut self.param_layouts {
            for index in layout {
                *index = left_indices[*index];
            }
        }
        self.param_layouts.push(right_indices);
        self.terms.push(term);
        term_idx
    }

    fn merge(&self, other: &Self) -> (Self, Vec<usize>, Vec<usize>) {
        let mut registry = Self::default();
        let mut left_map = Vec::with_capacity(self.terms.len());
        for term in &self.terms {
            let idx = registry.push_term(dyn_clone::clone_box(&**term));
            left_map.push(idx);
        }
        let mut right_map = Vec::with_capacity(other.terms.len());
        for term in &other.terms {
            let idx = registry.push_term(dyn_clone::clone_box(&**term));
            right_map.push(idx);
        }
        (registry, left_map, right_map)
    }

    fn fix_parameter(&self, name: &str, value: f64) -> LadduResult<()> {
        self.parameter_map.read().fix_parameter(name, value)?;
        for term in &self.terms {
            if term.parameter_map().contains_key(name) {
                term.fix_parameter(name, value)?;
            }
        }
        Ok(())
    }

    fn free_parameter(&self, name: &str) -> LadduResult<()> {
        self.parameter_map.read().free_parameter(name)?;
        for term in &self.terms {
            if term.parameter_map().contains_key(name) {
                term.free_parameter(name)?;
            }
        }
        Ok(())
    }

    fn rename_parameter(&self, old: &str, new: &str) -> LadduResult<()> {
        self.parameter_map.write().rename_parameter(old, new)?;
        for term in &self.terms {
            if term.parameter_map().contains_key(old) {
                term.rename_parameter(old, new)?;
            }
        }
        Ok(())
    }

    fn rename_parameters(&self, mapping: &HashMap<String, String>) -> LadduResult<()> {
        self.parameter_map.write().rename_parameters(mapping)?;
        for term in &self.terms {
            for (old, new) in mapping.iter() {
                if term.parameter_map().contains_key(old) {
                    term.rename_parameter(old, new)?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "mpi")]
    use std::fs;
    use std::sync::Arc;

    use approx::assert_relative_eq;
    #[cfg(feature = "mpi")]
    use laddu_core::mpi::{finalize_mpi, get_world, use_mpi, LadduMPI};
    use laddu_core::{
        amplitudes::{Amplitude, AmplitudeID, ExpressionDependence, Parameter},
        data::{Dataset, DatasetMetadata, EventData},
        parameter,
        resources::{Cache, ParameterID, Parameters, Resources, ScalarID},
        vectors::Vec4,
        Expression, LadduError, LadduResult,
    };
    #[cfg(feature = "mpi")]
    use mpi::topology::{Communicator, SimpleCommunicator};
    #[cfg(feature = "mpi")]
    use mpi_test::mpi_test;
    use nalgebra::DVector;
    use num::complex::Complex64;
    use serde::{Deserialize, Serialize};

    use crate::likelihood::{LikelihoodScalar, LikelihoodTerm, NLL};

    const LENGTH_MISMATCH_MESSAGE_FRAGMENT: &str = "length mismatch";
    const AMPLITUDE_NOT_FOUND_MESSAGE_FRAGMENT: &str = "No registered amplitude";

    #[derive(Clone, Serialize, Deserialize)]
    struct ConstantAmplitude {
        name: String,
        parameter: Parameter,
        pid: ParameterID,
    }

    impl ConstantAmplitude {
        #[allow(clippy::new_ret_no_self)]
        fn new(name: &str, parameter: Parameter) -> LadduResult<Expression> {
            Self {
                name: name.to_string(),
                parameter,
                pid: ParameterID::default(),
            }
            .into_expression()
        }
    }

    #[typetag::serde]
    impl Amplitude for ConstantAmplitude {
        fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID> {
            self.pid = resources.register_parameter(&self.parameter)?;
            resources.register_amplitude(&self.name)
        }

        fn dependence_hint(&self) -> ExpressionDependence {
            ExpressionDependence::ParameterOnly
        }

        fn compute(&self, parameters: &Parameters, _cache: &Cache) -> Complex64 {
            Complex64::new(parameters.get(self.pid), 0.0)
        }

        fn compute_gradient(
            &self,
            parameters: &Parameters,
            _cache: &Cache,
            gradient: &mut DVector<Complex64>,
        ) {
            if let Some(index) = parameters.free_index(self.pid) {
                gradient[index] = Complex64::ONE;
            }
        }
    }

    #[derive(Clone, Serialize, Deserialize)]
    struct CachedBeamScaleAmplitude {
        name: String,
        parameter: Parameter,
        pid: ParameterID,
        sid: ScalarID,
        p4_index: usize,
    }

    impl CachedBeamScaleAmplitude {
        #[allow(clippy::new_ret_no_self)]
        fn new(name: &str, parameter: Parameter, p4_index: usize) -> LadduResult<Expression> {
            Self {
                name: name.to_string(),
                parameter,
                pid: ParameterID::default(),
                sid: ScalarID::default(),
                p4_index,
            }
            .into_expression()
        }
    }

    #[typetag::serde]
    impl Amplitude for CachedBeamScaleAmplitude {
        fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID> {
            self.pid = resources.register_parameter(&self.parameter)?;
            self.sid = resources.register_scalar(Some(&format!("{}.beam_energy", self.name)));
            resources.register_amplitude(&self.name)
        }

        fn dependence_hint(&self) -> ExpressionDependence {
            ExpressionDependence::Mixed
        }

        fn precompute(&self, event: &laddu_core::data::Event<'_>, cache: &mut Cache) {
            cache.store_scalar(self.sid, event.p4_at(self.p4_index).e());
        }

        fn compute(&self, parameters: &Parameters, cache: &Cache) -> Complex64 {
            Complex64::new(parameters.get(self.pid), 0.0) * cache.get_scalar(self.sid)
        }

        fn compute_gradient(
            &self,
            parameters: &Parameters,
            cache: &Cache,
            gradient: &mut DVector<Complex64>,
        ) {
            if let Some(index) = parameters.free_index(self.pid) {
                gradient[index] = Complex64::new(cache.get_scalar(self.sid), 0.0);
            }
        }
    }

    #[derive(Clone, Serialize, Deserialize)]
    struct CacheOnlyBeamAmplitude {
        name: String,
        sid: ScalarID,
        p4_index: usize,
    }

    impl CacheOnlyBeamAmplitude {
        #[allow(clippy::new_ret_no_self)]
        fn new(name: &str, p4_index: usize) -> LadduResult<Expression> {
            Self {
                name: name.to_string(),
                sid: ScalarID::default(),
                p4_index,
            }
            .into_expression()
        }
    }

    #[typetag::serde]
    impl Amplitude for CacheOnlyBeamAmplitude {
        fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID> {
            self.sid = resources.register_scalar(Some(&format!("{}.beam_energy", self.name)));
            resources.register_amplitude(&self.name)
        }

        fn dependence_hint(&self) -> ExpressionDependence {
            ExpressionDependence::CacheOnly
        }

        fn precompute(&self, event: &laddu_core::data::Event<'_>, cache: &mut Cache) {
            cache.store_scalar(self.sid, event.p4_at(self.p4_index).e());
        }

        fn compute(&self, _parameters: &Parameters, cache: &Cache) -> Complex64 {
            Complex64::new(cache.get_scalar(self.sid), 0.0)
        }
    }

    fn dataset_with_weights(weights: &[f64]) -> Arc<Dataset> {
        let metadata = Arc::new(DatasetMetadata::default());
        let events = weights
            .iter()
            .map(|&weight| {
                Arc::new(EventData {
                    p4s: vec![Vec4::new(0.0, 0.0, 0.0, 1.0)],
                    aux: vec![],
                    weight,
                })
            })
            .collect();
        Arc::new(Dataset::new_with_metadata(events, metadata))
    }

    fn dataset_with_two_p4_and_weights(
        beam_energies: &[(f64, f64)],
        weights: &[f64],
    ) -> Arc<Dataset> {
        assert_eq!(beam_energies.len(), weights.len());
        let metadata = Arc::new(DatasetMetadata::default());
        let events = beam_energies
            .iter()
            .zip(weights.iter())
            .map(|(&(e0, e1), &weight)| {
                Arc::new(EventData {
                    p4s: vec![Vec4::new(0.0, 0.0, 0.0, e0), Vec4::new(0.0, 0.0, 0.0, e1)],
                    aux: vec![],
                    weight,
                })
            })
            .collect();
        Arc::new(Dataset::new_with_metadata(events, metadata))
    }

    #[cfg(feature = "mpi")]
    fn read_resident_rss_kb() -> Option<u64> {
        #[cfg(target_os = "linux")]
        {
            let status = fs::read_to_string("/proc/self/status").ok()?;
            let vm_rss = status
                .lines()
                .find(|line| line.starts_with("VmRSS:"))?
                .split_whitespace()
                .nth(1)?;
            vm_rss.parse::<u64>().ok()
        }

        #[cfg(not(target_os = "linux"))]
        {
            None
        }
    }

    #[cfg(feature = "mpi")]
    fn generated_two_p4_dataset(
        n_events: usize,
        base_energy: f64,
        weight_scale: f64,
    ) -> Arc<Dataset> {
        let metadata = Arc::new(DatasetMetadata::default());
        let events = (0..n_events)
            .map(|index| {
                let idx = index as f64;
                let beam_e0 = base_energy + (idx % 17.0) * 0.35 + idx * 0.0025;
                let beam_e1 = 0.5 * base_energy + (idx % 11.0) * 0.2 + idx * 0.0015;
                let weight = 0.75 + weight_scale * (1.0 + (index % 9) as f64);
                Arc::new(EventData {
                    p4s: vec![
                        Vec4::new(0.0, 0.0, 0.0, beam_e0),
                        Vec4::new(0.0, 0.0, 0.0, beam_e1),
                    ],
                    aux: vec![],
                    weight,
                })
            })
            .collect();
        Arc::new(Dataset::new_with_metadata(events, metadata))
    }

    fn make_constant_nll() -> (Box<NLL>, Vec<f64>) {
        let amp = ConstantAmplitude::new("amp", parameter!("scale")).unwrap();
        let expr = amp.norm_sqr();
        let data = dataset_with_weights(&[1.0, 2.0]);
        let mc = dataset_with_weights(&[0.5, 1.5]);
        let nll = NLL::new(&expr, &data, &mc, None).unwrap();
        (nll, vec![2.0])
    }

    fn make_two_parameter_nll() -> (Box<NLL>, Vec<f64>) {
        let amp_a = ConstantAmplitude::new("amp_a", parameter!("alpha")).unwrap();
        let amp_b = ConstantAmplitude::new("amp_b", parameter!("beta")).unwrap();
        let expr = (amp_a + amp_b).norm_sqr();
        let data = dataset_with_weights(&[1.0, 2.0, 3.0, 1.0]);
        let mc = dataset_with_weights(&[0.5, 1.5, 2.5, 0.5]);
        let nll = NLL::new(&expr, &data, &mc, None).unwrap();
        (nll, vec![0.75, -1.25])
    }

    #[test]
    fn nll_exposes_expression_and_current_compiled_expression() {
        let (nll, _) = make_two_parameter_nll();

        let expression_display = nll.expression().compiled_expression().to_string();
        assert!(expression_display.contains("amp_a(id=0)"));
        assert!(expression_display.contains("amp_b(id=1)"));

        nll.deactivate("amp_b");
        let compiled = nll.compiled_expression().to_string();
        assert!(compiled.contains("amp_a(id=0)"));
        assert!(!compiled.contains("amp_b(id=1)"));
        assert!(compiled.contains("const 0"));
    }

    #[test]
    fn stochastic_nll_exposes_expression_and_current_compiled_expression() {
        let (nll, _) = make_two_parameter_nll();
        let stochastic = nll
            .to_stochastic(2, Some(0))
            .expect("stochastic NLL should build");

        assert!(stochastic
            .expression()
            .compiled_expression()
            .to_string()
            .contains("amp_a(id=0)"));
        assert!(stochastic
            .compiled_expression()
            .to_string()
            .contains("amp_b(id=1)"));
    }

    #[derive(Clone, Copy)]
    enum DeterministicModelKind {
        Separable,
        Partial,
        NonSeparable,
    }

    struct DeterministicNllFixture {
        nll: Box<NLL>,
        parameters: Vec<f64>,
    }

    const DETERMINISTIC_STRICT_ABS_TOL: f64 = 1e-12;
    const DETERMINISTIC_STRICT_REL_TOL: f64 = 1e-10;

    fn assert_nll_fixture_matches_weighted_baseline(fixture: &DeterministicNllFixture) {
        let expected_value = crate::likelihood::nll::evaluate_weighted_expression_sum_local(
            &fixture.nll.data_evaluator,
            &fixture.parameters,
            |l| f64::ln(l.re),
        )
        .expect("evaluate should succeed");
        let expected_mc_term = fixture
            .nll
            .accmc_evaluator
            .evaluate_weighted_value_sum_local(&fixture.parameters)
            .expect("evaluate should succeed");
        let expected_value = -2.0 * (expected_value - expected_mc_term / fixture.nll.n_mc);

        let expected_data_gradient = fixture
            .nll
            .evaluate_data_gradient_term_local(&fixture.parameters)
            .expect("evaluate should succeed");
        let expected_mc_gradient = fixture
            .nll
            .accmc_evaluator
            .evaluate_weighted_gradient_sum_local(&fixture.parameters)
            .expect("evaluate should succeed");
        let expected_gradient =
            -2.0 * (expected_data_gradient - expected_mc_gradient / fixture.nll.n_mc);

        let actual_value = fixture
            .nll
            .evaluate_local(&fixture.parameters)
            .expect("evaluate should succeed");
        assert_relative_eq!(
            actual_value,
            expected_value,
            epsilon = DETERMINISTIC_STRICT_ABS_TOL,
            max_relative = DETERMINISTIC_STRICT_REL_TOL
        );

        let actual_gradient = fixture
            .nll
            .evaluate_gradient_local(&fixture.parameters)
            .expect("evaluate should succeed");
        assert_eq!(
            actual_gradient.len(),
            expected_gradient.len(),
            "fixture NLL gradient length mismatch (actual={}, expected={})",
            actual_gradient.len(),
            expected_gradient.len()
        );
        for (actual_item, expected_item) in actual_gradient.iter().zip(expected_gradient.iter()) {
            assert_relative_eq!(
                *actual_item,
                *expected_item,
                epsilon = DETERMINISTIC_STRICT_ABS_TOL,
                max_relative = DETERMINISTIC_STRICT_REL_TOL
            );
        }
    }

    #[cfg(feature = "mpi")]
    fn assert_nll_fixture_matches_mpi_reduced_baseline(
        fixture: &DeterministicNllFixture,
        world: &SimpleCommunicator,
    ) {
        let data_term_local = crate::likelihood::nll::evaluate_weighted_expression_sum_local(
            &fixture.nll.data_evaluator,
            &fixture.parameters,
            |l| f64::ln(l.re),
        )
        .expect("evaluate should succeed");
        let mc_term_local = fixture
            .nll
            .accmc_evaluator
            .evaluate_weighted_value_sum_local(&fixture.parameters)
            .expect("evaluate should succeed");
        let data_term = crate::likelihood::nll::reduce_scalar(world, data_term_local);
        let mc_term = crate::likelihood::nll::reduce_scalar(world, mc_term_local);
        let expected_value = -2.0 * (data_term - mc_term / fixture.nll.n_mc);
        let mpi_value = fixture
            .nll
            .evaluate_mpi(&fixture.parameters, world)
            .expect("evaluate should succeed");
        assert_relative_eq!(
            mpi_value,
            expected_value,
            epsilon = DETERMINISTIC_STRICT_ABS_TOL,
            max_relative = DETERMINISTIC_STRICT_REL_TOL
        );

        let data_gradient_local = fixture
            .nll
            .evaluate_data_gradient_term_local(&fixture.parameters)
            .expect("evaluate should succeed");
        let mc_gradient_local = fixture
            .nll
            .accmc_evaluator
            .evaluate_weighted_gradient_sum_local(&fixture.parameters)
            .expect("evaluate should succeed");
        let data_gradient = crate::likelihood::nll::reduce_gradient(world, &data_gradient_local);
        let mc_gradient = crate::likelihood::nll::reduce_gradient(world, &mc_gradient_local);
        let expected_gradient = -2.0 * (data_gradient - mc_gradient / fixture.nll.n_mc);
        let mpi_gradient = fixture
            .nll
            .evaluate_gradient_mpi(&fixture.parameters, world)
            .expect("evaluate should succeed");
        assert_eq!(
            mpi_gradient.len(),
            expected_gradient.len(),
            "fixture MPI gradient length mismatch (actual={}, expected={})",
            mpi_gradient.len(),
            expected_gradient.len()
        );
        for (actual_item, expected_item) in mpi_gradient.iter().zip(expected_gradient.iter()) {
            assert_relative_eq!(
                *actual_item,
                *expected_item,
                epsilon = DETERMINISTIC_STRICT_ABS_TOL,
                max_relative = DETERMINISTIC_STRICT_REL_TOL
            );
        }
    }

    fn make_deterministic_nll_fixture(kind: DeterministicModelKind) -> DeterministicNllFixture {
        let data = dataset_with_two_p4_and_weights(
            &[
                (1.0, 0.8),
                (2.5, 1.7),
                (4.0, 2.4),
                (3.3, 1.1),
                (5.2, 2.8),
                (1.7, 0.9),
            ],
            &[0.7, 1.2, 0.9, 1.5, 0.8, 1.1],
        );
        let mc = dataset_with_two_p4_and_weights(
            &[
                (1.5, 1.0),
                (3.0, 2.1),
                (5.5, 2.9),
                (2.0, 1.2),
                (4.2, 1.8),
                (2.8, 1.4),
            ],
            &[0.8, 1.4, 0.6, 1.1, 0.75, 1.25],
        );

        match kind {
            DeterministicModelKind::Separable => {
                let p1 = ConstantAmplitude::new("p1", parameter!("p1"))
                    .expect("separable p1 should build");
                let p2 = ConstantAmplitude::new("p2", parameter!("p2"))
                    .expect("separable p2 should build");
                let c1 = CacheOnlyBeamAmplitude::new("c1", 0).expect("separable c1 should build");
                let c2 = CacheOnlyBeamAmplitude::new("c2", 1).expect("separable c2 should build");
                let expression = (&p1 * &c1) + &(&p2 * &c2);
                DeterministicNllFixture {
                    nll: NLL::new(&expression, &data, &mc, None)
                        .expect("separable NLL should build"),
                    parameters: vec![0.4, 0.2],
                }
            }
            DeterministicModelKind::Partial => {
                let p =
                    ConstantAmplitude::new("p", parameter!("p")).expect("partial p should build");
                let c = CacheOnlyBeamAmplitude::new("c", 0).expect("partial c should build");
                let m = CachedBeamScaleAmplitude::new("m", parameter!("m"), 1)
                    .expect("partial m should build");
                let expression = (&p * &c) + &m;
                DeterministicNllFixture {
                    nll: NLL::new(&expression, &data, &mc, None).expect("partial NLL should build"),
                    parameters: vec![0.35, 0.25],
                }
            }
            DeterministicModelKind::NonSeparable => {
                let m1 = CachedBeamScaleAmplitude::new("m1", parameter!("m1"), 0)
                    .expect("non-separable m1 should build");
                let m2 = CachedBeamScaleAmplitude::new("m2", parameter!("m2"), 1)
                    .expect("non-separable m2 should build");
                let expression = &m1 * &m2;
                DeterministicNllFixture {
                    nll: NLL::new(&expression, &data, &mc, None)
                        .expect("non-separable NLL should build"),
                    parameters: vec![0.2, 0.15],
                }
            }
        }
    }

    #[cfg(feature = "mpi")]
    fn make_mixed_workload_nll_fixture(n_events: usize) -> DeterministicNllFixture {
        let data = generated_two_p4_dataset(n_events, 1.4, 0.08);
        let mc = generated_two_p4_dataset(n_events, 1.9, 0.11);
        let p =
            ConstantAmplitude::new("p", parameter!("p")).expect("mixed-workload p should build");
        let c = CacheOnlyBeamAmplitude::new("c", 0)
            .expect("mixed-workload cache amplitude should build");
        let m = CachedBeamScaleAmplitude::new("m", parameter!("m"), 1)
            .expect("mixed-workload beam amplitude should build");
        let expression = (&p * &c) + &m;
        DeterministicNllFixture {
            nll: NLL::new(&expression, &data, &mc, None).expect("mixed-workload NLL should build"),
            parameters: vec![0.35, 0.25],
        }
    }

    fn case_nll_evaluate_short(nll: &NLL) -> LadduResult<()> {
        nll.evaluate(&[]).map(|_| ())
    }

    fn case_nll_evaluate_gradient_long(nll: &NLL) -> LadduResult<()> {
        nll.evaluate_gradient(&[1.0, 2.0]).map(|_| ())
    }

    fn case_nll_project_short(nll: &NLL) -> LadduResult<()> {
        nll.project_weights(&[], None).map(|_| ())
    }

    fn case_nll_project_weights_and_gradients_long(nll: &NLL) -> LadduResult<()> {
        nll.project_weights_and_gradients(&[1.0, 2.0], None)
            .map(|_| ())
    }

    fn case_nll_project_weights_subset_short(nll: &NLL) -> LadduResult<()> {
        nll.project_weights_subset_local::<&str>(&[], &["missing_amplitude"], None)
            .map(|_| ())
    }

    fn case_nll_project_weights_and_gradients_subset_long(nll: &NLL) -> LadduResult<()> {
        nll.project_weights_and_gradients_subset_local::<&str>(
            &[1.0, 2.0],
            &["missing_amplitude"],
            None,
        )
        .map(|_| ())
    }

    fn case_likelihood_evaluate_short() -> LadduResult<()> {
        let alpha = LikelihoodScalar::new("alpha")?;
        alpha.evaluate(&[]).map(|_| ())
    }

    fn case_likelihood_gradient_long() -> LadduResult<()> {
        let alpha = LikelihoodScalar::new("alpha")?;
        alpha.evaluate_gradient(&[1.0, 2.0]).map(|_| ())
    }

    #[test]
    fn table_driven_length_mismatch_errors() {
        let (nll, _) = make_constant_nll();
        let cases: [(&str, LadduResult<()>); 8] = [
            ("nll.evaluate short", case_nll_evaluate_short(nll.as_ref())),
            (
                "nll.evaluate_gradient long",
                case_nll_evaluate_gradient_long(nll.as_ref()),
            ),
            (
                "nll.project_weights short",
                case_nll_project_short(nll.as_ref()),
            ),
            (
                "nll.project_weights_and_gradients long",
                case_nll_project_weights_and_gradients_long(nll.as_ref()),
            ),
            (
                "nll.project_weights_subset short",
                case_nll_project_weights_subset_short(nll.as_ref()),
            ),
            (
                "nll.project_weights_and_gradients_subset long",
                case_nll_project_weights_and_gradients_subset_long(nll.as_ref()),
            ),
            (
                "likelihood.evaluate short",
                case_likelihood_evaluate_short(),
            ),
            (
                "likelihood.evaluate_gradient long",
                case_likelihood_gradient_long(),
            ),
        ];
        for (label, result) in cases {
            let err = result.unwrap_err();
            assert!(
                matches!(err, LadduError::LengthMismatch { .. }),
                "expected LengthMismatch for {label}, got {err:?}"
            );
            assert!(
                err.to_string().contains(LENGTH_MISMATCH_MESSAGE_FRAGMENT),
                "expected message containing \"{LENGTH_MISMATCH_MESSAGE_FRAGMENT}\" for {label}, got {}",
                err
            );
        }
    }

    #[test]
    fn table_driven_unknown_amplitude_errors() {
        let (nll, params) = make_constant_nll();
        let cases: [(&str, LadduResult<()>); 4] = [
            (
                "activate_strict unknown",
                nll.activate_strict("missing_amplitude"),
            ),
            (
                "isolate_strict unknown",
                nll.isolate_strict("missing_amplitude"),
            ),
            (
                "project_weights_subset unknown",
                nll.project_weights_subset_local_strict::<&str>(
                    &params,
                    &["missing_amplitude"],
                    None,
                )
                .map(|_| ()),
            ),
            (
                "project_weights_and_gradients_subset unknown",
                nll.project_weights_and_gradients_subset_local_strict::<&str>(
                    &params,
                    &["missing_amplitude"],
                    None,
                )
                .map(|_| ()),
            ),
        ];
        for (label, result) in cases {
            let err = result.unwrap_err();
            assert!(
                matches!(err, LadduError::AmplitudeNotFoundError { .. }),
                "expected AmplitudeNotFoundError for {label}, got {err:?}"
            );
            assert!(
                err.to_string()
                    .contains(AMPLITUDE_NOT_FOUND_MESSAGE_FRAGMENT),
                "expected message containing \"{AMPLITUDE_NOT_FOUND_MESSAGE_FRAGMENT}\" for {label}, got {}",
                err
            );
        }
    }

    #[test]
    fn likelihood_expression_evaluates_scalar_sum() {
        let alpha = LikelihoodScalar::new("alpha").unwrap();
        let beta = LikelihoodScalar::new("beta").unwrap();
        let expr = &alpha + &beta;
        assert_eq!(expr.parameters(), vec!["alpha", "beta"]);
        let params = vec![2.0, 3.0];
        assert_relative_eq!(expr.evaluate(&params).unwrap(), 5.0);
        let grad = expr.evaluate_gradient(&params).unwrap();
        assert_relative_eq!(grad[0], 1.0);
        assert_relative_eq!(grad[1], 1.0);
    }

    #[test]
    fn likelihood_expression_evaluates_scalar_product() {
        let alpha = LikelihoodScalar::new("alpha").unwrap();
        let beta = LikelihoodScalar::new("beta").unwrap();
        let expr = &alpha * &beta;
        let params = vec![2.0, 3.0];
        assert_relative_eq!(expr.evaluate(&params).unwrap(), 6.0);
        let grad = expr.evaluate_gradient(&params).unwrap();
        assert_relative_eq!(grad[0], 3.0);
        assert_relative_eq!(grad[1], 2.0);
    }

    #[test]
    fn likelihood_expression_tracks_fixed_parameters() {
        let alpha = LikelihoodScalar::new("alpha").unwrap();
        let beta = LikelihoodScalar::new("beta").unwrap();
        let expr = &alpha + &beta;
        expr.fix_parameter("alpha", 1.5).unwrap();
        assert_eq!(expr.parameters(), vec!["alpha", "beta"]);
        assert_eq!(expr.free_parameters(), vec!["beta"]);
        assert_eq!(expr.fixed_parameters(), vec!["alpha"]);
        let params_free = vec![2.0];
        assert_relative_eq!(expr.evaluate(&params_free).unwrap(), 3.5);
        let grad_free = expr.evaluate_gradient(&params_free).unwrap();
        assert_eq!(grad_free.len(), 1);
        assert_relative_eq!(grad_free[0], 1.0);
    }

    #[test]
    fn likelihood_expression_handles_term_local_fixed_parameters() {
        let alpha = LikelihoodScalar::new("alpha").unwrap();
        alpha.fix_parameter("alpha", 1.5).unwrap();
        let beta = LikelihoodScalar::new("beta").unwrap();
        let expr = &alpha + &beta;
        assert_eq!(expr.parameters(), vec!["alpha", "beta"]);
        assert_eq!(expr.free_parameters(), vec!["beta"]);
        assert_eq!(expr.fixed_parameters(), vec!["alpha"]);

        let params_free = vec![2.0];
        assert_relative_eq!(expr.evaluate(&params_free).unwrap(), 3.5);
        let grad_free = expr.evaluate_gradient(&params_free).unwrap();
        assert_eq!(grad_free.len(), 1);
        assert_relative_eq!(grad_free[0], 1.0);
    }

    #[test]
    fn likelihood_product_handles_term_local_fixed_parameters() {
        let alpha = LikelihoodScalar::new("alpha").unwrap();
        alpha.fix_parameter("alpha", 1.5).unwrap();
        let beta = LikelihoodScalar::new("beta").unwrap();
        let expr = &alpha * &beta;
        assert_eq!(expr.parameters(), vec!["alpha", "beta"]);
        assert_eq!(expr.free_parameters(), vec!["beta"]);
        assert_eq!(expr.fixed_parameters(), vec!["alpha"]);

        let params_free = vec![2.0];
        assert_relative_eq!(expr.evaluate(&params_free).unwrap(), 3.0);
        let grad_free = expr.evaluate_gradient(&params_free).unwrap();
        assert_eq!(grad_free.len(), 1);
        assert_relative_eq!(grad_free[0], 1.5);
    }

    #[test]
    fn nll_evaluate_and_gradient_match_closed_form() {
        let (nll, params) = make_constant_nll();
        let intensity = params[0] * params[0];
        let weight_sum = 3.0;
        let expected = -2.0 * (weight_sum * intensity.ln() - intensity);
        assert_relative_eq!(nll.evaluate(&params).unwrap(), expected, epsilon = 1e-12);
        let grad = nll.evaluate_gradient(&params).unwrap();
        let expected_grad = -4.0 * (weight_sum / params[0] - params[0]);
        assert_relative_eq!(grad[0], expected_grad, epsilon = 1e-12);
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn gradient_scratch_reuse_is_thread_safe_across_parallel_calls() {
        let (nll_single, params_single) = make_constant_nll();
        let (nll_multi, params_multi) = make_two_parameter_nll();
        let nll_single = Arc::new(*nll_single);
        let nll_multi = Arc::new(*nll_multi);
        let expected_single = nll_single
            .evaluate_gradient(&params_single)
            .expect("single-parameter gradient should evaluate");
        let expected_multi = nll_multi
            .evaluate_gradient(&params_multi)
            .expect("two-parameter gradient should evaluate");
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let nll_single = Arc::clone(&nll_single);
                let nll_multi = Arc::clone(&nll_multi);
                let params_single = params_single.clone();
                let params_multi = params_multi.clone();
                let expected_single = expected_single.clone();
                let expected_multi = expected_multi.clone();
                scope.spawn(move || {
                    for _ in 0..100 {
                        let single_gradient = nll_single
                            .evaluate_gradient(&params_single)
                            .expect("single-parameter gradient should evaluate");
                        assert_relative_eq!(
                            single_gradient[0],
                            expected_single[0],
                            epsilon = 1e-12
                        );
                        let multi_gradient = nll_multi
                            .evaluate_gradient(&params_multi)
                            .expect("two-parameter gradient should evaluate");
                        assert_eq!(multi_gradient.len(), expected_multi.len());
                        for index in 0..expected_multi.len() {
                            assert_relative_eq!(
                                multi_gradient[index],
                                expected_multi[index],
                                epsilon = 1e-12
                            );
                        }
                    }
                });
            }
        });
    }

    #[test]
    fn nll_value_matches_mixed_scale_weighted_closed_form() {
        let amp = ConstantAmplitude::new("amp", parameter!("scale")).unwrap();
        let expr = amp.norm_sqr();
        let data = dataset_with_weights(&[1.0e12, 1.0e-12, 3.5, 7.25e4, 2.0e-3]);
        let mc = dataset_with_weights(&[4.0e9, 9.0e-6, 1.25, 2.5e2, 8.0e-4]);
        let nll = NLL::new(&expr, &data, &mc, None).unwrap();
        let params = vec![1.125];

        let intensity: f64 = params[0] * params[0];
        let data_weight_sum = data.weights_local().iter().copied().sum::<f64>();
        let mc_weight_sum = mc.weights_local().iter().copied().sum::<f64>();
        let n_mc = mc.n_events_weighted();
        let expected = -2.0 * (data_weight_sum * intensity.ln() - mc_weight_sum * intensity / n_mc);

        let value = nll.evaluate(&params).unwrap();
        assert_relative_eq!(value, expected, epsilon = 1e-9, max_relative = 1e-12);
    }

    #[test]
    fn nll_evaluate_and_gradient_match_hardcoded_weighted_reference() {
        let amp_a = CachedBeamScaleAmplitude::new("amp_a", parameter!("alpha"), 0).unwrap();
        let amp_b = CachedBeamScaleAmplitude::new("amp_b", parameter!("beta"), 1).unwrap();
        let expr = (&amp_a + &amp_b).norm_sqr();
        let data = dataset_with_two_p4_and_weights(
            &[(1.0, 0.8), (2.5, 1.7), (4.0, 2.4), (3.3, 1.1)],
            &[0.7, 1.2, 0.9, 1.5],
        );
        let mc = dataset_with_two_p4_and_weights(
            &[(1.5, 1.0), (3.0, 2.1), (5.5, 2.9), (2.0, 1.2), (4.2, 1.8)],
            &[0.8, 1.4, 0.6, 1.1, 0.75],
        );
        let nll = NLL::new(&expr, &data, &mc, None).unwrap();
        let params = vec![0.6, 1.1];
        assert_eq!(nll.free_parameters(), vec!["alpha", "beta"]);

        let value = nll.evaluate(&params).unwrap();
        assert_relative_eq!(value, 12.242296380697244, epsilon = 1e-12);

        let gradient = nll.evaluate_gradient(&params).unwrap();
        assert_eq!(gradient.len(), 2);
        assert_relative_eq!(gradient[0], 37.78259267741666, epsilon = 1e-12);
        assert_relative_eq!(gradient[1], 21.8538272590435, epsilon = 1e-12);
    }

    #[test]
    fn nll_deterministic_fixtures_cover_separable_partial_and_non_separable_models() {
        let separable = make_deterministic_nll_fixture(DeterministicModelKind::Separable);
        let partial = make_deterministic_nll_fixture(DeterministicModelKind::Partial);
        let non_separable = make_deterministic_nll_fixture(DeterministicModelKind::NonSeparable);

        for fixture in [separable, partial, non_separable] {
            assert_nll_fixture_matches_weighted_baseline(&fixture);
        }
    }

    #[test]
    fn nll_deterministic_fixture_matches_baseline_across_activation_toggles() {
        let fixture = make_deterministic_nll_fixture(DeterministicModelKind::Partial);
        assert_nll_fixture_matches_weighted_baseline(&fixture);

        fixture.nll.isolate_many(&["p", "c"]);
        assert_nll_fixture_matches_weighted_baseline(&fixture);

        fixture.nll.activate_all();
        assert_nll_fixture_matches_weighted_baseline(&fixture);
    }

    #[test]
    fn nll_project_returns_weighted_intensity() {
        let (nll, params) = make_constant_nll();
        let projection = nll.project_weights_local(&params, None).unwrap();
        assert_relative_eq!(projection[0], 1.0, epsilon = 1e-12);
        assert_relative_eq!(projection[1], 3.0, epsilon = 1e-12);
    }

    #[test]
    fn nll_project_reports_structured_length_error() {
        let (nll, _) = make_constant_nll();
        let err = nll.project_weights(&[], None).unwrap_err();
        assert!(matches!(
            err,
            LadduError::LengthMismatch {
                expected: 1,
                actual: 0,
                ..
            }
        ));
    }

    #[test]
    fn nll_project_weights_subset_reports_structured_missing_amplitude_error() {
        let (nll, params) = make_constant_nll();
        let err = nll
            .project_weights_subset_local_strict::<&str>(&params, &["missing_amplitude"], None)
            .unwrap_err();
        assert!(matches!(err, LadduError::AmplitudeNotFoundError { .. }));
    }

    #[test]
    fn nll_project_weights_subsets_matches_repeated_project_weights_subset_calls() {
        let (nll, params) = make_two_parameter_nll();
        let subsets = vec![
            vec!["amp_a".to_string()],
            vec!["amp_b".to_string()],
            vec!["amp_a".to_string(), "amp_b".to_string()],
        ];
        let batched = nll
            .project_weights_subsets_local(&params, &subsets, None)
            .expect("batched projection should evaluate");
        let repeated = subsets
            .iter()
            .map(|subset| {
                nll.project_weights_subset_local(&params, subset, None)
                    .expect("single subset projection should evaluate")
            })
            .collect::<Vec<_>>();
        assert_eq!(batched.len(), repeated.len());
        for (lhs, rhs) in batched.iter().zip(repeated.iter()) {
            assert_eq!(lhs.len(), rhs.len());
            for (lhs_value, rhs_value) in lhs.iter().zip(rhs.iter()) {
                assert_relative_eq!(lhs_value, rhs_value, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn nll_project_weights_subsets_handles_empty_and_duplicate_subsets() {
        let (nll, params) = make_two_parameter_nll();
        let empty: Vec<Vec<String>> = Vec::new();
        let empty_projection = nll
            .project_weights_subsets_local(&params, &empty, None)
            .expect("empty subset list should evaluate");
        assert!(empty_projection.is_empty());

        let subsets = vec![
            vec!["amp_b".to_string()],
            vec!["amp_a".to_string()],
            vec!["amp_a".to_string(), "amp_b".to_string()],
            vec!["amp_a".to_string()],
            vec!["amp_b".to_string()],
        ];
        let batched = nll
            .project_weights_subsets_local(&params, &subsets, None)
            .expect("batched projection should evaluate");
        let repeated = subsets
            .iter()
            .map(|subset| {
                nll.project_weights_subset_local(&params, subset, None)
                    .expect("single subset projection should evaluate")
            })
            .collect::<Vec<_>>();
        assert_eq!(batched.len(), repeated.len());
        for (lhs, rhs) in batched.iter().zip(repeated.iter()) {
            assert_eq!(lhs.len(), rhs.len());
            for (lhs_value, rhs_value) in lhs.iter().zip(rhs.iter()) {
                assert_relative_eq!(lhs_value, rhs_value, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn nll_project_weights_subsets_reports_missing_amplitude_error() {
        let (nll, params) = make_two_parameter_nll();
        let subsets = vec![vec!["amp_a".to_string()], vec!["missing".to_string()]];
        let err = nll
            .project_weights_subsets_local_strict(&params, &subsets, None)
            .expect_err("missing amplitude should fail");
        assert!(matches!(err, LadduError::AmplitudeNotFoundError { .. }));
    }

    #[test]
    fn nll_project_weights_and_gradients_subset_matches_repeated_calls() {
        let (nll, params) = make_two_parameter_nll();
        let subsets = vec![
            vec!["amp_b".to_string()],
            vec!["amp_a".to_string()],
            vec!["amp_a".to_string(), "amp_b".to_string()],
            vec!["amp_a".to_string()],
        ];
        for subset in subsets {
            let (weights_local, gradients_local) = nll
                .project_weights_and_gradients_subset_local(&params, &subset, None)
                .expect("local gradient projection should evaluate");
            let (weights_auto, gradients_auto) = nll
                .project_weights_and_gradients_subset(&params, &subset, None)
                .expect("auto gradient projection should evaluate");
            assert_eq!(weights_local.len(), weights_auto.len());
            assert_eq!(gradients_local.len(), gradients_auto.len());
            for (lhs, rhs) in weights_local.iter().zip(weights_auto.iter()) {
                assert_relative_eq!(lhs, rhs, epsilon = 1e-12);
            }
            for (lhs, rhs) in gradients_local.iter().zip(gradients_auto.iter()) {
                assert_relative_eq!(lhs, rhs, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn nll_activation_changes_invalidate_projection_mask_cache() {
        let (nll, params) = make_constant_nll();
        assert!(nll.projection_active_mask_cache.lock().is_empty());

        let _ = nll
            .project_weights_subset_local::<&str>(&params, &["amp"], None)
            .unwrap();
        assert!(!nll.projection_active_mask_cache.lock().is_empty());

        nll.deactivate("amp");
        assert!(nll.projection_active_mask_cache.lock().is_empty());

        let projection = nll
            .project_weights_subset_local::<&str>(&params, &["amp"], None)
            .unwrap();
        assert_relative_eq!(projection[0], 1.0, epsilon = 1e-12);
        assert_relative_eq!(projection[1], 3.0, epsilon = 1e-12);
    }

    #[test]
    fn nll_project_weights_subset_validates_length_before_isolation() {
        let (nll, _) = make_constant_nll();
        let err = nll
            .project_weights_subset_local::<&str>(&[], &["missing_amplitude"], None)
            .unwrap_err();
        assert!(matches!(
            err,
            LadduError::LengthMismatch {
                expected: 1,
                actual: 0,
                ..
            }
        ));
    }

    #[test]
    fn nll_project_weights_and_gradients_subset_validates_length_before_isolation() {
        let (nll, _) = make_constant_nll();
        let err = nll
            .project_weights_and_gradients_subset_local::<&str>(
                &[1.0, 2.0],
                &["missing_amplitude"],
                None,
            )
            .unwrap_err();
        assert!(matches!(
            err,
            LadduError::LengthMismatch {
                expected: 1,
                actual: 2,
                ..
            }
        ));
    }

    #[test]
    fn stochastic_nll_validates_batch_size() {
        let (nll, _params) = make_constant_nll();
        let err_zero = match nll.to_stochastic(0, Some(0)) {
            Ok(_) => panic!("expected batch_size=0 to return an error"),
            Err(err) => err,
        };
        assert!(matches!(
            err_zero,
            LadduError::LengthMismatch {
                expected: 2,
                actual: 0,
                ..
            }
        ));

        let err_large = match nll.to_stochastic(3, Some(0)) {
            Ok(_) => panic!("expected oversized batch to return an error"),
            Err(err) => err,
        };
        assert!(matches!(
            err_large,
            LadduError::LengthMismatch {
                expected: 2,
                actual: 3,
                ..
            }
        ));
    }

    #[test]
    fn stochastic_nll_accepts_full_dataset_batch() {
        let (nll, params) = make_constant_nll();
        let stochastic = nll.to_stochastic(2, Some(0)).unwrap();
        let value = stochastic.evaluate(&params).unwrap();
        assert!(value.is_finite());
    }

    #[test]
    fn stochastic_nll_matches_closed_form_on_full_batch() {
        let (nll, params) = make_constant_nll();
        let stochastic = nll
            .to_stochastic(nll.data_evaluator.dataset.n_events(), Some(0))
            .unwrap();
        let stochastic_value = stochastic.evaluate(&params).unwrap();
        let deterministic_value = nll.evaluate(&params).unwrap();
        assert_relative_eq!(stochastic_value, deterministic_value, epsilon = 1e-12);
    }

    #[test]
    fn likelihood_evaluator_reports_length_mismatch() {
        let alpha = LikelihoodScalar::new("alpha").unwrap();

        let err_short = alpha.evaluate(&[]).unwrap_err();
        assert!(matches!(
            err_short,
            LadduError::LengthMismatch {
                expected: 1,
                actual: 0,
                ..
            }
        ));

        let err_long = alpha.evaluate_gradient(&[1.0, 2.0]).unwrap_err();
        assert!(matches!(
            err_long,
            LadduError::LengthMismatch {
                expected: 1,
                actual: 2,
                ..
            }
        ));
    }

    #[cfg(feature = "mpi")]
    #[mpi_test(np = [2])]
    fn mpi_negative_paths_report_structured_errors() {
        use_mpi(true);
        let world = get_world().expect("MPI world should be initialized");
        let (nll, params) = make_constant_nll();

        let err_len = nll.project_weights_mpi(&[], None, &world).unwrap_err();
        assert!(matches!(
            err_len,
            LadduError::LengthMismatch {
                expected: 1,
                actual: 0,
                ..
            }
        ));

        let err_amp = nll
            .project_weights_subset_mpi_strict::<&str>(
                &params,
                &["missing_amplitude"],
                None,
                &world,
            )
            .unwrap_err();
        assert!(matches!(err_amp, LadduError::AmplitudeNotFoundError { .. }));
        finalize_mpi();
    }

    #[cfg(feature = "mpi")]
    #[mpi_test(np = [2])]
    fn mpi_value_and_gradient_match_total_non_mpi() {
        use_mpi(true);
        let world = get_world().expect("MPI world should be initialized");
        let (nll, params) = make_constant_nll();
        let data_term_local = crate::likelihood::nll::evaluate_weighted_expression_sum_local(
            &nll.data_evaluator,
            &params,
            |l| f64::ln(l.re),
        )
        .expect("evaluate should succeed");
        let mc_term_local = nll
            .accmc_evaluator
            .evaluate_weighted_value_sum_local(&params)
            .expect("evaluate should succeed");
        let data_term = crate::likelihood::nll::reduce_scalar(&world, data_term_local);
        let mc_term = crate::likelihood::nll::reduce_scalar(&world, mc_term_local);
        let expected_value = -2.0 * (data_term - mc_term / nll.n_mc);

        let mpi_value = nll
            .evaluate_mpi(&params, &world)
            .expect("evaluate should succeed");
        assert_relative_eq!(mpi_value, expected_value);

        let data_gradient_local = nll
            .evaluate_data_gradient_term_local(&params)
            .expect("evaluate should succeed");
        let mc_gradient_local = nll
            .accmc_evaluator
            .evaluate_weighted_gradient_sum_local(&params)
            .expect("evaluate should succeed");
        let data_gradient = crate::likelihood::nll::reduce_gradient(&world, &data_gradient_local);
        let mc_gradient = crate::likelihood::nll::reduce_gradient(&world, &mc_gradient_local);
        let expected_gradient = -2.0 * (data_gradient - mc_gradient / nll.n_mc);
        let mpi_gradient = nll
            .evaluate_gradient_mpi(&params, &world)
            .expect("evaluate should succeed");
        assert_relative_eq!(mpi_gradient, expected_gradient);

        finalize_mpi();
    }

    #[cfg(feature = "mpi")]
    #[mpi_test(np = [2])]
    fn mpi_deterministic_fixture_matches_local_and_reduced_baselines_across_activation_toggles() {
        use_mpi(true);
        let world = get_world().expect("MPI world should be initialized");

        let fixture = make_deterministic_nll_fixture(DeterministicModelKind::Partial);
        assert_nll_fixture_matches_weighted_baseline(&fixture);
        assert_nll_fixture_matches_mpi_reduced_baseline(&fixture, &world);

        fixture.nll.isolate_many(&["p", "c"]);
        assert_nll_fixture_matches_weighted_baseline(&fixture);
        assert_nll_fixture_matches_mpi_reduced_baseline(&fixture, &world);

        fixture.nll.activate_all();
        assert_nll_fixture_matches_weighted_baseline(&fixture);
        assert_nll_fixture_matches_mpi_reduced_baseline(&fixture, &world);

        finalize_mpi();
    }

    #[cfg(feature = "mpi")]
    #[mpi_test(np = [2])]
    fn mpi_mixed_scale_value_matches_local_evaluate() {
        use_mpi(true);
        let world = get_world().expect("MPI world should be initialized");
        let amp_a = CachedBeamScaleAmplitude::new("amp_a", parameter!("scale_a"), 0).unwrap();
        let amp_b = CachedBeamScaleAmplitude::new("amp_b", parameter!("scale_b"), 1).unwrap();
        let expr = (amp_a + amp_b).norm_sqr();
        let data = dataset_with_two_p4_and_weights(
            &[(1.0, 0.5), (10.0, 1.0), (3.0, 5.0), (1.0e2, 2.0e-1)],
            &[1.0e12, 1.0e-12, 3.5, 7.25e4],
        );
        let mc = dataset_with_two_p4_and_weights(
            &[(4.0, 0.1), (6.0, 2.0), (8.0, 1.5), (1.0e1, 3.0)],
            &[4.0e9, 9.0e-6, 1.25, 2.5e2],
        );
        let nll = NLL::new(&expr, &data, &mc, None).unwrap();
        let params = vec![1.125, -0.375];

        let data_local = nll
            .data_evaluator
            .evaluate_local(&params)
            .expect("evaluate should succeed");
        let mc_local = nll
            .accmc_evaluator
            .evaluate_local(&params)
            .expect("evaluate should succeed");
        let data_term_local: f64 = data_local
            .iter()
            .zip(nll.data_evaluator.dataset.weights_local().iter())
            .map(|(value, event)| *event * value.re.ln())
            .sum();
        let mc_term_local: f64 = mc_local
            .iter()
            .zip(nll.accmc_evaluator.dataset.weights_local().iter())
            .map(|(value, event)| *event * value.re)
            .sum();
        let data_term = crate::likelihood::nll::reduce_scalar(&world, data_term_local);
        let mc_term = crate::likelihood::nll::reduce_scalar(&world, mc_term_local);
        let expected = -2.0 * (data_term - mc_term / nll.n_mc);
        let mpi_value = nll
            .evaluate_mpi(&params, &world)
            .expect("evaluate should succeed");
        assert_relative_eq!(mpi_value, expected, epsilon = 1e-9, max_relative = 1e-12);
        finalize_mpi();
    }

    #[cfg(feature = "mpi")]
    #[mpi_test(np = [2])]
    fn mpi_projection_paths_are_explicit_global_gathers() {
        use_mpi(true);
        let world = get_world().expect("MPI world should be initialized");
        let (nll, params) = make_constant_nll();

        let local_projection = nll
            .project_weights_local(&params, None)
            .expect("local projection should evaluate");
        let gathered_projection = nll
            .project_weights_mpi(&params, None, &world)
            .expect("mpi projection should gather global projection");
        let local_len = nll.accmc_evaluator.dataset.n_events_local();
        let total_len = nll.accmc_evaluator.dataset.n_events();
        assert_eq!(local_projection.len(), local_len);
        assert_eq!(gathered_projection.len(), total_len);

        let (counts, displs) = world.get_counts_displs(total_len);
        let rank = world.rank() as usize;
        let start = displs[rank] as usize;
        let end = start + counts[rank] as usize;
        assert_eq!(
            &gathered_projection[start..end],
            local_projection.as_slice()
        );

        let (local_weights, local_gradients) = nll
            .project_weights_and_gradients_local(&params, None)
            .expect("local projection gradient should evaluate");
        let (gathered_weights, gathered_gradients) = nll
            .project_weights_and_gradients_mpi(&params, None, &world)
            .expect("mpi projection gradient should gather global projection");
        assert_eq!(local_weights.len(), local_len);
        assert_eq!(local_gradients.len(), local_len);
        assert_eq!(gathered_weights.len(), total_len);
        assert_eq!(gathered_gradients.len(), total_len);
        assert_eq!(&gathered_weights[start..end], local_weights.as_slice());

        let local_grad_slice = &gathered_gradients[start..end];
        for (lhs, rhs) in local_grad_slice.iter().zip(local_gradients.iter()) {
            assert_relative_eq!(lhs, rhs);
        }
        finalize_mpi();
    }

    #[cfg(feature = "mpi")]
    #[mpi_test(np = [2])]
    fn mpi_project_weights_subsets_matches_repeated_project_weights_subset_mpi() {
        use_mpi(true);
        let world = get_world().expect("MPI world should be initialized");
        let (nll, params) = make_two_parameter_nll();
        let subsets = vec![
            vec!["amp_b".to_string()],
            vec!["amp_a".to_string()],
            vec!["amp_a".to_string(), "amp_b".to_string()],
            vec!["amp_a".to_string()],
        ];
        let batched = nll
            .project_weights_subsets_mpi(&params, &subsets, None, &world)
            .expect("batched mpi projection should evaluate");
        let repeated = subsets
            .iter()
            .map(|subset| {
                nll.project_weights_subset_mpi(&params, subset, None, &world)
                    .expect("single mpi subset projection should evaluate")
            })
            .collect::<Vec<_>>();
        assert_eq!(batched.len(), repeated.len());
        for (lhs, rhs) in batched.iter().zip(repeated.iter()) {
            assert_eq!(lhs.len(), rhs.len());
            for (lhs_value, rhs_value) in lhs.iter().zip(rhs.iter()) {
                assert_relative_eq!(lhs_value, rhs_value, epsilon = 1e-12);
            }
        }
        finalize_mpi();
    }

    #[cfg(feature = "mpi")]
    #[mpi_test(np = [2])]
    fn mpi_project_weights_and_gradients_subset_matches_repeated_project_weights_and_gradients_subset_mpi(
    ) {
        use_mpi(true);
        let world = get_world().expect("MPI world should be initialized");
        let (nll, params) = make_two_parameter_nll();
        let subsets = vec![
            vec!["amp_b".to_string()],
            vec!["amp_a".to_string()],
            vec!["amp_a".to_string(), "amp_b".to_string()],
        ];
        for subset in subsets {
            let (weights_mpi, gradients_mpi) = nll
                .project_weights_and_gradients_subset_mpi(&params, &subset, None, &world)
                .expect("mpi gradient projection should evaluate");
            let (weights_auto, gradients_auto) = nll
                .project_weights_and_gradients_subset(&params, &subset, None)
                .expect("auto gradient projection should evaluate");
            assert_eq!(weights_mpi.len(), weights_auto.len());
            assert_eq!(gradients_mpi.len(), gradients_auto.len());
            for (lhs, rhs) in weights_mpi.iter().zip(weights_auto.iter()) {
                assert_relative_eq!(lhs, rhs, epsilon = 1e-12);
            }
            for (lhs, rhs) in gradients_mpi.iter().zip(gradients_auto.iter()) {
                assert_relative_eq!(lhs, rhs, epsilon = 1e-12);
            }
        }
        finalize_mpi();
    }

    #[cfg(feature = "mpi")]
    #[mpi_test(np = [2])]
    fn mpi_mixed_workload_rss_stays_bounded() {
        use_mpi(true);
        let world = get_world().expect("MPI world should be initialized");
        let fixture = make_mixed_workload_nll_fixture(2_048);

        let baseline_value = fixture
            .nll
            .evaluate_mpi(&fixture.parameters, &world)
            .expect("evaluate should succeed");
        let baseline_gradient = fixture
            .nll
            .evaluate_gradient_mpi(&fixture.parameters, &world)
            .expect("evaluate should succeed");
        let baseline_weights = fixture
            .nll
            .project_weights_mpi(&fixture.parameters, None, &world)
            .expect("baseline MPI projection should evaluate");
        let (baseline_projection_weights, baseline_projection_gradients) = fixture
            .nll
            .project_weights_and_gradients_mpi(&fixture.parameters, None, &world)
            .expect("baseline MPI projection gradient should evaluate");
        let mut post_warmup_rss_kb = Vec::new();

        assert_relative_eq!(
            baseline_weights.as_slice(),
            baseline_projection_weights.as_slice(),
            epsilon = DETERMINISTIC_STRICT_ABS_TOL,
            max_relative = DETERMINISTIC_STRICT_REL_TOL
        );

        for pass_index in 0..24 {
            let value = fixture
                .nll
                .evaluate_mpi(&fixture.parameters, &world)
                .expect("evaluate should succeed");
            assert_relative_eq!(
                value,
                baseline_value,
                epsilon = DETERMINISTIC_STRICT_ABS_TOL,
                max_relative = DETERMINISTIC_STRICT_REL_TOL
            );

            let gradient = fixture
                .nll
                .evaluate_gradient_mpi(&fixture.parameters, &world)
                .expect("evaluate should succeed");
            assert_eq!(
                gradient.len(),
                baseline_gradient.len(),
                "mixed-workload MPI gradient length should remain stable"
            );
            for (actual_item, expected_item) in gradient.iter().zip(baseline_gradient.iter()) {
                assert_relative_eq!(
                    *actual_item,
                    *expected_item,
                    epsilon = DETERMINISTIC_STRICT_ABS_TOL,
                    max_relative = DETERMINISTIC_STRICT_REL_TOL
                );
            }

            let weights = fixture
                .nll
                .project_weights_mpi(&fixture.parameters, None, &world)
                .expect("MPI projection should remain evaluable");
            assert_eq!(
                weights.len(),
                baseline_weights.len(),
                "mixed-workload MPI projection length should remain stable"
            );
            for (actual_item, expected_item) in weights.iter().zip(baseline_weights.iter()) {
                assert_relative_eq!(
                    *actual_item,
                    *expected_item,
                    epsilon = DETERMINISTIC_STRICT_ABS_TOL,
                    max_relative = DETERMINISTIC_STRICT_REL_TOL
                );
            }

            let (projection_weights, projection_gradients) = fixture
                .nll
                .project_weights_and_gradients_mpi(&fixture.parameters, None, &world)
                .expect("MPI projection gradients should remain evaluable");
            assert_eq!(
                projection_weights.len(),
                baseline_projection_weights.len(),
                "mixed-workload MPI projection-gradient weight length should remain stable"
            );
            assert_eq!(
                projection_gradients.len(),
                baseline_projection_gradients.len(),
                "mixed-workload MPI projection-gradient length should remain stable"
            );
            for (actual_item, expected_item) in projection_weights
                .iter()
                .zip(baseline_projection_weights.iter())
            {
                assert_relative_eq!(
                    *actual_item,
                    *expected_item,
                    epsilon = DETERMINISTIC_STRICT_ABS_TOL,
                    max_relative = DETERMINISTIC_STRICT_REL_TOL
                );
            }
            for (actual_gradient, expected_gradient) in projection_gradients
                .iter()
                .zip(baseline_projection_gradients.iter())
            {
                assert_eq!(
                    actual_gradient.len(),
                    expected_gradient.len(),
                    "mixed-workload MPI projection-gradient vector length should remain stable"
                );
                for (actual_item, expected_item) in
                    actual_gradient.iter().zip(expected_gradient.iter())
                {
                    assert_relative_eq!(
                        *actual_item,
                        *expected_item,
                        epsilon = DETERMINISTIC_STRICT_ABS_TOL,
                        max_relative = DETERMINISTIC_STRICT_REL_TOL
                    );
                }
            }

            if pass_index >= 3 {
                if let Some(rss_kb) = read_resident_rss_kb() {
                    post_warmup_rss_kb.push(rss_kb);
                }
            }
        }

        if let Some((&first_rss_kb, rest_rss_kb)) = post_warmup_rss_kb.split_first() {
            let last_rss_kb = *rest_rss_kb.last().unwrap_or(&first_rss_kb);
            let min_rss_kb = post_warmup_rss_kb
                .iter()
                .copied()
                .min()
                .expect("post-warmup RSS sample should exist");
            let max_rss_kb = post_warmup_rss_kb
                .iter()
                .copied()
                .max()
                .expect("post-warmup RSS sample should exist");
            const MAX_POST_WARMUP_RSS_GROWTH_KB: u64 = 64 * 1024;
            const MAX_POST_WARMUP_RSS_SPREAD_KB: u64 = 64 * 1024;
            assert!(
                last_rss_kb.saturating_sub(first_rss_kb) <= MAX_POST_WARMUP_RSS_GROWTH_KB,
                "mixed-workload post-warmup RSS grew by {} KiB (first={} KiB, last={} KiB)",
                last_rss_kb.saturating_sub(first_rss_kb),
                first_rss_kb,
                last_rss_kb
            );
            assert!(
                max_rss_kb.saturating_sub(min_rss_kb) <= MAX_POST_WARMUP_RSS_SPREAD_KB,
                "mixed-workload post-warmup RSS spread was {} KiB (min={} KiB, max={} KiB)",
                max_rss_kb.saturating_sub(min_rss_kb),
                min_rss_kb,
                max_rss_kb
            );
        }

        finalize_mpi();
    }
}
