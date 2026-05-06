use approx::assert_relative_eq;
#[cfg(feature = "mpi")]
use mpi_test::mpi_test;
use serde::{Deserialize, Serialize};

use super::*;
use crate::{
    amplitudes::TestAmplitude,
    data::{test_dataset, test_event, DatasetMetadata, EventData},
    parameter,
    parameters::Parameter,
    resources::{Cache, ParameterID, Parameters, Resources, ScalarID},
    vectors::Vec4,
};

#[derive(Clone, Serialize, Deserialize)]
pub struct ComplexScalar {
    name: String,
    re: Parameter,
    pid_re: ParameterID,
    im: Parameter,
    pid_im: ParameterID,
}

impl ComplexScalar {
    #[allow(clippy::new_ret_no_self)]
    pub fn new(name: &str, re: Parameter, im: Parameter) -> LadduResult<Expression> {
        Self {
            name: name.to_string(),
            re,
            pid_re: Default::default(),
            im,
            pid_im: Default::default(),
        }
        .into_expression()
    }
}

#[typetag::serde]
impl Amplitude for ComplexScalar {
    fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID> {
        self.pid_re = resources.register_parameter(&self.re)?;
        self.pid_im = resources.register_parameter(&self.im)?;
        resources.register_amplitude(&self.name)
    }

    fn compute(&self, parameters: &Parameters, _cache: &Cache) -> Complex64 {
        Complex64::new(parameters.get(self.pid_re), parameters.get(self.pid_im))
    }

    fn compute_gradient(
        &self,
        parameters: &Parameters,
        _cache: &Cache,
        gradient: &mut DVector<Complex64>,
    ) {
        if let Some(ind) = parameters.free_index(self.pid_re) {
            gradient[ind] = Complex64::ONE;
        }
        if let Some(ind) = parameters.free_index(self.pid_im) {
            gradient[ind] = Complex64::I;
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ParameterOnlyScalar {
    name: String,
    value: Parameter,
    pid: ParameterID,
}

impl ParameterOnlyScalar {
    #[allow(clippy::new_ret_no_self)]
    pub fn new(name: &str, value: Parameter) -> LadduResult<Expression> {
        Self {
            name: name.to_string(),
            value,
            pid: Default::default(),
        }
        .into_expression()
    }
}

#[typetag::serde]
impl Amplitude for ParameterOnlyScalar {
    fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID> {
        self.pid = resources.register_parameter(&self.value)?;
        resources.register_amplitude(&self.name)
    }

    fn dependence_hint(&self) -> ExpressionDependence {
        ExpressionDependence::ParameterOnly
    }

    fn real_valued_hint(&self) -> bool {
        true
    }

    fn compute(&self, parameters: &Parameters, _cache: &Cache) -> Complex64 {
        Complex64::new(parameters.get(self.pid), 0.0)
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct CacheOnlyScalar {
    name: String,
    beam_energy: ScalarID,
}

impl CacheOnlyScalar {
    #[allow(clippy::new_ret_no_self)]
    pub fn new(name: &str) -> LadduResult<Expression> {
        Self {
            name: name.to_string(),
            beam_energy: Default::default(),
        }
        .into_expression()
    }
}

#[typetag::serde]
impl Amplitude for CacheOnlyScalar {
    fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID> {
        self.beam_energy = resources.register_scalar(Some(&format!("{}.beam_energy", self.name)));
        resources.register_amplitude(&self.name)
    }

    fn dependence_hint(&self) -> ExpressionDependence {
        ExpressionDependence::CacheOnly
    }

    fn real_valued_hint(&self) -> bool {
        true
    }

    fn precompute(&self, event: &Event<'_>, cache: &mut Cache) {
        cache.store_scalar(self.beam_energy, event.p4_at(0).e());
    }

    fn compute(&self, _parameters: &Parameters, cache: &Cache) -> Complex64 {
        Complex64::new(cache.get_scalar(self.beam_energy), 0.0)
    }
}

#[derive(Clone, Copy)]
enum DeterministicFixtureKind {
    Separable,
    Partial,
    NonSeparable,
}

struct DeterministicFixture {
    expression: Expression,
    dataset: Arc<Dataset>,
    parameters: Vec<f64>,
}

const DETERMINISTIC_STRICT_ABS_TOL: f64 = 1e-12;
const DETERMINISTIC_STRICT_REL_TOL: f64 = 1e-10;

fn deterministic_fixture_dataset() -> Arc<Dataset> {
    let metadata = Arc::new(DatasetMetadata::default());
    let events = vec![
        Arc::new(EventData {
            p4s: vec![Vec4::new(0.0, 0.0, 0.0, 1.0)],
            aux: vec![],
            weight: 0.5,
        }),
        Arc::new(EventData {
            p4s: vec![Vec4::new(0.0, 0.0, 0.0, 2.0)],
            aux: vec![],
            weight: -1.25,
        }),
        Arc::new(EventData {
            p4s: vec![Vec4::new(0.0, 0.0, 0.0, 3.0)],
            aux: vec![],
            weight: 2.0,
        }),
        Arc::new(EventData {
            p4s: vec![Vec4::new(0.0, 0.0, 0.0, 5.0)],
            aux: vec![],
            weight: 0.75,
        }),
    ];
    Arc::new(Dataset::new_with_metadata(events, metadata))
}

fn make_deterministic_fixture(kind: DeterministicFixtureKind) -> DeterministicFixture {
    let dataset = deterministic_fixture_dataset();
    match kind {
        DeterministicFixtureKind::Separable => {
            let p1 = ParameterOnlyScalar::new("p1", parameter!("p1"))
                .expect("separable p1 should build");
            let p2 = ParameterOnlyScalar::new("p2", parameter!("p2"))
                .expect("separable p2 should build");
            let c1 = CacheOnlyScalar::new("c1").expect("separable c1 should build");
            let c2 = CacheOnlyScalar::new("c2").expect("separable c2 should build");
            DeterministicFixture {
                expression: (&p1 * &c1) + &(&p2 * &c2),
                dataset,
                parameters: vec![0.4, -0.3],
            }
        }
        DeterministicFixtureKind::Partial => {
            let p = ParameterOnlyScalar::new("p", parameter!("p")).expect("partial p should build");
            let c = CacheOnlyScalar::new("c").expect("partial c should build");
            let m = TestAmplitude::new("m", parameter!("mr"), parameter!("mi"))
                .expect("partial m should build");
            DeterministicFixture {
                expression: (&p * &c) + &m,
                dataset,
                parameters: vec![0.55, 0.2, -0.15],
            }
        }
        DeterministicFixtureKind::NonSeparable => {
            let m1 = TestAmplitude::new("m1", parameter!("m1r"), parameter!("m1i"))
                .expect("non-separable m1 should build");
            let m2 = TestAmplitude::new("m2", parameter!("m2r"), parameter!("m2i"))
                .expect("non-separable m2 should build");
            DeterministicFixture {
                expression: &m1 * &m2,
                dataset,
                parameters: vec![0.25, -0.4, 0.6, 0.1],
            }
        }
    }
}

fn assert_weighted_sum_matches_eventwise_baseline(fixture: &DeterministicFixture) {
    let evaluator = fixture
        .expression
        .load(&fixture.dataset)
        .expect("fixture evaluator should load");
    let expected_value = evaluator
        .evaluate_local(&fixture.parameters)
        .expect("evaluation should succeed")
        .iter()
        .zip(fixture.dataset.weights_local().iter())
        .fold(0.0, |accum, (value, event)| accum + *event * value.re);
    let expected_gradient = evaluator
        .evaluate_gradient_local(&fixture.parameters)
        .expect("evaluation should succeed")
        .iter()
        .zip(fixture.dataset.weights_local().iter())
        .fold(
            DVector::zeros(fixture.parameters.len()),
            |mut accum, (gradient, event)| {
                accum += gradient.map(|value| value.re).scale(*event);
                accum
            },
        );
    let actual_value = evaluator
        .evaluate_weighted_value_sum_local(&fixture.parameters)
        .expect("evaluation should succeed");
    let actual_gradient = evaluator
        .evaluate_weighted_gradient_sum_local(&fixture.parameters)
        .expect("evaluation should succeed");
    assert_relative_eq!(
        actual_value,
        expected_value,
        epsilon = DETERMINISTIC_STRICT_ABS_TOL,
        max_relative = DETERMINISTIC_STRICT_REL_TOL
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
fn assert_mixed_normalization_components_match_combined_path(fixture: &DeterministicFixture) {
    let evaluator = fixture
        .expression
        .load(&fixture.dataset)
        .expect("fixture evaluator should load");
    let state = {
        let resources = evaluator.resources.read();
        evaluator.ensure_cached_integral_cache_state(&resources)
    }
    .expect("state should be available");
    assert!(
        !state.values.is_empty(),
        "fixture should exercise cached normalization terms"
    );
    assert!(
        !state.execution_sets.residual_amplitudes.is_empty(),
        "fixture should exercise residual normalization amplitudes"
    );

    let (residual_value_sum, cached_value_sum) = evaluator
        .evaluate_weighted_value_sum_local_components(&fixture.parameters)
        .expect("evaluation should succeed");
    assert!(residual_value_sum.abs() > DETERMINISTIC_STRICT_ABS_TOL);
    assert!(cached_value_sum.abs() > DETERMINISTIC_STRICT_ABS_TOL);
    let combined_value = evaluator
        .evaluate_weighted_value_sum_local(&fixture.parameters)
        .expect("evaluation should succeed");
    assert_relative_eq!(
        residual_value_sum + cached_value_sum,
        combined_value,
        epsilon = DETERMINISTIC_STRICT_ABS_TOL,
        max_relative = DETERMINISTIC_STRICT_REL_TOL
    );

    let (residual_gradient_sum, cached_gradient_sum) = evaluator
        .evaluate_weighted_gradient_sum_local_components(&fixture.parameters)
        .expect("evaluation should succeed");
    let combined_gradient = evaluator
        .evaluate_weighted_gradient_sum_local(&fixture.parameters)
        .expect("evaluation should succeed");
    assert!(residual_gradient_sum
        .iter()
        .any(|value| value.abs() > DETERMINISTIC_STRICT_ABS_TOL));
    assert!(cached_gradient_sum
        .iter()
        .any(|value| value.abs() > DETERMINISTIC_STRICT_ABS_TOL));
    for ((residual_item, cached_item), combined_item) in residual_gradient_sum
        .iter()
        .zip(cached_gradient_sum.iter())
        .zip(combined_gradient.iter())
    {
        assert_relative_eq!(
            residual_item + cached_item,
            *combined_item,
            epsilon = DETERMINISTIC_STRICT_ABS_TOL,
            max_relative = DETERMINISTIC_STRICT_REL_TOL
        );
    }
}

#[test]
fn test_deterministic_fixture_weighted_sums_stable_across_activation_mask_toggle() {
    let fixture = make_deterministic_fixture(DeterministicFixtureKind::Partial);
    let evaluator = fixture
        .expression
        .load(&fixture.dataset)
        .expect("fixture evaluator should load");
    let original_mask = evaluator.active_mask();

    let original_value = evaluator
        .evaluate_weighted_value_sum_local(&fixture.parameters)
        .expect("evaluation should succeed");

    evaluator.isolate_many(&["p", "c"]);
    assert_ne!(evaluator.active_mask(), original_mask);

    evaluator
        .set_active_mask(&original_mask)
        .expect("original fixture active mask should restore");
    assert_eq!(evaluator.active_mask(), original_mask);
    let actual_value = evaluator
        .evaluate_weighted_value_sum_local(&fixture.parameters)
        .expect("evaluation should succeed");
    assert_relative_eq!(
        actual_value,
        original_value,
        epsilon = DETERMINISTIC_STRICT_ABS_TOL,
        max_relative = DETERMINISTIC_STRICT_REL_TOL
    );
}

#[test]
fn test_deterministic_fixtures_match_eventwise_weighted_sums() {
    let separable = make_deterministic_fixture(DeterministicFixtureKind::Separable);
    let partial = make_deterministic_fixture(DeterministicFixtureKind::Partial);
    let non_separable = make_deterministic_fixture(DeterministicFixtureKind::NonSeparable);

    assert_weighted_sum_matches_eventwise_baseline(&separable);
    assert_weighted_sum_matches_eventwise_baseline(&partial);
    assert_weighted_sum_matches_eventwise_baseline(&non_separable);
}
#[test]
fn test_deterministic_fixtures_cover_separable_partial_non_separable_models() {
    let separable = make_deterministic_fixture(DeterministicFixtureKind::Separable);
    let partial = make_deterministic_fixture(DeterministicFixtureKind::Partial);
    let non_separable = make_deterministic_fixture(DeterministicFixtureKind::NonSeparable);

    let separable_evaluator = separable
        .expression
        .load(&separable.dataset)
        .expect("separable evaluator should load");
    let partial_evaluator = partial
        .expression
        .load(&partial.dataset)
        .expect("partial evaluator should load");
    let non_separable_evaluator = non_separable
        .expression
        .load(&non_separable.dataset)
        .expect("non-separable evaluator should load");

    assert_eq!(
        separable_evaluator
            .expression_precomputed_cached_integrals()
            .expect("integrals should be computed")
            .len(),
        2
    );
    assert_eq!(
        partial_evaluator
            .expression_precomputed_cached_integrals()
            .expect("integrals should be computed")
            .len(),
        1
    );
    assert!(non_separable_evaluator
        .expression_precomputed_cached_integrals()
        .expect("integrals should be computed")
        .is_empty());
}
#[test]
fn test_partial_fixture_combined_normalization_components_match_total() {
    let partial = make_deterministic_fixture(DeterministicFixtureKind::Partial);
    assert_mixed_normalization_components_match_combined_path(&partial);
}
#[test]
fn test_non_separable_fixture_normalization_components_stay_residual_only() {
    let fixture = make_deterministic_fixture(DeterministicFixtureKind::NonSeparable);
    let evaluator = fixture
        .expression
        .load(&fixture.dataset)
        .expect("fixture evaluator should load");
    let resources = evaluator.resources.read();
    let state = evaluator
        .ensure_cached_integral_cache_state(&resources)
        .expect("state should be available");
    assert!(state.values.is_empty());

    let (residual_value_sum, cached_value_sum) = evaluator
        .evaluate_weighted_value_sum_local_components(&fixture.parameters)
        .expect("evaluation should succeed");
    assert_relative_eq!(
        cached_value_sum,
        0.0,
        epsilon = DETERMINISTIC_STRICT_ABS_TOL
    );
    assert_relative_eq!(
        residual_value_sum,
        evaluator
            .evaluate_weighted_value_sum_local(&fixture.parameters)
            .expect("evaluation should succeed"),
        epsilon = DETERMINISTIC_STRICT_ABS_TOL,
        max_relative = DETERMINISTIC_STRICT_REL_TOL
    );

    let (residual_gradient_sum, cached_gradient_sum) = evaluator
        .evaluate_weighted_gradient_sum_local_components(&fixture.parameters)
        .expect("evaluation should succeed");
    assert!(cached_gradient_sum
        .iter()
        .all(|value| value.abs() <= DETERMINISTIC_STRICT_ABS_TOL));
    let combined_gradient = evaluator
        .evaluate_weighted_gradient_sum_local(&fixture.parameters)
        .expect("evaluation should succeed");
    for (residual_item, combined_item) in residual_gradient_sum.iter().zip(combined_gradient.iter())
    {
        assert_relative_eq!(
            *residual_item,
            *combined_item,
            epsilon = DETERMINISTIC_STRICT_ABS_TOL,
            max_relative = DETERMINISTIC_STRICT_REL_TOL
        );
    }
}

#[test]
fn test_batch_evaluation() {
    let expr = TestAmplitude::new("test", parameter!("real"), parameter!("imag")).unwrap();
    let mut event1 = test_event();
    event1.p4s[0].t = 10.0;
    let mut event2 = test_event();
    event2.p4s[0].t = 11.0;
    let mut event3 = test_event();
    event3.p4s[0].t = 12.0;
    let dataset = Arc::new(Dataset::new_with_metadata(
        vec![Arc::new(event1), Arc::new(event2), Arc::new(event3)],
        Arc::new(DatasetMetadata::default()),
    ));
    let evaluator = expr.load(&dataset).unwrap();
    let result = evaluator
        .evaluate_batch(&[1.1, 2.2], &[0, 2])
        .expect("evaluation should succeed");
    assert_eq!(result.len(), 2);
    assert_eq!(result[0], Complex64::new(1.1, 2.2) * 10.0);
    assert_eq!(result[1], Complex64::new(1.1, 2.2) * 12.0);
    let result_grad = evaluator
        .evaluate_gradient_batch(&[1.1, 2.2], &[0, 2])
        .expect("evaluation should succeed");
    assert_eq!(result_grad.len(), 2);
    assert_eq!(result_grad[0][0], Complex64::new(10.0, 0.0));
    assert_eq!(result_grad[0][1], Complex64::new(0.0, 10.0));
    assert_eq!(result_grad[1][0], Complex64::new(12.0, 0.0));
    assert_eq!(result_grad[1][1], Complex64::new(0.0, 12.0));
}

#[test]
fn test_load_compiles_expression_ir_once() {
    let expr = (TestAmplitude::new("a", parameter!("ar"), parameter!("ai")).unwrap()
        + TestAmplitude::new("b", parameter!("br"), parameter!("bi")).unwrap())
    .norm_sqr();
    let dataset = Arc::new(Dataset::new(vec![Arc::new(test_event())]));
    let evaluator = expr.load(&dataset).unwrap();
    assert!(evaluator.expression_slot_count() > 0);
}
#[test]
fn test_expression_ir_value_matches_lowered_runtime() {
    let expr = ((TestAmplitude::new("a", parameter!("ar"), parameter!("ai")).unwrap()
        + TestAmplitude::new("b", parameter!("br"), parameter!("bi")).unwrap())
        * TestAmplitude::new("c", parameter!("cr"), parameter!("ci")).unwrap())
    .conj()
    .norm_sqr();
    let dataset = Arc::new(Dataset::new(vec![Arc::new(test_event())]));
    let evaluator = expr.load(&dataset).unwrap();
    let resources = evaluator.resources.read();
    let parameters = resources
        .parameter_map
        .assemble(&[1.0, 0.25, -0.8, 0.5, 0.2, -1.1])
        .expect("parameters should assemble");
    let mut amplitude_values = vec![Complex64::ZERO; evaluator.amplitudes.len()];
    evaluator.fill_amplitude_values(
        &mut amplitude_values,
        resources.active_indices(),
        &parameters,
        &resources.caches[0],
    );
    let mut ir_slots = vec![Complex64::ZERO; evaluator.expression_ir().node_count()];
    let lowered_runtime = evaluator.lowered_runtime();
    let lowered_program = lowered_runtime.value_program();
    let mut lowered_slots = vec![Complex64::ZERO; lowered_program.scratch_slots()];
    let lowered_value =
        evaluator.evaluate_expression_value_with_scratch(&amplitude_values, &mut ir_slots);
    let direct_lowered_value = lowered_program.evaluate_into(&amplitude_values, &mut lowered_slots);
    let ir_value = evaluator
        .expression_ir()
        .evaluate_into(&amplitude_values, &mut ir_slots);
    assert_relative_eq!(lowered_value.re, direct_lowered_value.re);
    assert_relative_eq!(lowered_value.im, direct_lowered_value.im);
    assert_relative_eq!(lowered_value.re, ir_value.re);
    assert_relative_eq!(lowered_value.im, ir_value.im);
}
#[test]
fn test_expression_ir_load_initializes_with_lowered_value_runtime() {
    let expr = TestAmplitude::new("a", parameter!("ar"), parameter!("ai"))
        .unwrap()
        .norm_sqr();
    let dataset = Arc::new(Dataset::new(vec![Arc::new(test_event())]));
    let evaluator = expr.load(&dataset).unwrap();
    let lowered_runtime = evaluator.lowered_runtime();
    assert_eq!(
        lowered_runtime.value_program().kind(),
        lowered::LoweredProgramKind::Value
    );
    assert_eq!(
        lowered_runtime.gradient_program().kind(),
        lowered::LoweredProgramKind::Gradient
    );
    assert_eq!(
        lowered_runtime.value_gradient_program().kind(),
        lowered::LoweredProgramKind::ValueGradient
    );
}
#[test]
fn test_expression_ir_gradient_matches_lowered_runtime() {
    let expr = (TestAmplitude::new("a", parameter!("ar"), parameter!("ai")).unwrap()
        * TestAmplitude::new("b", parameter!("br"), parameter!("bi")).unwrap())
    .norm_sqr();
    let dataset = Arc::new(Dataset::new(vec![Arc::new(test_event())]));
    let evaluator = expr.load(&dataset).unwrap();
    let resources = evaluator.resources.read();
    let parameters = resources
        .parameter_map
        .assemble(&[1.0, 0.25, -0.8, 0.5])
        .expect("parameters should assemble");
    let mut amplitude_values = vec![Complex64::ZERO; evaluator.amplitudes.len()];
    evaluator.fill_amplitude_values(
        &mut amplitude_values,
        resources.active_indices(),
        &parameters,
        &resources.caches[0],
    );
    let mut active_mask = vec![false; evaluator.amplitudes.len()];
    for &index in resources.active_indices() {
        active_mask[index] = true;
    }
    let mut amplitude_gradients = (0..evaluator.amplitudes.len())
        .map(|_| DVector::zeros(parameters.len()))
        .collect::<Vec<_>>();
    evaluator.fill_amplitude_gradients(
        &mut amplitude_gradients,
        &active_mask,
        &parameters,
        &resources.caches[0],
    );
    let mut ir_value_slots = vec![Complex64::ZERO; evaluator.expression_ir().node_count()];
    let mut ir_gradient_slots: Vec<DVector<Complex64>> =
        (0..evaluator.expression_ir().node_count())
            .map(|_| DVector::zeros(parameters.len()))
            .collect();
    let lowered_runtime = evaluator.lowered_runtime();
    let lowered_program = lowered_runtime.gradient_program();
    let mut lowered_value_slots = vec![Complex64::ZERO; lowered_program.scratch_slots()];
    let mut lowered_gradient_slots: Vec<DVector<Complex64>> = (0..lowered_program.scratch_slots())
        .map(|_| DVector::zeros(parameters.len()))
        .collect();
    let active_gradient = evaluator.evaluate_expression_gradient_with_scratch(
        &amplitude_values,
        &amplitude_gradients,
        &mut ir_value_slots,
        &mut ir_gradient_slots,
    );
    let ir_gradient = evaluator.expression_ir().evaluate_gradient_into(
        &amplitude_values,
        &amplitude_gradients,
        &mut ir_value_slots,
        &mut ir_gradient_slots,
    );
    let lowered_gradient = lowered_program.evaluate_gradient_into(
        &amplitude_values,
        &amplitude_gradients,
        &mut lowered_value_slots,
        &mut lowered_gradient_slots,
    );
    for (active, lowered) in active_gradient.iter().zip(lowered_gradient.iter()) {
        assert_relative_eq!(active.re, lowered.re);
        assert_relative_eq!(active.im, lowered.im);
    }
    for (lowered, ir) in lowered_gradient.iter().zip(ir_gradient.iter()) {
        assert_relative_eq!(lowered.re, ir.re);
        assert_relative_eq!(lowered.im, ir.im);
    }
}
#[test]
fn test_expression_ir_value_gradient_matches_lowered_runtime() {
    let expr = ((TestAmplitude::new("a", parameter!("ar"), parameter!("ai")).unwrap()
        + TestAmplitude::new("b", parameter!("br"), parameter!("bi")).unwrap())
        * TestAmplitude::new("c", parameter!("cr"), parameter!("ci")).unwrap())
    .norm_sqr();
    let dataset = Arc::new(Dataset::new(vec![Arc::new(test_event())]));
    let evaluator = expr.load(&dataset).unwrap();
    let resources = evaluator.resources.read();
    let parameters = resources
        .parameter_map
        .assemble(&[1.0, 0.25, -0.8, 0.5, 0.2, -1.1])
        .expect("parameters should assemble");
    let mut amplitude_values = vec![Complex64::ZERO; evaluator.amplitudes.len()];
    evaluator.fill_amplitude_values(
        &mut amplitude_values,
        resources.active_indices(),
        &parameters,
        &resources.caches[0],
    );
    let mut active_mask = vec![false; evaluator.amplitudes.len()];
    for &index in resources.active_indices() {
        active_mask[index] = true;
    }
    let mut amplitude_gradients = (0..evaluator.amplitudes.len())
        .map(|_| DVector::zeros(parameters.len()))
        .collect::<Vec<_>>();
    evaluator.fill_amplitude_gradients(
        &mut amplitude_gradients,
        &active_mask,
        &parameters,
        &resources.caches[0],
    );
    let mut ir_value_slots = vec![Complex64::ZERO; evaluator.expression_ir().node_count()];
    let mut ir_gradient_slots: Vec<DVector<Complex64>> =
        (0..evaluator.expression_ir().node_count())
            .map(|_| DVector::zeros(parameters.len()))
            .collect();
    let lowered_runtime = evaluator.lowered_runtime();
    let lowered_program = lowered_runtime.value_gradient_program();
    let mut lowered_value_slots = vec![Complex64::ZERO; lowered_program.scratch_slots()];
    let mut lowered_gradient_slots: Vec<DVector<Complex64>> = (0..lowered_program.scratch_slots())
        .map(|_| DVector::zeros(parameters.len()))
        .collect();

    let active_value_gradient = evaluator.evaluate_expression_value_gradient_with_scratch(
        &amplitude_values,
        &amplitude_gradients,
        &mut ir_value_slots,
        &mut ir_gradient_slots,
    );
    let ir_value_gradient = evaluator.expression_ir().evaluate_value_gradient_into(
        &amplitude_values,
        &amplitude_gradients,
        &mut ir_value_slots,
        &mut ir_gradient_slots,
    );
    let lowered_value_gradient = lowered_program.evaluate_value_gradient_into(
        &amplitude_values,
        &amplitude_gradients,
        &mut lowered_value_slots,
        &mut lowered_gradient_slots,
    );

    assert_relative_eq!(active_value_gradient.0.re, lowered_value_gradient.0.re);
    assert_relative_eq!(active_value_gradient.0.im, lowered_value_gradient.0.im);
    for (active, lowered) in active_value_gradient
        .1
        .iter()
        .zip(lowered_value_gradient.1.iter())
    {
        assert_relative_eq!(active.re, lowered.re);
        assert_relative_eq!(active.im, lowered.im);
    }
    assert_relative_eq!(lowered_value_gradient.0.re, ir_value_gradient.0.re);
    assert_relative_eq!(lowered_value_gradient.0.im, ir_value_gradient.0.im);
    for (lowered, ir) in lowered_value_gradient
        .1
        .iter()
        .zip(ir_value_gradient.1.iter())
    {
        assert_relative_eq!(lowered.re, ir.re);
        assert_relative_eq!(lowered.im, ir.im);
    }
}
#[test]
fn test_expression_runtime_diagnostics_reports_lowered_programs() {
    let fixture = make_deterministic_fixture(DeterministicFixtureKind::Partial);
    let evaluator = fixture
        .expression
        .load(&fixture.dataset)
        .expect("fixture evaluator should load");

    let diagnostics = evaluator.expression_runtime_diagnostics();
    assert!(diagnostics.ir_planning_enabled);
    assert!(diagnostics.lowered_value_program_present);
    assert!(diagnostics.lowered_gradient_program_present);
    assert!(diagnostics.lowered_value_gradient_program_present);
    assert!(diagnostics.residual_runtime_present);
    assert_eq!(
        diagnostics.specialization_status,
        Some(ExpressionSpecializationStatus {
            origin: ExpressionSpecializationOrigin::InitialLoad,
        })
    );
}
#[test]
fn test_expression_runtime_diagnostics_reports_specialization_origin() {
    let fixture = make_deterministic_fixture(DeterministicFixtureKind::Partial);
    let evaluator = fixture
        .expression
        .load(&fixture.dataset)
        .expect("fixture evaluator should load");

    assert_eq!(
        evaluator
            .expression_runtime_diagnostics()
            .specialization_status,
        Some(ExpressionSpecializationStatus {
            origin: ExpressionSpecializationOrigin::InitialLoad,
        })
    );

    evaluator.isolate_many(&["p"]);
    assert_eq!(
        evaluator
            .expression_runtime_diagnostics()
            .specialization_status,
        Some(ExpressionSpecializationStatus {
            origin: ExpressionSpecializationOrigin::CacheMissRebuild,
        })
    );
}
#[test]
fn test_compiled_expression_display_reports_dag_refs() {
    let a = TestAmplitude::new("a", parameter!("ar"), parameter!("ai")).unwrap();
    let b = TestAmplitude::new("b", parameter!("br"), parameter!("bi")).unwrap();
    let term = &a * &b;
    let expr = &term + &term;
    let dataset = Arc::new(Dataset::new(vec![Arc::new(test_event())]));
    let evaluator = expr.load(&dataset).unwrap();

    let compiled = evaluator.compiled_expression();
    let display = compiled.to_string();

    assert_eq!(compiled.root(), compiled.nodes().len() - 1);
    assert!(display.contains("#"));
    assert!(display.contains("+"));
    assert!(display.contains("×"));
    assert!(display.contains("a(id=0)"));
    assert!(display.contains("b(id=1)"));
    assert!(display.contains("(ref)"));
}

#[test]
fn test_expression_compiled_expression_display_reports_dag_refs_without_loading() {
    let a = TestAmplitude::new("a", parameter!("ar"), parameter!("ai")).unwrap();
    let b = TestAmplitude::new("b", parameter!("br"), parameter!("bi")).unwrap();
    let term = &a * &b;
    let expr = &term + &term;

    let compiled = expr.compiled_expression();
    let display = compiled.to_string();

    assert_eq!(compiled.root(), compiled.nodes().len() - 1);
    assert!(display.contains("#"));
    assert!(display.contains("+"));
    assert!(display.contains("×"));
    assert!(display.contains("a(id=0)"));
    assert!(display.contains("b(id=1)"));
    assert!(display.contains("(ref)"));
}

#[test]
fn test_compiled_expression_display_uses_current_active_mask() {
    let expr = TestAmplitude::new("a", parameter!("ar"), parameter!("ai")).unwrap()
        + TestAmplitude::new("b", parameter!("br"), parameter!("bi")).unwrap();
    let dataset = Arc::new(Dataset::new(vec![Arc::new(test_event())]));
    let evaluator = expr.load(&dataset).unwrap();
    evaluator.deactivate("b");

    let compiled = evaluator.compiled_expression().to_string();

    assert!(compiled.contains("a(id=0)"));
    assert!(!compiled.contains("b(id=1)"));
    assert!(compiled.contains("const 0"));
}

#[test]
fn test_evaluator_expression_reconstructs_expression() {
    let expr = TestAmplitude::new("a", parameter!("ar"), parameter!("ai")).unwrap();
    let dataset = Arc::new(Dataset::new(vec![Arc::new(test_event())]));
    let evaluator = expr.load(&dataset).unwrap();

    assert_eq!(
        evaluator.expression().compiled_expression(),
        expr.compiled_expression()
    );
}

#[test]
fn test_active_mask_override_ignores_current_ir_specialization() {
    let expr = ComplexScalar::new("amp", parameter!("scale"), parameter!("amp_im", 0.0))
        .unwrap()
        .norm_sqr();
    let dataset = Arc::new(test_dataset());
    let evaluator = expr.load(&dataset).unwrap();
    let params = vec![2.0];

    evaluator.deactivate("amp");
    assert_eq!(
        evaluator
            .evaluate(&params)
            .expect("evaluation should succeed")[0],
        Complex64::new(0.0, 0.0)
    );

    let overridden = evaluator
        .evaluate_local_with_active_mask(&params, &[true])
        .unwrap();
    assert_eq!(overridden[0], Complex64::new(4.0, 0.0));

    let overridden_fused = evaluator
        .evaluate_with_gradient_local_with_active_mask(&params, &[true])
        .unwrap();
    assert_eq!(overridden_fused[0].0, Complex64::new(4.0, 0.0));
    assert_eq!(overridden_fused[0].1[0], Complex64::new(4.0, 0.0));
}
#[test]
fn test_expression_ir_dependence_diagnostics_surface() {
    let expr = (TestAmplitude::new("a", parameter!("ar"), parameter!("ai")).unwrap()
        + TestAmplitude::new("b", parameter!("br"), parameter!("bi")).unwrap())
    .norm_sqr();
    let dataset = Arc::new(Dataset::new(vec![Arc::new(test_event())]));
    let evaluator = expr.load(&dataset).unwrap();
    let annotations = evaluator
        .expression_node_dependence_annotations()
        .expect("annotations should exist");
    assert_eq!(annotations.len(), evaluator.expression_ir().node_count());
    assert!(annotations
        .iter()
        .all(|dependence| *dependence == ExpressionDependence::Mixed));
    assert_eq!(
        evaluator
            .expression_root_dependence()
            .expect("root dependence should exist"),
        ExpressionDependence::Mixed
    );
}
#[test]
fn test_expression_ir_default_dependence_hint_is_mixed() {
    let expr = ComplexScalar::new("c", parameter!("cr"), parameter!("ci")).unwrap();
    let dataset = Arc::new(Dataset::new(vec![Arc::new(test_event())]));
    let evaluator = expr.load(&dataset).unwrap();
    assert_eq!(
        evaluator
            .expression_root_dependence()
            .expect("root dependence should exist"),
        ExpressionDependence::Mixed
    );
}
#[test]
fn test_expression_ir_parameter_only_dependence_hint_propagates() {
    let expr = ParameterOnlyScalar::new("p", parameter!("p")).unwrap();
    let dataset = Arc::new(Dataset::new(vec![Arc::new(test_event())]));
    let evaluator = expr.load(&dataset).unwrap();
    assert_eq!(
        evaluator
            .expression_root_dependence()
            .expect("root dependence should exist"),
        ExpressionDependence::ParameterOnly
    );
}
#[test]
fn test_expression_ir_cache_only_dependence_hint_propagates() {
    let expr = CacheOnlyScalar::new("k").unwrap();
    let dataset = Arc::new(Dataset::new(vec![Arc::new(test_event())]));
    let evaluator = expr.load(&dataset).unwrap();
    assert_eq!(
        evaluator
            .expression_root_dependence()
            .expect("root dependence should exist"),
        ExpressionDependence::CacheOnly
    );
}
#[test]
fn test_expression_ir_real_valued_hint_folds_imag_projection_to_zero() {
    let expr = ParameterOnlyScalar::new("p", parameter!("p"))
        .unwrap()
        .imag();
    let dataset = Arc::new(Dataset::new(vec![Arc::new(test_event())]));
    let evaluator = expr.load(&dataset).unwrap();
    let ir = evaluator.expression_ir();

    assert!(matches!(
        ir.nodes()[ir.root()],
        ir::IrNode::Constant(value) if value == Complex64::ZERO
    ));
    assert_eq!(
        evaluator
            .evaluate(&[2.5])
            .expect("evaluation should succeed")[0],
        Complex64::ZERO
    );
}
#[test]
fn test_expression_ir_real_valued_hint_simplifies_conjugation() {
    let expr = ParameterOnlyScalar::new("p", parameter!("p"))
        .unwrap()
        .conj();
    let dataset = Arc::new(Dataset::new(vec![Arc::new(test_event())]));
    let evaluator = expr.load(&dataset).unwrap();
    let ir = evaluator.expression_ir();

    assert!(matches!(ir.nodes()[ir.root()], ir::IrNode::Amp(0)));
    assert_eq!(
        evaluator
            .evaluate(&[2.5])
            .expect("evaluation should succeed")[0],
        Complex64::new(2.5, 0.0)
    );
}
#[test]
fn test_expression_ir_dependence_warnings_surface() {
    let expr = ParameterOnlyScalar::new("p", parameter!("p")).unwrap()
        + &CacheOnlyScalar::new("k").unwrap();
    let dataset = Arc::new(Dataset::new(vec![Arc::new(test_event())]));
    let evaluator = expr.load(&dataset).unwrap();
    assert!(evaluator
        .expression_dependence_warnings()
        .expect("warnings should exist")
        .iter()
        .any(|warning| warning.contains("both ParameterOnly and CacheOnly")));
}
#[test]
fn test_expression_ir_normalization_plan_explain_surface() {
    let expr = ParameterOnlyScalar::new("p", parameter!("p")).unwrap()
        * &CacheOnlyScalar::new("k").unwrap();
    let dataset = Arc::new(Dataset::new(vec![Arc::new(test_event())]));
    let evaluator = expr.load(&dataset).unwrap();
    let explain = evaluator
        .expression_normalization_plan_explain()
        .expect("plan should exist");
    assert_eq!(explain.root_dependence, ExpressionDependence::Mixed);
    assert_eq!(explain.separable_mul_candidate_nodes.len(), 1);
    assert_eq!(
        explain.cached_separable_nodes,
        explain.separable_mul_candidate_nodes
    );
    assert!(explain.residual_terms.iter().all(|index| {
        !explain
            .separable_mul_candidate_nodes
            .iter()
            .any(|candidate| candidate == index)
    }));
}
#[test]
fn test_expression_ir_normalization_execution_sets_surface() {
    let expr = ParameterOnlyScalar::new("p", parameter!("p")).unwrap()
        * &CacheOnlyScalar::new("k").unwrap();
    let dataset = Arc::new(Dataset::new(vec![Arc::new(test_event())]));
    let evaluator = expr.load(&dataset).unwrap();
    let sets = evaluator
        .expression_normalization_execution_sets()
        .expect("sets should exist");
    assert_eq!(sets.cached_parameter_amplitudes, vec![0]);
    assert_eq!(sets.cached_cache_amplitudes, vec![1]);
    assert!(sets.residual_amplitudes.is_empty());
}
#[test]
fn test_expression_ir_normalization_execution_sets_partial_surface() {
    let expr = (ParameterOnlyScalar::new("p", parameter!("p")).unwrap()
        * &CacheOnlyScalar::new("k").unwrap())
        + &TestAmplitude::new("m", parameter!("mr"), parameter!("mi")).unwrap();
    let dataset = Arc::new(Dataset::new(vec![Arc::new(test_event())]));
    let evaluator = expr.load(&dataset).unwrap();
    let sets = evaluator
        .expression_normalization_execution_sets()
        .expect("sets should exist");
    assert_eq!(sets.cached_parameter_amplitudes, vec![0]);
    assert_eq!(sets.cached_cache_amplitudes, vec![1]);
    assert_eq!(sets.residual_amplitudes, vec![2]);
}
#[test]
fn test_expression_ir_precomputed_cached_integrals_at_load() {
    let expr = ParameterOnlyScalar::new("p", parameter!("p")).unwrap()
        * &CacheOnlyScalar::new("k").unwrap();
    let dataset = Arc::new(Dataset::new(vec![Arc::new(test_event())]));
    let evaluator = expr.load(&dataset).unwrap();
    let precomputed = evaluator
        .expression_precomputed_cached_integrals()
        .expect("integrals should exist");
    assert_eq!(precomputed.len(), 1);
    let cache_reference = CacheOnlyScalar::new("k_ref")
        .unwrap()
        .load(&dataset)
        .unwrap();
    let cache_values = cache_reference
        .evaluate_local(&[])
        .expect("evaluation should succeed");
    let expected_weighted_sum = cache_values
        .iter()
        .zip(dataset.weights_local().iter())
        .fold(Complex64::ZERO, |acc, (value, event)| {
            acc + (*value * *event)
        });
    assert_relative_eq!(
        precomputed[0].weighted_cache_sum.re,
        expected_weighted_sum.re
    );
    assert_relative_eq!(
        precomputed[0].weighted_cache_sum.im,
        expected_weighted_sum.im
    );
}
#[test]
fn test_expression_ir_precomputed_cached_integrals_empty_when_non_separable() {
    let expr = TestAmplitude::new("m", parameter!("mr"), parameter!("mi")).unwrap()
        * &CacheOnlyScalar::new("k").unwrap();
    let dataset = Arc::new(Dataset::new(vec![Arc::new(test_event())]));
    let evaluator = expr.load(&dataset).unwrap();
    assert!(evaluator
        .expression_precomputed_cached_integrals()
        .expect("integrals should exist")
        .is_empty());
}
#[test]
fn test_expression_ir_precomputed_cached_integrals_recompute_on_activation_change() {
    let expr = ParameterOnlyScalar::new("p", parameter!("p")).unwrap()
        * &CacheOnlyScalar::new("k").unwrap();
    let dataset = Arc::new(Dataset::new(vec![Arc::new(test_event())]));
    let evaluator = expr.load(&dataset).unwrap();
    assert_eq!(
        evaluator
            .expression_precomputed_cached_integrals()
            .expect("integrals should exist")
            .len(),
        1
    );

    evaluator.isolate_many(&["p"]);
    assert!(evaluator
        .expression_precomputed_cached_integrals()
        .expect("integrals should exist")
        .is_empty());
}
#[test]
fn test_expression_ir_precomputed_cached_integrals_recompute_on_dataset_change() {
    let expr = ParameterOnlyScalar::new("p", parameter!("p")).unwrap()
        * &CacheOnlyScalar::new("k").unwrap();
    let dataset = Arc::new(Dataset::new(vec![Arc::new(test_event())]));
    let mut evaluator = expr.load(&dataset).unwrap();
    drop(dataset);
    let before = evaluator
        .expression_precomputed_cached_integrals()
        .expect("integrals should exist");
    assert_eq!(before.len(), 1);

    Arc::get_mut(&mut evaluator.dataset)
        .expect("evaluator should own dataset Arc in this test")
        .clear_events_local();
    let after = evaluator
        .expression_precomputed_cached_integrals()
        .expect("integrals should exist");
    assert_eq!(after.len(), 1);
    assert_eq!(after[0].weighted_cache_sum, Complex64::ZERO);
    assert!(before[0].weighted_cache_sum != after[0].weighted_cache_sum);
}
#[test]
fn test_expression_ir_precomputed_cached_integral_gradient_terms_scale_by_cache_integrals() {
    let expr = ParameterOnlyScalar::new("p", parameter!("p")).unwrap()
        * &CacheOnlyScalar::new("k").unwrap();
    let dataset = Arc::new(Dataset::new(vec![
        Arc::new(test_event()),
        Arc::new(test_event()),
    ]));
    let evaluator = expr.load(&dataset).unwrap();
    let cached_integrals = evaluator
        .expression_precomputed_cached_integrals()
        .expect("integrals should exist");
    assert_eq!(cached_integrals.len(), 1);
    let gradient_terms = evaluator
        .expression_precomputed_cached_integral_gradient_terms(&[1.25])
        .expect("evaluation should succeed");
    assert_eq!(gradient_terms.len(), 1);
    assert_eq!(gradient_terms[0].weighted_gradient.len(), 1);
    assert_relative_eq!(
        gradient_terms[0].weighted_gradient[0].re,
        cached_integrals[0].weighted_cache_sum.re,
        epsilon = 1e-6
    );
    assert_relative_eq!(
        gradient_terms[0].weighted_gradient[0].im,
        cached_integrals[0].weighted_cache_sum.im,
        epsilon = 1e-6
    );
}
#[test]
fn test_expression_ir_precomputed_cached_integral_gradient_terms_empty_when_not_separable() {
    let expr = TestAmplitude::new("m", parameter!("mr"), parameter!("mi")).unwrap()
        * &CacheOnlyScalar::new("k").unwrap();
    let dataset = Arc::new(Dataset::new(vec![Arc::new(test_event())]));
    let evaluator = expr.load(&dataset).unwrap();
    assert!(evaluator
        .expression_precomputed_cached_integral_gradient_terms(&[0.1, -0.2])
        .expect("evaluation should succeed")
        .is_empty());
}
#[test]
fn test_expression_ir_lowered_cached_factor_programs_match_ir_cached_paths() {
    let expr = (ParameterOnlyScalar::new("p", parameter!("p")).unwrap()
        * &CacheOnlyScalar::new("k").unwrap())
        + &TestAmplitude::new("m", parameter!("mr"), parameter!("mi")).unwrap();
    let dataset = Arc::new(test_dataset());
    let evaluator = expr.load(&dataset).unwrap();
    let resources = evaluator.resources.read();
    let state = evaluator
        .ensure_cached_integral_cache_state(&resources)
        .expect("state should be available");
    let lowered_artifacts = evaluator.active_lowered_artifacts().unwrap();
    let parameters = resources
        .parameter_map
        .assemble(&[0.55, 0.2, -0.15])
        .expect("parameters should assemble");

    let mut amplitude_values = vec![Complex64::ZERO; evaluator.amplitudes.len()];
    evaluator.fill_amplitude_values(
        &mut amplitude_values,
        &state.execution_sets.cached_parameter_amplitudes,
        &parameters,
        &resources.caches[0],
    );
    let cached_value_ir =
        evaluator.evaluate_cached_weighted_value_sum_ir(&state, &amplitude_values);
    let cached_value_lowered = evaluator
        .evaluate_cached_weighted_value_sum_lowered(
            &state,
            lowered_artifacts.as_ref(),
            &amplitude_values,
        )
        .expect("cached value lowering should succeed");
    assert_relative_eq!(cached_value_lowered, cached_value_ir, epsilon = 1e-12);

    let mut cached_parameter_mask = vec![false; evaluator.amplitudes.len()];
    for &index in &state.execution_sets.cached_parameter_amplitudes {
        cached_parameter_mask[index] = true;
    }
    let mut amplitude_gradients = (0..evaluator.amplitudes.len())
        .map(|_| DVector::zeros(parameters.len()))
        .collect::<Vec<_>>();
    evaluator.fill_amplitude_gradients(
        &mut amplitude_gradients,
        &cached_parameter_mask,
        &parameters,
        &resources.caches[0],
    );
    let cached_gradient_ir = evaluator.evaluate_cached_weighted_gradient_sum_ir(
        &state,
        &amplitude_values,
        &amplitude_gradients,
        parameters.len(),
    );
    let cached_gradient_lowered = evaluator
        .evaluate_cached_weighted_gradient_sum_lowered(
            &state,
            lowered_artifacts.as_ref(),
            &amplitude_values,
            &amplitude_gradients,
            parameters.len(),
        )
        .expect("cached gradient lowering should succeed");
    for (lowered, ir) in cached_gradient_lowered
        .iter()
        .zip(cached_gradient_ir.iter())
    {
        assert_relative_eq!(*lowered, *ir, epsilon = 1e-12);
    }
}
#[test]
fn test_expression_ir_lowered_residual_runtime_matches_zeroed_node_path() {
    let expr = (ParameterOnlyScalar::new("p", parameter!("p")).unwrap()
        * &CacheOnlyScalar::new("k").unwrap())
        + &TestAmplitude::new("m", parameter!("mr"), parameter!("mi")).unwrap();
    let dataset = Arc::new(test_dataset());
    let evaluator = expr.load(&dataset).unwrap();
    let resources = evaluator.resources.read();
    let state = evaluator
        .ensure_cached_integral_cache_state(&resources)
        .expect("state should be available");
    let lowered_artifacts = evaluator.active_lowered_artifacts().unwrap();
    let parameters = resources
        .parameter_map
        .assemble(&[0.55, 0.2, -0.15])
        .expect("parameters should assemble");

    let mut amplitude_values = vec![Complex64::ZERO; evaluator.amplitudes.len()];
    evaluator.fill_amplitude_values(
        &mut amplitude_values,
        &state.execution_sets.residual_amplitudes,
        &parameters,
        &resources.caches[0],
    );
    let residual_value_ir = evaluator.evaluate_residual_value_ir(&state, &amplitude_values);
    let residual_program = lowered_artifacts
        .residual_runtime
        .as_ref()
        .map(|runtime| runtime.value_program())
        .expect("residual value lowering should succeed");
    let mut value_slots = vec![Complex64::ZERO; residual_program.scratch_slots()];
    let residual_value_lowered =
        residual_program.evaluate_into(&amplitude_values, &mut value_slots);
    assert_relative_eq!(
        residual_value_lowered.re,
        residual_value_ir.re,
        epsilon = 1e-12
    );
    assert_relative_eq!(
        residual_value_lowered.im,
        residual_value_ir.im,
        epsilon = 1e-12
    );

    let mut residual_active_mask = vec![false; evaluator.amplitudes.len()];
    for &index in &state.execution_sets.residual_amplitudes {
        residual_active_mask[index] = true;
    }
    let mut amplitude_gradients = (0..evaluator.amplitudes.len())
        .map(|_| DVector::zeros(parameters.len()))
        .collect::<Vec<_>>();
    evaluator.fill_amplitude_gradients(
        &mut amplitude_gradients,
        &residual_active_mask,
        &parameters,
        &resources.caches[0],
    );
    let residual_gradient_ir = evaluator.evaluate_residual_gradient_ir(
        &state,
        &amplitude_values,
        &amplitude_gradients,
        parameters.len(),
    );

    let program = lowered_artifacts
        .residual_runtime
        .as_ref()
        .map(|runtime| runtime.gradient_program())
        .expect("gradient lowering should succeed");
    let mut value_slots = vec![Complex64::ZERO; program.scratch_slots()];
    let mut gradient_slots = vec![Complex64::ZERO; program.scratch_slots() * parameters.len()];
    let residual_gradient_lowered = program.evaluate_gradient_into_flat(
        &amplitude_values,
        &amplitude_gradients,
        &mut value_slots,
        &mut gradient_slots,
        parameters.len(),
    );

    for (lowered, ir) in residual_gradient_lowered
        .iter()
        .zip(residual_gradient_ir.iter())
    {
        assert_relative_eq!(lowered.re, ir.re, epsilon = 1e-12);
        assert_relative_eq!(lowered.im, ir.im, epsilon = 1e-12);
    }
}
#[test]
fn test_expression_ir_reuses_lowered_artifacts_when_dataset_key_changes() {
    let expr = (ParameterOnlyScalar::new("p", parameter!("p")).unwrap()
        * &CacheOnlyScalar::new("k").unwrap())
        + &TestAmplitude::new("m", parameter!("mr"), parameter!("mi")).unwrap();
    let dataset = Arc::new(test_dataset());
    let mut evaluator = expr.load(&dataset).unwrap();
    drop(dataset);

    assert_eq!(evaluator.specialization_cache_len(), 1);
    assert_eq!(evaluator.lowered_artifact_cache_len(), 1);

    evaluator.reset_expression_compile_metrics();
    evaluator.reset_expression_specialization_metrics();

    Arc::get_mut(&mut evaluator.dataset)
        .expect("evaluator should own dataset Arc in this test")
        .clear_events_local();

    let cached_integrals = evaluator
        .expression_precomputed_cached_integrals()
        .expect("integrals should exist");
    assert_eq!(cached_integrals.len(), 1);
    assert_eq!(cached_integrals[0].weighted_cache_sum, Complex64::ZERO);

    assert_eq!(evaluator.specialization_cache_len(), 2);
    assert_eq!(evaluator.lowered_artifact_cache_len(), 1);
    assert_eq!(
        evaluator.expression_specialization_metrics(),
        ExpressionSpecializationMetrics {
            cache_hits: 0,
            cache_misses: 1,
        }
    );

    let compile_metrics = evaluator.expression_compile_metrics();
    assert_eq!(compile_metrics.specialization_cache_hits, 0);
    assert_eq!(compile_metrics.specialization_cache_misses, 1);
    assert_eq!(compile_metrics.specialization_lowering_cache_hits, 1);
    assert_eq!(compile_metrics.specialization_lowering_cache_misses, 0);
    assert!(compile_metrics.specialization_ir_compile_nanos > 0);
    assert!(compile_metrics.specialization_cached_integrals_nanos > 0);
    assert_eq!(compile_metrics.specialization_lowering_nanos, 0);
}

#[test]
fn test_evaluate_weighted_gradient_sum_local_matches_eventwise_baseline() {
    let p1 = ParameterOnlyScalar::new("p1", parameter!("p1")).unwrap();
    let p2 = ParameterOnlyScalar::new("p2", parameter!("p2")).unwrap();
    let c1 = CacheOnlyScalar::new("c1").unwrap();
    let c2 = CacheOnlyScalar::new("c2").unwrap();
    let c3 = CacheOnlyScalar::new("c3").unwrap();
    let m1 = TestAmplitude::new("m1", parameter!("m1r"), parameter!("m1i")).unwrap();
    let expr = (&p1 * &c1) + &(&p2 * &c2) + &(&(&m1 * &p1) * &c3);
    let dataset = Arc::new(test_dataset());
    let evaluator = expr.load(&dataset).unwrap();
    assert_eq!(
        evaluator
            .expression_precomputed_cached_integrals()
            .expect("integrals should exist")
            .len(),
        2
    );
    let params = vec![0.2, -0.3, 1.1, -0.7];
    let expected = evaluator
        .evaluate_gradient_local(&params)
        .expect("evaluation should succeed")
        .iter()
        .zip(dataset.weights_local().iter())
        .fold(
            DVector::zeros(params.len()),
            |mut accum, (gradient, event)| {
                accum += gradient.map(|value| value.re).scale(*event);
                accum
            },
        );
    let actual = evaluator
        .evaluate_weighted_gradient_sum_local(&params)
        .expect("evaluation should succeed");
    for (actual_item, expected_item) in actual.iter().zip(expected.iter()) {
        assert_relative_eq!(*actual_item, *expected_item, epsilon = 1e-10);
    }
}

#[test]
fn test_evaluate_weighted_value_sum_local_matches_eventwise_baseline() {
    let p1 = ParameterOnlyScalar::new("p1", parameter!("p1")).unwrap();
    let p2 = ParameterOnlyScalar::new("p2", parameter!("p2")).unwrap();
    let c1 = CacheOnlyScalar::new("c1").unwrap();
    let c2 = CacheOnlyScalar::new("c2").unwrap();
    let c3 = CacheOnlyScalar::new("c3").unwrap();
    let m1 = TestAmplitude::new("m1", parameter!("m1r"), parameter!("m1i")).unwrap();
    let expr = (&p1 * &c1) + &(&p2 * &c2) + &(&(&m1 * &p1) * &c3);
    let dataset = Arc::new(test_dataset());
    let evaluator = expr.load(&dataset).unwrap();
    assert_eq!(
        evaluator
            .expression_precomputed_cached_integrals()
            .expect("integrals should exist")
            .len(),
        2
    );
    let params = vec![0.2, -0.3, 1.1, -0.7];
    let expected = evaluator
        .evaluate_local(&params)
        .expect("evaluation should succeed")
        .iter()
        .zip(dataset.weights_local().iter())
        .fold(0.0, |accum, (value, event)| accum + *event * value.re);
    let actual = evaluator
        .evaluate_weighted_value_sum_local(&params)
        .expect("evaluation should succeed");
    assert_relative_eq!(actual, expected, epsilon = 1e-10);
}

#[test]
fn test_weighted_sums_match_hardcoded_reference_values() {
    let p1 = ParameterOnlyScalar::new("p1", parameter!("p1")).unwrap();
    let p2 = ParameterOnlyScalar::new("p2", parameter!("p2")).unwrap();
    let c1 = CacheOnlyScalar::new("c1").unwrap();
    let c2 = CacheOnlyScalar::new("c2").unwrap();
    let c3 = CacheOnlyScalar::new("c3").unwrap();
    let m1 = TestAmplitude::new("m1", parameter!("m1r"), parameter!("m1i")).unwrap();
    let expr = (&p1 * &c1) + &(&p2 * &c2) + &(&(&m1 * &p1) * &c3);

    let metadata = Arc::new(DatasetMetadata::default());
    let dataset = Arc::new(Dataset::new_with_metadata(
        vec![
            Arc::new(EventData {
                p4s: vec![Vec4::new(0.0, 0.0, 0.0, 2.0)],
                aux: vec![],
                weight: 0.5,
            }),
            Arc::new(EventData {
                p4s: vec![Vec4::new(0.0, 0.0, 0.0, 3.0)],
                aux: vec![],
                weight: -1.25,
            }),
            Arc::new(EventData {
                p4s: vec![Vec4::new(0.0, 0.0, 0.0, 5.0)],
                aux: vec![],
                weight: 2.0,
            }),
        ],
        metadata,
    ));
    let evaluator = expr.load(&dataset).unwrap();
    let params = vec![0.7, -1.1, 0.9, -0.4];

    let weighted_value_sum = evaluator
        .evaluate_weighted_value_sum_local(&params)
        .expect("evaluation should succeed");
    assert_relative_eq!(weighted_value_sum, 22.7725, epsilon = 1e-12);

    let weighted_gradient_sum = evaluator
        .evaluate_weighted_gradient_sum_local(&params)
        .expect("evaluation should succeed");
    let free_parameters = evaluator
        .parameters()
        .free()
        .names()
        .into_iter()
        .map(|name| name.to_string())
        .collect::<Vec<_>>();
    assert_eq!(free_parameters, vec!["p1", "p2", "m1r", "m1i"]);
    let expected_gradient = [43.925, 7.25, 28.525, 0.0];
    assert_eq!(weighted_gradient_sum.len(), expected_gradient.len());
    for (actual, expected) in weighted_gradient_sum.iter().zip(expected_gradient.iter()) {
        assert_relative_eq!(*actual, *expected, epsilon = 1e-9);
    }
}
#[test]
fn test_evaluate_weighted_gradient_sum_local_respects_signed_cached_terms() {
    let expr = Expression::one()
        - &(ParameterOnlyScalar::new("p", parameter!("p")).unwrap()
            * &CacheOnlyScalar::new("k").unwrap());
    let dataset = Arc::new(test_dataset());
    let evaluator = expr.load(&dataset).unwrap();
    assert_eq!(
        evaluator
            .expression_precomputed_cached_integrals()
            .expect("integrals should exist")
            .len(),
        1
    );
    assert_eq!(
        evaluator
            .expression_precomputed_cached_integrals()
            .expect("integrals should exist")[0]
            .coefficient,
        -1
    );
    let params = vec![0.75];
    let expected = evaluator
        .evaluate_gradient_local(&params)
        .expect("evaluation should succeed")
        .iter()
        .zip(dataset.weights_local().iter())
        .fold(
            DVector::zeros(params.len()),
            |mut accum, (gradient, event)| {
                accum += gradient.map(|value| value.re).scale(*event);
                accum
            },
        );
    let actual = evaluator
        .evaluate_weighted_gradient_sum_local(&params)
        .expect("evaluation should succeed");
    for (actual_item, expected_item) in actual.iter().zip(expected.iter()) {
        assert_relative_eq!(*actual_item, *expected_item, epsilon = 1e-10);
    }
}
#[test]
fn test_evaluate_weighted_value_sum_local_respects_signed_cached_terms() {
    let expr = Expression::one()
        - &(ParameterOnlyScalar::new("p", parameter!("p")).unwrap()
            * &CacheOnlyScalar::new("k").unwrap());
    let dataset = Arc::new(test_dataset());
    let evaluator = expr.load(&dataset).unwrap();
    assert_eq!(
        evaluator
            .expression_precomputed_cached_integrals()
            .expect("integrals should exist")
            .len(),
        1
    );
    assert_eq!(
        evaluator
            .expression_precomputed_cached_integrals()
            .expect("integrals should exist")[0]
            .coefficient,
        -1
    );
    let params = vec![0.75];
    let expected = evaluator
        .evaluate_local(&params)
        .expect("evaluation should succeed")
        .iter()
        .zip(dataset.weights_local().iter())
        .fold(0.0, |accum, (value, event)| accum + *event * value.re);
    let actual = evaluator
        .evaluate_weighted_value_sum_local(&params)
        .expect("evaluation should succeed");
    assert_relative_eq!(actual, expected, epsilon = 1e-10);
}
#[test]
fn test_expression_ir_diagnostics_follow_activation_changes() {
    let expr = (ParameterOnlyScalar::new("p", parameter!("p")).unwrap()
        * &CacheOnlyScalar::new("k").unwrap())
        + &TestAmplitude::new("m", parameter!("mr"), parameter!("mi")).unwrap();
    let dataset = Arc::new(Dataset::new(vec![Arc::new(test_event())]));
    let evaluator = expr.load(&dataset).unwrap();

    let all_active = evaluator
        .expression_normalization_plan_explain()
        .expect("plan should exist");
    assert_eq!(all_active.cached_separable_nodes.len(), 1);
    assert_eq!(
        evaluator
            .expression_root_dependence()
            .expect("root dependence should exist"),
        ExpressionDependence::Mixed
    );

    evaluator.isolate_many(&["p"]);
    let param_only = evaluator
        .expression_normalization_plan_explain()
        .expect("plan should exist");
    assert!(param_only.cached_separable_nodes.is_empty());
    assert_eq!(
        evaluator
            .expression_root_dependence()
            .expect("root dependence should exist"),
        ExpressionDependence::ParameterOnly
    );
}
#[test]
fn test_expression_ir_specialization_cache_reuses_prior_mask_specializations() {
    let expr = (ParameterOnlyScalar::new("p", parameter!("p")).unwrap()
        * &CacheOnlyScalar::new("k").unwrap())
        + &TestAmplitude::new("m", parameter!("mr"), parameter!("mi")).unwrap();
    let dataset = Arc::new(Dataset::new(vec![Arc::new(test_event())]));
    let evaluator = expr.load(&dataset).unwrap();

    let initial_compile_metrics = evaluator.expression_compile_metrics();
    assert!(initial_compile_metrics.initial_ir_compile_nanos > 0);
    assert!(initial_compile_metrics.initial_cached_integrals_nanos > 0);
    assert!(initial_compile_metrics.initial_lowering_nanos > 0);
    assert_eq!(initial_compile_metrics.specialization_cache_hits, 0);
    assert_eq!(initial_compile_metrics.specialization_cache_misses, 0);
    assert_eq!(
        initial_compile_metrics.specialization_lowering_cache_hits,
        0
    );
    assert_eq!(
        initial_compile_metrics.specialization_lowering_cache_misses,
        1
    );

    assert_eq!(evaluator.specialization_cache_len(), 1);
    assert_eq!(evaluator.lowered_artifact_cache_len(), 1);
    assert_eq!(
        evaluator.expression_specialization_metrics(),
        ExpressionSpecializationMetrics {
            cache_hits: 0,
            cache_misses: 1,
        }
    );
    let all_active_cached_integrals = evaluator
        .expression_precomputed_cached_integrals()
        .expect("integrals should exist");

    evaluator.isolate_many(&["p"]);
    assert_eq!(evaluator.specialization_cache_len(), 2);
    assert_eq!(
        evaluator.expression_specialization_metrics(),
        ExpressionSpecializationMetrics {
            cache_hits: 0,
            cache_misses: 2,
        }
    );
    let after_cache_miss_metrics = evaluator.expression_compile_metrics();
    assert_eq!(after_cache_miss_metrics.specialization_cache_hits, 0);
    assert_eq!(after_cache_miss_metrics.specialization_cache_misses, 1);
    assert_eq!(
        after_cache_miss_metrics.specialization_lowering_cache_hits,
        0
    );
    assert_eq!(
        after_cache_miss_metrics.specialization_lowering_cache_misses,
        2
    );
    assert!(after_cache_miss_metrics.specialization_ir_compile_nanos > 0);
    assert!(after_cache_miss_metrics.specialization_cached_integrals_nanos > 0);
    assert!(after_cache_miss_metrics.specialization_lowering_nanos > 0);
    assert!(evaluator
        .expression_precomputed_cached_integrals()
        .expect("integrals should exist")
        .is_empty());

    evaluator.activate_many(&["k", "m"]);
    assert_eq!(evaluator.specialization_cache_len(), 2);
    assert_eq!(
        evaluator.expression_specialization_metrics(),
        ExpressionSpecializationMetrics {
            cache_hits: 1,
            cache_misses: 2,
        }
    );
    assert_eq!(
        evaluator
            .expression_precomputed_cached_integrals()
            .expect("integrals should exist"),
        all_active_cached_integrals
    );
    let after_cache_hit_metrics = evaluator.expression_compile_metrics();
    assert_eq!(after_cache_hit_metrics.specialization_cache_hits, 1);
    assert_eq!(after_cache_hit_metrics.specialization_cache_misses, 1);
    assert_eq!(
        after_cache_hit_metrics.specialization_lowering_cache_hits,
        0
    );
    assert_eq!(
        after_cache_hit_metrics.specialization_lowering_cache_misses,
        2
    );
    assert!(after_cache_hit_metrics.specialization_cache_restore_nanos > 0);
}

#[test]
fn test_weighted_sums_match_baseline_after_activation_changes() {
    let p1 = ParameterOnlyScalar::new("p1", parameter!("p1")).unwrap();
    let p2 = ParameterOnlyScalar::new("p2", parameter!("p2")).unwrap();
    let c1 = CacheOnlyScalar::new("c1").unwrap();
    let c2 = CacheOnlyScalar::new("c2").unwrap();
    let c3 = CacheOnlyScalar::new("c3").unwrap();
    let m1 = TestAmplitude::new("m1", parameter!("m1r"), parameter!("m1i")).unwrap();
    let expr = (&p1 * &c1) + &(&p2 * &c2) + &(&(&m1 * &p1) * &c3);
    let dataset = Arc::new(test_dataset());
    let evaluator = expr.load(&dataset).unwrap();
    let params = vec![0.2, -0.3, 1.1, -0.7];

    evaluator.isolate_many(&["p1", "c1", "m1", "c3"]);

    let expected_value = evaluator
        .evaluate_local(&params)
        .expect("evaluation should succeed")
        .iter()
        .zip(dataset.weights_local().iter())
        .fold(0.0, |accum, (value, event)| accum + *event * value.re);
    assert_relative_eq!(
        evaluator
            .evaluate_weighted_value_sum_local(&params)
            .expect("evaluation should succeed"),
        expected_value,
        epsilon = 1e-10
    );
}

#[test]
fn test_evaluate_local_does_not_depend_on_dataset_rows() {
    let expr = TestAmplitude::new("test", parameter!("real"), parameter!("imag"))
        .unwrap()
        .norm_sqr();
    let mut event1 = test_event();
    event1.p4s[0].t = 7.5;
    let mut event2 = test_event();
    event2.p4s[0].t = 8.25;
    let mut event3 = test_event();
    event3.p4s[0].t = 9.0;
    let dataset = Arc::new(Dataset::new_with_metadata(
        vec![Arc::new(event1), Arc::new(event2), Arc::new(event3)],
        Arc::new(DatasetMetadata::default()),
    ));
    let mut evaluator = expr.load(&dataset).unwrap();
    drop(dataset);
    let expected_len = evaluator.resources.read().caches.len();
    Arc::get_mut(&mut evaluator.dataset)
        .expect("evaluator should own dataset Arc in this test")
        .clear_events_local();
    let cached = evaluator
        .evaluate_local(&[1.25, -0.75])
        .expect("evaluation should succeed");
    assert_eq!(cached.len(), expected_len);
}

#[test]
fn test_evaluate_gradient_local_does_not_depend_on_dataset_rows() {
    let expr = TestAmplitude::new("test", parameter!("real"), parameter!("imag"))
        .unwrap()
        .norm_sqr();
    let mut event1 = test_event();
    event1.p4s[0].t = 7.5;
    let mut event2 = test_event();
    event2.p4s[0].t = 8.25;
    let mut event3 = test_event();
    event3.p4s[0].t = 9.0;
    let dataset = Arc::new(Dataset::new_with_metadata(
        vec![Arc::new(event1), Arc::new(event2), Arc::new(event3)],
        Arc::new(DatasetMetadata::default()),
    ));
    let mut evaluator = expr.load(&dataset).unwrap();
    drop(dataset);
    let expected_len = evaluator.resources.read().caches.len();
    Arc::get_mut(&mut evaluator.dataset)
        .expect("evaluator should own dataset Arc in this test")
        .clear_events_local();
    let cached = evaluator
        .evaluate_gradient_local(&[1.25, -0.75])
        .expect("evaluation should succeed");
    assert_eq!(cached.len(), expected_len);
}

#[test]
fn test_evaluate_with_gradient_local_matches_separate_paths() {
    let expr = TestAmplitude::new("test", parameter!("real"), parameter!("imag"))
        .unwrap()
        .norm_sqr();
    let dataset = Arc::new(Dataset::new(vec![
        Arc::new(test_event()),
        Arc::new(test_event()),
        Arc::new(test_event()),
    ]));
    let evaluator = expr.load(&dataset).unwrap();
    let params = [1.25, -0.75];
    let values = evaluator
        .evaluate_local(&params)
        .expect("evaluation should succeed");
    let gradients = evaluator
        .evaluate_gradient_local(&params)
        .expect("evaluation should succeed");
    let fused = evaluator
        .evaluate_with_gradient_local(&params)
        .expect("evaluation should succeed");
    assert_eq!(fused.len(), values.len());
    assert_eq!(fused.len(), gradients.len());
    for ((value_gradient, value), gradient) in fused.iter().zip(values.iter()).zip(gradients.iter())
    {
        let (fused_value, fused_gradient) = value_gradient;
        assert_relative_eq!(fused_value.re, value.re, epsilon = 1e-12);
        assert_relative_eq!(fused_value.im, value.im, epsilon = 1e-12);
        assert_eq!(fused_gradient.len(), gradient.len());
        for (fused_item, item) in fused_gradient.iter().zip(gradient.iter()) {
            assert_relative_eq!(fused_item.re, item.re, epsilon = 1e-12);
            assert_relative_eq!(fused_item.im, item.im, epsilon = 1e-12);
        }
    }
}

#[test]
fn test_evaluate_with_gradient_batch_local_matches_separate_paths() {
    let expr = TestAmplitude::new("test", parameter!("real"), parameter!("imag"))
        .unwrap()
        .norm_sqr();
    let dataset = Arc::new(Dataset::new(vec![
        Arc::new(test_event()),
        Arc::new(test_event()),
        Arc::new(test_event()),
        Arc::new(test_event()),
    ]));
    let evaluator = expr.load(&dataset).unwrap();
    let params = [0.5, -1.25];
    let indices = vec![0, 2, 3];
    let values = evaluator
        .evaluate_batch_local(&params, &indices)
        .expect("evaluation should succeed");
    let gradients = evaluator
        .evaluate_gradient_batch_local(&params, &indices)
        .expect("evaluation should succeed");
    let fused = evaluator
        .evaluate_with_gradient_batch_local(&params, &indices)
        .expect("evaluation should succeed");
    assert_eq!(fused.len(), values.len());
    assert_eq!(fused.len(), gradients.len());
    for ((value_gradient, value), gradient) in fused.iter().zip(values.iter()).zip(gradients.iter())
    {
        let (fused_value, fused_gradient) = value_gradient;
        assert_relative_eq!(fused_value.re, value.re, epsilon = 1e-12);
        assert_relative_eq!(fused_value.im, value.im, epsilon = 1e-12);
        assert_eq!(fused_gradient.len(), gradient.len());
        for (fused_item, item) in fused_gradient.iter().zip(gradient.iter()) {
            assert_relative_eq!(fused_item.re, item.re, epsilon = 1e-12);
            assert_relative_eq!(fused_item.im, item.im, epsilon = 1e-12);
        }
    }
}

#[test]
fn test_precompute_all_columnar_populates_cache() {
    let mut event1 = test_event();
    event1.p4s[0].t = 7.5;
    let mut event2 = test_event();
    event2.p4s[0].t = 8.25;
    let mut event3 = test_event();
    event3.p4s[0].t = 9.0;
    let dataset = Dataset::new_with_metadata(
        vec![Arc::new(event1), Arc::new(event2), Arc::new(event3)],
        Arc::new(DatasetMetadata::default()),
    );
    let mut amplitude = TestAmplitude {
        tags: Tags::new(["test"]),
        re: parameter!("real"),
        pid_re: ParameterID::default(),
        im: parameter!("imag"),
        pid_im: ParameterID::default(),
        beam_energy: Default::default(),
    };
    let mut resources = Resources::default();
    amplitude
        .register(&mut resources)
        .expect("test amplitude should register");
    resources.reserve_cache(dataset.n_events());
    amplitude.precompute_all(&dataset, &mut resources);
    for cache in &resources.caches {
        assert!(cache.get_scalar(amplitude.beam_energy) > 0.0);
    }
}

#[cfg(feature = "mpi")]
#[mpi_test(np = [2])]
fn test_load_reserves_local_cache_size_in_mpi() {
    use crate::mpi::{finalize_mpi, get_world, use_mpi};

    use_mpi(true);
    assert!(get_world().is_some(), "MPI world should be initialized");

    let expr = ComplexScalar::new(
        "constant",
        parameter!("const_re", 2.0),
        parameter!("const_im", 3.0),
    )
    .expect("constant amplitude should construct");
    let events = vec![
        Arc::new(test_event()),
        Arc::new(test_event()),
        Arc::new(test_event()),
        Arc::new(test_event()),
    ];
    let dataset = Arc::new(Dataset::new_with_metadata(
        events,
        Arc::new(DatasetMetadata::default()),
    ));
    let evaluator = expr.load(&dataset).expect("evaluator should load");
    let local_events = dataset.n_events_local();
    let cache_len = evaluator.resources.read().caches.len();

    assert_eq!(
        cache_len, local_events,
        "cache length must match local event count under MPI"
    );
    finalize_mpi();
}

#[cfg(feature = "mpi")]
#[mpi_test(np = [2])]
fn test_expression_ir_cached_integrals_are_rank_local_in_mpi() {
    use mpi::{collective::SystemOperation, topology::Communicator, traits::*};

    use crate::mpi::{finalize_mpi, get_world, use_mpi};

    use_mpi(true);
    let world = get_world().expect("MPI world should be initialized");

    let expr = ParameterOnlyScalar::new("p", parameter!("p")).unwrap()
        * &CacheOnlyScalar::new("k").unwrap();
    let events = vec![
        Arc::new(EventData {
            p4s: vec![Vec4::new(0.0, 0.0, 0.0, 1.0)],
            aux: vec![],
            weight: 0.5,
        }),
        Arc::new(EventData {
            p4s: vec![Vec4::new(0.0, 0.0, 0.0, 2.0)],
            aux: vec![],
            weight: 1.0,
        }),
        Arc::new(EventData {
            p4s: vec![Vec4::new(0.0, 0.0, 0.0, 3.0)],
            aux: vec![],
            weight: 1.5,
        }),
        Arc::new(EventData {
            p4s: vec![Vec4::new(0.0, 0.0, 0.0, 4.0)],
            aux: vec![],
            weight: 2.0,
        }),
        Arc::new(EventData {
            p4s: vec![Vec4::new(0.0, 0.0, 0.0, 5.0)],
            aux: vec![],
            weight: 2.5,
        }),
        Arc::new(EventData {
            p4s: vec![Vec4::new(0.0, 0.0, 0.0, 6.0)],
            aux: vec![],
            weight: 3.0,
        }),
        Arc::new(EventData {
            p4s: vec![Vec4::new(0.0, 0.0, 0.0, 7.0)],
            aux: vec![],
            weight: 3.5,
        }),
        Arc::new(EventData {
            p4s: vec![Vec4::new(0.0, 0.0, 0.0, 8.0)],
            aux: vec![],
            weight: 4.0,
        }),
    ];
    let dataset = Arc::new(Dataset::new_with_metadata(
        events,
        Arc::new(DatasetMetadata::default()),
    ));
    let evaluator = expr.load(&dataset).expect("evaluator should load");
    let cached_integrals = evaluator
        .expression_precomputed_cached_integrals()
        .expect("integrals should exist");
    assert_eq!(cached_integrals.len(), 1);

    let local_expected =
        dataset
            .weights_local()
            .iter()
            .enumerate()
            .fold(0.0, |acc, (index, weight)| {
                let event = dataset.event_local(index).expect("event should exist");
                acc + *weight * event.p4_at(0).e()
            });
    let cached_local = cached_integrals[0].weighted_cache_sum;
    assert_relative_eq!(cached_local.re, local_expected, epsilon = 1e-12);
    assert_relative_eq!(cached_local.im, 0.0, epsilon = 1e-12);

    let weighted_value_sum = evaluator
        .evaluate_weighted_value_sum_local(&[2.0])
        .expect("evaluate should succeed");
    assert_relative_eq!(weighted_value_sum, 2.0 * local_expected, epsilon = 1e-10);

    let mut global_expected = 0.0;
    world.all_reduce_into(
        &local_expected,
        &mut global_expected,
        SystemOperation::sum(),
    );
    if world.size() > 1 {
        assert!(
            (cached_local.re - global_expected).abs() > 1e-12,
            "cached integral should remain rank-local before MPI reduction"
        );
    }
    finalize_mpi();
}

#[cfg(feature = "mpi")]
#[mpi_test(np = [2])]
fn test_expression_ir_weighted_sum_mpi_matches_global_eventwise_baseline() {
    use mpi::{collective::SystemOperation, traits::*};

    use crate::mpi::{finalize_mpi, get_world, use_mpi};

    use_mpi(true);
    let world = get_world().expect("MPI world should be initialized");

    let p1 = ParameterOnlyScalar::new("p1", parameter!("p1")).unwrap();
    let p2 = ParameterOnlyScalar::new("p2", parameter!("p2")).unwrap();
    let c1 = CacheOnlyScalar::new("c1").unwrap();
    let c2 = CacheOnlyScalar::new("c2").unwrap();
    let c3 = CacheOnlyScalar::new("c3").unwrap();
    let m1 = TestAmplitude::new("m1", parameter!("m1r"), parameter!("m1i")).unwrap();
    let expr = (&p1 * &c1) + &(&p2 * &c2) + &(&(&m1 * &p1) * &c3);
    let events = vec![
        Arc::new(EventData {
            p4s: vec![Vec4::new(0.0, 0.0, 0.0, 1.0)],
            aux: vec![],
            weight: 0.5,
        }),
        Arc::new(EventData {
            p4s: vec![Vec4::new(0.0, 0.0, 0.0, 2.0)],
            aux: vec![],
            weight: -1.25,
        }),
        Arc::new(EventData {
            p4s: vec![Vec4::new(0.0, 0.0, 0.0, 3.0)],
            aux: vec![],
            weight: 0.75,
        }),
        Arc::new(EventData {
            p4s: vec![Vec4::new(0.0, 0.0, 0.0, 4.0)],
            aux: vec![],
            weight: 1.5,
        }),
        Arc::new(EventData {
            p4s: vec![Vec4::new(0.0, 0.0, 0.0, 5.0)],
            aux: vec![],
            weight: 2.25,
        }),
        Arc::new(EventData {
            p4s: vec![Vec4::new(0.0, 0.0, 0.0, 6.0)],
            aux: vec![],
            weight: -0.5,
        }),
        Arc::new(EventData {
            p4s: vec![Vec4::new(0.0, 0.0, 0.0, 7.0)],
            aux: vec![],
            weight: 3.5,
        }),
        Arc::new(EventData {
            p4s: vec![Vec4::new(0.0, 0.0, 0.0, 8.0)],
            aux: vec![],
            weight: 1.25,
        }),
    ];
    let dataset = Arc::new(Dataset::new_with_metadata(
        events,
        Arc::new(DatasetMetadata::default()),
    ));
    let evaluator = expr.load(&dataset).expect("evaluator should load");
    let params = vec![0.2, -0.3, 1.1, -0.7];

    let local_expected_value = evaluator
        .evaluate_local(&params)
        .expect("evaluate should succeed")
        .iter()
        .zip(dataset.weights_local().iter())
        .fold(0.0, |accum, (value, event)| accum + *event * value.re);
    let mut global_expected_value = 0.0;
    world.all_reduce_into(
        &local_expected_value,
        &mut global_expected_value,
        SystemOperation::sum(),
    );
    let mpi_value = evaluator
        .evaluate_weighted_value_sum_mpi(&params, &world)
        .expect("evaluate should succeed");
    assert_relative_eq!(mpi_value, global_expected_value, epsilon = 1e-10);

    let local_expected_gradient = evaluator
        .evaluate_gradient_local(&params)
        .expect("evaluate should succeed")
        .iter()
        .zip(dataset.weights_local().iter())
        .fold(
            DVector::zeros(params.len()),
            |mut accum, (gradient, event)| {
                accum += gradient.map(|value| value.re).scale(*event);
                accum
            },
        );
    let mut global_expected_gradient = vec![0.0; local_expected_gradient.len()];
    world.all_reduce_into(
        local_expected_gradient.as_slice(),
        &mut global_expected_gradient,
        SystemOperation::sum(),
    );
    let mpi_gradient = evaluator
        .evaluate_weighted_gradient_sum_mpi(&params, &world)
        .expect("evaluate should succeed");
    for (actual, expected) in mpi_gradient.iter().zip(global_expected_gradient.iter()) {
        assert_relative_eq!(*actual, *expected, epsilon = 1e-10);
    }

    finalize_mpi();
}

#[test]
fn test_evaluate_local_succeeds_for_constant_amplitude() {
    let expr = ComplexScalar::new(
        "constant",
        parameter!("const_re", 2.0),
        parameter!("const_im", 3.0),
    )
    .unwrap();
    let dataset = Arc::new(Dataset::new_with_metadata(
        vec![Arc::new(test_event())],
        Arc::new(DatasetMetadata::default()),
    ));
    let evaluator = expr.load(&dataset).unwrap();
    let values = evaluator
        .evaluate_local(&[])
        .expect("evaluation should succeed");
    assert_eq!(values.len(), 1);
    let gradients = evaluator
        .evaluate_gradient_local(&[])
        .expect("evaluation should succeed");
    assert_eq!(gradients.len(), 1);
}

#[test]
fn test_constant_amplitude() {
    let expr = ComplexScalar::new(
        "constant",
        parameter!("const_re", 2.0),
        parameter!("const_im", 3.0),
    )
    .unwrap();
    let dataset = Arc::new(Dataset::new_with_metadata(
        vec![Arc::new(test_event())],
        Arc::new(DatasetMetadata::default()),
    ));
    let evaluator = expr.load(&dataset).unwrap();
    let result = evaluator.evaluate(&[]).expect("evaluation should succeed");
    assert_eq!(result[0], Complex64::new(2.0, 3.0));
}

#[test]
fn test_parametric_amplitude() {
    let expr = ComplexScalar::new(
        "parametric",
        parameter!("test_param_re"),
        parameter!("test_param_im"),
    )
    .unwrap();
    let dataset = Arc::new(test_dataset());
    let evaluator = expr.load(&dataset).unwrap();
    let result = evaluator
        .evaluate(&[2.0, 3.0])
        .expect("evaluation should succeed");
    assert_eq!(result[0], Complex64::new(2.0, 3.0));
}

#[test]
fn test_expression_operations() {
    let expr1 = ComplexScalar::new(
        "const1",
        parameter!("const1_re", 2.0),
        parameter!("const1_im", 0.0),
    )
    .unwrap();
    let expr2 = ComplexScalar::new(
        "const2",
        parameter!("const2_re", 0.0),
        parameter!("const2_im", 1.0),
    )
    .unwrap();
    let expr3 = ComplexScalar::new(
        "const3",
        parameter!("const3_re", 3.0),
        parameter!("const3_im", 4.0),
    )
    .unwrap();

    let dataset = Arc::new(test_dataset());

    // Test (amp) addition
    let expr_add = &expr1 + &expr2;
    let result_add = expr_add
        .load(&dataset)
        .unwrap()
        .evaluate(&[])
        .expect("evaluation should succeed");
    assert_eq!(result_add[0], Complex64::new(2.0, 1.0));

    // Test (amp) subtraction
    let expr_sub = &expr1 - &expr2;
    let result_sub = expr_sub
        .load(&dataset)
        .unwrap()
        .evaluate(&[])
        .expect("evaluation should succeed");
    assert_eq!(result_sub[0], Complex64::new(2.0, -1.0));

    // Test (amp) multiplication
    let expr_mul = &expr1 * &expr2;
    let result_mul = expr_mul
        .load(&dataset)
        .unwrap()
        .evaluate(&[])
        .expect("evaluation should succeed");
    assert_eq!(result_mul[0], Complex64::new(0.0, 2.0));

    // Test (amp) division
    let expr_div = &expr1 / &expr3;
    let result_div = expr_div
        .load(&dataset)
        .unwrap()
        .evaluate(&[])
        .expect("evaluation should succeed");
    assert_eq!(result_div[0], Complex64::new(6.0 / 25.0, -8.0 / 25.0));

    // Test (amp) neg
    let expr_neg = -&expr3;
    let result_neg = expr_neg
        .load(&dataset)
        .unwrap()
        .evaluate(&[])
        .expect("evaluation should succeed");
    assert_eq!(result_neg[0], Complex64::new(-3.0, -4.0));

    // Test (expr) addition
    let expr_add2 = &expr_add + &expr_mul;
    let result_add2 = expr_add2
        .load(&dataset)
        .unwrap()
        .evaluate(&[])
        .expect("evaluation should succeed");
    assert_eq!(result_add2[0], Complex64::new(2.0, 3.0));

    // Test (expr) subtraction
    let expr_sub2 = &expr_add - &expr_mul;
    let result_sub2 = expr_sub2
        .load(&dataset)
        .unwrap()
        .evaluate(&[])
        .expect("evaluation should succeed");
    assert_eq!(result_sub2[0], Complex64::new(2.0, -1.0));

    // Test (expr) multiplication
    let expr_mul2 = &expr_add * &expr_mul;
    let result_mul2 = expr_mul2
        .load(&dataset)
        .unwrap()
        .evaluate(&[])
        .expect("evaluation should succeed");
    assert_eq!(result_mul2[0], Complex64::new(-2.0, 4.0));

    // Test (expr) division
    let expr_div2 = &expr_add / &expr_add2;
    let result_div2 = expr_div2
        .load(&dataset)
        .unwrap()
        .evaluate(&[])
        .expect("evaluation should succeed");
    assert_eq!(result_div2[0], Complex64::new(7.0 / 13.0, -4.0 / 13.0));

    // Test (expr) neg
    let expr_neg2 = -&expr_mul2;
    let result_neg2 = expr_neg2
        .load(&dataset)
        .unwrap()
        .evaluate(&[])
        .expect("evaluation should succeed");
    assert_eq!(result_neg2[0], Complex64::new(2.0, -4.0));

    // Test (amp) real
    let expr_real = expr3.real();
    let result_real = expr_real
        .load(&dataset)
        .unwrap()
        .evaluate(&[])
        .expect("evaluation should succeed");
    assert_eq!(result_real[0], Complex64::new(3.0, 0.0));

    // Test (expr) real
    let expr_mul2_real = expr_mul2.real();
    let result_mul2_real = expr_mul2_real
        .load(&dataset)
        .unwrap()
        .evaluate(&[])
        .expect("evaluation should succeed");
    assert_eq!(result_mul2_real[0], Complex64::new(-2.0, 0.0));

    // Test (amp) imag
    let expr_imag = expr3.imag();
    let result_imag = expr_imag
        .load(&dataset)
        .unwrap()
        .evaluate(&[])
        .expect("evaluation should succeed");
    assert_eq!(result_imag[0], Complex64::new(4.0, 0.0));

    // Test (expr) imag
    let expr_mul2_imag = expr_mul2.imag();
    let result_mul2_imag = expr_mul2_imag
        .load(&dataset)
        .unwrap()
        .evaluate(&[])
        .expect("evaluation should succeed");
    assert_eq!(result_mul2_imag[0], Complex64::new(4.0, 0.0));

    // Test (amp) conj
    let expr_conj = expr3.conj();
    let result_conj = expr_conj
        .load(&dataset)
        .unwrap()
        .evaluate(&[])
        .expect("evaluation should succeed");
    assert_eq!(result_conj[0], Complex64::new(3.0, -4.0));

    // Test (expr) conj
    let expr_mul2_conj = expr_mul2.conj();
    let result_mul2_conj = expr_mul2_conj
        .load(&dataset)
        .unwrap()
        .evaluate(&[])
        .expect("evaluation should succeed");
    assert_eq!(result_mul2_conj[0], Complex64::new(-2.0, -4.0));

    // Test (amp) norm_sqr
    let expr_norm = expr1.norm_sqr();
    let result_norm = expr_norm
        .load(&dataset)
        .unwrap()
        .evaluate(&[])
        .expect("evaluation should succeed");
    assert_eq!(result_norm[0], Complex64::new(4.0, 0.0));

    // Test (expr) norm_sqr
    let expr_mul2_norm = expr_mul2.norm_sqr();
    let result_mul2_norm = expr_mul2_norm
        .load(&dataset)
        .unwrap()
        .evaluate(&[])
        .expect("evaluation should succeed");
    assert_eq!(result_mul2_norm[0], Complex64::new(20.0, 0.0));
}

#[test]
fn test_amplitude_activation() {
    let expr1 = ComplexScalar::new(
        "const1",
        parameter!("const1_re_act", 1.0),
        parameter!("const1_im_act", 0.0),
    )
    .unwrap();
    let expr2 = ComplexScalar::new(
        "const2",
        parameter!("const2_re_act", 2.0),
        parameter!("const2_im_act", 0.0),
    )
    .unwrap();

    let dataset = Arc::new(test_dataset());
    let expr = &expr1 + &expr2;
    let evaluator = expr.load(&dataset).unwrap();

    // Test initial state (all active)
    let result = evaluator.evaluate(&[]).expect("evaluation should succeed");
    assert_eq!(result[0], Complex64::new(3.0, 0.0));

    // Test deactivation
    evaluator.deactivate_strict("const1").unwrap();
    let result = evaluator.evaluate(&[]).expect("evaluation should succeed");
    assert_eq!(result[0], Complex64::new(2.0, 0.0));

    // Test isolation
    evaluator.isolate_strict("const1").unwrap();
    let result = evaluator.evaluate(&[]).expect("evaluation should succeed");
    assert_eq!(result[0], Complex64::new(1.0, 0.0));

    // Test reactivation
    evaluator.activate_all();
    let result = evaluator.evaluate(&[]).expect("evaluation should succeed");
    assert_eq!(result[0], Complex64::new(3.0, 0.0));
}

#[test]
fn test_gradient() {
    let expr1 = ComplexScalar::new(
        "parametric_1",
        parameter!("test_param_re_1"),
        parameter!("test_param_im_1"),
    )
    .unwrap();
    let expr2 = ComplexScalar::new(
        "parametric_2",
        parameter!("test_param_re_2"),
        parameter!("test_param_im_2"),
    )
    .unwrap();

    let dataset = Arc::new(test_dataset());
    let params = vec![2.0, 3.0, 4.0, 5.0];

    let expr = &expr1 + &expr2;
    let evaluator = expr.load(&dataset).unwrap();

    let gradient = evaluator
        .evaluate_gradient(&params)
        .expect("evaluation should succeed");

    assert_relative_eq!(gradient[0][0].re, 1.0);
    assert_relative_eq!(gradient[0][0].im, 0.0);
    assert_relative_eq!(gradient[0][1].re, 0.0);
    assert_relative_eq!(gradient[0][1].im, 1.0);
    assert_relative_eq!(gradient[0][2].re, 1.0);
    assert_relative_eq!(gradient[0][2].im, 0.0);
    assert_relative_eq!(gradient[0][3].re, 0.0);
    assert_relative_eq!(gradient[0][3].im, 1.0);

    let expr = &expr1 - &expr2;
    let evaluator = expr.load(&dataset).unwrap();

    let gradient = evaluator
        .evaluate_gradient(&params)
        .expect("evaluation should succeed");

    assert_relative_eq!(gradient[0][0].re, 1.0);
    assert_relative_eq!(gradient[0][0].im, 0.0);
    assert_relative_eq!(gradient[0][1].re, 0.0);
    assert_relative_eq!(gradient[0][1].im, 1.0);
    assert_relative_eq!(gradient[0][2].re, -1.0);
    assert_relative_eq!(gradient[0][2].im, 0.0);
    assert_relative_eq!(gradient[0][3].re, 0.0);
    assert_relative_eq!(gradient[0][3].im, -1.0);

    let expr = &expr1 * &expr2;
    let evaluator = expr.load(&dataset).unwrap();

    let gradient = evaluator
        .evaluate_gradient(&params)
        .expect("evaluation should succeed");

    assert_relative_eq!(gradient[0][0].re, 4.0);
    assert_relative_eq!(gradient[0][0].im, 5.0);
    assert_relative_eq!(gradient[0][1].re, -5.0);
    assert_relative_eq!(gradient[0][1].im, 4.0);
    assert_relative_eq!(gradient[0][2].re, 2.0);
    assert_relative_eq!(gradient[0][2].im, 3.0);
    assert_relative_eq!(gradient[0][3].re, -3.0);
    assert_relative_eq!(gradient[0][3].im, 2.0);

    let expr = &expr1 / &expr2;
    let evaluator = expr.load(&dataset).unwrap();

    let gradient = evaluator
        .evaluate_gradient(&params)
        .expect("evaluation should succeed");

    assert_relative_eq!(gradient[0][0].re, 4.0 / 41.0);
    assert_relative_eq!(gradient[0][0].im, -5.0 / 41.0);
    assert_relative_eq!(gradient[0][1].re, 5.0 / 41.0);
    assert_relative_eq!(gradient[0][1].im, 4.0 / 41.0);
    assert_relative_eq!(gradient[0][2].re, -102.0 / 1681.0);
    assert_relative_eq!(gradient[0][2].im, 107.0 / 1681.0);
    assert_relative_eq!(gradient[0][3].re, -107.0 / 1681.0);
    assert_relative_eq!(gradient[0][3].im, -102.0 / 1681.0);

    let expr = -(&expr1 * &expr2);
    let evaluator = expr.load(&dataset).unwrap();

    let gradient = evaluator
        .evaluate_gradient(&params)
        .expect("evaluation should succeed");

    assert_relative_eq!(gradient[0][0].re, -4.0);
    assert_relative_eq!(gradient[0][0].im, -5.0);
    assert_relative_eq!(gradient[0][1].re, 5.0);
    assert_relative_eq!(gradient[0][1].im, -4.0);
    assert_relative_eq!(gradient[0][2].re, -2.0);
    assert_relative_eq!(gradient[0][2].im, -3.0);
    assert_relative_eq!(gradient[0][3].re, 3.0);
    assert_relative_eq!(gradient[0][3].im, -2.0);

    let expr = (&expr1 * &expr2).real();
    let evaluator = expr.load(&dataset).unwrap();

    let gradient = evaluator
        .evaluate_gradient(&params)
        .expect("evaluation should succeed");

    assert_relative_eq!(gradient[0][0].re, 4.0);
    assert_relative_eq!(gradient[0][0].im, 0.0);
    assert_relative_eq!(gradient[0][1].re, -5.0);
    assert_relative_eq!(gradient[0][1].im, 0.0);
    assert_relative_eq!(gradient[0][2].re, 2.0);
    assert_relative_eq!(gradient[0][2].im, 0.0);
    assert_relative_eq!(gradient[0][3].re, -3.0);
    assert_relative_eq!(gradient[0][3].im, 0.0);

    let expr = (&expr1 * &expr2).imag();
    let evaluator = expr.load(&dataset).unwrap();

    let gradient = evaluator
        .evaluate_gradient(&params)
        .expect("evaluation should succeed");

    assert_relative_eq!(gradient[0][0].re, 5.0);
    assert_relative_eq!(gradient[0][0].im, 0.0);
    assert_relative_eq!(gradient[0][1].re, 4.0);
    assert_relative_eq!(gradient[0][1].im, 0.0);
    assert_relative_eq!(gradient[0][2].re, 3.0);
    assert_relative_eq!(gradient[0][2].im, 0.0);
    assert_relative_eq!(gradient[0][3].re, 2.0);
    assert_relative_eq!(gradient[0][3].im, 0.0);

    let expr = (&expr1 * &expr2).conj();
    let evaluator = expr.load(&dataset).unwrap();

    let gradient = evaluator
        .evaluate_gradient(&params)
        .expect("evaluation should succeed");

    assert_relative_eq!(gradient[0][0].re, 4.0);
    assert_relative_eq!(gradient[0][0].im, -5.0);
    assert_relative_eq!(gradient[0][1].re, -5.0);
    assert_relative_eq!(gradient[0][1].im, -4.0);
    assert_relative_eq!(gradient[0][2].re, 2.0);
    assert_relative_eq!(gradient[0][2].im, -3.0);
    assert_relative_eq!(gradient[0][3].re, -3.0);
    assert_relative_eq!(gradient[0][3].im, -2.0);

    let expr = (&expr1 * &expr2).norm_sqr();
    let evaluator = expr.load(&dataset).unwrap();

    let gradient = evaluator
        .evaluate_gradient(&params)
        .expect("evaluation should succeed");

    assert_relative_eq!(gradient[0][0].re, 164.0);
    assert_relative_eq!(gradient[0][0].im, 0.0);
    assert_relative_eq!(gradient[0][1].re, 246.0);
    assert_relative_eq!(gradient[0][1].im, 0.0);
    assert_relative_eq!(gradient[0][2].re, 104.0);
    assert_relative_eq!(gradient[0][2].im, 0.0);
    assert_relative_eq!(gradient[0][3].re, 130.0);
    assert_relative_eq!(gradient[0][3].im, 0.0);
}

#[test]
fn test_expression_function_gradients() {
    let expr1 = ComplexScalar::new(
        "function_parametric_1",
        parameter!("function_test_param_re_1"),
        parameter!("function_test_param_im_1"),
    )
    .unwrap();
    let expr2 = ComplexScalar::new(
        "function_parametric_2",
        parameter!("function_test_param_re_2"),
        parameter!("function_test_param_im_2"),
    )
    .unwrap();

    let sin = expr1.sin();
    let cos = expr1.cos();
    let trig = &sin * &cos;
    let pow = expr1.pow(&expr2);
    let mut expr = expr1.sqrt();
    expr = &expr + &expr1.exp();
    expr = &expr + &expr1.powi(2);
    expr = &expr + &expr1.powf(1.7);
    expr = &expr + &trig;
    expr = &expr + &expr1.log();
    expr = &expr + &expr1.cis();
    expr = &expr + &pow;

    let dataset = Arc::new(test_dataset());
    let evaluator = expr.load(&dataset).unwrap();
    let params = vec![2.0, 0.5, 1.2, -0.3];
    let gradient = evaluator
        .evaluate_gradient(&params)
        .expect("evaluation should succeed");
    let eps = 1e-6;

    for param_index in 0..params.len() {
        let mut plus = params.clone();
        plus[param_index] += eps;
        let mut minus = params.clone();
        minus[param_index] -= eps;
        let finite_diff = (evaluator
            .evaluate(&plus)
            .expect("evaluation should succeed")[0]
            - evaluator
                .evaluate(&minus)
                .expect("evaluation should succeed")[0])
            / Complex64::new(2.0 * eps, 0.0);

        assert_relative_eq!(
            gradient[0][param_index].re,
            finite_diff.re,
            epsilon = 1e-6,
            max_relative = 1e-6
        );
        assert_relative_eq!(
            gradient[0][param_index].im,
            finite_diff.im,
            epsilon = 1e-6,
            max_relative = 1e-6
        );
    }
}

#[test]
fn test_zeros_and_ones() {
    let amp = ComplexScalar::new(
        "parametric",
        parameter!("test_param_re"),
        parameter!("fixed_two", 2.0),
    )
    .unwrap();
    let dataset = Arc::new(test_dataset());
    let expr = (amp * Expression::one() + Expression::zero()).norm_sqr();
    let evaluator = expr.load(&dataset).unwrap();

    let params = vec![2.0];
    let value = evaluator
        .evaluate(&params)
        .expect("evaluation should succeed");
    let gradient = evaluator
        .evaluate_gradient(&params)
        .expect("evaluation should succeed");

    // For |f(x) * 1 + 0|^2 where f(x) = x+2i, the value should be x^2 + 4
    assert_relative_eq!(value[0].re, 8.0);
    assert_relative_eq!(value[0].im, 0.0);

    // For |f(x) * 1 + 0|^2 where f(x) = x+2i, the derivative should be 2x
    assert_relative_eq!(gradient[0][0].re, 4.0);
    assert_relative_eq!(gradient[0][0].im, 0.0);
}
#[test]
fn test_default_build_uses_lowered_expression_runtime() {
    let expr = ComplexScalar::new(
        "opt_in_gate",
        parameter!("opt_in_gate_re", 2.0),
        parameter!("opt_in_gate_im", 0.0),
    )
    .unwrap()
    .norm_sqr();
    let dataset = Arc::new(test_dataset());
    let evaluator = expr.load(&dataset).unwrap();

    let diagnostics = evaluator.expression_runtime_diagnostics();
    assert!(diagnostics.ir_planning_enabled);
    assert!(diagnostics.lowered_value_program_present);
    assert!(diagnostics.lowered_gradient_program_present);
    assert!(diagnostics.lowered_value_gradient_program_present);
    assert_eq!(
        evaluator.evaluate(&[]).expect("evaluation should succeed")[0],
        Complex64::new(4.0, 0.0)
    );
}

#[test]
fn parameter_name_only_creates_free_parameter() {
    let p = parameter!("mass");

    assert_eq!(p.name(), "mass");
    assert_eq!(p.fixed(), None);
    assert_eq!(p.initial(), None);
    assert_eq!(p.bounds(), (None, None));
    assert_eq!(p.unit(), None);
    assert_eq!(p.latex(), None);
    assert_eq!(p.description(), None);
    assert!(p.is_free());
    assert!(!p.is_fixed());
}

#[test]
fn parameter_name_and_value_creates_fixed_parameter() {
    let p = parameter!("width", 0.15);

    assert_eq!(p.name(), "width");
    assert_eq!(p.fixed(), Some(0.15));
    assert_eq!(p.initial(), Some(0.15));
    assert!(p.is_fixed());
    assert!(!p.is_free());
}

#[test]
fn keyword_initial_sets_initial_only() {
    let p = parameter!("alpha", initial: 1.25);

    assert_eq!(p.name(), "alpha");
    assert_eq!(p.fixed(), None);
    assert_eq!(p.initial(), Some(1.25));
    assert_eq!(p.bounds(), (None, None));
    assert!(p.is_free());
}

#[test]
fn keyword_fixed_sets_fixed_and_initial() {
    let p = parameter!("beta", fixed: 2.5);

    assert_eq!(p.name(), "beta");
    assert_eq!(p.fixed(), Some(2.5));
    assert_eq!(p.initial(), Some(2.5));
    assert!(p.is_fixed());
}

#[test]
fn bounds_accept_plain_numbers() {
    let p = parameter!("x", bounds: (0.0, 10.0));

    assert_eq!(p.bounds(), (Some(0.0), Some(10.0)));
}

#[test]
fn bounds_accept_none_and_number() {
    let p = parameter!("x", bounds: (None, 10.0));

    assert_eq!(p.bounds(), (None, Some(10.0)));
}

#[test]
fn bounds_accept_number_and_none() {
    let p = parameter!("x", bounds: (-1.0, None));

    assert_eq!(p.bounds(), (Some(-1.0), None));
}

#[test]
fn bounds_accept_both_none() {
    let p = parameter!("x", bounds: (None, None));

    assert_eq!(p.bounds(), (None, None));
}

#[test]
fn bounds_accept_arbitrary_expressions() {
    let lo = 1.0;
    let hi = 2.0 * 3.0;
    let p = parameter!("x", bounds: (lo - 0.5, hi));

    assert_eq!(p.bounds(), (Some(0.5), Some(6.0)));
}

#[test]
fn multiple_keyword_arguments_work_together() {
    let p = parameter!(
        "gamma",
        initial: 1.0,
        bounds: (0.0, 5.0),
        unit: "GeV",
        latex: r"\gamma",
        description: "test parameter",
    );

    assert_eq!(p.name(), "gamma");
    assert_eq!(p.fixed(), None);
    assert_eq!(p.initial(), Some(1.0));
    assert_eq!(p.bounds(), (Some(0.0), Some(5.0)));
    assert_eq!(p.unit().as_deref(), Some("GeV"));
    assert_eq!(p.latex().as_deref(), Some(r"\gamma"));
    assert_eq!(p.description().as_deref(), Some("test parameter"));
}

#[test]
fn fixed_can_be_combined_with_other_fields() {
    let p = parameter!(
        "delta",
        fixed: 3.0,
        bounds: (0.0, 10.0),
        unit: "rad",
    );

    assert_eq!(p.name(), "delta");
    assert_eq!(p.fixed(), Some(3.0));
    assert_eq!(p.initial(), Some(3.0));
    assert_eq!(p.bounds(), (Some(0.0), Some(10.0)));
    assert_eq!(p.unit().as_deref(), Some("rad"));
}

#[test]
fn trailing_comma_is_accepted() {
    let p = parameter!(
        "eps",
        initial: 0.5,
        bounds: (None, 1.0),
        unit: "arb",
    );

    assert_eq!(p.initial(), Some(0.5));
    assert_eq!(p.bounds(), (None, Some(1.0)));
    assert_eq!(p.unit().as_deref(), Some("arb"));
}

#[test]
fn test_parameter_registration() {
    let expr = ComplexScalar::new(
        "parametric",
        parameter!("test_param_re"),
        parameter!("fixed_two", 2.0),
    )
    .unwrap();
    let parameters = expr.parameters().free().names();
    assert_eq!(parameters.len(), 1);
    assert_eq!(parameters[0], "test_param_re");
}

#[test]
fn test_duplicate_amplitude_tag_registration_is_allowed() {
    let amp1 = ComplexScalar::new(
        "same_name",
        parameter!("dup_re1", 1.0),
        parameter!("dup_im1", 0.0),
    )
    .unwrap();
    let amp2 = ComplexScalar::new(
        "same_name",
        parameter!("dup_re2", 2.0),
        parameter!("dup_im2", 0.0),
    )
    .unwrap();
    let expr = amp1 + amp2;
    assert_eq!(
        expr.parameters().fixed().names(),
        vec!["dup_re1", "dup_im1", "dup_re2", "dup_im2"]
    );
}

#[test]
fn test_tree_printing() {
    let amp1 = ComplexScalar::new(
        "parametric_1",
        parameter!("test_param_re_1"),
        parameter!("test_param_im_1"),
    )
    .unwrap();
    let amp2 = ComplexScalar::new(
        "parametric_2",
        parameter!("test_param_re_2"),
        parameter!("test_param_im_2"),
    )
    .unwrap();
    let expr = &amp1.real() + &amp2.conj().imag() + Expression::one() * Complex64::new(-1.4, 2.0)
        - Expression::zero() / 1.0
        + (&amp1 * &amp2).norm_sqr();
    assert_eq!(
        expr.to_string(),
        "+
├─ -
│  ├─ +
│  │  ├─ +
│  │  │  ├─ Re
│  │  │  │  └─ parametric_1(id=0)
│  │  │  └─ Im
│  │  │     └─ *
│  │  │        └─ parametric_2(id=1)
│  │  └─ ×
│  │     ├─ 1 (exact)
│  │     └─ -1.4+2i
│  └─ ÷
│     ├─ 0 (exact)
│     └─ 1 (exact)
└─ NormSqr
   └─ ×
      ├─ parametric_1(id=0)
      └─ parametric_2(id=1)
"
    );
}
