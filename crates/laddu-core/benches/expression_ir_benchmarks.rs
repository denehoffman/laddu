use std::{sync::Arc, time::Duration};

use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput,
};
#[cfg(feature = "expression-ir")]
use laddu_core::amplitudes::ExpressionRuntimeBackend;
use laddu_core::{
    amplitudes::{
        parameter, Amplitude, AmplitudeID, ExpressionDependence, ParameterLike, TestAmplitude,
    },
    data::{read_parquet, DatasetMetadata, DatasetReadOptions, EventData, NamedEventView},
    resources::{Cache, ParameterID, Parameters, Resources, ScalarID},
    utils::vectors::Vec4,
    Dataset, Evaluator, Expression, LadduResult,
};
use nalgebra::DVector;
use num::complex::Complex64;
use serde::{Deserialize, Serialize};

const BENCH_DATASET_PATH: &str = "benches/bench.parquet";
const P4_NAMES: [&str; 4] = ["beam", "proton", "kshort1", "kshort2"];
const AUX_NAMES: [&str; 2] = ["pol_magnitude", "pol_angle"];
const PARAMS: [f64; 2] = [1.25, -0.75];
const SAMPLE_EVENTS: usize = 512;
const NORMALIZATION_BENCH_EVENTS_SMALL: usize = 4096;
const NORMALIZATION_BENCH_EVENTS_LARGE: usize = 32768;
const MEMORY_BENCH_EVENTS_LARGE_GRADIENT: usize = 4096;
const MEMORY_BENCH_GRADIENT_TERMS: usize = 16;

#[derive(Clone, Serialize, Deserialize)]
struct ParameterOnlyScalar {
    name: String,
    value: ParameterLike,
    pid: ParameterID,
}

impl ParameterOnlyScalar {
    #[allow(clippy::new_ret_no_self)]
    fn new(name: &str, value: ParameterLike) -> LadduResult<Expression> {
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

    fn compute(&self, parameters: &Parameters, _cache: &Cache) -> Complex64 {
        Complex64::new(parameters.get(self.pid), 0.0)
    }
}

#[derive(Clone, Serialize, Deserialize)]
struct CacheOnlyScalar {
    name: String,
    beam_energy: ScalarID,
}

impl CacheOnlyScalar {
    #[allow(clippy::new_ret_no_self)]
    fn new(name: &str) -> LadduResult<Expression> {
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

    fn precompute(&self, event: &NamedEventView<'_>, cache: &mut Cache) {
        cache.store_scalar(self.beam_energy, event.p4_at(0).e());
    }

    fn compute(&self, _parameters: &Parameters, cache: &Cache) -> Complex64 {
        Complex64::new(cache.get_scalar(self.beam_energy), 0.0)
    }
}

#[derive(Clone, Copy)]
enum ScenarioKind {
    Separable,
    Partial,
    NonSeparable,
}

struct ScenarioCase {
    label: &'static str,
    kind: ScenarioKind,
    evaluator: Evaluator,
    parameters: Vec<f64>,
    n_events: usize,
}

struct LargeGradientCase {
    label: &'static str,
    evaluator: Evaluator,
    parameters: Vec<f64>,
    n_events: usize,
    n_parameters: usize,
}

struct RealUnaryCase {
    label: &'static str,
    evaluator: Evaluator,
    parameters: Vec<f64>,
    n_events: usize,
}

fn synthetic_weighted_dataset(n_events: usize) -> Arc<Dataset> {
    let metadata = Arc::new(DatasetMetadata::default());
    let events = (0..n_events)
        .map(|index| {
            let beam_e = 1.0 + (index % 97) as f64 * 0.125;
            let weight_mag = 0.5 + (index % 11) as f64 * 0.08;
            let weight = if index % 2 == 0 {
                weight_mag
            } else {
                -0.75 * weight_mag
            };
            Arc::new(EventData {
                p4s: vec![Vec4::new(0.0, 0.0, 0.0, beam_e)],
                aux: vec![],
                weight,
            })
        })
        .collect::<Vec<_>>();
    Arc::new(Dataset::new_with_metadata(events, metadata))
}

fn build_scenario(dataset: &Arc<Dataset>, kind: ScenarioKind) -> ScenarioCase {
    let expression = build_scenario_expression(kind);
    let label = match kind {
        ScenarioKind::Separable => "separable",
        ScenarioKind::Partial => "partial",
        ScenarioKind::NonSeparable => "non_separable",
    };
    let parameters = match kind {
        ScenarioKind::Separable => vec![0.4, -0.3],
        ScenarioKind::Partial => vec![0.55, 0.2, -0.15],
        ScenarioKind::NonSeparable => vec![0.25, -0.4, 0.6, 0.1],
    };

    ScenarioCase {
        label,
        kind,
        evaluator: expression
            .load(dataset)
            .expect("scenario evaluator should load"),
        parameters,
        n_events: dataset.n_events_local(),
    }
}

fn build_scenario_expression(kind: ScenarioKind) -> Expression {
    match kind {
        ScenarioKind::Separable => {
            let p1 = ParameterOnlyScalar::new("sep_p1", parameter("sep_p1"))
                .expect("separable p1 should construct");
            let p2 = ParameterOnlyScalar::new("sep_p2", parameter("sep_p2"))
                .expect("separable p2 should construct");
            let c1 = CacheOnlyScalar::new("sep_c1").expect("separable c1 should construct");
            let c2 = CacheOnlyScalar::new("sep_c2").expect("separable c2 should construct");
            (&p1 * &c1) + &(&p2 * &c2)
        }
        ScenarioKind::Partial => {
            let p = ParameterOnlyScalar::new("partial_p", parameter("partial_p"))
                .expect("partial p should construct");
            let c = CacheOnlyScalar::new("partial_c").expect("partial c should construct");
            let m = TestAmplitude::new(
                "partial_m",
                parameter("partial_mr"),
                parameter("partial_mi"),
            )
            .expect("partial mixed amplitude should construct");
            (&p * &c) + &m
        }
        ScenarioKind::NonSeparable => {
            let m1 = TestAmplitude::new(
                "nonsep_m1",
                parameter("nonsep_m1r"),
                parameter("nonsep_m1i"),
            )
            .expect("non-separable m1 should construct");
            let m2 = TestAmplitude::new(
                "nonsep_m2",
                parameter("nonsep_m2r"),
                parameter("nonsep_m2i"),
            )
            .expect("non-separable m2 should construct");
            &m1 * &m2
        }
    }
}

fn build_large_gradient_case(dataset: &Arc<Dataset>, n_terms: usize) -> LargeGradientCase {
    let mut expression = TestAmplitude::new(
        "large_grad_amp_0",
        parameter("large_grad_re_0"),
        parameter("large_grad_im_0"),
    )
    .expect("large gradient term 0 should construct")
    .norm_sqr();
    for index in 1..n_terms {
        let amplitude = TestAmplitude::new(
            &format!("large_grad_amp_{index}"),
            parameter(&format!("large_grad_re_{index}")),
            parameter(&format!("large_grad_im_{index}")),
        )
        .expect("large gradient term should construct");
        expression = &expression + &amplitude.norm_sqr();
    }

    let evaluator = expression
        .load(dataset)
        .expect("large gradient evaluator should load");
    let n_parameters = n_terms * 2;
    let parameters = (0..n_parameters)
        .map(|index| 0.15 + index as f64 * 0.01)
        .collect::<Vec<_>>();

    LargeGradientCase {
        label: "large_gradient",
        evaluator,
        parameters,
        n_events: dataset.n_events_local(),
        n_parameters,
    }
}

fn build_real_unary_case(dataset: &Arc<Dataset>) -> RealUnaryCase {
    let expression = build_real_unary_expression();
    let evaluator = expression
        .load(dataset)
        .expect("real unary evaluator should load");

    RealUnaryCase {
        label: "real_unary_heavy",
        evaluator,
        parameters: vec![0.2, -0.15, 0.35, 0.1],
        n_events: dataset.n_events_local(),
    }
}

fn build_real_unary_expression() -> Expression {
    let amp0 = TestAmplitude::new(
        "real_unary_amp_0",
        parameter("real_unary_re_0"),
        parameter("real_unary_im_0"),
    )
    .expect("real unary term 0 should construct");
    let amp1 = TestAmplitude::new(
        "real_unary_amp_1",
        parameter("real_unary_re_1"),
        parameter("real_unary_im_1"),
    )
    .expect("real unary term 1 should construct");
    let cache =
        CacheOnlyScalar::new("real_unary_cache").expect("real unary cache factor should construct");

    let expression = &amp0.norm_sqr()
        + &amp0.real()
        + &amp0.imag()
        + &amp1.norm_sqr()
        + &(&cache * &amp1.real());
    expression
}

#[cfg(feature = "expression-ir")]
fn activation_churn_isolate_names(kind: ScenarioKind) -> &'static [&'static str] {
    match kind {
        ScenarioKind::Separable => &["sep_p1"],
        ScenarioKind::Partial => &["partial_p"],
        ScenarioKind::NonSeparable => &["nonsep_m1"],
    }
}

#[cfg(feature = "expression-ir")]
fn activation_churn_deactivate_names(kind: ScenarioKind) -> &'static [&'static str] {
    match kind {
        ScenarioKind::Separable => &["sep_c2"],
        ScenarioKind::Partial => &["partial_c"],
        ScenarioKind::NonSeparable => &["nonsep_m2"],
    }
}

fn eventwise_weighted_value_sum(evaluator: &Evaluator, parameters: &[f64]) -> f64 {
    evaluator
        .evaluate_local(parameters)
        .iter()
        .zip(evaluator.dataset.events_local().iter())
        .fold(0.0, |accum, (value, event)| {
            accum + event.weight() * value.re
        })
}

fn eventwise_weighted_gradient_sum(evaluator: &Evaluator, parameters: &[f64]) -> DVector<f64> {
    evaluator
        .evaluate_gradient_local(parameters)
        .iter()
        .zip(evaluator.dataset.events_local().iter())
        .fold(
            DVector::zeros(parameters.len()),
            |mut accum, (gradient, event)| {
                accum += gradient.map(|value| value.re).scale(event.weight());
                accum
            },
        )
}

fn load_bench_dataset() -> Arc<Dataset> {
    read_parquet(
        BENCH_DATASET_PATH,
        &DatasetReadOptions::default()
            .p4_names(P4_NAMES)
            .aux_names(AUX_NAMES),
    )
    .expect("benchmark dataset should open")
}

fn sampled_dataset(dataset: &Arc<Dataset>, n_take: usize) -> Arc<Dataset> {
    let n_take = n_take.min(dataset.events_local().len());
    let events = dataset
        .events_local()
        .iter()
        .take(n_take)
        .map(|event| event.data_arc())
        .collect();
    Arc::new(Dataset::new_with_metadata(events, dataset.metadata_arc()))
}

fn build_test_evaluator(dataset: &Arc<Dataset>) -> Evaluator {
    let expression = TestAmplitude::new("ir_probe", parameter("ir_re"), parameter("ir_im"))
        .expect("test amplitude should construct")
        .norm_sqr();
    expression
        .load(dataset)
        .expect("evaluator should load benchmark dataset")
}

#[cfg(feature = "expression-ir")]
fn backend_label(backend: ExpressionRuntimeBackend) -> &'static str {
    match backend {
        ExpressionRuntimeBackend::IrInterpreter => "ir_interpreter",
        ExpressionRuntimeBackend::Lowered => "lowered",
    }
}

#[cfg(feature = "expression-ir")]
fn evaluator_with_backend(evaluator: &Evaluator, backend: ExpressionRuntimeBackend) -> Evaluator {
    let mut evaluator = evaluator.clone();
    evaluator.set_expression_runtime_backend(backend);
    evaluator
}

fn expression_backend_benchmarks(c: &mut Criterion) {
    let dataset = load_bench_dataset();
    let sampled = sampled_dataset(&dataset, SAMPLE_EVENTS);
    let evaluator = build_test_evaluator(&sampled);
    let n_events = sampled.n_events() as u64;

    let mut group = c.benchmark_group("expression_backend_paths");
    group.sample_size(30);
    group.throughput(Throughput::Elements(n_events));

    group.bench_function("evaluate_value_small_sample", |b| {
        b.iter(|| black_box(evaluator.evaluate(black_box(&PARAMS))))
    });

    group.bench_function("evaluate_gradient_small_sample", |b| {
        b.iter(|| black_box(evaluator.evaluate_gradient(black_box(&PARAMS))))
    });

    group.bench_function("evaluate_value_and_gradient_small_sample", |b| {
        b.iter(|| {
            let values = evaluator.evaluate(black_box(&PARAMS));
            let gradients = evaluator.evaluate_gradient(black_box(&PARAMS));
            black_box((values, gradients))
        })
    });
    group.finish();

    #[cfg(feature = "expression-ir")]
    {
        let backends = [
            ExpressionRuntimeBackend::IrInterpreter,
            ExpressionRuntimeBackend::Lowered,
        ];
        let mut backend_group = c.benchmark_group("expression_ir_backend_microcomparisons");
        backend_group.sample_size(30);
        backend_group.warm_up_time(Duration::from_secs(2));
        backend_group.measurement_time(Duration::from_secs(6));
        backend_group.throughput(Throughput::Elements(n_events));

        for backend in backends {
            let backend_evaluator = evaluator_with_backend(&evaluator, backend);
            let backend_name = backend_label(backend);
            backend_group.bench_with_input(
                BenchmarkId::new("evaluate_local", backend_name),
                &backend_evaluator,
                |b, evaluator| b.iter(|| black_box(evaluator.evaluate(black_box(&PARAMS)))),
            );
            backend_group.bench_with_input(
                BenchmarkId::new("evaluate_gradient_local", backend_name),
                &backend_evaluator,
                |b, evaluator| {
                    b.iter(|| black_box(evaluator.evaluate_gradient(black_box(&PARAMS))))
                },
            );
            backend_group.bench_with_input(
                BenchmarkId::new("evaluate_value_gradient_local", backend_name),
                &backend_evaluator,
                |b, evaluator| {
                    b.iter(|| {
                        let values = evaluator.evaluate(black_box(&PARAMS));
                        let gradients = evaluator.evaluate_gradient(black_box(&PARAMS));
                        black_box((values, gradients))
                    })
                },
            );
        }
        backend_group.finish();
    }

    let real_unary_dataset = synthetic_weighted_dataset(NORMALIZATION_BENCH_EVENTS_SMALL);
    let real_unary_case = build_real_unary_case(&real_unary_dataset);
    let mut real_unary_group = c.benchmark_group("expression_ir_instruction_mix_generic");
    real_unary_group.sample_size(40);
    real_unary_group.warm_up_time(Duration::from_secs(2));
    real_unary_group.measurement_time(Duration::from_secs(6));
    real_unary_group.throughput(Throughput::Elements(real_unary_case.n_events as u64));

    real_unary_group.bench_with_input(
        BenchmarkId::new(
            "evaluate_local_real_unary",
            format!("{}@{}", real_unary_case.label, real_unary_case.n_events),
        ),
        &real_unary_case,
        |b, case| b.iter(|| black_box(case.evaluator.evaluate_local(black_box(&case.parameters)))),
    );

    real_unary_group.bench_with_input(
        BenchmarkId::new(
            "evaluate_gradient_local_real_unary",
            format!("{}@{}", real_unary_case.label, real_unary_case.n_events),
        ),
        &real_unary_case,
        |b, case| {
            b.iter(|| {
                black_box(
                    case.evaluator
                        .evaluate_gradient_local(black_box(&case.parameters)),
                )
            })
        },
    );

    real_unary_group.finish();
}

fn expression_ir_normalization_factorization_benchmarks(c: &mut Criterion) {
    let datasets = [
        (
            "n4k",
            synthetic_weighted_dataset(NORMALIZATION_BENCH_EVENTS_SMALL),
        ),
        (
            "n32k",
            synthetic_weighted_dataset(NORMALIZATION_BENCH_EVENTS_LARGE),
        ),
    ];

    let mut overall_value_group =
        c.benchmark_group("expression_ir_factorization_weighted_value_sum");
    overall_value_group.sample_size(50);
    overall_value_group.warm_up_time(Duration::from_secs(2));
    overall_value_group.measurement_time(Duration::from_secs(8));
    for (tier_label, dataset) in &datasets {
        let cases = [
            build_scenario(dataset, ScenarioKind::Separable),
            build_scenario(dataset, ScenarioKind::Partial),
            build_scenario(dataset, ScenarioKind::NonSeparable),
        ];
        for case in &cases {
            overall_value_group.throughput(Throughput::Elements(case.n_events as u64));
            overall_value_group.bench_with_input(
                BenchmarkId::new(
                    format!("optimized/{}", tier_label),
                    format!("{}@{}", case.label, case.n_events),
                ),
                case,
                |b, case| {
                    b.iter(|| {
                        black_box(
                            case.evaluator
                                .evaluate_weighted_value_sum_local(black_box(&case.parameters)),
                        )
                    })
                },
            );
            overall_value_group.bench_with_input(
                BenchmarkId::new(
                    format!("eventwise_baseline/{}", tier_label),
                    format!("{}@{}", case.label, case.n_events),
                ),
                case,
                |b, case| {
                    b.iter(|| {
                        black_box(eventwise_weighted_value_sum(
                            &case.evaluator,
                            black_box(&case.parameters),
                        ))
                    })
                },
            );
        }
    }
    overall_value_group.finish();
    let mut separable_value_group =
        c.benchmark_group("expression_ir_factorization_separable_weighted_value_sum");
    separable_value_group.sample_size(50);
    separable_value_group.warm_up_time(Duration::from_secs(2));
    separable_value_group.measurement_time(Duration::from_secs(8));
    for (tier_label, dataset) in &datasets {
        let case = build_scenario(dataset, ScenarioKind::Separable);
        separable_value_group.throughput(Throughput::Elements(case.n_events as u64));
        separable_value_group.bench_with_input(
            BenchmarkId::new(
                format!("optimized/{}", tier_label),
                format!("{}@{}", case.label, case.n_events),
            ),
            &case,
            |b, case| {
                b.iter(|| {
                    black_box(
                        case.evaluator
                            .evaluate_weighted_value_sum_local(black_box(&case.parameters)),
                    )
                })
            },
        );
        separable_value_group.bench_with_input(
            BenchmarkId::new(
                format!("eventwise_baseline/{}", tier_label),
                format!("{}@{}", case.label, case.n_events),
            ),
            &case,
            |b, case| {
                b.iter(|| {
                    black_box(eventwise_weighted_value_sum(
                        &case.evaluator,
                        black_box(&case.parameters),
                    ))
                })
            },
        );
    }
    separable_value_group.finish();
    let mut partial_value_group =
        c.benchmark_group("expression_ir_factorization_partial_weighted_value_sum");
    partial_value_group.sample_size(50);
    partial_value_group.warm_up_time(Duration::from_secs(2));
    partial_value_group.measurement_time(Duration::from_secs(8));
    for (tier_label, dataset) in &datasets {
        let case = build_scenario(dataset, ScenarioKind::Partial);
        partial_value_group.throughput(Throughput::Elements(case.n_events as u64));
        partial_value_group.bench_with_input(
            BenchmarkId::new(
                format!("optimized/{}", tier_label),
                format!("{}@{}", case.label, case.n_events),
            ),
            &case,
            |b, case| {
                b.iter(|| {
                    black_box(
                        case.evaluator
                            .evaluate_weighted_value_sum_local(black_box(&case.parameters)),
                    )
                })
            },
        );
        partial_value_group.bench_with_input(
            BenchmarkId::new(
                format!("eventwise_baseline/{}", tier_label),
                format!("{}@{}", case.label, case.n_events),
            ),
            &case,
            |b, case| {
                b.iter(|| {
                    black_box(eventwise_weighted_value_sum(
                        &case.evaluator,
                        black_box(&case.parameters),
                    ))
                })
            },
        );
    }
    partial_value_group.finish();

    let mut overall_gradient_group =
        c.benchmark_group("expression_ir_factorization_weighted_gradient_sum");
    overall_gradient_group.sample_size(50);
    overall_gradient_group.warm_up_time(Duration::from_secs(2));
    overall_gradient_group.measurement_time(Duration::from_secs(8));
    for (tier_label, dataset) in &datasets {
        let cases = [
            build_scenario(dataset, ScenarioKind::Separable),
            build_scenario(dataset, ScenarioKind::Partial),
            build_scenario(dataset, ScenarioKind::NonSeparable),
        ];
        for case in &cases {
            overall_gradient_group.throughput(Throughput::Elements(case.n_events as u64));
            overall_gradient_group.bench_with_input(
                BenchmarkId::new(
                    format!("optimized/{}", tier_label),
                    format!("{}@{}", case.label, case.n_events),
                ),
                case,
                |b, case| {
                    b.iter(|| {
                        black_box(
                            case.evaluator
                                .evaluate_weighted_gradient_sum_local(black_box(&case.parameters)),
                        )
                    })
                },
            );
            overall_gradient_group.bench_with_input(
                BenchmarkId::new(
                    format!("eventwise_baseline/{}", tier_label),
                    format!("{}@{}", case.label, case.n_events),
                ),
                case,
                |b, case| {
                    b.iter(|| {
                        black_box(eventwise_weighted_gradient_sum(
                            &case.evaluator,
                            black_box(&case.parameters),
                        ))
                    })
                },
            );
        }
    }
    overall_gradient_group.finish();
    let mut separable_gradient_group =
        c.benchmark_group("expression_ir_factorization_separable_weighted_gradient_sum");
    separable_gradient_group.sample_size(50);
    separable_gradient_group.warm_up_time(Duration::from_secs(2));
    separable_gradient_group.measurement_time(Duration::from_secs(8));
    for (tier_label, dataset) in &datasets {
        let case = build_scenario(dataset, ScenarioKind::Separable);
        separable_gradient_group.throughput(Throughput::Elements(case.n_events as u64));
        separable_gradient_group.bench_with_input(
            BenchmarkId::new(
                format!("optimized/{}", tier_label),
                format!("{}@{}", case.label, case.n_events),
            ),
            &case,
            |b, case| {
                b.iter(|| {
                    black_box(
                        case.evaluator
                            .evaluate_weighted_gradient_sum_local(black_box(&case.parameters)),
                    )
                })
            },
        );
        separable_gradient_group.bench_with_input(
            BenchmarkId::new(
                format!("eventwise_baseline/{}", tier_label),
                format!("{}@{}", case.label, case.n_events),
            ),
            &case,
            |b, case| {
                b.iter(|| {
                    black_box(eventwise_weighted_gradient_sum(
                        &case.evaluator,
                        black_box(&case.parameters),
                    ))
                })
            },
        );
    }
    separable_gradient_group.finish();
    let mut partial_gradient_group =
        c.benchmark_group("expression_ir_factorization_partial_weighted_gradient_sum");
    partial_gradient_group.sample_size(50);
    partial_gradient_group.warm_up_time(Duration::from_secs(2));
    partial_gradient_group.measurement_time(Duration::from_secs(8));
    for (tier_label, dataset) in &datasets {
        let case = build_scenario(dataset, ScenarioKind::Partial);
        partial_gradient_group.throughput(Throughput::Elements(case.n_events as u64));
        partial_gradient_group.bench_with_input(
            BenchmarkId::new(
                format!("optimized/{}", tier_label),
                format!("{}@{}", case.label, case.n_events),
            ),
            &case,
            |b, case| {
                b.iter(|| {
                    black_box(
                        case.evaluator
                            .evaluate_weighted_gradient_sum_local(black_box(&case.parameters)),
                    )
                })
            },
        );
        partial_gradient_group.bench_with_input(
            BenchmarkId::new(
                format!("eventwise_baseline/{}", tier_label),
                format!("{}@{}", case.label, case.n_events),
            ),
            &case,
            |b, case| {
                b.iter(|| {
                    black_box(eventwise_weighted_gradient_sum(
                        &case.evaluator,
                        black_box(&case.parameters),
                    ))
                })
            },
        );
    }
    partial_gradient_group.finish();

    let mut control_group = c.benchmark_group("expression_ir_factorization_controls");
    control_group.sample_size(40);
    control_group.warm_up_time(Duration::from_secs(2));
    control_group.measurement_time(Duration::from_secs(6));
    for (tier_label, dataset) in &datasets {
        let cases = [
            build_scenario(dataset, ScenarioKind::Separable),
            build_scenario(dataset, ScenarioKind::Partial),
            build_scenario(dataset, ScenarioKind::NonSeparable),
        ];
        for case in &cases {
            control_group.throughput(Throughput::Elements(case.n_events as u64));
            control_group.bench_with_input(
                BenchmarkId::new(
                    format!("evaluate_local/{}", tier_label),
                    format!("{}@{}", case.label, case.n_events),
                ),
                case,
                |b, case| {
                    b.iter(|| black_box(case.evaluator.evaluate_local(black_box(&case.parameters))))
                },
            );
            control_group.bench_with_input(
                BenchmarkId::new(
                    format!("evaluate_gradient_local/{}", tier_label),
                    format!("{}@{}", case.label, case.n_events),
                ),
                case,
                |b, case| {
                    b.iter(|| {
                        black_box(
                            case.evaluator
                                .evaluate_gradient_local(black_box(&case.parameters)),
                        )
                    })
                },
            );
        }
    }
    control_group.finish();

    let mut non_separable_control_group =
        c.benchmark_group("expression_ir_factorization_non_separable_controls");
    non_separable_control_group.sample_size(40);
    non_separable_control_group.warm_up_time(Duration::from_secs(2));
    non_separable_control_group.measurement_time(Duration::from_secs(6));
    for (tier_label, dataset) in &datasets {
        let case = build_scenario(dataset, ScenarioKind::NonSeparable);
        non_separable_control_group.throughput(Throughput::Elements(case.n_events as u64));
        non_separable_control_group.bench_with_input(
            BenchmarkId::new(
                format!("weighted_value_sum/{}", tier_label),
                format!("{}@{}", case.label, case.n_events),
            ),
            &case,
            |b, case| {
                b.iter(|| {
                    black_box(
                        case.evaluator
                            .evaluate_weighted_value_sum_local(black_box(&case.parameters)),
                    )
                })
            },
        );
        non_separable_control_group.bench_with_input(
            BenchmarkId::new(
                format!("weighted_gradient_sum/{}", tier_label),
                format!("{}@{}", case.label, case.n_events),
            ),
            &case,
            |b, case| {
                b.iter(|| {
                    black_box(
                        case.evaluator
                            .evaluate_weighted_gradient_sum_local(black_box(&case.parameters)),
                    )
                })
            },
        );
        non_separable_control_group.bench_with_input(
            BenchmarkId::new(
                format!("eventwise_baseline/{}", tier_label),
                format!("{}@{}", case.label, case.n_events),
            ),
            &case,
            |b, case| {
                b.iter(|| {
                    black_box((
                        eventwise_weighted_value_sum(&case.evaluator, black_box(&case.parameters)),
                        eventwise_weighted_gradient_sum(
                            &case.evaluator,
                            black_box(&case.parameters),
                        ),
                    ))
                })
            },
        );
    }
    non_separable_control_group.finish();

    let mut mixed_group = c.benchmark_group("expression_ir_factorization_mixed_components");
    mixed_group.sample_size(40);
    mixed_group.warm_up_time(Duration::from_secs(2));
    mixed_group.measurement_time(Duration::from_secs(6));
    for (tier_label, dataset) in &datasets {
        let case = build_scenario(dataset, ScenarioKind::Partial);
        mixed_group.throughput(Throughput::Elements(case.n_events as u64));
        mixed_group.bench_with_input(
            BenchmarkId::new(
                format!("weighted_value_sum_partial/{}", tier_label),
                format!("{}@{}", case.label, case.n_events),
            ),
            &case,
            |b, case| {
                b.iter(|| {
                    black_box(
                        case.evaluator
                            .evaluate_weighted_value_sum_local(black_box(&case.parameters)),
                    )
                })
            },
        );
        mixed_group.bench_with_input(
            BenchmarkId::new(
                format!("weighted_gradient_sum_partial/{}", tier_label),
                format!("{}@{}", case.label, case.n_events),
            ),
            &case,
            |b, case| {
                b.iter(|| {
                    black_box(
                        case.evaluator
                            .evaluate_weighted_gradient_sum_local(black_box(&case.parameters)),
                    )
                })
            },
        );
        mixed_group.bench_with_input(
            BenchmarkId::new(
                format!("eventwise_partial/{}", tier_label),
                format!("{}@{}", case.label, case.n_events),
            ),
            &case,
            |b, case| {
                b.iter(|| {
                    black_box((
                        eventwise_weighted_value_sum(&case.evaluator, black_box(&case.parameters)),
                        eventwise_weighted_gradient_sum(
                            &case.evaluator,
                            black_box(&case.parameters),
                        ),
                    ))
                })
            },
        );
    }
    mixed_group.finish();

    #[cfg(feature = "expression-ir")]
    {
        let backend_case = build_scenario(
            &synthetic_weighted_dataset(NORMALIZATION_BENCH_EVENTS_SMALL),
            ScenarioKind::Partial,
        );
        let backends = [
            ExpressionRuntimeBackend::IrInterpreter,
            ExpressionRuntimeBackend::Lowered,
        ];
        let mut backend_group =
            c.benchmark_group("expression_ir_backend_normalization_microcomparisons");
        backend_group.sample_size(30);
        backend_group.warm_up_time(Duration::from_secs(2));
        backend_group.measurement_time(Duration::from_secs(6));
        backend_group.throughput(Throughput::Elements(backend_case.n_events as u64));

        for backend in backends {
            let backend_evaluator = evaluator_with_backend(&backend_case.evaluator, backend);
            let backend_name = backend_label(backend);
            backend_group.bench_with_input(
                BenchmarkId::new("weighted_value_sum_partial", backend_name),
                &backend_evaluator,
                |b, evaluator| {
                    b.iter(|| {
                        black_box(
                            evaluator.evaluate_weighted_value_sum_local(black_box(
                                &backend_case.parameters,
                            )),
                        )
                    })
                },
            );
            backend_group.bench_with_input(
                BenchmarkId::new("weighted_gradient_sum_partial", backend_name),
                &backend_evaluator,
                |b, evaluator| {
                    b.iter(|| {
                        black_box(evaluator.evaluate_weighted_gradient_sum_local(black_box(
                            &backend_case.parameters,
                        )))
                    })
                },
            );
        }
        backend_group.finish();
    }

    let real_unary_datasets = [
        (
            "n4k",
            synthetic_weighted_dataset(NORMALIZATION_BENCH_EVENTS_SMALL),
        ),
        (
            "n32k",
            synthetic_weighted_dataset(NORMALIZATION_BENCH_EVENTS_LARGE),
        ),
    ];
    let mut real_unary_group =
        c.benchmark_group("expression_ir_instruction_mix_real_unary_normalization");
    real_unary_group.sample_size(40);
    real_unary_group.warm_up_time(Duration::from_secs(2));
    real_unary_group.measurement_time(Duration::from_secs(6));
    for (tier_label, dataset) in &real_unary_datasets {
        let case = build_real_unary_case(dataset);
        real_unary_group.throughput(Throughput::Elements(case.n_events as u64));
        real_unary_group.bench_with_input(
            BenchmarkId::new(
                format!("weighted_value_sum/{}", tier_label),
                format!("{}@{}", case.label, case.n_events),
            ),
            &case,
            |b, case| {
                b.iter(|| {
                    black_box(
                        case.evaluator
                            .evaluate_weighted_value_sum_local(black_box(&case.parameters)),
                    )
                })
            },
        );
        real_unary_group.bench_with_input(
            BenchmarkId::new(
                format!("weighted_gradient_sum/{}", tier_label),
                format!("{}@{}", case.label, case.n_events),
            ),
            &case,
            |b, case| {
                b.iter(|| {
                    black_box(
                        case.evaluator
                            .evaluate_weighted_gradient_sum_local(black_box(&case.parameters)),
                    )
                })
            },
        );
        real_unary_group.bench_with_input(
            BenchmarkId::new(
                format!("eventwise_baseline/{}", tier_label),
                format!("{}@{}", case.label, case.n_events),
            ),
            &case,
            |b, case| {
                b.iter(|| {
                    black_box((
                        eventwise_weighted_value_sum(&case.evaluator, black_box(&case.parameters)),
                        eventwise_weighted_gradient_sum(
                            &case.evaluator,
                            black_box(&case.parameters),
                        ),
                    ))
                })
            },
        );
    }
    real_unary_group.finish();
}

#[cfg(feature = "expression-ir")]
fn expression_ir_activation_churn_benchmarks(c: &mut Criterion) {
    let dataset = synthetic_weighted_dataset(NORMALIZATION_BENCH_EVENTS_SMALL);
    let cases = [
        build_scenario(&dataset, ScenarioKind::Separable),
        build_scenario(&dataset, ScenarioKind::Partial),
        build_scenario(&dataset, ScenarioKind::NonSeparable),
    ];

    let mut group = c.benchmark_group("expression_ir_activation_churn");
    group.sample_size(30);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(6));

    for case in &cases {
        let isolate_names = activation_churn_isolate_names(case.kind);
        let deactivate_names = activation_churn_deactivate_names(case.kind);
        group.throughput(Throughput::Elements(case.n_events as u64));
        group.bench_with_input(
            BenchmarkId::new(
                "cache_miss/isolate_param_only",
                format!("{}@{}", case.label, case.n_events),
            ),
            case,
            |b, case| {
                b.iter_batched(
                    || {
                        let evaluator = build_scenario(&dataset, case.kind).evaluator;
                        evaluator.reset_expression_specialization_metrics();
                        evaluator
                    },
                    |evaluator| {
                        evaluator.isolate_many(isolate_names);
                        black_box(evaluator.expression_specialization_metrics())
                    },
                    BatchSize::SmallInput,
                )
            },
        );
        group.bench_with_input(
            BenchmarkId::new(
                "cache_hit/restore_full",
                format!("{}@{}", case.label, case.n_events),
            ),
            case,
            |b, case| {
                b.iter_batched(
                    || {
                        let evaluator = build_scenario(&dataset, case.kind).evaluator;
                        evaluator.isolate_many(isolate_names);
                        evaluator.activate_all();
                        evaluator.reset_expression_specialization_metrics();
                        evaluator.isolate_many(isolate_names);
                        evaluator.reset_expression_specialization_metrics();
                        evaluator
                    },
                    |evaluator| {
                        evaluator.activate_all();
                        black_box(evaluator.expression_specialization_metrics())
                    },
                    BatchSize::SmallInput,
                )
            },
        );
        group.bench_with_input(
            BenchmarkId::new(
                "workflow/repeated_activation_cycle",
                format!("{}@{}", case.label, case.n_events),
            ),
            case,
            |b, case| {
                b.iter_batched(
                    || {
                        let evaluator = build_scenario(&dataset, case.kind).evaluator;
                        evaluator.reset_expression_specialization_metrics();
                        evaluator
                    },
                    |evaluator| {
                        evaluator.isolate_many(isolate_names);
                        evaluator.activate_all();
                        evaluator.deactivate_many(deactivate_names);
                        evaluator.activate_many(deactivate_names);
                        evaluator.isolate_many(isolate_names);
                        evaluator.activate_all();
                        black_box(evaluator.expression_specialization_metrics())
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

#[cfg(not(feature = "expression-ir"))]
fn expression_ir_activation_churn_benchmarks(_c: &mut Criterion) {}

#[cfg(feature = "expression-ir")]
fn expression_ir_compile_cost_benchmarks(c: &mut Criterion) {
    let dataset = synthetic_weighted_dataset(NORMALIZATION_BENCH_EVENTS_SMALL);
    let load_cases = [
        (ScenarioKind::Separable, "separable"),
        (ScenarioKind::Partial, "partial"),
        (ScenarioKind::NonSeparable, "non_separable"),
    ];

    let mut group = c.benchmark_group("expression_ir_compile_costs");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(4));

    for (kind, label) in load_cases {
        group.bench_with_input(BenchmarkId::new("initial_load", label), &kind, |b, kind| {
            b.iter_batched(
                || build_scenario_expression(*kind),
                |expression| {
                    let evaluator = expression.load(&dataset).expect("load bench should load");
                    black_box(evaluator.expression_compile_metrics())
                },
                BatchSize::SmallInput,
            )
        });
    }

    let activation_case = build_scenario(&dataset, ScenarioKind::Partial);
    let isolate_names = activation_churn_isolate_names(activation_case.kind);

    group.bench_with_input(
        BenchmarkId::new("specialization_cache_miss", activation_case.label),
        &activation_case,
        |b, case| {
            b.iter_batched(
                || {
                    let evaluator = build_scenario(&dataset, case.kind).evaluator;
                    evaluator.reset_expression_compile_metrics();
                    evaluator.reset_expression_specialization_metrics();
                    evaluator
                },
                |evaluator| {
                    evaluator.isolate_many(isolate_names);
                    black_box((
                        evaluator.expression_compile_metrics(),
                        evaluator.expression_specialization_metrics(),
                    ))
                },
                BatchSize::SmallInput,
            )
        },
    );

    group.bench_with_input(
        BenchmarkId::new("specialization_cache_hit_restore", activation_case.label),
        &activation_case,
        |b, case| {
            b.iter_batched(
                || {
                    let evaluator = build_scenario(&dataset, case.kind).evaluator;
                    evaluator.isolate_many(isolate_names);
                    evaluator.activate_all();
                    evaluator.reset_expression_compile_metrics();
                    evaluator.reset_expression_specialization_metrics();
                    evaluator.isolate_many(isolate_names);
                    evaluator.reset_expression_compile_metrics();
                    evaluator.reset_expression_specialization_metrics();
                    evaluator
                },
                |evaluator| {
                    evaluator.activate_all();
                    black_box((
                        evaluator.expression_compile_metrics(),
                        evaluator.expression_specialization_metrics(),
                    ))
                },
                BatchSize::SmallInput,
            )
        },
    );

    group.finish();
}

#[cfg(not(feature = "expression-ir"))]
fn expression_ir_compile_cost_benchmarks(_c: &mut Criterion) {}

#[cfg(feature = "expression-ir")]
fn expression_ir_memory_workload_benchmarks(c: &mut Criterion) {
    let large_gradient_dataset = synthetic_weighted_dataset(MEMORY_BENCH_EVENTS_LARGE_GRADIENT);
    let large_gradient_case =
        build_large_gradient_case(&large_gradient_dataset, MEMORY_BENCH_GRADIENT_TERMS);

    let mut group = c.benchmark_group("expression_ir_memory_workloads");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(4));
    group.throughput(Throughput::Elements(large_gradient_case.n_events as u64));

    group.bench_with_input(
        BenchmarkId::new(
            "gradient_local_large",
            format!(
                "{}@{}x{}",
                large_gradient_case.label,
                large_gradient_case.n_events,
                large_gradient_case.n_parameters
            ),
        ),
        &large_gradient_case,
        |b, case| {
            b.iter(|| {
                black_box(
                    case.evaluator
                        .evaluate_gradient_local(black_box(&case.parameters)),
                )
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::new(
            "value_gradient_local_large",
            format!(
                "{}@{}x{}",
                large_gradient_case.label,
                large_gradient_case.n_events,
                large_gradient_case.n_parameters
            ),
        ),
        &large_gradient_case,
        |b, case| {
            b.iter(|| {
                black_box(
                    case.evaluator
                        .evaluate_with_gradient_local(black_box(&case.parameters)),
                )
            })
        },
    );

    let activation_dataset = synthetic_weighted_dataset(NORMALIZATION_BENCH_EVENTS_LARGE);
    let activation_case = build_scenario(&activation_dataset, ScenarioKind::Partial);
    let isolate_names = activation_churn_isolate_names(activation_case.kind);
    let deactivate_names = activation_churn_deactivate_names(activation_case.kind);
    group.bench_with_input(
        BenchmarkId::new(
            "activation_cycle_large",
            format!("{}@{}", activation_case.label, activation_case.n_events),
        ),
        &activation_case,
        |b, case| {
            b.iter_batched(
                || {
                    let evaluator = build_scenario(&activation_dataset, case.kind).evaluator;
                    evaluator.reset_expression_specialization_metrics();
                    evaluator
                },
                |evaluator| {
                    evaluator.isolate_many(isolate_names);
                    evaluator.activate_all();
                    evaluator.deactivate_many(deactivate_names);
                    evaluator.activate_many(deactivate_names);
                    black_box(evaluator.expression_specialization_metrics())
                },
                BatchSize::SmallInput,
            )
        },
    );

    group.finish();
}

#[cfg(not(feature = "expression-ir"))]
fn expression_ir_memory_workload_benchmarks(_c: &mut Criterion) {}

criterion_group!(
    benches,
    expression_backend_benchmarks,
    expression_ir_normalization_factorization_benchmarks,
    expression_ir_activation_churn_benchmarks,
    expression_ir_compile_cost_benchmarks,
    expression_ir_memory_workload_benchmarks
);
criterion_main!(benches);
