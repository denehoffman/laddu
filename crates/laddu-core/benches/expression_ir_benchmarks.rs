use std::{sync::Arc, time::Duration};

use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput,
};
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
    match kind {
        ScenarioKind::Separable => {
            let p1 = ParameterOnlyScalar::new("sep_p1", parameter("sep_p1"))
                .expect("separable p1 should construct");
            let p2 = ParameterOnlyScalar::new("sep_p2", parameter("sep_p2"))
                .expect("separable p2 should construct");
            let c1 = CacheOnlyScalar::new("sep_c1").expect("separable c1 should construct");
            let c2 = CacheOnlyScalar::new("sep_c2").expect("separable c2 should construct");
            let expression = (&p1 * &c1) + &(&p2 * &c2);
            ScenarioCase {
                label: "separable",
                kind,
                evaluator: expression
                    .load(dataset)
                    .expect("separable evaluator should load"),
                parameters: vec![0.4, -0.3],
                n_events: dataset.n_events_local(),
            }
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
            let expression = (&p * &c) + &m;
            ScenarioCase {
                label: "partial",
                kind,
                evaluator: expression
                    .load(dataset)
                    .expect("partial evaluator should load"),
                parameters: vec![0.55, 0.2, -0.15],
                n_events: dataset.n_events_local(),
            }
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
            let expression = &m1 * &m2;
            ScenarioCase {
                label: "non_separable",
                kind,
                evaluator: expression
                    .load(dataset)
                    .expect("non-separable evaluator should load"),
                parameters: vec![0.25, -0.4, 0.6, 0.1],
                n_events: dataset.n_events_local(),
            }
        }
    }
}

#[cfg(feature = "expression-ir")]
fn activation_churn_isolate_names(kind: ScenarioKind) -> &'static [&'static str] {
    match kind {
        ScenarioKind::Separable => &["sep_p1"],
        ScenarioKind::Partial => &["partial_p"],
        ScenarioKind::NonSeparable => &["nonsep_m1"],
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

    let mut value_group = c.benchmark_group("expression_ir_factorization_weighted_value_sum");
    value_group.sample_size(50);
    value_group.warm_up_time(Duration::from_secs(2));
    value_group.measurement_time(Duration::from_secs(8));
    for (tier_label, dataset) in &datasets {
        let cases = [
            build_scenario(dataset, ScenarioKind::Separable),
            build_scenario(dataset, ScenarioKind::Partial),
            build_scenario(dataset, ScenarioKind::NonSeparable),
        ];
        for case in &cases {
            value_group.throughput(Throughput::Elements(case.n_events as u64));
            value_group.bench_with_input(
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
            value_group.bench_with_input(
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
    value_group.finish();

    let mut gradient_group = c.benchmark_group("expression_ir_factorization_weighted_gradient_sum");
    gradient_group.sample_size(50);
    gradient_group.warm_up_time(Duration::from_secs(2));
    gradient_group.measurement_time(Duration::from_secs(8));
    for (tier_label, dataset) in &datasets {
        let cases = [
            build_scenario(dataset, ScenarioKind::Separable),
            build_scenario(dataset, ScenarioKind::Partial),
            build_scenario(dataset, ScenarioKind::NonSeparable),
        ];
        for case in &cases {
            gradient_group.throughput(Throughput::Elements(case.n_events as u64));
            gradient_group.bench_with_input(
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
            gradient_group.bench_with_input(
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
    gradient_group.finish();

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
    }

    group.finish();
}

#[cfg(not(feature = "expression-ir"))]
fn expression_ir_activation_churn_benchmarks(_c: &mut Criterion) {}

criterion_group!(
    benches,
    expression_backend_benchmarks,
    expression_ir_normalization_factorization_benchmarks,
    expression_ir_activation_churn_benchmarks
);
criterion_main!(benches);
