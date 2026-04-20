use laddu_core::{
    amplitudes::{
        parameter, Amplitude, AmplitudeID, ExpressionDependence, Parameter, TestAmplitude,
    },
    data::{DatasetMetadata, EventData, NamedEventView},
    resources::{Cache, ParameterID, Parameters, Resources, ScalarID},
    utils::vectors::Vec4,
    Dataset, Expression, LadduResult,
};
use num::complex::Complex64;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
#[derive(Clone, Serialize, Deserialize)]
struct ParameterOnlyScalar {
    name: String,
    value: Parameter,
    pid: ParameterID,
}
impl ParameterOnlyScalar {
    #[allow(clippy::new_ret_no_self)]
    fn new(name: &str, value: Parameter) -> LadduResult<Expression> {
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
impl ScenarioKind {
    fn parse(raw: &str) -> Result<Self, String> {
        match raw {
            "separable" => Ok(Self::Separable),
            "partial" => Ok(Self::Partial),
            "non_separable" => Ok(Self::NonSeparable),
            _ => Err(format!("unknown scenario: {raw}")),
        }
    }
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
fn build_scenario_expression(kind: ScenarioKind) -> Expression {
    match kind {
        ScenarioKind::Separable => {
            let p1 = ParameterOnlyScalar::new("sep_p1", parameter("sep_p1")).unwrap();
            let p2 = ParameterOnlyScalar::new("sep_p2", parameter("sep_p2")).unwrap();
            let c1 = CacheOnlyScalar::new("sep_c1").unwrap();
            let c2 = CacheOnlyScalar::new("sep_c2").unwrap();
            (&p1 * &c1) + &(&p2 * &c2)
        }
        ScenarioKind::Partial => {
            let p = ParameterOnlyScalar::new("partial_p", parameter("partial_p")).unwrap();
            let c = CacheOnlyScalar::new("partial_c").unwrap();
            let m = TestAmplitude::new(
                "partial_m",
                parameter("partial_mr"),
                parameter("partial_mi"),
            )
            .unwrap();
            (&p * &c) + &m
        }
        ScenarioKind::NonSeparable => {
            let m1 = TestAmplitude::new(
                "nonsep_m1",
                parameter("nonsep_m1r"),
                parameter("nonsep_m1i"),
            )
            .unwrap();
            let m2 = TestAmplitude::new(
                "nonsep_m2",
                parameter("nonsep_m2r"),
                parameter("nonsep_m2i"),
            )
            .unwrap();
            &m1 * &m2
        }
    }
}
fn print_usage_and_exit() -> ! {
    eprintln!(
        "usage: expression_ir_compile_probe <initial_load|specialization_cache_miss|specialization_cache_hit_restore> <separable|partial|non_separable>"
    );
    std::process::exit(2);
}
fn compile_metrics_json(evaluator: &laddu_core::Evaluator) -> String {
    let metrics = evaluator.expression_compile_metrics();
    format!(
        concat!(
            "{{",
            "\"initial_ir_compile_nanos\":{},",
            "\"initial_cached_integrals_nanos\":{},",
            "\"initial_lowering_nanos\":{},",
            "\"specialization_cache_hits\":{},",
            "\"specialization_cache_misses\":{},",
            "\"specialization_ir_compile_nanos\":{},",
            "\"specialization_cached_integrals_nanos\":{},",
            "\"specialization_lowering_nanos\":{},",
            "\"specialization_lowering_cache_hits\":{},",
            "\"specialization_lowering_cache_misses\":{},",
            "\"specialization_cache_restore_nanos\":{}",
            "}}"
        ),
        metrics.initial_ir_compile_nanos,
        metrics.initial_cached_integrals_nanos,
        metrics.initial_lowering_nanos,
        metrics.specialization_cache_hits,
        metrics.specialization_cache_misses,
        metrics.specialization_ir_compile_nanos,
        metrics.specialization_cached_integrals_nanos,
        metrics.specialization_lowering_nanos,
        metrics.specialization_lowering_cache_hits,
        metrics.specialization_lowering_cache_misses,
        metrics.specialization_cache_restore_nanos,
    )
}
fn specialization_metrics_json(evaluator: &laddu_core::Evaluator) -> String {
    let metrics = evaluator.expression_specialization_metrics();
    format!(
        "{{\"cache_hits\":{},\"cache_misses\":{}}}",
        metrics.cache_hits, metrics.cache_misses
    )
}
fn main() {
    let mut args = std::env::args().skip(1);
    let operation = args.next().unwrap_or_else(|| print_usage_and_exit());
    let scenario_raw = args.next().unwrap_or_else(|| print_usage_and_exit());
    if args.next().is_some() {
        print_usage_and_exit();
    }

    let scenario = ScenarioKind::parse(&scenario_raw).unwrap_or_else(|_| print_usage_and_exit());
    let dataset = synthetic_weighted_dataset(4096);
    let expression = build_scenario_expression(scenario);
    let evaluator = expression
        .load(&dataset)
        .expect("probe evaluator should load");

    match operation.as_str() {
        "initial_load" => {}
        "specialization_cache_miss" => {
            evaluator.reset_expression_compile_metrics();
            evaluator.reset_expression_specialization_metrics();
            match scenario {
                ScenarioKind::Separable => evaluator.isolate_many(&["sep_p1"]),
                ScenarioKind::Partial => evaluator.isolate_many(&["partial_p"]),
                ScenarioKind::NonSeparable => evaluator.isolate_many(&["nonsep_m1"]),
            }
        }
        "specialization_cache_hit_restore" => {
            match scenario {
                ScenarioKind::Separable => evaluator.isolate_many(&["sep_p1"]),
                ScenarioKind::Partial => evaluator.isolate_many(&["partial_p"]),
                ScenarioKind::NonSeparable => evaluator.isolate_many(&["nonsep_m1"]),
            }
            evaluator.activate_all();
            evaluator.reset_expression_compile_metrics();
            evaluator.reset_expression_specialization_metrics();
            match scenario {
                ScenarioKind::Separable => evaluator.isolate_many(&["sep_p1"]),
                ScenarioKind::Partial => evaluator.isolate_many(&["partial_p"]),
                ScenarioKind::NonSeparable => evaluator.isolate_many(&["nonsep_m1"]),
            }
            evaluator.reset_expression_compile_metrics();
            evaluator.reset_expression_specialization_metrics();
            evaluator.activate_all();
        }
        _ => print_usage_and_exit(),
    }

    println!(
        "{{\"operation\":\"{}\",\"scenario\":\"{}\",\"compile_metrics\":{},\"specialization_metrics\":{}}}",
        operation,
        scenario_raw,
        compile_metrics_json(&evaluator),
        specialization_metrics_json(&evaluator)
    );
}
