//! Generated, experiment-neutral CPU workflows. Driven by scripts/cpu_benchmark.py.

use std::{
    collections::{BTreeMap, HashMap},
    error::Error,
    f64::consts::PI,
    fs,
    io::{self, Write},
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};

use laddu::prelude::ganesh::{
    algorithms::gradient::{LBFGSB, LBFGSBConfig},
    core::MaxSteps,
    traits::{Algorithm, SupportsParameterNames},
};
use laddu::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Config {
    size: String,
    workflow: String,
    data_events: usize,
    mc_events: usize,
    waves: usize,
    replicas: usize,
    threads: usize,
    repeat: usize,
    seed: u64,
    evaluations: usize,
    fit_steps: usize,
    periods: usize,
    shards_per_period: usize,
    bins: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    projection_components: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    projection_count: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    profile_batch: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    verify_blocks: Option<bool>,
}

#[derive(Default)]
struct Reads {
    traversals: AtomicU64,
    rows: AtomicU64,
}

struct CountingSource {
    source: Box<dyn EventSource>,
    reads: Arc<Reads>,
}

impl EventSource for CountingSource {
    fn schema(&self) -> LadduDataResult<Arc<Schema>> {
        self.source.schema()
    }
    fn capabilities(&self) -> SourceCapabilities {
        self.source.capabilities()
    }
    fn num_events(&self) -> LadduDataResult<Option<u64>> {
        self.source.num_events()
    }
    fn weighted_total(&self) -> LadduDataResult<Option<f64>> {
        self.source.weighted_total()
    }
    fn batches(&self, plan: ReadPlan) -> LadduDataResult<laddu::data::io::EventBatchIter> {
        self.reads.traversals.fetch_add(1, Ordering::Relaxed);
        let reads = Arc::clone(&self.reads);
        Ok(Box::new(self.source.batches(plan)?.inspect(move |batch| {
            if let Ok(batch) = batch {
                reads.rows.fetch_add(batch.len() as u64, Ordering::Relaxed);
            }
        })))
    }
}

// Counts actual underlying reads, including bootstrap/projection views sharing a source.
fn counted(source: impl EventSource + 'static, reads: &Arc<Reads>) -> Dataset {
    Dataset::new(CountingSource {
        source: Box::new(source),
        reads: Arc::clone(reads),
    })
    .fastest()
}

fn generated(events: usize, seed: u64) -> Result<MemorySource> {
    let schema = Arc::new(Schema::new(
        std::iter::empty::<&str>(),
        ["mass", "cos_theta", "phi"],
        true,
    )?);
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut batches = Vec::new();
    for offset in (0..events).step_by(4096) {
        let rows = (offset..events.min(offset + 4096)).map(|_| {
            OwnedEvent::weighted(
                vec![],
                vec![
                    1.0 + rng.f64(),
                    2.0 * rng.f64() - 1.0,
                    -PI + 2.0 * PI * rng.f64(),
                ],
                0.75 + 0.5 * rng.f64(),
            )
        });
        batches.push(EventBatch::from_events(Arc::clone(&schema), rows)?);
    }
    Ok(MemorySource::from_batches(batches)?)
}

fn model(waves: usize) -> Result<CompiledModel> {
    let terms = (0..waves).map(|wave| {
        let re = Expr::from(
            Parameter::free(format!("wave_{wave}_re"))
                .with_initial(if wave == 0 { 0.3 } else { 0.03 })
                .with_bounds(if wave == 0 { 0.001 } else { -2.0 }, 2.0),
        );
        // Fix the overall phase, avoiding an unidentifiable fit direction.
        let im = if wave == 0 {
            Expr::from(0.0)
        } else {
            Expr::from(
                Parameter::free(format!("wave_{wave}_im"))
                    .with_initial(0.01)
                    .with_bounds(-2.0, 2.0),
            )
        };
        let angle = event_scalar("phi") * wave as f64;
        let basis = complex(angle.clone().cos(), angle.sin())
            * (1.0 + 0.2 * (event_scalar("mass") * (wave + 1) as f64).cos())
            * (1.0 + 0.1 * event_scalar("cos_theta").powi((wave % 3 + 1) as i32));
        (complex(re, im) * basis).tagged(format!("wave_{wave}"))
    });
    let amplitude = terms.reduce(|sum, term| sum + term).ok_or("no waves")?;
    Ok(CompiledModel::from_expr(&amplitude.norm_sqr())?)
}

fn peak_rss() -> Result<u64> {
    let status = fs::read_to_string("/proc/self/status")?;
    let line = status
        .lines()
        .find(|line| line.starts_with("VmHWM:"))
        .ok_or("VmHWM unavailable")?;
    Ok(line
        .split_whitespace()
        .nth(1)
        .ok_or("invalid VmHWM")?
        .parse::<u64>()?
        * 1024)
}

struct Recorder {
    started: Instant,
    stages: BTreeMap<String, f64>,
    reads: Arc<Reads>,
    execution: Execution,
}

impl Recorder {
    fn measure<T>(&mut self, name: &str, action: impl FnOnce() -> Result<T>) -> Result<T> {
        let start = Instant::now();
        let result = action()?;
        let seconds = start.elapsed().as_secs_f64();
        self.stages.insert(name.into(), seconds);
        println!(
            "{}",
            json!({"stage": name, "seconds": seconds, "peak_rss_bytes": peak_rss()?,
            "source_traversals": self.reads.traversals.load(Ordering::Relaxed),
            "source_rows": self.reads.rows.load(Ordering::Relaxed),
            "memory": self.execution.memory_report()})
        );
        io::stdout().flush()?;
        Ok(result)
    }
}

fn diagnostics(likelihood: &Likelihood) -> Value {
    let report = likelihood.diagnostics();
    json!({
        "objective_evaluations": report.objective_evaluations(),
        "gradient_evaluations": report.gradient_evaluations(),
        "free_parameter_count": likelihood.default_params().len(),
        "datasets": report.datasets().iter().map(|dataset| json!({
            "term": dataset.term(), "role": format!("{:?}", dataset.role()),
            "resident_bytes": dataset.stats().resident_bytes(),
            "source_traversals": dataset.source_traversals(),
            "normalization": dataset.normalization().map(|normalization| json!({
                "strategy": format!("{:?}", normalization.strategy()),
                "retained_bytes": normalization.retained_bytes(),
                "preparation_passes": normalization.preparation_passes(),
                "cache_hit": normalization.cache_hit(),
            })),
        })).collect::<Vec<_>>(),
    })
}

fn fit(likelihood: &Likelihood, initial: &[f64], steps: usize) -> Result<(Vec<f64>, f64)> {
    let problem = FitProblem::<_, f64>::new(likelihood);
    let config = LBFGSBConfig::<f64>::default()
        .with_parameter_names(problem.parameter_names())
        .with_transform(problem.native_transform()?)?
        .with_bounds(problem.native_bounds())?;
    let fitted = LBFGSB::<f64>::default().process(
        &problem,
        &(),
        problem.vector(initial),
        config,
        LBFGSB::<f64>::default_callbacks().with_terminator(MaxSteps(steps)),
    )?;
    Ok((fitted.x.to_vec(), fitted.fx))
}

type Science = BTreeMap<String, Value>;

fn binned(estimate: &BinnedEstimate) -> Result<Value> {
    let covariance = estimate.covariance()?;
    Ok(
        json!({"values": estimate.values(), "draws": estimate.draws(),
        "bootstrap_std": covariance.iter().enumerate().map(|(i, row)| row[i].max(0.0).sqrt()).collect::<Vec<_>>()}),
    )
}

fn likelihood_workflow(config: &Config, recorder: &mut Recorder) -> Result<(Science, Value)> {
    let reads = Arc::clone(&recorder.reads);
    let (model, data, accepted, generated_mc) = recorder.measure("fixtures", || {
        Ok((
            model(config.waves)?,
            counted(generated(config.data_events, config.seed)?, &reads),
            counted(generated(config.mc_events, config.seed + 1)?, &reads),
            if config.workflow.starts_with("bootstrap") {
                Some(counted(
                    generated(config.mc_events, config.seed + 2)?,
                    &reads,
                ))
            } else {
                None
            },
        ))
    })?;
    let execution = recorder.execution.clone();
    let likelihood = recorder.measure("likelihood_preparation", || {
        Ok(Arc::new(Likelihood::with_execution(
            [ExtendedNllTerm::new("sample", &model, &data, &accepted)?],
            &execution,
        )?))
    })?;
    let initial = likelihood.default_params();
    let value = recorder.measure("fixed_nll", || {
        let mut value = 0.0;
        for _ in 0..config.evaluations {
            value = std::hint::black_box(likelihood.nll(&initial)?);
        }
        Ok(value)
    })?;
    let gradient = recorder.measure("fixed_gradient", || {
        let mut result = None;
        for _ in 0..config.evaluations {
            result = Some(std::hint::black_box(
                likelihood.nll_with_gradient(&initial)?,
            ));
        }
        Ok(result.ok_or("no gradient evaluations")?.into_parts())
    })?;
    let (parameters, nll) = recorder.measure("bounded_fit", || {
        fit(&likelihood, &initial, config.fit_steps)
    })?;
    let mut science = Science::from([
        ("initial_nll".into(), json!(value)),
        ("initial_gradient".into(), json!(gradient.1)),
        ("fit_nll".into(), json!(nll)),
        ("fit_parameters".into(), json!(parameters)),
    ]);
    if !config.workflow.starts_with("bootstrap") {
        return Ok((science, diagnostics(&likelihood)));
    }

    let mut replica_nlls = Vec::new();
    let ensemble = recorder.measure("bootstrap_fits", || {
        if config.workflow.starts_with("bootstrap-arbitrary") {
            // Independent rows and MC deliberately have no native sharing provenance.
            let mut replicas = Vec::with_capacity(config.replicas);
            let mut draws = Vec::with_capacity(config.replicas);
            for index in 0..config.replicas {
                let data = counted(
                    generated(config.data_events, config.seed + 1000 + index as u64)?,
                    &reads,
                );
                let accepted = if config.workflow == "bootstrap-arbitrary-data" {
                    accepted.clone()
                } else {
                    counted(
                        generated(config.mc_events, config.seed + 2000 + index as u64)?,
                        &reads,
                    )
                };
                let replica = Arc::new(Likelihood::with_execution(
                    [ExtendedNllTerm::new("sample", &model, &data, &accepted)?],
                    &execution,
                )?);
                let (draw, nll) = fit(&replica, &parameters, config.fit_steps)?;
                replica_nlls.push(nll);
                draws.push(draw);
                replicas.push(replica);
            }
            let names = likelihood
                .params()
                .free_params()
                .iter()
                .map(|id| likelihood.params().name(*id).map(str::to_owned))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            return Ok(Ensemble::with_replicas(names, draws, replicas)?);
        }
        Ok(Ensemble::bootstrap_fit(
            &likelihood,
            config.replicas,
            config.seed + 100,
            |replica, _| -> Result<Vec<f64>> {
                let (parameters, nll) = fit(replica, &parameters, config.fit_steps)?;
                replica_nlls.push(nll);
                Ok(parameters)
            },
        )
        .map_err(|error| error.to_string())?)
    })?;
    science.insert("replica_parameters".into(), json!(ensemble.draws()));
    science.insert("replica_nlls".into(), json!(replica_nlls));
    let pairing = json!({"replicas": ensemble.replicas().len(),
        "draws": ensemble.len(), "bootstrap_seed": ensemble.bootstrap_seed()});
    let replica_diagnostics = ensemble
        .replicas()
        .iter()
        .map(|replica| diagnostics(replica))
        .collect::<Vec<_>>();
    let mut projections = vec![
        Projection::new(
            "mass",
            vec![Axis::new(
                event_scalar("mass"),
                edges(1.0, 2.0, config.bins),
            )?],
        )?,
        Projection::new(
            "cos_theta",
            vec![Axis::new(
                event_scalar("cos_theta"),
                edges(-1.0, 1.0, config.bins),
            )?],
        )?,
    ];
    if let Some(count) = config.projection_count {
        if !(1..=4).contains(&count) {
            return Err("projection_count must be between one and four".into());
        }
        projections.push(Projection::new(
            "mass_fine",
            vec![Axis::new(
                event_scalar("mass"),
                edges(1.0, 2.0, config.bins * 2),
            )?],
        )?);
        projections.push(Projection::new(
            "mass_cos_theta",
            vec![
                Axis::new(event_scalar("mass"), edges(1.0, 2.0, config.bins))?,
                Axis::new(event_scalar("cos_theta"), edges(-1.0, 1.0, config.bins))?,
            ],
        )?);
        projections.truncate(count);
    }
    let component_count = config.projection_components.unwrap_or(config.waves);
    if component_count > config.waves {
        return Err("projection_components exceeds wave count".into());
    }
    let components = (0..component_count)
        .map(|wave| (format!("wave_{wave}"), vec![format!("wave_{wave}")]))
        .collect::<HashMap<_, _>>();
    let batch_profile = if config.profile_batch == Some(true) || config.verify_blocks == Some(true)
    {
        Some(profile_projection_batch(
            &model,
            &accepted,
            &execution,
            &parameters,
            &ensemble,
            component_count,
            config.verify_blocks == Some(true),
        )?)
    } else {
        None
    };
    // Optional differential check: both APIs receive these exact fitted inputs.
    // Verification runs are separate from performance measurements because they
    // execute and retain both paths in one process.
    let separate_inputs = std::env::var_os("LADDU_CPU_VERIFY_SHARED").map(|_| {
        (
            generated_mc.clone().expect("bootstrap generated MC"),
            parameters.clone(),
            ensemble.clone(),
        )
    });
    // The scalar modes exercise the standalone preparation path separately.
    let section = recorder.measure("cross_section_total", || {
        if config.workflow != "bootstrap" {
            return Ok(likelihood.cross_section(
                "sample",
                generated_mc.ok_or("bootstrap requires generated MC")?,
                Luminosity::new(10.0, AreaUnit::Nanobarn)?,
                parameters,
                Some(ensemble),
            )?);
        }
        Ok(likelihood.cross_section_with_projections(
            "sample",
            generated_mc.ok_or("bootstrap requires generated MC")?,
            Luminosity::new(10.0, AreaUnit::Nanobarn)?,
            parameters,
            Some(ensemble),
            &projections,
            &components,
        )?)
    })?;
    science.insert(
        "cross_section".into(),
        json!({"value": section.total().value(),
        "draws": section.total().draws(), "bootstrap_std": section.total().std()?}),
    );
    let projected = recorder.measure("component_projections", || {
        Ok(section.project_many(&projections, &components)?)
    })?;
    for (name, projection) in projected {
        science.insert(format!("projection_{name}"), binned(projection.total())?);
        for (component, estimate) in projection.components() {
            science.insert(format!("projection_{name}_{component}"), binned(estimate)?);
        }
    }
    let separate_science = if let Some((generated, parameters, ensemble)) = separate_inputs {
        let separate = likelihood.cross_section(
            "sample",
            generated,
            Luminosity::new(10.0, AreaUnit::Nanobarn)?,
            parameters,
            Some(ensemble),
        )?;
        let mut reference = science.clone();
        reference.insert(
            "cross_section".into(),
            json!({"value": separate.total().value(), "draws": separate.total().draws(),
                "bootstrap_std": separate.total().std()?}),
        );
        for (name, projection) in separate.project_many(&projections, &components)? {
            reference.insert(format!("projection_{name}"), binned(projection.total())?);
            for (component, estimate) in projection.components() {
                reference.insert(format!("projection_{name}_{component}"), binned(estimate)?);
            }
        }
        Some(reference)
    } else {
        None
    };
    Ok((
        science,
        json!({"likelihood": diagnostics(&likelihood), "replica_diagnostics": replica_diagnostics,
        "pairing": pairing, "component_count": components.len(),
        "projection_names": projections.iter().map(Projection::name).collect::<Vec<_>>(),
        "projection_axes": ["mass", "cos_theta"],
        "batch_profile": batch_profile,
        "separate_science": separate_science}),
    ))
}

// Isolate the event/parameter kernel from projection construction and binning.
// This optional diagnostic adds work and is excluded from workflow measurements.
fn profile_projection_batch(
    model: &CompiledModel,
    accepted: &Dataset,
    execution: &Execution,
    central: &[f64],
    ensemble: &Ensemble,
    components: usize,
    verify_blocks: bool,
) -> Result<Value> {
    let mut read_plan = accepted.read_plan();
    read_plan.chunk_size = Some(8192);
    let batch = accepted
        .stream_with_plan(read_plan)?
        .next()
        .ok_or("no batch to profile")??;
    let parameters = std::iter::once(central)
        .chain(ensemble.draws().iter().map(Vec::as_slice))
        .map(|draw| model.params().values(draw))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let mut timings = BTreeMap::new();
    let mut verified_values = 0usize;
    for index in 0..=components {
        let name = if index == 0 {
            "full".into()
        } else {
            format!("wave_{}", index - 1)
        };
        let selected = if index == 0 {
            model.clone()
        } else {
            model.project_tags([name.as_str()])?
        };
        let projection = selected.params().projection_from(model.params())?;
        let local = parameters
            .iter()
            .map(|parameters| projection.project(parameters))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let plan = laddu_runtime::PreparedModel::prepare(&selected, execution)?;
        let _workspace = execution
            .host_memory()
            .reserve(plan.batch_memory_estimate(batch.len()) as u64)?;
        let started = Instant::now();
        plan.visit_batch_many(execution, &local, &batch, |_, values| {
            std::hint::black_box(values);
            Ok(())
        })?;
        timings.insert(name, started.elapsed().as_secs_f64());
        if verify_blocks {
            // The serial visitor remains an independent, existing evaluation path.
            let serial = Execution::local(ExecutionOptions {
                device: Device::Cpu(CpuOptions {
                    threads: ThreadPolicy::Fixed(1),
                    jit: JitPolicy::Disabled,
                }),
                ..ExecutionOptions::default()
            })?;
            let bytes = local
                .len()
                .checked_mul(batch.len())
                .and_then(|n| n.checked_mul(std::mem::size_of::<[f64; 2]>()))
                .ok_or("verification workspace overflow")?;
            let _reference_workspace = execution.host_memory().reserve(bytes as u64)?;
            let mut reference = Vec::new();
            plan.visit_batch_many(&serial, &local, &batch, |_, values| {
                reference.push(values.to_vec());
                Ok(())
            })?;
            plan.visit_batch_many(execution, &local, &batch, |parameter, values| {
                let expected = &reference[parameter];
                if values.len() != expected.len()
                    || values.iter().zip(expected).any(|(actual, expected)| {
                        actual.re.to_bits() != expected.re.to_bits()
                            || actual.im.to_bits() != expected.im.to_bits()
                    })
                {
                    return Err(laddu_runtime::RuntimeError::Parameter(
                        "parallel and serial event evaluation differ at identical fitted inputs"
                            .into(),
                    ));
                }
                verified_values += values.len();
                Ok(())
            })?;
        }
    }
    Ok(
        json!({"batch_events": batch.len(), "parameter_vectors": parameters.len(),
        "components_seconds": timings, "verified_values": verified_values,
        "bitwise_equal_at_identical_inputs": verify_blocks.then_some(true)}),
    )
}

fn edges(lower: f64, upper: f64, bins: usize) -> Vec<f64> {
    (0..=bins)
        .map(|bin| lower + (upper - lower) * bin as f64 / bins as f64)
        .collect()
}

fn parquet_workflow(
    config: &Config,
    recorder: &mut Recorder,
    directory: &Path,
) -> Result<(Science, Value)> {
    let reads = Arc::clone(&recorder.reads);
    recorder.measure("parquet_generation", || {
        fs::create_dir_all(directory)?;
        for period in 0..config.periods {
            for shard in 0..config.shards_per_period {
                let count = config.mc_events / config.shards_per_period
                    + usize::from(shard < config.mc_events % config.shards_per_period);
                let source = generated(
                    count,
                    config.seed + (period * config.shards_per_period + shard) as u64,
                )?;
                let dataset = counted(source, &reads);
                let mut sink = ParquetSink::builder(
                    directory.join(format!("period_{period}_{shard}.parquet")),
                )
                .build();
                dataset.write_to(&mut sink)?;
            }
        }
        Ok(())
    })?;
    let sources = recorder.measure("parquet_metadata", || {
        (0..config.periods)
            .map(|period| {
                Ok(counted(
                    ParquetSource::open(
                        directory
                            .join(format!("period_{period}_*.parquet"))
                            .to_str()
                            .ok_or("non-UTF8 path")?,
                    )?,
                    &reads,
                ))
            })
            .collect::<Result<Vec<_>>>()
    })?;
    let loaded = recorder.measure("parquet_loading", || {
        sources
            .iter()
            .map(|dataset| {
                let batches = dataset.batches()?.collect::<LadduDataResult<Vec<_>>>()?;
                Ok(counted(MemorySource::from_batches(batches)?, &reads))
            })
            .collect::<Result<Vec<_>>>()
    })?;
    let execution = recorder.execution.clone();
    let bins = recorder.measure("mass_binning", || {
        loaded
            .iter()
            .map(|dataset| {
                Ok(dataset.bin_by(
                    &event_scalar("mass"),
                    BinSpec::edges(edges(1.0, 2.0, config.bins))?,
                    &execution,
                )?)
            })
            .collect::<Result<Vec<_>>>()
    })?;
    let mut science = Science::new();
    for (period, bins) in bins.iter().enumerate() {
        let stats = bins
            .iter()
            .map(|bin| bin.dataset().stats())
            .collect::<LadduDataResult<Vec<_>>>()?;
        science.insert(
            format!("period_{period}_bin_events"),
            json!(stats.iter().map(|stats| stats.events()).collect::<Vec<_>>()),
        );
        science.insert(
            format!("period_{period}_bin_weights"),
            json!(
                stats
                    .iter()
                    .map(|stats| stats.sum_weights())
                    .collect::<Vec<_>>()
            ),
        );
    }
    Ok((
        science,
        json!({"periods": config.periods, "shards_per_period": config.shards_per_period,
        "mass_edges": edges(1.0, 2.0, config.bins)}),
    ))
}

fn main() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 2 {
        return Err("expected CONFIG.json PARQUET_DIRECTORY".into());
    }
    let config: Config = serde_json::from_slice(&fs::read(&args[0])?)?;
    if ![
        "likelihood-5",
        "likelihood-12",
        "bootstrap",
        "bootstrap-scalar",
        "bootstrap-arbitrary",
        "bootstrap-arbitrary-data",
        "parquet",
    ]
    .contains(&config.workflow.as_str())
        || [
            config.data_events,
            config.mc_events,
            config.waves,
            config.replicas,
            config.threads,
            config.evaluations,
            config.fit_steps,
            config.periods,
            config.shards_per_period,
            config.bins,
        ]
        .contains(&0)
        || config.mc_events < config.shards_per_period
    {
        return Err("invalid workflow or zero/insufficient counts".into());
    }
    let execution = Execution::local(ExecutionOptions {
        device: Device::Cpu(CpuOptions {
            threads: ThreadPolicy::Fixed(config.threads),
            jit: JitPolicy::Disabled,
        }),
        precision: Precision::F64,
        ..ExecutionOptions::default()
    })?;
    let mut recorder = Recorder {
        started: Instant::now(),
        stages: BTreeMap::new(),
        reads: Arc::new(Reads::default()),
        execution,
    };
    let (science, diagnostics) = if config.workflow == "parquet" {
        parquet_workflow(&config, &mut recorder, Path::new(&args[1]))?
    } else {
        likelihood_workflow(&config, &mut recorder)?
    };
    println!(
        "{}",
        json!({"result": {
            "config": config, "stages_seconds": recorder.stages,
            "total_seconds": recorder.started.elapsed().as_secs_f64(), "peak_rss_bytes": peak_rss()?,
            "source_traversals": recorder.reads.traversals.load(Ordering::Relaxed),
            "source_rows": recorder.reads.rows.load(Ordering::Relaxed),
            "science": science, "diagnostics": diagnostics, "memory": recorder.execution.memory_report(),
            "execution": {"device": "cpu", "precision": "f64", "jit": "disabled", "normalization": "auto", "storage": "fastest"},
            "weights": {"distribution": "uniform", "lower": 0.75, "upper": 1.25},
        }})
    );
    Ok(())
}
