//! Shared in-memory generation-to-fit closure workflow.

use std::{
    error::Error,
    fs::{self, File},
    io::BufWriter,
    path::Path,
    time::{Duration, Instant},
};

use laddu::prelude::ganesh::{
    algorithms::gradient::{LBFGSB, LBFGSBConfig},
    core::{MaxSteps, MinimizationSummary},
    traits::{Algorithm, SupportsParameterNames},
};
use laddu::prelude::*;
use serde::Serialize;

use super::ksks::{
    F2_MAGNITUDE_TRUTH, F2_PHASE_TRUTH, ksks_channel, ksks_intensity, truth_parameters,
};

#[derive(Clone, Copy, Debug)]
pub struct ClosureConfig {
    pub data_events: usize,
    pub normalization_events: usize,
    pub pilot_proposals: usize,
    pub seed: u64,
    pub max_fit_steps: usize,
    pub projection_bins: usize,
}

impl Default for ClosureConfig {
    fn default() -> Self {
        Self {
            data_events: 10_000,
            normalization_events: 100_000,
            pilot_proposals: 50_000,
            seed: 0x0043_4c4f_5355_5245,
            max_fit_steps: 200,
            projection_bins: 50,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct HistogramSeries {
    pub id: &'static str,
    pub label: &'static str,
    pub histogram: Histogram,
}

#[derive(Debug, Serialize)]
pub struct FitParameter {
    pub name: String,
    pub value: f64,
    pub truth: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct ProjectionHistogram {
    pub schema_version: u32,
    pub title: &'static str,
    pub observable: &'static str,
    pub unit: &'static str,
    pub data: HistogramSeries,
    pub projections: Vec<HistogramSeries>,
    pub fit_parameters: Vec<FitParameter>,
    pub note: &'static str,
}

impl ProjectionHistogram {
    pub fn write_json(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn Error>> {
        let path = path.as_ref();
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)?;
        }
        let writer = BufWriter::new(File::create(path)?);
        serde_json::to_writer_pretty(writer, self)?;
        Ok(())
    }
}

pub struct ClosureResult {
    pub data_report: GenerationReport,
    pub normalization_report: GenerationReport,
    pub initial_nll: f64,
    pub initial_gradient: Vec<f64>,
    pub fit: MinimizationSummary,
    pub projection: ProjectionHistogram,
    pub data_generation_time: Duration,
    pub normalization_generation_time: Duration,
    pub likelihood_preparation_time: Duration,
    pub fit_time: Duration,
    pub projection_time: Duration,
}

impl ClosureResult {
    pub fn fitted(&self, name: &str) -> Option<f64> {
        self.fit
            .parameter_names
            .as_ref()?
            .iter()
            .position(|candidate| candidate == name)
            .map(|index| self.fit.x.get(index))
    }
}

fn fitted_parameter(fit: &MinimizationSummary, name: &str) -> Result<f64, Box<dyn Error>> {
    let index = fit
        .parameter_names
        .as_ref()
        .and_then(|names| names.iter().position(|candidate| candidate == name))
        .ok_or_else(|| format!("fit result does not contain parameter `{name}`"))?;
    Ok(fit.x.get(index))
}

pub fn wrapped_phase_residual(value: f64, truth: f64) -> f64 {
    (value - truth).sin().atan2((value - truth).cos())
}

pub fn run_closure(
    config: ClosureConfig,
    execution: Execution,
) -> Result<ClosureResult, Box<dyn Error>> {
    if config.data_events == 0
        || config.normalization_events == 0
        || config.pilot_proposals == 0
        || config.max_fit_steps == 0
        || config.projection_bins == 0
    {
        return Err("closure event counts, pilot proposals, and fit steps must be nonzero".into());
    }

    let channel = ksks_channel()?;
    let mass = channel.mass("X")?;
    let model = CompiledModel::from_expr(&ksks_intensity(&channel)?)?;
    let evaluator = ModelEvaluator::prepare(&model, truth_parameters(&model)?, &execution)?;
    let generator = ChannelGenerator::new(channel)?;

    let data_start = Instant::now();
    let (data, data_report) = generator.generate_unweighted_dataset(
        UnweightedConfig {
            events: config.data_events,
            max_proposals: None,
            memory: MemoryBudget::Bytes(256 * 1024 * 1024),
            seed: config.seed,
            diagnostics: false,
            envelope: EnvelopeMode::Pilot {
                proposals: config.pilot_proposals,
                safety_factor: 2.0,
            },
            envelope_overflow: EnvelopeOverflow::Grow { safety_factor: 1.5 },
        },
        Some(&evaluator),
    )?;
    let data_generation_time = data_start.elapsed();

    let normalization_start = Instant::now();
    let (normalization, normalization_report) = generator.generate_weighted_dataset(
        WeightedConfig {
            events: config.normalization_events,
            memory: MemoryBudget::Bytes(256 * 1024 * 1024),
            seed: config.seed.wrapping_add(1),
            diagnostics: false,
        },
        None,
    )?;
    let normalization_generation_time = normalization_start.elapsed();

    let likelihood_start = Instant::now();
    let likelihood = Likelihood::with_execution(
        [NllTerm::new("ksks", &model, &data, &normalization)?],
        &execution,
    )?;
    let likelihood_preparation_time = likelihood_start.elapsed();
    let initial = likelihood.sample_initial(config.seed.wrapping_add(2));
    let initial_evaluation = likelihood.nll_with_gradient(&initial)?;
    let (initial_nll, initial_gradient) = initial_evaluation.into_parts();
    let fit_start = Instant::now();
    let problem = FitProblem::<_, f64>::new(&likelihood);
    let fit_config = LBFGSBConfig::<f64>::default()
        .with_parameter_names(problem.parameter_names())
        .with_transform(problem.native_transform()?)?
        .with_bounds(problem.native_bounds())?;
    let fit = LBFGSB::<f64>::default().process(
        &problem,
        &(),
        problem.vector(&initial),
        fit_config,
        LBFGSB::<f64>::default_callbacks().with_terminator(MaxSteps(config.max_fit_steps)),
    )?;
    let fit_time = fit_start.elapsed();

    let projection_start = Instant::now();
    let fitted_magnitude = fitted_parameter(&fit, "f2_magnitude")?;
    let fitted_phase = fitted_parameter(&fit, "f2_phase")?;
    let fitted = fit.x.to_vec();
    let projection = build_projection(
        &likelihood,
        &data,
        &normalization,
        &mass,
        &fitted,
        fitted_magnitude,
        fitted_phase,
        config.projection_bins,
        &execution,
    )?;
    let projection_time = projection_start.elapsed();

    Ok(ClosureResult {
        data_report,
        normalization_report,
        initial_nll,
        initial_gradient,
        fit,
        projection,
        data_generation_time,
        normalization_generation_time,
        likelihood_preparation_time,
        fit_time,
        projection_time,
    })
}

#[allow(
    clippy::too_many_arguments,
    reason = "projection inputs are distinct analysis products and configuration values"
)]
fn build_projection(
    likelihood: &Likelihood,
    data: &Dataset,
    normalization: &Dataset,
    mass: &Expr,
    fitted: &[f64],
    fitted_magnitude: f64,
    fitted_phase: f64,
    bins: usize,
    execution: &Execution,
) -> Result<ProjectionHistogram, Box<dyn Error>> {
    let minimum = 2.0 * particles::K_SHORT.mass()?;
    let maximum = 2.0;
    let data_masses = data.evaluate_real(mass, execution)?;
    let normalization_masses = normalization.evaluate_real(mass, execution)?;
    let data_weights = dataset_weights(data)?;
    let coherent = likelihood.projection("ksks", normalization, ["f0", "f2"])?;
    let f0 = likelihood.projection("ksks", normalization, ["f0"])?;
    let f2 = likelihood.projection("ksks", normalization, ["f2"])?;
    let data_histogram =
        Histogram::from_values(&data_masses, bins, (minimum, maximum), Some(&data_weights))?;
    let coherent_histogram = Histogram::from_values(
        &normalization_masses,
        bins,
        (minimum, maximum),
        Some(&coherent.weights(fitted, true)?),
    )?;
    let f0_histogram = Histogram::from_values(
        &normalization_masses,
        bins,
        (minimum, maximum),
        Some(&f0.weights(fitted, true)?),
    )?;
    let f2_histogram = Histogram::from_values(
        &normalization_masses,
        bins,
        (minimum, maximum),
        Some(&f2.weights(fitted, true)?),
    )?;

    Ok(ProjectionHistogram {
        schema_version: 2,
        title: "$\\gamma p \\to K_S^0 K_S^0 p\\;\\mathrm{closure\\ fit}$",
        observable: "$m(K_S^0 K_S^0)$",
        unit: "GeV",
        data: HistogramSeries {
            id: "data",
            label: "$\\mathrm{Pseudo\\ data}$",
            histogram: data_histogram,
        },
        projections: vec![
            HistogramSeries {
                id: "fit",
                label: "$\\mathrm{Coherent\\ fit}$",
                histogram: coherent_histogram,
            },
            HistogramSeries {
                id: "f0",
                label: "$f_0(1500)$",
                histogram: f0_histogram,
            },
            HistogramSeries {
                id: "f2",
                label: "$f_2(1270)$",
                histogram: f2_histogram,
            },
        ],
        fit_parameters: vec![
            FitParameter {
                name: "f2_magnitude".into(),
                value: fitted_magnitude,
                truth: Some(F2_MAGNITUDE_TRUTH),
            },
            FitParameter {
                name: "f2_phase".into(),
                value: fitted_phase,
                truth: Some(F2_PHASE_TRUTH),
            },
        ],
        note: "Isolated resonances use the fitted coupling and the coherent fit's normalization; interference is shown only in the coherent curve.",
    })
}

fn dataset_weights(dataset: &Dataset) -> Result<Vec<f64>, Box<dyn Error>> {
    let mut weights = Vec::new();
    for batch in dataset.batches()? {
        let batch = batch?;
        weights.extend((0..batch.len()).map(|row| batch.weights_at(row)));
    }
    Ok(weights)
}
