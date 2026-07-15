//! Shared in-memory generation-to-fit closure workflow.

use std::{
    error::Error,
    fs::{self, File},
    io::{self, BufWriter},
    path::Path,
    time::{Duration, Instant},
};

use laddu::data::data::BatchEvent;
use laddu::prelude::ganesh::{
    algorithms::gradient::{LBFGSB, LBFGSBConfig},
    core::MaxSteps,
    traits::Algorithm,
};
use laddu::prelude::*;
use serde::Serialize;

use super::ksks::{
    F2_MAGNITUDE_TRUTH, F2_PHASE_TRUTH, ksks_channel, ksks_intensities, truth_parameters,
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
    pub values: Vec<f64>,
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
    pub bin_edges: Vec<f64>,
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
    pub fit: MinimizationResult,
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
            .parameters()
            .into_iter()
            .find_map(|(candidate, value)| (candidate == name).then_some(value))
    }
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
    let intensities = ksks_intensities(&channel)?;
    let model = CompiledModel::from_expr(&intensities.coherent)?;
    let f0_model = CompiledModel::from_expr(&intensities.f0)?;
    let f2_model = CompiledModel::from_expr(&intensities.f2)?;
    let evaluator = ModelEvaluator::prepare(&model, truth_parameters(&model)?, &execution)?;
    let generator = ChannelGenerator::new(channel)?;

    let data_start = Instant::now();
    let (data, data_report) = generator.generate_unweighted_dataset(
        UnweightedConfig {
            events: config.data_events,
            max_proposals: None,
            batch_size: 2_048,
            seed: config.seed,
            diagnostics: false,
            envelope_overflow: EnvelopeOverflow::Grow { safety_factor: 1.5 },
        },
        &evaluator,
        EnvelopeMode::Pilot {
            proposals: config.pilot_proposals,
            safety_factor: 2.0,
        },
    )?;
    let data_generation_time = data_start.elapsed();

    let normalization_start = Instant::now();
    let (normalization, normalization_report) = generator.generate_weighted_dataset(
        WeightedConfig {
            events: config.normalization_events,
            batch_size: 4_096,
            seed: config.seed.wrapping_add(1),
            diagnostics: false,
        },
        None,
    )?;
    let normalization_generation_time = normalization_start.elapsed();

    let likelihood_start = Instant::now();
    let likelihood = Likelihood::with_execution(
        [NllTerm::new("ksks", &model, &data, &normalization)?.boxed()],
        execution.clone(),
    )?;
    let likelihood_preparation_time = likelihood_start.elapsed();
    let initial_evaluation =
        likelihood.nll_with_gradient(likelihood.default_params().as_slice())?;
    let (initial_nll, initial_gradient) = initial_evaluation.into_parts();

    let problem = FitProblem::<_, f64>::new(&likelihood);
    let initial = problem.initial();
    let fit_config =
        problem.configure_lbfgsb(LBFGSBConfig::<f64>::default(), TransformOptions::default())?;
    let fit_start = Instant::now();
    let fit = problem.minimize(
        &mut LBFGSB::<f64>::default(),
        initial,
        fit_config,
        LBFGSB::<f64>::default_callbacks().with_terminator(MaxSteps(config.max_fit_steps)),
    )?;
    let fit_time = fit_start.elapsed();

    let projection_start = Instant::now();
    let fitted = fit.parameters();
    let fitted_magnitude = fitted_parameter(&fitted, "f2_magnitude")?;
    let fitted_phase = fitted_parameter(&fitted, "f2_phase")?;
    let projection = build_projection(
        &data,
        &normalization,
        [&model, &f0_model, &f2_model],
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

fn fitted_parameter(parameters: &[(String, f64)], name: &str) -> Result<f64, io::Error> {
    parameters
        .iter()
        .find_map(|(candidate, value)| (candidate == name).then_some(*value))
        .ok_or_else(|| io::Error::other(format!("fit did not return `{name}`")))
}

fn fitted_parameters(
    model: &CompiledModel,
    magnitude: f64,
    phase: f64,
) -> Result<ParamValues, ParamError> {
    let free = model
        .params()
        .free_values_with(|parameter| match parameter.name() {
            "f2_magnitude" => magnitude,
            "f2_phase" => phase,
            name => panic!("unexpected free parameter `{name}` in K_S K_S projection"),
        });
    model.params().values(&free)
}

fn build_projection(
    data: &Dataset,
    normalization: &Dataset,
    models: [&CompiledModel; 3],
    fitted_magnitude: f64,
    fitted_phase: f64,
    bins: usize,
    execution: &Execution,
) -> Result<ProjectionHistogram, Box<dyn Error>> {
    let minimum = 2.0 * particles::K_SHORT.mass()?;
    let maximum = 2.0;
    let width = (maximum - minimum) / bins as f64;
    let bin_edges = (0..=bins)
        .map(|index| minimum + index as f64 * width)
        .collect::<Vec<_>>();
    let mut data_values = vec![0.0; bins];
    for batch in data.batches()? {
        let batch = batch?;
        for row in 0..batch.len() {
            let event = batch.event(row);
            if let Some(bin) = bin_index(ksks_mass(&event)?, minimum, maximum, bins) {
                data_values[bin] += event.weight();
            }
        }
    }

    let evaluators = [
        ModelEvaluator::prepare(
            models[0],
            fitted_parameters(models[0], fitted_magnitude, fitted_phase)?,
            execution,
        )?,
        ModelEvaluator::prepare(
            models[1],
            fitted_parameters(models[1], fitted_magnitude, fitted_phase)?,
            execution,
        )?,
        ModelEvaluator::prepare(
            models[2],
            fitted_parameters(models[2], fitted_magnitude, fitted_phase)?,
            execution,
        )?,
    ];
    let mut projection_values = [vec![0.0; bins], vec![0.0; bins], vec![0.0; bins]];
    for batch in normalization.batches()? {
        let batch = batch?;
        let values = [
            evaluators[0].evaluate_batch(&batch)?,
            evaluators[1].evaluate_batch(&batch)?,
            evaluators[2].evaluate_batch(&batch)?,
        ];
        for (row, ((coherent, f0), f2)) in
            values[0].iter().zip(&values[1]).zip(&values[2]).enumerate()
        {
            let event = batch.event(row);
            let Some(bin) = bin_index(ksks_mass(&event)?, minimum, maximum, bins) else {
                continue;
            };
            let weight = event.weight();
            projection_values[0][bin] += weight * coherent;
            projection_values[1][bin] += weight * f0;
            projection_values[2][bin] += weight * f2;
        }
    }

    let data_yield = data_values.iter().sum::<f64>();
    let fit_yield = projection_values[0].iter().sum::<f64>();
    if !fit_yield.is_finite() || fit_yield <= 0.0 {
        return Err(io::Error::other("fitted projection has nonpositive normalization").into());
    }
    let scale = data_yield / fit_yield;
    for values in &mut projection_values {
        for value in values {
            *value *= scale;
        }
    }

    Ok(ProjectionHistogram {
        schema_version: 1,
        title: "γp → KₛKₛp closure fit",
        observable: "m(KₛKₛ)",
        unit: "GeV",
        bin_edges,
        data: HistogramSeries {
            id: "data",
            label: "Pseudo-data",
            values: data_values,
        },
        projections: vec![
            HistogramSeries {
                id: "fit",
                label: "Coherent fit",
                values: projection_values[0].clone(),
            },
            HistogramSeries {
                id: "f0",
                label: "f₀(1500)",
                values: projection_values[1].clone(),
            },
            HistogramSeries {
                id: "f2",
                label: "f₂(1270)",
                values: projection_values[2].clone(),
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

fn ksks_mass(event: &BatchEvent<'_>) -> Result<f64, Box<dyn Error>> {
    let ks1 = event
        .p4_named("ks1")
        .ok_or_else(|| io::Error::other("projection dataset is missing `ks1`"))?;
    let ks2 = event
        .p4_named("ks2")
        .ok_or_else(|| io::Error::other("projection dataset is missing `ks2`"))?;
    Ok((ks1 + ks2).m()?)
}

fn bin_index(value: f64, minimum: f64, maximum: f64, bins: usize) -> Option<usize> {
    if !value.is_finite() || value < minimum || value > maximum {
        return None;
    }
    if value == maximum {
        return Some(bins - 1);
    }
    Some((((value - minimum) / (maximum - minimum)) * bins as f64) as usize)
}
