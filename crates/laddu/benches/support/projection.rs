//! Deterministic cross-section projection workloads shared by tests and benchmarks.

#![allow(dead_code, reason = "each benchmark uses a subset of the fixture API")]

use std::{collections::HashMap, error::Error, f64::consts::PI, sync::Arc};

use laddu::prelude::*;

const FIXTURE_SEED: u64 = 0x50_52_4f_4a_45_43_54;
const SCALARS: [&str; 4] = ["cos_theta", "phi", "cos_theta_decay", "phi_decay"];

type FixtureResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

/// Prepared-data residency used by a projection workload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Storage {
    /// Ask the memory planner to retain prepared event data.
    Resident,
    /// Traverse prepared event data as a stream.
    Streaming,
}

#[derive(Clone, Copy)]
enum DatasetRole {
    Data,
    Accepted,
    Generated,
}

impl DatasetRole {
    const fn seed_offset(self) -> u64 {
        match self {
            Self::Data => 0,
            Self::Accepted => 1,
            Self::Generated => 2,
        }
    }
}

impl Storage {
    fn apply(self, dataset: Dataset) -> Dataset {
        match self {
            Self::Resident => dataset.fastest(),
            Self::Streaming => dataset.streaming(),
        }
    }

    /// Stable label used in benchmark identifiers and reports.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Resident => "resident",
            Self::Streaming => "streaming",
        }
    }
}

/// Representative four-period workload for independent differential calls.
pub struct ProjectionFixture {
    single: CrossSection,
    combined: CrossSection,
    axes: [Axis; 4],
    selections: HashMap<String, Vec<String>>,
}

impl ProjectionFixture {
    /// Builds a deterministic workload with four run periods and `draws` ensemble draws.
    pub fn new(
        events_per_sample: usize,
        draws: usize,
        storage: Storage,
        threads: ThreadPolicy,
    ) -> FixtureResult<Self> {
        let model = projection_model()?;
        let execution = Execution::local(ExecutionOptions {
            device: Device::Cpu(CpuOptions {
                threads,
                jit: JitPolicy::Disabled,
            }),
            ..ExecutionOptions::default()
        })?;

        let mut generated = Vec::with_capacity(4);
        let mut terms = Vec::with_capacity(4);
        for period in 0..4 {
            let data = dataset(events_per_sample, period, DatasetRole::Data, storage)?;
            let accepted = dataset(events_per_sample, period, DatasetRole::Accepted, storage)?;
            generated.push(dataset(
                events_per_sample,
                period,
                DatasetRole::Generated,
                storage,
            )?);
            terms.push(NllTerm::new(
                format!("period_{period}"),
                &model,
                &data,
                &accepted,
            )?);
        }
        let likelihood = Arc::new(Likelihood::with_execution(terms, &execution)?);
        let ensemble = deterministic_ensemble(&likelihood, draws)?;
        let parameters = likelihood.default_params();
        let members = generated
            .into_iter()
            .enumerate()
            .map(|(period, generated)| {
                likelihood.cross_section_with_ensemble(
                    format!("period_{period}"),
                    generated,
                    10.0 + period as f64 * 2.5,
                    parameters.clone(),
                    ensemble.clone(),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let single = members[0].clone();
        let combined = CrossSection::combine(members)?;

        Ok(Self {
            single,
            combined,
            axes: projection_axes()?,
            selections: HashMap::from([
                ("signal".to_owned(), vec!["signal".to_owned()]),
                (
                    "signal_alias".to_owned(),
                    vec!["signal".to_owned(), "signal".to_owned()],
                ),
                ("background".to_owned(), vec!["background".to_owned()]),
            ]),
        })
    }

    /// Number of run-period members in the representative combined cross section.
    pub const fn member_count(&self) -> usize {
        4
    }

    /// Number of independent one-dimensional projections in the workload.
    pub fn projection_count(&self) -> usize {
        self.axes.len()
    }

    /// Number of public named selections, including aliases.
    pub fn selection_count(&self) -> usize {
        self.selections.len()
    }

    /// Number of unique canonical tag selections.
    pub const fn unique_selection_count(&self) -> usize {
        2
    }

    /// Evaluates one or four independent public differential calls.
    pub fn evaluate_single(
        &self,
        projections: usize,
    ) -> FixtureResult<Vec<DifferentialCrossSection>> {
        self.evaluate(&self.single, projections, &self.selections)
    }

    /// Evaluates one or four independent projections through one projection-set call.
    pub fn evaluate_single_set(&self, projections: usize) -> FixtureResult<ProjectionSet> {
        if !matches!(projections, 1 | 4) {
            return Err(format!("projection count must be 1 or 4, got {projections}").into());
        }
        let projections = self.axes[..projections]
            .iter()
            .enumerate()
            .map(|(index, axis)| Projection::new(format!("projection_{index}"), vec![axis.clone()]))
            .collect::<LikelihoodResult<Vec<_>>>()?;
        self.single
            .projection_set(&projections, &self.selections)
            .map_err(Into::into)
    }

    /// Evaluates one or four independent public differential calls on all periods.
    pub fn evaluate_combined(
        &self,
        projections: usize,
    ) -> FixtureResult<Vec<DifferentialCrossSection>> {
        self.evaluate(&self.combined, projections, &self.selections)
    }

    /// Evaluates with aliases removed, isolating canonical-selection scaling.
    pub fn evaluate_combined_unique(
        &self,
        projections: usize,
    ) -> FixtureResult<Vec<DifferentialCrossSection>> {
        let selections = HashMap::from([
            ("signal".to_owned(), vec!["signal".to_owned()]),
            ("background".to_owned(), vec!["background".to_owned()]),
        ]);
        self.evaluate(&self.combined, projections, &selections)
    }

    fn evaluate(
        &self,
        cross_section: &CrossSection,
        projections: usize,
        selections: &HashMap<String, Vec<String>>,
    ) -> FixtureResult<Vec<DifferentialCrossSection>> {
        if !matches!(projections, 1 | 4) {
            return Err(format!("projection count must be 1 or 4, got {projections}").into());
        }
        self.axes[..projections]
            .iter()
            .map(|axis| {
                cross_section
                    .differential(std::slice::from_ref(axis), selections)
                    .map_err(Into::into)
            })
            .collect()
    }
}

fn projection_model() -> FixtureResult<CompiledModel> {
    let cos_theta = event_scalar("cos_theta");
    let phi = event_scalar("phi");
    let cos_theta_decay = event_scalar("cos_theta_decay");
    let phi_decay = event_scalar("phi_decay");
    let signal = (complex(
        Expr::from(parameter!("signal_re", initial: 1.1)),
        Expr::from(parameter!("signal_im", initial: -0.2)),
    ) * complex(1.0 + cos_theta.clone(), phi.clone().cos()))
    .tagged("signal");
    let background = (complex(
        Expr::from(parameter!("background_re", initial: 0.45)),
        Expr::from(parameter!("background_im", initial: 0.15)),
    ) * complex(1.0 + cos_theta_decay, phi_decay.sin()))
    .tagged("background");
    Ok(CompiledModel::from_expr(&(signal + background).norm_sqr())?)
}

fn dataset(
    events: usize,
    period: usize,
    role: DatasetRole,
    storage: Storage,
) -> FixtureResult<Dataset> {
    let schema = Arc::new(Schema::new(std::iter::empty::<&str>(), SCALARS, true)?);
    let mut rng = fastrand::Rng::with_seed(
        FIXTURE_SEED ^ ((period as u64) << 16) ^ (role.seed_offset() << 8),
    );
    let rows = (0..events).map(|index| {
        let phase = (index as f64 + 0.5) / events.max(1) as f64;
        let jitter = rng.f64() - 0.5;
        OwnedEvent::weighted(
            vec![],
            vec![
                (2.0 * phase - 1.0 + 0.03 * jitter).clamp(-0.999_999, 0.999_999),
                -PI + 2.0 * PI * ((phase * (period + 1) as f64 + 0.01 * jitter).fract()),
                (2.0 * ((phase * 1.3 + period as f64 * 0.07 + role.seed_offset() as f64 * 0.03)
                    .fract())
                    - 1.0)
                    .clamp(-0.999_999, 0.999_999),
                -PI + 2.0 * PI * ((phase * 1.7 + period as f64 * 0.11 + 0.02 * jitter).fract()),
            ],
            0.75 + 0.5 * rng.f64(),
        )
    });
    Ok(storage.apply(Dataset::from_events(schema, rows)?))
}

fn deterministic_ensemble(likelihood: &Likelihood, draws: usize) -> FixtureResult<Ensemble> {
    let names = likelihood
        .params()
        .free_parameters()
        .map(|parameter| parameter.name().to_owned())
        .collect();
    let defaults = likelihood.default_params();
    let mut rng = fastrand::Rng::with_seed(FIXTURE_SEED ^ 0x44_52_41_57_53);
    let values = (0..draws)
        .map(|_| {
            defaults
                .iter()
                .map(|value| value + 0.08 * (rng.f64() - 0.5))
                .collect()
        })
        .collect();
    Ok(Ensemble::new(names, values)?)
}

fn projection_axes() -> FixtureResult<[Axis; 4]> {
    Ok([
        Axis::new(event_scalar("cos_theta"), uniform_edges(-1.0, 1.0, 40))?,
        Axis::new(event_scalar("phi"), uniform_edges(-PI, PI, 40))?,
        Axis::new(
            event_scalar("cos_theta_decay"),
            uniform_edges(-1.0, 1.0, 40),
        )?,
        Axis::new(event_scalar("phi_decay"), uniform_edges(-PI, PI, 40))?,
    ])
}

fn uniform_edges(low: f64, high: f64, bins: usize) -> Vec<f64> {
    (0..=bins)
        .map(|index| low + (high - low) * index as f64 / bins as f64)
        .collect()
}
