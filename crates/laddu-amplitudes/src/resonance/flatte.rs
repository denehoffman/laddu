use laddu_core::{
    amplitude::{
        display_key, parameter_key, Amplitude, AmplitudeID, AmplitudeSemanticKey, Expression,
        IntoTags, Parameter, Tags,
    },
    data::{DatasetMetadata, Event},
    math::{rho_m, Sheet},
    resources::{Cache, ParameterID, Parameters, Resources},
    traits::Variable,
    variables::Mass,
    LadduResult, ScalarID,
};
use nalgebra::DVector;
use num::complex::Complex64;
use serde::{Deserialize, Serialize};

/// A Flatte [`Amplitude`].
#[derive(Clone, Serialize, Deserialize)]
pub struct Flatte {
    tags: Tags,
    mass: Parameter,
    observed_channel_coupling: Parameter,
    alternate_channel_coupling: Parameter,
    pid_mass: ParameterID,
    pid_observed_channel_coupling: ParameterID,
    pid_alternate_channel_coupling: ParameterID,
    observed_channel_daughter_masses: (Mass, Mass),
    alternate_channel_daughter_masses: (f64, f64),
    resonance_mass: Mass,
    observed_channel_mass_ids: (ScalarID, ScalarID),
    resonance_mass_id: ScalarID,
}

impl Flatte {
    /// Construct a [`Flatte`] with the given mass, couplings, and channel daughter masses.
    pub fn new(
        tags: impl IntoTags,
        mass: Parameter,
        observed_channel_coupling: Parameter,
        alternate_channel_coupling: Parameter,
        observed_channel_daughter_masses: (&Mass, &Mass),
        alternate_channel_daughter_masses: (f64, f64),
        resonance_mass: &Mass,
    ) -> LadduResult<Expression> {
        Self {
            tags: tags.into_tags(),
            mass,
            observed_channel_coupling,
            alternate_channel_coupling,
            pid_mass: ParameterID::default(),
            pid_observed_channel_coupling: ParameterID::default(),
            pid_alternate_channel_coupling: ParameterID::default(),
            observed_channel_daughter_masses: (
                observed_channel_daughter_masses.0.clone(),
                observed_channel_daughter_masses.1.clone(),
            ),
            alternate_channel_daughter_masses,
            resonance_mass: resonance_mass.clone(),
            observed_channel_mass_ids: (ScalarID::default(), ScalarID::default()),
            resonance_mass_id: ScalarID::default(),
        }
        .into_expression()
    }
}

#[typetag::serde]
impl Amplitude for Flatte {
    fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID> {
        self.pid_mass = resources.register_parameter(&self.mass)?;
        self.pid_observed_channel_coupling =
            resources.register_parameter(&self.observed_channel_coupling)?;
        self.pid_alternate_channel_coupling =
            resources.register_parameter(&self.alternate_channel_coupling)?;
        self.observed_channel_mass_ids = (
            resources.register_scalar(None),
            resources.register_scalar(None),
        );
        self.resonance_mass_id = resources.register_scalar(None);
        resources.register_amplitude(self.tags.clone())
    }

    fn semantic_key(&self) -> Option<AmplitudeSemanticKey> {
        Some(
            AmplitudeSemanticKey::new("Flatte")
                .with_field("mass", parameter_key(&self.mass))
                .with_field(
                    "observed_channel_coupling",
                    parameter_key(&self.observed_channel_coupling),
                )
                .with_field(
                    "alternate_channel_coupling",
                    parameter_key(&self.alternate_channel_coupling),
                )
                .with_field(
                    "observed_channel_m1",
                    display_key(&self.observed_channel_daughter_masses.0),
                )
                .with_field(
                    "observed_channel_m2",
                    display_key(&self.observed_channel_daughter_masses.1),
                )
                .with_field(
                    "alternate_channel_m1",
                    self.alternate_channel_daughter_masses.0.to_string(),
                )
                .with_field(
                    "alternate_channel_m2",
                    self.alternate_channel_daughter_masses.1.to_string(),
                )
                .with_field("resonance_mass", display_key(&self.resonance_mass)),
        )
    }

    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        self.observed_channel_daughter_masses.0.bind(metadata)?;
        self.observed_channel_daughter_masses.1.bind(metadata)?;
        self.resonance_mass.bind(metadata)?;
        Ok(())
    }

    fn precompute(&self, event: &Event<'_>, cache: &mut Cache) {
        cache.store_scalar(
            self.observed_channel_mass_ids.0,
            event.evaluate(&self.observed_channel_daughter_masses.0),
        );
        cache.store_scalar(
            self.observed_channel_mass_ids.1,
            event.evaluate(&self.observed_channel_daughter_masses.1),
        );
        cache.store_scalar(self.resonance_mass_id, event.evaluate(&self.resonance_mass));
    }

    fn compute(&self, parameters: &Parameters, cache: &Cache) -> Complex64 {
        let mass = cache.get_scalar(self.resonance_mass_id);
        let mass0 = parameters.get(self.pid_mass).abs();
        let observed_channel_coupling = parameters.get(self.pid_observed_channel_coupling).abs();
        let alternate_channel_coupling = parameters.get(self.pid_alternate_channel_coupling).abs();
        let observed_m1 = cache.get_scalar(self.observed_channel_mass_ids.0);
        let observed_m2 = cache.get_scalar(self.observed_channel_mass_ids.1);
        let (alternate_m1, alternate_m2) = self.alternate_channel_daughter_masses;
        let observed_rho = rho_m(mass, observed_m1, observed_m2, Sheet::Physical);
        let alternate_rho = rho_m(mass, alternate_m1, alternate_m2, Sheet::Physical);

        1.0 / (Complex64::from(mass0.powi(2) - mass.powi(2))
            - Complex64::I
                * mass0
                * (observed_channel_coupling * observed_rho
                    + alternate_channel_coupling * alternate_rho))
    }

    fn compute_gradient(
        &self,
        parameters: &Parameters,
        cache: &Cache,
        gradient: &mut DVector<Complex64>,
    ) {
        if let Some(index) = parameters.free_index(self.pid_mass) {
            self.central_difference_with_indices(&[index], parameters, cache, gradient);
        }
        if let Some(index) = parameters.free_index(self.pid_observed_channel_coupling) {
            self.central_difference_with_indices(&[index], parameters, cache, gradient);
        }
        if let Some(index) = parameters.free_index(self.pid_alternate_channel_coupling) {
            self.central_difference_with_indices(&[index], parameters, cache, gradient);
        }
    }
}
