use laddu_core::{
    amplitudes::{
        display_key, parameter_key, Amplitude, AmplitudeID, AmplitudeSemanticKey, Expression,
        IntoTags, Parameter, Tags,
    },
    data::{DatasetMetadata, Event},
    math::{blatt_weisskopf_m, q_m, BarrierKind, Sheet, QR_DEFAULT},
    resources::{Cache, ParameterID, Parameters, Resources},
    traits::Variable,
    variables::Mass,
    LadduResult, ScalarID,
};
use nalgebra::DVector;
use num::complex::Complex64;
use serde::{Deserialize, Serialize};

/// A relativistic Breit-Wigner [`Amplitude`].
#[derive(Clone, Serialize, Deserialize)]
pub struct BreitWigner {
    tags: Tags,
    mass: Parameter,
    width: Parameter,
    pid_mass: ParameterID,
    pid_width: ParameterID,
    l: usize,
    daughter_1_mass: Mass,
    daughter_2_mass: Mass,
    resonance_mass: Mass,
    barrier_factors: bool,
    daughter_1_mass_id: ScalarID,
    daughter_2_mass_id: ScalarID,
    resonance_mass_id: ScalarID,
}

impl BreitWigner {
    /// Construct a [`BreitWigner`] with barrier factors.
    pub fn new(
        tags: impl IntoTags,
        mass: Parameter,
        width: Parameter,
        l: usize,
        daughter_1_mass: &Mass,
        daughter_2_mass: &Mass,
        resonance_mass: &Mass,
    ) -> LadduResult<Expression> {
        Self {
            tags: tags.into_tags(),
            mass,
            width,
            pid_mass: ParameterID::default(),
            pid_width: ParameterID::default(),
            l,
            daughter_1_mass: daughter_1_mass.clone(),
            daughter_2_mass: daughter_2_mass.clone(),
            resonance_mass: resonance_mass.clone(),
            barrier_factors: true,
            daughter_1_mass_id: ScalarID::default(),
            daughter_2_mass_id: ScalarID::default(),
            resonance_mass_id: ScalarID::default(),
        }
        .into_expression()
    }

    /// Construct a [`BreitWigner`] without barrier factors.
    pub fn new_without_barrier_factors(
        tags: impl IntoTags,
        mass: Parameter,
        width: Parameter,
        l: usize,
        daughter_1_mass: &Mass,
        daughter_2_mass: &Mass,
        resonance_mass: &Mass,
    ) -> LadduResult<Expression> {
        Self {
            tags: tags.into_tags(),
            mass,
            width,
            pid_mass: ParameterID::default(),
            pid_width: ParameterID::default(),
            l,
            daughter_1_mass: daughter_1_mass.clone(),
            daughter_2_mass: daughter_2_mass.clone(),
            resonance_mass: resonance_mass.clone(),
            barrier_factors: false,
            daughter_1_mass_id: ScalarID::default(),
            daughter_2_mass_id: ScalarID::default(),
            resonance_mass_id: ScalarID::default(),
        }
        .into_expression()
    }
}

#[typetag::serde]
impl Amplitude for BreitWigner {
    fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID> {
        self.pid_mass = resources.register_parameter(&self.mass)?;
        self.pid_width = resources.register_parameter(&self.width)?;
        self.daughter_1_mass_id = resources.register_scalar(None);
        self.daughter_2_mass_id = resources.register_scalar(None);
        self.resonance_mass_id = resources.register_scalar(None);
        resources.register_amplitude(self.tags.clone())
    }

    fn semantic_key(&self) -> Option<AmplitudeSemanticKey> {
        Some(
            AmplitudeSemanticKey::new("BreitWigner")
                .with_field("mass", parameter_key(&self.mass))
                .with_field("width", parameter_key(&self.width))
                .with_field("l", self.l.to_string())
                .with_field("daughter_1_mass", display_key(&self.daughter_1_mass))
                .with_field("daughter_2_mass", display_key(&self.daughter_2_mass))
                .with_field("resonance_mass", display_key(&self.resonance_mass))
                .with_field("barrier_factors", self.barrier_factors.to_string()),
        )
    }

    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        self.daughter_1_mass.bind(metadata)?;
        self.daughter_2_mass.bind(metadata)?;
        self.resonance_mass.bind(metadata)?;
        Ok(())
    }

    fn precompute(&self, event: &Event<'_>, cache: &mut Cache) {
        cache.store_scalar(
            self.daughter_1_mass_id,
            event.evaluate(&self.daughter_1_mass),
        );
        cache.store_scalar(
            self.daughter_2_mass_id,
            event.evaluate(&self.daughter_2_mass),
        );
        cache.store_scalar(self.resonance_mass_id, event.evaluate(&self.resonance_mass));
    }

    fn compute(&self, parameters: &Parameters, cache: &Cache) -> Complex64 {
        let mass = cache.get_scalar(self.resonance_mass_id);
        let mass0 = parameters.get(self.pid_mass).abs();
        let width0 = parameters.get(self.pid_width).abs();
        let mass1 = cache.get_scalar(self.daughter_1_mass_id);
        let mass2 = cache.get_scalar(self.daughter_2_mass_id);
        let q0 = q_m(mass0, mass1, mass2, Sheet::Physical);
        let q = q_m(mass, mass1, mass2, Sheet::Physical);
        let width = if self.barrier_factors {
            let f0 = blatt_weisskopf_m(
                mass0,
                mass1,
                mass2,
                self.l,
                QR_DEFAULT,
                Sheet::Physical,
                BarrierKind::Full,
            );
            let f = blatt_weisskopf_m(
                mass,
                mass1,
                mass2,
                self.l,
                QR_DEFAULT,
                Sheet::Physical,
                BarrierKind::Full,
            );
            width0 * (mass0 / mass) * (q / q0) * (f / f0).powi(2)
        } else {
            width0 * (mass0 / mass) * (q / q0).powi((2 * self.l + 1) as i32)
        };
        1.0 / (Complex64::from(mass0.powi(2) - mass.powi(2)) - Complex64::I * mass0 * width)
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
        if let Some(index) = parameters.free_index(self.pid_width) {
            self.central_difference_with_indices(&[index], parameters, cache, gradient);
        }
    }
}

/// A non-relativistic Breit-Wigner [`Amplitude`].
#[derive(Clone, Serialize, Deserialize)]
pub struct BreitWignerNonRelativistic {
    tags: Tags,
    mass: Parameter,
    width: Parameter,
    pid_mass: ParameterID,
    pid_width: ParameterID,
    resonance_mass: Mass,
    resonance_mass_id: ScalarID,
}

impl BreitWignerNonRelativistic {
    /// Construct a [`BreitWignerNonRelativistic`] with activation tags, mass, and width.
    pub fn new(
        tags: impl IntoTags,
        mass: Parameter,
        width: Parameter,
        resonance_mass: &Mass,
    ) -> LadduResult<Expression> {
        Self {
            tags: tags.into_tags(),
            mass,
            width,
            pid_mass: ParameterID::default(),
            pid_width: ParameterID::default(),
            resonance_mass: resonance_mass.clone(),
            resonance_mass_id: ScalarID::default(),
        }
        .into_expression()
    }
}

#[typetag::serde]
impl Amplitude for BreitWignerNonRelativistic {
    fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID> {
        self.pid_mass = resources.register_parameter(&self.mass)?;
        self.pid_width = resources.register_parameter(&self.width)?;
        self.resonance_mass_id = resources.register_scalar(None);
        resources.register_amplitude(self.tags.clone())
    }

    fn semantic_key(&self) -> Option<AmplitudeSemanticKey> {
        Some(
            AmplitudeSemanticKey::new("BreitWignerNonRelativistic")
                .with_field("mass", parameter_key(&self.mass))
                .with_field("width", parameter_key(&self.width))
                .with_field("resonance_mass", display_key(&self.resonance_mass)),
        )
    }

    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        self.resonance_mass.bind(metadata)?;
        Ok(())
    }

    fn precompute(&self, event: &Event<'_>, cache: &mut Cache) {
        cache.store_scalar(self.resonance_mass_id, event.evaluate(&self.resonance_mass));
    }

    fn compute(&self, parameters: &Parameters, cache: &Cache) -> Complex64 {
        let mass = cache.get_scalar(self.resonance_mass_id);
        let mass0 = parameters.get(self.pid_mass).abs();
        let width0 = parameters.get(self.pid_width).abs();
        1.0 / Complex64::new(mass0.powi(2) - mass.powi(2), -(mass0 * width0))
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
        if let Some(index) = parameters.free_index(self.pid_width) {
            self.central_difference_with_indices(&[index], parameters, cache, gradient);
        }
    }
}
