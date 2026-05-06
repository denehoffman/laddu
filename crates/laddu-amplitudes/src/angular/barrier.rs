use laddu_core::{
    amplitudes::{
        debug_key, f64_key, Amplitude, AmplitudeID, AmplitudeSemanticKey, Expression, IntoTags,
        Tags,
    },
    data::{DatasetMetadata, Event},
    math::{blatt_weisskopf_m, BarrierKind, Sheet},
    resources::{Cache, ComplexScalarID, Parameters, Resources, ScalarID},
    traits::Variable,
    Decay, LadduResult, OrbitalAngularMomentum,
};
use nalgebra::DVector;
use num::complex::Complex64;
use serde::{Deserialize, Serialize};

/// A Blatt-Weisskopf barrier-factor amplitude for a two-body decay.
#[derive(Clone, Serialize, Deserialize)]
pub struct BlattWeisskopf {
    tags: Tags,
    decay_key: String,
    parent_mass: Box<dyn Variable>,
    daughter_1_mass: Box<dyn Variable>,
    daughter_2_mass: Box<dyn Variable>,
    l: OrbitalAngularMomentum,
    reference_mass: f64,
    q_r: f64,
    sheet: Sheet,
    kind: BarrierKind,
    parent_mass_id: ScalarID,
    daughter_1_mass_id: ScalarID,
    daughter_2_mass_id: ScalarID,
    value_id: ComplexScalarID,
}

impl BlattWeisskopf {
    /// Construct a new Blatt-Weisskopf barrier-factor amplitude.
    pub fn new(
        tags: impl IntoTags,
        decay: &Decay,
        l: OrbitalAngularMomentum,
        reference_mass: f64,
        q_r: f64,
        sheet: Sheet,
        kind: BarrierKind,
    ) -> LadduResult<Expression> {
        Self {
            tags: tags.into_tags(),
            decay_key: format!(
                "{} -> {} {}",
                decay.parent(),
                decay.daughter_1(),
                decay.daughter_2()
            ),
            parent_mass: Box::new(decay.parent_mass()),
            daughter_1_mass: Box::new(decay.daughter_1_mass()),
            daughter_2_mass: Box::new(decay.daughter_2_mass()),
            l,
            reference_mass,
            q_r,
            sheet,
            kind,
            parent_mass_id: ScalarID::default(),
            daughter_1_mass_id: ScalarID::default(),
            daughter_2_mass_id: ScalarID::default(),
            value_id: ComplexScalarID::default(),
        }
        .into_expression()
    }
}

#[typetag::serde]
impl Amplitude for BlattWeisskopf {
    fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID> {
        self.parent_mass_id = resources.register_scalar(None);
        self.daughter_1_mass_id = resources.register_scalar(None);
        self.daughter_2_mass_id = resources.register_scalar(None);
        self.value_id = resources.register_complex_scalar(None);
        resources.register_amplitude(self.tags.clone())
    }

    fn semantic_key(&self) -> Option<AmplitudeSemanticKey> {
        Some(
            AmplitudeSemanticKey::new("BlattWeisskopf")
                .with_field("decay", debug_key(&self.decay_key))
                .with_field("l", self.l.value().to_string())
                .with_field("reference_mass", f64_key(self.reference_mass))
                .with_field("q_r", f64_key(self.q_r))
                .with_field("sheet", debug_key(self.sheet))
                .with_field("kind", debug_key(self.kind)),
        )
    }

    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        self.parent_mass.bind(metadata)?;
        self.daughter_1_mass.bind(metadata)?;
        self.daughter_2_mass.bind(metadata)
    }

    fn precompute(&self, event: &Event<'_>, cache: &mut Cache) {
        let parent_mass = self.parent_mass.value(event);
        let daughter_1_mass = self.daughter_1_mass.value(event);
        let daughter_2_mass = self.daughter_2_mass.value(event);
        cache.store_scalar(self.parent_mass_id, parent_mass);
        cache.store_scalar(self.daughter_1_mass_id, daughter_1_mass);
        cache.store_scalar(self.daughter_2_mass_id, daughter_2_mass);
        let barrier = blatt_weisskopf_m(
            parent_mass,
            daughter_1_mass,
            daughter_2_mass,
            self.l.value() as usize,
            self.q_r,
            self.sheet,
            self.kind,
        );
        let reference_barrier = blatt_weisskopf_m(
            self.reference_mass,
            daughter_1_mass,
            daughter_2_mass,
            self.l.value() as usize,
            self.q_r,
            self.sheet,
            self.kind,
        );
        cache.store_complex_scalar(self.value_id, barrier / reference_barrier);
    }

    fn compute(&self, _parameters: &Parameters, cache: &Cache) -> Complex64 {
        cache.get_complex_scalar(self.value_id)
    }

    fn compute_gradient(
        &self,
        _parameters: &Parameters,
        _cache: &Cache,
        _gradient: &mut DVector<Complex64>,
    ) {
    }
}
