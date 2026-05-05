use laddu_core::{
    amplitudes::{
        debug_key, display_key, Amplitude, AmplitudeID, AmplitudeSemanticKey, Expression,
    },
    data::{DatasetMetadata, Event},
    math::{rho_m, Sheet},
    resources::{Cache, Parameters, Resources},
    traits::Variable,
    variables::{Mandelstam, Mass},
    LadduResult, ScalarID, PI,
};
use nalgebra::DVector;
use num::complex::Complex64;
use serde::{Deserialize, Serialize};

/// A phase-space [`Amplitude`] for `$a+b\\to c+d$` with `$c\\to 1+2$`.
#[derive(Clone, Serialize, Deserialize)]
pub struct PhaseSpaceFactor {
    name: String,
    recoil_mass: Mass,
    daughter_1_mass: Mass,
    daughter_2_mass: Mass,
    resonance_mass: Mass,
    mandelstam_s: Mandelstam,
    sid: ScalarID,
}

impl PhaseSpaceFactor {
    /// Construct a new [`PhaseSpaceFactor`] that models the two-body phase-space density.
    pub fn new(
        name: &str,
        recoil_mass: &Mass,
        daughter_1_mass: &Mass,
        daughter_2_mass: &Mass,
        resonance_mass: &Mass,
        mandelstam_s: &Mandelstam,
    ) -> LadduResult<Expression> {
        Self {
            name: name.to_string(),
            recoil_mass: recoil_mass.clone(),
            daughter_1_mass: daughter_1_mass.clone(),
            daughter_2_mass: daughter_2_mass.clone(),
            resonance_mass: resonance_mass.clone(),
            mandelstam_s: mandelstam_s.clone(),
            sid: ScalarID::default(),
        }
        .into_expression()
    }
}

#[typetag::serde]
impl Amplitude for PhaseSpaceFactor {
    fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID> {
        self.sid = resources.register_scalar(None);
        resources.register_amplitude(&self.name)
    }

    fn semantic_key(&self) -> Option<AmplitudeSemanticKey> {
        Some(
            AmplitudeSemanticKey::new("PhaseSpaceFactor")
                .with_field("name", debug_key(&self.name))
                .with_field("recoil_mass", display_key(&self.recoil_mass))
                .with_field("daughter_1_mass", display_key(&self.daughter_1_mass))
                .with_field("daughter_2_mass", display_key(&self.daughter_2_mass))
                .with_field("resonance_mass", display_key(&self.resonance_mass))
                .with_field("mandelstam_s", display_key(&self.mandelstam_s)),
        )
    }

    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        self.recoil_mass.bind(metadata)?;
        self.daughter_1_mass.bind(metadata)?;
        self.daughter_2_mass.bind(metadata)?;
        self.resonance_mass.bind(metadata)?;
        self.mandelstam_s.bind(metadata)?;
        Ok(())
    }

    fn precompute(&self, event: &Event<'_>, cache: &mut Cache) {
        let m_recoil = event.evaluate(&self.recoil_mass);
        let m_1 = event.evaluate(&self.daughter_1_mass);
        let m_2 = event.evaluate(&self.daughter_2_mass);
        let m_res = event.evaluate(&self.resonance_mass);
        let s = event.evaluate(&self.mandelstam_s);
        let term = rho_m(m_res, m_1, m_2, Sheet::Physical).re * m_res
            / (s - m_recoil.powi(2)).powi(2)
            / (2.0 * (4.0 * PI).powi(5));
        cache.store_scalar(self.sid, term.sqrt());
    }

    fn compute(&self, _parameters: &Parameters, cache: &Cache) -> Complex64 {
        cache.get_scalar(self.sid).into()
    }

    fn compute_gradient(
        &self,
        _parameters: &Parameters,
        _cache: &Cache,
        _gradient: &mut DVector<Complex64>,
    ) {
    }
}
