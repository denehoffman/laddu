use laddu_core::{
    amplitude::{
        display_key, Amplitude, AmplitudeID, AmplitudeSemanticKey, Expression, IntoTags, Tags,
    },
    data::{DatasetMetadata, Event},
    math::spherical_harmonic,
    resources::{Cache, ComplexScalarID, Parameters, Resources},
    traits::Variable,
    variables::Angles,
    LadduResult, Polarization, Sign,
};
use nalgebra::DVector;
use num::complex::Complex64;
use serde::{Deserialize, Serialize};

/// An [`Amplitude`] for the spherical harmonic function $`Y_\ell^m(\theta, \phi)`$.
#[derive(Clone, Serialize, Deserialize)]
pub struct Ylm {
    tags: Tags,
    l: usize,
    m: isize,
    angles: Angles,
    csid: ComplexScalarID,
}

impl Ylm {
    /// Construct a new [`Ylm`] with activation tags, angular momentum (`l`) and moment (`m`) over
    /// the given set of [`Angles`].
    pub fn new(
        tags: impl IntoTags,
        l: usize,
        m: isize,
        angles: &Angles,
    ) -> LadduResult<Expression> {
        Self {
            tags: tags.into_tags(),
            l,
            m,
            angles: angles.clone(),
            csid: ComplexScalarID::default(),
        }
        .into_expression()
    }
}

#[typetag::serde]
impl Amplitude for Ylm {
    fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID> {
        self.csid = resources.register_complex_scalar(None);
        resources.register_amplitude(self.tags.clone())
    }

    fn semantic_key(&self) -> Option<AmplitudeSemanticKey> {
        Some(
            AmplitudeSemanticKey::new("Ylm")
                .with_field("l", self.l.to_string())
                .with_field("m", self.m.to_string())
                .with_field("angles", display_key(&self.angles)),
        )
    }

    fn real_valued_hint(&self) -> bool {
        self.m == 0
    }

    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        self.angles.costheta.bind(metadata)?;
        self.angles.phi.bind(metadata)?;
        Ok(())
    }

    fn precompute(&self, event: &Event<'_>, cache: &mut Cache) {
        cache.store_complex_scalar(
            self.csid,
            spherical_harmonic(
                self.l,
                self.m,
                event.evaluate(&self.angles.costheta),
                event.evaluate(&self.angles.phi),
            ),
        );
    }

    fn compute(&self, _parameters: &Parameters, cache: &Cache) -> Complex64 {
        cache.get_complex_scalar(self.csid)
    }

    fn compute_gradient(
        &self,
        _parameters: &Parameters,
        _cache: &Cache,
        _gradient: &mut DVector<Complex64>,
    ) {
    }
}

/// An [`Amplitude`] representing an extension of the [`Ylm`] [`Amplitude`] for a linearly
/// polarized beam.
#[derive(Clone, Serialize, Deserialize)]
pub struct Zlm {
    tags: Tags,
    l: usize,
    m: isize,
    r: Sign,
    angles: Angles,
    polarization: Polarization,
    csid: ComplexScalarID,
}

impl Zlm {
    /// Construct a new [`Zlm`] with the given angular momentum, moment, reflectivity, angles, and
    /// polarization variables.
    pub fn new(
        tags: impl IntoTags,
        l: usize,
        m: isize,
        r: Sign,
        angles: &Angles,
        polarization: &Polarization,
    ) -> LadduResult<Expression> {
        Self {
            tags: tags.into_tags(),
            l,
            m,
            r,
            angles: angles.clone(),
            polarization: polarization.clone(),
            csid: ComplexScalarID::default(),
        }
        .into_expression()
    }
}

#[typetag::serde]
impl Amplitude for Zlm {
    fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID> {
        self.csid = resources.register_complex_scalar(None);
        resources.register_amplitude(self.tags.clone())
    }

    fn semantic_key(&self) -> Option<AmplitudeSemanticKey> {
        Some(
            AmplitudeSemanticKey::new("Zlm")
                .with_field("l", self.l.to_string())
                .with_field("m", self.m.to_string())
                .with_field("r", display_key(self.r))
                .with_field("angles", display_key(&self.angles))
                .with_field("polarization", display_key(&self.polarization)),
        )
    }

    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        self.angles.costheta.bind(metadata)?;
        self.angles.phi.bind(metadata)?;
        self.polarization.pol_angle.bind(metadata)?;
        self.polarization.pol_magnitude.bind(metadata)?;
        Ok(())
    }

    fn precompute(&self, event: &Event<'_>, cache: &mut Cache) {
        let ylm = spherical_harmonic(
            self.l,
            self.m,
            event.evaluate(&self.angles.costheta),
            event.evaluate(&self.angles.phi),
        );
        let pol_angle = event.evaluate(&self.polarization.pol_angle);
        let pgamma = event.evaluate(&self.polarization.pol_magnitude);
        let phase = Complex64::new(f64::cos(-pol_angle), f64::sin(-pol_angle));
        let zlm = ylm * phase;
        cache.store_complex_scalar(
            self.csid,
            match self.r {
                Sign::Positive => Complex64::new(
                    f64::sqrt(1.0 + pgamma) * zlm.re,
                    f64::sqrt(1.0 - pgamma) * zlm.im,
                ),
                Sign::Negative => Complex64::new(
                    f64::sqrt(1.0 - pgamma) * zlm.re,
                    f64::sqrt(1.0 + pgamma) * zlm.im,
                ),
            },
        );
    }

    fn compute(&self, _parameters: &Parameters, cache: &Cache) -> Complex64 {
        cache.get_complex_scalar(self.csid)
    }

    fn compute_gradient(
        &self,
        _parameters: &Parameters,
        _cache: &Cache,
        _gradient: &mut DVector<Complex64>,
    ) {
    }
}

/// An [`Amplitude`] representing the polarization phase $`P_\gamma e^{2 i \Phi}`$.
#[derive(Clone, Serialize, Deserialize)]
pub struct PolPhase {
    tags: Tags,
    polarization: Polarization,
    csid: ComplexScalarID,
}

impl PolPhase {
    /// Construct a new [`PolPhase`] from the given polarization variables.
    pub fn new(tags: impl IntoTags, polarization: &Polarization) -> LadduResult<Expression> {
        Self {
            tags: tags.into_tags(),
            polarization: polarization.clone(),
            csid: ComplexScalarID::default(),
        }
        .into_expression()
    }
}

#[typetag::serde]
impl Amplitude for PolPhase {
    fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID> {
        self.csid = resources.register_complex_scalar(None);
        resources.register_amplitude(self.tags.clone())
    }

    fn semantic_key(&self) -> Option<AmplitudeSemanticKey> {
        Some(
            AmplitudeSemanticKey::new("PolPhase")
                .with_field("polarization", display_key(&self.polarization)),
        )
    }

    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        self.polarization.pol_angle.bind(metadata)?;
        self.polarization.pol_magnitude.bind(metadata)?;
        Ok(())
    }

    fn precompute(&self, event: &Event<'_>, cache: &mut Cache) {
        let pol_angle = event.evaluate(&self.polarization.pol_angle);
        let pgamma = event.evaluate(&self.polarization.pol_magnitude);
        let phase = Complex64::new(f64::cos(2.0 * pol_angle), f64::sin(2.0 * pol_angle));
        cache.store_complex_scalar(self.csid, pgamma * phase);
    }

    fn compute(&self, _parameters: &Parameters, cache: &Cache) -> Complex64 {
        cache.get_complex_scalar(self.csid)
    }

    fn compute_gradient(
        &self,
        _parameters: &Parameters,
        _cache: &Cache,
        _gradient: &mut DVector<Complex64>,
    ) {
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use approx::assert_relative_eq;
    use laddu_core::{
        data::test_dataset,
        reaction::{Particle, Reaction},
        Frame,
    };

    use super::*;

    fn reaction_context() -> (Reaction, Angles) {
        let beam = Particle::stored("beam");
        let target = Particle::missing("target");
        let kshort1 = Particle::stored("kshort1");
        let kshort2 = Particle::stored("kshort2");
        let kk = Particle::composite("kk", (&kshort1, &kshort2)).unwrap();
        let proton = Particle::stored("proton");
        let reaction = Reaction::two_to_two(&beam, &target, &kk, &proton).unwrap();
        let angles = reaction
            .decay("kk")
            .unwrap()
            .angles("kshort1", Frame::Helicity)
            .unwrap();
        (reaction, angles)
    }

    #[test]
    fn test_ylm_evaluation() {
        let dataset = Arc::new(test_dataset());
        let (_, angles) = reaction_context();
        let expr = Ylm::new("ylm", 1, 1, &angles).unwrap();
        let evaluator = expr.load(&dataset).unwrap();
        let result = evaluator.evaluate(&[]).unwrap();
        assert_relative_eq!(result[0].re, 0.2713394403451035);
        assert_relative_eq!(result[0].im, 0.1426897184196572);
    }

    #[test]
    fn test_ylm_m_zero_reports_real_valued_hint() {
        let (_, angles) = reaction_context();
        let real_ylm = Ylm {
            tags: Tags::empty(),
            l: 1,
            m: 0,
            angles: angles.clone(),
            csid: ComplexScalarID::default(),
        };
        let complex_ylm = Ylm {
            tags: Tags::empty(),
            l: 1,
            m: 1,
            angles,
            csid: ComplexScalarID::default(),
        };
        assert!(Amplitude::real_valued_hint(&real_ylm));
        assert!(!Amplitude::real_valued_hint(&complex_ylm));
    }

    #[test]
    fn test_zlm_evaluation() {
        let dataset = Arc::new(test_dataset());
        let (reaction, angles) = reaction_context();
        let polarization = reaction.polarization("pol_magnitude", "pol_angle");
        let expr = Zlm::new("zlm", 1, 1, Sign::Positive, &angles, &polarization).unwrap();
        let evaluator = expr.load(&dataset).unwrap();
        let result = evaluator.evaluate(&[]).unwrap();
        assert_relative_eq!(result[0].re, 0.042841277808013944);
        assert_relative_eq!(result[0].im, -0.23859639139484332);
    }

    #[test]
    fn test_polphase_evaluation() {
        let dataset = Arc::new(test_dataset());
        let (reaction, _) = reaction_context();
        let polarization = reaction.polarization("pol_magnitude", "pol_angle");
        let expr = PolPhase::new("polphase", &polarization).unwrap();
        let evaluator = expr.load(&dataset).unwrap();
        let result = evaluator.evaluate(&[]).unwrap();
        assert_relative_eq!(result[0].re, -0.2872914473575512);
        assert_relative_eq!(result[0].im, -0.2572403880070272);
    }
}
