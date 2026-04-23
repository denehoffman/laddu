use nalgebra::DVector;
use num::complex::Complex64;
use serde::{Deserialize, Serialize};

use super::{Particle, Reaction};
use crate::{
    amplitudes::{Amplitude, AmplitudeID, AmplitudeSemanticKey, Expression},
    data::NamedEventView,
    math::{clebsch_gordon, WignerDMatrix},
    quantum::{
        AngularMomentum, AngularMomentumProjection, Frame, OrbitalAngularMomentum, SpinState,
    },
    resources::{Cache, ComplexScalarID, Parameters, Resources},
    traits::Variable,
    variables::{Angles, CosTheta, Mass, Phi},
    LadduError, LadduResult,
};

/// An isobar decay view used to derive variables and decay-local amplitudes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Decay {
    pub(super) reaction: Reaction,
    pub(super) parent: Particle,
    pub(super) daughter_1: Particle,
    pub(super) daughter_2: Particle,
}

impl Decay {
    /// Return the enclosing reaction.
    pub const fn reaction(&self) -> &Reaction {
        &self.reaction
    }

    /// Return the parent particle.
    pub const fn parent(&self) -> &Particle {
        &self.parent
    }

    /// Return the first daughter particle.
    pub const fn daughter_1(&self) -> &Particle {
        &self.daughter_1
    }

    /// Return the second daughter particle.
    pub const fn daughter_2(&self) -> &Particle {
        &self.daughter_2
    }

    /// Return the ordered daughter particles.
    pub fn daughters(&self) -> [&Particle; 2] {
        [&self.daughter_1, &self.daughter_2]
    }

    /// Return the parent mass variable.
    pub fn mass(&self) -> Mass {
        self.reaction.mass(&self.parent)
    }

    /// Return the parent mass variable.
    pub fn parent_mass(&self) -> Mass {
        self.mass()
    }

    /// Return the first daughter mass variable.
    pub fn daughter_1_mass(&self) -> Mass {
        self.reaction.mass(&self.daughter_1)
    }

    /// Return the second daughter mass variable.
    pub fn daughter_2_mass(&self) -> Mass {
        self.reaction.mass(&self.daughter_2)
    }

    /// Return the mass variable for one of the ordered daughters.
    pub fn daughter_mass(&self, daughter: &Particle) -> LadduResult<Mass> {
        self.validate_daughter(daughter)?;
        Ok(self.reaction.mass(daughter))
    }

    /// Return the decay costheta variable.
    pub fn costheta(&self, daughter: &Particle, frame: Frame) -> LadduResult<CosTheta> {
        self.validate_daughter(daughter)?;
        Ok(CosTheta::from_reaction(
            self.reaction.clone(),
            self.parent.clone(),
            daughter.clone(),
            frame,
        ))
    }

    /// Return the decay phi variable.
    pub fn phi(&self, daughter: &Particle, frame: Frame) -> LadduResult<Phi> {
        self.validate_daughter(daughter)?;
        Ok(Phi::from_reaction(
            self.reaction.clone(),
            self.parent.clone(),
            daughter.clone(),
            frame,
        ))
    }

    /// Return both decay angle variables.
    pub fn angles(&self, daughter: &Particle, frame: Frame) -> LadduResult<Angles> {
        self.validate_daughter(daughter)?;
        Ok(Angles::from_reaction(
            self.reaction.clone(),
            self.parent.clone(),
            daughter.clone(),
            frame,
        ))
    }

    /// Construct the helicity-basis angular factor for one explicit helicity term.
    ///
    /// The returned expression is `$D^{J*}_{M,\lambda}(\phi,\theta,0)$`, where
    /// `$\lambda = \lambda_1 - \lambda_2$`.
    #[allow(clippy::too_many_arguments)]
    pub fn helicity_factor(
        &self,
        name: &str,
        spin: AngularMomentum,
        projection: AngularMomentumProjection,
        daughter: &Particle,
        lambda_1: AngularMomentumProjection,
        lambda_2: AngularMomentumProjection,
        frame: Frame,
    ) -> LadduResult<Expression> {
        let lambda = AngularMomentumProjection::from_twice(lambda_1.value() - lambda_2.value());
        let angles = self.angles(daughter, frame)?;
        Ok(DecayWignerD::expression(name, spin, projection, lambda, &angles)?.conj())
    }

    /// Construct the canonical-basis spin-angular factor for one explicit LS/helicity term.
    ///
    /// The returned expression is
    /// `$\sqrt{2L+1}\langle L0;S\lambda|J\lambda\rangle
    /// \langle J_1\lambda_1;J_2-\lambda_2|S\lambda\rangle
    /// D^{J*}_{M,\lambda}(\phi,\theta,0)$`, where
    /// `$\lambda = \lambda_1 - \lambda_2$`.
    #[allow(clippy::too_many_arguments)]
    pub fn canonical_factor(
        &self,
        name: &str,
        spin: AngularMomentum,
        projection: AngularMomentumProjection,
        orbital_l: OrbitalAngularMomentum,
        coupled_spin: AngularMomentum,
        daughter: &Particle,
        daughter_1_spin: AngularMomentum,
        daughter_2_spin: AngularMomentum,
        lambda_1: AngularMomentumProjection,
        lambda_2: AngularMomentumProjection,
        frame: Frame,
    ) -> LadduResult<Expression> {
        let lambda = AngularMomentumProjection::from_twice(lambda_1.value() - lambda_2.value());
        let minus_lambda_2 = AngularMomentumProjection::from_twice(-lambda_2.value());
        let norm = FixedScalar::expression(
            &format!("{name}.norm"),
            f64::from(2 * orbital_l.value() + 1).sqrt(),
        )?;
        let orbital_spin_cg = ClebschGordanFactor::expression(
            &format!("{name}.orbital_spin_cg"),
            orbital_l.angular_momentum(),
            AngularMomentumProjection::integer(0),
            coupled_spin,
            lambda,
            spin,
            lambda,
        )?;
        let daughter_spin_cg = ClebschGordanFactor::expression(
            &format!("{name}.daughter_spin_cg"),
            daughter_1_spin,
            lambda_1,
            daughter_2_spin,
            minus_lambda_2,
            coupled_spin,
            lambda,
        )?;
        Ok(norm
            * orbital_spin_cg
            * daughter_spin_cg
            * self.helicity_factor(
                &format!("{name}.wigner_d"),
                spin,
                projection,
                daughter,
                lambda_1,
                lambda_2,
                frame,
            )?)
    }

    fn validate_daughter(&self, daughter: &Particle) -> LadduResult<()> {
        if daughter == &self.daughter_1 || daughter == &self.daughter_2 {
            Ok(())
        } else {
            Err(LadduError::Custom(format!(
                "particle '{}' is not an immediate daughter of '{}'",
                daughter.label(),
                self.parent.label()
            )))
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
struct FixedScalar {
    name: String,
    value: f64,
}

impl FixedScalar {
    fn expression(name: &str, value: f64) -> LadduResult<Expression> {
        Self {
            name: name.to_string(),
            value,
        }
        .into_expression()
    }
}

#[typetag::serde]
impl Amplitude for FixedScalar {
    fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID> {
        resources.register_amplitude(&self.name)
    }

    fn semantic_key(&self) -> Option<AmplitudeSemanticKey> {
        Some(
            AmplitudeSemanticKey::new("FixedScalar")
                .with_field("name", format!("{:?}", self.name))
                .with_field("value", f64_key(self.value)),
        )
    }

    fn real_valued_hint(&self) -> bool {
        true
    }

    fn compute(&self, _parameters: &Parameters, _cache: &Cache) -> Complex64 {
        Complex64::new(self.value, 0.0)
    }

    fn compute_gradient(
        &self,
        _parameters: &Parameters,
        _cache: &Cache,
        _gradient: &mut DVector<Complex64>,
    ) {
    }
}

#[derive(Clone, Serialize, Deserialize)]
struct ClebschGordanFactor;

impl ClebschGordanFactor {
    fn expression(
        name: &str,
        j1: AngularMomentum,
        m1: AngularMomentumProjection,
        j2: AngularMomentum,
        m2: AngularMomentumProjection,
        j: AngularMomentum,
        m: AngularMomentumProjection,
    ) -> LadduResult<Expression> {
        let value = clebsch_gordon(
            j1.value() as u64,
            j2.value() as u64,
            j.value() as u64,
            m1.value() as i64,
            m2.value() as i64,
            m.value() as i64,
        );
        FixedScalar::expression(name, value)
    }
}

#[derive(Clone, Serialize, Deserialize)]
struct DecayWignerD {
    name: String,
    spin: AngularMomentum,
    row_projection: AngularMomentumProjection,
    column_projection: AngularMomentumProjection,
    costheta: Box<dyn Variable>,
    phi: Box<dyn Variable>,
    angles_key: String,
    value_id: ComplexScalarID,
}

impl DecayWignerD {
    fn expression(
        name: &str,
        spin: AngularMomentum,
        row_projection: AngularMomentumProjection,
        column_projection: AngularMomentumProjection,
        angles: &Angles,
    ) -> LadduResult<Expression> {
        SpinState::new(spin, row_projection)?;
        SpinState::new(spin, column_projection)?;
        Self {
            name: name.to_string(),
            spin,
            row_projection,
            column_projection,
            costheta: angles.costheta_variable(),
            phi: angles.phi_variable(),
            angles_key: angles.to_string(),
            value_id: ComplexScalarID::default(),
        }
        .into_expression()
    }
}

#[typetag::serde]
impl Amplitude for DecayWignerD {
    fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID> {
        self.value_id = resources.register_complex_scalar(None);
        resources.register_amplitude(&self.name)
    }

    fn semantic_key(&self) -> Option<AmplitudeSemanticKey> {
        Some(
            AmplitudeSemanticKey::new("DecayWignerD")
                .with_field("name", format!("{:?}", self.name))
                .with_field("spin", self.spin.value().to_string())
                .with_field("row_projection", self.row_projection.value().to_string())
                .with_field(
                    "column_projection",
                    self.column_projection.value().to_string(),
                )
                .with_field("angles", format!("{:?}", self.angles_key)),
        )
    }

    fn bind(&mut self, metadata: &crate::data::DatasetMetadata) -> LadduResult<()> {
        self.costheta.bind(metadata)?;
        self.phi.bind(metadata)
    }

    fn precompute(&self, event: &NamedEventView<'_>, cache: &mut Cache) {
        let costheta = self.costheta.value(event).clamp(-1.0, 1.0);
        let theta = costheta.acos();
        let phi = self.phi.value(event);
        let d_matrix = WignerDMatrix::new(
            self.spin.value() as u64,
            self.row_projection.value() as i64,
            self.column_projection.value() as i64,
        );
        cache.store_complex_scalar(self.value_id, d_matrix.D(phi, theta, 0.0));
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

fn f64_key(value: f64) -> String {
    if value == 0.0 {
        "0".to_string()
    } else {
        format!("{:016x}", value.to_bits())
    }
}
