use laddu_core::{
    amplitudes::{debug_key, Amplitude, AmplitudeID, AmplitudeSemanticKey, Expression},
    data::{DatasetMetadata, NamedEventView},
    math::WignerDMatrix,
    resources::{Cache, ComplexScalarID, Parameters, Resources},
    traits::Variable,
    variables::Angles,
    AngularMomentum, AngularMomentumProjection, Decay, Frame, LadduResult, OrbitalAngularMomentum,
    SpinState,
};
use nalgebra::DVector;
use num::complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::angular::ClebschGordan;

/// An amplitude evaluating a Wigner-D matrix element from decay angles.
#[derive(Clone, Serialize, Deserialize)]
pub struct WignerD {
    name: String,
    spin: AngularMomentum,
    row_projection: AngularMomentumProjection,
    column_projection: AngularMomentumProjection,
    costheta: Box<dyn Variable>,
    phi: Box<dyn Variable>,
    angles_key: String,
    value_id: ComplexScalarID,
}

impl WignerD {
    /// Construct a new Wigner-D amplitude.
    ///
    /// The returned expression evaluates
    /// `D^j_{m' m}(phi, theta, 0)`, with `theta = acos(costheta)`.
    pub fn new(
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

/// Extension methods that build decay-local angular expressions from [`Decay`] objects.
pub trait DecayAmplitudeExt {
    /// Construct the helicity-basis angular factor for one explicit helicity term.
    #[allow(clippy::too_many_arguments)]
    fn helicity_factor(
        &self,
        name: &str,
        spin: AngularMomentum,
        projection: AngularMomentumProjection,
        daughter: &str,
        lambda_1: AngularMomentumProjection,
        lambda_2: AngularMomentumProjection,
        frame: Frame,
    ) -> LadduResult<Expression>;

    /// Construct the canonical-basis spin-angular factor for one explicit LS/helicity term.
    #[allow(clippy::too_many_arguments)]
    fn canonical_factor(
        &self,
        name: &str,
        spin: AngularMomentum,
        projection: AngularMomentumProjection,
        orbital_l: OrbitalAngularMomentum,
        coupled_spin: AngularMomentum,
        daughter: &str,
        daughter_1_spin: AngularMomentum,
        daughter_2_spin: AngularMomentum,
        lambda_1: AngularMomentumProjection,
        lambda_2: AngularMomentumProjection,
        frame: Frame,
    ) -> LadduResult<Expression>;
}

impl DecayAmplitudeExt for Decay {
    fn helicity_factor(
        &self,
        name: &str,
        spin: AngularMomentum,
        projection: AngularMomentumProjection,
        daughter: &str,
        lambda_1: AngularMomentumProjection,
        lambda_2: AngularMomentumProjection,
        frame: Frame,
    ) -> LadduResult<Expression> {
        let lambda = AngularMomentumProjection::from_twice(lambda_1.value() - lambda_2.value());
        let angles = self.angles(daughter, frame)?;
        Ok(WignerD::new(name, spin, projection, lambda, &angles)?.conj())
    }

    fn canonical_factor(
        &self,
        name: &str,
        spin: AngularMomentum,
        projection: AngularMomentumProjection,
        orbital_l: OrbitalAngularMomentum,
        coupled_spin: AngularMomentum,
        daughter: &str,
        daughter_1_spin: AngularMomentum,
        daughter_2_spin: AngularMomentum,
        lambda_1: AngularMomentumProjection,
        lambda_2: AngularMomentumProjection,
        frame: Frame,
    ) -> LadduResult<Expression> {
        let lambda = AngularMomentumProjection::from_twice(lambda_1.value() - lambda_2.value());
        let minus_lambda_2 = AngularMomentumProjection::from_twice(-lambda_2.value());
        Ok(
            Expression::from(f64::from(2 * orbital_l.value() + 1).sqrt())
                * ClebschGordan::new(
                    &format!("{name}.orbital_spin_cg"),
                    orbital_l.angular_momentum(),
                    AngularMomentumProjection::integer(0),
                    coupled_spin,
                    lambda,
                    spin,
                    lambda,
                )?
                * ClebschGordan::new(
                    &format!("{name}.daughter_spin_cg"),
                    daughter_1_spin,
                    lambda_1,
                    daughter_2_spin,
                    minus_lambda_2,
                    coupled_spin,
                    lambda,
                )?
                * self.helicity_factor(
                    &format!("{name}.wigner_d"),
                    spin,
                    projection,
                    daughter,
                    lambda_1,
                    lambda_2,
                    frame,
                )?,
        )
    }
}

#[typetag::serde]
impl Amplitude for WignerD {
    fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID> {
        self.value_id = resources.register_complex_scalar(None);
        resources.register_amplitude(&self.name)
    }

    fn semantic_key(&self) -> Option<AmplitudeSemanticKey> {
        Some(
            AmplitudeSemanticKey::new("WignerD")
                .with_field("name", debug_key(&self.name))
                .with_field("spin", self.spin.value().to_string())
                .with_field("row_projection", self.row_projection.value().to_string())
                .with_field(
                    "column_projection",
                    self.column_projection.value().to_string(),
                )
                .with_field("angles", debug_key(&self.angles_key)),
        )
    }

    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
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
