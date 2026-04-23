#[cfg(any(test, feature = "python"))]
use laddu_core::math::QR_DEFAULT;
use laddu_core::{
    amplitudes::{Amplitude, AmplitudeID, AmplitudeSemanticKey, Expression},
    data::{DatasetMetadata, NamedEventView},
    math::{blatt_weisskopf_m, clebsch_gordon, wigner_3j, BarrierKind, Sheet, WignerDMatrix},
    parameter,
    resources::{Cache, ComplexScalarID, Parameters, Resources, ScalarID},
    traits::Variable,
    variables::Angles,
    AngularMomentum, AngularMomentumProjection, Decay, LadduError, LadduResult,
    OrbitalAngularMomentum, Polarization, SpinState,
};
#[cfg(feature = "python")]
use laddu_python::{
    amplitudes::PyExpression,
    utils::variables::{PyAngles, PyDecay, PyPolarization},
};
use nalgebra::DVector;
use num::complex::Complex64;
#[cfg(feature = "python")]
use num::rational::Ratio;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyAny, PyBool};
use serde::{Deserialize, Serialize};

use crate::{
    common::Scalar,
    semantic_key::{debug_key, f64_key},
};

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

/// A Blatt-Weisskopf barrier-factor amplitude for a two-body decay.
#[derive(Clone, Serialize, Deserialize)]
pub struct BlattWeisskopf {
    name: String,
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
        name: &str,
        decay: &Decay,
        l: OrbitalAngularMomentum,
        reference_mass: f64,
        q_r: f64,
        sheet: Sheet,
        kind: BarrierKind,
    ) -> LadduResult<Expression> {
        Self {
            name: name.to_string(),
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
        self.parent_mass_id =
            resources.register_scalar(Some(&format!("{}.parent_mass", self.name)));
        self.daughter_1_mass_id =
            resources.register_scalar(Some(&format!("{}.daughter_1_mass", self.name)));
        self.daughter_2_mass_id =
            resources.register_scalar(Some(&format!("{}.daughter_2_mass", self.name)));
        self.value_id = resources.register_complex_scalar(None);
        resources.register_amplitude(&self.name)
    }

    fn semantic_key(&self) -> Option<AmplitudeSemanticKey> {
        Some(
            AmplitudeSemanticKey::new("BlattWeisskopf")
                .with_field("name", debug_key(&self.name))
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

    fn precompute(&self, event: &NamedEventView<'_>, cache: &mut Cache) {
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

/// A Clebsch-Gordan coefficient expression.
pub struct ClebschGordan;

impl ClebschGordan {
    /// Construct a new constant expression for a Clebsch-Gordan coefficient.
    pub fn new(
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
        Scalar::new(name, parameter!(&format!("{name}.value"), value))
    }
}

/// A Wigner-3j symbol expression.
pub struct Wigner3j;

impl Wigner3j {
    /// Construct a new constant expression for a Wigner-3j symbol.
    pub fn new(
        name: &str,
        j1: AngularMomentum,
        m1: AngularMomentumProjection,
        j2: AngularMomentum,
        m2: AngularMomentumProjection,
        j3: AngularMomentum,
        m3: AngularMomentumProjection,
    ) -> LadduResult<Expression> {
        let value = wigner_3j(
            j1.value() as u64,
            j2.value() as u64,
            j3.value() as u64,
            m1.value() as i64,
            m2.value() as i64,
            m3.value() as i64,
        );
        Scalar::new(name, parameter!(&format!("{name}.value"), value))
    }
}

/// Photon polarization state used by [`PhotonSDME`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum PhotonPolarization {
    /// Unpolarized real photons.
    Unpolarized,
    /// Linearly polarized real photons.
    Linear(Box<Polarization>),
}

/// A real-photon helicity using the physical values `+1` and `-1`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PhotonHelicity(i8);

impl PhotonHelicity {
    /// Construct a photon helicity from a physical helicity value.
    pub fn new(value: i32) -> LadduResult<Self> {
        match value {
            -1 | 1 => Ok(Self(value as i8)),
            _ => Err(LadduError::Custom(
                "photon helicities must be physical values +/-1".to_string(),
            )),
        }
    }

    /// Return the physical helicity value.
    pub const fn value(self) -> i8 {
        self.0
    }
}

/// A photon spin-density matrix element.
#[derive(Clone, Serialize, Deserialize)]
pub struct PhotonSDME {
    name: String,
    polarization: PhotonPolarization,
    lambda: PhotonHelicity,
    lambda_prime: PhotonHelicity,
    value_id: ComplexScalarID,
}

impl PhotonSDME {
    /// Construct a new photon SDME amplitude.
    pub fn new(
        name: &str,
        polarization: PhotonPolarization,
        lambda: PhotonHelicity,
        lambda_prime: PhotonHelicity,
    ) -> LadduResult<Expression> {
        Self {
            name: name.to_string(),
            polarization,
            lambda,
            lambda_prime,
            value_id: ComplexScalarID::default(),
        }
        .into_expression()
    }
}

#[typetag::serde]
impl Amplitude for PhotonSDME {
    fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID> {
        self.value_id = resources.register_complex_scalar(None);
        resources.register_amplitude(&self.name)
    }

    fn semantic_key(&self) -> Option<AmplitudeSemanticKey> {
        Some(
            AmplitudeSemanticKey::new("PhotonSDME")
                .with_field("name", debug_key(&self.name))
                .with_field("polarization", debug_key(&self.polarization))
                .with_field("lambda", self.lambda.value().to_string())
                .with_field("lambda_prime", self.lambda_prime.value().to_string()),
        )
    }

    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        match &mut self.polarization {
            PhotonPolarization::Unpolarized => Ok(()),
            PhotonPolarization::Linear(polarization) => {
                polarization.pol_angle.bind(metadata)?;
                polarization.pol_magnitude.bind(metadata)
            }
        }
    }

    fn precompute(&self, event: &NamedEventView<'_>, cache: &mut Cache) {
        let value = match &self.polarization {
            PhotonPolarization::Unpolarized => {
                if self.lambda == self.lambda_prime {
                    Complex64::new(0.5, 0.0)
                } else {
                    Complex64::ZERO
                }
            }
            PhotonPolarization::Linear(polarization) => {
                if self.lambda == self.lambda_prime {
                    Complex64::new(0.5, 0.0)
                } else {
                    let magnitude = polarization.pol_magnitude.value(event);
                    let angle = polarization.pol_angle.value(event);
                    let sign = if self.lambda.value() > self.lambda_prime.value() {
                        -1.0
                    } else {
                        1.0
                    };
                    -0.5 * magnitude * Complex64::cis(sign * 2.0 * angle)
                }
            }
        };
        cache.store_complex_scalar(self.value_id, value);
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

/// Construct a Python Wigner-D amplitude.
#[cfg(feature = "python")]
#[pyfunction(name = "WignerD")]
pub fn py_wigner_d(
    name: &str,
    spin: &Bound<'_, PyAny>,
    row_projection: &Bound<'_, PyAny>,
    column_projection: &Bound<'_, PyAny>,
    angles: &Bound<'_, PyAny>,
) -> PyResult<PyExpression> {
    let spin = parse_angular_momentum(spin)?;
    let row_projection = parse_projection(row_projection)?;
    let column_projection = parse_projection(column_projection)?;
    if let Ok(angles) = angles.extract::<PyRef<'_, PyAngles>>() {
        Ok(PyExpression(WignerD::new(
            name,
            spin,
            row_projection,
            column_projection,
            &angles.0,
        )?))
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "angles must be an Angles object",
        ))
    }
}

/// Construct a Python Blatt-Weisskopf barrier-factor amplitude.
#[cfg(feature = "python")]
#[pyfunction(name = "BlattWeisskopf", signature = (name, decay, l, reference_mass, q_r = QR_DEFAULT, sheet = "physical", kind = "full"))]
pub fn py_blatt_weisskopf(
    name: &str,
    decay: &PyDecay,
    l: &Bound<'_, PyAny>,
    reference_mass: f64,
    q_r: f64,
    sheet: &str,
    kind: &str,
) -> PyResult<PyExpression> {
    let sheet = match sheet.to_ascii_lowercase().as_str() {
        "physical" => Sheet::Physical,
        "unphysical" => Sheet::Unphysical,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "sheet must be 'physical' or 'unphysical'",
            ));
        }
    };
    let kind = match kind.to_ascii_lowercase().as_str() {
        "full" => BarrierKind::Full,
        "tensor" => BarrierKind::Tensor,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "kind must be 'full' or 'tensor'",
            ));
        }
    };
    Ok(PyExpression(BlattWeisskopf::new(
        name,
        &decay.0,
        parse_orbital_angular_momentum(l)?,
        reference_mass,
        q_r,
        sheet,
        kind,
    )?))
}

/// Construct a Python Clebsch-Gordan constant expression.
#[cfg(feature = "python")]
#[pyfunction(name = "ClebschGordan")]
pub fn py_clebsch_gordan(
    name: &str,
    j1: &Bound<'_, PyAny>,
    m1: &Bound<'_, PyAny>,
    j2: &Bound<'_, PyAny>,
    m2: &Bound<'_, PyAny>,
    j: &Bound<'_, PyAny>,
    m: &Bound<'_, PyAny>,
) -> PyResult<PyExpression> {
    Ok(PyExpression(ClebschGordan::new(
        name,
        parse_angular_momentum(j1)?,
        parse_projection(m1)?,
        parse_angular_momentum(j2)?,
        parse_projection(m2)?,
        parse_angular_momentum(j)?,
        parse_projection(m)?,
    )?))
}

/// Construct a Python Wigner-3j constant expression.
#[cfg(feature = "python")]
#[pyfunction(name = "Wigner3j")]
pub fn py_wigner_3j(
    name: &str,
    j1: &Bound<'_, PyAny>,
    m1: &Bound<'_, PyAny>,
    j2: &Bound<'_, PyAny>,
    m2: &Bound<'_, PyAny>,
    j3: &Bound<'_, PyAny>,
    m3: &Bound<'_, PyAny>,
) -> PyResult<PyExpression> {
    Ok(PyExpression(Wigner3j::new(
        name,
        parse_angular_momentum(j1)?,
        parse_projection(m1)?,
        parse_angular_momentum(j2)?,
        parse_projection(m2)?,
        parse_angular_momentum(j3)?,
        parse_projection(m3)?,
    )?))
}

#[cfg(feature = "python")]
fn parse_angular_momentum(input: &Bound<'_, PyAny>) -> PyResult<AngularMomentum> {
    parse_ratio_like(input)
        .and_then(AngularMomentum::from_ratio)
        .map_err(py_value_error)
}

#[cfg(feature = "python")]
fn parse_projection(input: &Bound<'_, PyAny>) -> PyResult<AngularMomentumProjection> {
    parse_ratio_like(input)
        .and_then(AngularMomentumProjection::from_ratio)
        .map_err(py_value_error)
}

#[cfg(feature = "python")]
fn parse_orbital_angular_momentum(input: &Bound<'_, PyAny>) -> PyResult<OrbitalAngularMomentum> {
    parse_ratio_like(input)
        .and_then(OrbitalAngularMomentum::from_ratio)
        .map_err(py_value_error)
}

#[cfg(feature = "python")]
fn parse_ratio_like(input: &Bound<'_, PyAny>) -> LadduResult<Ratio<i32>> {
    if input.is_instance_of::<PyBool>() {
        return Err(LadduError::Custom(
            "quantum number cannot be a bool".to_string(),
        ));
    }
    if let Ok(value) = input.extract::<i32>() {
        return Ok(Ratio::from_integer(value));
    }
    if let Ok(value) = input.extract::<f64>() {
        let twice = AngularMomentumProjection::from_f64(value)?.value();
        return Ok(Ratio::new(twice, 2));
    }
    let numerator = input
        .getattr("numerator")
        .and_then(|value| value.extract::<i32>());
    let denominator = input
        .getattr("denominator")
        .and_then(|value| value.extract::<i32>());
    if let (Ok(numerator), Ok(denominator)) = (numerator, denominator) {
        if denominator == 0 {
            return Err(LadduError::Custom(
                "quantum number denominator cannot be zero".to_string(),
            ));
        }
        return Ok(Ratio::new(numerator, denominator));
    }
    Err(LadduError::Custom(
        "quantum number must be an int, float, or fractions.Fraction".to_string(),
    ))
}

#[cfg(feature = "python")]
fn py_value_error(err: LadduError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(err.to_string())
}

/// Construct a Python photon SDME amplitude.
#[cfg(feature = "python")]
#[pyfunction(name = "PhotonSDME", signature = (name, helicity, helicity_prime, polarization = None))]
pub fn py_photon_sdme(
    name: &str,
    helicity: i32,
    helicity_prime: i32,
    polarization: Option<&PyPolarization>,
) -> PyResult<PyExpression> {
    let polarization = polarization
        .map(|polarization| PhotonPolarization::Linear(Box::new(polarization.0.clone())))
        .unwrap_or(PhotonPolarization::Unpolarized);
    Ok(PyExpression(PhotonSDME::new(
        name,
        polarization,
        PhotonHelicity::new(helicity)?,
        PhotonHelicity::new(helicity_prime)?,
    )?))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use approx::assert_relative_eq;
    use laddu_core::{
        data::test_dataset,
        math::{BarrierKind, WignerDMatrix},
        reaction::{Particle, Reaction},
        Frame,
    };

    use super::*;

    fn reaction_context() -> (Reaction, Particle, Particle) {
        let beam = Particle::measured("beam", "beam");
        let target = Particle::missing("target");
        let kshort1 = Particle::measured("K_S1", "kshort1");
        let kshort2 = Particle::measured("K_S2", "kshort2");
        let kk = Particle::composite("KK", [&kshort1, &kshort2]).unwrap();
        let proton = Particle::measured("proton", "proton");
        (
            Reaction::two_to_two(&beam, &target, &kk, &proton).unwrap(),
            kk,
            kshort1,
        )
    }

    #[test]
    fn wigner_d_matches_core_function() {
        let dataset = Arc::new(test_dataset());
        let (reaction, kk, kshort1) = reaction_context();
        let decay = reaction.decay(&kk).unwrap();
        let angles = decay.angles(&kshort1, Frame::Helicity).unwrap();
        let expr = WignerD::new(
            "d",
            AngularMomentum::from_twice(2),
            AngularMomentumProjection::from_twice(2),
            AngularMomentumProjection::from_twice(0),
            &angles,
        )
        .unwrap();
        let evaluator = expr.load(&dataset).unwrap();
        let event = dataset.event_view(0);
        let mut costheta = angles.costheta.clone();
        let mut phi = angles.phi.clone();
        costheta.bind(dataset.metadata()).unwrap();
        phi.bind(dataset.metadata()).unwrap();
        let expected = WignerDMatrix::new(2, 2, 0).D(
            event.evaluate(&phi),
            event.evaluate(&costheta).clamp(-1.0, 1.0).acos(),
            0.0,
        );
        let value = evaluator.evaluate(&[]).unwrap()[0];

        assert_relative_eq!(value.re, expected.re);
        assert_relative_eq!(value.im, expected.im);
    }

    #[test]
    fn clebsch_gordan_constant_matches_core_function() {
        let dataset = Arc::new(test_dataset());
        let expr = ClebschGordan::new(
            "cg",
            AngularMomentum::from_twice(1),
            AngularMomentumProjection::from_twice(1),
            AngularMomentum::from_twice(1),
            AngularMomentumProjection::from_twice(-1),
            AngularMomentum::from_twice(2),
            AngularMomentumProjection::from_twice(0),
        )
        .unwrap();
        let value = expr.load(&dataset).unwrap().evaluate(&[]).unwrap()[0];

        assert_relative_eq!(value.re, 1.0 / 2.0_f64.sqrt());
        assert_relative_eq!(value.im, 0.0);
    }

    #[test]
    fn photon_sdme_unpolarized_is_diagonal() {
        let dataset = Arc::new(test_dataset());
        let diagonal = PhotonSDME::new(
            "rho_diag",
            PhotonPolarization::Unpolarized,
            PhotonHelicity::new(1).unwrap(),
            PhotonHelicity::new(1).unwrap(),
        )
        .unwrap();
        let off_diagonal = PhotonSDME::new(
            "rho_off",
            PhotonPolarization::Unpolarized,
            PhotonHelicity::new(1).unwrap(),
            PhotonHelicity::new(-1).unwrap(),
        )
        .unwrap();

        assert_relative_eq!(
            diagonal.load(&dataset).unwrap().evaluate(&[]).unwrap()[0].re,
            0.5
        );
        assert_relative_eq!(
            off_diagonal.load(&dataset).unwrap().evaluate(&[]).unwrap()[0].norm(),
            0.0
        );
    }

    #[test]
    fn blatt_weisskopf_accepts_reaction_decay_context() {
        let beam = laddu_core::Particle::measured("beam", "beam");
        let target = laddu_core::Particle::measured("target", "target");
        let k1 = laddu_core::Particle::measured("k1", "kshort1");
        let k2 = laddu_core::Particle::measured("k2", "kshort2");
        let x = laddu_core::Particle::composite("x", [&k1, &k2]).unwrap();
        let recoil = laddu_core::Particle::measured("recoil", "proton");
        let reaction = laddu_core::Reaction::two_to_two(&beam, &target, &x, &recoil).unwrap();
        let decay = reaction.decay(&x).unwrap();
        let expr = BlattWeisskopf::new(
            "b",
            &decay,
            OrbitalAngularMomentum::integer(2),
            1.5,
            QR_DEFAULT,
            Sheet::Physical,
            BarrierKind::Full,
        )
        .unwrap();
        let dataset = Arc::new(test_dataset());
        let event = dataset.event_view(0);
        let value = expr.load(&dataset).unwrap().evaluate(&[]).unwrap()[0];
        let expected = blatt_weisskopf_m(
            event.get_p4_sum(["kshort1", "kshort2"]).unwrap().m(),
            event.p4("kshort1").unwrap().m(),
            event.p4("kshort2").unwrap().m(),
            2,
            QR_DEFAULT,
            Sheet::Physical,
            BarrierKind::Full,
        ) / blatt_weisskopf_m(
            1.5,
            event.p4("kshort1").unwrap().m(),
            event.p4("kshort2").unwrap().m(),
            2,
            QR_DEFAULT,
            Sheet::Physical,
            BarrierKind::Full,
        );

        assert_relative_eq!(value.re, expected.re);
        assert_relative_eq!(value.im, expected.im);
    }

    #[test]
    fn helicity_factor_matches_conjugated_wigner_d() {
        let dataset = Arc::new(test_dataset());
        let (reaction, kk, kshort1) = reaction_context();
        let decay = reaction.decay(&kk).unwrap();
        let factor = decay
            .helicity_factor(
                "h",
                AngularMomentum::integer(2),
                AngularMomentumProjection::integer(1),
                &kshort1,
                AngularMomentumProjection::integer(1),
                AngularMomentumProjection::integer(0),
                Frame::Helicity,
            )
            .unwrap();
        let angles = decay.angles(&kshort1, Frame::Helicity).unwrap();
        let explicit = WignerD::new(
            "d",
            AngularMomentum::integer(2),
            AngularMomentumProjection::integer(1),
            AngularMomentumProjection::integer(1),
            &angles,
        )
        .unwrap()
        .conj();

        let factor_value = factor.load(&dataset).unwrap().evaluate(&[]).unwrap()[0];
        let explicit_value = explicit.load(&dataset).unwrap().evaluate(&[]).unwrap()[0];

        assert_relative_eq!(factor_value.re, explicit_value.re);
        assert_relative_eq!(factor_value.im, explicit_value.im);
    }

    #[test]
    fn canonical_factor_matches_explicit_product() {
        let dataset = Arc::new(test_dataset());
        let (reaction, kk, kshort1) = reaction_context();
        let decay = reaction.decay(&kk).unwrap();
        let factor = decay
            .canonical_factor(
                "c",
                AngularMomentum::integer(2),
                AngularMomentumProjection::integer(0),
                OrbitalAngularMomentum::integer(2),
                AngularMomentum::integer(0),
                &kshort1,
                AngularMomentum::integer(0),
                AngularMomentum::integer(0),
                AngularMomentumProjection::integer(0),
                AngularMomentumProjection::integer(0),
                Frame::Helicity,
            )
            .unwrap();
        let explicit = Scalar::new("norm", parameter!("norm.value", 5.0_f64.sqrt())).unwrap()
            * ClebschGordan::new(
                "orbital_spin",
                AngularMomentum::integer(2),
                AngularMomentumProjection::integer(0),
                AngularMomentum::integer(0),
                AngularMomentumProjection::integer(0),
                AngularMomentum::integer(2),
                AngularMomentumProjection::integer(0),
            )
            .unwrap()
            * ClebschGordan::new(
                "daughter_spin",
                AngularMomentum::integer(0),
                AngularMomentumProjection::integer(0),
                AngularMomentum::integer(0),
                AngularMomentumProjection::integer(0),
                AngularMomentum::integer(0),
                AngularMomentumProjection::integer(0),
            )
            .unwrap()
            * decay
                .helicity_factor(
                    "d",
                    AngularMomentum::integer(2),
                    AngularMomentumProjection::integer(0),
                    &kshort1,
                    AngularMomentumProjection::integer(0),
                    AngularMomentumProjection::integer(0),
                    Frame::Helicity,
                )
                .unwrap();

        let factor_value = factor.load(&dataset).unwrap().evaluate(&[]).unwrap()[0];
        let explicit_value = explicit.load(&dataset).unwrap().evaluate(&[]).unwrap()[0];

        assert_relative_eq!(factor_value.re, explicit_value.re);
        assert_relative_eq!(factor_value.im, explicit_value.im);
    }
}
