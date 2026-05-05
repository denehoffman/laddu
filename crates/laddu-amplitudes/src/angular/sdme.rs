use laddu_core::{
    amplitudes::{debug_key, Amplitude, AmplitudeID, AmplitudeSemanticKey, Expression},
    data::{DatasetMetadata, Event},
    resources::{Cache, ComplexScalarID, Parameters, Resources},
    traits::Variable,
    LadduError, LadduResult, Polarization,
};
use nalgebra::DVector;
use num::complex::Complex64;
use serde::{Deserialize, Serialize};

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

    fn precompute(&self, event: &Event<'_>, cache: &mut Cache) {
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
