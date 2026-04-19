use laddu_core::{
    amplitudes::{Amplitude, AmplitudeID, AmplitudeSemanticKey, Expression},
    data::{DatasetMetadata, NamedEventView},
    resources::{Cache, ComplexScalarID, Parameters, Resources},
    utils::{
        functions::spherical_harmonic,
        variables::{Angles, Variable},
    },
    LadduResult,
};
#[cfg(feature = "python")]
use laddu_python::{amplitudes::PyExpression, utils::variables::PyAngles};
use nalgebra::DVector;
use num::complex::Complex64;
#[cfg(feature = "python")]
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use crate::semantic_key::{debug_key, display_key};

/// An [`Amplitude`] for the spherical harmonic function $`Y_\ell^m(\theta, \phi)`$.
#[derive(Clone, Serialize, Deserialize)]
pub struct Ylm {
    name: String,
    l: usize,
    m: isize,
    angles: Angles,
    csid: ComplexScalarID,
}

impl Ylm {
    /// Construct a new [`Ylm`] with the given name, angular momentum (`l`) and moment (`m`) over
    /// the given set of [`Angles`].
    pub fn new(name: &str, l: usize, m: isize, angles: &Angles) -> LadduResult<Expression> {
        Self {
            name: name.to_string(),
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
        resources.register_amplitude(&self.name)
    }

    fn semantic_key(&self) -> Option<AmplitudeSemanticKey> {
        Some(
            AmplitudeSemanticKey::new("Ylm")
                .with_field("name", debug_key(&self.name))
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

    fn precompute(&self, event: &NamedEventView<'_>, cache: &mut Cache) {
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

/// An spherical harmonic Amplitude
///
/// Computes a spherical harmonic (:math:`Y_{\ell}^m(\theta, \varphi)`)
///
/// Parameters
/// ----------
/// name : str
///     The Amplitude name
/// l : int
///     The total orbital momentum (:math:`l \geq 0`)
/// m : int
///     The orbital moment (:math:`-l \leq m \leq l`)
/// angles : laddu.Angles
///     The spherical angles to use in the calculation
///
/// Returns
/// -------
/// laddu.Expression
///     An Expression which can be loaded and evaluated directly
///
#[cfg(feature = "python")]
#[pyfunction(name = "Ylm")]
pub fn py_ylm(name: &str, l: usize, m: isize, angles: &PyAngles) -> PyResult<PyExpression> {
    Ok(PyExpression(Ylm::new(name, l, m, &angles.0)?))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use approx::assert_relative_eq;
    use laddu_core::{
        data::test_dataset,
        utils::reaction::{Particle, Reaction},
        Frame,
    };

    fn angles() -> Angles {
        let beam = Particle::measured("beam", "beam");
        let target = Particle::missing("target");
        let kshort1 = Particle::measured("K_S1", "kshort1");
        let kshort2 = Particle::measured("K_S2", "kshort2");
        let kk = Particle::composite("KK", [&kshort1, &kshort2]).unwrap();
        let proton = Particle::measured("proton", "proton");
        let reaction = Reaction::two_to_two(&beam, &target, &kk, &proton).unwrap();
        reaction
            .decay(&kk)
            .unwrap()
            .angles(&kshort1, Frame::Helicity)
            .unwrap()
    }

    #[test]
    fn test_ylm_evaluation() {
        let dataset = Arc::new(test_dataset());
        let angles = angles();
        let expr = Ylm::new("ylm", 1, 1, &angles).unwrap();
        let evaluator = expr.load(&dataset).unwrap();

        let result = evaluator.evaluate(&[]);

        assert_relative_eq!(result[0].re, 0.2713394403451035);
        assert_relative_eq!(result[0].im, 0.1426897184196572);
    }

    #[test]
    fn test_ylm_gradient() {
        let dataset = Arc::new(test_dataset());
        let angles = angles();
        let expr = Ylm::new("ylm", 1, 1, &angles).unwrap();
        let evaluator = expr.load(&dataset).unwrap();

        let result = evaluator.evaluate_gradient(&[]);
        assert_eq!(result[0].len(), 0); // amplitude has no parameters
    }

    #[test]
    fn test_ylm_m_zero_reports_real_valued_hint() {
        let angles = angles();
        let real_ylm = Ylm {
            name: "ylm0".to_string(),
            l: 1,
            m: 0,
            angles: angles.clone(),
            csid: ComplexScalarID::default(),
        };
        let complex_ylm = Ylm {
            name: "ylm1".to_string(),
            l: 1,
            m: 1,
            angles,
            csid: ComplexScalarID::default(),
        };

        assert!(Amplitude::real_valued_hint(&real_ylm));
        assert!(!Amplitude::real_valued_hint(&complex_ylm));
    }
}
