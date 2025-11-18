use laddu_core::{
    amplitudes::{Amplitude, AmplitudeID, Expression},
    data::{DatasetMetadata, EventData},
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

    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        self.angles.costheta.bind(metadata)?;
        self.angles.phi.bind(metadata)?;
        Ok(())
    }

    fn precompute(&self, event: &EventData, cache: &mut Cache) {
        cache.store_complex_scalar(
            self.csid,
            spherical_harmonic(
                self.l,
                self.m,
                self.angles.costheta.value(event),
                self.angles.phi.value(event),
            ),
        );
    }

    fn compute(&self, _parameters: &Parameters, _event: &EventData, cache: &Cache) -> Complex64 {
        cache.get_complex_scalar(self.csid)
    }

    fn compute_gradient(
        &self,
        _parameters: &Parameters,
        _event: &EventData,
        _cache: &Cache,
        _gradient: &mut DVector<Complex64>,
    ) {
        // This amplitude is independent of free parameters
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
    use laddu_core::{data::test_dataset, utils::variables::Topology, Frame};

    fn reaction_topology() -> Topology {
        Topology::missing_k2("beam", ["kshort1", "kshort2"], "proton")
    }

    #[test]
    fn test_ylm_evaluation() {
        let dataset = Arc::new(test_dataset());
        let angles = Angles::new(reaction_topology(), "kshort1", Frame::Helicity);
        let expr = Ylm::new("ylm", 1, 1, &angles).unwrap();
        let evaluator = expr.load(&dataset).unwrap();

        let result = evaluator.evaluate(&[]);

        assert_relative_eq!(result[0].re, 0.2713394403451035);
        assert_relative_eq!(result[0].im, 0.1426897184196572);
    }

    #[test]
    fn test_ylm_gradient() {
        let dataset = Arc::new(test_dataset());
        let angles = Angles::new(reaction_topology(), "kshort1", Frame::Helicity);
        let expr = Ylm::new("ylm", 1, 1, &angles).unwrap();
        let evaluator = expr.load(&dataset).unwrap();

        let result = evaluator.evaluate_gradient(&[]);
        assert_eq!(result[0].len(), 0); // amplitude has no parameters
    }
}
