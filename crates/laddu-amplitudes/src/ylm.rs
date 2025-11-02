use laddu_core::{
    amplitudes::{Amplitude, AmplitudeID},
    resources::{CacheRow, ExprName, Parameters, Resources},
    utils::functions::spherical_harmonic_polars,
    ExprID, LadduResult,
};
#[cfg(feature = "python")]
use laddu_python::amplitudes::PyAmplitude;
use nalgebra::DVector;
use num::complex::Complex64;
use polars::prelude::Expr;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3_polars::PyExpr;
use serde::{Deserialize, Serialize};

/// An [`Amplitude`] for the spherical harmonic function $`Y_\ell^m(\theta, \phi)`$.
#[derive(Clone, Serialize, Deserialize)]
pub struct Ylm {
    name: String,
    l: usize,
    m: isize,
    angles: [Expr; 2],
    eid: ExprID,
}

impl Ylm {
    /// Construct a new [`Ylm`] with the given name, angular momentum (`l`) and moment (`m`) over
    /// the given set of [`Angles`].
    pub fn new(name: &str, l: usize, m: isize, angles: [Expr; 2]) -> Box<Self> {
        Self {
            name: name.to_string(),
            l,
            m,
            angles: angles.clone(),
            eid: Default::default(),
        }
        .into()
    }
}

#[typetag::serde]
impl Amplitude for Ylm {
    fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID> {
        let ylm = spherical_harmonic_polars(
            self.l,
            self.m,
            self.angles[0].clone(),
            self.angles[1].clone(),
        );
        self.eid = resources.register_cscalar(ExprName::Infer, ylm)?;
        resources.register_amplitude(&self.name)
    }

    fn compute(&self, _parameters: &Parameters, cache_row: &CacheRow) -> Complex64 {
        cache_row.get_cscalar(self.eid)
    }

    fn compute_gradient(
        &self,
        _parameters: &Parameters,
        _cache_row: &CacheRow,
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
/// laddu.Amplitude
///     An Amplitude which can be registered by a laddu.Manager
///
/// See Also
/// --------
/// laddu.Manager
///
#[cfg(feature = "python")]
#[pyfunction(name = "Ylm")]
pub fn py_ylm(name: &str, l: usize, m: isize, angles: Bound<PyAny>) -> PyResult<PyAmplitude> {
    let (costheta, phi) = angles.extract::<(PyExpr, PyExpr)>()?;
    Ok(PyAmplitude(Ylm::new(name, l, m, [costheta.0, phi.0])))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use laddu_core::{angles, data::test_dataset, Frame, Manager};

    #[test]
    fn test_ylm_evaluation() {
        let mut manager = Manager::default();
        let angles = angles(
            "beam",
            ["proton"],
            ["kshort1"],
            ["kshort1", "kshort2"],
            Frame::Helicity,
        );
        let amp = Ylm::new("ylm", 1, 1, angles);
        let aid = manager.register(amp).unwrap();

        let dataset = test_dataset();
        let expr = aid.into();
        let model = manager.model(&expr);
        let evaluator = model.load(&dataset).unwrap();

        let result = evaluator.evaluate(&[]).unwrap();

        assert_relative_eq!(result[0].re, 0.27133944, epsilon = f64::EPSILON.sqrt());
        assert_relative_eq!(result[0].im, 0.14268971, epsilon = f64::EPSILON.sqrt());
    }

    #[test]
    fn test_ylm_gradient() {
        let mut manager = Manager::default();
        let angles = angles(
            "beam",
            ["proton"],
            ["kshort1"],
            ["kshort1", "kshort2"],
            Frame::Helicity,
        );
        let amp = Ylm::new("ylm", 1, 1, angles);
        let aid = manager.register(amp).unwrap();

        let dataset = test_dataset();
        let expr = aid.into();
        let model = manager.model(&expr);
        let evaluator = model.load(&dataset).unwrap();

        let result = evaluator.evaluate_gradient(&[]).unwrap();
        assert_eq!(result[0].len(), 0); // amplitude has no parameters
    }
}
