use super::FixedKMatrix;
use laddu_core::{
    amplitudes::{Amplitude, AmplitudeID, Expression, ParameterLike},
    data::{DatasetMetadata, EventData},
    resources::{Cache, ComplexVectorID, MatrixID, ParameterID, Parameters, Resources},
    utils::variables::{Mass, Variable},
    LadduResult,
};
#[cfg(feature = "python")]
use laddu_python::{
    amplitudes::{PyExpression, PyParameterLike},
    utils::variables::PyMass,
};
use nalgebra::{matrix, vector, DVector, SVector};
use num::complex::Complex64;
#[cfg(feature = "python")]
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::array;

/// A K-matrix parameterization for $`\rho`$ particles described by Kopf et al.[^1] with fixed couplings and mass poles
/// (free production couplings only).
///
/// [^1]: Kopf, B., Albrecht, M., Koch, H., Küßner, M., Pychy, J., Qin, X., & Wiedner, U. (2021). Investigation of the lightest hybrid meson candidate with a coupled-channel analysis of $`\bar{p}p`$-, $`\pi^- p`$- and $`\pi \pi`$-Data. The European Physical Journal C, 81(12). [doi:10.1140/epjc/s10052-021-09821-2](https://doi.org/10.1140/epjc/s10052-021-09821-2)
#[derive(Clone, Serialize, Deserialize)]
pub struct KopfKMatrixRho {
    name: String,
    channel: usize,
    mass: Mass,
    constants: FixedKMatrix<3, 2>,
    couplings_real: [ParameterLike; 2],
    couplings_imag: [ParameterLike; 2],
    couplings_indices_real: [ParameterID; 2],
    couplings_indices_imag: [ParameterID; 2],
    ikc_cache_index: ComplexVectorID<3>,
    p_vec_cache_index: MatrixID<3, 2>,
}

impl KopfKMatrixRho {
    /// Construct a new [`KopfKMatrixRho`] with the given name, production couplings, channel,
    /// and input mass.
    ///
    /// | Channel index | Channel |
    /// | ------------- | ------- |
    /// | 0             | $`\pi\pi`$ |
    /// | 1             | $`2\pi 2\pi`$ |
    /// | 2             | $`K\bar{K}`$ |
    ///
    /// | Pole names |
    /// | ---------- |
    /// | $`\rho(770)`$ |
    /// | $`\rho(1700)`$ |
    pub fn new(
        name: &str,
        couplings: [[ParameterLike; 2]; 2],
        channel: usize,
        mass: &Mass,
    ) -> LadduResult<Expression> {
        let mut couplings_real: [ParameterLike; 2] = array::from_fn(|_| ParameterLike::default());
        let mut couplings_imag: [ParameterLike; 2] = array::from_fn(|_| ParameterLike::default());
        for i in 0..2 {
            couplings_real[i] = couplings[i][0].clone();
            couplings_imag[i] = couplings[i][1].clone();
        }
        Self {
            name: name.to_string(),
            channel,
            mass: mass.clone(),
            constants: FixedKMatrix {
                g: matrix![
                     0.28023,  0.16318;
                     0.01806,  0.53879;
                     0.06501,  0.00495
                ],
                c: matrix![
                    -0.06948,  0.00000,  0.07958;
                     0.00000,  0.00000,  0.00000;
                     0.07958,  0.00000, -0.60000
                ],
                m1s: vector![0.1349768, 2.0 * 0.1349768, 0.493677],
                m2s: vector![0.1349768, 2.0 * 0.1349768, 0.497611],
                mrs: vector![0.71093, 1.58660],
                adler_zero: None,
                l: 1,
            },
            couplings_real,
            couplings_imag,
            couplings_indices_real: [ParameterID::default(); 2],
            couplings_indices_imag: [ParameterID::default(); 2],
            ikc_cache_index: ComplexVectorID::default(),
            p_vec_cache_index: MatrixID::default(),
        }
        .into_expression()
    }
}

#[typetag::serde]
impl Amplitude for KopfKMatrixRho {
    fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID> {
        for i in 0..self.couplings_indices_real.len() {
            self.couplings_indices_real[i] =
                resources.register_parameter(&self.couplings_real[i])?;
            self.couplings_indices_imag[i] =
                resources.register_parameter(&self.couplings_imag[i])?;
        }
        self.ikc_cache_index = resources
            .register_complex_vector(Some(&format!("KopfKMatrixRho<{}> ikc_vec", self.name)));
        self.p_vec_cache_index =
            resources.register_matrix(Some(&format!("KopfKMatrixRho<{}> p_vec", self.name)));
        resources.register_amplitude(&self.name)
    }

    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        self.mass.bind(metadata)?;
        Ok(())
    }

    fn precompute(&self, event: &EventData, cache: &mut Cache) {
        let s = self.mass.value(event).powi(2);
        cache.store_complex_vector(
            self.ikc_cache_index,
            self.constants.ikc_inv_vec(s, self.channel),
        );
        cache.store_matrix(self.p_vec_cache_index, self.constants.p_vec_constants(s));
    }

    fn compute(&self, parameters: &Parameters, _event: &EventData, cache: &Cache) -> Complex64 {
        let betas = SVector::from_fn(|i, _| {
            Complex64::new(
                parameters.get(self.couplings_indices_real[i]),
                parameters.get(self.couplings_indices_imag[i]),
            )
        });
        let ikc_inv_vec = cache.get_complex_vector(self.ikc_cache_index);
        let p_vec_constants = cache.get_matrix(self.p_vec_cache_index);
        FixedKMatrix::compute(&betas, &ikc_inv_vec, &p_vec_constants)
    }

    fn compute_cached(&self, parameters: &Parameters, cache: &Cache) -> LadduResult<Complex64> {
        let betas = SVector::from_fn(|i, _| {
            Complex64::new(
                parameters.get(self.couplings_indices_real[i]),
                parameters.get(self.couplings_indices_imag[i]),
            )
        });
        let ikc_inv_vec = cache.get_complex_vector(self.ikc_cache_index);
        let p_vec_constants = cache.get_matrix(self.p_vec_cache_index);
        Ok(FixedKMatrix::compute(
            &betas,
            &ikc_inv_vec,
            &p_vec_constants,
        ))
    }

    fn compute_gradient(
        &self,
        _parameters: &Parameters,
        _event: &EventData,
        cache: &Cache,
        gradient: &mut DVector<Complex64>,
    ) {
        let ikc_inv_vec = cache.get_complex_vector(self.ikc_cache_index);
        let p_vec_constants = cache.get_matrix(self.p_vec_cache_index);
        let internal_gradient = FixedKMatrix::compute_gradient(&ikc_inv_vec, &p_vec_constants);
        for i in 0..2 {
            if let ParameterID::Parameter(index) = self.couplings_indices_real[i] {
                gradient[index] = internal_gradient[i];
            }
            if let ParameterID::Parameter(index) = self.couplings_indices_imag[i] {
                gradient[index] = Complex64::I * internal_gradient[i];
            }
        }
    }
}

/// A fixed K-Matrix Amplitude for :math:`\rho` mesons
///
/// Parameters
/// ----------
/// name : str
///     The Amplitude name
/// couplings : list of list of laddu.ParameterLike
///     Each initial-state coupling (as a list of pairs of real and imaginary parts)
/// channel : int
///     The channel onto which the K-Matrix is projected
/// mass: laddu.Mass
///     The total mass of the resonance
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
/// Notes
/// -----
/// This Amplitude follows the prescription of [Kopf]_ and fixes the K-Matrix to data
/// from that paper, leaving the couplings to the initial state free
///
/// +---------------+-------------------+
/// | Channel index | Channel           |
/// +===============+===================+
/// | 0             | :math:`\pi\pi`    |
/// +---------------+-------------------+
/// | 1             | :math:`2\pi 2\pi` |
/// +---------------+-------------------+
/// | 2             | :math:`K\bar{K}`  |
/// +---------------+-------------------+
///
/// +--------------------+
/// | Pole names         |
/// +====================+
/// | :math:`\rho(770)`  |
/// +--------------------+
/// | :math:`\rho(1700)` |
/// +--------------------+
///
#[cfg(feature = "python")]
#[pyfunction(name = "KopfKMatrixRho")]
pub fn py_kopf_kmatrix_rho(
    name: &str,
    couplings: [[PyParameterLike; 2]; 2],
    channel: usize,
    mass: PyMass,
) -> PyResult<PyExpression> {
    Ok(PyExpression(KopfKMatrixRho::new(
        name,
        array::from_fn(|i| array::from_fn(|j| couplings[i][j].clone().0)),
        channel,
        &mass.0,
    )?))
}

#[cfg(test)]
mod tests {
    // Note: These tests are not exhaustive, they only check one channel
    use std::sync::Arc;

    use super::*;
    use approx::assert_relative_eq;
    use laddu_core::{data::test_dataset, parameter, Mass};

    #[test]
    fn test_rho_evaluation() {
        let dataset = Arc::new(test_dataset());
        let res_mass = Mass::new(["kshort1", "kshort2"]);
        let expr = KopfKMatrixRho::new(
            "rho",
            [
                [parameter("p0"), parameter("p1")],
                [parameter("p2"), parameter("p3")],
            ],
            1,
            &res_mass,
        )
        .unwrap();
        let evaluator = expr.load(&dataset).unwrap();

        let result = evaluator.evaluate(&[0.1, 0.2, 0.3, 0.4]);

        assert_relative_eq!(result[0].re, 0.09483558754117698);
        assert_relative_eq!(result[0].im, 0.2609183741271106);
    }

    #[test]
    fn test_rho_gradient() {
        let dataset = Arc::new(test_dataset());
        let res_mass = Mass::new(["kshort1", "kshort2"]);
        let expr = KopfKMatrixRho::new(
            "rho",
            [
                [parameter("p0"), parameter("p1")],
                [parameter("p2"), parameter("p3")],
            ],
            1,
            &res_mass,
        )
        .unwrap();
        let evaluator = expr.load(&dataset).unwrap();

        let result = evaluator.evaluate_gradient(&[0.1, 0.2, 0.3, 0.4]);

        assert_relative_eq!(result[0][0].re, 0.026520319348816407);
        assert_relative_eq!(result[0][0].im, -0.026602652559793133);
        assert_relative_eq!(result[0][1].re, -result[0][0].im);
        assert_relative_eq!(result[0][1].im, result[0][0].re);
        assert_relative_eq!(result[0][2].re, 0.5172379289201292);
        assert_relative_eq!(result[0][2].im, 0.17073733305788397);
        assert_relative_eq!(result[0][3].re, -result[0][2].im);
        assert_relative_eq!(result[0][3].im, result[0][2].re);
    }
}
