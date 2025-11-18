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
use nalgebra::{matrix, vector, DVector, SMatrix, SVector};
use num::complex::Complex64;
#[cfg(feature = "python")]
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::array;

const G_A0: SMatrix<f64, 2, 2> = matrix![
     0.43215,  0.19000;
    -0.28825,  0.43372
];
const C_A0: SMatrix<f64, 2, 2> = matrix![
     0.00000,  0.00000;
     0.00000,  0.00000
];
const M_A0: SVector<f64, 2> = vector![0.95395, 1.26767];
const COV_A0: SMatrix<f64, 10, 10> = matrix![
    0.00012122797646, 0.00003066079649, -0.00028703990499, 0.00001292827118, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, -0.00002958942659, -0.00001922174568;
    0.00003066079649, 0.00038954159059, -0.00027049599621, -0.00021015802987, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, -0.00001858825050, -0.00003332104667;
    -0.00028703990499, -0.00027049599621, 0.00460975157866, 0.00002489860531, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, -0.00015362983837, 0.00008571810966;
    0.00001292827118, -0.00021015802987, 0.00002489860531, 0.00029242822529, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00001331759124, -0.00000794584049;
    0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000;
    0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000;
    0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000;
    0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000;
    -0.00002958942659, -0.00001858825050, -0.00015362983837, 0.00001331759124, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00003500612212, 0.00000327456443;
    -0.00001922174568, -0.00003332104667, 0.00008571810966, -0.00000794584049, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000327456443, 0.00002146933344;
];

/// A K-matrix parameterization for $`a_0`$ particles described by Kopf et al.[^1] with fixed couplings and mass poles
/// (free production couplings only).
///
/// [^1]: Kopf, B., Albrecht, M., Koch, H., Küßner, M., Pychy, J., Qin, X., & Wiedner, U. (2021). Investigation of the lightest hybrid meson candidate with a coupled-channel analysis of $`\bar{p}p`$-, $`\pi^- p`$- and $`\pi \pi`$-Data. The European Physical Journal C, 81(12). [doi:10.1140/epjc/s10052-021-09821-2](https://doi.org/10.1140/epjc/s10052-021-09821-2)
#[derive(Clone, Serialize, Deserialize)]
pub struct KopfKMatrixA0 {
    name: String,
    channel: usize,
    mass: Mass,
    constants: FixedKMatrix<2, 2>,
    couplings_real: [ParameterLike; 2],
    couplings_imag: [ParameterLike; 2],
    couplings_indices_real: [ParameterID; 2],
    couplings_indices_imag: [ParameterID; 2],
    ikc_cache_index: ComplexVectorID<2>,
    p_vec_cache_index: MatrixID<2, 2>,
}

impl KopfKMatrixA0 {
    /// Construct a new [`KopfKMatrixA0`] with the given name, production couplings, channel,
    /// and input mass.
    ///
    /// | Channel index | Channel |
    /// | ------------- | ------- |
    /// | 0             | $`\pi\eta`$ |
    /// | 1             | $`K\bar{K}`$ |
    ///
    /// | Pole names |
    /// | ---------- |
    /// | $`a_0(980)`$ |
    /// | $`a_0(1450)`$ |
    pub fn new(
        name: &str,
        couplings: [[ParameterLike; 2]; 2],
        channel: usize,
        mass: &Mass,
        seed: Option<usize>,
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
            constants: FixedKMatrix::new(
                G_A0,
                C_A0,
                vector![0.1349768, 0.493677],
                vector![0.547862, 0.497611],
                M_A0,
                None,
                0,
                COV_A0,
                seed,
            ),
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
impl Amplitude for KopfKMatrixA0 {
    fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID> {
        for i in 0..self.couplings_indices_real.len() {
            self.couplings_indices_real[i] = resources.register_parameter(&self.couplings_real[i]);
            self.couplings_indices_imag[i] = resources.register_parameter(&self.couplings_imag[i]);
        }
        self.ikc_cache_index = resources
            .register_complex_vector(Some(&format!("KopfKMatrixA0<{}> ikc_vec", self.name)));
        self.p_vec_cache_index =
            resources.register_matrix(Some(&format!("KopfKMatrixA0<{}> p_vec", self.name)));
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

/// A fixed K-Matrix Amplitude for :math:`a_0` mesons
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
/// seed: int, optional
///     Seed used to resample fixed K-matrix components according to their covariance
///     No resampling is done if seed is None
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
/// | 0             | :math:`\pi\eta`   |
/// +---------------+-------------------+
/// | 1             | :math:`K\bar{K}`  |
/// +---------------+-------------------+
///
/// +-------------------+
/// | Pole names        |
/// +===================+
/// | :math:`a_0(980)`  |
/// +-------------------+
/// | :math:`a_0(1450)` |
/// +-------------------+
///
#[cfg(feature = "python")]
#[pyfunction(name = "KopfKMatrixA0", signature = (name, couplings, channel, mass, seed = None))]
pub fn py_kopf_kmatrix_a0(
    name: &str,
    couplings: [[PyParameterLike; 2]; 2],
    channel: usize,
    mass: PyMass,
    seed: Option<usize>,
) -> PyResult<PyExpression> {
    Ok(PyExpression(KopfKMatrixA0::new(
        name,
        array::from_fn(|i| array::from_fn(|j| couplings[i][j].clone().0)),
        channel,
        &mass.0,
        seed,
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
    fn test_a0_evaluation() {
        let dataset = Arc::new(test_dataset());
        let res_mass = Mass::new(["kshort1", "kshort2"]);
        let expr = KopfKMatrixA0::new(
            "a0",
            [
                [parameter("p0"), parameter("p1")],
                [parameter("p2"), parameter("p3")],
            ],
            1,
            &res_mass,
            None,
        )
        .unwrap();
        let evaluator = expr.load(&dataset).unwrap();

        let result = evaluator.evaluate(&[0.1, 0.2, 0.3, 0.4]);

        assert_relative_eq!(result[0].re, -0.8002759157259999);
        assert_relative_eq!(result[0].im, -0.1359306632058216);
    }

    #[test]
    fn test_a0_gradient() {
        let dataset = Arc::new(test_dataset());
        let res_mass = Mass::new(["kshort1", "kshort2"]);
        let expr = KopfKMatrixA0::new(
            "a0",
            [
                [parameter("p0"), parameter("p1")],
                [parameter("p2"), parameter("p3")],
            ],
            1,
            &res_mass,
            None,
        )
        .unwrap();
        let evaluator = expr.load(&dataset).unwrap();

        let result = evaluator.evaluate_gradient(&[0.1, 0.2, 0.3, 0.4]);

        assert_relative_eq!(result[0][0].re, 0.2906192438344459);
        assert_relative_eq!(result[0][0].im, -0.09989060459904309);
        assert_relative_eq!(result[0][1].re, -result[0][0].im);
        assert_relative_eq!(result[0][1].im, result[0][0].re);
        assert_relative_eq!(result[0][2].re, -1.313683875655594);
        assert_relative_eq!(result[0][2].im, 1.1380269958314373);
        assert_relative_eq!(result[0][3].re, -result[0][2].im);
        assert_relative_eq!(result[0][3].im, result[0][2].re);
    }

    #[test]
    fn test_a0_resample() {
        let res_mass = Mass::new(["kshort1", "kshort2"]);
        let _amp = KopfKMatrixA0::new(
            "a0",
            [
                [parameter("p0"), parameter("p1")],
                [parameter("p2"), parameter("p3")],
            ],
            1,
            &res_mass,
            Some(1),
        )
        .unwrap();
    }
}
