use nalgebra::{matrix, vector};
use nalgebra::{SMatrix, SVector};
use serde::{Deserialize, Serialize};
use std::array;

use laddu_core::{
    amplitudes::{Amplitude, AmplitudeID, ParameterLike},
    data::Event,
    resources::{Cache, ComplexVectorID, MatrixID, ParameterID, Parameters, Resources},
    utils::variables::{Mass, Variable},
    Complex, DVector, Float, LadduError,
};

#[cfg(feature = "python")]
use laddu_python::{
    amplitudes::{PyAmplitude, PyParameterLike},
    utils::variables::PyMass,
};
#[cfg(feature = "python")]
use pyo3::prelude::*;

use super::FixedKMatrix;

const G_A2: SMatrix<Float, 3, 2> = matrix![
     0.30073,  0.68567;
     0.21426,  0.12543;
    -0.09162,  0.00184
];
const C_A2: SMatrix<Float, 3, 3> = matrix![
    -0.40184,  0.00033, -0.08707;
     0.00033, -0.21416, -0.06193;
    -0.08707, -0.06193, -0.17435
];
const M_A2: SVector<Float, 2> = vector![1.30080, 1.75351];

const COV_A2: SMatrix<Float, 17, 17> = matrix![
    0.00060, -0.00005, -0.00122, -0.00100, -0.00008, -0.00010, -0.00028, 0.00034, 0.00019, 0.00000, -0.00047, -0.00055, 0.00000, 0.00000, -0.00058, 0.00003, 0.00001;
    -0.00005, 0.00816, 0.00029, -0.00049, 0.00034, -0.00229, -0.00261, -0.00100, -0.00096, 0.00000, 0.00217, 0.00244, 0.00000, 0.00000, 0.00184, 0.00007, 0.00229;
    -0.00122, 0.00029, 0.00268, 0.00201, 0.00041, 0.00014, 0.00062, -0.00073, -0.00066, 0.00000, 0.00123, 0.00125, 0.00000, 0.00000, 0.00115, -0.00006, 0.00005;
    -0.00100, -0.00049, 0.00201, 0.01740, 0.00084, 0.00528, 0.00101, -0.00210, -0.00078, 0.00000, -0.00084, -0.00051, 0.00000, 0.00000, 0.00061, -0.00013, 0.00121;
    -0.00008, 0.00034, 0.00041, 0.00084, 0.00125, 0.00228, 0.00004, 0.00011, -0.00064, 0.00000, -0.00040, -0.00053, 0.00000, 0.00000, -0.00140, -0.00001, 0.00039;
    -0.00010, -0.00229, 0.00014, 0.00528, 0.00228, 0.03808, 0.00004, 0.00572, -0.00027, 0.00000, -0.01025, -0.01121, 0.00000, 0.00000, -0.02187, -0.00012, 0.00255;
    -0.00028, -0.00261, 0.00062, 0.00101, 0.00004, 0.00004, 0.00244, -0.00150, -0.00109, 0.00000, 0.00202, 0.00174, 0.00000, 0.00000, 0.00128, -0.00009, -0.00102;
    0.00034, -0.00100, -0.00073, -0.00210, 0.00011, 0.00572, -0.00150, 0.00505, 0.00221, 0.00000, -0.00834, -0.00676, 0.00000, 0.00000, -0.00567, -0.00004, 0.00058;
    0.00019, -0.00096, -0.00066, -0.00078, -0.00064, -0.00027, -0.00109, 0.00221, 0.00383, 0.00000, -0.00519, -0.00452, 0.00000, 0.00000, -0.00098, 0.00005, -0.00013;
    0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000;
    -0.00047, 0.00217, 0.00123, -0.00084, -0.00040, -0.01025, 0.00202, -0.00834, -0.00519, 0.00000, 0.02304, 0.01445, 0.00000, 0.00000, 0.00990, -0.00002, -0.00114;
    -0.00055, 0.00244, 0.00125, -0.00051, -0.00053, -0.01121, 0.00174, -0.00676, -0.00452, 0.00000, 0.01445, 0.01313, 0.00000, 0.00000, 0.01033, 0.00001, -0.00077;
    0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000;
    0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000;
    -0.00058, 0.00184, 0.00115, 0.00061, -0.00140, -0.02187, 0.00128, -0.00567, -0.00098, 0.00000, 0.00990, 0.01033, 0.00000, 0.00000, 0.02658, 0.00003, -0.00129;
    0.00003, 0.00007, -0.00006, -0.00013, -0.00001, -0.00012, -0.00009, -0.00004, 0.00005, 0.00000, -0.00002, 0.00001, 0.00000, 0.00000, 0.00003, 0.00001, 0.00002;
    0.00001, 0.00229, 0.00005, 0.00121, 0.00039, 0.00255, -0.00102, 0.00058, -0.00013, 0.00000, -0.00114, -0.00077, 0.00000, 0.00000, -0.00129, 0.00002, 0.00142;
];

/// A K-matrix parameterization for $`a_2`$ particles described by Kopf et al.[^1] with fixed couplings and mass poles
/// (free production couplings only).
///
/// [^1]: Kopf, B., Albrecht, M., Koch, H., Küßner, M., Pychy, J., Qin, X., & Wiedner, U. (2021). Investigation of the lightest hybrid meson candidate with a coupled-channel analysis of $`\bar{p}p`$-, $`\pi^- p`$- and $`\pi \pi`$-Data. The European Physical Journal C, 81(12). [doi:10.1140/epjc/s10052-021-09821-2](https://doi.org/10.1140/epjc/s10052-021-09821-2)
#[derive(Clone, Serialize, Deserialize)]
pub struct KopfKMatrixA2 {
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

impl KopfKMatrixA2 {
    /// Construct a new [`KopfKMatrixA2`] with the given name, production couplings, channel,
    /// and input mass.
    ///
    /// | Channel index | Channel |
    /// | ------------- | ------- |
    /// | 0             | $`\pi\eta`$ |
    /// | 1             | $`K\bar{K}`$ |
    /// | 2             | $`\pi\eta'`$ |
    ///
    /// | Pole names |
    /// | ---------- |
    /// | $`a_2(1320)`$ |
    /// | $`a_2(1700)`$ |
    pub fn new(
        name: &str,
        couplings: [[ParameterLike; 2]; 2],
        channel: usize,
        mass: &Mass,
        seed: Option<usize>,
    ) -> Box<Self> {
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
                G_A2,
                C_A2,
                vector![0.1349768, 0.493677, 0.1349768],
                vector![0.547862, 0.497611, 0.95778],
                M_A2,
                None,
                2,
                COV_A2,
                seed,
            ),
            couplings_real,
            couplings_imag,
            couplings_indices_real: [ParameterID::default(); 2],
            couplings_indices_imag: [ParameterID::default(); 2],
            ikc_cache_index: ComplexVectorID::default(),
            p_vec_cache_index: MatrixID::default(),
        }
        .into()
    }
}

#[typetag::serde]
impl Amplitude for KopfKMatrixA2 {
    fn register(&mut self, resources: &mut Resources) -> Result<AmplitudeID, LadduError> {
        for i in 0..self.couplings_indices_real.len() {
            self.couplings_indices_real[i] = resources.register_parameter(&self.couplings_real[i]);
            self.couplings_indices_imag[i] = resources.register_parameter(&self.couplings_imag[i]);
        }
        self.ikc_cache_index = resources
            .register_complex_vector(Some(&format!("KopfKMatrixA2<{}> ikc_vec", self.name)));
        self.p_vec_cache_index =
            resources.register_matrix(Some(&format!("KopfKMatrixA2<{}> p_vec", self.name)));
        resources.register_amplitude(&self.name)
    }

    fn precompute(&self, event: &Event, cache: &mut Cache) {
        let s = self.mass.value(event).powi(2);
        cache.store_complex_vector(
            self.ikc_cache_index,
            self.constants.ikc_inv_vec(s, self.channel),
        );
        cache.store_matrix(self.p_vec_cache_index, self.constants.p_vec_constants(s));
    }

    fn compute(&self, parameters: &Parameters, _event: &Event, cache: &Cache) -> Complex<Float> {
        let betas = SVector::from_fn(|i, _| {
            Complex::new(
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
        _event: &Event,
        cache: &Cache,
        gradient: &mut DVector<Complex<Float>>,
    ) {
        let ikc_inv_vec = cache.get_complex_vector(self.ikc_cache_index);
        let p_vec_constants = cache.get_matrix(self.p_vec_cache_index);
        let internal_gradient = FixedKMatrix::compute_gradient(&ikc_inv_vec, &p_vec_constants);
        for i in 0..2 {
            if let ParameterID::Parameter(index) = self.couplings_indices_real[i] {
                gradient[index] = internal_gradient[i];
            }
            if let ParameterID::Parameter(index) = self.couplings_indices_imag[i] {
                gradient[index] = Complex::<Float>::I * internal_gradient[i];
            }
        }
    }
}

/// A fixed K-Matrix Amplitude for :math:`a_2` mesons
///
/// This Amplitude follows the prescription of [Kopf]_ and fixes the K-Matrix to data
/// from that paper, leaving the couplings to the initial state free
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
/// +---------------+-------------------+
/// | Channel index | Channel           |
/// +===============+===================+
/// | 0             | :math:`\pi\eta`   |
/// +---------------+-------------------+
/// | 1             | :math:`K\bar{K}`  |
/// +---------------+-------------------+
/// | 2             | :math:`\pi\eta'`  |
/// +---------------+-------------------+
///
/// +-------------------+
/// | Pole names        |
/// +===================+
/// | :math:`a_2(1320)` |
/// +-------------------+
/// | :math:`a_2(1700)` |
/// +-------------------+
///
#[cfg(feature = "python")]
#[pyfunction(name = "KopfKMatrixA2", signature = (name, couplings, channel, mass, seed = None))]
pub fn py_kopf_kmatrix_a2(
    name: &str,
    couplings: [[PyParameterLike; 2]; 2],
    channel: usize,
    mass: PyMass,
    seed: Option<usize>,
) -> PyAmplitude {
    PyAmplitude(KopfKMatrixA2::new(
        name,
        array::from_fn(|i| array::from_fn(|j| couplings[i][j].clone().0)),
        channel,
        &mass.0,
        seed,
    ))
}

#[cfg(test)]
mod tests {
    // Note: These tests are not exhaustive, they only check one channel
    use std::sync::Arc;

    use super::*;
    use approx::assert_relative_eq;
    use laddu_core::{data::test_dataset, parameter, Manager, Mass};

    #[test]
    fn test_a2_evaluation() {
        let mut manager = Manager::default();
        let res_mass = Mass::new([2, 3]);
        let amp = KopfKMatrixA2::new(
            "a2",
            [
                [parameter("p0"), parameter("p1")],
                [parameter("p2"), parameter("p3")],
            ],
            1,
            &res_mass,
            None,
        );
        let aid = manager.register(amp).unwrap();

        let dataset = Arc::new(test_dataset());
        let expr = aid.into();
        let model = manager.model(&expr);
        let evaluator = model.load(&dataset);

        let result = evaluator.evaluate(&[0.1, 0.2, 0.3, 0.4]);

        assert_relative_eq!(result[0].re, -0.20926617, epsilon = Float::EPSILON.sqrt());
        assert_relative_eq!(result[0].im, -0.0985062, epsilon = Float::EPSILON.sqrt());
    }

    #[test]
    fn test_a2_gradient() {
        let mut manager = Manager::default();
        let res_mass = Mass::new([2, 3]);
        let amp = KopfKMatrixA2::new(
            "a2",
            [
                [parameter("p0"), parameter("p1")],
                [parameter("p2"), parameter("p3")],
            ],
            1,
            &res_mass,
            None,
        );
        let aid = manager.register(amp).unwrap();

        let dataset = Arc::new(test_dataset());
        let expr = aid.into();
        let model = manager.model(&expr);
        let evaluator = model.load(&dataset);

        let result = evaluator.evaluate_gradient(&[0.1, 0.2, 0.3, 0.4]);

        assert_relative_eq!(result[0][0].re, -0.5756896, epsilon = Float::EPSILON.cbrt());
        assert_relative_eq!(result[0][0].im, 0.9398863, epsilon = Float::EPSILON.cbrt());
        assert_relative_eq!(result[0][1].re, -result[0][0].im);
        assert_relative_eq!(result[0][1].im, result[0][0].re);
        assert_relative_eq!(result[0][2].re, -0.0811143, epsilon = Float::EPSILON.cbrt());
        assert_relative_eq!(result[0][2].im, -0.1522787, epsilon = Float::EPSILON.cbrt());
        assert_relative_eq!(result[0][3].re, -result[0][2].im);
        assert_relative_eq!(result[0][3].im, result[0][2].re);
    }
}
