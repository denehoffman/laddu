use super::FixedKMatrix;
use laddu_core::{
    amplitudes::{Amplitude, AmplitudeID, ParameterLike},
    resources::{CacheRow, ParameterID, Parameters, Resources},
    ExprID, LadduResult,
};
#[cfg(feature = "python")]
use laddu_python::amplitudes::{PyAmplitude, PyParameterLike};
use nalgebra::{matrix, vector, DVector, SMatrix, SVector};
use num::complex::Complex64;
use polars::prelude::Expr;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3_polars::PyExpr;
use serde::{Deserialize, Serialize};
use std::array;

const G_A2: SMatrix<f64, 3, 2> = matrix![
     0.30073,  0.68567;
     0.21426,  0.12543;
    -0.09162,  0.00184
];
const C_A2: SMatrix<f64, 3, 3> = matrix![
    -0.40184,  0.00033, -0.08707;
     0.00033, -0.21416, -0.06193;
    -0.08707, -0.06193, -0.17435
];
const M_A2: SVector<f64, 2> = vector![1.30080, 1.75351];

const COV_A2: SMatrix<f64, 17, 17> = matrix![
    0.00059780890382, -0.00004650414533, -0.00122139959949, -0.00100412815497, -0.00007866505645, -0.00009724104078, -0.00028402260113, 0.00033675733167, 0.00019309287802, 0.00000000000000, -0.00047453425354, -0.00055251965792, 0.00000000000000, 0.00000000000000, -0.00058162571106, 0.00002528793084, 0.00001313246125;
    -0.00004650414533, 0.00816345485311, 0.00028803035681, -0.00048905295096, 0.00033678639248, -0.00228837157908, -0.00261223341432, -0.00099695946129, -0.00095551450363, 0.00000000000000, 0.00216701717647, 0.00244247665991, 0.00000000000000, 0.00000000000000, 0.00184115829632, 0.00007185677441, 0.00229344038589;
    -0.00122139959949, 0.00028803035681, 0.00267823345505, 0.00200797132506, 0.00040938760161, 0.00014339020910, 0.00061791369793, -0.00073299967428, -0.00065626923226, 0.00000000000000, 0.00123438592734, 0.00125296175370, 0.00000000000000, 0.00000000000000, 0.00115172549976, -0.00006026929442, 0.00004754900279;
    -0.00100412815497, -0.00048905295096, 0.00200797132506, 0.01739528592062, 0.00083809165951, 0.00527992911733, 0.00100846854145, -0.00209658575996, -0.00077639236344, 0.00000000000000, -0.00084191312250, -0.00051100874285, 0.00000000000000, 0.00000000000000, 0.00060860962737, -0.00012813381932, 0.00120975110797;
    -0.00007866505645, 0.00033678639248, 0.00040938760161, 0.00083809165951, 0.00125020341499, 0.00228459594661, 0.00003627364401, 0.00011191202086, -0.00064171086926, 0.00000000000000, -0.00040124514834, -0.00052872757432, 0.00000000000000, 0.00000000000000, -0.00140322631874, -0.00000660258065, 0.00039202336349;
    -0.00009724104078, -0.00228837157908, 0.00014339020910, 0.00527992911733, 0.00228459594661, 0.03807860275785, 0.00004339578717, 0.00572394902368, -0.00027397007157, 0.00000000000000, -0.01025277758527, -0.01121332654228, 0.00000000000000, 0.00000000000000, -0.02186912382587, -0.00011760761236, 0.00254816916832;
    -0.00028402260113, -0.00261223341432, 0.00061791369793, 0.00100846854145, 0.00003627364401, 0.00004339578717, 0.00244476914478, -0.00150173579203, -0.00108808942209, 0.00000000000000, 0.00201742478074, 0.00174014427043, 0.00000000000000, 0.00000000000000, 0.00127671488574, -0.00009144859346, -0.00101555125535;
    0.00033675733167, -0.00099695946129, -0.00073299967428, -0.00209658575996, 0.00011191202086, 0.00572394902368, -0.00150173579203, 0.00505371782152, 0.00221007500202, 0.00000000000000, -0.00834279099006, -0.00675597670314, 0.00000000000000, 0.00000000000000, -0.00567465418788, -0.00003696696237, 0.00057688731760;
    0.00019309287802, -0.00095551450363, -0.00065626923226, -0.00077639236344, -0.00064171086926, -0.00027397007157, -0.00108808942209, 0.00221007500202, 0.00382723590287, 0.00000000000000, -0.00518698951963, -0.00452170820191, 0.00000000000000, 0.00000000000000, -0.00097524015944, 0.00005216866839, -0.00013320909664;
    0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000;
    -0.00047453425354, 0.00216701717647, 0.00123438592734, -0.00084191312250, -0.00040124514834, -0.01025277758527, 0.00201742478074, -0.00834279099006, -0.00518698951963, 0.00000000000000, 0.02304114324508, 0.01444631382928, 0.00000000000000, 0.00000000000000, 0.00990420809879, -0.00001763810617, -0.00113889702043;
    -0.00055251965792, 0.00244247665991, 0.00125296175370, -0.00051100874285, -0.00052872757432, -0.01121332654228, 0.00174014427043, -0.00675597670314, -0.00452170820191, 0.00000000000000, 0.01444631382928, 0.01313065284683, 0.00000000000000, 0.00000000000000, 0.01032940622768, 0.00001091475272, -0.00076628899003;
    0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000;
    0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000;
    -0.00058162571106, 0.00184115829632, 0.00115172549976, 0.00060860962737, -0.00140322631874, -0.02186912382587, 0.00127671488574, -0.00567465418788, -0.00097524015944, 0.00000000000000, 0.00990420809879, 0.01032940622768, 0.00000000000000, 0.00000000000000, 0.02657589618818, 0.00003287193182, -0.00128520134838;
    0.00002528793084, 0.00007185677441, -0.00006026929442, -0.00012813381932, -0.00000660258065, -0.00011760761236, -0.00009144859346, -0.00003696696237, 0.00005216866839, 0.00000000000000, -0.00001763810617, 0.00001091475272, 0.00000000000000, 0.00000000000000, 0.00003287193182, 0.00001389727103, 0.00002074066023;
    0.00001313246125, 0.00229344038589, 0.00004754900279, 0.00120975110797, 0.00039202336349, 0.00254816916832, -0.00101555125535, 0.00057688731760, -0.00013320909664, 0.00000000000000, -0.00113889702043, -0.00076628899003, 0.00000000000000, 0.00000000000000, -0.00128520134838, 0.00002074066023, 0.00142306642239;
];

/// A K-matrix parameterization for $`a_2`$ particles described by Kopf et al.[^1] with fixed couplings and mass poles
/// (free production couplings only).
///
/// [^1]: Kopf, B., Albrecht, M., Koch, H., Küßner, M., Pychy, J., Qin, X., & Wiedner, U. (2021). Investigation of the lightest hybrid meson candidate with a coupled-channel analysis of $`\bar{p}p`$-, $`\pi^- p`$- and $`\pi \pi`$-Data. The European Physical Journal C, 81(12). [doi:10.1140/epjc/s10052-021-09821-2](https://doi.org/10.1140/epjc/s10052-021-09821-2)
#[derive(Clone, Serialize, Deserialize)]
pub struct KopfKMatrixA2 {
    name: String,
    channel: usize,
    mass: Expr,
    constants: FixedKMatrix<3, 2>,
    couplings_real: [ParameterLike; 2],
    couplings_imag: [ParameterLike; 2],
    couplings_indices_real: [ParameterID; 2],
    couplings_indices_imag: [ParameterID; 2],
    ikc_cache_index: ExprID,
    p_vec_cache_index: ExprID,
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
        mass: &Expr,
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
            ikc_cache_index: Default::default(),
            p_vec_cache_index: Default::default(),
        }
        .into()
    }
}

#[typetag::serde]
impl Amplitude for KopfKMatrixA2 {
    fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID> {
        for i in 0..self.couplings_indices_real.len() {
            self.couplings_indices_real[i] = resources.register_parameter(&self.couplings_real[i]);
            self.couplings_indices_imag[i] = resources.register_parameter(&self.couplings_imag[i]);
        }
        let s = self.mass.clone().pow(2);
        self.ikc_cache_index = resources.register_cvector(
            format!("KopfKMatrixA2<{}> ikc_vec", self.name).into(),
            self.constants.ikc_inv_vec_expr(s.clone(), self.channel),
        )?;
        self.p_vec_cache_index = resources.register_matrix(
            format!("KopfKMatrixA2<{}> p_vec", self.name).into(),
            self.constants.p_vec_constants_expr(s),
        )?;
        resources.register_amplitude(&self.name)
    }

    fn compute(&self, parameters: &Parameters, cache_row: &CacheRow) -> Complex64 {
        let betas = SVector::from_fn(|i, _| {
            Complex64::new(
                parameters.get(self.couplings_indices_real[i]),
                parameters.get(self.couplings_indices_imag[i]),
            )
        });
        let ikc_inv_vec = cache_row.get_cvector::<3>(self.ikc_cache_index);
        let p_vec_constants = cache_row.get_matrix::<3, 2>(self.p_vec_cache_index);
        FixedKMatrix::compute(&betas, &ikc_inv_vec, &p_vec_constants)
    }

    fn compute_gradient(
        &self,
        _parameters: &Parameters,
        cache_row: &CacheRow,
        gradient: &mut DVector<Complex64>,
    ) {
        let ikc_inv_vec = cache_row.get_cvector::<3>(self.ikc_cache_index);
        let p_vec_constants = cache_row.get_matrix::<3, 2>(self.p_vec_cache_index);
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
    mass: Bound<PyAny>,
    seed: Option<usize>,
) -> PyResult<PyAmplitude> {
    Ok(PyAmplitude(KopfKMatrixA2::new(
        name,
        array::from_fn(|i| array::from_fn(|j| couplings[i][j].clone().0)),
        channel,
        &mass.extract::<PyExpr>()?.0,
        seed,
    )))
}

#[cfg(test)]
mod tests {
    // Note: These tests are not exhaustive, they only check one channel
    use super::*;
    use approx::assert_relative_eq;
    use laddu_core::{data::test_dataset, mass, parameter, Manager};

    #[test]
    fn test_a2_evaluation() {
        let mut manager = Manager::default();
        let res_mass = mass(["kshort1", "kshort2"]);
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

        let dataset = test_dataset();
        let expr = aid.into();
        let model = manager.model(&expr);
        let evaluator = model.load(&dataset).unwrap();

        let result = evaluator.evaluate(&[0.1, 0.2, 0.3, 0.4]).unwrap();

        assert_relative_eq!(result[0].re, -0.20926617, epsilon = f64::EPSILON.sqrt());
        assert_relative_eq!(result[0].im, -0.0985062, epsilon = f64::EPSILON.sqrt());
    }

    #[test]
    fn test_a2_gradient() {
        let mut manager = Manager::default();
        let res_mass = mass(["kshort1", "kshort2"]);
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

        let dataset = test_dataset();
        let expr = aid.into();
        let model = manager.model(&expr);
        let evaluator = model.load(&dataset).unwrap();

        let result = evaluator.evaluate_gradient(&[0.1, 0.2, 0.3, 0.4]).unwrap();

        assert_relative_eq!(result[0][0].re, -0.5756896, epsilon = f64::EPSILON.cbrt());
        assert_relative_eq!(result[0][0].im, 0.9398863, epsilon = f64::EPSILON.cbrt());
        assert_relative_eq!(result[0][1].re, -result[0][0].im);
        assert_relative_eq!(result[0][1].im, result[0][0].re);
        assert_relative_eq!(result[0][2].re, -0.0811143, epsilon = f64::EPSILON.cbrt());
        assert_relative_eq!(result[0][2].im, -0.1522787, epsilon = f64::EPSILON.cbrt());
        assert_relative_eq!(result[0][3].re, -result[0][2].im);
        assert_relative_eq!(result[0][3].im, result[0][2].re);
    }

    #[test]
    fn test_a2_resample() {
        let res_mass = mass(["kshort1", "kshort2"]);
        let _amp = KopfKMatrixA2::new(
            "a2",
            [
                [parameter("p0"), parameter("p1")],
                [parameter("p2"), parameter("p3")],
            ],
            1,
            &res_mass,
            Some(1),
        );
    }
}
