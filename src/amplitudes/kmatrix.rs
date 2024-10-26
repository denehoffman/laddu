use std::array;

use nalgebra::{matrix, vector, DVector};
use nalgebra::{SMatrix, SVector};
use num::traits::ConstOne;
use num::traits::FloatConst;
use num::Complex;

use crate::LadduError;
use crate::{
    amplitudes::{AmplitudeID, ParameterLike},
    data::Event,
    resources::{Cache, ComplexVectorID, MatrixID, ParameterID, Parameters, Resources},
    utils::{
        functions::{blatt_weisskopf, chi_plus, rho},
        variables::{Mass, Variable},
    },
    Float,
};

use super::Amplitude;

/// An Adler zero term used in a K-matrix.
#[derive(Copy, Clone, Debug)]
pub struct AdlerZero {
    /// The zero position $`s_0`$.
    pub s_0: Float,
    /// The normalization factor $`s_\text{norm}`$.
    pub s_norm: Float,
}

/// Methods for computing various parts of a K-matrix with fixed couplings and mass poles.
#[derive(Clone, Debug)]
pub struct FixedKMatrix<const CHANNELS: usize, const RESONANCES: usize> {
    g: SMatrix<Float, CHANNELS, RESONANCES>,
    c: SMatrix<Float, CHANNELS, CHANNELS>,
    m1s: SVector<Float, CHANNELS>,
    m2s: SVector<Float, CHANNELS>,
    mrs: SVector<Float, RESONANCES>,
    adler_zero: Option<AdlerZero>,
    l: usize,
}
impl<const CHANNELS: usize, const RESONANCES: usize> FixedKMatrix<CHANNELS, RESONANCES> {
    fn c_mat(&self, s: Float) -> SMatrix<Complex<Float>, CHANNELS, CHANNELS> {
        SMatrix::from_diagonal(&SVector::from_fn(|i, _| {
            let m1 = self.m1s[i];
            let m2 = self.m2s[i];
            ((rho(s, m1, m2)
                * Complex::ln(
                    (chi_plus(s, m1, m2) + rho(s, m1, m2)) / (chi_plus(s, m1, m2) - rho(s, m1, m2)),
                ))
                - (chi_plus(s, m1, m2) * ((m2 - m1) / (m1 + m2)) * Float::ln(m2 / m1)))
                / Float::PI()
        }))
    }
    fn barrier_mat(&self, s: Float) -> SMatrix<Float, CHANNELS, RESONANCES> {
        let m0 = Float::sqrt(s);
        SMatrix::from_fn(|i, a| {
            let m1 = self.m1s[i];
            let m2 = self.m2s[i];
            let mr = self.mrs[a];
            blatt_weisskopf(m0, m1, m2, self.l) / blatt_weisskopf(mr, m1, m2, self.l)
        })
    }
    fn product_of_poles(&self, s: Float) -> Float {
        self.mrs.map(|m| m.powi(2) - s).product()
    }
    fn product_of_poles_except_one(&self, s: Float, a_i: usize) -> Float {
        self.mrs
            .iter()
            .enumerate()
            .filter_map(|(a_j, m_j)| {
                if a_j != a_i {
                    Some(m_j.powi(2) - s)
                } else {
                    None
                }
            })
            .product()
    }

    fn k_mat(&self, s: Float) -> SMatrix<Complex<Float>, CHANNELS, CHANNELS> {
        let bf = self.barrier_mat(s);
        SMatrix::from_fn(|i, j| {
            self.adler_zero
                .map_or(Float::ONE, |az| (s - az.s_0) / az.s_norm)
                * (0..RESONANCES)
                    .map(|a| {
                        Complex::from(
                            bf[(i, a)] * bf[(j, a)] * self.g[(i, a)] * self.g[(j, a)]
                                + (self.c[(i, j)] * (self.mrs[a].powi(2) - s)),
                        ) * self.product_of_poles_except_one(s, a)
                    })
                    .sum::<Complex<Float>>()
        })
    }

    fn ikc_inv_vec(&self, s: Float, channel: usize) -> SVector<Complex<Float>, CHANNELS> {
        let i_mat: SMatrix<Complex<Float>, CHANNELS, CHANNELS> = SMatrix::identity();
        let k_mat = self.k_mat(s);
        let c_mat = self.c_mat(s);
        let ikc_mat = i_mat.scale(self.product_of_poles(s)) + k_mat * c_mat;
        let ikc_inv_mat = ikc_mat.try_inverse().unwrap();
        ikc_inv_mat.row(channel).transpose()
    }

    fn p_vec_constants(&self, s: Float) -> SMatrix<Float, CHANNELS, RESONANCES> {
        let barrier_mat = self.barrier_mat(s);
        SMatrix::from_fn(|i, a| {
            barrier_mat[(i, a)] * self.g[(i, a)] * self.product_of_poles_except_one(s, a)
        })
    }

    fn compute(
        betas: &SVector<Complex<Float>, RESONANCES>,
        ikc_inv_vec: &SVector<Complex<Float>, CHANNELS>,
        p_vec_constants: &SMatrix<Float, CHANNELS, RESONANCES>,
    ) -> Complex<Float> {
        let p_vec: SVector<Complex<Float>, CHANNELS> = SVector::from_fn(|j, _| {
            (0..RESONANCES)
                .map(|a| betas[a] * p_vec_constants[(j, a)])
                .sum()
        });
        ikc_inv_vec.dot(&p_vec)
    }

    fn compute_gradient(
        ikc_inv_vec: &SVector<Complex<Float>, CHANNELS>,
        p_vec_constants: &SMatrix<Float, CHANNELS, RESONANCES>,
    ) -> DVector<Complex<Float>> {
        DVector::from_fn(RESONANCES, |a, _| {
            (0..RESONANCES)
                .map(|j| ikc_inv_vec[j] * p_vec_constants[(j, a)])
                .sum()
        })
    }
}

/// A K-matrix parameterization for $`f_0`$ particles described by Kopf et al.[^1] with fixed couplings and mass poles
/// (free production couplings only).
///
/// [^1]: Kopf, B., Albrecht, M., Koch, H., Küßner, M., Pychy, J., Qin, X., & Wiedner, U. (2021). Investigation of the lightest hybrid meson candidate with a coupled-channel analysis of $`\bar{p}p`$-, $`\pi^- p`$- and $`\pi \pi`$-Data. The European Physical Journal C, 81(12). [doi:10.1140/epjc/s10052-021-09821-2](https://doi.org/10.1140/epjc/s10052-021-09821-2)
#[derive(Clone)]
pub struct KopfKMatrixF0 {
    name: String,
    channel: usize,
    mass: Mass,
    constants: FixedKMatrix<5, 5>,
    couplings_real: [ParameterLike; 5],
    couplings_imag: [ParameterLike; 5],
    couplings_indices_real: [ParameterID; 5],
    couplings_indices_imag: [ParameterID; 5],
    ikc_cache_index: ComplexVectorID<5>,
    p_vec_cache_index: MatrixID<5, 5>,
}

impl KopfKMatrixF0 {
    /// Construct a new [`KopfKMatrixF0`] with the given name, production couplings, channel,
    /// and input mass.
    ///
    /// | Channel index | Channel |
    /// | ------------- | ------- |
    /// | 0             | $`\pi\pi`$ |
    /// | 1             | $`2\pi 2\pi`$ |
    /// | 2             | $`K\bar{K}`$ |
    /// | 3             | $`\eta\eta`$ |
    /// | 4             | $`\eta\eta'`$ |
    ///
    /// | Pole names |
    /// | ---------- |
    /// | $`f_0(500)`$ |
    /// | $`f_0(980)`$ |
    /// | $`f_0(1370)`$ |
    /// | $`f_0(1500)`$ |
    /// | $`f_0(1710)`$ |
    pub fn new(
        name: &str,
        couplings: [[ParameterLike; 2]; 5],
        channel: usize,
        mass: &Mass,
    ) -> Box<Self> {
        let mut couplings_real: [ParameterLike; 5] = array::from_fn(|_| ParameterLike::default());
        let mut couplings_imag: [ParameterLike; 5] = array::from_fn(|_| ParameterLike::default());
        for i in 0..5 {
            couplings_real[i] = couplings[i][0].clone();
            couplings_imag[i] = couplings[i][1].clone();
        }
        Self {
            name: name.to_string(),
            channel,
            mass: mass.clone(),
            constants: FixedKMatrix {
                g: matrix![
                     0.74987,  0.06401, -0.23417,  0.01270, -0.14242;
                    -0.01257,  0.00204, -0.01032,  0.26700,  0.22780;
                     0.27536,  0.77413,  0.72283,  0.09214,  0.15981;
                    -0.15102,  0.50999,  0.11934,  0.02742,  0.16272;
                     0.36103,  0.13112,  0.36792, -0.04025, -0.17397

                ],
                c: matrix![
                     0.03728,  0.00000, -0.01398, -0.02203,  0.01397;
                     0.00000,  0.00000,  0.00000,  0.00000,  0.00000;
                    -0.01398,  0.00000,  0.02349,  0.03101, -0.04003;
                    -0.02203,  0.00000,  0.03101, -0.13769, -0.06722;
                     0.01397,  0.00000, -0.04003, -0.06722, -0.28401
                ],
                m1s: vector![0.1349768, 2.0 * 0.1349768, 0.493677, 0.547862, 0.547862],
                m2s: vector![0.1349768, 2.0 * 0.1349768, 0.497611, 0.547862, 0.95778],
                mrs: vector![0.51461, 0.90630, 1.23089, 1.46104, 1.69611],
                adler_zero: Some(AdlerZero {
                    s_0: 0.0091125,
                    s_norm: 1.0,
                }),
                l: 0,
            },
            couplings_real,
            couplings_imag,
            couplings_indices_real: [ParameterID::default(); 5],
            couplings_indices_imag: [ParameterID::default(); 5],
            ikc_cache_index: ComplexVectorID::default(),
            p_vec_cache_index: MatrixID::default(),
        }
        .into()
    }
}

impl Amplitude for KopfKMatrixF0 {
    fn register(&mut self, resources: &mut Resources) -> Result<AmplitudeID, LadduError> {
        for i in 0..self.couplings_indices_real.len() {
            self.couplings_indices_real[i] = resources.register_parameter(&self.couplings_real[i]);
            self.couplings_indices_imag[i] = resources.register_parameter(&self.couplings_imag[i]);
        }
        self.ikc_cache_index = resources
            .register_complex_vector(Some(&format!("KopfKMatrixF0<{}> ikc_vec", self.name)));
        self.p_vec_cache_index =
            resources.register_matrix(Some(&format!("KopfKMatrixF0<{}> p_vec", self.name)));
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
        for i in 0..5 {
            if let ParameterID::Parameter(index) = self.couplings_indices_real[i] {
                gradient[index] = internal_gradient[i];
            }
            if let ParameterID::Parameter(index) = self.couplings_indices_imag[i] {
                gradient[index] = Complex::<Float>::I * internal_gradient[i];
            }
        }
    }
}

/// A K-matrix parameterization for $`f_2`$ particles described by Kopf et al.[^1] with fixed couplings and mass poles
/// (free production couplings only).
///
/// [^1]: Kopf, B., Albrecht, M., Koch, H., Küßner, M., Pychy, J., Qin, X., & Wiedner, U. (2021). Investigation of the lightest hybrid meson candidate with a coupled-channel analysis of $`\bar{p}p`$-, $`\pi^- p`$- and $`\pi \pi`$-Data. The European Physical Journal C, 81(12). [doi:10.1140/epjc/s10052-021-09821-2](https://doi.org/10.1140/epjc/s10052-021-09821-2)
#[derive(Clone)]
pub struct KopfKMatrixF2 {
    name: String,
    channel: usize,
    mass: Mass,
    constants: FixedKMatrix<4, 4>,
    couplings_real: [ParameterLike; 4],
    couplings_imag: [ParameterLike; 4],
    couplings_indices_real: [ParameterID; 4],
    couplings_indices_imag: [ParameterID; 4],
    ikc_cache_index: ComplexVectorID<4>,
    p_vec_cache_index: MatrixID<4, 4>,
}

impl KopfKMatrixF2 {
    /// Construct a new [`KopfKMatrixF2`] with the given name, production couplings, channel,
    /// and input mass.
    ///
    /// | Channel index | Channel |
    /// | ------------- | ------- |
    /// | 0             | $`\pi\pi`$ |
    /// | 1             | $`2\pi 2\pi`$ |
    /// | 2             | $`K\bar{K}`$ |
    /// | 3             | $`\eta\eta`$ |
    ///
    /// | Pole names |
    /// | ---------- |
    /// | $`f_2(1270)`$ |
    /// | $`f_2'(1525)`$ |
    /// | $`f_2(1810)`$ |
    /// | $`f_2(1950)`$ |
    pub fn new(
        name: &str,
        couplings: [[ParameterLike; 2]; 4],
        channel: usize,
        mass: &Mass,
    ) -> Box<Self> {
        let mut couplings_real: [ParameterLike; 4] = array::from_fn(|_| ParameterLike::default());
        let mut couplings_imag: [ParameterLike; 4] = array::from_fn(|_| ParameterLike::default());
        for i in 0..4 {
            couplings_real[i] = couplings[i][0].clone();
            couplings_imag[i] = couplings[i][1].clone();
        }
        Self {
            name: name.to_string(),
            channel,
            mass: mass.clone(),
            constants: FixedKMatrix {
                g: matrix![
                     0.40033,  0.01820, -0.06709, -0.49924;
                     0.15479,  0.17300,  0.22941,  0.19295;
                    -0.08900,  0.32393, -0.43133,  0.27975;
                    -0.00113,  0.15256,  0.23721, -0.03987
                ],
                c: matrix![
                    -0.04319,  0.00000,  0.00984,  0.01028;
                     0.00000,  0.00000,  0.00000,  0.00000;
                     0.00984,  0.00000, -0.07344,  0.05533;
                     0.01028,  0.00000,  0.05533, -0.05183
                ],
                m1s: vector![0.1349768, 2.0 * 0.1349768, 0.493677, 0.547862],
                m2s: vector![0.1349768, 2.0 * 0.1349768, 0.497611, 0.547862],
                mrs: vector![1.15299, 1.48359, 1.72923, 1.96700],
                adler_zero: None,
                l: 2,
            },
            couplings_real,
            couplings_imag,
            couplings_indices_real: [ParameterID::default(); 4],
            couplings_indices_imag: [ParameterID::default(); 4],
            ikc_cache_index: ComplexVectorID::default(),
            p_vec_cache_index: MatrixID::default(),
        }
        .into()
    }
}

impl Amplitude for KopfKMatrixF2 {
    fn register(&mut self, resources: &mut Resources) -> Result<AmplitudeID, LadduError> {
        for i in 0..self.couplings_indices_real.len() {
            self.couplings_indices_real[i] = resources.register_parameter(&self.couplings_real[i]);
            self.couplings_indices_imag[i] = resources.register_parameter(&self.couplings_imag[i]);
        }
        self.ikc_cache_index = resources
            .register_complex_vector(Some(&format!("KopfKMatrixF2<{}> ikc_vec", self.name)));
        self.p_vec_cache_index =
            resources.register_matrix(Some(&format!("KopfKMatrixF2<{}> p_vec", self.name)));
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
        for i in 0..4 {
            if let ParameterID::Parameter(index) = self.couplings_indices_real[i] {
                gradient[index] = internal_gradient[i];
            }
            if let ParameterID::Parameter(index) = self.couplings_indices_imag[i] {
                gradient[index] = Complex::<Float>::I * internal_gradient[i];
            }
        }
    }
}

/// A K-matrix parameterization for $`a_0`$ particles described by Kopf et al.[^1] with fixed couplings and mass poles
/// (free production couplings only).
///
/// [^1]: Kopf, B., Albrecht, M., Koch, H., Küßner, M., Pychy, J., Qin, X., & Wiedner, U. (2021). Investigation of the lightest hybrid meson candidate with a coupled-channel analysis of $`\bar{p}p`$-, $`\pi^- p`$- and $`\pi \pi`$-Data. The European Physical Journal C, 81(12). [doi:10.1140/epjc/s10052-021-09821-2](https://doi.org/10.1140/epjc/s10052-021-09821-2)
#[derive(Clone)]
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
            constants: FixedKMatrix {
                g: matrix![
                     0.43215,  0.19000;
                    -0.28825,  0.43372
                ],
                c: matrix![
                     0.00000,  0.00000;
                     0.00000,  0.00000
                ],
                m1s: vector![0.1349768, 0.493677],
                m2s: vector![0.547862, 0.497611],
                mrs: vector![0.95395, 1.26767],
                adler_zero: None,
                l: 0,
            },
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

impl Amplitude for KopfKMatrixA0 {
    fn register(&mut self, resources: &mut Resources) -> Result<AmplitudeID, LadduError> {
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

/// A K-matrix parameterization for $`a_2`$ particles described by Kopf et al.[^1] with fixed couplings and mass poles
/// (free production couplings only).
///
/// [^1]: Kopf, B., Albrecht, M., Koch, H., Küßner, M., Pychy, J., Qin, X., & Wiedner, U. (2021). Investigation of the lightest hybrid meson candidate with a coupled-channel analysis of $`\bar{p}p`$-, $`\pi^- p`$- and $`\pi \pi`$-Data. The European Physical Journal C, 81(12). [doi:10.1140/epjc/s10052-021-09821-2](https://doi.org/10.1140/epjc/s10052-021-09821-2)
#[derive(Clone)]
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
            constants: FixedKMatrix {
                g: matrix![
                     0.30073,  0.68567;
                     0.21426,  0.12543;
                    -0.09162,  0.00184

                ],
                c: matrix![
                    -0.40184,  0.00033, -0.08707;
                     0.00033, -0.21416, -0.06193;
                    -0.08707, -0.06193, -0.17435
                ],
                m1s: vector![0.1349768, 0.493677, 0.1349768],
                m2s: vector![0.547862, 0.497611, 0.95778],
                mrs: vector![1.30080, 1.75351],
                adler_zero: None,
                l: 2,
            },
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

/// A K-matrix parameterization for $`\rho`$ particles described by Kopf et al.[^1] with fixed couplings and mass poles
/// (free production couplings only).
///
/// [^1]: Kopf, B., Albrecht, M., Koch, H., Küßner, M., Pychy, J., Qin, X., & Wiedner, U. (2021). Investigation of the lightest hybrid meson candidate with a coupled-channel analysis of $`\bar{p}p`$-, $`\pi^- p`$- and $`\pi \pi`$-Data. The European Physical Journal C, 81(12). [doi:10.1140/epjc/s10052-021-09821-2](https://doi.org/10.1140/epjc/s10052-021-09821-2)
#[derive(Clone)]
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
    /// | 1             | $`K\bar{K}`$ |
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
        .into()
    }
}

impl Amplitude for KopfKMatrixRho {
    fn register(&mut self, resources: &mut Resources) -> Result<AmplitudeID, LadduError> {
        for i in 0..self.couplings_indices_real.len() {
            self.couplings_indices_real[i] = resources.register_parameter(&self.couplings_real[i]);
            self.couplings_indices_imag[i] = resources.register_parameter(&self.couplings_imag[i]);
        }
        self.ikc_cache_index = resources
            .register_complex_vector(Some(&format!("KopfKMatrixRho<{}> ikc_vec", self.name)));
        self.p_vec_cache_index =
            resources.register_matrix(Some(&format!("KopfKMatrixRho<{}> p_vec", self.name)));
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

/// A K-matrix parameterization for the $`\pi_1`$ hybrid candidate described by Kopf et al.[^1] with fixed couplings and mass poles
/// (free production couplings only).
///
/// [^1]: Kopf, B., Albrecht, M., Koch, H., Küßner, M., Pychy, J., Qin, X., & Wiedner, U. (2021). Investigation of the lightest hybrid meson candidate with a coupled-channel analysis of $`\bar{p}p`$-, $`\pi^- p`$- and $`\pi \pi`$-Data. The European Physical Journal C, 81(12). [doi:10.1140/epjc/s10052-021-09821-2](https://doi.org/10.1140/epjc/s10052-021-09821-2)
#[derive(Clone)]
pub struct KopfKMatrixPi1 {
    name: String,
    channel: usize,
    mass: Mass,
    constants: FixedKMatrix<2, 1>,
    couplings_real: [ParameterLike; 1],
    couplings_imag: [ParameterLike; 1],
    couplings_indices_real: [ParameterID; 1],
    couplings_indices_imag: [ParameterID; 1],
    ikc_cache_index: ComplexVectorID<2>,
    p_vec_cache_index: MatrixID<2, 1>,
}

impl KopfKMatrixPi1 {
    /// Construct a new [`KopfKMatrixPi1`] with the given name, production couplings, channel,
    /// and input mass.
    ///
    /// | Channel index | Channel |
    /// | ------------- | ------- |
    /// | 0             | $`\pi\eta`$ |
    /// | 1             | $`\pi\eta'`$ |
    ///
    /// | Pole names |
    /// | ---------- |
    /// | $`\pi_1(1600)`$ |
    pub fn new(
        name: &str,
        couplings: [[ParameterLike; 2]; 1],
        channel: usize,
        mass: &Mass,
    ) -> Box<Self> {
        let mut couplings_real: [ParameterLike; 1] = array::from_fn(|_| ParameterLike::default());
        let mut couplings_imag: [ParameterLike; 1] = array::from_fn(|_| ParameterLike::default());
        for i in 0..1 {
            couplings_real[i] = couplings[i][0].clone();
            couplings_imag[i] = couplings[i][1].clone();
        }
        Self {
            name: name.to_string(),
            channel,
            mass: mass.clone(),
            constants: FixedKMatrix {
                g: matrix![
                     0.80564;
                     1.04595
                ],
                c: matrix![
                    1.05000,  0.15163;
                    0.15163, -0.24611
                ],
                m1s: vector![0.1349768, 0.1349768],
                m2s: vector![0.547862, 0.95778],
                mrs: vector![1.38552],
                adler_zero: None,
                l: 1,
            },
            couplings_real,
            couplings_imag,
            couplings_indices_real: [ParameterID::default(); 1],
            couplings_indices_imag: [ParameterID::default(); 1],
            ikc_cache_index: ComplexVectorID::default(),
            p_vec_cache_index: MatrixID::default(),
        }
        .into()
    }
}

impl Amplitude for KopfKMatrixPi1 {
    fn register(&mut self, resources: &mut Resources) -> Result<AmplitudeID, LadduError> {
        for i in 0..self.couplings_indices_real.len() {
            self.couplings_indices_real[i] = resources.register_parameter(&self.couplings_real[i]);
            self.couplings_indices_imag[i] = resources.register_parameter(&self.couplings_imag[i]);
        }
        self.ikc_cache_index = resources
            .register_complex_vector(Some(&format!("KopfKMatrixPi1<{}> ikc_vec", self.name)));
        self.p_vec_cache_index =
            resources.register_matrix(Some(&format!("KopfKMatrixPi1<{}> p_vec", self.name)));
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
        if let ParameterID::Parameter(index) = self.couplings_indices_real[0] {
            gradient[index] = internal_gradient[0];
        }
        if let ParameterID::Parameter(index) = self.couplings_indices_imag[0] {
            gradient[index] = Complex::<Float>::I * internal_gradient[0];
        }
    }
}
