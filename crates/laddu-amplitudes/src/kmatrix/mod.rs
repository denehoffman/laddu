use fastrand::Rng;
use fastrand_contrib::RngExt;
use laddu_core::{
    utils::functions::{blatt_weisskopf, chi_plus, rho},
    Float,
};
use nalgebra::{Cholesky, DMatrix, DVector, SMatrix, SVector};
use num::{
    traits::{ConstOne, FloatConst},
    Complex,
};
use serde::{Deserialize, Serialize};

fn sample_normal<const PARAMETERS: usize>(
    mu: SVector<Float, PARAMETERS>,
    cov: SMatrix<Float, PARAMETERS, PARAMETERS>,
    rng: &mut Rng,
) -> SVector<Float, PARAMETERS> {
    #[cfg(not(feature = "f32"))]
    let mut normal = || rng.f64_normal(0.0, 1.0);
    #[cfg(feature = "f32")]
    let mut normal = || rng.f32_normal(0.0, 1.0);
    let active: Vec<usize> = (0..mu.len())
        .filter(|&i| cov.row(i).iter().any(|&x| x != 0.0))
        .collect();
    if active.is_empty() {
        return mu;
    }
    let mu_active = DVector::from_iterator(active.len(), active.iter().map(|&i| mu[i]));
    let cov_active = DMatrix::from_fn(active.len(), active.len(), |i, j| {
        cov[(active[i], active[j])]
    });

    let cholesky =
        Cholesky::new(cov_active).expect("Active covariance matrix not positive definite");
    let a = cholesky.l();
    let z = DVector::from_iterator(mu_active.len(), (0..mu_active.len()).map(|_| normal()));
    let sampled_active = mu_active + a * z;
    let mut result = mu;
    for (k, &i) in active.iter().enumerate() {
        result[i] = sampled_active[k];
    }
    result
}

/// An Adler zero term used in a K-matrix.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct AdlerZero {
    /// The zero position $`s_0`$.
    pub s_0: Float,
    /// The normalization factor $`s_\text{norm}`$.
    pub s_norm: Float,
}

/// Methods for computing various parts of a K-matrix with fixed couplings and mass poles.
#[derive(Clone, Debug, Serialize, Deserialize)]
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
    #[allow(clippy::too_many_arguments)]
    fn new<const PARAMETERS: usize>(
        g: SMatrix<Float, CHANNELS, RESONANCES>,
        c: SMatrix<Float, CHANNELS, CHANNELS>,
        m1s: SVector<Float, CHANNELS>,
        m2s: SVector<Float, CHANNELS>,
        mrs: SVector<Float, RESONANCES>,
        adler_zero: Option<AdlerZero>,
        l: usize,
        cov: SMatrix<Float, PARAMETERS, PARAMETERS>,
        seed: Option<usize>,
    ) -> Self {
        let (g, c, mrs, adler_zero) = if let Some(seed) = seed {
            let mut rng = fastrand::Rng::with_seed(seed as u64);
            let mut flat = SVector::<Float, PARAMETERS>::zeros();
            let mut i = 0;

            for val in g.iter() {
                flat[i] = *val;
                i += 1;
            }
            for val in c.iter() {
                flat[i] = *val;
                i += 1;
            }
            for val in mrs.iter() {
                flat[i] = *val;
                i += 1;
            }
            if let Some(az) = adler_zero {
                flat[i] = az.s_0;
            }
            let flat = sample_normal(flat, cov, &mut rng);
            let mut i = 0;

            let g = SMatrix::<Float, CHANNELS, RESONANCES>::from_iterator(
                flat.iter().skip(i).take(CHANNELS * RESONANCES).cloned(),
            );
            i += CHANNELS * RESONANCES;

            let c = SMatrix::<Float, CHANNELS, CHANNELS>::from_iterator(
                flat.iter().skip(i).take(CHANNELS * CHANNELS).cloned(),
            );
            i += CHANNELS * CHANNELS;

            let mrs = SVector::<Float, RESONANCES>::from_iterator(
                flat.iter().skip(i).take(RESONANCES).cloned(),
            );
            i += RESONANCES;
            let adler_zero = if let Some(az) = adler_zero {
                let az_s_0 = *flat.iter().skip(i).take(1).collect::<Vec<_>>()[0];
                Some(AdlerZero {
                    s_0: az_s_0,
                    s_norm: az.s_norm,
                })
            } else {
                adler_zero
            };
            (g, c, mrs, adler_zero)
        } else {
            (g, c, mrs, adler_zero)
        };
        Self {
            g,
            c,
            m1s,
            m2s,
            mrs,
            adler_zero,
            l,
        }
    }
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
        let ikc_inv_mat = ikc_mat.try_inverse().expect("Matrix inverse failed!");
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

/// Module containing the $`f_0`$ K-matrix.
pub mod f0;
pub use f0::KopfKMatrixF0;

/// Module containing the $`f_2`$ K-matrix.
pub mod f2;
pub use f2::KopfKMatrixF2;

/// Module containing the $`a_0`$ K-matrix.
pub mod a0;
pub use a0::KopfKMatrixA0;

/// Module containing the $`a_2`$ K-matrix.
pub mod a2;
pub use a2::KopfKMatrixA2;

/// Module containing the $`\rho`$ K-matrix.
pub mod rho;
pub use rho::KopfKMatrixRho;

/// Module containing the $`\pi_1`$ K-matrix.
pub mod pi1;
pub use pi1::KopfKMatrixPi1;

#[cfg(feature = "python")]
pub use a0::py_kopf_kmatrix_a0;
#[cfg(feature = "python")]
pub use a2::py_kopf_kmatrix_a2;
#[cfg(feature = "python")]
pub use f0::py_kopf_kmatrix_f0;
#[cfg(feature = "python")]
pub use f2::py_kopf_kmatrix_f2;
#[cfg(feature = "python")]
pub use pi1::py_kopf_kmatrix_pi1;
#[cfg(feature = "python")]
pub use rho::py_kopf_kmatrix_rho;

#[cfg(test)]
mod tests {
    // Note: These tests are not exhaustive, they only check one channel
    use std::sync::Arc;

    use super::*;
    use approx::assert_relative_eq;
    use laddu_core::{data::test_dataset, parameter, Manager, Mass};

    #[test]
    fn test_resampled_evaluation() {
        let mut manager = Manager::default();
        let res_mass = Mass::new([2, 3]);
        let amp = KopfKMatrixA0::new(
            "a0",
            [
                [parameter("p0"), parameter("p1")],
                [parameter("p2"), parameter("p3")],
            ],
            1,
            &res_mass,
            Some(1),
        );
        let aid = manager.register(amp).unwrap();

        let dataset = Arc::new(test_dataset());
        let expr = aid.into();
        let model = manager.model(&expr);
        let evaluator = model.load(&dataset);

        let result = evaluator.evaluate(&[0.1, 0.2, 0.3, 0.4]);

        #[cfg(not(feature = "f32"))]
        {
            assert_relative_eq!(result[0].re, -0.83799097, epsilon = Float::EPSILON.sqrt());
            assert_relative_eq!(result[0].im, -0.02275711, epsilon = Float::EPSILON.sqrt());
        }
        // NOTE: the f32 feature implies a different standard normal RNG which makes all these
        // values different than the f64 version but still reproducible
        #[cfg(feature = "f32")]
        {
            assert_relative_eq!(result[0].re, -0.78789650, epsilon = Float::EPSILON.sqrt());
            assert_relative_eq!(result[0].im, -0.16928345, epsilon = Float::EPSILON.sqrt());
        }
    }

    #[test]
    fn test_resampled_gradient() {
        let mut manager = Manager::default();
        let res_mass = Mass::new([2, 3]);
        let amp = KopfKMatrixA0::new(
            "a0",
            [
                [parameter("p0"), parameter("p1")],
                [parameter("p2"), parameter("p3")],
            ],
            1,
            &res_mass,
            Some(1),
        );
        let aid = manager.register(amp).unwrap();

        let dataset = Arc::new(test_dataset());
        let expr = aid.into();
        let model = manager.model(&expr);
        let evaluator = model.load(&dataset);

        let result = evaluator.evaluate_gradient(&[0.1, 0.2, 0.3, 0.4]);

        #[cfg(not(feature = "f32"))]
        {
            assert_relative_eq!(result[0][0].re, 0.3067087, epsilon = Float::EPSILON.cbrt());
            assert_relative_eq!(result[0][0].im, -0.0483031, epsilon = Float::EPSILON.cbrt());
            assert_relative_eq!(result[0][1].re, -result[0][0].im);
            assert_relative_eq!(result[0][1].im, result[0][0].re);
            assert_relative_eq!(result[0][2].re, -1.1808166, epsilon = Float::EPSILON.cbrt());
            assert_relative_eq!(result[0][2].im, 1.3101937, epsilon = Float::EPSILON.cbrt());
            assert_relative_eq!(result[0][3].re, -result[0][2].im);
            assert_relative_eq!(result[0][3].im, result[0][2].re);
        }
        // NOTE: the f32 feature implies a different standard normal RNG which makes all these
        // values different than the f64 version but still reproducible
        #[cfg(feature = "f32")]
        {
            assert_relative_eq!(result[0][0].re, 0.2876683, epsilon = Float::EPSILON.cbrt());
            assert_relative_eq!(result[0][0].im, -0.1252435, epsilon = Float::EPSILON.cbrt());
            assert_relative_eq!(result[0][1].re, -result[0][0].im);
            assert_relative_eq!(result[0][1].im, result[0][0].re);
            assert_relative_eq!(result[0][2].re, -1.3529229, epsilon = Float::EPSILON.cbrt());
            assert_relative_eq!(result[0][2].im, 1.0895880, epsilon = Float::EPSILON.cbrt());
            assert_relative_eq!(result[0][3].re, -result[0][2].im);
            assert_relative_eq!(result[0][3].im, result[0][2].re);
        }
    }
}
