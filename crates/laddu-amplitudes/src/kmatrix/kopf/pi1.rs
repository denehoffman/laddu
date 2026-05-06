use std::array;

use laddu_core::{
    amplitudes::{
        display_key, parameter_array_key, Amplitude, AmplitudeID, AmplitudeSemanticKey, IntoTags,
        Parameter, Tags,
    },
    data::{DatasetMetadata, Event},
    resources::{Cache, ComplexVectorID, MatrixID, ParameterID, Parameters, Resources},
    traits::Variable,
    variables::Mass,
    Expression, LadduResult,
};
use nalgebra::{matrix, vector, DVector, SVector};
use num::complex::Complex64;
use serde::{Deserialize, Serialize};

use super::{FixedKMatrix, KopfKMatrixPi1Channel};

/// A K-matrix parameterization for the $`\pi_1`$ hybrid candidate described by Kopf et al.[^1] with fixed couplings and mass poles
/// (free production couplings only).
///
/// [^1]: Kopf, B., Albrecht, M., Koch, H., Küßner, M., Pychy, J., Qin, X., & Wiedner, U. (2021). Investigation of the lightest hybrid meson candidate with a coupled-channel analysis of $`\bar{p}p`$-, $`\pi^- p`$- and $`\pi \pi`$-Data. The European Physical Journal C, 81(12). [doi:10.1140/epjc/s10052-021-09821-2](https://doi.org/10.1140/epjc/s10052-021-09821-2)
#[derive(Clone, Serialize, Deserialize)]
pub struct KopfKMatrixPi1 {
    tags: Tags,
    channel: KopfKMatrixPi1Channel,
    mass: Mass,
    constants: FixedKMatrix<2, 1>,
    couplings_real: [Parameter; 1],
    couplings_imag: [Parameter; 1],
    couplings_indices_real: [ParameterID; 1],
    couplings_indices_imag: [ParameterID; 1],
    ikc_cache_index: ComplexVectorID<2>,
    p_vec_cache_index: MatrixID<2, 1>,
}

impl KopfKMatrixPi1 {
    /// Construct a new [`KopfKMatrixPi1`] with activation tags, production couplings, channel,
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
        tags: impl IntoTags,
        couplings: [[Parameter; 2]; 1],
        channel: KopfKMatrixPi1Channel,
        mass: &Mass,
    ) -> LadduResult<Expression> {
        let mut couplings_real: [Parameter; 1] = array::from_fn(|_| Parameter::default());
        let mut couplings_imag: [Parameter; 1] = array::from_fn(|_| Parameter::default());
        for i in 0..1 {
            couplings_real[i] = couplings[i][0].clone();
            couplings_imag[i] = couplings[i][1].clone();
        }
        Self {
            tags: tags.into_tags(),
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
        .into_expression()
    }
}

#[typetag::serde]
impl Amplitude for KopfKMatrixPi1 {
    fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID> {
        for i in 0..self.couplings_indices_real.len() {
            self.couplings_indices_real[i] =
                resources.register_parameter(&self.couplings_real[i])?;
            self.couplings_indices_imag[i] =
                resources.register_parameter(&self.couplings_imag[i])?;
        }
        self.ikc_cache_index = resources.register_complex_vector(None);
        self.p_vec_cache_index = resources.register_matrix(None);
        resources.register_amplitude(self.tags.clone())
    }

    fn semantic_key(&self) -> Option<AmplitudeSemanticKey> {
        Some(
            AmplitudeSemanticKey::new("KopfKMatrixPi1")
                .with_field("channel", self.channel.to_string())
                .with_field("mass", display_key(&self.mass))
                .with_field("couplings_real", parameter_array_key(&self.couplings_real))
                .with_field("couplings_imag", parameter_array_key(&self.couplings_imag)),
        )
    }

    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        self.mass.bind(metadata)?;
        Ok(())
    }

    fn precompute(&self, event: &Event<'_>, cache: &mut Cache) {
        let s = self.mass.value(event).powi(2);
        cache.store_complex_vector(
            self.ikc_cache_index,
            self.constants.ikc_inv_vec(s, self.channel.index()),
        );
        cache.store_matrix(self.p_vec_cache_index, self.constants.p_vec_constants(s));
    }
    fn compute(&self, parameters: &Parameters, cache: &Cache) -> Complex64 {
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
        parameters: &Parameters,
        cache: &Cache,
        gradient: &mut DVector<Complex64>,
    ) {
        let ikc_inv_vec = cache.get_complex_vector(self.ikc_cache_index);
        let p_vec_constants = cache.get_matrix(self.p_vec_cache_index);
        let internal_gradient = FixedKMatrix::compute_gradient(&ikc_inv_vec, &p_vec_constants);
        if let Some(index) = parameters.free_index(self.couplings_indices_real[0]) {
            gradient[index] = internal_gradient[0];
        }
        if let Some(index) = parameters.free_index(self.couplings_indices_imag[0]) {
            gradient[index] = Complex64::I * internal_gradient[0];
        }
    }
}

/// A fixed K-Matrix Amplitude for the :math:`\pi_1(1600)` hybrid meson
///
/// Parameters
/// ----------
/// tags : str
///     Activation tag(s) for the amplitude.
/// couplings : list of list of laddu.Parameter
///     Each initial-state coupling (as a list of pairs of real and imaginary parts)
/// channel : laddu.KopfKMatrixPi1Channel
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
/// | 0             | :math:`\pi\eta`   |
/// +---------------+-------------------+
/// | 1             | :math:`\pi\eta'`  |
/// +---------------+-------------------+
///
/// +---------------------+
/// | Pole names          |
/// +=====================+
/// | :math:`\pi_1(1600)` |
/// +---------------------+
///
#[cfg(test)]
mod tests {
    // Note: These tests are not exhaustive, they only check one channel
    use std::sync::Arc;

    use approx::assert_relative_eq;
    use laddu_core::{data::test_dataset, parameter, Mass};

    use super::*;

    #[test]
    fn test_pi1_evaluation() {
        let dataset = Arc::new(test_dataset());
        let res_mass = Mass::new(["kshort1", "kshort2"]);
        let expr = KopfKMatrixPi1::new(
            "pi1",
            [[parameter!("p0"), parameter!("p1")]],
            KopfKMatrixPi1Channel::PiEtaPrime,
            &res_mass,
        )
        .unwrap();
        let evaluator = expr.load(&dataset).unwrap();

        let result = evaluator.evaluate(&[0.1, 0.2]).unwrap();

        assert_relative_eq!(result[0].re, -0.11017586807747382);
        assert_relative_eq!(result[0].im, 0.2638717244927635);
    }

    #[test]
    fn test_pi1_gradient() {
        let dataset = Arc::new(test_dataset());
        let res_mass = Mass::new(["kshort1", "kshort2"]);
        let expr = KopfKMatrixPi1::new(
            "pi1",
            [[parameter!("p0"), parameter!("p1")]],
            KopfKMatrixPi1Channel::PiEtaPrime,
            &res_mass,
        )
        .unwrap();
        let evaluator = expr.load(&dataset).unwrap();

        let result = evaluator.evaluate_gradient(&[0.1, 0.2]).unwrap();

        assert_relative_eq!(result[0][0].re, -14.79871746893747);
        assert_relative_eq!(result[0][0].im, -5.843009428873964);
        assert_relative_eq!(result[0][1].re, -result[0][0].im);
        assert_relative_eq!(result[0][1].im, result[0][0].re);
    }
}
