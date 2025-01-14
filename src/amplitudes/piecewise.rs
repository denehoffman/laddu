use num::Complex;
use serde::{Deserialize, Serialize};

use crate::{
    amplitudes::{AmplitudeID, ParameterLike},
    data::Event,
    resources::{Cache, ParameterID, Parameters, Resources},
    traits::Variable,
    CosTheta, DVector, Float, LadduError, Mandelstam, Mass, Phi, PolAngle, PolMagnitude, ScalarID,
};

use super::Amplitude;

fn generate_bin_edges(bins: usize, range: (Float, Float)) -> Vec<Float> {
    let bin_width = (range.1 - range.0) / bins as Float;
    (0..=bins)
        .map(|i| range.0 + i as Float * bin_width)
        .collect()
}

fn get_index(value: Float, bin_edges: &[Float]) -> Option<usize> {
    if value < bin_edges[0] || value >= bin_edges[bin_edges.len() - 1] {
        return None;
    }
    bin_edges
        .windows(2)
        .position(|edges| value >= edges[0] && value < edges[1])
}

/// A piecewise scalar-valued [`Amplitude`] which just contains a single parameter for each bin as its value.
#[derive(Clone, Serialize, Deserialize)]
pub struct PiecewiseScalar<V: Variable> {
    name: String,
    variable: V,
    bin_edges: Vec<Float>,
    values: Vec<ParameterLike>,
    pids: Vec<ParameterID>,
    bin_index: ScalarID,
}
impl<V: Variable> PiecewiseScalar<V> {
    /// Create a new [`PiecewiseScalar`] with the given name and parameter value.
    pub fn new(
        name: &str,
        variable: &V,
        bins: usize,
        range: (Float, Float),
        values: Vec<ParameterLike>,
    ) -> Box<Self> {
        assert_eq!(
            bins,
            values.len(),
            "Number of bins must match number of parameters!"
        );
        Self {
            name: name.to_string(),
            variable: variable.clone(),
            bin_edges: generate_bin_edges(bins, range),
            values,
            pids: Default::default(),
            bin_index: Default::default(),
        }
        .into()
    }
}
macro_rules! impl_amplitude_for_piecewise_scalar {
    ($($generic:ty),*) => {
        $(
            #[typetag::serde]
            impl Amplitude for PiecewiseScalar<$generic> {
                fn register(&mut self, resources: &mut Resources) -> Result<AmplitudeID, LadduError> {
                    self.pids = self
                        .values
                        .iter()
                        .map(|value| resources.register_parameter(value))
                        .collect();
                    self.bin_index = resources.register_scalar(None);
                    resources.register_amplitude(&self.name)
                }

                fn precompute(&self, event: &Event, cache: &mut Cache) {
                    let maybe_bin_index = get_index(self.variable.value(event), &self.bin_edges);
                    if let Some(bin_index) = maybe_bin_index {
                        cache.store_scalar(self.bin_index, bin_index as Float);
                    } else {
                        cache.store_scalar(self.bin_index, self.bin_edges.len() as Float);
                        // store ibin = nbins + 1 if outside range
                    }
                }

                fn compute(&self, parameters: &Parameters, _event: &Event, cache: &Cache) -> Complex<Float> {
                    let bin_index: usize = cache.get_scalar(self.bin_index) as usize;
                    if bin_index == self.bin_edges.len() {
                        Complex::ZERO
                    } else {
                        Complex::from(parameters.get(self.pids[bin_index]))
                    }
                }

                fn compute_gradient(
                    &self,
                    _parameters: &Parameters,
                    _event: &Event,
                    cache: &Cache,
                    gradient: &mut DVector<Complex<Float>>,
                ) {
                    let bin_index: usize = cache.get_scalar(self.bin_index) as usize;
                    if bin_index < self.bin_edges.len() {
                        gradient[bin_index] = Complex::ONE;
                    }
                }
            }
        )*
    };
}
/// A piecewise complex-valued [`Amplitude`] which just contains two parameters representing its real and
/// imaginary parts.
#[derive(Clone, Serialize, Deserialize)]
pub struct PiecewiseComplexScalar<V: Variable> {
    name: String,
    variable: V,
    bin_edges: Vec<Float>,
    re_ims: Vec<(ParameterLike, ParameterLike)>,
    pids_re_im: Vec<(ParameterID, ParameterID)>,
    bin_index: ScalarID,
}
impl<V: Variable> PiecewiseComplexScalar<V> {
    /// Create a new [`PiecewiseComplexScalar`] with the given name and parameter value.
    pub fn new(
        name: &str,
        variable: &V,
        bins: usize,
        range: (Float, Float),
        re_ims: Vec<(ParameterLike, ParameterLike)>,
    ) -> Box<Self> {
        assert_eq!(
            bins,
            re_ims.len(),
            "Number of bins must match number of parameters!"
        );
        Self {
            name: name.to_string(),
            variable: variable.clone(),
            bin_edges: generate_bin_edges(bins, range),
            re_ims,
            pids_re_im: Default::default(),
            bin_index: Default::default(),
        }
        .into()
    }
}

macro_rules! impl_amplitude_for_piecewise_complex_scalar {
    ($($generic:ty),*) => {
        $(
            #[typetag::serde]
            impl Amplitude for PiecewiseComplexScalar<$generic> {
                fn register(&mut self, resources: &mut Resources) -> Result<AmplitudeID, LadduError> {
                    self.pids_re_im = self
                        .re_ims
                        .iter()
                        .map(|(re, im)| (resources.register_parameter(re), resources.register_parameter(im)))
                        .collect();
                    self.bin_index = resources.register_scalar(None);
                    resources.register_amplitude(&self.name)
                }

                fn precompute(&self, event: &Event, cache: &mut Cache) {
                    let maybe_bin_index = get_index(self.variable.value(event), &self.bin_edges);
                    if let Some(bin_index) = maybe_bin_index {
                        cache.store_scalar(self.bin_index, bin_index as Float);
                    } else {
                        cache.store_scalar(self.bin_index, self.bin_edges.len() as Float);
                        // store ibin = nbins + 1 if outside range
                    }
                }

                fn compute(&self, parameters: &Parameters, _event: &Event, cache: &Cache) -> Complex<Float> {
                    let bin_index: usize = cache.get_scalar(self.bin_index) as usize;
                    if bin_index == self.bin_edges.len() {
                        Complex::ZERO
                    } else {
                        let pid_re_im = self.pids_re_im[bin_index];
                        Complex::new(
                            parameters.get(pid_re_im.0),
                            parameters.get(pid_re_im.1),
                        )
                    }
                }

                fn compute_gradient(
                    &self,
                    _parameters: &Parameters,
                    _event: &Event,
                    cache: &Cache,
                    gradient: &mut DVector<Complex<Float>>,
                ) {
                    let bin_index: usize = cache.get_scalar(self.bin_index) as usize;
                    if bin_index < self.bin_edges.len() {
                        let pid_re_im = self.pids_re_im[bin_index];
                        if let ParameterID::Parameter(ind) = pid_re_im.0 {
                            gradient[ind] = Complex::ONE;
                        }
                        if let ParameterID::Parameter(ind) = pid_re_im.1 {
                            gradient[ind] = Complex::I;
                        }
                    }
                }
            }
        )*
    };
}

/// A piecewise complex-valued [`Amplitude`] which just contains two parameters representing its magnitude and
/// phase.
#[derive(Clone, Serialize, Deserialize)]
pub struct PiecewisePolarComplexScalar<V: Variable> {
    name: String,
    variable: V,
    bin_edges: Vec<Float>,
    r_thetas: Vec<(ParameterLike, ParameterLike)>,
    pids_r_theta: Vec<(ParameterID, ParameterID)>,
    bin_index: ScalarID,
}
impl<V: Variable> PiecewisePolarComplexScalar<V> {
    /// Create a new [`PiecewiseComplexScalar`] with the given name and parameter value.
    pub fn new(
        name: &str,
        variable: &V,
        bins: usize,
        range: (Float, Float),
        r_thetas: Vec<(ParameterLike, ParameterLike)>,
    ) -> Box<Self> {
        assert_eq!(
            bins,
            r_thetas.len(),
            "Number of bins must match number of parameters!"
        );
        Self {
            name: name.to_string(),
            variable: variable.clone(),
            bin_edges: generate_bin_edges(bins, range),
            r_thetas,
            pids_r_theta: Default::default(),
            bin_index: Default::default(),
        }
        .into()
    }
}

macro_rules! impl_amplitude_for_piecewise_polar_complex_scalar {
    ($($generic:ty),*) => {
        $(
            #[typetag::serde]
            impl Amplitude for PiecewisePolarComplexScalar<$generic> {
                fn register(&mut self, resources: &mut Resources) -> Result<AmplitudeID, LadduError> {
                    self.pids_r_theta = self
                        .r_thetas
                        .iter()
                        .map(|(r, theta)| (resources.register_parameter(r), resources.register_parameter(theta)))
                        .collect();
                    self.bin_index = resources.register_scalar(None);
                    resources.register_amplitude(&self.name)
                }

                fn precompute(&self, event: &Event, cache: &mut Cache) {
                    let maybe_bin_index = get_index(self.variable.value(event), &self.bin_edges);
                    if let Some(bin_index) = maybe_bin_index {
                        cache.store_scalar(self.bin_index, bin_index as Float);
                    } else {
                        cache.store_scalar(self.bin_index, self.bin_edges.len() as Float);
                        // store ibin = nbins + 1 if outside range
                    }
                }

                fn compute(&self, parameters: &Parameters, _event: &Event, cache: &Cache) -> Complex<Float> {
                    let bin_index: usize = cache.get_scalar(self.bin_index) as usize;
                    if bin_index == self.bin_edges.len() {
                        Complex::ZERO
                    } else {
                        let pid_r_theta = self.pids_r_theta[bin_index];
                        Complex::from_polar(
                            parameters.get(pid_r_theta.0),
                            parameters.get(pid_r_theta.1),
                        )
                    }
                }

                fn compute_gradient(
                    &self,
                    parameters: &Parameters,
                    _event: &Event,
                    cache: &Cache,
                    gradient: &mut DVector<Complex<Float>>,
                ) {
                    let bin_index: usize = cache.get_scalar(self.bin_index) as usize;
                    if bin_index < self.bin_edges.len() {
                        let pid_r_theta = self.pids_r_theta[bin_index];
                        let r = parameters.get(pid_r_theta.0);
                        let theta = parameters.get(pid_r_theta.1);
                        let exp_i_theta = Complex::cis(theta);
                        if let ParameterID::Parameter(ind) = pid_r_theta.0 {
                            gradient[ind] = exp_i_theta;
                        }
                        if let ParameterID::Parameter(ind) = pid_r_theta.1 {
                            gradient[ind] = Complex::<Float>::I
                                * Complex::from_polar(r, theta);
                        }
                    }
                }
            }
        )*
    };
}

impl_amplitude_for_piecewise_scalar!(Mass, CosTheta, Phi, PolAngle, PolMagnitude, Mandelstam);
impl_amplitude_for_piecewise_complex_scalar!(
    Mass,
    CosTheta,
    Phi,
    PolAngle,
    PolMagnitude,
    Mandelstam
);
impl_amplitude_for_piecewise_polar_complex_scalar!(
    Mass,
    CosTheta,
    Phi,
    PolAngle,
    PolMagnitude,
    Mandelstam
);

#[cfg(feature = "python")]
impl_amplitude_for_piecewise_scalar!(crate::python::laddu::PyVariable);
#[cfg(feature = "python")]
impl_amplitude_for_piecewise_complex_scalar!(crate::python::laddu::PyVariable);
#[cfg(feature = "python")]
impl_amplitude_for_piecewise_polar_complex_scalar!(crate::python::laddu::PyVariable);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::amplitudes::{parameter, Manager};
    use crate::data::test_dataset;
    use crate::PI;
    use approx::assert_relative_eq;
    use std::sync::Arc;

    #[test]
    fn test_binning() {
        let v = Mass::new([2]);
        let dataset = Arc::new(test_dataset());
        let bin_edges = generate_bin_edges(3, (0.0, 1.0));
        let bin_index = get_index(v.value_on(&dataset)[0], &bin_edges);
        assert_eq!(bin_index, Some(1));
        let bin_index = get_index(0.0, &bin_edges);
        assert_eq!(bin_index, Some(0));
        let bin_index = get_index(0.1, &bin_edges);
        assert_eq!(bin_index, Some(0));
        let bin_index = get_index(0.9, &bin_edges);
        assert_eq!(bin_index, Some(2));
        let bin_index = get_index(1.0, &bin_edges);
        assert_eq!(bin_index, None);
        let bin_index = get_index(2.0, &bin_edges);
        assert_eq!(bin_index, None);
    }

    #[test]
    fn test_piecewise_scalar_creation_and_evaluation() {
        let mut manager = Manager::default();
        let v = Mass::new([2]);
        let amp = PiecewiseScalar::new(
            "test_scalar",
            &v,
            3,
            (0.0, 1.0),
            vec![
                parameter("test_param0"),
                parameter("test_param1"),
                parameter("test_param2"),
            ],
        );
        let aid = manager.register(amp).unwrap();

        let dataset = Arc::new(test_dataset());
        let expr = aid.into(); // Direct amplitude evaluation
        let model = manager.model(&expr);
        let evaluator = model.load(&dataset);

        let params = vec![1.1, 2.2, 3.3];
        let result = evaluator.evaluate(&params);

        assert_relative_eq!(result[0].re, 2.2);
        assert_relative_eq!(result[0].im, 0.0);
    }

    #[test]
    fn test_piecewise_scalar_gradient() {
        let mut manager = Manager::default();
        let v = Mass::new([2]);
        let amp = PiecewiseScalar::new(
            "test_scalar",
            &v,
            3,
            (0.0, 1.0),
            vec![
                parameter("test_param0"),
                parameter("test_param1"),
                parameter("test_param2"),
            ],
        );
        let aid = manager.register(amp).unwrap();

        let dataset = Arc::new(test_dataset());
        let expr = aid.norm_sqr(); // |f(x)|^2
        let model = manager.model(&expr);
        let evaluator = model.load(&dataset);

        let params = vec![1.0, 2.0, 3.0];
        let gradient = evaluator.evaluate_gradient(&params);

        // For |f(x)|^2 where f(x) = x, the derivative should be 2x
        assert_relative_eq!(gradient[0][0].re, 0.0);
        assert_relative_eq!(gradient[0][0].im, 0.0);
        assert_relative_eq!(gradient[0][1].re, 4.0);
        assert_relative_eq!(gradient[0][1].im, 0.0);
        assert_relative_eq!(gradient[0][2].re, 0.0);
        assert_relative_eq!(gradient[0][2].im, 0.0);
    }

    #[test]
    fn test_piecewise_complex_scalar_evaluation() {
        let mut manager = Manager::default();
        let v = Mass::new([2]);
        let amp = PiecewiseComplexScalar::new(
            "test_complex",
            &v,
            3,
            (0.0, 1.0),
            vec![
                (parameter("re_param0"), parameter("im_param0")),
                (parameter("re_param1"), parameter("im_param1")),
                (parameter("re_param2"), parameter("im_param2")),
            ],
        );
        let aid = manager.register(amp).unwrap();

        let dataset = Arc::new(test_dataset());
        let expr = aid.into();
        let model = manager.model(&expr);
        let evaluator = model.load(&dataset);

        let params = vec![1.1, 1.2, 2.1, 2.2, 3.1, 3.2]; // Real and imaginary parts
        let result = evaluator.evaluate(&params);

        assert_relative_eq!(result[0].re, 2.1);
        assert_relative_eq!(result[0].im, 2.2);
    }

    #[test]
    fn test_piecewise_complex_scalar_gradient() {
        let mut manager = Manager::default();
        let v = Mass::new([2]);
        let amp = PiecewiseComplexScalar::new(
            "test_complex",
            &v,
            3,
            (0.0, 1.0),
            vec![
                (parameter("re_param0"), parameter("im_param0")),
                (parameter("re_param1"), parameter("im_param1")),
                (parameter("re_param2"), parameter("im_param2")),
            ],
        );
        let aid = manager.register(amp).unwrap();

        let dataset = Arc::new(test_dataset());
        let expr = aid.norm_sqr(); // |f(x + iy)|^2
        let model = manager.model(&expr);
        let evaluator = model.load(&dataset);

        let params = vec![1.1, 1.2, 2.1, 2.2, 3.1, 3.2]; // Real and imaginary parts
        let gradient = evaluator.evaluate_gradient(&params);

        // For |f(x + iy)|^2, partial derivatives should be 2x and 2y
        assert_relative_eq!(gradient[0][0].re, 0.0);
        assert_relative_eq!(gradient[0][0].im, 0.0);
        assert_relative_eq!(gradient[0][1].re, 0.0);
        assert_relative_eq!(gradient[0][1].im, 0.0);
        assert_relative_eq!(gradient[0][2].re, 4.2);
        assert_relative_eq!(gradient[0][2].im, 0.0);
        assert_relative_eq!(gradient[0][3].re, 4.4);
        assert_relative_eq!(gradient[0][3].im, 0.0);
        assert_relative_eq!(gradient[0][4].re, 0.0);
        assert_relative_eq!(gradient[0][4].im, 0.0);
        assert_relative_eq!(gradient[0][5].re, 0.0);
        assert_relative_eq!(gradient[0][5].im, 0.0);
    }

    #[test]
    fn test_piecewise_polar_complex_scalar_evaluation() {
        let mut manager = Manager::default();
        let v = Mass::new([2]);
        let amp = PiecewisePolarComplexScalar::new(
            "test_polar",
            &v,
            3,
            (0.0, 1.0),
            vec![
                (parameter("r_param0"), parameter("theta_param0")),
                (parameter("r_param1"), parameter("theta_param1")),
                (parameter("r_param2"), parameter("theta_param2")),
            ],
        );
        let aid = manager.register(amp).unwrap();

        let dataset = Arc::new(test_dataset());
        let expr = aid.into();
        let model = manager.model(&expr);
        let evaluator = model.load(&dataset);

        let r = 2.0;
        let theta = PI / 4.3;
        let params = vec![
            1.1 * r,
            1.2 * theta,
            2.1 * r,
            2.2 * theta,
            3.1 * r,
            3.2 * theta,
        ];
        let result = evaluator.evaluate(&params);

        // r * (cos(theta) + i*sin(theta))
        assert_relative_eq!(result[0].re, 2.1 * r * (2.2 * theta).cos());
        assert_relative_eq!(result[0].im, 2.1 * r * (2.2 * theta).sin());
    }

    #[test]
    fn test_piecewise_polar_complex_scalar_gradient() {
        let mut manager = Manager::default();
        let v = Mass::new([2]);
        let amp = PiecewisePolarComplexScalar::new(
            "test_polar",
            &v,
            3,
            (0.0, 1.0),
            vec![
                (parameter("r_param0"), parameter("theta_param0")),
                (parameter("r_param1"), parameter("theta_param1")),
                (parameter("r_param2"), parameter("theta_param2")),
            ],
        );
        let aid = manager.register(amp).unwrap();

        let dataset = Arc::new(test_dataset());
        let expr = aid.into(); // f(r,θ) = re^(iθ)
        let model = manager.model(&expr);
        let evaluator = model.load(&dataset);

        let r = 2.0;
        let theta = PI / 4.3;
        let params = vec![
            1.1 * r,
            1.2 * theta,
            2.1 * r,
            2.2 * theta,
            3.1 * r,
            3.2 * theta,
        ];
        let gradient = evaluator.evaluate_gradient(&params);

        // d/dr re^(iθ) = e^(iθ), d/dθ re^(iθ) = ire^(iθ)
        assert_relative_eq!(gradient[0][0].re, 0.0);
        assert_relative_eq!(gradient[0][0].im, 0.0);
        assert_relative_eq!(gradient[0][1].re, 0.0);
        assert_relative_eq!(gradient[0][1].im, 0.0);
        assert_relative_eq!(gradient[0][2].re, Float::cos(2.2 * theta));
        assert_relative_eq!(gradient[0][2].im, Float::sin(2.2 * theta));
        assert_relative_eq!(gradient[0][3].re, -2.1 * r * Float::sin(2.2 * theta));
        assert_relative_eq!(gradient[0][3].im, 2.1 * r * Float::cos(2.2 * theta));
        assert_relative_eq!(gradient[0][4].re, 0.0);
        assert_relative_eq!(gradient[0][4].im, 0.0);
        assert_relative_eq!(gradient[0][5].re, 0.0);
        assert_relative_eq!(gradient[0][5].im, 0.0);
    }
}
