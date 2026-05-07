use std::fmt::Display;

use nalgebra::DVector;
use num::complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::amplitude::Tags;

/// A helper struct that contains the value of each amplitude for a particular event.
#[derive(Debug)]
pub struct AmplitudeValues(pub Vec<Complex64>);

/// A helper struct that contains the gradient of each amplitude for a particular event.
#[derive(Debug)]
pub struct GradientValues(pub usize, pub Vec<DVector<Complex64>>);

/// A tag set which refers to a registered [`Amplitude`](crate::amplitude::Amplitude).
///
/// This is the base object which can be used to build
/// [`Expression`](crate::expression::Expression)s and should be obtained from the
/// [`Resources::register_amplitude`](crate::resources::Resources::register_amplitude) method.
#[derive(Clone, Default, Debug, Serialize, Deserialize)]
pub struct AmplitudeID(pub(crate) Tags, pub(crate) usize);

impl Display for AmplitudeID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}(id={})", self.0.display_label(), self.1)
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub(crate) struct AmplitudeUseSite {
    pub(crate) amplitude_index: usize,
    pub(crate) tags: Tags,
}

/// Utility function to calculate a central finite difference gradient.
pub fn central_difference<F: Fn(&[f64]) -> f64>(parameters: &[f64], func: F) -> DVector<f64> {
    let mut gradient = DVector::zeros(parameters.len());
    let x = parameters.to_owned();
    let h: DVector<f64> = x
        .iter()
        .map(|&xi| f64::cbrt(f64::EPSILON) * (xi.abs() + 1.0))
        .collect::<Vec<_>>()
        .into();
    for i in 0..parameters.len() {
        let mut x_plus = x.clone();
        let mut x_minus = x.clone();
        x_plus[i] += h[i];
        x_minus[i] -= h[i];
        let f_plus = func(&x_plus);
        let f_minus = func(&x_minus);
        gradient[i] = (f_plus - f_minus) / (2.0 * h[i]);
    }
    gradient
}
