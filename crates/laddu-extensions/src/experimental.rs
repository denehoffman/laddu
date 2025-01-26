use laddu_core::{traits::Variable, utils::histogram, DVector, Float};

use crate::{likelihoods::LikelihoodTerm, NLL};

#[cfg(feature = "python")]
use crate::likelihoods::{PyLikelihoodTerm, PyNLL};
#[cfg(feature = "python")]
use laddu_python::utils::variables::PyVariable;
#[cfg(feature = "python")]
use pyo3::prelude::*;

/// A [`LikelihoodTerm`] whose size is proportional to the χ²-distance from a binned projection of
/// the fit to a provided set of datapoints representing the true values in each bin.
///
/// This is intended to be used as follows. Suppose we perform a binned fit to a simple amplitude
/// which is not parameterized over the binning variable. We then form a new [`Model`] which *is*
/// parameterized over said variable, and we wish to perform an unbinned fit. If we can isolate
/// terms which are not interfering, we could imagine fitting the unbinned data with a cost
/// function that minimizes the distance to the result from the binned fit. From there, it is up to
/// the user to decide what to do with this minimum. Caution should be used, as this will not be
/// the minimum of the [`NLL`], but of the guide term only. However, this minimum could be used as
/// an intermediate for getting close to a global minimum if the likelihood landscape has many
/// local minima. Then a true fit could be performed, starting at this intermediate point.
#[derive(Clone)]
pub struct BinnedGuideTerm {
    nll: Box<NLL>,
    values: Vec<Float>,
    amplitude_sets: Vec<Vec<String>>,
    bins: usize,
    range: (Float, Float),
    count_sets: Vec<Vec<Float>>,
    error_sets: Vec<Vec<Float>>,
}

impl BinnedGuideTerm {
    /// Construct a new [`BinnedGuideTerm`]
    ///
    /// This term takes a list of subsets of amplitudes, activates each set, and compares the projected
    /// histogram to the known one provided at construction. Both `count_sets` and `error_sets` should
    /// have the same shape, and their first dimension should be the same as that of `amplitude_sets`.
    ///
    /// The intended usage is to provide some sets of amplitudes to isolate, like `[["amp1", "amp2"], ["amp3"]]`,
    /// along with some known counts for a binned fit (`count_sets ~ [[histogram counts involving "amp1" and "amp2"], [histogram counts involving "amp3"]]` and simlar for `error_sets`).
    pub fn new<
        V: Variable + 'static,
        L: AsRef<str>,
        T: AsRef<[L]>,
        U: AsRef<[Float]>,
        E: AsRef<[Float]>,
    >(
        nll: Box<NLL>,
        variable: &V,
        amplitude_sets: &[T],
        bins: usize,
        range: (Float, Float),
        count_sets: &[U],
        error_sets: Option<&[E]>,
    ) -> Box<Self> {
        let values = variable.value_on(&nll.accmc_evaluator.dataset);
        let amplitude_sets: Vec<Vec<String>> = amplitude_sets
            .iter()
            .map(|t| t.as_ref().iter().map(|s| s.as_ref().to_string()).collect())
            .collect();
        let count_sets: Vec<Vec<Float>> = count_sets.iter().map(|f| f.as_ref().to_vec()).collect();
        let error_sets: Vec<Vec<Float>> = if let Some(error_sets) = error_sets {
            error_sets.iter().map(|f| f.as_ref().to_vec()).collect()
        } else {
            count_sets
                .iter()
                .map(|v| v.iter().map(|f| f.sqrt()).collect())
                .collect()
        };
        assert_eq!(amplitude_sets.len(), count_sets.len());
        assert_eq!(count_sets.len(), error_sets.len());
        Box::new(Self {
            nll,
            amplitude_sets,
            values,
            bins,
            range,
            count_sets,
            error_sets,
        })
    }
}

impl LikelihoodTerm for BinnedGuideTerm {
    fn evaluate(&self, parameters: &[Float]) -> Float {
        let mut result = 0.0;
        for ((counts, errors), amplitudes) in self
            .count_sets
            .iter()
            .zip(self.error_sets.iter())
            .zip(self.amplitude_sets.iter())
        {
            let weights = self.nll.project_with(parameters, amplitudes, None).unwrap();
            let eval_hist = histogram(&self.values, self.bins, self.range, Some(&weights));
            let chisqr: Float = eval_hist
                .counts
                .iter()
                .zip(counts.iter())
                .zip(errors.iter())
                .map(|((o, c), e)| (o - c).powi(2) / e.powi(2))
                .sum();
            result += chisqr;
        }
        result
    }

    fn parameters(&self) -> Vec<String> {
        self.nll.parameters()
    }

    fn evaluate_gradient(&self, parameters: &[Float]) -> laddu_core::DVector<Float> {
        let mut gradient = DVector::zeros(parameters.len());
        let bin_width = (self.range.1 - self.range.0) / self.bins as Float;
        for ((counts, errors), amplitudes) in self
            .count_sets
            .iter()
            .zip(self.error_sets.iter())
            .zip(self.amplitude_sets.iter())
        {
            let (weights, weights_gradient) = self
                .nll
                .project_gradient_with(parameters, amplitudes, None)
                .unwrap();
            let mut eval_counts = vec![0.0; self.bins];
            let mut eval_count_gradient: Vec<DVector<Float>> =
                vec![DVector::zeros(parameters.len()); self.bins];

            for (j, &value) in self.values.iter().enumerate() {
                if value >= self.range.0 && value < self.range.1 {
                    let bin_idx =
                        (((value - self.range.0) / bin_width).floor() as usize).min(self.bins - 1);
                    eval_counts[bin_idx] += weights[j];
                    for k in 0..parameters.len() {
                        eval_count_gradient[bin_idx][k] += weights_gradient[j][k];
                    }
                }
            }
            for i in 0..self.bins {
                let o_i = eval_counts[i];
                let c_i = counts[i];
                let e_i = errors[i];
                let residual = o_i - c_i;
                let residual_gradient = &eval_count_gradient[i];
                for k in 0..parameters.len() {
                    gradient[k] += 2.0 * residual * residual_gradient[k] / e_i.powi(2);
                }
            }
        }
        gradient
    }
}

/// A χ²-like term which uses a known binned result to guide the fit
///
/// This term takes a list of subsets of amplitudes, activates each set, and compares the projected
/// histogram to the known one provided at construction. Both `count_sets` and `error_sets` should
/// have the same shape, and their first dimension should be the same as that of `amplitude_sets`.
///
/// Parameters
/// ----------
/// nll: NLL
/// variable : {laddu.Mass, laddu.CosTheta, laddu.Phi, laddu.PolAngle, laddu.PolMagnitude, laddu.Mandelstam}
///     The variable to use for binning
/// amplitude_sets : list of list of str
///     A list of lists of amplitudes to activate, with each inner list representing a set that
///     corresponds to the provided binned data
/// bins : int
/// range : tuple of (min, max)
///     The range of the variable to use for binning
/// count_sets : list of list of float
///      A list of binned counts for each amplitude set
/// error_sets : list of list of float, optional
///      A list of bin errors for each amplitude set (square root of `count_sets` if None is
///      provided)
///
/// Returns
/// -------
/// LikelihoodTerm
///
#[cfg(feature = "python")]
#[pyfunction(name = "BinnedGuideTerm", signature = (nll, variable, amplitude_sets, bins, range, count_sets, error_sets = None))]
pub fn py_binned_guide_term(
    nll: PyNLL,
    variable: Bound<'_, PyAny>,
    amplitude_sets: Vec<Vec<String>>,
    bins: usize,
    range: (Float, Float),
    count_sets: Vec<Vec<Float>>,
    error_sets: Option<Vec<Vec<Float>>>,
) -> PyResult<PyLikelihoodTerm> {
    let variable = variable.extract::<PyVariable>()?;
    Ok(PyLikelihoodTerm(BinnedGuideTerm::new(
        nll.0.clone(),
        &variable,
        &amplitude_sets,
        bins,
        range,
        &count_sets,
        error_sets.as_deref(),
    )))
}
