use laddu_extensions::experimental::{BinnedGuideTerm, Regularizer};
use pyo3::{exceptions::PyValueError, prelude::*};

use crate::{
    extensions::likelihood::{PyLikelihoodExpression, PyNLL},
    variables::PyVariable,
};
/// A χ²-like term which uses a known binned result to guide the fit
///
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
/// LikelihoodExpression
///     A term that can be combined with other likelihood expressions.
#[pyfunction(name = "BinnedGuideTerm", signature = (nll, variable, amplitude_sets, bins, range, count_sets, error_sets = None))]
pub fn py_binned_guide_term(
    nll: PyNLL,
    variable: Bound<'_, PyAny>,
    amplitude_sets: Vec<Vec<String>>,
    bins: usize,
    range: (f64, f64),
    count_sets: Vec<Vec<f64>>,
    error_sets: Option<Vec<Vec<f64>>>,
) -> PyResult<PyLikelihoodExpression> {
    let variable = variable.extract::<PyVariable>()?;
    Ok(PyLikelihoodExpression(BinnedGuideTerm::new(
        nll.0.clone(),
        &variable,
        &amplitude_sets,
        bins,
        range,
        &count_sets,
        error_sets.as_deref(),
    )?))
}

/// An weighted :math:`\ell_p` regularization term which acts as a maximum a posteriori (MAP) prior.
///
/// This can be interpreted as a prior of the form
///
/// .. math:: f(\vec{x}) = \frac{p\lambda^{1/p}}{2\Gamma(1/p)}e^{-\lambda|\vec{x}|^p}
///
/// which becomes a Laplace distribution for :math:`p=1` and a Gaussian for :math:`p=2`. These are commonly
/// interpreted as :math:`\ell_p` regularizers for linear regression models, with :math:`p=1` and :math:`p=2`
/// corresponding to LASSO and ridge regression, respectively. When used in nonlinear regression,
/// these should be interpeted as the prior listed above when used in maximum a posteriori (MAP)
/// estimation. Explicitly, when the logarithm is taken, this term becomes
///
/// .. math:: \lambda \left(\sum_{j} w_j |x_j|^p\right)^{1/p}
///
/// plus some additional constant terms which do not depend on free parameters.
///
/// Weights can be specified to vary the influence of each parameter used in the regularization.
/// These weights are typically assigned by first fitting without a regularization term to obtain
/// parameter values :math:`\vec{\beta}`, choosing a value :math:`\gamma>0`, and setting the weights to
/// :math:`\vec{w} = 1/|\vec{\beta}|^\gamma` according to [Zou]_.
///
/// References
/// ----------
/// .. [Zou] Zou, H. (2006). The Adaptive Lasso and Its Oracle Properties.
///    Journal of the American Statistical Association, 101(476), 1418–1429.
///    doi:10.1198/016214506000000735
///
/// Parameters
/// ----------
/// parameters : list of str
///     The names of the parameters to regularize
/// lda : float
///     The regularization parameter :math:`\lambda`
/// p : {1, 2}
///     The degree of the norm :math:`\ell_p`
/// weights : list of float, optional
///     Weights to apply in the regularization to each parameter
///
/// Raises
/// ------
/// ValueError
///     If :math:`p` is not 1 or 2
/// Exception
///     If the number of parameters and weights is not equal
///
/// Returns
/// -------
/// LikelihoodExpression
///     A term that can be combined with other likelihood expressions.
#[allow(rustdoc::broken_intra_doc_links)]
#[pyfunction(name = "Regularizer", signature = (parameters, lda, p=1, weights=None))]
pub fn py_regularizer(
    parameters: Vec<String>,
    lda: f64,
    p: usize,
    weights: Option<Vec<f64>>,
) -> PyResult<PyLikelihoodExpression> {
    if p == 1 {
        Ok(PyLikelihoodExpression(Regularizer::<1>::new(
            parameters, lda, weights,
        )?))
    } else if p == 2 {
        Ok(PyLikelihoodExpression(Regularizer::<2>::new(
            parameters, lda, weights,
        )?))
    } else {
        Err(PyValueError::new_err(
            "'Regularizer' only supports p = 1 or 2",
        ))
    }
}
