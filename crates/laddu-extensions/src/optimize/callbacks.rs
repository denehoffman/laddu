use ganesh::traits::{Algorithm, Observer, Status};

use crate::likelihood::LikelihoodTerm;

/// An observer which calls [`LikelihoodTerm::update`] on each step of the algorithm.
///
/// This should generally be used with any algorithm, but it mostly impacts
/// [`StochasticNLL`](`crate::likelihood::StochasticNLL`) terms which need to update random state
/// at each step.
#[derive(Copy, Clone)]
pub struct LikelihoodTermObserver;

impl<A, P, S, U, E, C> Observer<A, P, S, U, E, C> for LikelihoodTermObserver
where
    A: Algorithm<P, S, U, E, Config = C>,
    P: LikelihoodTerm,
    S: Status,
{
    fn observe(
        &mut self,
        _current_step: usize,
        _algorithm: &A,
        problem: &P,
        _status: &S,
        _args: &U,
        _config: &C,
    ) {
        problem.update();
    }
}
