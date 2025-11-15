//! # laddu-extensions
//!
//! This is an internal crate used by `laddu`.
#![warn(clippy::perf, clippy::style, missing_docs)]

/// Experimental extensions to the `laddu` ecosystem
///
/// <div class="warning">
///
/// This module contains experimental code which may be untested or unreliable. Use at your own
/// risk! The features contained here may eventually be moved into the standard crate modules.
///
/// </div>
pub mod experimental;

/// A module containing the `laddu` interface with the [`ganesh`] library
pub mod ganesh_ext;

/// Extended maximum likelihood cost functions with support for additive terms
pub mod likelihoods;

// pub use ganesh_ext::{MCMCOptions, MinimizerOptions};
pub use likelihoods::{
    LikelihoodEvaluator, LikelihoodExpression, LikelihoodID, LikelihoodManager, LikelihoodScalar,
    NLL,
};

use fastrand::Rng;
use rapidhash::{HashSetExt, RapidHashSet};

/// An extension to [`Rng`] which allows for sampling from a subset of the integers `[0..n)`
/// without replacement.
pub trait RngSubsetExtension {
    /// Draw a random subset of `m` indices between `0` and `n`.
    fn subset(&mut self, m: usize, n: usize) -> Vec<usize>;
}

// Nice write-up here:
// https://www.nowherenearithaca.com/2013/05/robert-floyds-tiny-and-beautiful.html
fn floyd_sample(m: usize, n: usize, rng: &mut Rng) -> RapidHashSet<usize> {
    let mut set = RapidHashSet::with_capacity(m * 2);
    for j in (n - m)..n {
        let t = rng.usize(..=j);
        if !set.insert(t) {
            set.insert(j);
        }
    }
    set
}

impl RngSubsetExtension for Rng {
    fn subset(&mut self, m: usize, n: usize) -> Vec<usize> {
        assert!(m < n);
        if m > n / 2 {
            let k = n - m;
            let exclude = floyd_sample(k, n, self);
            let mut res = Vec::with_capacity(m);
            for i in 0..n {
                if !exclude.contains(&i) {
                    res.push(i);
                }
            }
            return res;
        }
        floyd_sample(m, n, self).into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn floyd_sample_draws_unique_values() {
        let mut rng = Rng::with_seed(0);
        let sample = floyd_sample(4, 16, &mut rng);
        assert_eq!(sample.len(), 4, "must return exactly m entries");
        assert!(sample.iter().all(|&value| value < 16));
        let mut sorted: Vec<_> = sample.iter().copied().collect();
        sorted.sort_unstable();
        assert_eq!(
            sorted,
            vec![0, 1, 4, 7],
            "sampling should be deterministic with seed"
        );
    }

    #[test]
    fn subset_prefers_complement_for_large_m() {
        let mut rng = Rng::with_seed(0);
        let picks = rng.subset(8, 10);
        assert_eq!(picks.len(), 8);
        let unique: HashSet<_> = picks.iter().copied().collect();
        assert_eq!(unique.len(), picks.len(), "values must be unique");
        let missing: Vec<_> = (0..10).filter(|idx| !unique.contains(idx)).collect();
        assert_eq!(missing.len(), 2, "exactly n-m elements should be excluded");
        assert_eq!(
            missing,
            vec![0, 5],
            "complement should be deterministic with seed 0"
        );
    }

    #[test]
    fn subset_handles_small_samples() {
        let mut rng = Rng::with_seed(0);
        let picks = rng.subset(3, 10);
        assert_eq!(picks.len(), 3);
        assert!(picks.iter().all(|&value| value < 10));
        let mut sorted = picks.clone();
        sorted.sort_unstable();
        assert!(
            sorted.windows(2).all(|pair| pair[0] != pair[1]),
            "duplicates detected"
        );
        assert_eq!(
            sorted,
            vec![0, 4, 9],
            "sample should be deterministic with seed 0"
        );
    }
}
