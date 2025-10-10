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
