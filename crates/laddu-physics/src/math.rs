//! Special functions and numerical helpers used across amplitudes and variables.

mod functions;
mod wigner;

pub use functions::*;
pub use wigner::*;

/// Evaluate a literal Clebsch-Gordan coefficient.
///
/// This macro accepts `j1, m1, j2, m2 | j, m`, optionally wrapped in angle
/// brackets. Values may be integer literals or fractions with denominator `2`.
///
/// # Examples
///
/// ```rust
/// # use laddu_physics::clebsch_gordan;
/// let cg = clebsch_gordan!(<1 / 2, -1 / 2, 1, -1 | 3 / 2, -3 / 2>);
/// assert_eq!(cg, 1.0);
/// let cg = clebsch_gordan!(1, 1, 1, -1 | 1, 0);
/// assert_eq!(cg, f64::sqrt(1.0 / 2.0));
/// ```
///
/// Incorrect integer/half-integer pairings are rejected at compile time:
///
/// ```compile_fail
/// # use laddu_physics::clebsch_gordan;
/// let _ = clebsch_gordan!(<1, 1 / 2, 1, -1 | 1, 0>);
/// ```
#[macro_export]
macro_rules! clebsch_gordan {
    (< $n:literal / 2, $($rest:tt)*) => {
        $crate::clebsch_gordan!(@m1_half [$crate::j!($n / 2)] $($rest)*)
    };
    (< $n:literal, $($rest:tt)*) => {
        $crate::clebsch_gordan!(@m1_int [$crate::j!($n)] $($rest)*)
    };
    ($n:literal / 2, $($rest:tt)*) => {
        $crate::clebsch_gordan!(@m1_half [$crate::j!($n / 2)] $($rest)*)
    };
    ($n:literal, $($rest:tt)*) => {
        $crate::clebsch_gordan!(@m1_int [$crate::j!($n)] $($rest)*)
    };
    (@m1_half [$j1:expr] - $n:literal / 2, $($rest:tt)*) => {
        $crate::clebsch_gordan!(@j2 [$j1, $crate::m!(-$n / 2)] $($rest)*)
    };
    (@m1_half [$j1:expr] $n:literal / 2, $($rest:tt)*) => {
        $crate::clebsch_gordan!(@j2 [$j1, $crate::m!($n / 2)] $($rest)*)
    };
    (@m1_int [$j1:expr] - $n:literal, $($rest:tt)*) => {
        $crate::clebsch_gordan!(@j2 [$j1, $crate::m!(-$n)] $($rest)*)
    };
    (@m1_int [$j1:expr] $n:literal, $($rest:tt)*) => {
        $crate::clebsch_gordan!(@j2 [$j1, $crate::m!($n)] $($rest)*)
    };
    (@j2 [$j1:expr, $m1:expr] $n:literal / 2, $($rest:tt)*) => {
        $crate::clebsch_gordan!(@m2_half [$j1, $m1, $crate::j!($n / 2)] $($rest)*)
    };
    (@j2 [$j1:expr, $m1:expr] $n:literal, $($rest:tt)*) => {
        $crate::clebsch_gordan!(@m2_int [$j1, $m1, $crate::j!($n)] $($rest)*)
    };
    (@m2_half [$j1:expr, $m1:expr, $j2:expr] - $n:literal / 2 | $($rest:tt)*) => {
        $crate::clebsch_gordan!(@j [$j1, $m1, $j2, $crate::m!(-$n / 2)] $($rest)*)
    };
    (@m2_half [$j1:expr, $m1:expr, $j2:expr] $n:literal / 2 | $($rest:tt)*) => {
        $crate::clebsch_gordan!(@j [$j1, $m1, $j2, $crate::m!($n / 2)] $($rest)*)
    };
    (@m2_int [$j1:expr, $m1:expr, $j2:expr] - $n:literal | $($rest:tt)*) => {
        $crate::clebsch_gordan!(@j [$j1, $m1, $j2, $crate::m!(-$n)] $($rest)*)
    };
    (@m2_int [$j1:expr, $m1:expr, $j2:expr] $n:literal | $($rest:tt)*) => {
        $crate::clebsch_gordan!(@j [$j1, $m1, $j2, $crate::m!($n)] $($rest)*)
    };
    (@j [$j1:expr, $m1:expr, $j2:expr, $m2:expr] $n:literal / 2, $($rest:tt)*) => {
        $crate::clebsch_gordan!(@m_half [$j1, $m1, $j2, $m2, $crate::j!($n / 2)] $($rest)*)
    };
    (@j [$j1:expr, $m1:expr, $j2:expr, $m2:expr] $n:literal, $($rest:tt)*) => {
        $crate::clebsch_gordan!(@m_int [$j1, $m1, $j2, $m2, $crate::j!($n)] $($rest)*)
    };
    (@m_half [$j1:expr, $m1:expr, $j2:expr, $m2:expr, $j:expr] - $n:literal / 2 $(>)?) => {
        $crate::math::clebsch_gordan($j1, $m1, $j2, $m2, $j, $crate::m!(-$n / 2))
    };
    (@m_half [$j1:expr, $m1:expr, $j2:expr, $m2:expr, $j:expr] $n:literal / 2 $(>)?) => {
        $crate::math::clebsch_gordan($j1, $m1, $j2, $m2, $j, $crate::m!($n / 2))
    };
    (@m_int [$j1:expr, $m1:expr, $j2:expr, $m2:expr, $j:expr] - $n:literal $(>)?) => {
        $crate::math::clebsch_gordan($j1, $m1, $j2, $m2, $j, $crate::m!(-$n))
    };
    (@m_int [$j1:expr, $m1:expr, $j2:expr, $m2:expr, $j:expr] $n:literal $(>)?) => {
        $crate::math::clebsch_gordan($j1, $m1, $j2, $m2, $j, $crate::m!($n))
    };
    ($($bad:tt)*) => {
        compile_error!(
            "expected clebsch_gordan!(<j1, m1, j2, m2 | j, m>) with each j/m pair both integer or both half-integer"
        )
    };
}

#[cfg(test)]
mod tests {
    #[test]
    fn clebsch_gordan_macro_accepts_wrapped_and_unwrapped_symbols() {
        assert_eq!(
            crate::clebsch_gordan!(<1 / 2, -1 / 2, 1, -1 | 3 / 2, -3 / 2>),
            1.0
        );
        assert_eq!(
            crate::clebsch_gordan!(1 / 2, -1 / 2, 1, -1 | 3 / 2, -3 / 2),
            1.0
        );
    }
}
