//! Efficient helper methods to calculate Clebsch-Gordon coefficients and Wigner 3j-symbols.
//!
//! Ported from <https://github.com/0382/WignerSymbol> to Rust by N. D. Hoffman, 2026.

use num::complex::Complex64;

use crate::utils::wigner::utils::{binomial, const_imax, const_umin};

mod utils {
    const MAX_BINOMIAL: u64 = 67;
    const SIZE: usize = table_size(MAX_BINOMIAL);

    const fn table_size(n: u64) -> usize {
        let x = n / 2 + 1;
        (x * (x + (n & 1))) as usize
    }
    const fn index(n: u64, k: u64) -> usize {
        let x = n / 2 + 1;
        (x * (x - (1 - (n & 1))) + k) as usize
    }
    pub(crate) const fn const_umin(a: u64, b: u64) -> u64 {
        if a < b {
            a
        } else {
            b
        }
    }
    pub(crate) const fn const_imax(a: i64, b: i64) -> i64 {
        if a > b {
            a
        } else {
            b
        }
    }
    const fn build_binomial_table() -> [u64; SIZE] {
        let mut data = [0u64; SIZE];
        data[0] = 1;
        let mut n = 1;
        while n <= MAX_BINOMIAL {
            let mut k = 0;
            while k <= n / 2 {
                let value = if k == 0 {
                    1
                } else {
                    let nm1 = n - 1;
                    let a_k = const_umin(k, nm1 - k);
                    let km1 = k - 1;
                    let b_k = const_umin(km1, nm1 - km1);
                    data[index(nm1, a_k)] + data[index(nm1, b_k)]
                };
                data[index(n, k)] = value;
                k += 1;
            }
            n += 1;
        }
        data
    }
    static BINOMIAL_TABLE: [u64; SIZE] = build_binomial_table();

    #[inline]
    pub(crate) const fn binomial(n: u64, k: u64) -> u64 {
        if n > MAX_BINOMIAL || k > n {
            return 0;
        }
        let k = const_umin(k, n - k);
        BINOMIAL_TABLE[index(n, k)]
    }

    #[cfg(test)]
    mod tests {
        use super::binomial;
        #[test]
        fn test_binomial() {
            assert_eq!(binomial(0, 0), 1);
            assert_eq!(binomial(5, 0), 1);
            assert_eq!(binomial(5, 1), 5);
            assert_eq!(binomial(5, 2), 10);
            assert_eq!(binomial(5, 3), 10);
            assert_eq!(binomial(5, 4), 5);
            assert_eq!(binomial(5, 5), 1);
            assert_eq!(binomial(67, 33), 14_226_520_737_620_288_370);
            assert_eq!(binomial(67, 34), 14_226_520_737_620_288_370);
            assert_eq!(binomial(68, 1), 0);
            assert_eq!(binomial(10, 11), 0);
        }
    }
}

/// (-1)^x but efficient
#[inline]
const fn phase(x: u64) -> i64 {
    1 - (2 * (x & 1) as i64)
}

/// true if both j and m are either both integers or both half-integers, false for mixed cases
#[inline]
const fn check_parity(dj: i64, dm: i64) -> bool {
    (dj ^ dm) & 1 == 0
}
#[inline]
const fn check_jm(dj: i64, dm: i64) -> bool {
    check_parity(dj, dm) && (dm.abs() <= dj)
}
#[inline]
const fn check_coupling(dj1: i64, dj2: i64, dj3: i64) -> bool {
    (dj1 >= 0)
        && (dj2 >= 0)
        && (dj3 >= (dj1 - dj2).abs())
        && check_parity(dj1 + dj2, dj3)
        && (dj3 <= (dj1 + dj2))
}

/// Computes the Clebsch–Gordon coefficient $`\langle j_1 m_1; j_2 m_2 \mid j_3 m_3\rangle`$ using the doubled-quantum-number convention.
///
/// # Parameters
///
/// All angular-momentum values are represented in doubled form:
/// - `dj1 = 2*j₁`
/// - `dj2 = 2*j₂`
/// - `dj3 = 2*j₃`
/// - `dm1 = 2*m₁`
/// - `dm2 = 2*m₂`
/// - `dm3 = 2*m₃`
///
/// This lets the function work with both integer and half-integer quantum numbers
/// using integer arithmetic.
///
/// # Returns
///
/// Returns the Clebsch–Gordon coefficient as `f64`.
///
/// The function returns `0.0` when any standard selection rule is violated:
/// - `(dj, dm)` is not a valid angular-momentum projection pair
/// - `j₁`, `j₂`, and `j₃` do not satisfy the triangle relation
/// - `m₁ + m₂ != m₃`
///
/// # Examples
///
/// Integer angular momentum:
///
/// ```rust
/// # use laddu_core::utils::wigner::clebsch_gordon;
/// let cg = clebsch_gordon(2, 2, 2, 2, -2, 0); // ⟨1,1; 1,-1 | 1,0⟩
/// assert_eq!(cg, f64::sqrt(1.0 / 2.0));
/// ```
///
/// Half-integer angular momentum:
///
/// ```rust
/// # use laddu_core::utils::wigner::clebsch_gordon;
/// let cg = clebsch_gordon(1, 1, 2, 1, -1, 0); // ⟨1/2,1/2; 1/2,-1/2 | 1,0⟩
/// assert_eq!(cg, f64::sqrt(1.0 / 2.0));
/// ```
///
/// # Convention
///
/// This coefficient is related to the Wigner 3-j symbol by
///
/// $`\langle j_1 m_1; j_2 m_2 \mid j_3 m_3\rangle = (-1)^{j_1 - j_2 + m_3} \sqrt{2j_3 + 1}\begin{pmatrix}j_1 & j_2 & j_3\\m_1 & m_2 & -m_3\end{pmatrix}`$
pub fn clebsch_gordon(dj1: u64, dj2: u64, dj3: u64, dm1: i64, dm2: i64, dm3: i64) -> f64 {
    if !(check_jm(dj1 as i64, dm1) && check_jm(dj2 as i64, dm2) && check_jm(dj3 as i64, dm3)) {
        return 0.0;
    }
    if !check_coupling(dj1 as i64, dj2 as i64, dj3 as i64) {
        return 0.0;
    }
    if dm1 + dm2 != dm3 {
        return 0.0;
    }
    if dm1 == 0 && dm2 == 0 && dm3 == 0 {
        let j1 = dj1 / 2;
        let j2 = dj2 / 2;
        let j3 = dj3 / 2;
        let j = j1 + j2 + j3;
        let g = j / 2;
        return phase(g - j3) as f64 * (binomial(g, j3) * binomial(j3, g - j1)) as f64
            / ((binomial(j + 1, dj3 + 1) * binomial(dj3, j - dj1)) as f64).sqrt();
    }
    let j = (dj1 + dj2 + dj3) / 2;
    let jm1 = j - dj1;
    let jm2 = j - dj2;
    let jm3 = j - dj3;
    let j1mm1 = (dj1 as i64 - dm1) as u64 / 2;
    let j2mm2 = (dj2 as i64 - dm2) as u64 / 2;
    let j3mm3 = (dj3 as i64 - dm3) as u64 / 2;
    let j2pm2 = (dj2 as i64 + dm2) as u64 / 2;
    let a = ((binomial(dj1, jm2) * binomial(dj2, jm3)) as f64
        / (binomial(j + 1, jm3)
            * binomial(dj1, j1mm1)
            * binomial(dj2, j2mm2)
            * binomial(dj3, j3mm3)) as f64)
        .sqrt();
    let mut b: i64 = 0;
    let k_min = const_imax(
        0,
        const_imax(j1mm1 as i64 - jm2 as i64, j2pm2 as i64 - jm1 as i64),
    ) as u64;
    let k_max = const_umin(jm3, const_umin(j1mm1, j2pm2));
    for z in k_min..=k_max {
        b = -b + (binomial(jm3, z) * binomial(jm2, j1mm1 - z) * binomial(jm1, j2pm2 - z)) as i64;
    }
    a * (phase(k_max) * b) as f64
}

/// Computes the Wigner 3-j symbol
///
/// $`\begin{pmatrix} j_1 & j_2 & j_3 \\ m_1 & m_2 & m_3 \end{pmatrix}`$
///
/// using the doubled-quantum-number convention.
///
/// # Parameters
///
/// All quantum numbers are passed in doubled form:
/// - `dj1 = 2*j₁`
/// - `dj2 = 2*j₂`
/// - `dj3 = 2*j₃`
/// - `dm1 = 2*m₁`
/// - `dm2 = 2*m₂`
/// - `dm3 = 2*m₃`
///
/// This representation supports both integer and half-integer values
/// without using floating-point input.
///
/// # Returns
///
/// Returns the Wigner 3-j symbol as `f64`.
///
/// The function returns `0.0` when any selection rule fails:
/// - `(dj, dm)` is invalid for any input pair
/// - `j₁`, `j₂`, and `j₃` violate the triangle condition
/// - `m₁ + m₂ + m₃ != 0`
///
/// # Notes
///
/// - The implementation uses a finite alternating sum over binomial factors.
///
/// # Examples
///
/// Integer angular momentum:
///
/// ```rust
/// # use laddu_core::utils::wigner::wigner_3j;
/// let w = wigner_3j(2, 2, 2, 2, -2, 0); // (1 1 1; 1 -1 0)
/// assert_eq!(w, f64::sqrt(1.0 / 6.0))
/// ```
///
/// Half-integer angular momentum:
///
/// ```rust
/// # use laddu_core::utils::wigner::wigner_3j;
/// let w = wigner_3j(1, 1, 2, 1, -1, 0); // (1/2 1/2 1; 1/2 -1/2 0)
/// assert_eq!(w, f64::sqrt(1.0 / 6.0))
/// ```
///
/// # Convention
///
/// This symbol is related to the Clebsch–Gordon coefficient by
///
/// $`\begin{pmatrix} j_1 & j_2 & j_3 \\ m_1 & m_2 & m_3 \end{pmatrix} = (-1)^{j_1 - j_2 - m_3} \sqrt{2j_3 + 1} \langle j_1 m_1; j_2 m_2 | j_3, -m_3\rangle`$
pub fn wigner_3j(dj1: u64, dj2: u64, dj3: u64, dm1: i64, dm2: i64, dm3: i64) -> f64 {
    if !(check_jm(dj1 as i64, dm1) && check_jm(dj2 as i64, dm2) && check_jm(dj3 as i64, dm3)) {
        return 0.0;
    }
    if !check_coupling(dj1 as i64, dj2 as i64, dj3 as i64) {
        return 0.0;
    }
    if dm1 + dm2 + dm3 != 0 {
        return 0.0;
    }
    let j = (dj1 + dj2 + dj3) / 2;
    let jm1 = j - dj1;
    let jm2 = j - dj2;
    let jm3 = j - dj3;
    let j1mm1 = (dj1 as i64 - dm1) as u64 / 2;
    let j2mm2 = (dj2 as i64 - dm2) as u64 / 2;
    let j3mm3 = (dj3 as i64 - dm3) as u64 / 2;
    let j1pm1 = (dj1 as i64 + dm1) as u64 / 2;
    let a = ((binomial(dj1, jm2) * binomial(dj2, jm1)) as f64
        / ((j + 1)
            * binomial(j, jm3)
            * binomial(dj1, j1mm1)
            * binomial(dj2, j2mm2)
            * binomial(dj3, j3mm3)) as f64)
        .sqrt();
    let mut b: i64 = 0;
    let k_min = const_imax(
        0,
        const_imax(j1pm1 as i64 - jm2 as i64, j2mm2 as i64 - jm1 as i64),
    ) as u64;
    let k_max = const_umin(jm3, const_umin(j1pm1, j2mm2));
    for z in k_min..=k_max {
        b = -b + (binomial(jm3, z) * binomial(jm2, j1pm1 - z) * binomial(jm1, j2mm2 - z)) as i64;
    }
    a * (phase(dj1 + (dj3 as i64 + dm3) as u64 / 2 + k_max) * b) as f64
}

/// Precomputed helper for Wigner rotation matrix elements.
///
/// Stores all coefficients needed for repeated evaluation of
///
/// $`d^j_{m' m}(\beta)`$ and $`D^j_{m' m}(\alpha,\beta,\gamma)`$.
///
/// # Convention
///
/// Doubled quantum numbers:
///
/// - $`dj = 2j`$
/// - $`dmp = 2m'`$
/// - $`dm = 2m`$
///
/// # Definitions
///
/// $`D^j_{m' m}(\alpha,\beta,\gamma) = e^{-i m' \alpha} d^j_{m' m}(\beta) e^{-i m \gamma}`$
///
/// # Notes
///
/// Designed for reuse across many angle evaluations.
pub struct WignerDMatrix {
    dj: i64,    // 2 * j
    dmp: i64,   // 2 * m'
    dm: i64,    // 2 * m
    jpm: i64,   // j + m
    jmmp: i64,  // j - m'
    delta: i64, // m' - m
    s_min: i64,
    s_max: i64,
    p_c0: i32,  // 2j+m-m'-2s_min
    p_s0: i32,  // m'-m+2s_min
    amp0: f64,  // initial prefactor / denominator with s=s_min
    sign0: f64, // 1 if s_min is even else -1
}
impl WignerDMatrix {
    /// Constructs a Wigner small-$`d`$/full-$`D`$ matrix element helper for fixed
    /// quantum numbers $`j`$, $`m'`$, and $`m`$.
    ///
    /// All angular-momentum quantum numbers are passed in doubled form:
    /// - `dj = 2*j`
    /// - `dmp = 2*m'`
    /// - `dm = 2*m`
    ///
    /// This allows both integer and half-integer values to be represented
    /// exactly using integer types.
    ///
    /// The constructed value precomputes the combinatorial factors and
    /// summation bounds needed for repeated evaluation of:
    /// - the reduced Wigner matrix element $`d^j_{m' m}(\beta)`$, and
    /// - the full Wigner matrix element $`D^j_{m' m}(\alpha,\beta,\gamma)`$.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `|m'| > j`
    /// - `|m| > j`
    /// - `j` and `m'` do not have matching integer/half-integer parity
    /// - `j` and `m` do not have matching integer/half-integer parity
    /// - the internally derived summation bounds are inconsistent
    ///
    /// # Notes
    ///
    /// The returned struct is intended for reuse when evaluating the same
    /// matrix element for many angles. This avoids recomputing factorial-based
    /// prefactors on every call.
    ///
    /// # Examples
    ///
    /// Integer angular momentum:
    ///
    /// ```rust
    /// use laddu_core::utils::wigner::WignerDMatrix;
    /// let w = WignerDMatrix::new(2, 2, 0); // j = 1, m' = 1, m = 0
    /// ```
    ///
    /// Half-integer angular momentum:
    ///
    /// ```rust
    /// use laddu_core::utils::wigner::WignerDMatrix;
    /// let w = WignerDMatrix::new(1, 1, -1); // j = 1/2, m' = 1/2, m = -1/2
    /// ```
    pub fn new(dj: u64, dmp: i64, dm: i64) -> Self {
        let dj = dj as i64;
        assert!(
            dmp.abs() <= dj,
            "|m'| > j is not allowed! (2*j = {}, 2*m' = {})",
            dj,
            dmp
        );
        assert!(
            dm.abs() <= dj,
            "|m| > j is not allowed! (2*j = {}, 2*m = {})",
            dj,
            dm
        );
        assert!(
            check_parity(dj, dmp),
            "j and m' must either both be integers or both be half-integers! (2*j = {}, 2*m' = {})",
            dj,
            dmp
        );
        assert!(
            check_parity(dj, dm),
            "j and m must either both be integers or both be half-integers! (2*j = {}, 2*m = {})",
            dj,
            dm
        );
        let jpmp = (dj + dmp) / 2;
        let jmmp = (dj - dmp) / 2;
        let jpm = (dj + dm) / 2;
        let jmm = (dj - dm) / 2;
        let delta = (dmp - dm) / 2;
        let s_min = 0.max(-delta);
        let s_max = jpm.min(jmmp);
        assert!(
            s_min <= s_max,
            "summation bounds are incorrect (this shouldn't happen)!"
        );
        let mut ln_factorial = vec![0.0; dj as usize + 1];
        for i in 1..=dj as usize {
            ln_factorial[i] = ln_factorial[i - 1] + (i as f64).ln();
        }
        let ln_prefactor = 0.5
            * (ln_factorial[jpmp as usize]
                + ln_factorial[jmmp as usize]
                + ln_factorial[jpm as usize]
                + ln_factorial[jmm as usize]);
        let denom_ln_s_min = ln_factorial[(jpm - s_min) as usize]
            + ln_factorial[s_min as usize]
            + ln_factorial[(delta + s_min) as usize]
            + ln_factorial[(jmmp - s_min) as usize];
        let p_c0 = (dj - delta - 2 * s_min) as i32;
        let p_s0 = (delta + 2 * s_min) as i32;
        let amp0 = (ln_prefactor - denom_ln_s_min).exp();
        let sign0 = if ((s_min + delta) & 1) == 0 {
            1.0
        } else {
            -1.0
        };
        Self {
            dj,
            dmp,
            dm,
            jpm,
            jmmp,
            delta,
            s_min,
            s_max,
            p_c0,
            p_s0,
            amp0,
            sign0,
        }
    }
    #[inline(always)]
    fn d_half(&self, ch: f64, sh: f64) -> f64 {
        if sh.abs() < f64::EPSILON {
            return if self.dmp == self.dm { 1.0 } else { 0.0 };
        }
        if ch.abs() < f64::EPSILON {
            if self.dmp != -self.dm {
                return 0.0;
            }
            return if (((self.dj + self.dm) / 2) & 1) == 0 {
                1.0
            } else {
                -1.0
            };
        }
        let ratio = (sh * sh) / (ch * ch);
        let mut term = self.sign0 * self.amp0 * ch.powi(self.p_c0) * sh.powi(self.p_s0);
        let mut sum = term;
        for s in self.s_min..self.s_max {
            let num = (self.jpm - s) as f64 * (self.jmmp - s) as f64;
            let den = (s + 1) as f64 * (self.delta + s + 1) as f64;
            term *= -num / den * ratio;
            sum += term;
        }
        sum
    }
    /// Evaluates the reduced Wigner small-$`d`$ matrix element $`d^j_{m' m}(\beta)`$.
    ///
    /// The quantum numbers $`j`$, $`m'`$, and $`m`$ are those fixed when the
    /// [`WignerDMatrix`] was constructed.
    ///
    /// # Parameters
    ///
    /// - `beta`: the middle Euler angle $`\beta`$, in radians
    ///
    /// # Returns
    ///
    /// Returns the real-valued reduced Wigner matrix element $`d^j_{m' m}(\beta)`$.
    ///
    /// # Notes
    ///
    /// This method evaluates the standard finite sum in powers of
    /// $`\cos(\beta/2)`$ and $`\sin(\beta/2)`$, with special handling near
    /// $`\sin(\beta/2) = 0`$ and $`\cos(\beta/2) = 0`$ to avoid unnecessary numerical
    /// instability.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use laddu_core::utils::wigner::WignerDMatrix;
    /// # use approx::assert_relative_eq;
    /// let w = WignerDMatrix::new(2, 2, 0); // j = 1, m' = 1, m = 0
    /// let val = w.d(std::f64::consts::FRAC_PI_2);
    /// assert_relative_eq!(val, -std::f64::consts::FRAC_1_SQRT_2);
    /// ```
    #[inline(always)]
    pub fn d(&self, beta: f64) -> f64 {
        let h = 0.5 * beta;
        self.d_half(h.cos(), h.sin())
    }

    /// Evaluates the full Wigner $`D`$ matrix element $`D^j_{m' m}(\alpha,\beta,\gamma)`$.
    ///
    /// The implemented convention is
    ///
    /// $`D^j_{m' m}(\alpha,\beta,\gamma) = e^{-\imath m' \alpha} d^j_{m' m}(\beta) e^{-\imath m \gamma}`$,
    ///
    /// # Parameters
    ///
    /// - `alpha`: first Euler angle $`\alpha`$, in radians
    /// - `beta`: middle Euler angle $`\beta`$, in radians
    /// - `gamma`: third Euler angle $`\gamma`$, in radians
    ///
    /// # Returns
    ///
    /// Returns the complex Wigner $`D`$ matrix element as `Complex64`.
    ///
    /// # Notes
    ///
    /// Since $`d^j_{m' m}(\beta)`$ is real for real $`\beta`$, the complex phase comes
    /// entirely from the $`\alpha`$ and $`\gamma`$ dependence.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use laddu_core::utils::wigner::WignerDMatrix;
    /// # use approx::assert_relative_eq;
    /// # use num::complex::Complex64;
    /// let w = WignerDMatrix::new(2, 2, 0); // j = 1, m' = 1, m = 0
    /// let val = w.D(0.1, 0.2, 0.3);
    /// let truth =  Complex64::cis(-0.1) * -std::f64::consts::FRAC_1_SQRT_2 * f64::sin(0.2);
    /// assert_relative_eq!(val.re, truth.re);
    /// assert_relative_eq!(val.im, truth.im);
    /// ```
    #[inline(always)]
    #[allow(non_snake_case)]
    pub fn D(&self, alpha: f64, beta: f64, gamma: f64) -> Complex64 {
        let d = self.d(beta);
        let phi = -0.5 * ((self.dmp as f64) * alpha + (self.dm as f64) * gamma);
        Complex64::new(phi.cos() * d, phi.sin() * d)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use num::complex::Complex64;
    use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_2, PI};

    use super::*;
    #[test]
    fn test_phase() {
        assert_eq!(phase(0), 1);
        assert_eq!(phase(1), -1);
        assert_eq!(phase(2), 1);
        assert_eq!(phase(3), -1);
    }

    #[test]
    fn singlet_triplet_for_two_spin_half() {
        // <1/2,1/2; 1/2,1/2 | 1,1> = 1
        assert_relative_eq!(clebsch_gordon(1, 1, 2, 1, 1, 2), 1.0);

        // <1/2,1/2; 1/2,-1/2 | 1,0> = 1/sqrt(2)
        assert_relative_eq!(clebsch_gordon(1, 1, 2, 1, -1, 0), FRAC_1_SQRT_2);

        // <1/2,-1/2; 1/2,1/2 | 1,0> = 1/sqrt(2)
        assert_relative_eq!(clebsch_gordon(1, 1, 2, -1, 1, 0), FRAC_1_SQRT_2);

        // <1/2,1/2; 1/2,-1/2 | 0,0> = 1/sqrt(2)
        assert_relative_eq!(clebsch_gordon(1, 1, 0, 1, -1, 0), FRAC_1_SQRT_2);

        // <1/2,-1/2; 1/2,1/2 | 0,0> = -1/sqrt(2)
        assert_relative_eq!(clebsch_gordon(1, 1, 0, -1, 1, 0), -FRAC_1_SQRT_2);
    }

    #[test]
    fn highest_weight_state_is_one() {
        // <1,1; 1,1 | 2,2> = 1
        // doubled notation: j1=j2=1 -> dj1=dj2=2, m1=m2=1 -> dm1=dm2=2
        assert_relative_eq!(clebsch_gordon(2, 2, 4, 2, 2, 4), 1.0);
    }

    #[test]
    fn known_spin_one_couplings() {
        // <1,1; 1,0 | 2,1> = 1/sqrt(2)
        assert_relative_eq!(clebsch_gordon(2, 2, 4, 2, 0, 2), FRAC_1_SQRT_2);

        // <1,0; 1,0 | 2,0> = sqrt(2/3)
        assert_relative_eq!(clebsch_gordon(2, 2, 4, 0, 0, 0), (2.0 / 3.0_f64).sqrt());

        // <1,0; 1,0 | 0,0> = -1/sqrt(3)
        assert_relative_eq!(clebsch_gordon(2, 2, 0, 0, 0, 0), -1.0 / 3.0_f64.sqrt());
    }

    #[test]
    fn zero_when_m_sum_fails() {
        // dm1 + dm2 != dm3
        assert_eq!(clebsch_gordon(1, 1, 2, 1, 1, 0), 0.0);
    }

    #[test]
    fn zero_when_triangle_rule_fails() {
        // 1/2 + 1/2 cannot couple to j=2
        assert_eq!(clebsch_gordon(1, 1, 4, 1, 1, 2), 0.0);
    }

    #[test]
    fn zero_when_m_out_of_range() {
        // For dj=1 (j=1/2), dm must be ±1 only
        assert_eq!(clebsch_gordon(1, 1, 2, 3, -1, 2), 0.0);
    }

    #[test]
    fn normalization_for_fixed_jm() {
        // For j1=j2=1/2 and total J=1, M=0:
        // |<+,-|1,0>|^2 + |<-,+|1,0>|^2 = 1
        let c1 = clebsch_gordon(1, 1, 2, 1, -1, 0);
        let c2 = clebsch_gordon(1, 1, 2, -1, 1, 0);
        assert_relative_eq!(c1 * c1 + c2 * c2, 1.0);
    }

    #[test]
    fn normalization_for_singlet() {
        // For j1=j2=1/2 and total J=0, M=0:
        // |<+,-|0,0>|^2 + |<-,+|0,0>|^2 = 1
        let c1 = clebsch_gordon(1, 1, 0, 1, -1, 0);
        let c2 = clebsch_gordon(1, 1, 0, -1, 1, 0);
        assert_relative_eq!(c1 * c1 + c2 * c2, 1.0);
    }

    #[test]
    fn two_spin_half_cases() {
        // (1/2 1/2 1 ; 1/2 1/2 -1) = -1/sqrt(3)
        assert_relative_eq!(wigner_3j(1, 1, 2, 1, 1, -2), -1.0 / 3.0_f64.sqrt());

        // (1/2 1/2 0 ; 1/2 -1/2 0) = 1/sqrt(2)
        assert_relative_eq!(wigner_3j(1, 1, 0, 1, -1, 0), FRAC_1_SQRT_2);

        // (1/2 1/2 0 ; -1/2 1/2 0) = -1/sqrt(2)
        assert_relative_eq!(wigner_3j(1, 1, 0, -1, 1, 0), -FRAC_1_SQRT_2);
    }

    #[test]
    fn spin_one_cases() {
        // (1 1 0 ; 0 0 0) = -1/sqrt(3)
        assert_relative_eq!(wigner_3j(2, 2, 0, 0, 0, 0), -1.0 / 3.0_f64.sqrt());

        // (1 1 2 ; 1 -1 0) = 1/sqrt(30)
        assert_relative_eq!(wigner_3j(2, 2, 4, 2, -2, 0), 1.0 / 30.0_f64.sqrt());

        // (1 1 2 ; 0 0 0) = sqrt(2/15)
        assert_relative_eq!(wigner_3j(2, 2, 4, 0, 0, 0), (2.0 / 15.0_f64).sqrt());
    }

    #[test]
    fn selection_rule_failures_return_zero() {
        // m1 + m2 + m3 != 0
        assert_eq!(wigner_3j(1, 1, 0, 1, -1, 1), 0.0);

        // triangle rule fails: 1/2 + 1/2 cannot couple to 2
        assert_eq!(wigner_3j(1, 1, 4, 1, -1, 0), 0.0);

        // invalid m for j = 1/2
        assert_eq!(wigner_3j(1, 1, 0, 3, -1, -2), 0.0);
    }

    #[test]
    fn odd_j_sum_with_all_zero_ms_vanishes() {
        // For integer j's, (j1 j2 j3; 0 0 0) vanishes if j1+j2+j3 is odd.
        // Here 1+1+1 = 3 is odd.
        assert_eq!(wigner_3j(2, 2, 2, 0, 0, 0), 0.0);
    }

    #[test]
    fn column_swap_symmetry_even_case() {
        // Swapping first two columns gives factor (-1)^(j1+j2+j3).
        // Here 1+1+2 = 4 is even, so unchanged.
        let a = wigner_3j(2, 2, 4, 2, -2, 0);
        let b = wigner_3j(2, 2, 4, -2, 2, 0);
        assert_relative_eq!(a, b);
    }

    #[test]
    fn column_swap_symmetry_odd_case() {
        // Here 1/2 + 1/2 + 0 = 1 is odd, so swap picks up a minus sign.
        let a = wigner_3j(1, 1, 0, 1, -1, 0);
        let b = wigner_3j(1, 1, 0, -1, 1, 0);
        assert_relative_eq!(a, -b);
    }

    #[test]
    fn sign_flip_symmetry() {
        // (j1 j2 j3; -m1 -m2 -m3) = (-1)^(j1+j2+j3) (j1 j2 j3; m1 m2 m3)
        // For j1=j2=1/2, j3=0, total is odd => minus sign.
        let a = wigner_3j(1, 1, 0, 1, -1, 0);
        let b = wigner_3j(1, 1, 0, -1, 1, 0);
        assert_relative_eq!(b, -a);

        // For j1=j2=1, j3=2, total is even => same sign.
        let c = wigner_3j(2, 2, 4, 2, -2, 0);
        let d = wigner_3j(2, 2, 4, -2, 2, 0);
        assert_relative_eq!(d, c);
    }

    #[test]
    fn relation_to_clebsch_gordon_examples() {
        // Using:
        // (j1 j2 j3; m1 m2 -m3) = (-1)^(j1-j2+m3) / sqrt(2j3+1) * <j1 m1 j2 m2 | j3 m3>
        //
        // In doubled notation the phase exponent becomes (dj1 - dj2 + dm3)/2.

        // <1/2,1/2; 1/2,-1/2 | 0,0> = 1/sqrt(2)
        // => (1/2 1/2 0 ; 1/2 -1/2 0) = 1/sqrt(2)
        let cg = clebsch_gordon(1, 1, 0, 1, -1, 0);
        let w3j = wigner_3j(1, 1, 0, 1, -1, 0);
        assert_relative_eq!(w3j, cg);

        // <1,1; 1,-1 | 2,0> = 1/sqrt(6)
        // => (1 1 2 ; 1 -1 0) = 1/sqrt(30)
        let cg = clebsch_gordon(2, 2, 4, 2, -2, 0);
        let expected = cg / 5.0_f64.sqrt(); // sqrt(2j3+1)=sqrt(5)
        let w3j = wigner_3j(2, 2, 4, 2, -2, 0);
        assert_relative_eq!(w3j, expected);
    }

    #[test]
    fn construct_integer_case() {
        let _ = WignerDMatrix::new(2, 2, 0); // j = 1, m' = 1, m = 0
    }

    #[test]
    fn construct_half_integer_case() {
        let _ = WignerDMatrix::new(1, 1, -1); // j = 1/2, m' = 1/2, m = -1/2
    }

    #[test]
    #[should_panic]
    fn panic_when_mp_out_of_range() {
        let _ = WignerDMatrix::new(2, 4, 0);
    }

    #[test]
    #[should_panic]
    fn panic_when_m_out_of_range() {
        let _ = WignerDMatrix::new(2, 0, 4);
    }

    #[test]
    #[should_panic]
    fn panic_when_parity_mismatch_mp() {
        let _ = WignerDMatrix::new(2, 1, 0);
    }

    #[test]
    #[should_panic]
    fn panic_when_parity_mismatch_m() {
        let _ = WignerDMatrix::new(2, 0, 1);
    }

    #[test]
    fn d_beta_zero_is_identity_integer_j() {
        let vals = [-2, 0, 2];
        for &mp in &vals {
            for &m in &vals {
                let w = WignerDMatrix::new(2, mp, m);
                let expected = if mp == m { 1.0 } else { 0.0 };
                assert_relative_eq!(w.d(0.0), expected);
            }
        }
    }

    #[test]
    fn d_beta_zero_is_identity_half_integer_j() {
        let vals = [-1, 1];
        for &mp in &vals {
            for &m in &vals {
                let w = WignerDMatrix::new(1, mp, m);
                let expected = if mp == m { 1.0 } else { 0.0 };
                assert_relative_eq!(w.d(0.0), expected);
            }
        }
    }

    #[test]
    fn d_beta_pi_selection_rule_j_one() {
        let vals = [-2, 0, 2];
        for &mp in &vals {
            for &m in &vals {
                let w = WignerDMatrix::new(2, mp, m);
                let expected = if mp == -m {
                    let jm = (2 + m) / 2; // j + m with j=1
                    if (jm & 1) == 0 {
                        1.0
                    } else {
                        -1.0
                    }
                } else {
                    0.0
                };
                assert_relative_eq!(w.d(PI), expected);
            }
        }
    }

    #[test]
    fn d_j_half_closed_forms() {
        let beta = 0.73;
        let c = f64::cos(beta / 2.0);
        let s = f64::sin(beta / 2.0);

        let w_pp = WignerDMatrix::new(1, 1, 1);
        let w_pm = WignerDMatrix::new(1, 1, -1);
        let w_mp = WignerDMatrix::new(1, -1, 1);
        let w_mm = WignerDMatrix::new(1, -1, -1);

        assert_relative_eq!(w_pp.d(beta), c);
        assert_relative_eq!(w_pm.d(beta), -s);
        assert_relative_eq!(w_mp.d(beta), s);
        assert_relative_eq!(w_mm.d(beta), c);
    }

    #[test]
    fn d_j_one_closed_forms() {
        let beta = 1.1;
        let cb = f64::cos(beta);
        let sb = f64::sin(beta);

        let w_11 = WignerDMatrix::new(2, 2, 2);
        let w_10 = WignerDMatrix::new(2, 2, 0);
        let w_1m1 = WignerDMatrix::new(2, 2, -2);
        let w_00 = WignerDMatrix::new(2, 0, 0);
        let w_01 = WignerDMatrix::new(2, 0, 2);
        let w_0m1 = WignerDMatrix::new(2, 0, -2);
        let w_m11 = WignerDMatrix::new(2, -2, 2);
        let w_m10 = WignerDMatrix::new(2, -2, 0);
        let w_m1m1 = WignerDMatrix::new(2, -2, -2);

        assert_relative_eq!(w_11.d(beta), 0.5 * (1.0 + cb));
        assert_relative_eq!(w_10.d(beta), -FRAC_1_SQRT_2 * sb);
        assert_relative_eq!(w_1m1.d(beta), 0.5 * (1.0 - cb));

        assert_relative_eq!(w_01.d(beta), FRAC_1_SQRT_2 * sb);
        assert_relative_eq!(w_00.d(beta), cb);
        assert_relative_eq!(w_0m1.d(beta), -FRAC_1_SQRT_2 * sb);

        assert_relative_eq!(w_m11.d(beta), 0.5 * (1.0 - cb));
        assert_relative_eq!(w_m10.d(beta), FRAC_1_SQRT_2 * sb);
        assert_relative_eq!(w_m1m1.d(beta), 0.5 * (1.0 + cb));
    }

    #[test]
    fn d_j_one_special_value() {
        let w = WignerDMatrix::new(2, 2, 0);
        assert_relative_eq!(w.d(FRAC_PI_2), -FRAC_1_SQRT_2);
    }

    #[test]
    fn full_d_matches_phase_definition() {
        let alpha = 0.31;
        let beta = 0.82;
        let gamma = -0.47;

        let w = WignerDMatrix::new(3, 1, -1); // j = 3/2, m' = 1/2, m = -1/2
        let d = w.d(beta);
        let expected_phase = Complex64::cis(-0.5 * (alpha - gamma));
        let expected = expected_phase * d;
        let val = w.D(alpha, beta, gamma);

        assert_relative_eq!(val.re, expected.re);
        assert_relative_eq!(val.im, expected.im);
    }

    #[test]
    fn full_d_has_no_gamma_dependence_when_m_zero() {
        let w = WignerDMatrix::new(2, 2, 0); // j=1, m'=1, m=0
        let a = w.D(0.2, 0.7, 0.0);
        let b = w.D(0.2, 0.7, 1.3);

        assert_relative_eq!(a.re, b.re);
        assert_relative_eq!(a.im, b.im);
    }

    #[test]
    fn full_d_has_no_alpha_dependence_when_mp_zero() {
        let w = WignerDMatrix::new(2, 0, 2); // j=1, m'=0, m=1
        let a = w.D(0.0, 0.7, 0.2);
        let b = w.D(1.3, 0.7, 0.2);

        assert_relative_eq!(a.re, b.re);
        assert_relative_eq!(a.im, b.im);
    }

    #[test]
    fn d_symmetry_minus_indices() {
        let beta = 0.91;
        let w1 = WignerDMatrix::new(4, 2, -2); // j=2, m'=1, m=-1
        let w2 = WignerDMatrix::new(4, -2, 2); // j=2, m'=-1, m=1

        let lhs = w1.d(beta);
        let rhs = w2.d(beta);

        // d^j_{m' m}(beta) = (-1)^(m'-m) d^j_{-m', -m}(beta)
        // here m'-m = 2, so sign is +1
        assert_relative_eq!(lhs, rhs);
    }

    #[test]
    fn d_symmetry_transpose_relation() {
        let beta = 0.64;
        let w1 = WignerDMatrix::new(3, 1, -1); // j=3/2, m'=1/2, m=-1/2
        let w2 = WignerDMatrix::new(3, -1, 1); // swapped

        let lhs = w1.d(beta);
        let rhs = w2.d(beta);

        // d^j_{m' m}(beta) = (-1)^(m'-m) d^j_{m m'}(beta)
        // here m'-m = 1, so sign is -1
        assert_relative_eq!(lhs, -rhs);
    }

    #[test]
    fn d_is_real_for_real_beta() {
        let w = WignerDMatrix::new(6, 2, -4); // j=3, m'=1, m=-2
        let d = w.d(0.37);
        assert!(d.is_finite());
    }

    #[test]
    fn full_d_magnitude_equals_abs_small_d() {
        let w = WignerDMatrix::new(5, 1, -3); // j=5/2, m'=1/2, m=-3/2
        let d = w.d(1.23).abs();
        #[allow(non_snake_case)]
        let D = w.D(0.4, 1.23, -0.9).norm();
        assert_relative_eq!(D, d);
    }
}
