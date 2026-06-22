use std::f64::consts::PI;

use num::complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::{
    LadduPhysicsError, LadduPhysicsResult,
    quantum::{L, M},
};

#[inline]
fn complex_is_finite(value: Complex64) -> bool {
    value.re.is_finite() && value.im.is_finite()
}

#[inline]
fn validate_finite_real(name: &str, value: f64) -> LadduPhysicsResult<()> {
    if !value.is_finite() {
        return Err(LadduPhysicsError::invalid_value(name, "finite", value));
    }
    Ok(())
}

#[inline]
fn validate_finite_complex(name: &str, value: Complex64) -> LadduPhysicsResult<()> {
    if !complex_is_finite(value) {
        return Err(LadduPhysicsError::invalid_value(name, "finite", value));
    }
    Ok(())
}

#[inline]
fn validate_real_mass(name: &str, value: f64) -> LadduPhysicsResult<()> {
    validate_finite_real(name, value)?;

    if value < 0.0 {
        return Err(LadduPhysicsError::invalid_value(name, "nonnegative", value));
    }

    Ok(())
}

#[inline]
fn validate_nonzero_real(name: &str, value: f64) -> LadduPhysicsResult<()> {
    validate_finite_real(name, value)?;

    if value == 0.0 {
        return Err(LadduPhysicsError::invalid_value(name, "nonzero", value));
    }

    Ok(())
}

#[inline]
fn validate_nonzero_complex(name: &str, value: Complex64) -> LadduPhysicsResult<()> {
    validate_finite_complex(name, value)?;

    if value == Complex64::new(0.0, 0.0) {
        return Err(LadduPhysicsError::invalid_value(name, "nonzero", value));
    }

    Ok(())
}

#[inline]
fn validate_parent_mass(name: &str, value: f64) -> LadduPhysicsResult<()> {
    validate_real_mass(name, value)?;

    if value == 0.0 {
        return Err(LadduPhysicsError::invalid_value(name, "nonzero", value));
    }

    Ok(())
}

#[inline]
fn validate_costheta(costheta: f64) -> LadduPhysicsResult<()> {
    validate_finite_real("costheta", costheta)?;

    if costheta.abs() > 1.0 {
        return Err(LadduPhysicsError::invalid_value(
            "costheta",
            "between -1 and 1 inclusive",
            costheta,
        ));
    }

    Ok(())
}

#[inline]
fn validate_spherical_harmonic_quantum_numbers(l: L, m: M) -> LadduPhysicsResult<usize> {
    if !m.is_integer() {
        return Err(LadduPhysicsError::invalid_relation(format!(
            "m must be an integer, got {m}"
        )));
    }
    let abs_m = (m.doubled() / 2).unsigned_abs();

    if abs_m > l.value() {
        return Err(LadduPhysicsError::invalid_relation(format!(
            "|m| <= l, got l = {l}, m = {m}"
        )));
    }

    if l.value().checked_add(abs_m).is_none() {
        return Err(LadduPhysicsError::numeric_overflow(format!(
            "l + |m| for l = {l}, m = {m}"
        )));
    }

    Ok(abs_m as usize)
}

#[inline]
fn validate_blatt_weisskopf_l(l: L) -> LadduPhysicsResult<()> {
    if l.value() as usize > BLATT_WEISSKOPF_MAX_L {
        return Err(LadduPhysicsError::unsupported_value(
            "l",
            format!("l <= {BLATT_WEISSKOPF_MAX_L}"),
            l,
        ));
    }

    Ok(())
}

#[inline]
fn validate_q_r(q_r: f64) -> LadduPhysicsResult<()> {
    validate_finite_real("q_r", q_r)?;

    if q_r <= 0.0 {
        return Err(LadduPhysicsError::invalid_value("q_r", "positive", q_r));
    }

    Ok(())
}

#[inline]
fn validate_tensor_barrier_q(l: L, q: Complex64) -> LadduPhysicsResult<()> {
    validate_finite_complex("q", q)?;

    if l.value() > 0 && q == Complex64::new(0.0, 0.0) {
        return Err(LadduPhysicsError::invalid_relation(format!(
            "tensor Blatt-Weisskopf factor is singular at q = 0 for l = {l}"
        )));
    }

    Ok(())
}

#[inline]
fn alp_pos_m_unchecked(l: usize, m: usize, x: f64) -> f64 {
    let mut p = 1.0;

    if l == 0 && m == 0 {
        return p;
    }

    let y = f64::sqrt(1.0 - x.powi(2));

    for m_p in 0..m {
        p *= -((2 * m_p + 1) as f64) * y;
    }

    if l == m {
        return p;
    }

    let mut p_min_2 = p;
    let mut p_min_1 = (2 * m + 1) as f64 * x * p_min_2;

    if l == m + 1 {
        return p_min_1;
    }

    for l_p in (m + 1)..l {
        p = ((2 * l_p + 1) as f64 * x * p_min_1 - (l_p + m) as f64 * p_min_2)
            / (l_p - m + 1) as f64;
        p_min_2 = p_min_1;
        p_min_1 = p;
    }

    p
}

#[inline]
fn factorial_ratio_l_minus_over_l_plus_unchecked(l: usize, abs_m: usize) -> f64 {
    let mut ratio = 1.0;

    for k in (l - abs_m + 1)..=(l + abs_m) {
        ratio /= k as f64;
    }

    ratio
}

/// A validated spherical-harmonic mode.
///
/// This stores fixed quantum numbers `l` and `m`, so they are validated once.
/// Use [`SphericalHarmonic::evaluate_unchecked`] in hot loops when `costheta`
/// and `phi` are already known to be valid.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SphericalHarmonic {
    l: L,
    m: M,
}

impl SphericalHarmonic {
    pub fn new(l: impl TryInto<L>, m: impl TryInto<M>) -> LadduPhysicsResult<Self> {
        let l = l
            .try_into()
            .map_err(|_| LadduPhysicsError::ConversionError("L"))?;
        let m = m
            .try_into()
            .map_err(|_| LadduPhysicsError::ConversionError("M"))?;
        validate_spherical_harmonic_quantum_numbers(l, m)?;
        Ok(Self { l, m })
    }

    #[inline]
    pub const fn l(self) -> L {
        self.l
    }

    #[inline]
    pub const fn m(self) -> M {
        self.m
    }

    pub fn evaluate(self, costheta: f64, phi: f64) -> LadduPhysicsResult<Complex64> {
        validate_costheta(costheta)?;
        validate_finite_real("phi", phi)?;

        Ok(self.evaluate_unchecked(costheta, phi))
    }

    /// Evaluate without validation.
    ///
    /// # Preconditions
    ///
    /// - `|m| <= l`, guaranteed by [`SphericalHarmonic::new`]
    /// - `costheta` is finite and satisfies `|costheta| <= 1`
    /// - `phi` is finite
    #[inline(always)]
    pub fn evaluate_unchecked(self, costheta: f64, phi: f64) -> Complex64 {
        spherical_harmonic_unchecked(
            self.l.value() as usize,
            self.m.doubled() as isize / 2,
            costheta,
            phi,
        )
    }
}

/// Computes the spherical harmonic $`Y_\ell^m(\theta, \phi)`$ given $`\cos\theta`$ and $`\phi`$.
///
/// This checked version validates `|m| <= l`, `|costheta| <= 1`, and finite inputs.
pub fn spherical_harmonic(l: L, m: M, costheta: f64, phi: f64) -> LadduPhysicsResult<Complex64> {
    SphericalHarmonic::new(l, m)?.evaluate(costheta, phi)
}

/// Computes the spherical harmonic without validation.
///
/// # Preconditions
///
/// - `|m| <= l`
/// - `l + |m|` does not overflow `usize`
/// - `costheta` is finite and satisfies `|costheta| <= 1`
/// - `phi` is finite
#[inline(always)]
pub fn spherical_harmonic_unchecked(l: usize, m: isize, costheta: f64, phi: f64) -> Complex64 {
    let abs_m = m.unsigned_abs();

    let mut res = alp_pos_m_unchecked(l, abs_m, costheta);

    res *= f64::sqrt(
        (2 * l + 1) as f64 / (4.0 * PI) * factorial_ratio_l_minus_over_l_plus_unchecked(l, abs_m),
    );

    if m < 0 && abs_m % 2 != 0 {
        res = -res;
    }

    Complex64::new(
        res * f64::cos(m as f64 * phi),
        res * f64::sin(m as f64 * phi),
    )
}

/// A validated two-body channel with fixed daughter masses.
///
/// The masses are validated once at construction. The checked methods validate
/// dynamic variables like `s` or `m`; the unchecked methods skip those checks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TwoBodyChannel {
    m1: f64,
    m2: f64,
}

impl TwoBodyChannel {
    pub fn new(m1: f64, m2: f64) -> LadduPhysicsResult<Self> {
        validate_real_mass("m1", m1)?;
        validate_real_mass("m2", m2)?;

        Ok(Self { m1, m2 })
    }

    #[inline]
    pub const fn m1(self) -> f64 {
        self.m1
    }

    #[inline]
    pub const fn m2(self) -> f64 {
        self.m2
    }

    #[inline]
    pub fn threshold(self) -> f64 {
        self.m1 + self.m2
    }

    #[inline]
    pub fn threshold_s(self) -> f64 {
        self.threshold().powi(2)
    }

    #[inline]
    pub fn pseudothreshold(self) -> f64 {
        (self.m1 - self.m2).abs()
    }

    #[inline]
    pub fn pseudothreshold_s(self) -> f64 {
        self.pseudothreshold().powi(2)
    }

    pub fn chi_plus(self, s: f64) -> LadduPhysicsResult<f64> {
        validate_nonzero_real("s", s)?;
        Ok(self.chi_plus_unchecked(s))
    }

    #[inline(always)]
    pub fn chi_plus_unchecked(self, s: f64) -> f64 {
        chi_plus_unchecked(s, self.m1, self.m2)
    }

    pub fn chi_minus(self, s: f64) -> LadduPhysicsResult<f64> {
        validate_nonzero_real("s", s)?;
        Ok(self.chi_minus_unchecked(s))
    }

    #[inline(always)]
    pub fn chi_minus_unchecked(self, s: f64) -> f64 {
        chi_minus_unchecked(s, self.m1, self.m2)
    }

    pub fn q_s(self, s: Complex64, sheet: Sheet) -> LadduPhysicsResult<Complex64> {
        validate_nonzero_complex("s", s)?;
        Ok(self.q_s_unchecked(s, sheet))
    }

    #[inline(always)]
    pub fn q_s_unchecked(self, s: Complex64, sheet: Sheet) -> Complex64 {
        q_s_unchecked(s, self.m1, self.m2, sheet)
    }

    pub fn q_m(self, m: f64, sheet: Sheet) -> LadduPhysicsResult<Complex64> {
        validate_parent_mass("m", m)?;
        Ok(self.q_m_unchecked(m, sheet))
    }

    #[inline(always)]
    pub fn q_m_unchecked(self, m: f64, sheet: Sheet) -> Complex64 {
        q_m_unchecked(m, self.m1, self.m2, sheet)
    }

    pub fn rho_s(self, s: Complex64, sheet: Sheet) -> LadduPhysicsResult<Complex64> {
        validate_nonzero_complex("s", s)?;
        Ok(self.rho_s_unchecked(s, sheet))
    }

    #[inline(always)]
    pub fn rho_s_unchecked(self, s: Complex64, sheet: Sheet) -> Complex64 {
        rho_s_unchecked(s, self.m1, self.m2, sheet)
    }

    pub fn rho_m(self, m: f64, sheet: Sheet) -> LadduPhysicsResult<Complex64> {
        validate_parent_mass("m", m)?;
        Ok(self.rho_m_unchecked(m, sheet))
    }

    #[inline(always)]
    pub fn rho_m_unchecked(self, m: f64, sheet: Sheet) -> Complex64 {
        rho_m_unchecked(m, self.m1, self.m2, sheet)
    }
}

/// Computes $`\chi_+(s, m_1, m_2) = 1 - \frac{(m_1 + m_2)^2}{s}`$.
pub fn chi_plus(s: f64, m1: f64, m2: f64) -> LadduPhysicsResult<f64> {
    TwoBodyChannel::new(m1, m2)?.chi_plus(s)
}

/// Computes $`\chi_+(s, m_1, m_2)`$ without validation.
///
/// # Preconditions
///
/// - `s` is finite and nonzero
/// - `m1` and `m2` are finite and nonnegative
#[inline(always)]
pub fn chi_plus_unchecked(s: f64, m1: f64, m2: f64) -> f64 {
    1.0 - (m1 + m2) * (m1 + m2) / s
}

/// Computes $`\chi_-(s, m_1, m_2) = 1 - \frac{(m_1 - m_2)^2}{s}`$.
pub fn chi_minus(s: f64, m1: f64, m2: f64) -> LadduPhysicsResult<f64> {
    TwoBodyChannel::new(m1, m2)?.chi_minus(s)
}

/// Computes $`\chi_-(s, m_1, m_2)`$ without validation.
///
/// # Preconditions
///
/// - `s` is finite and nonzero
/// - `m1` and `m2` are finite and nonnegative
#[inline(always)]
pub fn chi_minus_unchecked(s: f64, m1: f64, m2: f64) -> f64 {
    1.0 - (m1 - m2) * (m1 - m2) / s
}

/// Selects the Riemann sheet used for analytic continuation of two-body kinematic functions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Sheet {
    /// The physical sheet of the complex energy plane.
    Physical,
    /// The unphysical sheet obtained by flipping the sign of the channel momentum.
    Unphysical,
}

/// Computes the complex breakup momentum $`q(s)`$ for a two-body channel on a chosen Riemann sheet.
///
/// This checked version validates finite nonnegative masses and finite nonzero `s`.
pub fn q_s(s: Complex64, m1: f64, m2: f64, sheet: Sheet) -> LadduPhysicsResult<Complex64> {
    TwoBodyChannel::new(m1, m2)?.q_s(s, sheet)
}

/// Computes the complex breakup momentum $`q(s)`$ without validation.
///
/// # Preconditions
///
/// - `s` is finite and nonzero
/// - `m1` and `m2` are finite and nonnegative
#[inline(always)]
pub fn q_s_unchecked(s: Complex64, m1: f64, m2: f64, sheet: Sheet) -> Complex64 {
    let sp = Complex64::from((m1 + m2).powi(2));
    let sm = Complex64::from((m1 - m2).powi(2));
    let q_phys = ((s - sp) * (s - sm)).sqrt() / (2.0 * s.sqrt());

    match sheet {
        Sheet::Physical => q_phys,
        Sheet::Unphysical => -q_phys,
    }
}

/// Computes the complex breakup momentum $`q(m)`$ for a real parent mass.
///
/// This checked version validates finite nonnegative masses and `m > 0`.
pub fn q_m(m: f64, m1: f64, m2: f64, sheet: Sheet) -> LadduPhysicsResult<Complex64> {
    TwoBodyChannel::new(m1, m2)?.q_m(m, sheet)
}

/// Computes the complex breakup momentum $`q(m)`$ without validation.
///
/// # Preconditions
///
/// - `m` is finite and positive
/// - `m1` and `m2` are finite and nonnegative
#[inline(always)]
pub fn q_m_unchecked(m: f64, m1: f64, m2: f64, sheet: Sheet) -> Complex64 {
    q_s_unchecked(m.powi(2).into(), m1, m2, sheet)
}

/// Computes the complex two-body phase-space factor $`\rho(s)`$ on a chosen Riemann sheet.
///
/// This checked version validates finite nonnegative masses and finite nonzero `s`.
pub fn rho_s(s: Complex64, m1: f64, m2: f64, sheet: Sheet) -> LadduPhysicsResult<Complex64> {
    TwoBodyChannel::new(m1, m2)?.rho_s(s, sheet)
}

/// Computes the complex two-body phase-space factor $`\rho(s)`$ without validation.
///
/// # Preconditions
///
/// - `s` is finite and nonzero
/// - `m1` and `m2` are finite and nonnegative
#[inline(always)]
pub fn rho_s_unchecked(s: Complex64, m1: f64, m2: f64, sheet: Sheet) -> Complex64 {
    2.0 * q_s_unchecked(s, m1, m2, sheet) / s.sqrt()
}

/// Computes the complex two-body phase-space factor $`\rho(m)`$ for a real parent mass.
///
/// This checked version validates finite nonnegative masses and `m > 0`.
pub fn rho_m(m: f64, m1: f64, m2: f64, sheet: Sheet) -> LadduPhysicsResult<Complex64> {
    TwoBodyChannel::new(m1, m2)?.rho_m(m, sheet)
}

/// Computes the complex two-body phase-space factor $`\rho(m)`$ without validation.
///
/// # Preconditions
///
/// - `m` is finite and positive
/// - `m1` and `m2` are finite and nonnegative
#[inline(always)]
pub fn rho_m_unchecked(m: f64, m1: f64, m2: f64, sheet: Sheet) -> Complex64 {
    rho_s_unchecked(m.powi(2).into(), m1, m2, sheet)
}

/// Selects which form of the Blatt-Weisskopf barrier factor is returned.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BarrierKind {
    /// The full barrier factor, including threshold powers of $`q`$.
    Full,
    /// The tensor barrier factor, with the explicit $`q^\ell`$ dependence removed.
    Tensor,
}

pub const BLATT_WEISSKOPF_MAX_L: usize = 8;

/// Default Blatt-Weisskopf radius parameter $`q_R`$ in GeV.
pub const QR_DEFAULT: f64 = 0.1973;

/// Validated Blatt-Weisskopf barrier-factor configuration.
///
/// This validates `l`, `q_r`, and `kind` once. Use `evaluate_*_unchecked`
/// methods in hot loops when the dynamic inputs have already been validated.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlattWeisskopf {
    l: L,
    q_r: f64,
    kind: BarrierKind,
}

impl BlattWeisskopf {
    pub fn new(l: impl TryInto<L>, q_r: f64, kind: BarrierKind) -> LadduPhysicsResult<Self> {
        let l = l
            .try_into()
            .map_err(|_| LadduPhysicsError::ConversionError("L"))?;
        validate_blatt_weisskopf_l(l)?;
        validate_q_r(q_r)?;

        Ok(Self { l, q_r, kind })
    }

    pub fn new_default(l: impl TryInto<L>, kind: BarrierKind) -> LadduPhysicsResult<Self> {
        Self::new(l, QR_DEFAULT, kind)
    }

    #[inline]
    pub const fn l(self) -> L {
        self.l
    }

    #[inline]
    pub const fn q_r(self) -> f64 {
        self.q_r
    }

    #[inline]
    pub const fn kind(self) -> BarrierKind {
        self.kind
    }

    pub fn evaluate_q(self, q: Complex64) -> LadduPhysicsResult<Complex64> {
        validate_finite_complex("q", q)?;

        if self.kind == BarrierKind::Tensor {
            validate_tensor_barrier_q(self.l, q)?;
        }

        Ok(self.evaluate_q_unchecked(q))
    }

    /// Evaluate from a precomputed breakup momentum without validation.
    ///
    /// # Preconditions
    ///
    /// - `q` is finite
    /// - if `kind == BarrierKind::Tensor` and `l > 0`, then `q != 0`
    #[inline(always)]
    pub fn evaluate_q_unchecked(self, q: Complex64) -> Complex64 {
        let z = q * q / (self.q_r * self.q_r);
        let full = blatt_weisskopf_polynomial_unchecked(z, self.l);

        match self.kind {
            BarrierKind::Full => full,
            BarrierKind::Tensor => full / q.powu(self.l.value()),
        }
    }

    pub fn evaluate_s(
        self,
        s: Complex64,
        channel: TwoBodyChannel,
        sheet: Sheet,
    ) -> LadduPhysicsResult<Complex64> {
        let q = channel.q_s(s, sheet)?;
        self.evaluate_q(q)
    }

    /// Evaluate from `s` without validation.
    ///
    /// # Preconditions
    ///
    /// - `s` is finite and nonzero
    /// - `channel` was constructed with valid masses
    /// - if `kind == BarrierKind::Tensor` and `l > 0`, then `q(s) != 0`
    #[inline(always)]
    pub fn evaluate_s_unchecked(
        self,
        s: Complex64,
        channel: TwoBodyChannel,
        sheet: Sheet,
    ) -> Complex64 {
        let q = channel.q_s_unchecked(s, sheet);
        self.evaluate_q_unchecked(q)
    }

    pub fn evaluate_m(
        self,
        m0: f64,
        channel: TwoBodyChannel,
        sheet: Sheet,
    ) -> LadduPhysicsResult<Complex64> {
        let q = channel.q_m(m0, sheet)?;
        self.evaluate_q(q)
    }

    /// Evaluate from real parent mass without validation.
    ///
    /// # Preconditions
    ///
    /// - `m0` is finite and positive
    /// - `channel` was constructed with valid masses
    /// - if `kind == BarrierKind::Tensor` and `l > 0`, then `q(m0) != 0`
    #[inline(always)]
    pub fn evaluate_m_unchecked(self, m0: f64, channel: TwoBodyChannel, sheet: Sheet) -> Complex64 {
        let q = channel.q_m_unchecked(m0, sheet);
        self.evaluate_q_unchecked(q)
    }
}

/// Computes the Blatt-Weisskopf polynomial factor from `z`.
///
/// This checked version validates that `l <= BLATT_WEISSKOPF_MAX_L` and that
/// `z` is finite.
pub fn blatt_weisskopf_polynomial(z: Complex64, l: L) -> LadduPhysicsResult<Complex64> {
    validate_finite_complex("z", z)?;
    validate_blatt_weisskopf_l(l)?;

    Ok(blatt_weisskopf_polynomial_unchecked(z, l))
}

/// Computes the Blatt-Weisskopf polynomial factor from `z` without validation.
///
/// # Preconditions
///
/// - `z` is finite
/// - `l <= BLATT_WEISSKOPF_MAX_L`
#[inline(always)]
pub(crate) fn blatt_weisskopf_polynomial_unchecked(z: Complex64, l: L) -> Complex64 {
    let value = match l.value() {
        0 => Complex64::new(1.0, 0.0),
        1 => (2.0 * z) / (z + 1.0),
        2 => (13.0 * z.powu(2)) / ((z - 3.0).powu(2) + 9.0 * z),
        3 => (277.0 * z.powu(3)) / (z.powu(3) + 6.0 * z.powu(2) + 45.0 * z + 225.0),
        4 => {
            (12746.0 * z.powu(4))
                / (z.powu(4) + 10.0 * z.powu(3) + 135.0 * z.powu(2) + 1575.0 * z + 11025.0)
        }
        5 => {
            (998881.0 * z.powu(5))
                / (z.powu(5)
                    + 15.0 * z.powu(4)
                    + 315.0 * z.powu(3)
                    + 6300.0 * z.powu(2)
                    + 99225.0 * z
                    + 893025.0)
        }
        6 => {
            (118394977.0 * z.powu(6))
                / (z.powu(6)
                    + 21.0 * z.powu(5)
                    + 630.0 * z.powu(4)
                    + 18900.0 * z.powu(3)
                    + 496125.0 * z.powu(2)
                    + 9823275.0 * z
                    + 18261468225.0)
        }
        7 => {
            (19727003738.0 * z.powu(7))
                / (z.powu(7)
                    + 28.0 * z.powu(6)
                    + 1134.0 * z.powu(5)
                    + 47250.0 * z.powu(4)
                    + 1819125.0 * z.powu(3)
                    + 58939650.0 * z.powu(2)
                    + 1404728325.0 * z
                    + 18261468225.0)
        }
        8 => {
            (4392846440677.0 * z.powu(8))
                / (z.powu(8)
                    + 36.0 * z.powu(7)
                    + 1890.0 * z.powu(6)
                    + 103950.0 * z.powu(5)
                    + 5457375.0 * z.powu(4)
                    + 255405150.0 * z.powu(3)
                    + 9833098275.0 * z.powu(2)
                    + 273922023375.0 * z
                    + 4108830350625.0)
        }
        _ => {
            unreachable!("l must be validated before calling blatt_weisskopf_polynomial_unchecked")
        }
    };

    value.sqrt()
}

/// Computes the Blatt-Weisskopf centrifugal barrier factor from precomputed `q`.
pub fn blatt_weisskopf_q(
    q: Complex64,
    l: usize,
    q_r: f64,
    kind: BarrierKind,
) -> LadduPhysicsResult<Complex64> {
    BlattWeisskopf::new(l, q_r, kind)?.evaluate_q(q)
}

/// Computes the Blatt-Weisskopf centrifugal barrier factor from precomputed `q`
/// without validation.
///
/// # Preconditions
///
/// - `q` is finite
/// - `l <= BLATT_WEISSKOPF_MAX_L`
/// - `q_r` is finite and positive
/// - if `kind == BarrierKind::Tensor` and `l > 0`, then `q != 0`
#[inline(always)]
pub fn blatt_weisskopf_q_unchecked(q: Complex64, l: L, q_r: f64, kind: BarrierKind) -> Complex64 {
    let barrier = BlattWeisskopf { l, q_r, kind };
    barrier.evaluate_q_unchecked(q)
}

/// Computes the Blatt-Weisskopf centrifugal barrier factor in terms of `s`.
pub fn blatt_weisskopf_s(
    s: Complex64,
    m1: f64,
    m2: f64,
    l: L,
    q_r: f64,
    sheet: Sheet,
    kind: BarrierKind,
) -> LadduPhysicsResult<Complex64> {
    let channel = TwoBodyChannel::new(m1, m2)?;
    let barrier = BlattWeisskopf::new(l, q_r, kind)?;

    barrier.evaluate_s(s, channel, sheet)
}

/// Computes the Blatt-Weisskopf centrifugal barrier factor in terms of `s`
/// without validation.
///
/// # Preconditions
///
/// - `s` is finite and nonzero
/// - `m1` and `m2` are finite and nonnegative
/// - `l <= BLATT_WEISSKOPF_MAX_L`
/// - `q_r` is finite and positive
/// - if `kind == BarrierKind::Tensor` and `l > 0`, then `q(s) != 0`
#[inline(always)]
pub fn blatt_weisskopf_s_unchecked(
    s: Complex64,
    m1: f64,
    m2: f64,
    l: L,
    q_r: f64,
    sheet: Sheet,
    kind: BarrierKind,
) -> Complex64 {
    let channel = TwoBodyChannel { m1, m2 };
    let barrier = BlattWeisskopf { l, q_r, kind };

    barrier.evaluate_s_unchecked(s, channel, sheet)
}

/// Computes the Blatt-Weisskopf centrifugal barrier factor for a real parent mass.
pub fn blatt_weisskopf_m(
    m0: f64,
    m1: f64,
    m2: f64,
    l: L,
    q_r: f64,
    sheet: Sheet,
    kind: BarrierKind,
) -> LadduPhysicsResult<Complex64> {
    let channel = TwoBodyChannel::new(m1, m2)?;
    let barrier = BlattWeisskopf::new(l, q_r, kind)?;

    barrier.evaluate_m(m0, channel, sheet)
}

/// Computes the Blatt-Weisskopf centrifugal barrier factor for a real parent mass
/// without validation.
///
/// # Preconditions
///
/// - `m0` is finite and positive
/// - `m1` and `m2` are finite and nonnegative
/// - `l <= BLATT_WEISSKOPF_MAX_L`
/// - `q_r` is finite and positive
/// - if `kind == BarrierKind::Tensor` and `l > 0`, then `q(m0) != 0`
#[inline(always)]
pub fn blatt_weisskopf_m_unchecked(
    m0: f64,
    m1: f64,
    m2: f64,
    l: L,
    q_r: f64,
    sheet: Sheet,
    kind: BarrierKind,
) -> Complex64 {
    let channel = TwoBodyChannel { m1, m2 };
    let barrier = BlattWeisskopf { l, q_r, kind };

    barrier.evaluate_m_unchecked(m0, channel, sheet)
}

#[cfg(test)]
mod test {
    use std::f64::consts::PI;

    use approx::assert_relative_eq;
    use num::complex::Complex64;

    use super::*;
    use crate::{l, m};

    const EPS: f64 = 1.0e-12;
    const LOOSE_EPS: f64 = 1.0e-8;

    fn assert_complex_relative_eq(actual: Complex64, expected: Complex64, epsilon: f64) {
        assert_relative_eq!(actual.re, expected.re, epsilon = epsilon);
        assert_relative_eq!(actual.im, expected.im, epsilon = epsilon);
    }

    fn expected_spherical_harmonic(l: usize, m: isize, costheta: f64, phi: f64) -> Complex64 {
        match (l, m) {
            (0, 0) => Complex64::from(f64::sqrt(1.0 / (4.0 * PI))),

            (1, -1) => Complex64::from_polar(
                f64::sqrt(3.0 / (8.0 * PI)) * f64::sin(f64::acos(costheta)),
                -phi,
            ),
            (1, 0) => Complex64::from(f64::sqrt(3.0 / (4.0 * PI)) * costheta),
            (1, 1) => Complex64::from_polar(
                -f64::sqrt(3.0 / (8.0 * PI)) * f64::sin(f64::acos(costheta)),
                phi,
            ),

            (2, -2) => Complex64::from_polar(
                f64::sqrt(15.0 / (32.0 * PI)) * f64::sin(f64::acos(costheta)).powi(2),
                -2.0 * phi,
            ),
            (2, -1) => Complex64::from_polar(
                f64::sqrt(15.0 / (8.0 * PI)) * f64::sin(f64::acos(costheta)) * costheta,
                -phi,
            ),
            (2, 0) => {
                Complex64::from(f64::sqrt(5.0 / (16.0 * PI)) * (3.0 * costheta.powi(2) - 1.0))
            }
            (2, 1) => Complex64::from_polar(
                -f64::sqrt(15.0 / (8.0 * PI)) * f64::sin(f64::acos(costheta)) * costheta,
                phi,
            ),
            (2, 2) => Complex64::from_polar(
                f64::sqrt(15.0 / (32.0 * PI)) * f64::sin(f64::acos(costheta)).powi(2),
                2.0 * phi,
            ),

            _ => unreachable!("test only covers l <= 2"),
        }
    }

    #[test]
    fn spherical_harmonics_match_known_values_and_unchecked_paths() {
        let modes = [
            (l!(0), m!(0)),
            (l!(1), m!(-1)),
            (l!(1), m!(0)),
            (l!(1), m!(1)),
            (l!(2), m!(-2)),
            (l!(2), m!(-1)),
            (l!(2), m!(0)),
            (l!(2), m!(1)),
            (l!(2), m!(2)),
        ];
        let costhetas = [-1.0, -0.8, -0.3, 0.0, 0.3, 0.8, 1.0];
        let phis = [0.0, 0.3, 0.5, 0.8, 1.0].map(|v| v * 2.0 * PI);

        for (l, m) in modes {
            let mode = SphericalHarmonic::new(l, m).unwrap();

            assert_eq!(mode.l(), l);
            assert_eq!(mode.m(), m);

            for costheta in costhetas {
                for phi in phis {
                    let expected = expected_spherical_harmonic(
                        l.value() as usize,
                        m.doubled() as isize / 2,
                        costheta,
                        phi,
                    );

                    let standalone = spherical_harmonic(l, m, costheta, phi).unwrap();
                    let standalone_unchecked = spherical_harmonic_unchecked(
                        l.value() as usize,
                        m.doubled() as isize / 2,
                        costheta,
                        phi,
                    );
                    let struct_checked = mode.evaluate(costheta, phi).unwrap();
                    let struct_unchecked = mode.evaluate_unchecked(costheta, phi);

                    assert_complex_relative_eq(standalone, expected, EPS);
                    assert_complex_relative_eq(standalone_unchecked, expected, EPS);
                    assert_complex_relative_eq(struct_checked, expected, EPS);
                    assert_complex_relative_eq(struct_unchecked, expected, EPS);
                }
            }
        }
    }

    #[test]
    fn two_body_kinematics_match_known_values_and_unchecked_paths() {
        let channel = TwoBodyChannel::new(0.51, 0.62).unwrap();

        assert_relative_eq!(channel.m1(), 0.51, epsilon = EPS);
        assert_relative_eq!(channel.m2(), 0.62, epsilon = EPS);
        assert_relative_eq!(channel.threshold(), 1.13, epsilon = EPS);
        assert_relative_eq!(channel.threshold_s(), 1.2769, epsilon = EPS);
        assert_relative_eq!(channel.pseudothreshold(), 0.11, epsilon = EPS);
        assert_relative_eq!(channel.pseudothreshold_s(), 0.0121, epsilon = EPS);

        let s = 1.3;
        let s_complex = Complex64::from(s);
        let m = f64::sqrt(s);

        assert_relative_eq!(
            chi_plus(s, 0.51, 0.62).unwrap(),
            0.01776923076923098,
            epsilon = EPS
        );
        assert_relative_eq!(
            chi_plus_unchecked(s, 0.51, 0.62),
            0.01776923076923098,
            epsilon = EPS
        );
        assert_relative_eq!(
            channel.chi_plus(s).unwrap(),
            0.01776923076923098,
            epsilon = EPS
        );
        assert_relative_eq!(
            channel.chi_plus_unchecked(s),
            0.01776923076923098,
            epsilon = EPS
        );

        assert_relative_eq!(
            chi_minus(s, 0.51, 0.62).unwrap(),
            0.9906923076923076,
            epsilon = EPS
        );
        assert_relative_eq!(
            chi_minus_unchecked(s, 0.51, 0.62),
            0.9906923076923076,
            epsilon = EPS
        );
        assert_relative_eq!(
            channel.chi_minus(s).unwrap(),
            0.9906923076923076,
            epsilon = EPS
        );
        assert_relative_eq!(
            channel.chi_minus_unchecked(s),
            0.9906923076923076,
            epsilon = EPS
        );

        let rho_expected = Complex64::new(0.13267946426138, 0.0);
        assert_complex_relative_eq(
            rho_m(m, 0.51, 0.62, Sheet::Physical).unwrap(),
            rho_expected,
            EPS,
        );
        assert_complex_relative_eq(
            rho_m_unchecked(m, 0.51, 0.62, Sheet::Physical),
            rho_expected,
            EPS,
        );
        assert_complex_relative_eq(
            channel.rho_m(m, Sheet::Physical).unwrap(),
            rho_expected,
            EPS,
        );
        assert_complex_relative_eq(
            channel.rho_m_unchecked(m, Sheet::Physical),
            rho_expected,
            EPS,
        );
        assert_complex_relative_eq(
            channel.rho_s(s_complex, Sheet::Physical).unwrap(),
            rho_expected,
            EPS,
        );
        assert_complex_relative_eq(
            channel.rho_s_unchecked(s_complex, Sheet::Physical),
            rho_expected,
            EPS,
        );

        let q_expected = Complex64::new(0.3954823004889093, 0.0);
        let q_channel = TwoBodyChannel::new(0.4, 0.5).unwrap();

        assert_complex_relative_eq(
            q_m(1.2, 0.4, 0.5, Sheet::Physical).unwrap(),
            q_expected,
            EPS,
        );
        assert_complex_relative_eq(
            q_m_unchecked(1.2, 0.4, 0.5, Sheet::Physical),
            q_expected,
            EPS,
        );
        assert_complex_relative_eq(
            q_channel.q_m(1.2, Sheet::Physical).unwrap(),
            q_expected,
            EPS,
        );
        assert_complex_relative_eq(
            q_channel.q_m_unchecked(1.2, Sheet::Physical),
            q_expected,
            EPS,
        );
        assert_complex_relative_eq(
            q_channel
                .q_s(Complex64::from(1.44), Sheet::Physical)
                .unwrap(),
            q_expected,
            EPS,
        );
        assert_complex_relative_eq(
            q_channel.q_s_unchecked(Complex64::from(1.44), Sheet::Physical),
            q_expected,
            EPS,
        );

        assert_complex_relative_eq(
            q_channel.q_m(1.2, Sheet::Unphysical).unwrap(),
            -q_expected,
            EPS,
        );

        let below_channel = TwoBodyChannel::new(1.23, 0.62).unwrap();
        let below_rho_expected = Complex64::new(0.0, 1.0795209736472833);
        let below_q_expected = Complex64::new(0.0, 1.3154464282347478);

        assert_complex_relative_eq(
            below_channel.rho_m(m, Sheet::Physical).unwrap(),
            below_rho_expected,
            EPS,
        );
        assert_complex_relative_eq(
            q_m(1.2, 1.4, 1.5, Sheet::Physical).unwrap(),
            below_q_expected,
            EPS,
        );
    }

    #[test]
    fn blatt_weisskopf_matches_known_values_and_unchecked_paths() {
        let above_channel = TwoBodyChannel::new(0.4, 0.5).unwrap();
        let below_channel = TwoBodyChannel::new(1.4, 1.5).unwrap();

        let above_expected = [
            (l!(0), 1.0),
            (l!(1), 1.2654752018685698),
            (l!(2), 2.375285855793918),
            (l!(3), 5.62658768678507),
            (l!(4), 12.747554064467208),
        ];

        let below_expected = [
            (l!(0), 1.0),
            (l!(1), 1.430394249144933),
            (l!(2), 3.724659004227952),
            (l!(3), 17.689297320491015),
            (l!(4), 124.05258418258987),
            (l!(5), 1138.5868292398761),
            (l!(6), 6211.480561374802),
            (l!(7), 172727.17381791578),
            (l!(8), 2630882.804294494),
        ];

        for (l, expected_re) in above_expected {
            let barrier = BlattWeisskopf::new_default(l, BarrierKind::Full).unwrap();
            let expected = Complex64::new(expected_re, 0.0);
            let q = above_channel.q_m(1.2, Sheet::Physical).unwrap();

            assert_eq!(barrier.l(), l);
            assert_relative_eq!(barrier.q_r(), QR_DEFAULT, epsilon = EPS);
            assert_eq!(barrier.kind(), BarrierKind::Full);

            assert_complex_relative_eq(
                blatt_weisskopf_m(
                    1.2,
                    0.4,
                    0.5,
                    l,
                    QR_DEFAULT,
                    Sheet::Physical,
                    BarrierKind::Full,
                )
                .unwrap(),
                expected,
                LOOSE_EPS,
            );
            assert_complex_relative_eq(
                blatt_weisskopf_m_unchecked(
                    1.2,
                    0.4,
                    0.5,
                    l,
                    QR_DEFAULT,
                    Sheet::Physical,
                    BarrierKind::Full,
                ),
                expected,
                LOOSE_EPS,
            );
            assert_complex_relative_eq(
                barrier
                    .evaluate_m(1.2, above_channel, Sheet::Physical)
                    .unwrap(),
                expected,
                LOOSE_EPS,
            );
            assert_complex_relative_eq(
                barrier.evaluate_m_unchecked(1.2, above_channel, Sheet::Physical),
                expected,
                LOOSE_EPS,
            );
            assert_complex_relative_eq(barrier.evaluate_q(q).unwrap(), expected, LOOSE_EPS);
            assert_complex_relative_eq(barrier.evaluate_q_unchecked(q), expected, LOOSE_EPS);
        }

        for (l, expected_re) in below_expected {
            let barrier = BlattWeisskopf::new_default(l, BarrierKind::Full).unwrap();
            let expected = Complex64::new(expected_re, 0.0);

            assert_complex_relative_eq(
                blatt_weisskopf_m(
                    1.2,
                    1.4,
                    1.5,
                    l,
                    QR_DEFAULT,
                    Sheet::Physical,
                    BarrierKind::Full,
                )
                .unwrap(),
                expected,
                LOOSE_EPS,
            );
            assert_complex_relative_eq(
                barrier
                    .evaluate_m(1.2, below_channel, Sheet::Physical)
                    .unwrap(),
                expected,
                LOOSE_EPS,
            );
        }
    }

    #[test]
    fn tensor_barrier_is_full_barrier_with_explicit_q_power_removed() {
        let q = Complex64::new(0.3, 0.2);

        for l in 0..=BLATT_WEISSKOPF_MAX_L {
            let full = blatt_weisskopf_q(q, l, QR_DEFAULT, BarrierKind::Full).unwrap();
            let tensor = blatt_weisskopf_q(q, l, QR_DEFAULT, BarrierKind::Tensor).unwrap();

            assert_complex_relative_eq(tensor * q.powu(l as u32), full, LOOSE_EPS);
        }

        let tensor_l0_at_threshold =
            blatt_weisskopf_q(Complex64::new(0.0, 0.0), 0, QR_DEFAULT, BarrierKind::Tensor)
                .unwrap();

        assert_complex_relative_eq(tensor_l0_at_threshold, Complex64::new(1.0, 0.0), EPS);
    }

    #[test]
    fn checked_apis_reject_invalid_inputs() {
        assert!(SphericalHarmonic::new(l!(1), m!(2)).is_err());
        assert!(SphericalHarmonic::new(l!(1), m!(1 / 2)).is_err());
        assert!(spherical_harmonic(l!(1), m!(0), 1.0 + f64::EPSILON.sqrt(), 0.0).is_err());
        assert!(spherical_harmonic(l!(1), m!(0), 0.0, f64::NAN).is_err());

        assert!(TwoBodyChannel::new(-0.1, 0.2).is_err());
        assert!(TwoBodyChannel::new(0.1, f64::NAN).is_err());

        let channel = TwoBodyChannel::new(0.4, 0.5).unwrap();

        assert!(channel.chi_plus(0.0).is_err());
        assert!(channel.chi_minus(f64::INFINITY).is_err());
        assert!(
            channel
                .q_s(Complex64::new(0.0, 0.0), Sheet::Physical)
                .is_err()
        );
        assert!(
            channel
                .q_s(Complex64::new(f64::NAN, 0.0), Sheet::Physical)
                .is_err()
        );
        assert!(channel.q_m(0.0, Sheet::Physical).is_err());
        assert!(
            channel
                .rho_s(Complex64::new(0.0, 0.0), Sheet::Physical)
                .is_err()
        );
        assert!(channel.rho_m(f64::INFINITY, Sheet::Physical).is_err());

        assert!(chi_plus(1.0, -0.1, 0.2).is_err());
        assert!(chi_minus(0.0, 0.1, 0.2).is_err());
        assert!(q_s(Complex64::new(0.0, 0.0), 0.1, 0.2, Sheet::Physical).is_err());
        assert!(q_m(0.0, 0.1, 0.2, Sheet::Physical).is_err());
        assert!(rho_s(Complex64::new(0.0, 0.0), 0.1, 0.2, Sheet::Physical).is_err());
        assert!(rho_m(0.0, 0.1, 0.2, Sheet::Physical).is_err());

        assert!(
            BlattWeisskopf::new(BLATT_WEISSKOPF_MAX_L + 1, QR_DEFAULT, BarrierKind::Full).is_err()
        );
        assert!(BlattWeisskopf::new_default(BLATT_WEISSKOPF_MAX_L + 1, BarrierKind::Full).is_err());
        assert!(BlattWeisskopf::new(0, 0.0, BarrierKind::Full).is_err());
        assert!(BlattWeisskopf::new(0, f64::NAN, BarrierKind::Full).is_err());

        assert!(blatt_weisskopf_polynomial(Complex64::new(f64::NAN, 0.0), l!(0),).is_err());
        assert!(
            blatt_weisskopf_polynomial(
                Complex64::new(1.0, 0.0),
                L::try_from(BLATT_WEISSKOPF_MAX_L + 1).unwrap(),
            )
            .is_err()
        );

        let tensor = BlattWeisskopf::new_default(1, BarrierKind::Tensor).unwrap();

        assert!(tensor.evaluate_q(Complex64::new(0.0, 0.0)).is_err());
        assert!(
            blatt_weisskopf_q(Complex64::new(0.0, 0.0), 1, QR_DEFAULT, BarrierKind::Tensor,)
                .is_err()
        );
        assert!(
            blatt_weisskopf_s(
                Complex64::new(0.0, 0.0),
                0.4,
                0.5,
                l!(1),
                QR_DEFAULT,
                Sheet::Physical,
                BarrierKind::Full,
            )
            .is_err()
        );
        assert!(
            blatt_weisskopf_m(
                0.0,
                0.4,
                0.5,
                l!(1),
                QR_DEFAULT,
                Sheet::Physical,
                BarrierKind::Full,
            )
            .is_err()
        );
    }
}
