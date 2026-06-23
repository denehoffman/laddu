use std::f64::consts::PI;

use laddu_expr::Expr;
use num::complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::{
    LadduPhysicsError, LadduPhysicsResult,
    quantum::{L, M},
};

#[inline]
fn validate_finite_real(name: &str, value: f64) -> LadduPhysicsResult<()> {
    if !value.is_finite() {
        return Err(LadduPhysicsError::invalid_value(name, "finite", value));
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
fn associated_legendre_pos_m(l: usize, m: usize, x: Expr) -> Expr {
    let mut p: Expr = 1.0.into();

    if l == 0 && m == 0 {
        return p;
    }

    let y = (1.0 - x.powi(2)).sqrt();

    for m_p in 0..m {
        p *= -((2 * m_p + 1) as f64) * &y;
    }

    if l == m {
        return p;
    }

    let mut p_min_2 = p;
    let mut p_min_1 = (2 * m + 1) as f64 * &x * &p_min_2;

    if l == m + 1 {
        return p_min_1;
    }

    for l_p in (m + 1)..l {
        let next = ((2 * l_p + 1) as f64 * &x * &p_min_1 - (l_p + m) as f64 * &p_min_2)
            / (l_p - m + 1) as f64;
        p_min_2 = p_min_1;
        p_min_1 = next;
    }

    p_min_1
}

#[inline]
fn factorial_ratio_l_minus_over_l_plus(l: usize, abs_m: usize) -> f64 {
    let mut ratio = 1.0;

    for k in (l - abs_m + 1)..=(l + abs_m) {
        ratio /= k as f64;
    }

    ratio
}

/// A validated spherical-harmonic mode.
///
/// This stores fixed quantum numbers `l` and `m`, so they are validated once.
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
    pub const fn l(&self) -> L {
        self.l
    }

    #[inline]
    pub const fn m(&self) -> M {
        self.m
    }

    pub fn evaluate(&self, costheta: impl Into<Expr>, phi: impl Into<Expr>) -> Expr {
        spherical_harmonic_expr(
            self.l.value() as usize,
            self.m.doubled() as isize / 2,
            costheta.into(),
            phi.into(),
        )
    }
}

/// Computes the spherical harmonic $`Y_\ell^m(\theta, \phi)`$ given $`\cos\theta`$ and $`\phi`$.
///
/// This checked version validates `|m| <= l`
pub fn spherical_harmonic(l: L, m: M, costheta: Expr, phi: Expr) -> LadduPhysicsResult<Expr> {
    Ok(SphericalHarmonic::new(l, m)?.evaluate(costheta, phi))
}

#[inline(always)]
fn spherical_harmonic_expr(l: usize, m: isize, costheta: Expr, phi: Expr) -> Expr {
    let abs_m = m.unsigned_abs();

    let mut res = associated_legendre_pos_m(l, abs_m, costheta);

    res *=
        f64::sqrt((2 * l + 1) as f64 / (4.0 * PI) * factorial_ratio_l_minus_over_l_plus(l, abs_m));

    if m < 0 && abs_m % 2 != 0 {
        res = -res;
    }

    &res * (m as f64 * &phi).cos() + Complex64::I * res * (m as f64 * phi).sin()
}

/// A two-body channel with fixed daughter masses.
#[derive(Clone, Debug)]
pub struct TwoBodyChannel {
    m1: Expr,
    m2: Expr,
}

impl TwoBodyChannel {
    pub fn new(m1: impl Into<Expr>, m2: impl Into<Expr>) -> Self {
        Self {
            m1: m1.into(),
            m2: m2.into(),
        }
    }

    #[inline]
    pub fn m1(&self) -> Expr {
        self.m1.clone()
    }

    #[inline]
    pub fn m2(&self) -> Expr {
        self.m2.clone()
    }

    #[inline]
    pub fn threshold(&self) -> Expr {
        &self.m1 + &self.m2
    }

    #[inline]
    pub fn threshold_s(&self) -> Expr {
        self.threshold().powi(2)
    }

    #[inline]
    pub fn pseudothreshold(&self) -> Expr {
        (&self.m1 - &self.m2).powi(2).sqrt()
    }

    #[inline]
    pub fn pseudothreshold_s(&self) -> Expr {
        self.pseudothreshold().powi(2)
    }

    pub fn chi_plus(&self, s: impl Into<Expr>) -> Expr {
        chi_plus_expr(s.into(), &self.m1, &self.m2)
    }

    pub fn chi_minus(&self, s: impl Into<Expr>) -> Expr {
        chi_minus_expr(s.into(), &self.m1, &self.m2)
    }

    pub fn q_s(&self, s: impl Into<Expr>, sheet: Sheet) -> Expr {
        q_s_expr(s.into(), &self.m1, &self.m2, sheet)
    }

    pub fn q_m(&self, m: impl Into<Expr>, sheet: Sheet) -> Expr {
        q_m_expr(m.into(), &self.m1, &self.m2, sheet)
    }

    pub fn rho_s(&self, s: impl Into<Expr>, sheet: Sheet) -> Expr {
        rho_s_expr(s.into(), &self.m1, &self.m2, sheet)
    }

    pub fn rho_m(&self, m: impl Into<Expr>, sheet: Sheet) -> Expr {
        rho_m_expr(m.into(), &self.m1, &self.m2, sheet)
    }
}

#[inline(always)]
fn chi_plus_expr(s: Expr, m1: impl Into<Expr>, m2: impl Into<Expr>) -> Expr {
    let m1 = m1.into();
    let m2 = m2.into();
    1.0 - (m1 + m2).powi(2) / s
}

#[inline(always)]
fn chi_minus_expr(s: Expr, m1: impl Into<Expr>, m2: impl Into<Expr>) -> Expr {
    let m1 = m1.into();
    let m2 = m2.into();
    1.0 - (m1 - m2).powi(2) / s
}

/// Selects the Riemann sheet used for analytic continuation of two-body kinematic functions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Sheet {
    /// The physical sheet of the complex energy plane.
    Physical,
    /// The unphysical sheet obtained by flipping the sign of the channel momentum.
    Unphysical,
}

#[inline(always)]
fn q_s_expr(s: Expr, m1: impl Into<Expr>, m2: impl Into<Expr>, sheet: Sheet) -> Expr {
    let m1 = m1.into();
    let m2 = m2.into();
    let sp = (&m1 + &m2).powi(2);
    let sm = (m1 - m2).powi(2);
    let q_phys = ((&s - sp) * (&s - sm)).sqrt() / (2.0 * s.sqrt());

    match sheet {
        Sheet::Physical => q_phys,
        Sheet::Unphysical => -q_phys,
    }
}

#[inline(always)]
fn q_m_expr(m: Expr, m1: impl Into<Expr>, m2: impl Into<Expr>, sheet: Sheet) -> Expr {
    q_s_expr(m.powi(2), m1, m2, sheet)
}

#[inline(always)]
fn rho_s_expr(s: Expr, m1: impl Into<Expr>, m2: impl Into<Expr>, sheet: Sheet) -> Expr {
    2.0 * q_s_expr(s.clone(), m1, m2, sheet) / s.sqrt()
}

#[inline(always)]
fn rho_m_expr(m: Expr, m1: impl Into<Expr>, m2: impl Into<Expr>, sheet: Sheet) -> Expr {
    rho_s_expr(m.powi(2), m1, m2, sheet)
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
    pub const fn l(&self) -> L {
        self.l
    }

    #[inline]
    pub const fn q_r(&self) -> f64 {
        self.q_r
    }

    #[inline]
    pub const fn kind(&self) -> BarrierKind {
        self.kind
    }

    pub fn evaluate_q(&self, q: impl Into<Expr>) -> Expr {
        let q = q.into();
        let z = q.powi(2) / self.q_r.powi(2);
        let full = blatt_weisskopf_polynomial_expr(z, self.l);

        match self.kind {
            BarrierKind::Full => full,
            BarrierKind::Tensor => full / q.powi(self.l.value() as i32),
        }
    }

    pub fn evaluate_s(&self, s: impl Into<Expr>, channel: &TwoBodyChannel, sheet: Sheet) -> Expr {
        let q = channel.q_s(s, sheet);
        self.evaluate_q(q)
    }

    pub fn evaluate_m(&self, m0: impl Into<Expr>, channel: &TwoBodyChannel, sheet: Sheet) -> Expr {
        let q = channel.q_m(m0, sheet);
        self.evaluate_q(q)
    }
}

/// Computes the Blatt-Weisskopf polynomial factor from `z`.
///
/// This checked version validates that `l <= BLATT_WEISSKOPF_MAX_L` and that
/// `z` is finite.
pub fn blatt_weisskopf_polynomial(z: Expr, l: L) -> LadduPhysicsResult<Expr> {
    validate_blatt_weisskopf_l(l)?;

    Ok(blatt_weisskopf_polynomial_expr(z, l))
}

#[inline(always)]
fn blatt_weisskopf_polynomial_expr(z: Expr, l: L) -> Expr {
    let value = match l.value() {
        0 => Complex64::new(1.0, 0.0).into(),
        1 => (2.0 * &z) / (z + 1.0),
        2 => (13.0 * z.powi(2)) / ((&z - 3.0).powi(2) + 9.0 * z),
        3 => (277.0 * z.powi(3)) / (z.powi(3) + 6.0 * z.powi(2) + 45.0 * z + 225.0),
        4 => {
            (12746.0 * z.powi(4))
                / (z.powi(4) + 10.0 * z.powi(3) + 135.0 * z.powi(2) + 1575.0 * z + 11025.0)
        }
        5 => {
            (998881.0 * z.powi(5))
                / (z.powi(5)
                    + 15.0 * z.powi(4)
                    + 315.0 * z.powi(3)
                    + 6300.0 * z.powi(2)
                    + 99225.0 * z
                    + 893025.0)
        }
        6 => {
            (118394977.0 * z.powi(6))
                / (z.powi(6)
                    + 21.0 * z.powi(5)
                    + 630.0 * z.powi(4)
                    + 18900.0 * z.powi(3)
                    + 496125.0 * z.powi(2)
                    + 9823275.0 * z
                    + 18261468225.0)
        }
        7 => {
            (19727003738.0 * z.powi(7))
                / (z.powi(7)
                    + 28.0 * z.powi(6)
                    + 1134.0 * z.powi(5)
                    + 47250.0 * z.powi(4)
                    + 1819125.0 * z.powi(3)
                    + 58939650.0 * z.powi(2)
                    + 1404728325.0 * z
                    + 18261468225.0)
        }
        8 => {
            (4392846440677.0 * z.powi(8))
                / (z.powi(8)
                    + 36.0 * z.powi(7)
                    + 1890.0 * z.powi(6)
                    + 103950.0 * z.powi(5)
                    + 5457375.0 * z.powi(4)
                    + 255405150.0 * z.powi(3)
                    + 9833098275.0 * z.powi(2)
                    + 273922023375.0 * z
                    + 4108830350625.0)
        }
        _ => {
            unreachable!("l must be validated before calling blatt_weisskopf_polynomial_expr")
        }
    };

    value.sqrt()
}

#[cfg(test)]
mod test {
    use std::f64::consts::PI;

    use approx::assert_relative_eq;
    use laddu_compile::CompiledModel;
    use laddu_runtime::CpuBackend;
    use num::complex::Complex64;

    use super::*;
    use crate::{l, m};

    const EPS: f64 = 1.0e-12;
    const LOOSE_EPS: f64 = 1.0e-8;

    fn evaluate(expr: Expr) -> Complex64 {
        let model = CompiledModel::from_expr(&expr).unwrap();
        let params = model.params().default_values();
        CpuBackend.prepare(&model).evaluate(&params).unwrap()
    }

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
    fn spherical_harmonics_match_known_values() {
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

                    assert_complex_relative_eq(
                        evaluate(spherical_harmonic(l, m, costheta.into(), phi.into()).unwrap()),
                        expected,
                        EPS,
                    );
                    assert_complex_relative_eq(
                        evaluate(mode.evaluate(costheta, phi)),
                        expected,
                        EPS,
                    );
                }
            }
        }
    }

    #[test]
    fn two_body_kinematics_match_known_values() {
        let channel = TwoBodyChannel::new(0.51, 0.62);
        let s = 1.3;
        let m = f64::sqrt(s);

        assert_complex_relative_eq(evaluate(channel.m1()), 0.51.into(), EPS);
        assert_complex_relative_eq(evaluate(channel.m2()), 0.62.into(), EPS);
        assert_complex_relative_eq(evaluate(channel.threshold()), 1.13.into(), EPS);
        assert_complex_relative_eq(evaluate(channel.threshold_s()), 1.2769.into(), EPS);
        assert_complex_relative_eq(evaluate(channel.pseudothreshold()), 0.11.into(), EPS);
        assert_complex_relative_eq(evaluate(channel.pseudothreshold_s()), 0.0121.into(), EPS);

        assert_complex_relative_eq(
            evaluate(channel.chi_plus(s)),
            0.01776923076923098.into(),
            EPS,
        );
        assert_complex_relative_eq(
            evaluate(channel.chi_minus(s)),
            0.9906923076923076.into(),
            EPS,
        );

        let rho_expected = Complex64::new(0.13267946426138, 0.0);
        assert_complex_relative_eq(
            evaluate(channel.rho_m(m, Sheet::Physical)),
            rho_expected,
            EPS,
        );
        assert_complex_relative_eq(
            evaluate(channel.rho_s(Complex64::from(s), Sheet::Physical)),
            rho_expected,
            EPS,
        );

        let q_expected = Complex64::new(0.3954823004889093, 0.0);
        let q_channel = TwoBodyChannel::new(0.4, 0.5);
        assert_complex_relative_eq(
            evaluate(q_channel.q_m(1.2, Sheet::Physical)),
            q_expected,
            EPS,
        );
        assert_complex_relative_eq(
            evaluate(q_channel.q_s(Complex64::from(1.44), Sheet::Physical)),
            q_expected,
            EPS,
        );
        assert_complex_relative_eq(
            evaluate(q_channel.q_m(1.2, Sheet::Unphysical)),
            -q_expected,
            EPS,
        );

        let below_channel = TwoBodyChannel::new(1.23, 0.62);
        assert_complex_relative_eq(
            evaluate(below_channel.rho_m(m, Sheet::Physical)),
            Complex64::new(0.0, 1.0795209736472833),
            EPS,
        );
        assert_complex_relative_eq(
            evaluate(TwoBodyChannel::new(1.4, 1.5).q_m(1.2, Sheet::Physical)),
            Complex64::new(0.0, 1.3154464282347478),
            EPS,
        );
    }

    #[test]
    fn blatt_weisskopf_matches_known_values() {
        let above_channel = TwoBodyChannel::new(0.4, 0.5);
        let below_channel = TwoBodyChannel::new(1.4, 1.5);

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
            assert_eq!(barrier.l(), l);
            assert_relative_eq!(barrier.q_r(), QR_DEFAULT, epsilon = EPS);
            assert_eq!(barrier.kind(), BarrierKind::Full);

            assert_complex_relative_eq(
                evaluate(barrier.evaluate_m(1.2, &above_channel, Sheet::Physical)),
                Complex64::new(expected_re, 0.0),
                LOOSE_EPS,
            );
            let q = above_channel.q_m(1.2, Sheet::Physical);
            assert_complex_relative_eq(
                evaluate(barrier.evaluate_q(q)),
                Complex64::new(expected_re, 0.0),
                LOOSE_EPS,
            );
        }

        for (l, expected_re) in below_expected {
            let barrier = BlattWeisskopf::new_default(l, BarrierKind::Full).unwrap();
            assert_complex_relative_eq(
                evaluate(barrier.evaluate_m(1.2, &below_channel, Sheet::Physical)),
                Complex64::new(expected_re, 0.0),
                LOOSE_EPS,
            );
        }
    }

    #[test]
    fn tensor_barrier_is_full_barrier_with_explicit_q_power_removed() {
        let q = Complex64::new(0.3, 0.2);

        for l_raw in 0..=BLATT_WEISSKOPF_MAX_L {
            let l = L::try_from(l_raw).unwrap();
            let full = BlattWeisskopf::new(l, QR_DEFAULT, BarrierKind::Full)
                .unwrap()
                .evaluate_q(q);
            let tensor = BlattWeisskopf::new(l, QR_DEFAULT, BarrierKind::Tensor)
                .unwrap()
                .evaluate_q(q);

            assert_complex_relative_eq(
                evaluate(tensor) * q.powu(l_raw as u32),
                evaluate(full),
                LOOSE_EPS,
            );
        }

        let tensor_l0_at_threshold = BlattWeisskopf::new(l!(0), QR_DEFAULT, BarrierKind::Tensor)
            .unwrap()
            .evaluate_q(Complex64::new(0.0, 0.0));
        assert_complex_relative_eq(
            evaluate(tensor_l0_at_threshold),
            Complex64::new(1.0, 0.0),
            EPS,
        );
    }

    #[test]
    fn constructors_reject_invalid_static_configuration() {
        assert!(SphericalHarmonic::new(l!(1), m!(2)).is_err());
        assert!(SphericalHarmonic::new(l!(1), m!(1 / 2)).is_err());
        assert!(
            BlattWeisskopf::new(BLATT_WEISSKOPF_MAX_L + 1, QR_DEFAULT, BarrierKind::Full).is_err()
        );
        assert!(BlattWeisskopf::new_default(BLATT_WEISSKOPF_MAX_L + 1, BarrierKind::Full).is_err());
        assert!(BlattWeisskopf::new(l!(0), 0.0, BarrierKind::Full).is_err());
        assert!(BlattWeisskopf::new(l!(0), f64::NAN, BarrierKind::Full).is_err());
        assert!(
            blatt_weisskopf_polynomial(
                Complex64::new(1.0, 0.0).into(),
                L::try_from(BLATT_WEISSKOPF_MAX_L + 1).unwrap(),
            )
            .is_err()
        );
    }
}
