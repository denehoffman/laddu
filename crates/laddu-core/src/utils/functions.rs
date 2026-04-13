use factorial::Factorial;
use nalgebra::ComplexField;
use num::{complex::Complex64, Integer};
use std::f64::consts::PI;

fn alp_pos_m(l: usize, m: usize, x: f64) -> f64 {
    let mut p = 1.0;
    if l == 0 && m == 0 {
        return p;
    }
    let y = f64::sqrt(1.0 - f64::powi(x, 2));
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

/// Computes the spherical harmonic $`Y_{\ell}^m(\theta, \phi)`$ (given $`\cos\theta`$). Note that
/// this formulation includes the Condon-Shortley phase.
pub fn spherical_harmonic(l: usize, m: isize, costheta: f64, phi: f64) -> Complex64 {
    let abs_m = isize::abs(m) as usize;
    let mut res = alp_pos_m(l, abs_m, costheta); // Includes Condon-Shortley phase already
    res *= f64::sqrt(
        (2 * l + 1) as f64 / (4.0 * PI) * ((l - abs_m).factorial()) as f64
            / ((l + abs_m).factorial()) as f64,
    );
    if m < 0 {
        res *= if abs_m.is_even() { 1.0 } else { -1.0 }; // divide out Condon-Shortley phase
                                                         // (it's just +/-1 so division is the
                                                         // same as multiplication here)
    }
    Complex64::new(
        res * f64::cos(m as f64 * phi),
        res * f64::sin(m as f64 * phi),
    )
}

/// Computes $`\chi_+(s, m_1, m_2) = 1 - \frac{(m_1 + m_2)^2}{s}`$.
pub fn chi_plus(s: f64, m1: f64, m2: f64) -> f64 {
    1.0 - (m1 + m2) * (m1 + m2) / s
}

/// Computes $`\chi_-(s, m_1, m_2) = 1 - \frac{(m_1 - m_2)^2}{s}`$.
pub fn chi_minus(s: f64, m1: f64, m2: f64) -> f64 {
    1.0 - (m1 - m2) * (m1 - m2) / s
}

/// Selects the Riemann sheet used for analytic continuation of two-body kinematic functions.
///
/// For a two-body channel, the breakup momentum
/// `q(s) = sqrt((s - s_+)(s - s_-)) / (2 sqrt(s))`
/// is a square-root-valued function and therefore lives on two sheets.
///
/// Conventions used here:
///
/// - `Physical`: the principal branch, chosen so that just above threshold on the real axis,
///   `q` is real and positive.
/// - `Unphysical`: the branch reached by crossing the channel cut once, implemented here
///   by flipping the sign of `q`.
///
/// This distinction matters for:
///
/// - analytic continuation of amplitudes into the complex plane,
/// - pole searches,
/// - coupled-channel Flatté-like parameterizations.
///
/// For ordinary evaluation of amplitudes on the real mass axis, `Physical` is usually the
/// appropriate choice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sheet {
    /// The physical sheet of the complex energy plane.
    Physical,
    /// The unphysical sheet obtained by flipping the sign of the channel momentum.
    Unphysical,
}

/// Computes the complex breakup momentum `q(s)` for a two-body channel on a chosen Riemann sheet.
///
/// The definition used is
///
/// `q(s, m1, m2) = sqrt((s - s_+)(s - s_-)) / (2 sqrt(s))`,
///
/// where
///
/// - `s_+ = (m1 + m2)^2`,
/// - `s_- = (m1 - m2)^2`.
///
/// On the physical sheet:
///
/// - above threshold, `q` is real and positive,
/// - below threshold, `q` is purely imaginary,
/// - for complex `s`, this gives the analytic continuation of the physical branch.
///
/// On the unphysical sheet, the sign of the result is flipped:
///
/// `q_unphysical(s) = -q_physical(s)`.
///
/// This function is the canonical analytic object from which the phase-space factor
/// `rho(s) = 2 q(s) / sqrt(s)` is constructed.
///
/// # Arguments
///
/// - `s`: squared invariant mass, possibly complex.
/// - `m1`: mass of the first daughter particle.
/// - `m2`: mass of the second daughter particle.
/// - `sheet`: the Riemann sheet to evaluate on.
///
/// # Returns
///
/// The complex breakup momentum on the requested sheet.
pub fn q_s(s: Complex64, m1: f64, m2: f64, sheet: Sheet) -> Complex64 {
    let sp = Complex64::from((m1 + m2).powi(2));
    let sm = Complex64::from((m1 - m2).powi(2));
    let q_phys = ((s - sp) * (s - sm)).sqrt() / (2.0 * s.sqrt());
    match sheet {
        Sheet::Physical => q_phys,
        Sheet::Unphysical => -q_phys,
    }
}

/// Computes the complex breakup momentum `q(m)` for a real mass `m` on a chosen Riemann sheet.
///
/// This is a convenience wrapper around [`q_s`], using `s = m^2`.
///
/// The definition is
///
/// `q(m, m1, m2) = q_s(m^2, m1, m2, sheet)`.
///
/// On the physical sheet:
///
/// - for `m > m1 + m2`, the result is real and positive,
/// - for `m < m1 + m2`, the result is purely imaginary.
///
/// # Arguments
///
/// - `m`: invariant mass of the parent state.
/// - `m1`: mass of the first daughter particle.
/// - `m2`: mass of the second daughter particle.
/// - `sheet`: the Riemann sheet to evaluate on.
///
/// # Returns
///
/// The complex breakup momentum evaluated at `s = m^2`.
pub fn q_m(m: f64, m1: f64, m2: f64, sheet: Sheet) -> Complex64 {
    q_s(m.powi(2).into(), m1, m2, sheet)
}

/// Computes the complex two-body phase-space factor `rho(s)` on a chosen Riemann sheet.
///
/// The definition used is
///
/// `rho(s, m1, m2) = 2 q(s, m1, m2) / sqrt(s)`.
///
/// Equivalently,
///
/// `rho(s, m1, m2) = sqrt((1 - (m1 + m2)^2 / s) (1 - (m1 - m2)^2 / s))`,
///
/// with the branch determined by the choice of [`Sheet`].
///
/// On the physical sheet:
///
/// - above threshold, `rho` is real and positive,
/// - below threshold, `rho` is purely imaginary.
///
/// # Arguments
///
/// - `s`: squared invariant mass, possibly complex.
/// - `m1`: mass of the first daughter particle.
/// - `m2`: mass of the second daughter particle.
/// - `sheet`: the Riemann sheet to evaluate on.
///
/// # Returns
///
/// The complex phase-space factor on the requested sheet.
pub fn rho_s(s: Complex64, m1: f64, m2: f64, sheet: Sheet) -> Complex64 {
    2.0 * q_s(s, m1, m2, sheet) / s.sqrt()
}

/// Computes the complex two-body phase-space factor `rho(m)` for a real mass `m` on a chosen
/// Riemann sheet.
///
/// This is a convenience wrapper around [`rho_s`], using `s = m^2`.
///
/// # Arguments
///
/// - `m`: invariant mass of the parent state.
/// - `m1`: mass of the first daughter particle.
/// - `m2`: mass of the second daughter particle.
/// - `sheet`: the Riemann sheet to evaluate on.
///
/// # Returns
///
/// The complex phase-space factor evaluated at `s = m^2`.
///
/// # Notes
///
/// On the physical sheet:
///
/// - above threshold, the result is real,
/// - below threshold, the result is imaginary.
pub fn rho_m(m: f64, m1: f64, m2: f64, sheet: Sheet) -> Complex64 {
    rho_s(m.powi(2).into(), m1, m2, sheet)
}

/// Selects which form of the Blatt-Weisskopf barrier factor is returned.
///
/// Two related conventions are supported:
///
/// - `Full`: the full barrier factor, which includes the threshold behavior
///   proportional to `q^L`.
/// - `Tensor`: the corresponding smooth "tensor" barrier factor, obtained by dividing the
///   full factor by `q^L`.
///
/// In other words, schematically,
///
/// `Full ~ q^L F_L(q)`,
///
/// while
///
/// `Tensor ~ F_L(q)`.
///
/// The tensor form is often useful when the explicit threshold dependence already appears in
/// a covariant tensor amplitude or elsewhere in the model, and one only wants the smooth
/// finite-size suppression factor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BarrierKind {
    /// The full barrier factor, including threshold powers of `q`.
    Full,
    /// The tensor barrier factor, with the explicit `q^L` dependence removed.
    Tensor,
}

fn blatt_weisskopf_polynomial(z: Complex64, l: usize) -> Complex64 {
    match l {
        0 => Complex64::ONE,
        1 => ((2.0 * z) / (z + 1.0)).sqrt(),
        2 => ((13.0 * z.powi(2)) / ((z - 3.0).powi(2) + 9.0 * z)).sqrt(),
        3 => {
            ((277.0 * z.powi(3)) / (z * (z - 15.0).powi(2) + 9.0 * (2.0 * z - 5.0).powi(2))).sqrt()
        }
        4 => ((12746.0 * z.powi(4))
            / ((z.powi(2) - 45.0 * z + 105.0).powi(2) + 25.0 * z * (2.0 * z - 21.0).powi(2)))
        .sqrt(),
        l => panic!("L = {l} is not yet implemented"),
    }
}

/// Default Blatt-Weisskopf radius parameter `q_R` in GeV.
///
/// Since the barrier factor depends on the dimensionless combination `z = q^2 / q_R^2`,
/// changing `q_R` changes the scale at which centrifugal suppression becomes important.
pub const QR_DEFAULT: f64 = 0.1973;

/// Computes the Blatt-Weisskopf centrifugal barrier factor in terms of the squared invariant mass
/// `s`, on a chosen Riemann sheet.
///
/// The computation proceeds by:
///
/// 1. evaluating the complex breakup momentum `q(s)` on the chosen sheet,
/// 2. forming `z = q^2 / q_R^2`,
/// 3. evaluating the typical Blatt-Weisskopf polynomial in `z`,
/// 4. optionally dividing by `q^L` if [`BarrierKind::Tensor`] is requested.
///
/// # Arguments
///
/// - `s`: squared invariant mass, possibly complex.
/// - `m1`: mass of the first daughter particle.
/// - `m2`: mass of the second daughter particle.
/// - `l`: orbital angular momentum.
/// - `q_r`: Blatt-Weisskopf radius parameter.
/// - `sheet`: the Riemann sheet used for the breakup momentum.
/// - `kind`: selects either the full or tensor version of the barrier factor.
///
/// # Returns
///
/// The requested complex barrier factor.
///
/// # Conventions
///
/// - `BarrierKind::Full` returns the full factor, which includes threshold powers
///   of `q`.
/// - `BarrierKind::Tensor` returns the smooth factor with those powers removed.
///
/// # Notes
///
/// This function uses the fully analytic complex momentum. As a result, the barrier factor can
/// become complex below threshold or when evaluated at complex `s`.
///
/// This behavior is appropriate for:
///
/// - analytic continuation,
/// - pole extraction,
/// - fully complex amplitude constructions.
///
/// # Panics
///
/// Panics if `l > 4`.
pub fn blatt_weisskopf_s(
    s: Complex64,
    m1: f64,
    m2: f64,
    l: usize,
    q_r: f64,
    sheet: Sheet,
    kind: BarrierKind,
) -> Complex64 {
    let q = q_s(s, m1, m2, sheet);
    let z = q * q / (q_r * q_r);
    let full = blatt_weisskopf_polynomial(z, l);
    match kind {
        BarrierKind::Full => full,
        BarrierKind::Tensor => full / q.powu(l as u32),
    }
}

/// Computes the Blatt-Weisskopf centrifugal barrier factor for a real parent mass `m0`,
/// on a chosen Riemann sheet.
///
/// This is a convenience wrapper around [`blatt_weisskopf_s`], using `s = m0^2`.
///
/// # Arguments
///
/// - `m0`: invariant mass of the parent state.
/// - `m1`: mass of the first daughter particle.
/// - `m2`: mass of the second daughter particle.
/// - `l`: orbital angular momentum.
/// - `q_r`: Blatt-Weisskopf radius parameter.
/// - `sheet`: the Riemann sheet used for the breakup momentum.
/// - `kind`: selects either the full or tensor version of the barrier factor.
///
/// # Returns
///
/// The requested complex barrier factor evaluated at `s = m0^2`.
///
/// # Notes
///
/// On the physical sheet, this is suitable for ordinary line-shape evaluation on the real axis.
/// On the unphysical sheet, it can be used in analytically continued amplitudes or pole studies.
///
/// As with [`blatt_weisskopf_s`], the returned value may be complex below threshold.
pub fn blatt_weisskopf_m(
    m0: f64,
    m1: f64,
    m2: f64,
    l: usize,
    q_r: f64,
    sheet: Sheet,
    kind: BarrierKind,
) -> Complex64 {
    blatt_weisskopf_s(m0.powi(2).into(), m1, m2, l, q_r, sheet, kind)
}

#[cfg(test)]
mod test {
    use approx::assert_relative_eq;
    use num::complex::Complex64;

    use crate::utils::functions::{BarrierKind, QR_DEFAULT};

    use super::{blatt_weisskopf_m, chi_minus, chi_plus, q_m, rho_m, spherical_harmonic, Sheet};

    #[test]
    fn test_spherical_harmonics() {
        use std::f64::consts::PI;
        let costhetas = [-1.0, -0.8, -0.3, 0.0, 0.3, 0.8, 1.0];
        let phis = [0.0, 0.3, 0.5, 0.8, 1.0].map(|v| v * PI * 2.0);
        for costheta in costhetas {
            for phi in phis {
                // L = 0
                let y00 = spherical_harmonic(0, 0, costheta, phi);
                let y00_true = Complex64::from(f64::sqrt(1.0 / (4.0 * PI)));
                assert_relative_eq!(y00.re, y00_true.re);
                assert_relative_eq!(y00.im, y00_true.im);
                // L = 1
                let y1n1 = spherical_harmonic(1, -1, costheta, phi);
                let y1n1_true = Complex64::from_polar(
                    f64::sqrt(3.0 / (8.0 * PI)) * f64::sin(f64::acos(costheta)),
                    -phi,
                );
                assert_relative_eq!(y1n1.re, y1n1_true.re);
                assert_relative_eq!(y1n1.im, y1n1_true.im);
                let y10 = spherical_harmonic(1, 0, costheta, phi);
                let y10_true = Complex64::from(f64::sqrt(3.0 / (4.0 * PI)) * costheta);
                assert_relative_eq!(y10.re, y10_true.re);
                assert_relative_eq!(y10.im, y10_true.im);
                let y11 = spherical_harmonic(1, 1, costheta, phi);
                let y11_true = Complex64::from_polar(
                    -f64::sqrt(3.0 / (8.0 * PI)) * f64::sin(f64::acos(costheta)),
                    phi,
                );
                assert_relative_eq!(y11.re, y11_true.re);
                assert_relative_eq!(y11.im, y11_true.im);
                // L = 2
                let y2n2 = spherical_harmonic(2, -2, costheta, phi);
                let y2n2_true = Complex64::from_polar(
                    f64::sqrt(15.0 / (32.0 * PI)) * f64::sin(f64::acos(costheta)).powi(2),
                    -2.0 * phi,
                );
                assert_relative_eq!(y2n2.re, y2n2_true.re);
                assert_relative_eq!(y2n2.im, y2n2_true.im);
                let y2n1 = spherical_harmonic(2, -1, costheta, phi);
                let y2n1_true = Complex64::from_polar(
                    f64::sqrt(15.0 / (8.0 * PI)) * f64::sin(f64::acos(costheta)) * costheta,
                    -phi,
                );
                assert_relative_eq!(y2n1.re, y2n1_true.re);
                assert_relative_eq!(y2n1.im, y2n1_true.im);
                let y20 = spherical_harmonic(2, 0, costheta, phi);
                let y20_true =
                    Complex64::from(f64::sqrt(5.0 / (16.0 * PI)) * (3.0 * costheta.powi(2) - 1.0));
                assert_relative_eq!(y20.re, y20_true.re);
                assert_relative_eq!(y20.im, y20_true.im);
                let y21 = spherical_harmonic(2, 1, costheta, phi);
                let y21_true = Complex64::from_polar(
                    -f64::sqrt(15.0 / (8.0 * PI)) * f64::sin(f64::acos(costheta)) * costheta,
                    phi,
                );
                assert_relative_eq!(y21.re, y21_true.re);
                assert_relative_eq!(y21.im, y21_true.im);
                let y22 = spherical_harmonic(2, 2, costheta, phi);
                let y22_true = Complex64::from_polar(
                    f64::sqrt(15.0 / (32.0 * PI)) * f64::sin(f64::acos(costheta)).powi(2),
                    2.0 * phi,
                );
                assert_relative_eq!(y22.re, y22_true.re);
                assert_relative_eq!(y22.im, y22_true.im);
            }
        }
    }

    #[test]
    fn test_momentum_functions() {
        assert_relative_eq!(chi_plus(1.3, 0.51, 0.62), 0.01776923076923098,);
        assert_relative_eq!(chi_minus(1.3, 0.51, 0.62), 0.9906923076923076,);
        let x0 = rho_m(f64::sqrt(1.3), 0.51, 0.62, Sheet::Physical);
        assert_relative_eq!(x0.re, 0.13267946426138);
        assert_relative_eq!(x0.im, 0.0);
        let x1 = rho_m(f64::sqrt(1.3), 1.23, 0.62, Sheet::Physical);
        assert_relative_eq!(x1.re, 0.0);
        assert_relative_eq!(x1.im, 1.0795209736472833);
        let y0 = q_m(1.2, 0.4, 0.5, Sheet::Physical);
        assert_relative_eq!(y0.re, 0.3954823004889093);
        assert_relative_eq!(y0.im, 0.0);
        let y1 = q_m(1.2, 1.4, 1.5, Sheet::Physical);
        assert_relative_eq!(y1.re, 0.0);
        assert_relative_eq!(y1.im, 1.3154464282347478);

        let w0 = blatt_weisskopf_m(
            1.2,
            0.4,
            0.5,
            0,
            QR_DEFAULT,
            Sheet::Physical,
            BarrierKind::Full,
        );
        assert_relative_eq!(w0.re, 1.0);
        assert_relative_eq!(w0.im, 0.0);
        let w1 = blatt_weisskopf_m(
            1.2,
            0.4,
            0.5,
            1,
            QR_DEFAULT,
            Sheet::Physical,
            BarrierKind::Full,
        );
        assert_relative_eq!(w1.re, 1.2654752018685698);
        assert_relative_eq!(w1.im, 0.0);
        let w2 = blatt_weisskopf_m(
            1.2,
            0.4,
            0.5,
            2,
            QR_DEFAULT,
            Sheet::Physical,
            BarrierKind::Full,
        );
        assert_relative_eq!(w2.re, 2.375285855793918);
        assert_relative_eq!(w2.im, 0.0);
        let w3 = blatt_weisskopf_m(
            1.2,
            0.4,
            0.5,
            3,
            QR_DEFAULT,
            Sheet::Physical,
            BarrierKind::Full,
        );
        assert_relative_eq!(w3.re, 5.62658768678507);
        assert_relative_eq!(w3.im, 0.0, epsilon = f64::EPSILON.sqrt());
        let w4 = blatt_weisskopf_m(
            1.2,
            0.4,
            0.5,
            4,
            QR_DEFAULT,
            Sheet::Physical,
            BarrierKind::Full,
        );
        assert_relative_eq!(w4.re, 12.747554064467208);
        assert_relative_eq!(w4.im, 0.0, epsilon = f64::EPSILON.sqrt());
        let w0im = blatt_weisskopf_m(
            1.2,
            1.4,
            1.5,
            0,
            QR_DEFAULT,
            Sheet::Physical,
            BarrierKind::Full,
        );
        assert_relative_eq!(w0im.re, 1.0);
        assert_relative_eq!(w0im.im, 0.0);
        let w1im = blatt_weisskopf_m(
            1.2,
            1.4,
            1.5,
            1,
            QR_DEFAULT,
            Sheet::Physical,
            BarrierKind::Full,
        );
        assert_relative_eq!(w1im.re, 1.430394249144933);
        assert_relative_eq!(w1im.im, 0.0);
        let w2im = blatt_weisskopf_m(
            1.2,
            1.4,
            1.5,
            2,
            QR_DEFAULT,
            Sheet::Physical,
            BarrierKind::Full,
        );
        assert_relative_eq!(w2im.re, 3.724659004227952);
        assert_relative_eq!(w2im.im, 0.0, epsilon = f64::EPSILON.sqrt());
        let w3im = blatt_weisskopf_m(
            1.2,
            1.4,
            1.5,
            3,
            QR_DEFAULT,
            Sheet::Physical,
            BarrierKind::Full,
        );
        assert_relative_eq!(w3im.re, 17.689297320491015);
        assert_relative_eq!(w3im.im, 0.0, epsilon = f64::EPSILON.sqrt());
        let w4im = blatt_weisskopf_m(
            1.2,
            1.4,
            1.5,
            4,
            QR_DEFAULT,
            Sheet::Physical,
            BarrierKind::Full,
        );
        assert_relative_eq!(w4im.re, 124.0525841825899);
        assert_relative_eq!(w4im.im, 0.0, epsilon = f64::EPSILON.sqrt());
    }
    #[test]
    #[should_panic]
    fn panicking_blatt_weisskopf() {
        blatt_weisskopf_m(
            1.2,
            0.4,
            0.5,
            5,
            QR_DEFAULT,
            Sheet::Physical,
            BarrierKind::Full,
        );
    }
}
