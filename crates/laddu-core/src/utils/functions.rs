use factorial::Factorial;
use nalgebra::ComplexField;
use num::{complex::Complex64, Integer};
use polars::prelude::*;
use std::f64::consts::PI;

use crate::utils::vectorize;

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

/// Computes the phase-space factor $`\rho(s, m_1, m_2) = \sqrt(\chi_+(s, m_1, m_2)\chi_-(s, m_1, m_2))`$
pub fn rho(s: f64, m1: f64, m2: f64) -> Complex64 {
    let x: Complex64 = (chi_plus(s, m1, m2) * chi_minus(s, m1, m2)).into();
    x.sqrt()
}

/// Computes the breakup momentum (often denoted $`q`$) for a particle with mass $`m_0`$ decaying
/// into two particles with masses $`m_1`$ and $`m_2`$: $`\frac{m_0 \left|\rho(m_0^2, m_1, m_2)\right|}{2}`$.
///
/// Note that this version will always yield a real-valued number.
pub fn breakup_momentum(m0: f64, m1: f64, m2: f64) -> f64 {
    let s = m0.powi(2);
    let x: Complex64 = (chi_plus(s, m1, m2) * chi_minus(s, m1, m2)).into();
    x.abs().sqrt() * m0 / 2.0
}

/// Computes the breakup momentum (often denoted $`q`$) for a particle with mass $`m_0`$ decaying
/// into two particles with masses $`m_1`$ and $`m_2`$: $`\frac{m_0 \rho(m_0^2, m_1, m_2)}{2}`$.
pub fn complex_breakup_momentum(m0: f64, m1: f64, m2: f64) -> Complex64 {
    rho(f64::powi(m0, 2), m1, m2) * m0 / 2.0
}

/// Computes the Blatt-Weisskopf centrifugal barrier factor for a particle with mass $`m_0`$ and
/// angular momentum $`\ell`$ decaying to two particles with masses $`m_1`$ and $`m_2`$.
///
/// Note that this version uses absolute form of the breakup momentum, so the results are
/// real-valued. Additionally, this method uses an impact parameter of $`0.1973\text{GeV}^{-1}`$.
/// Currently, only values of $`\ell <= 4`$ are implemented.
pub fn blatt_weisskopf(m0: f64, m1: f64, m2: f64, l: usize) -> f64 {
    let q = breakup_momentum(m0, m1, m2);
    let z = q.powi(2) / 0.1973.powi(2);
    match l {
        0 => 1.0,
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

/// Computes the Blatt-Weisskopf centrifugal barrier factor for a particle with mass $`m_0`$ and
/// angular momentum $`\ell`$ decaying to two particles with masses $`m_1`$ and $`m_2`$.
///
/// Note that this method uses an impact parameter of $`0.1973\text{GeV}^{-1}`$. Currently, only
/// values of $`\ell <= 4`$ are implemented.
pub fn complex_blatt_weisskopf(m0: f64, m1: f64, m2: f64, l: usize) -> Complex64 {
    let q = complex_breakup_momentum(m0, m1, m2);
    let z = q.powi(2) / 0.1973.powi(2);
    match l {
        0 => Complex64::ONE,
        1 => ((z * 2.0) / (z + 1.0)).sqrt(),
        2 => ((z.powi(2) * 13.0) / ((z - 3.0).powi(2) + z * 9.0)).sqrt(),
        3 => {
            ((z.powi(3) * 277.0) / (z * (z - 15.0).powi(2) + (z * 2.0 - 5.0).powi(2) * 9.0)).sqrt()
        }
        4 => ((z.powi(4) * 12746.0)
            / ((z.powi(2) - z * 45.0 + 105.0).powi(2) + z * 25.0 * (z * 2.0 - 21.0).powi(2)))
        .sqrt(),
        l => panic!("L = {l} is not yet implemented"),
    }
}

pub fn spherical_harmonic_polars(l: usize, m: isize, costheta: Expr, phi: Expr) -> (Expr, Expr) {
    let base = format!("Y{l}{m}");
    vectorize::<Complex64, _, _>(base, [costheta, phi], move |row| {
        spherical_harmonic(l, m, row[0], row[1])
    })
    .as_cscalar()
}

pub fn chi_plus_polars(s: Expr, m1: Expr, m2: Expr) -> Expr {
    let base = "chi_plus";
    vectorize::<f64, _, _>(base, [s, m1, m2], move |row| {
        chi_plus(row[0], row[1], row[2])
    })
    .as_scalar()
}

pub fn chi_minus_polars(s: Expr, m1: Expr, m2: Expr) -> Expr {
    let base = "chi_minus";
    vectorize::<f64, _, _>(base, [s, m1, m2], move |row| {
        chi_minus(row[0], row[1], row[2])
    })
    .as_scalar()
}
pub fn rho_polars(s: Expr, m1: Expr, m2: Expr) -> (Expr, Expr) {
    let base = "rho";
    vectorize::<Complex64, _, _>(base, [s, m1, m2], move |row| rho(row[0], row[1], row[2]))
        .as_cscalar()
}
pub fn breakup_momentum_polars(m0: Expr, m1: Expr, m2: Expr) -> Expr {
    let base = "q";
    vectorize::<f64, _, _>(base, [m0, m1, m2], move |row| {
        breakup_momentum(row[0], row[1], row[2])
    })
    .as_scalar()
}

pub fn complex_breakup_momentum_polars(m0: Expr, m1: Expr, m2: Expr) -> (Expr, Expr) {
    let base = "q";
    vectorize::<Complex64, _, _>(base, [m0, m1, m2], move |row| {
        complex_breakup_momentum(row[0], row[1], row[2])
    })
    .as_cscalar()
}

pub fn blatt_weisskopf_polars(m0: Expr, m1: Expr, m2: Expr, l: usize) -> Expr {
    let base = format!("B_l{l}");
    vectorize::<f64, _, _>(base, [m0, m1, m2], move |row| {
        blatt_weisskopf(row[0], row[1], row[2], l)
    })
    .as_scalar()
}

pub fn complex_blatt_weisskopf_polars(m0: Expr, m1: Expr, m2: Expr, l: usize) -> (Expr, Expr) {
    let base = format!("B_l{l}");
    vectorize::<Complex64, _, _>(base, [m0, m1, m2], move |row| {
        complex_blatt_weisskopf(row[0], row[1], row[2], l)
    })
    .as_cscalar()
}

#[cfg(test)]
mod test {
    use approx::assert_relative_eq;
    use num::complex::Complex64;

    use crate::utils::functions::{
        blatt_weisskopf, breakup_momentum, chi_minus, complex_breakup_momentum, rho,
        spherical_harmonic,
    };

    use super::{chi_plus, complex_blatt_weisskopf};

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
        assert_relative_eq!(
            chi_plus(1.3, 0.51, 0.62),
            0.01776923076923098,
            epsilon = f64::EPSILON.sqrt()
        );
        assert_relative_eq!(
            chi_minus(1.3, 0.51, 0.62),
            0.9906923076923076,
            epsilon = f64::EPSILON.sqrt()
        );
        let x0 = rho(1.3, 0.51, 0.62);
        assert_relative_eq!(x0.re, 0.1326794642613792, epsilon = f64::EPSILON.sqrt());
        assert_relative_eq!(x0.im, 0.0);
        let x1 = rho(1.3, 1.23, 0.62);
        assert_relative_eq!(x1.re, 0.0);
        assert_relative_eq!(x1.im, 1.0795209736472833, epsilon = f64::EPSILON.sqrt());
        let y0 = breakup_momentum(1.2, 0.4, 0.5);
        assert_relative_eq!(y0, 0.3954823004889093, epsilon = f64::EPSILON.sqrt());
        let y1 = breakup_momentum(1.2, 1.4, 1.5);
        assert_relative_eq!(y1, 1.3154464282347478, epsilon = f64::EPSILON.sqrt());
        let y2 = complex_breakup_momentum(1.2, 0.4, 0.5);
        assert_relative_eq!(y2.re, 0.3954823004889093, epsilon = f64::EPSILON.sqrt());
        assert_relative_eq!(y2.im, 0.0);
        let y3 = complex_breakup_momentum(1.2, 1.4, 1.5);
        assert_relative_eq!(y3.re, 0.0);
        assert_relative_eq!(y3.im, 1.3154464282347478, epsilon = f64::EPSILON.sqrt());
        assert_relative_eq!(y0, y2.re);
        assert_relative_eq!(y1, y3.im);

        let z0 = blatt_weisskopf(1.2, 0.4, 0.5, 0);
        assert_relative_eq!(z0, 1.0);
        let z1 = blatt_weisskopf(1.2, 0.4, 0.5, 1);
        assert_relative_eq!(z1, 1.2654752018685698, epsilon = f64::EPSILON.sqrt());
        let z2 = blatt_weisskopf(1.2, 0.4, 0.5, 2);
        assert_relative_eq!(z2, 2.375285855793918, epsilon = f64::EPSILON.sqrt());
        let z3 = blatt_weisskopf(1.2, 0.4, 0.5, 3);
        assert_relative_eq!(z3, 5.6265876867850695, epsilon = f64::EPSILON.sqrt());
        let z4 = blatt_weisskopf(1.2, 0.4, 0.5, 4);
        assert_relative_eq!(z4, 12.747554064467208, epsilon = f64::EPSILON.sqrt());
        let z0im = blatt_weisskopf(1.2, 1.4, 0.5, 0);
        assert_relative_eq!(z0im, 1.0);
        let z1im = blatt_weisskopf(1.2, 1.4, 1.5, 1);
        assert_relative_eq!(z1im, 1.398569848337654, epsilon = f64::EPSILON.sqrt());
        let z2im = blatt_weisskopf(1.2, 1.4, 1.5, 2);
        assert_relative_eq!(z2im, 3.482294988252171, epsilon = f64::EPSILON.sqrt());
        let z3im = blatt_weisskopf(1.2, 1.4, 1.5, 3);
        assert_relative_eq!(z3im, 15.450855647831101, epsilon = f64::EPSILON.sqrt());
        let z4im = blatt_weisskopf(1.2, 1.4, 1.5, 4);
        assert_relative_eq!(z4im, 98.48799450927207, epsilon = f64::EPSILON.sqrt());

        let w0 = complex_blatt_weisskopf(1.2, 0.4, 0.5, 0);
        assert_relative_eq!(w0.re, 1.0);
        assert_relative_eq!(w0.im, 0.0);
        let w1 = complex_blatt_weisskopf(1.2, 0.4, 0.5, 1);
        assert_relative_eq!(w1.re, 1.2654752018685698, epsilon = f64::EPSILON.sqrt());
        assert_relative_eq!(w1.im, 0.0);
        let w2 = complex_blatt_weisskopf(1.2, 0.4, 0.5, 2);
        assert_relative_eq!(w2.re, 2.375285855793918, epsilon = f64::EPSILON.sqrt());
        assert_relative_eq!(w2.im, 0.0);
        let w3 = complex_blatt_weisskopf(1.2, 0.4, 0.5, 3);
        assert_relative_eq!(w3.re, 5.62658768678507, epsilon = f64::EPSILON.sqrt());
        assert_relative_eq!(w3.im, 0.0, epsilon = f64::EPSILON.sqrt());
        let w4 = complex_blatt_weisskopf(1.2, 0.4, 0.5, 4);
        assert_relative_eq!(w4.re, 12.747554064467208, epsilon = f64::EPSILON.sqrt());
        assert_relative_eq!(w4.im, 0.0, epsilon = f64::EPSILON.sqrt());
        let w0im = complex_blatt_weisskopf(1.2, 1.4, 1.5, 0);
        assert_relative_eq!(w0im.re, 1.0);
        assert_relative_eq!(w0im.im, 0.0);
        let w1im = complex_blatt_weisskopf(1.2, 1.4, 1.5, 1);
        assert_relative_eq!(w1im.re, 1.430394249144933, epsilon = f64::EPSILON.sqrt());
        assert_relative_eq!(w1im.im, 0.0);
        let w2im = complex_blatt_weisskopf(1.2, 1.4, 1.5, 2);
        assert_relative_eq!(w2im.re, 3.724659004227952, epsilon = f64::EPSILON.sqrt());
        assert_relative_eq!(w2im.im, 0.0, epsilon = f64::EPSILON.sqrt());
        let w3im = complex_blatt_weisskopf(1.2, 1.4, 1.5, 3);
        assert_relative_eq!(w3im.re, 17.689297320491015, epsilon = f64::EPSILON.sqrt());
        assert_relative_eq!(w3im.im, 0.0, epsilon = f64::EPSILON.sqrt());
        let w4im = complex_blatt_weisskopf(1.2, 1.4, 1.5, 4);
        assert_relative_eq!(w4im.re, 124.0525841825899, epsilon = f64::EPSILON.sqrt());
        assert_relative_eq!(w4im.im, 0.0, epsilon = f64::EPSILON.sqrt());

        assert_relative_eq!(z0, w0.re);
        assert_relative_eq!(z1, w1.re);
        assert_relative_eq!(z2, w2.re);
        assert_relative_eq!(z3, w3.re);
        assert_relative_eq!(z4, w4.re);
    }
    #[test]
    #[should_panic]
    fn panicking_blatt_weisskopf() {
        blatt_weisskopf(1.2, 0.4, 0.5, 5);
    }
    #[test]
    #[should_panic]
    fn panicking_complex_blatt_weisskopf() {
        complex_blatt_weisskopf(1.2, 0.4, 0.5, 5);
    }
}
