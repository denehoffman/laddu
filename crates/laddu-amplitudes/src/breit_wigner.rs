use laddu_expr::Expr;
use laddu_physics::{
    LadduPhysicsError, LadduPhysicsResult,
    math::{BarrierKind, QR_DEFAULT, Sheet, blatt_weisskopf_custom, q},
    quantum::L,
};
use num::complex::Complex64;

pub fn breit_wigner(s: impl Into<Expr>, mass: impl Into<Expr>, width: impl Into<Expr>) -> Expr {
    let mass = mass.into();
    let width = width.into();
    1.0 / (mass.powi(2) - s.into() - Complex64::I * mass * width)
}

pub fn relativistic_breit_wigner(
    s: impl Into<Expr>,
    mass: impl Into<Expr>,
    width: impl Into<Expr>,
    mass1: impl Into<Expr>,
    mass2: impl Into<Expr>,
    l: impl TryInto<L>,
) -> LadduPhysicsResult<Expr> {
    relativistic_breit_wigner_custom(s, mass, width, mass1, mass2, l, true, QR_DEFAULT)
}

#[allow(clippy::too_many_arguments)]
pub fn relativistic_breit_wigner_custom(
    s: impl Into<Expr>,
    mass: impl Into<Expr>,
    width: impl Into<Expr>,
    mass1: impl Into<Expr>,
    mass2: impl Into<Expr>,
    l: impl TryInto<L>,
    barrier_factors: bool,
    q_r: f64,
) -> LadduPhysicsResult<Expr> {
    let s = s.into();
    let mass = mass.into();
    let width = width.into();
    let mass1 = mass1.into();
    let mass2 = mass2.into();
    let l = l
        .try_into()
        .map_err(|_| LadduPhysicsError::ConversionError("L"))?;
    let q0 = q(mass.powi(2), &mass1, &mass2, Sheet::Physical);
    let q = q(&s, &mass1, &mass2, Sheet::Physical);
    let running_width = if barrier_factors {
        let f0 = blatt_weisskopf_custom(&q0, l, BarrierKind::Full, q_r)?;
        let f = blatt_weisskopf_custom(&q, l, BarrierKind::Full, q_r)?;
        width * (&mass / s.sqrt()) * (q / q0) * (f / f0).powi(2)
    } else {
        width * (&mass / s.sqrt()) * (q / q0).powi((2 * l.value() + 1) as i32)
    };
    Ok(Expr::from(1.0) / (mass.powi(2) - &s - Complex64::I * mass * running_width))
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use laddu_compile::CompiledModel;
    use laddu_physics::{
        l,
        math::{BLATT_WEISSKOPF_MAX_L, QR_DEFAULT},
    };
    use laddu_runtime::CpuBackend;

    use super::*;

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

    #[test]
    fn breit_wigner_variants_match_pole_normalization() {
        let mass = 1.2;
        let width = 0.13;
        let s = mass * mass;
        let expected_pole = Complex64::new(0.0, 1.0 / (mass * width));

        assert_complex_relative_eq(evaluate(breit_wigner(s, mass, width)), expected_pole, EPS);
        assert_complex_relative_eq(
            evaluate(relativistic_breit_wigner(s, mass, width, 0.4, 0.5, l!(1)).unwrap()),
            expected_pole,
            LOOSE_EPS,
        );
        assert_complex_relative_eq(
            evaluate(
                relativistic_breit_wigner_custom(s, mass, width, 0.4, 0.5, l!(1), true, QR_DEFAULT)
                    .unwrap(),
            ),
            expected_pole,
            LOOSE_EPS,
        );
    }

    #[test]
    fn invalid_static_configuration_is_rejected() {
        assert!(
            relativistic_breit_wigner(1.0, 0.1, 0.01, 0.2, 0.3, BLATT_WEISSKOPF_MAX_L + 1).is_err()
        );
        assert!(
            relativistic_breit_wigner_custom(1.0, 1.0, 0.1, 0.2, 0.3, l!(0), true, 0.0).is_err()
        );
    }
}
