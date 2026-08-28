use crate::{KMatrixError, KMatrixResult};
use laddu_expr::{Expr, ExprShape, matmul, matrix_from_flat, solve, vector};
use laddu_physics::{
    LadduPhysicsError,
    math::{BarrierKind, Sheet, blatt_weisskopf_custom, q},
    quantum::L,
};

/// Constructs channel-by-pole Blatt-Weisskopf barrier ratios.
///
/// # Errors
///
/// Returns [`KMatrixError`] when inputs have incompatible shapes, contain no
/// channels or poles, `l` is invalid, `q_r` is not scalar, or a literal
/// `q_r` is not positive, finite, and real. For other expressions, the caller
/// must keep their values positive, finite, and real during evaluation.
pub fn blatt_weisskopf_barriers(
    s: impl Into<Expr>,
    channel_mass_1: impl Into<Expr>,
    channel_mass_2: impl Into<Expr>,
    pole_masses: impl Into<Expr>,
    l: impl TryInto<L>,
    q_r: impl Into<Expr>,
) -> KMatrixResult<Expr> {
    let s = s.into();
    let channel_mass_1 = channel_mass_1.into();
    let channel_mass_2 = channel_mass_2.into();
    let pole_masses = pole_masses.into();
    let q_r = q_r.into();
    let l = l
        .try_into()
        .map_err(|_| LadduPhysicsError::ConversionError("L"))?;

    if q_r.shape()? != ExprShape::Scalar {
        return Err(KMatrixError::InvalidShape(format!(
            "q_r must be scalar, got {}",
            q_r.shape()?
        )));
    }

    if s.shape()? != ExprShape::Scalar {
        return Err(KMatrixError::InvalidShape(format!(
            "s must be scalar, got {}",
            s.shape()?
        )));
    }
    let ExprShape::Vector { len: channels } = channel_mass_1.shape()? else {
        return Err(KMatrixError::InvalidShape(format!(
            "channel_mass_1 must be a vector, got {}",
            channel_mass_1.shape()?
        )));
    };
    if channel_mass_2.shape()? != (ExprShape::Vector { len: channels }) {
        return Err(KMatrixError::InvalidShape(format!(
            "channel_mass_2 must be vector[{channels}], got {}",
            channel_mass_2.shape()?
        )));
    }
    let ExprShape::Vector { len: poles } = pole_masses.shape()? else {
        return Err(KMatrixError::InvalidShape(format!(
            "pole_masses must be a vector, got {}",
            pole_masses.shape()?
        )));
    };
    if channels == 0 || poles == 0 {
        return Err(KMatrixError::InvalidShape(
            "barrier matrices require at least one channel and one pole".into(),
        ));
    }

    let mut elements = Vec::with_capacity(channels * poles);
    for channel in 0..channels {
        let mass_1 = channel_mass_1.component(channel);
        let mass_2 = channel_mass_2.component(channel);
        let q_s = q(&s, &mass_1, &mass_2, Sheet::Physical);
        let numerator = blatt_weisskopf_custom(q_s, l, BarrierKind::Full, &q_r)?;
        for pole in 0..poles {
            let pole_mass = pole_masses.component(pole);
            let q_pole = q(pole_mass.powi(2), &mass_1, &mass_2, Sheet::Physical);
            let denominator = blatt_weisskopf_custom(q_pole, l, BarrierKind::Full, &q_r)?;
            elements.push(&numerator / denominator);
        }
    }
    Ok(matrix_from_flat(channels, poles, elements)?)
}

/// Constructs the pole contribution to a coupled-channel K matrix.
///
/// # Errors
///
/// Returns [`KMatrixError`] when scalar, vector, or matrix input shapes are
/// inconsistent or contain no channels or poles.
pub fn k_matrix(
    s: impl Into<Expr>,
    pole_masses: impl Into<Expr>,
    couplings: impl Into<Expr>,
    barriers: impl Into<Expr>,
) -> KMatrixResult<Expr> {
    k_matrix_impl(
        s.into(),
        pole_masses.into(),
        couplings.into(),
        barriers.into(),
        None,
    )
}

/// Constructs a coupled-channel K matrix with a non-pole background.
///
/// # Errors
///
/// Returns [`KMatrixError`] when input shapes are inconsistent, contain no
/// channels or poles, or `background` is not channel-by-channel.
pub fn k_matrix_with_background(
    s: impl Into<Expr>,
    pole_masses: impl Into<Expr>,
    couplings: impl Into<Expr>,
    barriers: impl Into<Expr>,
    background: impl Into<Expr>,
) -> KMatrixResult<Expr> {
    k_matrix_impl(
        s.into(),
        pole_masses.into(),
        couplings.into(),
        barriers.into(),
        Some(background.into()),
    )
}

fn k_matrix_impl(
    s: Expr,
    pole_masses: Expr,
    couplings: Expr,
    barriers: Expr,
    background: Option<Expr>,
) -> KMatrixResult<Expr> {
    let (channels, poles) = validate_pole_inputs(&s, &pole_masses, &couplings, &barriers)?;
    if let Some(background) = &background {
        let expected = ExprShape::Matrix {
            rows: channels,
            cols: channels,
        };
        if background.shape()? != expected {
            return Err(KMatrixError::InvalidShape(format!(
                "K background must be {expected}, got {}",
                background.shape()?
            )));
        }
    }
    let (pole_product, reduced_products) = pole_products(&s, &pole_masses, poles);
    let mut elements = Vec::with_capacity(channels * channels);
    for i in 0..channels {
        for j in 0..channels {
            let mut element = Expr::from(0.0);
            for (pole, reduced_product) in reduced_products.iter().enumerate() {
                element += barriers.matrix_element(i, pole)
                    * barriers.matrix_element(j, pole)
                    * couplings.matrix_element(i, pole)
                    * couplings.matrix_element(j, pole)
                    * reduced_product;
            }
            if let Some(background) = &background {
                element += background.matrix_element(i, j) * &pole_product;
            }
            elements.push(element);
        }
    }
    Ok(matrix_from_flat(channels, channels, elements)?)
}

/// Constructs a coupled-channel production vector from pole terms.
///
/// # Errors
///
/// Returns [`KMatrixError`] when input shapes are inconsistent or contain no
/// channels or poles.
pub fn p_vector(
    s: impl Into<Expr>,
    pole_masses: impl Into<Expr>,
    production: impl Into<Expr>,
    couplings: impl Into<Expr>,
    barriers: impl Into<Expr>,
) -> KMatrixResult<Expr> {
    p_vector_impl(
        s.into(),
        pole_masses.into(),
        production.into(),
        couplings.into(),
        barriers.into(),
        None,
    )
}

/// Constructs a production vector with a non-pole background.
///
/// # Errors
///
/// Returns [`KMatrixError`] when input shapes are inconsistent, contain no
/// channels or poles, or `background` has the wrong channel length.
pub fn p_vector_with_background(
    s: impl Into<Expr>,
    pole_masses: impl Into<Expr>,
    production: impl Into<Expr>,
    couplings: impl Into<Expr>,
    barriers: impl Into<Expr>,
    background: impl Into<Expr>,
) -> KMatrixResult<Expr> {
    p_vector_impl(
        s.into(),
        pole_masses.into(),
        production.into(),
        couplings.into(),
        barriers.into(),
        Some(background.into()),
    )
}

fn p_vector_impl(
    s: Expr,
    pole_masses: Expr,
    production: Expr,
    couplings: Expr,
    barriers: Expr,
    background: Option<Expr>,
) -> KMatrixResult<Expr> {
    let (channels, poles) = validate_pole_inputs(&s, &pole_masses, &couplings, &barriers)?;
    let expected_production = ExprShape::Vector { len: poles };
    if production.shape()? != expected_production {
        return Err(KMatrixError::InvalidShape(format!(
            "production must be {expected_production}, got {}",
            production.shape()?
        )));
    }
    if let Some(background) = &background {
        let expected = ExprShape::Vector { len: channels };
        if background.shape()? != expected {
            return Err(KMatrixError::InvalidShape(format!(
                "P background must be {expected}, got {}",
                background.shape()?
            )));
        }
    }
    let (pole_product, reduced_products) = pole_products(&s, &pole_masses, poles);
    let mut elements = Vec::with_capacity(channels);
    for channel in 0..channels {
        let mut element = Expr::from(0.0);
        for (pole, reduced_product) in reduced_products.iter().enumerate() {
            element += production.component(pole)
                * barriers.matrix_element(channel, pole)
                * couplings.matrix_element(channel, pole)
                * reduced_product;
        }
        if let Some(background) = &background {
            element += background.component(channel) * &pole_product;
        }
        elements.push(element);
    }
    Ok(vector(elements))
}

/// Solves the coupled-channel final-state-interaction equation for an F vector.
///
/// # Errors
///
/// Returns [`KMatrixError`] when the supplied scalar, vectors, and matrices
/// have incompatible channel or pole dimensions.
pub fn f_vector(
    s: impl Into<Expr>,
    pole_masses: impl Into<Expr>,
    k: impl Into<Expr>,
    p: impl Into<Expr>,
    phase_space: impl Into<Expr>,
) -> KMatrixResult<Expr> {
    let s = s.into();
    let pole_masses = pole_masses.into();
    let k = k.into();
    let p = p.into();
    let phase_space = phase_space.into();
    if s.shape()? != ExprShape::Scalar {
        return Err(KMatrixError::InvalidShape(format!(
            "s must be scalar, got {}",
            s.shape()?
        )));
    }
    let ExprShape::Vector { len: poles } = pole_masses.shape()? else {
        return Err(KMatrixError::InvalidShape(format!(
            "pole_masses must be a vector, got {}",
            pole_masses.shape()?
        )));
    };
    let ExprShape::Vector { len: channels } = p.shape()? else {
        return Err(KMatrixError::InvalidShape(format!(
            "P must be a vector, got {}",
            p.shape()?
        )));
    };
    let expected_matrix = ExprShape::Matrix {
        rows: channels,
        cols: channels,
    };
    if k.shape()? != expected_matrix {
        return Err(KMatrixError::InvalidShape(format!(
            "K must be {expected_matrix}, got {}",
            k.shape()?
        )));
    }
    if phase_space.shape()? != expected_matrix {
        return Err(KMatrixError::InvalidShape(format!(
            "phase_space must be {expected_matrix}, got {}",
            phase_space.shape()?
        )));
    }
    if poles == 0 || channels == 0 {
        return Err(KMatrixError::InvalidShape(
            "F vectors require at least one channel and one pole".into(),
        ));
    }

    let (pole_product, _) = pole_products(&s, &pole_masses, poles);
    let kc = matmul(k, phase_space);
    let mut system = Vec::with_capacity(channels * channels);
    for row in 0..channels {
        for col in 0..channels {
            let identity = if row == col {
                pole_product.clone()
            } else {
                Expr::from(0.0)
            };
            system.push(identity + kc.matrix_element(row, col));
        }
    }
    Ok(solve(matrix_from_flat(channels, channels, system)?, p))
}

fn validate_pole_inputs(
    s: &Expr,
    pole_masses: &Expr,
    couplings: &Expr,
    barriers: &Expr,
) -> KMatrixResult<(usize, usize)> {
    if s.shape()? != ExprShape::Scalar {
        return Err(KMatrixError::InvalidShape(format!(
            "s must be scalar, got {}",
            s.shape()?
        )));
    }
    let ExprShape::Vector { len: poles } = pole_masses.shape()? else {
        return Err(KMatrixError::InvalidShape(format!(
            "pole_masses must be a vector, got {}",
            pole_masses.shape()?
        )));
    };
    let ExprShape::Matrix {
        rows: channels,
        cols,
    } = couplings.shape()?
    else {
        return Err(KMatrixError::InvalidShape(format!(
            "couplings must be a matrix, got {}",
            couplings.shape()?
        )));
    };
    if cols != poles {
        return Err(KMatrixError::InvalidShape(format!(
            "couplings has {cols} pole columns but pole_masses has length {poles}"
        )));
    }
    let expected_barriers = ExprShape::Matrix {
        rows: channels,
        cols: poles,
    };
    if barriers.shape()? != expected_barriers {
        return Err(KMatrixError::InvalidShape(format!(
            "barriers must be {expected_barriers}, got {}",
            barriers.shape()?
        )));
    }
    if channels == 0 || poles == 0 {
        return Err(KMatrixError::InvalidShape(
            "K-matrix expressions require at least one channel and one pole".into(),
        ));
    }
    Ok((channels, poles))
}

fn pole_products(s: &Expr, pole_masses: &Expr, poles: usize) -> (Expr, Vec<Expr>) {
    let denominators = (0..poles)
        .map(|pole| pole_masses.component(pole).powi(2) - s)
        .collect::<Vec<_>>();
    let pole_product = denominators
        .iter()
        .fold(Expr::from(1.0), |product, denominator| {
            product * denominator
        });
    let reduced_products = (0..poles)
        .map(|excluded| {
            denominators
                .iter()
                .enumerate()
                .filter(|(pole, _)| *pole != excluded)
                .fold(Expr::from(1.0), |product, (_, denominator)| {
                    product * denominator
                })
        })
        .collect();
    (pole_product, reduced_products)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use laddu_compile::CompiledModel;
    use laddu_expr::{BinaryOp, ExprNode, complex, matrix, parameters::Parameter};
    use laddu_runtime::CpuBackend;
    use nalgebra::{Matrix2, Vector2};
    use num::complex::Complex64;

    use super::*;

    fn evaluate(expr: Expr) -> Complex64 {
        let model = CompiledModel::from_expr(&expr).unwrap();
        let params = model.params().default_values();
        CpuBackend.prepare(&model).evaluate(&params).unwrap()
    }

    #[test]
    fn barrier_matrix_preserves_reference_momentum_expression_gradients() {
        let radius = 0.2;
        let q_r = 2.0 * Expr::from(Parameter::free("radius").with_initial(radius));
        let barriers = blatt_weisskopf_barriers(
            1.1,
            vector([0.2, 0.3]),
            vector([0.3, 0.4]),
            vector([1.3, 1.5]),
            1,
            q_r,
        )
        .unwrap();
        let step = 1.0e-6;
        for row in 0..2 {
            for col in 0..2 {
                let model = CompiledModel::from_expr(&barriers.matrix_element(row, col)).unwrap();
                let result = CpuBackend
                    .prepare(&model)
                    .evaluate_with_gradient(&model.params().default_values())
                    .unwrap();
                let values = [radius, radius + step, radius - step].map(|value| {
                    let numeric = blatt_weisskopf_barriers(
                        1.1,
                        vector([0.2, 0.3]),
                        vector([0.3, 0.4]),
                        vector([1.3, 1.5]),
                        1,
                        2.0 * value,
                    )
                    .unwrap();
                    evaluate(numeric.matrix_element(row, col))
                });
                let finite_difference = (values[1] - values[2]) / (2.0 * step);
                assert_relative_eq!(result.value().re, values[0].re, epsilon = 1.0e-12);
                assert_relative_eq!(
                    result.gradient()[0].re,
                    finite_difference.re,
                    epsilon = 1.0e-8
                );
                assert_relative_eq!(
                    result.gradient()[0].im,
                    finite_difference.im,
                    epsilon = 1.0e-8
                );
                assert!(result.gradient()[0].norm() > 1.0e-5);
            }
        }
    }

    #[test]
    fn one_channel_k_p_and_f_match_cleared_scalar_equation() {
        let s = 1.5;
        let masses = vector([2.0]);
        let couplings = matrix([[3.0]]);
        let barriers = matrix([[1.0]]);
        let production = vector([Complex64::new(0.5, 0.25)]);
        let k_background = matrix([[0.2]]);
        let p_background = vector([Complex64::new(0.1, -0.05)]);
        let phase_space = matrix([[Complex64::new(0.0, -0.4)]]);

        let k = k_matrix_with_background(s, &masses, &couplings, &barriers, k_background).unwrap();
        let p = p_vector_with_background(s, &masses, production, couplings, barriers, p_background)
            .unwrap();
        let f = f_vector(s, masses, &k, &p, phase_space).unwrap();

        let d = 4.0 - s;
        let expected_k = 9.0 + 0.2 * d;
        let expected_p = Complex64::new(0.5, 0.25) * 3.0 + Complex64::new(0.1, -0.05) * d;
        let expected = expected_p / (d + expected_k * Complex64::new(0.0, -0.4));
        let actual = evaluate(f.component(0));
        assert_relative_eq!(actual.re, expected.re, epsilon = 1.0e-12);
        assert_relative_eq!(actual.im, expected.im, epsilon = 1.0e-12);
        assert_relative_eq!(evaluate(k.matrix_element(0, 0)).re, expected_k);
        assert_relative_eq!(evaluate(p.component(0)).re, expected_p.re);
    }

    #[test]
    fn one_channel_f_gradient_matches_closed_scalar_equation() {
        let s = 1.5;
        let mass = 2.0_f64;
        let coupling = 3.0;
        let background = 0.2;
        let phase_space = Complex64::new(0.0, -0.4);
        let beta = complex(
            Parameter::free("beta_re").with_initial(0.5),
            Parameter::free("beta_im").with_initial(0.25),
        );
        let masses = vector([mass]);
        let k = k_matrix_with_background(
            s,
            &masses,
            matrix([[coupling]]),
            matrix([[1.0]]),
            matrix([[background]]),
        )
        .unwrap();
        let p = p_vector(
            s,
            &masses,
            vector([beta]),
            matrix([[coupling]]),
            matrix([[1.0]]),
        )
        .unwrap();
        let expression = f_vector(s, masses, k, p, matrix([[phase_space]]))
            .unwrap()
            .component(0);
        let model = CompiledModel::from_expr(&expression).unwrap();
        let params = model.params().default_values();
        let result = CpuBackend
            .prepare(&model)
            .evaluate_with_gradient(&params)
            .unwrap();

        let pole = mass.powi(2) - s;
        let denominator = pole + (coupling.powi(2) + background * pole) * phase_space;
        let expected_value = Complex64::new(0.5, 0.25) * coupling / denominator;
        let expected_beta_re = Complex64::from(coupling) / denominator;
        assert_relative_eq!(result.value().re, expected_value.re, epsilon = 1.0e-12);
        assert_relative_eq!(result.value().im, expected_value.im, epsilon = 1.0e-12);
        assert_relative_eq!(
            result.gradient()[0].re,
            expected_beta_re.re,
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            result.gradient()[0].im,
            expected_beta_re.im,
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            result.gradient()[1].re,
            -expected_beta_re.im,
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            result.gradient()[1].im,
            expected_beta_re.re,
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn cleared_pole_construction_contains_no_division_nodes() {
        let masses = vector([1.0, 2.0]);
        let k = k_matrix(
            1.2,
            masses,
            matrix([[0.3, 0.4], [0.5, 0.6]]),
            matrix([[1.0, 1.0], [1.0, 1.0]]),
        )
        .unwrap();
        assert!(!k.to_graph().nodes().iter().any(|node| {
            matches!(
                node,
                ExprNode::Binary {
                    op: BinaryOp::Div,
                    ..
                }
            )
        }));
    }

    #[test]
    fn mismatched_shapes_are_rejected_before_compilation() {
        let error = k_matrix(
            1.0,
            vector([1.2, 1.4]),
            matrix([[0.1], [0.2]]),
            matrix([[1.0], [1.0]]),
        )
        .unwrap_err();
        assert!(error.to_string().contains("pole columns"));
    }

    #[test]
    fn two_channel_f_vector_matches_direct_non_diagonal_solve() {
        let s = 1.3;
        let pole_masses = vector([1.5, 2.0]);
        let couplings = matrix([[0.2, 0.4], [0.3, 0.5]]);
        let barriers = matrix([[1.0, 0.8], [0.9, 1.1]]);
        let production = vector([Complex64::new(0.6, 0.1), Complex64::new(-0.2, 0.3)]);
        let k_background = matrix([[0.1, -0.02], [-0.02, 0.05]]);
        let p_background = vector([Complex64::new(0.03, 0.01), Complex64::new(-0.04, 0.02)]);
        let phase_space_values = [
            Complex64::new(0.1, -0.3),
            Complex64::new(0.02, 0.01),
            Complex64::new(-0.01, 0.03),
            Complex64::new(0.2, -0.25),
        ];
        let phase_space = matrix_from_flat(2, 2, phase_space_values).unwrap();

        let k =
            k_matrix_with_background(s, &pole_masses, &couplings, &barriers, k_background).unwrap();
        let p = p_vector_with_background(
            s,
            &pole_masses,
            production,
            &couplings,
            &barriers,
            p_background,
        )
        .unwrap();
        let f = f_vector(s, pole_masses, &k, &p, phase_space).unwrap();

        let k_numeric = Matrix2::new(
            evaluate(k.matrix_element(0, 0)),
            evaluate(k.matrix_element(0, 1)),
            evaluate(k.matrix_element(1, 0)),
            evaluate(k.matrix_element(1, 1)),
        );
        let p_numeric = Vector2::new(evaluate(p.component(0)), evaluate(p.component(1)));
        let c_numeric = Matrix2::from_row_slice(&phase_space_values);
        let pole_product = (1.5_f64.powi(2) - s) * (2.0_f64.powi(2) - s);
        let expected = (Matrix2::identity() * Complex64::from(pole_product)
            + k_numeric * c_numeric)
            .lu()
            .solve(&p_numeric)
            .unwrap();

        for channel in 0..2 {
            let actual = evaluate(f.component(channel));
            assert_relative_eq!(actual.re, expected[channel].re, epsilon = 1.0e-12);
            assert_relative_eq!(actual.im, expected[channel].im, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn cleared_form_evaluates_at_a_pole_without_dividing_by_zero() {
        let pole_mass = 1.5;
        let coupling = 0.4;
        let production = Complex64::new(0.7, -0.2);
        let phase_space = Complex64::new(0.1, -0.3);
        let pole_masses = vector([pole_mass]);
        let k = k_matrix(
            pole_mass.powi(2),
            &pole_masses,
            matrix([[coupling]]),
            matrix([[1.0]]),
        )
        .unwrap();
        let p = p_vector(
            pole_mass.powi(2),
            &pole_masses,
            vector([production]),
            matrix([[coupling]]),
            matrix([[1.0]]),
        )
        .unwrap();
        let f = f_vector(
            pole_mass.powi(2),
            pole_masses,
            k,
            p,
            matrix([[phase_space]]),
        )
        .unwrap();
        let actual = evaluate(f.component(0));
        let expected = production * coupling / (coupling.powi(2) * phase_space);
        assert!(actual.re.is_finite() && actual.im.is_finite());
        assert_relative_eq!(actual.re, expected.re, epsilon = 1.0e-12);
        assert_relative_eq!(actual.im, expected.im, epsilon = 1.0e-12);
    }
}
