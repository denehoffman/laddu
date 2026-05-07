use crate::{vectors::Vec3, LadduError, LadduResult};

pub(super) fn checked_boost_vector(beta: Vec3, context: &str) -> LadduResult<Vec3> {
    let beta2 = beta.mag2();
    if !beta2.is_finite() || beta2 >= 1.0 {
        return Err(LadduError::Custom(format!(
            "{context} boost must have |beta| < 1"
        )));
    }
    Ok(beta)
}

pub(super) fn unit_vector(vector: Vec3, name: &str) -> LadduResult<Vec3> {
    let mag2 = vector.mag2();
    if !mag2.is_finite() || mag2 <= f64::EPSILON * f64::EPSILON {
        return Err(LadduError::Custom(format!("{name} must be non-zero")));
    }
    Ok(vector.unit())
}
