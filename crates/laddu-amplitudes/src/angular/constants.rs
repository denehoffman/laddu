use laddu_core::{
    amplitude::{Expression, IntoTags},
    math::wigner_3j,
    LadduResult, J, M,
};

/// A Wigner-3j symbol expression.
pub struct Wigner3j;

impl Wigner3j {
    /// Construct a new constant expression for a Wigner-3j symbol.
    pub fn new(
        tags: impl IntoTags,
        j1: J,
        m1: M,
        j2: J,
        m2: M,
        j3: J,
        m3: M,
    ) -> LadduResult<Expression> {
        let value = wigner_3j(
            j1.value() as u64,
            j2.value() as u64,
            j3.value() as u64,
            m1.value() as i64,
            m2.value() as i64,
            m3.value() as i64,
        );
        let _ = tags.into_tags();
        Ok(value.into())
    }
}
