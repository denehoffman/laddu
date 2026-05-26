use laddu_core::{
    amplitude::{Expression, IntoTags},
    math::{clebsch_gordon, wigner_3j},
    LadduResult, J, M,
};

/// A Clebsch-Gordan coefficient expression.
pub struct ClebschGordan;

impl ClebschGordan {
    /// Construct a new constant expression for a Clebsch-Gordan coefficient.
    pub fn new(
        tags: impl IntoTags,
        j1: J,
        m1: M,
        j2: J,
        m2: M,
        j: J,
        m: M,
    ) -> LadduResult<Expression> {
        let value = clebsch_gordon(
            j1.value() as u64,
            j2.value() as u64,
            j.value() as u64,
            m1.value() as i64,
            m2.value() as i64,
            m.value() as i64,
        );
        let _ = tags.into_tags();
        Ok(value.into())
    }
}

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
