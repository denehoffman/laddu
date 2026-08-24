//! Binding vocabulary shared by shader declarations, layouts, and dispatch.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub(crate) enum Binding {
    Parameters = 0,
    Cache = 1,
    Output = 2,
    Weights = 3,
    Config = 4,
    Partials = 5,
    ReductionError = 6,
    SolveError = 7,
}

impl Binding {
    pub(crate) const fn index(self) -> u32 {
        self as u32
    }
}
