use serde::{Deserialize, Serialize};
use serde_with::serde_as;

/// A tag for retrieving or storing a scalar value in a [`Cache`](crate::resources::Cache).
#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize)]
pub struct ScalarID(pub(crate) usize);

/// A tag for retrieving or storing a complex scalar value in a [`Cache`](crate::resources::Cache).
#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize)]
pub struct ComplexScalarID(pub(crate) usize, pub(crate) usize);

/// A tag for retrieving or storing a vector in a [`Cache`](crate::resources::Cache).
#[serde_as]
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct VectorID<const R: usize>(#[serde_as(as = "[_; R]")] pub(crate) [usize; R]);

impl<const R: usize> Default for VectorID<R> {
    fn default() -> Self {
        Self([0; R])
    }
}

/// A tag for retrieving or storing a complex-valued vector in a [`Cache`](crate::resources::Cache).
#[serde_as]
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct ComplexVectorID<const R: usize>(
    #[serde_as(as = "[_; R]")] pub(crate) [usize; R],
    #[serde_as(as = "[_; R]")] pub(crate) [usize; R],
);

impl<const R: usize> Default for ComplexVectorID<R> {
    fn default() -> Self {
        Self([0; R], [0; R])
    }
}

/// A tag for retrieving or storing a matrix in a [`Cache`](crate::resources::Cache).
#[serde_as]
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct MatrixID<const R: usize, const C: usize>(
    #[serde_as(as = "[[_; C]; R]")] pub(crate) [[usize; C]; R],
);

impl<const R: usize, const C: usize> Default for MatrixID<R, C> {
    fn default() -> Self {
        Self([[0; C]; R])
    }
}

/// A tag for retrieving or storing a complex-valued matrix in a [`Cache`](crate::resources::Cache).
#[serde_as]
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct ComplexMatrixID<const R: usize, const C: usize>(
    #[serde_as(as = "[[_; C]; R]")] pub(crate) [[usize; C]; R],
    #[serde_as(as = "[[_; C]; R]")] pub(crate) [[usize; C]; R],
);

impl<const R: usize, const C: usize> Default for ComplexMatrixID<R, C> {
    fn default() -> Self {
        Self([[0; C]; R], [[0; C]; R])
    }
}
