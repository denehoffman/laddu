use pyo3::prelude::*;
use pyo3::types::PyDict;

pub mod amplitudes;
pub mod data;
pub mod utils;

pub trait GetStrExtractObj {
    fn get_extract<T>(&self, key: &str) -> PyResult<Option<T>>
    where
        T: for<'py> FromPyObject<'py>;
}

impl GetStrExtractObj for Bound<'_, PyDict> {
    fn get_extract<T>(&self, key: &str) -> PyResult<Option<T>>
    where
        T: for<'py> FromPyObject<'py>,
    {
        self.get_item(key)?
            .map(|value| value.extract::<T>())
            .transpose()
    }
}
