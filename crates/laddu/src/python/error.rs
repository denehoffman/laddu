use pyo3::{PyErr, create_exception, exceptions::PyException};

create_exception!(
    laddu,
    LadduError,
    PyException,
    "Base exception raised by Laddu's Python API."
);

pub(crate) fn to_py_err(error: impl std::fmt::Display) -> PyErr {
    LadduError::new_err(error.to_string())
}
