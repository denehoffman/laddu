use pyo3::{PyErr, create_exception, exceptions::PyException};

create_exception!(
    laddu,
    LadduError,
    PyException,
    "Base exception raised by laddu's Python API."
);

pub(crate) fn to_py_err(error: impl std::fmt::Display) -> PyErr {
    LadduError::new_err(error.to_string())
}

#[cfg(feature = "fit")]
create_exception!(
    laddu,
    FitDomainError,
    LadduError,
    "The likelihood is undefined at the fit point."
);

#[cfg(feature = "fit")]
pub(crate) fn to_py_fit_err(error: laddu_fit::FitError) -> PyErr {
    if error.is_objective_domain() {
        FitDomainError::new_err(error.to_string())
    } else {
        to_py_err(error)
    }
}

#[cfg(feature = "fit")]
create_exception!(
    laddu,
    BootstrapFitError,
    LadduError,
    "A bootstrap replica failed to fit."
);

#[cfg(feature = "fit")]
create_exception!(
    laddu,
    ReplicaDidNotConverge,
    LadduError,
    "A bootstrap replica did not converge."
);
