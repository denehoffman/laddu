use pyo3::prelude::*;

#[pyfunction]
fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
