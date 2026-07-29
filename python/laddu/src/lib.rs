//! Primary Python extension module with automatic MPI-backend discovery.

laddu_core::laddu_python_module!(
    laddu,
    "local",
    #[pymodule_init]
    fn select_mpi_backend(module: &Bound<'_, PyModule>) -> PyResult<()> {
        let py = module.py();
        match py.import("_laddu_mpi") {
            Ok(mpi) => {
                for (name, value) in mpi.dict().iter() {
                    let name: &str = name.extract()?;
                    if !matches!(name, "__name__" | "__package__" | "__loader__" | "__spec__") {
                        module.setattr(name, value)?;
                    }
                }
                Ok(())
            }
            Err(error) if error.is_instance_of::<pyo3::exceptions::PyModuleNotFoundError>(py) => {
                Ok(())
            }
            Err(error) => Err(error),
        }
    }
);
