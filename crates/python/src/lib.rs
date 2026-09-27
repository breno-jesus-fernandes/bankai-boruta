//! PyO3 extension for bankai-boruta.

use pyo3::prelude::*;

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("ENGINE_READY", false)?;
    Ok(())
}
