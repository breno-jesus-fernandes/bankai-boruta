//! PyO3 extension for bankai-boruta.

use boruta_core::MatrixView;
use numpy::Element;
use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

fn validate_borrowed<T>(array: PyReadonlyArray2<'_, T>) -> PyResult<(usize, usize)>
where
    T: Element + Copy + Into<f64>,
{
    let view = array.as_array();
    let rows = view.nrows();
    let columns = view.ncols();
    if rows == 0 || columns == 0 {
        return Err(PyValueError::new_err(
            "X must have at least one row and feature",
        ));
    }
    let strides = view.strides();
    let row_stride = usize::try_from(strides[0])
        .map_err(|_| PyValueError::new_err("X must have positive element strides"))?;
    let column_stride = usize::try_from(strides[1])
        .map_err(|_| PyValueError::new_err("X must have positive element strides"))?;
    let c_contiguous = column_stride == 1 && (rows == 1 || row_stride == columns);
    let f_contiguous = row_stride == 1 && (columns == 1 || column_stride == rows);
    if !c_contiguous && !f_contiguous {
        return Err(PyValueError::new_err(
            "zero_copy requires a C-contiguous or Fortran-contiguous float32 or float64 NumPy array; use input_mode='copy'",
        ));
    }
    let values = view.as_slice_memory_order().ok_or_else(|| {
        PyValueError::new_err(
            "zero_copy requires a C-contiguous or Fortran-contiguous float32 or float64 NumPy array; use input_mode='copy'",
        )
    })?;
    let matrix = MatrixView::new(values, rows, columns, row_stride, column_stride)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    debug_assert!(matrix.shares_storage_with(values));
    Ok(matrix.shape())
}

#[pyfunction(signature = (x, input_mode="zero_copy"))]
fn inspect_matrix(
    py: Python<'_>,
    x: &Bound<'_, PyAny>,
    input_mode: &str,
) -> PyResult<(usize, usize, usize)> {
    match input_mode {
        "zero_copy" => {
            if let Ok(array) = x.extract::<PyReadonlyArray2<'_, f64>>() {
                let (rows, columns) = validate_borrowed(array)?;
                Ok((rows, columns, 0))
            } else if let Ok(array) = x.extract::<PyReadonlyArray2<'_, f32>>() {
                let (rows, columns) = validate_borrowed(array)?;
                Ok((rows, columns, 0))
            } else {
                Err(PyValueError::new_err(
                    "zero_copy requires a two-dimensional float32 or float64 NumPy array; use input_mode='copy'",
                ))
            }
        }
        "copy" => {
            let numpy = py.import("numpy")?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("dtype", "float64")?;
            kwargs.set_item("order", "C")?;
            kwargs.set_item("copy", true)?;
            let copied = numpy.call_method("array", (x,), Some(&kwargs))?;
            let array = copied.extract::<PyReadonlyArray2<'_, f64>>().map_err(|_| {
                PyValueError::new_err("copy mode requires a two-dimensional numeric array")
            })?;
            let (rows, columns) = validate_borrowed(array)?;
            let bytes = rows
                .checked_mul(columns)
                .and_then(|elements| elements.checked_mul(std::mem::size_of::<f64>()))
                .ok_or_else(|| PyValueError::new_err("X is too large"))?;
            Ok((rows, columns, bytes))
        }
        _ => Err(PyValueError::new_err(
            "input_mode must be 'zero_copy' or 'copy'",
        )),
    }
}

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("ENGINE_READY", false)?;
    module.add_function(wrap_pyfunction!(inspect_matrix, module)?)?;
    Ok(())
}
