//! PyO3 extension for bankai-boruta.

use boruta_core::{
    BorutaConfig, ImportanceScale, MatrixView, MaxFeatures, PermutationImportanceProvider,
    PruningMode, run_boruta,
};
use numpy::{Element, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

fn validate_borrowed<T>(array: &PyReadonlyArray2<'_, T>) -> PyResult<(usize, usize, usize, usize)>
where
    T: Element + Copy + Into<f64>,
{
    if !array.data().cast_const().is_aligned() {
        return Err(PyValueError::new_err(
            "X data must be aligned for its NumPy dtype",
        ));
    }
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
    if view.iter().any(|value| {
        let value: f64 = (*value).into();
        !value.is_finite()
    }) {
        return Err(PyValueError::new_err("X must contain only finite values"));
    }
    Ok((rows, columns, row_stride, column_stride))
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
                let (rows, columns, _, _) = validate_borrowed(&array)?;
                Ok((rows, columns, 0))
            } else if let Ok(array) = x.extract::<PyReadonlyArray2<'_, f32>>() {
                let (rows, columns, _, _) = validate_borrowed(&array)?;
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
            let (rows, columns, _, _) = validate_borrowed(&array)?;
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

#[allow(clippy::too_many_arguments)]
#[pyfunction]
fn fit_boruta<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    y: PyReadonlyArray1<'py, i64>,
    n_classes: usize,
    n_estimators: usize,
    max_iter: usize,
    alpha: f64,
    max_features: &Bound<'py, PyAny>,
    pruning: &str,
    max_depth: Option<usize>,
    n_jobs: usize,
    random_state: Option<u64>,
    importance_scale: &str,
    store_history: bool,
    input_mode: &str,
) -> PyResult<Py<PyDict>> {
    let labels = y
        .as_array()
        .iter()
        .map(|&label| {
            usize::try_from(label)
                .map_err(|_| PyValueError::new_err("encoded y must contain non-negative classes"))
        })
        .collect::<PyResult<Vec<_>>>()?;
    let max_features = parse_max_features(max_features)?;
    let importance_scale = match importance_scale {
        "raw" => ImportanceScale::Raw,
        "zscore" => ImportanceScale::ZScore,
        _ => {
            return Err(PyValueError::new_err(
                "importance_scale must be 'raw' or 'zscore'",
            ));
        }
    };
    let pruning = match pruning {
        "canonical" => PruningMode::Canonical,
        "aggressive" => PruningMode::Aggressive,
        _ => return Err(PyValueError::new_err("invalid pruning mode")),
    };
    let config = BorutaConfig {
        n_estimators,
        max_iter,
        alpha,
        pruning,
        max_features,
        max_depth,
        n_jobs,
        random_state: random_state.unwrap_or_else(random_seed),
        importance_scale,
        store_history,
    };

    let (result, copied_bytes) = match input_mode {
        "zero_copy" => {
            if let Ok(array) = x.extract::<PyReadonlyArray2<'py, f64>>() {
                (fit_array(array, labels, n_classes, &config)?, 0)
            } else if let Ok(array) = x.extract::<PyReadonlyArray2<'py, f32>>() {
                (fit_array(array, labels, n_classes, &config)?, 0)
            } else {
                return Err(PyValueError::new_err(
                    "zero_copy requires a two-dimensional float32 or float64 NumPy array; use input_mode='copy'",
                ));
            }
        }
        "copy" => {
            let numpy = py.import("numpy")?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("dtype", "float64")?;
            kwargs.set_item("order", "C")?;
            kwargs.set_item("copy", true)?;
            let copied = numpy.call_method("array", (x,), Some(&kwargs))?;
            let array = copied
                .extract::<PyReadonlyArray2<'py, f64>>()
                .map_err(|_| {
                    PyValueError::new_err("copy mode requires a two-dimensional numeric array")
                })?;
            let (rows, columns, _, _) = validate_borrowed(&array)?;
            let bytes = rows
                .checked_mul(columns)
                .and_then(|elements| elements.checked_mul(std::mem::size_of::<f64>()))
                .ok_or_else(|| PyValueError::new_err("X is too large"))?;
            (fit_array(array, labels, n_classes, &config)?, bytes)
        }
        _ => {
            return Err(PyValueError::new_err(
                "input_mode must be 'zero_copy' or 'copy'",
            ));
        }
    };
    result_to_python(py, result, copied_bytes)
}

fn fit_array<T>(
    array: PyReadonlyArray2<'_, T>,
    labels: Vec<usize>,
    n_classes: usize,
    config: &BorutaConfig,
) -> PyResult<boruta_core::BorutaResult>
where
    T: Element + Copy + Into<f64> + Send + Sync,
{
    let view = array.as_array();
    let (rows, columns, row_stride, column_stride) = validate_borrowed(&array)?;
    let values = view.as_slice_memory_order().ok_or_else(|| {
        PyValueError::new_err(
            "zero_copy requires a C-contiguous or Fortran-contiguous float32 or float64 NumPy array; use input_mode='copy'",
        )
    })?;
    let matrix = MatrixView::new(values, rows, columns, row_stride, column_stride)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    debug_assert!(matrix.shares_storage_with(values));
    let mut provider = PermutationImportanceProvider;
    run_boruta(matrix, labels, n_classes, config, &mut provider)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

fn parse_max_features(value: &Bound<'_, PyAny>) -> PyResult<MaxFeatures> {
    if value.is_none() {
        return Ok(MaxFeatures::All);
    }
    if let Ok(name) = value.extract::<String>() {
        return match name.as_str() {
            "sqrt" => Ok(MaxFeatures::Sqrt),
            "log2" => Ok(MaxFeatures::Log2),
            _ => Err(PyValueError::new_err("invalid max_features")),
        };
    }
    if let Ok(count) = value.extract::<usize>() {
        return Ok(MaxFeatures::Count(count));
    }
    if let Ok(fraction) = value.extract::<f64>() {
        return Ok(MaxFeatures::Fraction(fraction));
    }
    Err(PyValueError::new_err("invalid max_features"))
}

fn random_seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos() as u64)
}

fn result_to_python(
    py: Python<'_>,
    result: boruta_core::BorutaResult,
    copied_bytes: usize,
) -> PyResult<Py<PyDict>> {
    use boruta_core::{FeatureDecision, StopReason};
    let output = PyDict::new(py);
    let decisions = result
        .decisions
        .iter()
        .map(|decision| match decision {
            FeatureDecision::Rejected => 0_usize,
            FeatureDecision::Confirmed => 1_usize,
            FeatureDecision::Tentative => 2_usize,
        })
        .collect::<Vec<_>>();
    output.set_item("decisions", decisions)?;
    output.set_item("n_iter", result.n_iter)?;
    output.set_item(
        "stop_reason",
        match result.stop_reason {
            StopReason::NoTentativeFeatures => "no_tentative_features",
            StopReason::MaxIterations => "max_iter",
        },
    )?;
    let diagnostics = PyDict::new(py);
    diagnostics.set_item("copied_bytes", copied_bytes)?;
    diagnostics.set_item(
        "undefined_importance_count",
        result.undefined_importance_count,
    )?;
    diagnostics.set_item(
        "non_finite_importance_count",
        result.non_finite_importance_count,
    )?;
    if let Some(active_counts) = result.active_feature_counts {
        diagnostics.set_item("active_features_per_iteration", active_counts)?;
    }
    output.set_item("diagnostics", diagnostics)?;
    if let Some(history) = result.hit_history {
        output.set_item("history", history)?;
    }
    Ok(output.unbind())
}

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("ENGINE_READY", true)?;
    module.add_function(wrap_pyfunction!(inspect_matrix, module)?)?;
    module.add_function(wrap_pyfunction!(fit_boruta, module)?)?;
    Ok(())
}
