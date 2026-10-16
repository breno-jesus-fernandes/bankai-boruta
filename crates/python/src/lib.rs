//! NumPy boundary; shadow columns are read through row permutation indices.
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn gain_importances(
    x: PyReadonlyArray2<'_, f64>,
    y: PyReadonlyArray1<'_, i64>,
    active: PyReadonlyArray1<'_, i64>,
    sources: PyReadonlyArray1<'_, i64>,
    permutations: PyReadonlyArray2<'_, i64>,
    trees: usize,
    depth: usize,
    seed: u64,
    bins: usize,
    extra_seed: u64,
) -> PyResult<(Vec<f64>, f64, f64)> {
    let preparation_start = std::time::Instant::now();
    let x = x.as_array();
    let active = active.as_array();
    let sources = sources.as_array();
    let permutations = permutations.as_array();
    let labels = y
        .as_array()
        .iter()
        .map(|&v| usize::try_from(v))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| PyValueError::new_err("invalid labels"))?;
    let (rows, columns) = x.dim();
    if rows == 0
        || columns == 0
        || labels.len() != rows
        || trees == 0
        || depth == 0
        || !(2..=255).contains(&bins)
        || active.is_empty()
        || sources.is_empty()
        || active
            .iter()
            .chain(sources.iter())
            .any(|&v| v < 0 || v as usize >= columns)
        || permutations.dim() != (sources.len(), rows)
        || permutations.iter().any(|&v| v < 0 || v as usize >= rows)
        || x.iter().any(|v| !v.is_finite())
    {
        return Err(PyValueError::new_err("invalid training arrays or settings"));
    }
    // a15 canonicalizes dense training rows lexicographically, then by label.
    let mut order: Vec<usize> = (0..rows).collect();
    let value = |row: usize, col: usize| {
        if col < active.len() {
            x[[row, active[col] as usize]]
        } else {
            let s = col - active.len();
            x[[permutations[[s, row]] as usize, sources[s] as usize]]
        }
    };
    let width = active.len() + sources.len();
    order.sort_by(|&a, &b| {
        for col in 0..width {
            let comparison = value(a, col)
                .partial_cmp(&value(b, col))
                .expect("finite values");
            if !comparison.is_eq() {
                return comparison;
            }
        }
        labels[a].cmp(&labels[b])
    });
    let sorted_labels = order.iter().map(|&r| labels[r]).collect();
    let boundary_seconds = preparation_start.elapsed().as_secs_f64();
    let result = boruta_core::gain_importances(
        rows,
        width,
        sorted_labels,
        trees,
        depth,
        seed,
        bins,
        extra_seed,
        |row, col| value(order[row], col),
    )
    .map_err(PyValueError::new_err)?;
    Ok((
        result.importances,
        boundary_seconds + result.preparation_seconds,
        result.training_seconds,
    ))
}

#[pymodule]
fn _boruta_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gain_importances, m)?)?;
    Ok(())
}
