//! NumPy boundary; shadow columns are read through row permutation indices.
use numpy::{
    Element, IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

trait Value: Element + Copy {
    fn value(self) -> f64;
}
impl Value for f32 {
    fn value(self) -> f64 {
        self as f64
    }
}
impl Value for f64 {
    fn value(self) -> f64 {
        self
    }
}
trait Index: Element + Copy {
    fn index(self) -> Option<usize>;
}
impl Index for u32 {
    fn index(self) -> Option<usize> {
        usize::try_from(self).ok()
    }
}
impl Index for i64 {
    fn index(self) -> Option<usize> {
        usize::try_from(self).ok()
    }
}

#[allow(clippy::too_many_arguments)]
fn gain_typed<V: Value, I: Index>(
    x: PyReadonlyArray2<'_, V>,
    y: PyReadonlyArray1<'_, i64>,
    active: PyReadonlyArray1<'_, i64>,
    sources: PyReadonlyArray1<'_, i64>,
    permutations: PyReadonlyArray2<'_, I>,
    trees: usize,
    depth: usize,
    seed: u64,
    bins: usize,
    extra_seed: u64,
    threads: usize,
    max_features: usize,
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
        || threads == 0
        || max_features == 0
        || max_features > active.len() + sources.len()
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
        || permutations
            .iter()
            .any(|&v| v.index().is_none_or(|r| r >= rows))
        || x.iter().any(|v| !v.value().is_finite())
    {
        return Err(PyValueError::new_err("invalid training arrays or settings"));
    }
    // a15 canonicalizes dense training rows lexicographically, then by label.
    let mut order: Vec<usize> = (0..rows).collect();
    // Resolve feature offsets and strides once per column, including strided
    // input views. The hot accessor only indexes one-dimensional views.
    let column_views: Vec<_> = active
        .iter()
        .chain(sources.iter())
        .map(|&col| x.column(col as usize))
        .collect();
    let permutation_rows: Vec<_> = permutations.rows().into_iter().collect();
    let value = |row: usize, col: usize| {
        let source_row = if col < active.len() {
            row
        } else {
            permutation_rows[col - active.len()][row]
                .index()
                .expect("validated index")
        };
        column_views[col][source_row].value()
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
        threads,
        max_features,
        |row, col| value(order[row], col),
    )
    .map_err(PyValueError::new_err)?;
    Ok((
        result.importances,
        boundary_seconds + result.preparation_seconds,
        result.training_seconds,
    ))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn gain_importances(
    x: &Bound<'_, PyAny>,
    y: PyReadonlyArray1<'_, i64>,
    active: PyReadonlyArray1<'_, i64>,
    sources: PyReadonlyArray1<'_, i64>,
    permutations: &Bound<'_, PyAny>,
    trees: usize,
    depth: usize,
    seed: u64,
    bins: usize,
    extra_seed: u64,
    threads: usize,
    max_features: usize,
) -> PyResult<(Vec<f64>, f64, f64)> {
    macro_rules! dispatch {
        ($values:ty, $indices:ty) => {
            if let (Ok(x), Ok(permutations)) = (
                x.extract::<PyReadonlyArray2<'_, $values>>(),
                permutations.extract::<PyReadonlyArray2<'_, $indices>>(),
            ) {
                return gain_typed(
                    x,
                    y,
                    active,
                    sources,
                    permutations,
                    trees,
                    depth,
                    seed,
                    bins,
                    extra_seed,
                    threads,
                    max_features,
                );
            }
        };
    }
    dispatch!(f32, u32);
    dispatch!(f32, i64);
    dispatch!(f64, u32);
    dispatch!(f64, i64);
    Err(PyValueError::new_err(
        "expected float32/float64 values and uint32/int64 permutations",
    ))
}

// NumPy RandomState's MT19937 and random_interval, specialized for a u32
// swap. NumPy's legacy shuffle only specializes pointer-sized swaps; its
// generic memcpy path otherwise dominates preparation with compact indices.
struct Mt19937 {
    words: [u32; 624],
    position: usize,
}
impl Mt19937 {
    fn next(&mut self) -> u32 {
        if self.position == 624 {
            for i in 0..624 {
                let combined =
                    (self.words[i] & 0x8000_0000) | (self.words[(i + 1) % 624] & 0x7fff_ffff);
                self.words[i] = self.words[(i + 397) % 624]
                    ^ (combined >> 1)
                    ^ if combined & 1 != 0 { 0x9908_b0df } else { 0 };
            }
            self.position = 0;
        }
        let mut value = self.words[self.position];
        self.position += 1;
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c_5680;
        value ^= (value << 15) & 0xefc6_0000;
        value ^ (value >> 18)
    }
}

#[pyfunction]
fn shuffle_uint32<'py>(
    py: Python<'py>,
    mut values: PyReadwriteArray1<'py, u32>,
    state: PyReadonlyArray1<'py, u32>,
    position: usize,
) -> PyResult<(Bound<'py, PyArray1<u32>>, usize)> {
    let words = state
        .as_slice()
        .map_err(|_| PyValueError::new_err("MT19937 state must be contiguous"))?;
    if words.len() != 624 || position > 624 || values.as_array().len() as u128 > (1u128 << 32) {
        return Err(PyValueError::new_err(
            "invalid MT19937 state or permutation size",
        ));
    }
    let mut rng = Mt19937 {
        words: words.try_into().expect("validated state"),
        position,
    };
    let values = values
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("permutation must be contiguous"))?;
    // Identical descending Fisher-Yates draws, including rejection sampling.
    for i in (1..values.len()).rev() {
        let maximum = i as u32;
        let mask = u32::MAX >> maximum.leading_zeros();
        let j = loop {
            let candidate = rng.next() & mask;
            if candidate <= maximum {
                break candidate as usize;
            }
        };
        values.swap(i, j);
    }
    Ok((rng.words.to_vec().into_pyarray(py), rng.position))
}

#[pymodule]
fn _boruta_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gain_importances, m)?)?;
    m.add_function(wrap_pyfunction!(shuffle_uint32, m)?)?;
    Ok(())
}
