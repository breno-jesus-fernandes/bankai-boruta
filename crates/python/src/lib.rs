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
    bins: Option<usize>,
    extra_seed: u64,
    threads: usize,
    max_features: usize,
    backend: &str,
    criterion: &str,
    min_samples_split: usize,
    min_samples_leaf: usize,
    min_leaf_weight: f64,
    min_impurity_decrease: f64,
    max_leaves: Option<usize>,
    bootstrap: bool,
    sample_size: Option<usize>,
    binning_strategy: &str,
    bin_sample_size: usize,
    binning_seed: u64,
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
        || bins.is_some_and(|value| !(2..=255).contains(&value))
        || backend != if bins.is_some() { "binned" } else { "exact" }
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
        criterion,
        min_samples_split,
        min_samples_leaf,
        min_leaf_weight,
        min_impurity_decrease,
        max_leaves.unwrap_or(usize::MAX),
        bootstrap,
        sample_size,
        binning_strategy,
        bin_sample_size,
        binning_seed,
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
#[pyo3(signature = (x, y, active, sources, permutations, trees, depth, seed, bins, extra_seed, threads, max_features, backend="binned", criterion="gini", min_samples_split=2, min_samples_leaf=1, min_leaf_weight=0.0, min_impurity_decrease=0.0, max_leaves=None, bootstrap=true, sample_size=None, binning_strategy="exact_sort", bin_sample_size=200000, binning_seed=6))]
fn gain_importances(
    x: &Bound<'_, PyAny>,
    y: PyReadonlyArray1<'_, i64>,
    active: PyReadonlyArray1<'_, i64>,
    sources: PyReadonlyArray1<'_, i64>,
    permutations: &Bound<'_, PyAny>,
    trees: usize,
    depth: usize,
    seed: u64,
    bins: Option<usize>,
    extra_seed: u64,
    threads: usize,
    max_features: usize,
    backend: &str,
    criterion: &str,
    min_samples_split: usize,
    min_samples_leaf: usize,
    min_leaf_weight: f64,
    min_impurity_decrease: f64,
    max_leaves: Option<usize>,
    bootstrap: bool,
    sample_size: Option<usize>,
    binning_strategy: &str,
    bin_sample_size: usize,
    binning_seed: u64,
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
                    backend,
                    criterion,
                    min_samples_split,
                    min_samples_leaf,
                    min_leaf_weight,
                    min_impurity_decrease,
                    max_leaves,
                    bootstrap,
                    sample_size,
                    binning_strategy,
                    bin_sample_size,
                    binning_seed,
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

// Owned snapshots prevent mutation of caller arrays from changing a running fit.
#[pyclass]
struct ExactSession {
    bank: boruta_core::ExactColumnBank,
    labels: Vec<usize>,
    active: Vec<usize>,
    real_order: Vec<usize>,
    ties: Vec<(usize, usize)>,
    order: std::sync::Arc<Vec<usize>>,
    permutations: std::sync::Arc<boruta_core::ExactRowIndices>,
    cache_rebuilds: usize,
}

#[pymethods]
impl ExactSession {
    #[new]
    fn new(x: &Bound<'_, PyAny>, y: PyReadonlyArray1<'_, i64>) -> PyResult<Self> {
        let bank = if let Ok(x) = x.extract::<PyReadonlyArray2<'_, f32>>() {
            let x = x.as_array();
            boruta_core::ExactColumnBank::from_f32(
                x.columns().into_iter().flat_map(|c| c.to_vec()).collect(),
                x.nrows(),
                x.ncols(),
            )
        } else if let Ok(x) = x.extract::<PyReadonlyArray2<'_, f64>>() {
            let x = x.as_array();
            boruta_core::ExactColumnBank::from_f64(
                x.columns().into_iter().flat_map(|c| c.to_vec()).collect(),
                x.nrows(),
                x.ncols(),
            )
        } else {
            return Err(PyValueError::new_err("expected float32/float64 values"));
        }
        .map_err(PyValueError::new_err)?;
        let labels = y
            .as_array()
            .iter()
            .map(|&v| usize::try_from(v))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| PyValueError::new_err("invalid labels"))?;
        if labels.len() != bank.rows()
            || labels
                .iter()
                .max()
                .is_none_or(|&v| v == 0 || v >= bank.rows())
        {
            return Err(PyValueError::new_err("invalid labels"));
        }
        Ok(Self {
            bank,
            labels,
            active: Vec::new(),
            real_order: Vec::new(),
            ties: Vec::new(),
            order: std::sync::Arc::new(Vec::new()),
            permutations: std::sync::Arc::new(boruta_core::ExactRowIndices::U32(Vec::new())),
            cache_rebuilds: 0,
        })
    }

    /// Fill explicit independent permutations in the owned reusable buffer.
    /// Same descending Fisher-Yates and rejection draws as NumPy RandomState.
    fn shuffle_uint32<'py>(
        &mut self,
        py: Python<'py>,
        state: PyReadonlyArray1<'py, u32>,
        position: usize,
        shadows: usize,
    ) -> PyResult<(Bound<'py, PyArray1<u32>>, usize)> {
        let words = state
            .as_slice()
            .map_err(|_| PyValueError::new_err("MT19937 state must be contiguous"))?;
        let rows = self.bank.rows();
        if words.len() != 624 || position > 624 || shadows == 0 || rows as u128 > (1u128 << 32) {
            return Err(PyValueError::new_err(
                "invalid MT19937 state or permutation size",
            ));
        }
        let cells = rows
            .checked_mul(shadows)
            .filter(|&n| n <= isize::MAX as usize / 4)
            .ok_or_else(|| {
                PyValueError::new_err("permutation shape exceeds platform addressing limits")
            })?;
        let mut rng = Mt19937 {
            words: words.try_into().expect("validated state"),
            position,
        };
        let dst = std::sync::Arc::make_mut(&mut self.permutations);
        if !matches!(dst, boruta_core::ExactRowIndices::U32(_)) {
            *dst = boruta_core::ExactRowIndices::U32(Vec::new());
        }
        if let boruta_core::ExactRowIndices::U32(values) = dst {
            values.resize(cells, 0);
            for permutation in values.chunks_mut(rows) {
                for (i, value) in permutation.iter_mut().enumerate() {
                    *value = i as u32;
                }
                rng.shuffle(permutation);
            }
        }
        Ok((rng.words.to_vec().into_pyarray(py), rng.position))
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (active, sources, permutations, trees, depth, seed, bins, extra_seed, threads, max_features, backend="exact", criterion="gini", min_samples_split=2, min_samples_leaf=1, min_leaf_weight=0.0, min_impurity_decrease=0.0, max_leaves=None, bootstrap=true, sample_size=None, binning_strategy="exact_sort", bin_sample_size=200000, binning_seed=6))]
    fn gain_importances(
        &mut self,
        active: PyReadonlyArray1<'_, i64>,
        sources: PyReadonlyArray1<'_, i64>,
        permutations: &Bound<'_, PyAny>,
        trees: usize,
        depth: usize,
        seed: u64,
        bins: Option<usize>,
        extra_seed: u64,
        threads: usize,
        max_features: usize,
        backend: &str,
        criterion: &str,
        min_samples_split: usize,
        min_samples_leaf: usize,
        min_leaf_weight: f64,
        min_impurity_decrease: f64,
        max_leaves: Option<usize>,
        bootstrap: bool,
        sample_size: Option<usize>,
        binning_strategy: &str,
        bin_sample_size: usize,
        binning_seed: u64,
    ) -> PyResult<(Vec<f64>, f64, f64, std::collections::HashMap<String, f64>)> {
        use boruta_core::{ExactRowIndices, ExactVirtualColumns};
        use std::time::Instant;
        let start = Instant::now();
        let rows = self.bank.rows();
        let map = |v: PyReadonlyArray1<'_, i64>| -> PyResult<Vec<usize>> {
            v.as_array()
                .iter()
                .map(|&v| {
                    usize::try_from(v)
                        .ok()
                        .filter(|&c| c < self.bank.columns())
                        .ok_or_else(|| PyValueError::new_err("invalid feature map"))
                })
                .collect()
        };
        let active = map(active)?;
        let sources = map(sources)?;
        if active.is_empty() || sources.is_empty() || bins.is_some() || backend != "exact" {
            return Err(PyValueError::new_err(
                "exact session requires nonempty maps and exact backend",
            ));
        }
        // Preserve compact index width and reuse the owned round buffer.
        let copied_bytes;
        if permutations.is_none() {
            if rows.checked_mul(sources.len()) != Some(self.permutations.len()) {
                return Err(PyValueError::new_err(
                    "shuffle buffer shape does not match source map",
                ));
            }
            copied_bytes = 0;
        } else if let Ok(p) = permutations.extract::<PyReadonlyArray2<'_, u32>>() {
            let p = p.as_array();
            if p.dim() != (sources.len(), rows) {
                return Err(PyValueError::new_err("invalid permutation shape"));
            }
            let dst = std::sync::Arc::make_mut(&mut self.permutations);
            if !matches!(dst, ExactRowIndices::U32(_)) {
                *dst = ExactRowIndices::U32(Vec::new());
            }
            if let ExactRowIndices::U32(v) = dst {
                v.clear();
                v.extend(p.iter().copied());
            }
            copied_bytes = self.permutations.bytes();
        } else if let Ok(p) = permutations.extract::<PyReadonlyArray2<'_, i64>>() {
            let p = p.as_array();
            if p.dim() != (sources.len(), rows) {
                return Err(PyValueError::new_err("invalid permutation shape"));
            }
            let dst = std::sync::Arc::make_mut(&mut self.permutations);
            if !matches!(dst, ExactRowIndices::Usize(_)) {
                *dst = ExactRowIndices::Usize(Vec::new());
            }
            if let ExactRowIndices::Usize(v) = dst {
                v.clear();
                for &r in p.iter() {
                    v.push(
                        usize::try_from(r)
                            .map_err(|_| PyValueError::new_err("invalid permutation index"))?,
                    );
                }
            }
            copied_bytes = self.permutations.bytes();
        } else {
            return Err(PyValueError::new_err("expected uint32/int64 permutations"));
        }
        if (0..self.permutations.len()).any(|i| self.permutations.at(i) >= rows) {
            return Err(PyValueError::new_err("invalid permutation index"));
        }
        let validation_seconds = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let real_cmp = |a: usize, b: usize| {
            for &c in &active {
                let cmp = self
                    .bank
                    .value_at(a, c)
                    .partial_cmp(&self.bank.value_at(b, c))
                    .expect("finite bank");
                if !cmp.is_eq() {
                    return cmp;
                }
            }
            std::cmp::Ordering::Equal
        };
        if self.active != active {
            self.real_order.clear();
            self.real_order.extend(0..rows);
            self.real_order.sort_by(|&a, &b| real_cmp(a, b));
            self.ties.clear();
            let mut first = 0;
            for end in 1..=rows {
                if end == rows || !real_cmp(self.real_order[first], self.real_order[end]).is_eq() {
                    if end - first > 1 {
                        self.ties.push((first, end));
                    }
                    first = end;
                }
            }
            self.active.clone_from(&active);
            self.cache_rebuilds += 1;
        }
        let order = std::sync::Arc::make_mut(&mut self.order);
        order.clone_from(&self.real_order);
        for &(first, end) in &self.ties {
            order[first..end].sort_by(|&a, &b| {
                for (shadow, &source) in sources.iter().enumerate() {
                    let va = self
                        .bank
                        .value_at(self.permutations.at(shadow * rows + a), source);
                    let vb = self
                        .bank
                        .value_at(self.permutations.at(shadow * rows + b), source);
                    let cmp = va.partial_cmp(&vb).expect("finite bank");
                    if !cmp.is_eq() {
                        return cmp;
                    }
                }
                self.labels[a].cmp(&self.labels[b])
            });
        }
        let labels = order.iter().map(|&r| self.labels[r]).collect();
        let sorting_seconds = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let storage = ExactVirtualColumns::new(
            self.bank.clone(),
            active.clone(),
            sources.clone(),
            self.order.clone(),
            self.permutations.clone(),
        )
        .map_err(PyValueError::new_err)?;
        let storage_seconds = start.elapsed().as_secs_f64();
        let result = boruta_core::gain_importances_with_storage(
            rows,
            active.len() + sources.len(),
            labels,
            trees,
            depth,
            seed,
            None,
            extra_seed,
            threads,
            max_features,
            criterion,
            min_samples_split,
            min_samples_leaf,
            min_leaf_weight,
            min_impurity_decrease,
            max_leaves.unwrap_or(usize::MAX),
            bootstrap,
            sample_size,
            binning_strategy,
            bin_sample_size,
            binning_seed,
            |_, _| unreachable!("virtual storage never materializes"),
            Some(storage),
        )
        .map_err(PyValueError::new_err)?;
        let metrics = [
            ("validation_seconds", validation_seconds + storage_seconds),
            ("sorting_seconds", sorting_seconds),
            ("storage_seconds", result.preparation_seconds),
            ("real_bank_bytes", self.bank.bytes() as f64),
            (
                "native_permutation_bytes",
                self.permutations.allocated_bytes() as f64,
            ),
            ("native_permutation_copied_bytes", copied_bytes as f64),
            (
                "order_cache_bytes",
                (self.real_order.capacity() + self.order.capacity()) as f64
                    * std::mem::size_of::<usize>() as f64
                    + self.ties.capacity() as f64 * 2. * std::mem::size_of::<usize>() as f64,
            ),
            ("sort_cache_rebuilds", self.cache_rebuilds as f64),
            ("materialized_bytes", 0.),
            (
                "feature_workspace_peak_bytes_per_worker",
                result.feature_workspace_peak_bytes_per_worker,
            ),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v))
        .collect();
        Ok((
            result.importances,
            validation_seconds + sorting_seconds + storage_seconds + result.preparation_seconds,
            result.training_seconds,
            metrics,
        ))
    }
}

// NumPy RandomState's MT19937 and random_interval, specialized for a u32
// swap. NumPy's legacy shuffle only specializes pointer-sized swaps; its
// generic memcpy path otherwise dominates preparation with compact indices.
struct Mt19937 {
    words: [u32; 624],
    position: usize,
}
impl Mt19937 {
    fn shuffle(&mut self, values: &mut [u32]) {
        // Identical descending Fisher-Yates draws, including rejection sampling.
        for i in (1..values.len()).rev() {
            let maximum = i as u32;
            let mask = u32::MAX >> maximum.leading_zeros();
            let j = loop {
                let candidate = self.next() & mask;
                if candidate <= maximum {
                    break candidate as usize;
                }
            };
            values.swap(i, j);
        }
    }
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
    rng.shuffle(values);
    Ok((rng.words.to_vec().into_pyarray(py), rng.position))
}

/// One fit's float32 bank in canonical row order plus a reusable shadow
/// buffer. Shadows are shuffled values, so no permutation maps are stored.
#[pyclass]
struct ShadowSession {
    rows: usize,
    columns: usize,
    bank: std::sync::Arc<Vec<f32>>,
    labels: Vec<usize>,
    order: Vec<usize>,
    shadows: std::sync::Arc<Vec<f32>>,
    binned: Option<BinnedBank>,
}

/// Real columns binned once per fit; the float bank is then released.
struct BinnedBank {
    codes: Vec<u8>,
    edges: Vec<Vec<f64>>,
    max_bins: usize,
}

#[pymethods]
impl ShadowSession {
    #[new]
    #[pyo3(signature = (x, y, threads, bins=None, binning_strategy="exact_sort", bin_sample_size=200000, binning_seed=6))]
    fn new(x: PyReadonlyArray2<'_, f32>, y: PyReadonlyArray1<'_, i64>, threads: usize, bins: Option<usize>,
           binning_strategy: &str, bin_sample_size: usize, binning_seed: u64) -> PyResult<Self> {
        let x = x.as_array();
        let (rows, columns) = x.dim();
        let labels = y
            .as_array()
            .iter()
            .map(|&v| usize::try_from(v))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| PyValueError::new_err("invalid labels"))?;
        if rows == 0 || columns == 0 || threads == 0 || labels.len() != rows
            || labels.iter().max().is_none_or(|&v| v == 0 || v >= rows)
            || x.iter().any(|v| !v.is_finite())
        {
            return Err(PyValueError::new_err("invalid training arrays or settings"));
        }
        let order = boruta_core::canonical_row_order(rows, columns, &labels, |r, c| x[[r, c]]);
        let labels = order.iter().map(|&r| labels[r]).collect();
        let (bank, binned) = match bins {
            None => (boruta_core::real_bank(rows, columns, &order, threads, |r, c| x[[r, c]]), None),
            Some(max_bins) => {
                // Bin straight from the input in canonical order; no float bank.
                let strategy = boruta_core::BinningStrategy::parse(binning_strategy).map_err(PyValueError::new_err)?;
                let (codes, edges) = boruta_core::DenseInput::bin_columns(rows, columns, max_bins, threads, strategy,
                    bin_sample_size, binning_seed, |r, c| f64::from(x[[order[r], c]])).map_err(PyValueError::new_err)?;
                (Vec::new(), Some(BinnedBank { codes, edges, max_bins }))
            }
        };
        Ok(Self { rows, columns, bank: std::sync::Arc::new(bank), labels, order, shadows: std::sync::Arc::new(Vec::new()), binned })
    }

    #[getter]
    fn bank_bytes(&self) -> usize { self.binned.as_ref().map_or(self.bank.len() * 4, |b| b.codes.len()) }

    fn _canonical_order(&self) -> Vec<usize> { self.order.clone() }

    /// Training-order real columns and last round's shadows, for parity tests.
    fn _training_matrix<'py>(&self, py: Python<'py>, active: Vec<usize>, shadows: usize) -> PyResult<Bound<'py, numpy::PyArray2<f32>>> {
        if self.binned.is_some() {
            return Err(PyValueError::new_err("binned sessions keep codes only"));
        }
        let rows = self.rows;
        let mut out = numpy::ndarray::Array2::<f32>::zeros((rows, active.len() + shadows));
        for (j, &c) in active.iter().enumerate() {
            out.column_mut(j).assign(&numpy::ndarray::ArrayView1::from(&self.bank[c * rows..(c + 1) * rows]));
        }
        for j in 0..shadows {
            out.column_mut(active.len() + j).assign(&numpy::ndarray::ArrayView1::from(&self.shadows[j * rows..(j + 1) * rows]));
        }
        Ok(numpy::PyArray2::from_owned_array(py, out))
    }

    fn _training_labels(&self) -> Vec<usize> { self.labels.clone() }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (active, sources, shadow_seed, trees, depth, seed, extra_seed, threads, max_features, criterion="gini", min_samples_split=2, min_samples_leaf=1, min_leaf_weight=0.0, min_impurity_decrease=0.0, max_leaves=None, bootstrap=true, sample_size=None, extra_trees=true))]
    fn gain_importances(
        &mut self,
        active: PyReadonlyArray1<'_, i64>,
        sources: PyReadonlyArray1<'_, i64>,
        shadow_seed: u64,
        trees: usize,
        depth: usize,
        seed: u64,
        extra_seed: u64,
        threads: usize,
        max_features: usize,
        criterion: &str,
        min_samples_split: usize,
        min_samples_leaf: usize,
        min_leaf_weight: f64,
        min_impurity_decrease: f64,
        max_leaves: Option<usize>,
        bootstrap: bool,
        sample_size: Option<usize>,
        extra_trees: bool,
    ) -> PyResult<(Vec<f64>, f64, f64, std::collections::HashMap<String, f64>)> {
        let start = std::time::Instant::now();
        let to_columns = |values: PyReadonlyArray1<'_, i64>| -> PyResult<Vec<usize>> {
            values.as_array().iter()
                .map(|&v| usize::try_from(v).ok().filter(|&c| c < self.columns))
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| PyValueError::new_err("invalid feature indices"))
        };
        let (active, sources) = (to_columns(active)?, to_columns(sources)?);
        if active.is_empty() || sources.is_empty() || threads == 0 {
            return Err(PyValueError::new_err("invalid training arrays or settings"));
        }
        if let Some(binned) = &self.binned {
            if !extra_trees {
                return Err(PyValueError::new_err("binned sessions train Extra Trees"));
            }
            let rows = self.rows;
            let mut codes = Vec::with_capacity((active.len() + sources.len()) * rows);
            for &c in &active {
                codes.extend_from_slice(&binned.codes[c * rows..(c + 1) * rows]);
            }
            codes.resize((active.len() + sources.len()) * rows, 0);
            boruta_core::fill_shadows(&binned.codes, rows, &sources, shadow_seed, threads, &mut codes[active.len() * rows..]);
            let edges = active.iter().chain(&sources).map(|&c| binned.edges[c].clone()).collect();
            let shuffle_seconds = start.elapsed().as_secs_f64();
            let codes_bytes = codes.len() as f64;
            let result = boruta_core::gain_importances_binned(
                rows, codes, edges, self.labels.clone(), binned.max_bins, trees, depth, seed, extra_seed, threads,
                max_features, criterion, min_samples_split, min_samples_leaf, min_leaf_weight,
                min_impurity_decrease, max_leaves.unwrap_or(usize::MAX), bootstrap, sample_size,
            )
            .map_err(PyValueError::new_err)?;
            let metrics = [
                ("shuffle_seconds", shuffle_seconds),
                ("storage_seconds", result.preparation_seconds),
                ("real_bank_bytes", binned.codes.len() as f64),
                ("shadow_buffer_bytes", codes_bytes),
                ("materialized_bytes", 0.),
            ]
            .into_iter()
            .map(|(k, v)| (k.to_owned(), v))
            .collect();
            return Ok((result.importances, shuffle_seconds + result.preparation_seconds, result.training_seconds, metrics));
        }
        // The previous round's forest is gone, so the buffer is uniquely owned.
        let buffer = std::sync::Arc::make_mut(&mut self.shadows);
        buffer.resize(sources.len() * self.rows, 0.);
        boruta_core::fill_shadows(&self.bank, self.rows, &sources, shadow_seed, threads, buffer);
        let shuffle_seconds = start.elapsed().as_secs_f64();
        let table = boruta_core::ExactColumnTable::new(self.rows, self.bank.clone(), self.shadows.clone(), &active, sources.len())
            .map_err(PyValueError::new_err)?;
        let result = boruta_core::gain_importances_columns(
            table, self.labels.clone(), extra_trees, trees, depth, seed, extra_seed, threads, max_features,
            criterion, min_samples_split, min_samples_leaf, min_leaf_weight, min_impurity_decrease,
            max_leaves.unwrap_or(usize::MAX), bootstrap, sample_size,
        )
        .map_err(PyValueError::new_err)?;
        let metrics = [
            ("shuffle_seconds", shuffle_seconds),
            ("storage_seconds", result.preparation_seconds),
            ("real_bank_bytes", (self.bank.len() * 4) as f64),
            ("shadow_buffer_bytes", (self.shadows.capacity() * 4) as f64),
            ("materialized_bytes", 0.),
            ("feature_workspace_peak_bytes_per_worker", result.feature_workspace_peak_bytes_per_worker),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v))
        .collect();
        Ok((result.importances, shuffle_seconds + result.preparation_seconds, result.training_seconds, metrics))
    }
}

/// Shadow columns of a column-major float32 bank, as a session fills them.
#[pyfunction]
fn _fill_shadows_f32(bank: Vec<f32>, rows: usize, sources: Vec<usize>, seed: u64, threads: usize) -> PyResult<Vec<f32>> {
    if rows == 0 || bank.len() % rows != 0 || sources.iter().any(|&c| c >= bank.len() / rows) {
        return Err(PyValueError::new_err("invalid bank or sources"));
    }
    let mut shadows = vec![0.; sources.len() * rows];
    boruta_core::fill_shadows(&bank, rows, &sources, seed, threads.max(1), &mut shadows);
    Ok(shadows)
}

#[pymodule]
fn _boruta_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ExactSession>()?;
    m.add_class::<ShadowSession>()?;
    m.add_function(wrap_pyfunction!(_fill_shadows_f32, m)?)?;
    m.add_function(wrap_pyfunction!(gain_importances, m)?)?;
    m.add_function(wrap_pyfunction!(shuffle_uint32, m)?)?;
    Ok(())
}
