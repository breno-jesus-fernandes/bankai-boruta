//! BankaiRF a15 training with virtual shadow access and per-tree gain normalization.
#![forbid(unsafe_code)]
use bankai_random_forest::dense::{BinningStrategy, Criterion, DenseInput};
pub use bankai_random_forest::dense::{ExactColumnBank, ExactRowIndices, ExactVirtualColumns};
use std::time::Instant;
use xrf::Forest;

pub struct GainResult {
    pub importances: Vec<f64>,
    pub preparation_seconds: f64,
    pub training_seconds: f64,
    pub feature_workspace_peak_bytes_per_worker: f64,
}

#[allow(clippy::too_many_arguments)]
pub fn gain_importances<F: Fn(usize, usize) -> f64 + Sync>(
    rows: usize,
    columns: usize,
    labels: Vec<usize>,
    trees: usize,
    depth: usize,
    seed: u64,
    bins: Option<usize>,
    extra_seed: u64,
    threads: usize,
    max_features: usize,
    criterion: &str,
    min_samples_split: usize,
    min_samples_leaf: usize,
    min_leaf_weight: f64,
    min_impurity_decrease: f64,
    max_leaves: usize,
    bootstrap: bool,
    sample_size: Option<usize>,
    binning_strategy: &str,
    bin_sample_size: usize,
    binning_seed: u64,
    value: F,
) -> Result<GainResult, String> {
    gain_importances_with_storage(
        rows,
        columns,
        labels,
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
        max_leaves,
        bootstrap,
        sample_size,
        binning_strategy,
        bin_sample_size,
        binning_seed,
        value,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn gain_importances_with_storage<F: Fn(usize, usize) -> f64 + Sync>(
    rows: usize,
    columns: usize,
    labels: Vec<usize>,
    trees: usize,
    depth: usize,
    seed: u64,
    bins: Option<usize>,
    extra_seed: u64,
    threads: usize,
    max_features: usize,
    criterion: &str,
    min_samples_split: usize,
    min_samples_leaf: usize,
    min_leaf_weight: f64,
    min_impurity_decrease: f64,
    max_leaves: usize,
    bootstrap: bool,
    sample_size: Option<usize>,
    binning_strategy: &str,
    bin_sample_size: usize,
    binning_seed: u64,
    value: F,
    virtual_storage: Option<ExactVirtualColumns>,
) -> Result<GainResult, String> {
    let start = Instant::now();
    if rows == 0
        || columns == 0
        || labels.len() != rows
        || threads == 0
        || max_features == 0
        || max_features > columns
        || trees == 0
        || depth == 0
        || bins.is_some_and(|value| !(2..=255).contains(&value))
        || min_samples_split < 2
        || min_samples_leaf == 0
        || !min_leaf_weight.is_finite()
        || min_leaf_weight < 0.0
        || !min_impurity_decrease.is_finite()
        || min_impurity_decrease < 0.0
        || (sample_size.is_some() && !bootstrap)
        || (bins.is_some() && bin_sample_size == 0)
    {
        return Err("invalid training settings".into());
    }
    let max_label = labels.iter().max().copied().ok_or("empty target")?;
    if max_label == 0 || max_label >= rows {
        return Err("invalid classification labels".into());
    }
    let classes = max_label + 1;
    let criterion = match criterion {
        "gini" => Criterion::Gini,
        "entropy" | "log_loss" => Criterion::Entropy,
        _ => return Err("criterion must be 'gini', 'entropy', or 'log_loss'".into()),
    };
    let binning_strategy = BinningStrategy::parse(binning_strategy)?;
    let profile = virtual_storage
        .as_ref()
        .map(|_| std::sync::Arc::new(bankai_random_forest::dense::FitProfile::default()));
    let input = if let Some(bins) = bins {
        DenseInput::training_binned_with_accessor_and_seed(
            rows,
            columns,
            labels,
            vec![1.; rows],
            classes,
            min_leaf_weight,
            criterion,
            min_samples_split,
            min_samples_leaf,
            min_impurity_decrease,
            bins,
            threads,
            binning_strategy,
            bin_sample_size,
            binning_seed,
            value,
        )?
        .with_extra_trees(true, extra_seed)
        .with_float32_column_storage(threads)
        .with_histogram_threads(1)
        .with_binned_extra_trees(threads)
    } else if let Some(storage) = virtual_storage {
        DenseInput::training_exact_virtual(
            storage,
            labels,
            classes,
            min_leaf_weight,
            criterion,
            min_samples_split,
            min_samples_leaf,
            min_impurity_decrease,
        )?
        .with_extra_trees(true, extra_seed)
        .with_histogram_threads(1)
    } else {
        DenseInput::training_exact_f32_with_accessor(
            rows,
            columns,
            labels,
            vec![1.; rows],
            classes,
            min_leaf_weight,
            criterion,
            min_samples_split,
            min_samples_leaf,
            min_impurity_decrease,
            threads,
            |row, column| value(row, column) as f32,
        )?
        .with_extra_trees(true, extra_seed)
        .with_histogram_threads(1)
    };
    let input = input.with_profiling(profile.clone());
    let preparation_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let forest = Forest::new_parallel_with_settings(
        &input,
        trees,
        max_features,
        true,
        false,
        false,
        seed,
        threads.min(trees),
        depth,
        max_leaves,
        bootstrap,
        sample_size,
    );
    let mut result = vec![0.; columns];
    let mut gains = vec![0.; columns];
    for tree in forest.tree_refs() {
        gains.fill(0.);
        for node in &tree.nodes {
            if let xrf::Node::Branch(feature, _, score, _, _, _) = node {
                gains[*feature] += score.abs();
            }
        }
        let total: f64 = gains.iter().sum();
        if total > 0. {
            for (sum, gain) in result.iter_mut().zip(&gains) {
                *sum += gain / total;
            }
        }
    }
    let total: f64 = result.iter().sum();
    if total > 0. {
        for gain in &mut result {
            *gain /= total;
        }
    }
    Ok(GainResult {
        importances: result,
        preparation_seconds,
        training_seconds: start.elapsed().as_secs_f64(),
        feature_workspace_peak_bytes_per_worker: profile
            .as_ref()
            .map_or(0., |p| p.snapshot()["feature_buffer_peak_bytes_per_worker"]),
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn rejects_invalid_native_controls() {
        for (threads, max_features) in [(0, 1), (1, 0), (1, 3)] {
            let result = super::gain_importances(
                3,
                2,
                vec![0, 1, 0],
                2,
                3,
                1,
                Some(63),
                6,
                threads,
                max_features,
                "gini",
                2,
                1,
                0.,
                0.,
                usize::MAX,
                true,
                None,
                "exact_sort",
                200000,
                6,
                |row, column| (row + column) as f64,
            );
            assert!(matches!(result, Err(message) if message == "invalid training settings"));
        }
    }
}
