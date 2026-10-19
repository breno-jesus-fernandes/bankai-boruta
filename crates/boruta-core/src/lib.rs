//! BankaiRF a15 training with virtual shadow access and per-tree gain normalization.
#![forbid(unsafe_code)]
use bankai_random_forest::dense::{BinningStrategy, Criterion, DenseInput};
pub use bankai_random_forest::dense::{ExactColumnBank, ExactColumnTable, ExactRowIndices, ExactVirtualColumns};
use std::time::Instant;
use xrf::{Forest, RfInput};

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
    Ok(GainResult {
        importances: tree_normalized_gains(&forest, columns),
        preparation_seconds,
        training_seconds: start.elapsed().as_secs_f64(),
        feature_workspace_peak_bytes_per_worker: profile
            .as_ref()
            .map_or(0., |p| p.snapshot()["feature_buffer_peak_bytes_per_worker"]),
    })
}

/// Rows sorted by every column in order, then label, then original index.
/// The primary column is cached contiguously and only tie groups are refined.
pub fn canonical_row_order<F: Fn(usize, usize) -> f32>(rows: usize, columns: usize, labels: &[usize], value: F) -> Vec<usize> {
    let mut primary: Vec<(f32, usize)> = (0..rows).map(|row| (value(row, 0), row)).collect();
    primary.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let mut order: Vec<usize> = primary.iter().map(|&(_, row)| row).collect();
    let mut start = 0;
    while start < rows {
        let mut end = start + 1;
        while end < rows && primary[end].0.total_cmp(&primary[start].0).is_eq() {
            end += 1;
        }
        if end - start > 1 {
            order[start..end].sort_by(|&a, &b| {
                (1..columns)
                    .map(|column| value(a, column).total_cmp(&value(b, column)))
                    .find(|ordering| !ordering.is_eq())
                    .unwrap_or_else(|| labels[a].cmp(&labels[b]).then(a.cmp(&b)))
            });
        }
        start = end;
    }
    order
}

/// Column-major float32 copy of every column in the given row order.
pub fn real_bank<F: Fn(usize, usize) -> f32 + Sync>(rows: usize, columns: usize, order: &[usize], threads: usize, value: F) -> Vec<f32> {
    let mut bank = vec![0f32; rows * columns];
    let per_worker = columns.div_ceil(threads.max(1)).max(1);
    std::thread::scope(|scope| {
        for (block, dst) in bank.chunks_mut(per_worker * rows).enumerate() {
            let (value, order) = (&value, order);
            scope.spawn(move || {
                for (offset, column) in dst.chunks_mut(rows).enumerate() {
                    let source = block * per_worker + offset;
                    for (cell, &row) in column.iter_mut().zip(order) {
                        *cell = value(row, source);
                    }
                }
            });
        }
    });
    bank
}

/// xoshiro256** seeded by SplitMix64; one independent stream per shadow column.
struct ShadowRng([u64; 4]);
impl ShadowRng {
    fn new(seed: u64, stream: u64) -> Self {
        let mut state = seed ^ stream.wrapping_mul(0xd1b5_4a32_d192_ed03);
        let mut next = || {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        };
        Self([next(), next(), next(), next()])
    }
    fn next(&mut self) -> u64 {
        let s = &mut self.0;
        let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }
    /// Unbiased draw in `0..bound` (Lemire's multiply with rejection).
    fn below(&mut self, bound: u64) -> u64 {
        let mut product = u128::from(self.next()) * u128::from(bound);
        if (product as u64) < bound {
            let threshold = bound.wrapping_neg() % bound;
            while (product as u64) < threshold {
                product = u128::from(self.next()) * u128::from(bound);
            }
        }
        (product >> 64) as u64
    }
}

/// Fill one shadow per source: copy the bank column, then Fisher-Yates in place.
/// Each column has its own stream, so the result does not depend on `threads`.
pub fn fill_shadows(bank: &[f32], rows: usize, sources: &[usize], seed: u64, threads: usize, shadows: &mut Vec<f32>) {
    shadows.resize(sources.len() * rows, 0.);
    let per_worker = sources.len().div_ceil(threads.max(1)).max(1);
    std::thread::scope(|scope| {
        for (block, dst) in shadows.chunks_mut(per_worker * rows).enumerate() {
            scope.spawn(move || {
                for (offset, column) in dst.chunks_mut(rows).enumerate() {
                    let shadow = block * per_worker + offset;
                    let source = sources[shadow];
                    column.copy_from_slice(&bank[source * rows..(source + 1) * rows]);
                    let mut rng = ShadowRng::new(seed, shadow as u64);
                    for i in (1..rows).rev() {
                        column.swap(i, rng.below(i as u64 + 1) as usize);
                    }
                }
            });
        }
    });
}

/// Mean decrease in impurity: per-tree normalized split gains, averaged.
/// Parallel workers return trees in completion order, so tree shares are
/// summed in 2^-60 fixed point: exact, hence independent of that order.
fn tree_normalized_gains(forest: &Forest<DenseInput>, columns: usize) -> Vec<f64> {
    const SCALE: f64 = (1u64 << 60) as f64;
    let mut fixed = vec![0u128; columns];
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
            for (sum, gain) in fixed.iter_mut().zip(&gains) {
                *sum += (gain / total * SCALE).round() as u128;
            }
        }
    }
    let mut result: Vec<f64> = fixed.iter().map(|&sum| sum as f64 / SCALE).collect();
    let total: f64 = result.iter().sum();
    if total > 0. {
        for gain in &mut result {
            *gain /= total;
        }
    }
    result
}

/// Exact Extra Trees over real bank columns and a shadow buffer, both already
/// in training-row order; nothing is gathered or materialized per round.
#[allow(clippy::too_many_arguments)]
pub fn gain_importances_columns(
    table: ExactColumnTable,
    labels: Vec<usize>,
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
    max_leaves: usize,
    bootstrap: bool,
    sample_size: Option<usize>,
) -> Result<GainResult, String> {
    let start = Instant::now();
    let rows = labels.len();
    if threads == 0 || trees == 0 || depth == 0 || max_features == 0
        || min_samples_split < 2 || min_samples_leaf == 0
        || !min_leaf_weight.is_finite() || min_leaf_weight < 0.0
        || !min_impurity_decrease.is_finite() || min_impurity_decrease < 0.0
        || (sample_size.is_some() && !bootstrap)
    {
        return Err("invalid training settings".into());
    }
    let max_label = labels.iter().max().copied().ok_or("empty target")?;
    if max_label == 0 || max_label >= rows {
        return Err("invalid classification labels".into());
    }
    let criterion = match criterion {
        "gini" => Criterion::Gini,
        "entropy" | "log_loss" => Criterion::Entropy,
        _ => return Err("criterion must be 'gini', 'entropy', or 'log_loss'".into()),
    };
    let profile = std::sync::Arc::new(bankai_random_forest::dense::FitProfile::default());
    let input = DenseInput::training_exact_columns(
        table, labels, max_label + 1, min_leaf_weight, criterion,
        min_samples_split, min_samples_leaf, min_impurity_decrease,
    )?
    .with_extra_trees(true, extra_seed)
    .with_histogram_threads(1)
    .with_profiling(Some(profile.clone()));
    let columns = input.feature_count();
    if max_features > columns {
        return Err("invalid training settings".into());
    }
    let preparation_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let forest = Forest::new_parallel_with_settings(
        &input, trees, max_features, true, false, false, seed,
        threads.min(trees), depth, max_leaves, bootstrap, sample_size,
    );
    Ok(GainResult {
        importances: tree_normalized_gains(&forest, columns),
        preparation_seconds,
        training_seconds: start.elapsed().as_secs_f64(),
        feature_workspace_peak_bytes_per_worker: profile.snapshot()["feature_buffer_peak_bytes_per_worker"],
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

#[cfg(test)]
mod shadow_session_tests {
    use super::*;

    #[test]
    fn shadows_permute_sources_deterministically_for_any_thread_count() {
        let rows = 257;
        let bank: Vec<f32> = (0..rows * 3).map(|i| i as f32).collect();
        let sources = [2, 0, 2, 1, 0];
        let mut reference = Vec::new();
        fill_shadows(&bank, rows, &sources, 11, 1, &mut reference);
        for threads in [2, 3, 8] {
            let mut other = vec![7.; 3];
            fill_shadows(&bank, rows, &sources, 11, threads, &mut other);
            assert_eq!(reference, other);
        }
        for (shadow, &source) in sources.iter().enumerate() {
            let mut column = reference[shadow * rows..(shadow + 1) * rows].to_vec();
            assert_ne!(column, bank[source * rows..(source + 1) * rows]);
            column.sort_by(f32::total_cmp);
            assert_eq!(column, bank[source * rows..(source + 1) * rows]);
        }
        // Repeated sources draw independent permutations.
        assert_ne!(reference[..rows], reference[2 * rows..3 * rows]);
        let mut reseeded = Vec::new();
        fill_shadows(&bank, rows, &sources, 12, 2, &mut reseeded);
        assert_ne!(reference, reseeded);
    }

    #[test]
    fn shadow_positions_are_uniform() {
        let rows = 8;
        let bank: Vec<f32> = (0..rows).map(|i| i as f32).collect();
        let mut counts = [[0u32; 8]; 8];
        let mut shadows = Vec::new();
        for seed in 0..20_000 {
            fill_shadows(&bank, rows, &[0], seed, 1, &mut shadows);
            for (position, &value) in shadows.iter().enumerate() {
                counts[value as usize][position] += 1;
            }
        }
        for row in counts {
            for count in row {
                assert!((2_200..2_800).contains(&count), "{count}");
            }
        }
    }

    #[test]
    fn canonical_order_matches_full_lexicographic_sort() {
        let rows = 300;
        let x: Vec<[f32; 3]> = (0..rows).map(|r| [(r % 4) as f32, ((r * 7) % 5) as f32, ((r * 13) % 3) as f32]).collect();
        let labels: Vec<usize> = (0..rows).map(|r| (r * 3) % 2).collect();
        let order = canonical_row_order(rows, 3, &labels, |r, c| x[r][c]);
        let mut expected: Vec<usize> = (0..rows).collect();
        expected.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap().then(labels[a].cmp(&labels[b])).then(a.cmp(&b)));
        assert_eq!(order, expected);
        let bank = real_bank(rows, 3, &order, 2, |r, c| x[r][c]);
        assert!((0..rows).all(|i| (0..3).all(|c| bank[c * rows + i] == x[order[i]][c])));
    }
}
