//! BankaiRF a15 training with virtual shadow access and per-tree gain normalization.
#![forbid(unsafe_code)]
use bankai_random_forest::dense::{BinningStrategy, Criterion, DenseInput};
use std::time::Instant;
use xrf::Forest;

pub struct GainResult {
    pub importances: Vec<f64>,
    pub preparation_seconds: f64,
    pub training_seconds: f64,
}

#[allow(clippy::too_many_arguments)]
pub fn gain_importances<F: Fn(usize, usize) -> f64 + Sync>(
    rows: usize,
    columns: usize,
    labels: Vec<usize>,
    trees: usize,
    depth: usize,
    seed: u64,
    bins: usize,
    extra_seed: u64,
    value: F,
) -> Result<GainResult, String> {
    let start = Instant::now();
    if rows == 0
        || columns == 0
        || labels.len() != rows
        || trees == 0
        || depth == 0
        || !(2..=255).contains(&bins)
    {
        return Err("invalid training settings".into());
    }
    let max_label = labels.iter().max().copied().ok_or("empty target")?;
    if max_label == 0 || max_label >= rows {
        return Err("invalid classification labels".into());
    }
    let classes = max_label + 1;
    let threads = std::thread::available_parallelism().map_or(1, usize::from);
    let input = DenseInput::training_binned_with_accessor_and_seed(
        rows,
        columns,
        labels,
        vec![1.; rows],
        classes,
        0.,
        Criterion::Gini,
        2,
        1,
        0.,
        bins,
        threads,
        BinningStrategy::ExactSort,
        200000,
        6,
        value,
    )?
    .with_extra_trees(true, extra_seed)
    .with_histogram_threads(1)
    .with_binned_extra_trees(threads);
    let preparation_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let forest = Forest::new_parallel_with_settings(
        &input,
        trees,
        (columns as f64).sqrt() as usize,
        true,
        false,
        false,
        seed,
        threads.min(trees),
        depth,
        usize::MAX,
        true,
        None,
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
    })
}
