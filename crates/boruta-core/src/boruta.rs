use crate::statistics::decision_table;
use crate::{
    DecisionLedger, FeatureDecision, FeatureStorage, LogicalMatrix, MatrixForestInput,
    MatrixViewError,
};
use bankai_forest_core::xrf::Forest;
use std::sync::Arc;
use thiserror::Error;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MaxFeatures {
    All,
    Sqrt,
    Log2,
    Count(usize),
    Fraction(f64),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImportanceScale {
    Raw,
    ZScore,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PruningMode {
    Canonical,
    Aggressive,
}

#[derive(Clone, Debug)]
pub struct BorutaConfig {
    pub n_estimators: usize,
    pub max_iter: usize,
    pub alpha: f64,
    pub pruning: PruningMode,
    pub max_features: MaxFeatures,
    pub max_depth: Option<usize>,
    pub n_jobs: usize,
    pub random_state: u64,
    pub importance_scale: ImportanceScale,
    pub store_history: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StopReason {
    NoTentativeFeatures,
    MaxIterations,
}

#[derive(Clone, Debug)]
pub struct BorutaResult {
    pub decisions: Vec<FeatureDecision>,
    pub n_iter: usize,
    pub stop_reason: StopReason,
    pub undefined_importance_count: usize,
    pub non_finite_importance_count: usize,
    pub hit_history: Option<Vec<Vec<u32>>>,
    pub active_feature_counts: Option<Vec<usize>>,
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum BorutaError {
    #[error("classification requires at least two classes")]
    ClassCount,
    #[error("at least two observations are required for out-of-bag importance")]
    ObservationCount,
    #[error("Boruta configuration contains an invalid value")]
    InvalidConfiguration,
    #[error("out-of-bag importance has no valid observations in iteration {iteration}")]
    NoOobEvidence { iteration: usize },
    #[error("importance provider returned {actual} scores; expected {expected}")]
    ImportanceLength { expected: usize, actual: usize },
    #[error("importance provider failed: {0}")]
    ImportanceProvider(String),
    #[error("forest input is invalid: {0}")]
    ForestInput(String),
    #[error("matrix view is invalid: {0}")]
    MatrixView(#[from] MatrixViewError),
}

/// Pluggable source of one aggregate importance vector per Boruta iteration.
/// Implementations stay in Rust; Python callbacks are intentionally excluded.
pub trait ImportanceProvider<S: FeatureStorage + Send + Sync> {
    fn compute(
        &mut self,
        matrix: LogicalMatrix<S>,
        labels: Arc<Vec<usize>>,
        n_classes: usize,
        active_features: &[usize],
        config: &BorutaConfig,
        seed: u64,
    ) -> Result<(Vec<f64>, bool, usize), BorutaError>;
}

#[derive(Default)]
pub struct PermutationImportanceProvider;

impl<S: FeatureStorage + Send + Sync> ImportanceProvider<S> for PermutationImportanceProvider {
    fn compute(
        &mut self,
        matrix: LogicalMatrix<S>,
        labels: Arc<Vec<usize>>,
        n_classes: usize,
        active_features: &[usize],
        config: &BorutaConfig,
        seed: u64,
    ) -> Result<(Vec<f64>, bool, usize), BorutaError> {
        let matrix = matrix.with_shuffled_shadows(active_features, seed.rotate_left(17))?;
        let input = MatrixForestInput::new_shared(matrix.clone(), labels, n_classes)
            .map_err(|error| BorutaError::ForestInput(error.to_string()))?;
        let (_, columns) = matrix.shape();
        let tries = resolve_max_features(config.max_features, columns);
        let max_depth = config.max_depth.unwrap_or(usize::MAX);
        let forest = if config.n_jobs == 1 {
            Forest::new_with_max_depth(
                &input,
                config.n_estimators,
                tries,
                false,
                true,
                true,
                seed,
                max_depth,
            )
        } else {
            Forest::new_parallel_deterministic_with_settings(
                &input,
                config.n_estimators,
                tries,
                false,
                true,
                true,
                seed,
                config.n_jobs,
                max_depth,
                usize::MAX,
                true,
                None,
            )
        };
        let valid_oob = forest.oob().any(|(_, votes)| votes.observations() > 0);
        let mut scores = vec![0.0; columns];
        let (importances, undefined) = match config.importance_scale {
            ImportanceScale::Raw => (forest.importance().collect::<Vec<_>>(), 0),
            ImportanceScale::ZScore => {
                let importances = forest.importance_normalised().collect::<Vec<_>>();
                let undefined = columns.saturating_sub(importances.len());
                (importances, undefined)
            }
        };
        for (feature, score) in importances {
            if let Some(slot) = scores.get_mut(feature) {
                *slot = score;
            }
        }
        Ok((scores, valid_oob, undefined))
    }
}

fn resolve_max_features(spec: MaxFeatures, columns: usize) -> usize {
    let count = match spec {
        MaxFeatures::All => columns,
        MaxFeatures::Sqrt => (columns as f64).sqrt().floor() as usize,
        MaxFeatures::Log2 => (columns as f64).log2().floor() as usize,
        MaxFeatures::Count(count) => count,
        MaxFeatures::Fraction(fraction) => (fraction * columns as f64).ceil() as usize,
    };
    count.clamp(1, columns)
}

pub fn run_boruta<S, P>(
    matrix: LogicalMatrix<S>,
    labels: Vec<usize>,
    n_classes: usize,
    config: &BorutaConfig,
    provider: &mut P,
) -> Result<BorutaResult, BorutaError>
where
    S: FeatureStorage + Send + Sync,
    P: ImportanceProvider<S>,
{
    validate_config(config)?;
    let (rows, columns) = matrix.shape();
    if rows < 2 {
        return Err(BorutaError::ObservationCount);
    }
    if n_classes < 2 {
        return Err(BorutaError::ClassCount);
    }
    if labels.len() != rows || labels.iter().any(|&label| label >= n_classes) {
        return Err(BorutaError::ForestInput(
            "target labels must match X and use contiguous class IDs".to_string(),
        ));
    }
    let mut observed_classes = vec![false; n_classes];
    for &label in &labels {
        observed_classes[label] = true;
    }
    if observed_classes
        .iter()
        .filter(|&&observed| observed)
        .count()
        < 2
    {
        return Err(BorutaError::ClassCount);
    }
    let labels = Arc::new(labels);
    let mut ledger = DecisionLedger::new(columns);
    let mut hits = vec![0_u32; columns];
    let mut hit_history = config.store_history.then(Vec::new);
    let mut active_feature_counts = config.store_history.then(Vec::new);
    let mut undefined_importance_count = 0;
    let mut non_finite_importance_count = 0;
    let mut n_iter = 0;

    for iteration in 0..config.max_iter {
        let tentative = ledger
            .decisions()
            .iter()
            .enumerate()
            .filter_map(|(feature, decision)| {
                (*decision == FeatureDecision::Tentative).then_some(feature)
            })
            .collect::<Vec<_>>();
        if tentative.is_empty() {
            break;
        }
        let active = ledger
            .decisions()
            .iter()
            .enumerate()
            .filter_map(|(feature, decision)| {
                let active = match config.pruning {
                    PruningMode::Canonical => *decision != FeatureDecision::Rejected,
                    PruningMode::Aggressive => *decision == FeatureDecision::Tentative,
                };
                active.then_some(feature)
            })
            .collect::<Vec<_>>();
        if let Some(history) = &mut active_feature_counts {
            history.push(active.len());
        }
        let mut active_positions = vec![usize::MAX; columns];
        for (position, &feature) in active.iter().enumerate() {
            active_positions[feature] = position;
        }
        let seed = iteration_seed(config.random_state, iteration);
        let (mut scores, valid_oob, undefined) = provider.compute(
            matrix.clone(),
            Arc::clone(&labels),
            n_classes,
            &active,
            config,
            seed,
        )?;
        n_iter += 1;
        if !valid_oob {
            return Err(BorutaError::NoOobEvidence { iteration });
        }
        let shadow_count = active.len().max(5);
        let expected = active.len() + shadow_count;
        if scores.len() != expected {
            return Err(BorutaError::ImportanceLength {
                expected,
                actual: scores.len(),
            });
        }
        undefined_importance_count += undefined;
        for score in &mut scores {
            if !score.is_finite() {
                *score = 0.0;
                non_finite_importance_count += 1;
            }
        }
        let shadow_max = scores[active.len()..]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        for &feature in &tentative {
            let position = active_positions[feature];
            debug_assert_ne!(position, usize::MAX, "tentative features remain active");
            if scores[position] > shadow_max {
                hits[feature] += 1;
            }
        }
        let decisions = decision_table(n_iter, columns, config.alpha);
        for feature in tentative {
            let decision = decisions[hits[feature] as usize];
            if decision != FeatureDecision::Tentative {
                ledger
                    .decide(feature, decision)
                    .expect("each tentative feature is decided at most once");
            }
        }
        if let Some(history) = &mut hit_history {
            history.push(hits.clone());
        }
    }

    let stop_reason = if ledger
        .decisions()
        .iter()
        .all(|decision| *decision != FeatureDecision::Tentative)
    {
        StopReason::NoTentativeFeatures
    } else {
        StopReason::MaxIterations
    };
    Ok(BorutaResult {
        decisions: ledger.decisions().to_vec(),
        n_iter,
        stop_reason,
        undefined_importance_count,
        non_finite_importance_count,
        hit_history,
        active_feature_counts,
    })
}

fn iteration_seed(seed: u64, iteration: usize) -> u64 {
    let mut value = seed ^ (iteration as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn validate_config(config: &BorutaConfig) -> Result<(), BorutaError> {
    if config.n_estimators == 0
        || config.max_iter == 0
        || config.n_jobs == 0
        || config.max_depth == Some(0)
        || !config.alpha.is_finite()
        || !(0.0..1.0).contains(&config.alpha)
        || matches!(config.max_features, MaxFeatures::Count(0))
        || matches!(config.max_features, MaxFeatures::Fraction(fraction) if !fraction.is_finite() || fraction <= 0.0 || fraction > 1.0)
    {
        return Err(BorutaError::InvalidConfiguration);
    }
    Ok(())
}
