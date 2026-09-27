//! Core state and statistics for the native Boruta implementation.
#![forbid(unsafe_code)]

use bankai_forest_core::best_split_gini_exact;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use thiserror::Error;

mod statistics;
pub use statistics::{binomial_lower_tail, binomial_upper_tail, decide_from_hits};
mod boruta;
pub use boruta::{
    BorutaConfig, BorutaError, BorutaResult, ImportanceProvider, ImportanceScale, MaxFeatures,
    PermutationImportanceProvider, StopReason, run_boruta,
};

/// The current, irreversible decision for a feature from the original input.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FeatureDecision {
    Tentative,
    Confirmed,
    Rejected,
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum LedgerError {
    #[error("feature id {feature_id} is outside the original feature range 0..{feature_count}")]
    InvalidFeatureId {
        feature_id: usize,
        feature_count: usize,
    },
    #[error("feature {feature_id} already has a final decision")]
    AlreadyDecided { feature_id: usize },
    #[error("a feature decision must be Confirmed or Rejected")]
    TentativeIsNotFinal,
}

/// Stores decisions by original feature id, independent of the active training view.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecisionLedger {
    decisions: Vec<FeatureDecision>,
}

impl DecisionLedger {
    pub fn new(feature_count: usize) -> Self {
        Self {
            decisions: vec![FeatureDecision::Tentative; feature_count],
        }
    }

    pub fn decisions(&self) -> &[FeatureDecision] {
        &self.decisions
    }

    pub fn decide(
        &mut self,
        feature_id: usize,
        decision: FeatureDecision,
    ) -> Result<(), LedgerError> {
        let feature_count = self.decisions.len();
        let current = self
            .decisions
            .get_mut(feature_id)
            .ok_or(LedgerError::InvalidFeatureId {
                feature_id,
                feature_count,
            })?;

        if decision == FeatureDecision::Tentative {
            return Err(LedgerError::TentativeIsNotFinal);
        }
        if *current != FeatureDecision::Tentative {
            return Err(LedgerError::AlreadyDecided { feature_id });
        }
        *current = decision;
        Ok(())
    }
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum MatrixViewError {
    #[error("matrix must have at least one row and one source feature")]
    EmptyShape,
    #[error("matrix strides overflow addressable storage")]
    OffsetOverflow,
    #[error("matrix buffer does not contain the requested shape and strides")]
    BufferTooSmall,
    #[error("matrix value at row {row}, feature {feature} is not finite")]
    NonFiniteValue { row: usize, feature: usize },
    #[error("the logical feature view must contain at least one feature")]
    EmptyFeatureView,
    #[error("feature id {feature_id} is outside 0..{feature_count}")]
    InvalidFeatureId {
        feature_id: usize,
        feature_count: usize,
    },
    #[error("feature id {0} appears more than once in the active feature list")]
    DuplicateFeatureId(usize),
    #[error("shadow feature {feature_id} has a malformed row permutation")]
    InvalidPermutation { feature_id: usize },
}

/// A logical shadow column sourced from an original input feature.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShadowSpec {
    pub source_feature: usize,
    pub row_permutation: Arc<[usize]>,
}

#[derive(Clone, Debug)]
struct ColumnView {
    source_feature: usize,
    row_permutation: Option<std::sync::Arc<[usize]>>,
}

/// Borrowed strided matrix plus a logical feature map; shadow columns stay virtual.
#[derive(Clone, Debug)]
pub struct MatrixView<'a, T = f64>
where
    T: Copy + Into<f64>,
{
    values: &'a [T],
    rows: usize,
    source_features: usize,
    row_stride: usize,
    column_stride: usize,
    columns: Vec<ColumnView>,
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum MatrixForestInputError {
    #[error("target length must match the matrix row count")]
    TargetLength,
    #[error("classification requires at least two classes")]
    ClassCount,
    #[error("target labels must be encoded in 0..n_classes")]
    TargetEncoding,
}

/// XRF adapter for a borrowed logical matrix and owned, shared classification labels.
#[derive(Clone)]
pub struct MatrixForestInput<'a, T = f64>
where
    T: Copy + Into<f64>,
{
    matrix: MatrixView<'a, T>,
    labels: std::sync::Arc<Vec<usize>>,
    n_classes: usize,
}

impl<'a, T> MatrixForestInput<'a, T>
where
    T: Copy + Into<f64>,
{
    pub fn new(
        matrix: MatrixView<'a, T>,
        labels: Vec<usize>,
        n_classes: usize,
    ) -> Result<Self, MatrixForestInputError> {
        Self::new_shared(matrix, std::sync::Arc::new(labels), n_classes)
    }

    pub fn new_shared(
        matrix: MatrixView<'a, T>,
        labels: std::sync::Arc<Vec<usize>>,
        n_classes: usize,
    ) -> Result<Self, MatrixForestInputError> {
        if labels.len() != matrix.rows {
            return Err(MatrixForestInputError::TargetLength);
        }
        if n_classes < 2 {
            return Err(MatrixForestInputError::ClassCount);
        }
        if labels.iter().any(|&label| label >= n_classes) {
            return Err(MatrixForestInputError::TargetEncoding);
        }
        Ok(Self {
            matrix,
            labels,
            n_classes,
        })
    }

    pub fn matrix(&self) -> &MatrixView<'a, T> {
        &self.matrix
    }
}

#[derive(Clone)]
pub struct MatrixVotes {
    classes: Vec<usize>,
    observations: usize,
}

impl MatrixVotes {
    pub fn observations(&self) -> usize {
        self.observations
    }
}

impl<T> bankai_forest_core::xrf::VoteAggregator<MatrixForestInput<'_, T>> for MatrixVotes
where
    T: Copy + Into<f64>,
{
    fn new(input: &MatrixForestInput<'_, T>) -> Self {
        Self {
            classes: vec![0; input.n_classes],
            observations: 0,
        }
    }

    fn ingest_vote(&mut self, vote: usize) {
        self.classes[vote] += 1;
        self.observations += 1;
    }

    fn merge(&mut self, other: &Self) {
        for (into, from) in self.classes.iter_mut().zip(&other.classes) {
            *into += *from;
        }
        self.observations += other.observations;
    }
}

impl<'a, T> MatrixView<'a, T>
where
    T: Copy + Into<f64>,
{
    /// Construct a view whose strides are measured in elements, not bytes.
    pub fn new(
        values: &'a [T],
        rows: usize,
        source_features: usize,
        row_stride: usize,
        column_stride: usize,
    ) -> Result<Self, MatrixViewError> {
        if rows == 0 || source_features == 0 {
            return Err(MatrixViewError::EmptyShape);
        }
        let last_row = (rows - 1)
            .checked_mul(row_stride)
            .ok_or(MatrixViewError::OffsetOverflow)?;
        let last_column = (source_features - 1)
            .checked_mul(column_stride)
            .ok_or(MatrixViewError::OffsetOverflow)?;
        let last_offset = last_row
            .checked_add(last_column)
            .ok_or(MatrixViewError::OffsetOverflow)?;
        if last_offset >= values.len() {
            return Err(MatrixViewError::BufferTooSmall);
        }
        for row in 0..rows {
            for feature in 0..source_features {
                let offset = row * row_stride + feature * column_stride;
                if !values[offset].into().is_finite() {
                    return Err(MatrixViewError::NonFiniteValue { row, feature });
                }
            }
        }
        Ok(Self {
            values,
            rows,
            source_features,
            row_stride,
            column_stride,
            columns: (0..source_features)
                .map(|source_feature| ColumnView {
                    source_feature,
                    row_permutation: None,
                })
                .collect(),
        })
    }

    /// Replace the logical columns with selected real features followed by shadows.
    pub fn with_virtual_shadows(
        &self,
        active_features: &[usize],
        shadows: &[ShadowSpec],
    ) -> Result<Self, MatrixViewError> {
        if active_features.is_empty() {
            return Err(MatrixViewError::EmptyFeatureView);
        }
        let mut seen = HashSet::with_capacity(active_features.len());
        let mut columns = Vec::with_capacity(active_features.len() + shadows.len());
        for &feature_id in active_features {
            if feature_id >= self.source_features {
                return Err(MatrixViewError::InvalidFeatureId {
                    feature_id,
                    feature_count: self.source_features,
                });
            }
            if !seen.insert(feature_id) {
                return Err(MatrixViewError::DuplicateFeatureId(feature_id));
            }
            columns.push(ColumnView {
                source_feature: feature_id,
                row_permutation: None,
            });
        }
        for shadow in shadows {
            if shadow.source_feature >= self.source_features {
                return Err(MatrixViewError::InvalidFeatureId {
                    feature_id: shadow.source_feature,
                    feature_count: self.source_features,
                });
            }
            let mut rows_seen = vec![false; self.rows];
            if shadow.row_permutation.len() != self.rows
                || shadow.row_permutation.iter().any(|&row| {
                    if row >= self.rows || rows_seen[row] {
                        true
                    } else {
                        rows_seen[row] = true;
                        false
                    }
                })
            {
                return Err(MatrixViewError::InvalidPermutation {
                    feature_id: shadow.source_feature,
                });
            }
            columns.push(ColumnView {
                source_feature: shadow.source_feature,
                row_permutation: Some(shadow.row_permutation.clone()),
            });
        }
        Ok(Self {
            values: self.values,
            rows: self.rows,
            source_features: self.source_features,
            row_stride: self.row_stride,
            column_stride: self.column_stride,
            columns,
        })
    }

    /// Create one independently permuted shadow per active feature, with a floor of five.
    pub fn with_shuffled_shadows(
        &self,
        active_features: &[usize],
        seed: u64,
    ) -> Result<Self, MatrixViewError> {
        use bankai_forest_core::xrf::RfRng;

        if active_features.is_empty() {
            return Err(MatrixViewError::EmptyFeatureView);
        }
        let mut rng = RfRng::from_seed(seed, 1);
        let shadow_count = active_features.len().max(5);
        let mut shadows = Vec::with_capacity(shadow_count);
        for shadow in 0..shadow_count {
            let mut row_permutation = (0..self.rows).collect::<Vec<_>>();
            for row in 0..self.rows.saturating_sub(1) {
                let other = row + rng.up_to(self.rows - row);
                row_permutation.swap(row, other);
            }
            shadows.push(ShadowSpec {
                source_feature: active_features[shadow % active_features.len()],
                row_permutation: row_permutation.into(),
            });
        }
        self.with_virtual_shadows(active_features, &shadows)
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.columns.len())
    }

    pub fn get(&self, row: usize, feature: usize) -> Option<f64> {
        if row >= self.rows {
            return None;
        }
        let column = self.columns.get(feature)?;
        let source_row = column
            .row_permutation
            .as_ref()
            .map_or(row, |permutation| permutation[row]);
        let offset = source_row * self.row_stride + column.source_feature * self.column_stride;
        self.values.get(offset).copied().map(Into::into)
    }

    /// Diagnostic that verifies this view still points to the caller's buffer.
    pub fn shares_storage_with(&self, values: &[T]) -> bool {
        std::ptr::eq(self.values.as_ptr(), values.as_ptr()) && self.values.len() == values.len()
    }
}

pub struct MatrixDecisionSlice {
    labels: Vec<usize>,
    class_counts: Vec<usize>,
}

impl bankai_forest_core::xrf::DecisionSlice<usize> for MatrixDecisionSlice {
    fn is_pure(&self) -> bool {
        self.class_counts.iter().filter(|&&count| count > 0).count() <= 1
    }

    fn condense(&self, rng: &mut bankai_forest_core::xrf::RfRng) -> usize {
        let mut best = bankai_forest_core::xrf::FairBest::new();
        for (class, &count) in self.class_counts.iter().enumerate() {
            best.ingest(count as f64, class, rng);
        }
        best.consume().map_or(0, |(_, class)| class)
    }

    fn condense_with_bounds(
        &self,
        rng: &mut bankai_forest_core::xrf::RfRng,
        lower_bound: f64,
        upper_bound: f64,
    ) -> usize {
        if self.class_counts.len() != 2 || self.labels.is_empty() {
            return self.condense(rng);
        }
        let probability = (self.class_counts[1] as f64 / self.labels.len() as f64)
            .clamp(lower_bound, upper_bound);
        usize::from(probability > 0.5)
    }

    fn positive_probability(&self) -> Option<f64> {
        (self.class_counts.len() == 2)
            .then(|| self.class_counts[1] as f64 / self.labels.len().max(1) as f64)
    }

    fn pruning_weight(&self) -> Option<f64> {
        Some(self.labels.len() as f64)
    }

    fn pruning_risk(&self, root_weight: f64) -> Option<f64> {
        if root_weight <= 0.0 {
            return Some(0.0);
        }
        let total = self.labels.len() as f64;
        let impurity = 1.0
            - self
                .class_counts
                .iter()
                .map(|&count| {
                    let probability = count as f64 / total;
                    probability * probability
                })
                .sum::<f64>();
        Some(total * impurity / root_weight)
    }

    fn pruning_vote(&self) -> Option<usize> {
        self.class_counts
            .iter()
            .enumerate()
            .max_by_key(|&(class, count)| (*count, std::cmp::Reverse(class)))
            .map(|(class, _)| class)
    }
}

pub struct MatrixFeatureSampler {
    feature_count: usize,
}

impl<T> bankai_forest_core::xrf::FeatureSampler<MatrixForestInput<'_, T>> for MatrixFeatureSampler
where
    T: Copy + Into<f64>,
{
    fn random_feature(&mut self, rng: &mut bankai_forest_core::xrf::RfRng) -> usize {
        rng.up_to(self.feature_count)
    }

    fn reload(&mut self) {}

    fn reset(&mut self) {}
}

pub struct MatrixImportance {
    direct: Vec<Option<usize>>,
    drops: HashMap<usize, isize>,
    denominator: usize,
    labels: Arc<Vec<usize>>,
}

impl<T> bankai_forest_core::xrf::AccuracyDecreaseAggregator<MatrixForestInput<'_, T>>
    for MatrixImportance
where
    T: Copy + Into<f64>,
{
    fn new(input: &MatrixForestInput<'_, T>, on: &bankai_forest_core::xrf::Mask, n: usize) -> Self {
        Self {
            direct: vec![None; n],
            drops: HashMap::new(),
            denominator: on.len(),
            labels: input.labels.clone(),
        }
    }

    fn ingest(
        &mut self,
        permuted: Option<usize>,
        mask: &bankai_forest_core::xrf::Mask,
        vote: &usize,
    ) {
        if let Some(feature) = permuted {
            let difference = mask
                .iter()
                .map(|&row| {
                    let direct = self.direct[row]
                        .expect("direct OOB votes must be collected before permutation");
                    if direct == *vote {
                        return 0;
                    }
                    match (self.labels[row] == *vote, self.labels[row] == direct) {
                        (true, false) => -1,
                        (false, true) => 1,
                        (false, false) => 0,
                        (true, true) => unreachable!("different votes cannot both match truth"),
                    }
                })
                .sum::<isize>();
            *self.drops.entry(feature).or_insert(0) += difference;
        } else {
            for &row in mask.iter() {
                self.direct[row] = Some(*vote);
            }
        }
    }

    fn mda_iter(&self) -> impl Iterator<Item = (usize, f64)> {
        self.drops
            .iter()
            .map(|(&feature, &drop)| (feature, drop as f64 / self.denominator.max(1) as f64))
    }

    fn get_direct_vote(&self, row: usize) -> usize {
        self.direct[row].expect("direct OOB vote must be available")
    }
}

impl<T> bankai_forest_core::xrf::RfInput for MatrixForestInput<'_, T>
where
    T: Copy + Into<f64>,
{
    type FeatureId = usize;
    type Pivot = f64;
    type Vote = usize;
    type VoteAggregator = MatrixVotes;
    type DecisionSlice = MatrixDecisionSlice;
    type AccuracyDecreaseAggregator = MatrixImportance;
    type FeatureSampler = MatrixFeatureSampler;
    type SplitCache = ();

    fn observation_count(&self) -> usize {
        self.matrix.rows
    }

    fn feature_count(&self) -> usize {
        self.matrix.columns.len()
    }

    fn decision_slice(&self, mask: &bankai_forest_core::xrf::Mask) -> Self::DecisionSlice {
        let labels = mask.iter().map(|&row| self.labels[row]).collect::<Vec<_>>();
        let mut class_counts = vec![0; self.n_classes];
        for &label in &labels {
            class_counts[label] += 1;
        }
        MatrixDecisionSlice {
            labels,
            class_counts,
        }
    }

    fn feature_sampler(&self) -> Self::FeatureSampler {
        MatrixFeatureSampler {
            feature_count: self.feature_count(),
        }
    }

    fn split_cache(&self, _: &bankai_forest_core::xrf::Mask) -> Self::SplitCache {}

    fn split_cache_children(
        &self,
        _: &Self::SplitCache,
        _: &bankai_forest_core::xrf::Mask,
        _: &bankai_forest_core::xrf::Mask,
        _: &bankai_forest_core::xrf::Mask,
    ) -> (Self::SplitCache, Self::SplitCache) {
        ((), ())
    }

    fn new_split(
        &self,
        mask: &bankai_forest_core::xrf::Mask,
        feature: usize,
        target: &Self::DecisionSlice,
        _: &Self::SplitCache,
        _: &mut bankai_forest_core::xrf::RfRng,
    ) -> Option<(f64, bool, f64)> {
        let mut ranked = mask
            .iter()
            .copied()
            .zip(target.labels.iter().copied())
            .map(|(row, label)| Some((self.matrix.get(row, feature)?, label, 1.0)))
            .collect::<Option<Vec<_>>>()?;
        let class_weights = target
            .class_counts
            .iter()
            .map(|&count| count as f64)
            .collect::<Vec<_>>();
        best_split_gini_exact(
            &mut ranked,
            &class_weights,
            target.labels.len() as f64,
            1,
            0.0,
            0.0,
        )
        .map(|(pivot, missing_left, gain)| {
            (
                pivot,
                missing_left,
                gain * target.labels.len() as f64 / self.matrix.rows as f64,
            )
        })
    }

    fn split_iter<'b>(
        &'b self,
        mask: &'b bankai_forest_core::xrf::Mask,
        feature: usize,
        pivot: &'b f64,
        _: bool,
    ) -> impl Iterator<Item = bool> + 'b {
        mask.iter().map(move |&row| {
            self.matrix
                .get(row, feature)
                .expect("XRF supplied a valid matrix coordinate")
                > *pivot
        })
    }
}
