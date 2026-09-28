use boruta_core::{
    BorutaConfig, BorutaError, BorutaResult, FeatureDecision, FeatureStorage, ImportanceOutput,
    ImportanceProvider, ImportanceScale, LogicalMatrix, MatrixView, MaxFeatures, PruningMode,
    StopReason, run_boruta,
};
use std::sync::Arc;

struct ScriptedImportance {
    calls: usize,
    active_history: Vec<Vec<usize>>,
    missing_oob: bool,
    wrong_length: bool,
}

impl<S: FeatureStorage + Send + Sync> ImportanceProvider<S> for ScriptedImportance {
    fn compute(
        &mut self,
        matrix: LogicalMatrix<S>,
        _: Arc<Vec<usize>>,
        _: usize,
        active_features: &[usize],
        _: &BorutaConfig,
        _: u64,
    ) -> Result<ImportanceOutput, BorutaError> {
        self.calls += 1;
        self.active_history.push(active_features.to_vec());
        assert_eq!(matrix.shape().0, 24);
        let mut scores = vec![1.0; active_features.len() + active_features.len().max(5)];
        for (logical, &feature) in active_features.iter().enumerate() {
            scores[logical] = match feature {
                0 => 2.0,
                1 => 0.0,
                2 if self.calls <= 7 => 1.0,
                _ => 2.0,
            };
        }
        if self.wrong_length {
            scores.pop();
        }
        Ok(ImportanceOutput {
            scores,
            valid_oob: Some(!self.missing_oob),
            undefined_count: 0,
        })
    }
}

struct NonOobImportance;

impl<S: FeatureStorage + Send + Sync> ImportanceProvider<S> for NonOobImportance {
    fn compute(
        &mut self,
        _: LogicalMatrix<S>,
        _: Arc<Vec<usize>>,
        _: usize,
        active_features: &[usize],
        _: &BorutaConfig,
        _: u64,
    ) -> Result<ImportanceOutput, BorutaError> {
        Ok(ImportanceOutput {
            scores: vec![1.0; active_features.len() + active_features.len().max(5)],
            valid_oob: None,
            undefined_count: 0,
        })
    }
}

struct NonFiniteImportance;

impl<S: FeatureStorage + Send + Sync> ImportanceProvider<S> for NonFiniteImportance {
    fn compute(
        &mut self,
        _: LogicalMatrix<S>,
        _: Arc<Vec<usize>>,
        _: usize,
        active_features: &[usize],
        _: &BorutaConfig,
        _: u64,
    ) -> Result<ImportanceOutput, BorutaError> {
        let mut scores = vec![f64::NAN; active_features.len() + active_features.len().max(5)];
        scores[0] = f64::INFINITY;
        Ok(ImportanceOutput {
            scores,
            valid_oob: None,
            undefined_count: 0,
        })
    }
}

fn config() -> BorutaConfig {
    BorutaConfig {
        n_estimators: 16,
        max_iter: 30,
        alpha: 0.01,
        pruning: PruningMode::Canonical,
        max_features: MaxFeatures::Sqrt,
        max_depth: Some(16),
        n_jobs: 1,
        random_state: 17,
        importance_scale: ImportanceScale::ZScore,
        store_history: true,
    }
}

fn run(provider: &mut ScriptedImportance) -> Result<BorutaResult, BorutaError> {
    let rows = 24;
    let values = (0..rows * 3).map(|value| value as f64).collect::<Vec<_>>();
    let matrix = MatrixView::new(&values, rows, 3, 3, 1).unwrap();
    run_boruta(
        matrix,
        (0..rows).map(|row| row % 2).collect(),
        2,
        &config(),
        provider,
    )
}

#[test]
fn canonical_loop_updates_original_ids_and_keeps_confirmed_features_as_context() {
    let mut provider = ScriptedImportance {
        calls: 0,
        active_history: Vec::new(),
        missing_oob: false,
        wrong_length: false,
    };
    let result = run(&mut provider).unwrap();

    assert_eq!(
        result.decisions,
        [
            FeatureDecision::Confirmed,
            FeatureDecision::Rejected,
            FeatureDecision::Confirmed,
        ]
    );
    assert_eq!(result.stop_reason, StopReason::NoTentativeFeatures);
    assert!(result.n_iter > 9);
    assert_eq!(provider.active_history[9], [0, 2]);
    assert_eq!(result.hit_history.as_ref().unwrap()[0][2], 0);
    assert!(result.hit_history.unwrap().len() == result.n_iter);
}

#[test]
fn aggressive_loop_removes_confirmed_features_from_later_training_views() {
    let mut provider = ScriptedImportance {
        calls: 0,
        active_history: Vec::new(),
        missing_oob: false,
        wrong_length: false,
    };
    let mut cfg = config();
    cfg.pruning = PruningMode::Aggressive;
    let rows = 24;
    let values = (0..rows * 3).map(|value| value as f64).collect::<Vec<_>>();
    let matrix = MatrixView::new(&values, rows, 3, 3, 1).unwrap();
    let result = run_boruta(
        matrix,
        (0..rows).map(|row| row % 2).collect(),
        2,
        &cfg,
        &mut provider,
    )
    .unwrap();

    assert_eq!(result.decisions[0], FeatureDecision::Confirmed);
    assert_eq!(result.decisions[1], FeatureDecision::Rejected);
    assert_eq!(result.decisions[2], FeatureDecision::Confirmed);
    assert_eq!(provider.active_history[9], [2]);
}

#[test]
fn tentative_features_remain_tentative_when_the_iteration_limit_is_reached() {
    let mut provider = ScriptedImportance {
        calls: 0,
        active_history: Vec::new(),
        missing_oob: false,
        wrong_length: false,
    };
    let mut cfg = config();
    cfg.max_iter = 3;
    let rows = 24;
    let values = (0..rows * 3).map(|value| value as f64).collect::<Vec<_>>();
    let matrix = MatrixView::new(&values, rows, 3, 3, 1).unwrap();
    let result = run_boruta(
        matrix,
        (0..rows).map(|row| row % 2).collect(),
        2,
        &cfg,
        &mut provider,
    )
    .unwrap();

    assert_eq!(result.n_iter, 3);
    assert_eq!(result.stop_reason, StopReason::MaxIterations);
    assert_eq!(result.decisions, [FeatureDecision::Tentative; 3]);
}

#[test]
fn no_oob_evidence_and_malformed_scores_fail_clearly() {
    let mut no_oob = ScriptedImportance {
        calls: 0,
        active_history: Vec::new(),
        missing_oob: true,
        wrong_length: false,
    };
    assert_eq!(
        run(&mut no_oob).unwrap_err(),
        BorutaError::NoOobEvidence { iteration: 0 }
    );

    let mut malformed = ScriptedImportance {
        calls: 0,
        active_history: Vec::new(),
        missing_oob: false,
        wrong_length: true,
    };
    assert!(matches!(
        run(&mut malformed),
        Err(BorutaError::ImportanceLength { .. })
    ));
}

#[test]
fn provider_without_oob_requirement_runs_through_the_rust_controller() {
    let mut provider = NonOobImportance;
    let rows = 24;
    let values = (0..rows * 3).map(|value| value as f64).collect::<Vec<_>>();
    let matrix = MatrixView::new(&values, rows, 3, 3, 1).unwrap();
    let result = run_boruta(
        matrix,
        (0..rows).map(|row| row % 2).collect(),
        2,
        &config(),
        &mut provider,
    )
    .unwrap();
    assert_eq!(result.n_iter, 9);
    assert_eq!(result.decisions, [FeatureDecision::Rejected; 3]);
}

#[test]
fn rust_controller_sanitizes_non_finite_scores_and_counts_them() {
    let mut provider = NonFiniteImportance;
    let rows = 24;
    let values = (0..rows * 3).map(|value| value as f64).collect::<Vec<_>>();
    let matrix = MatrixView::new(&values, rows, 3, 3, 1).unwrap();
    let result = run_boruta(
        matrix,
        (0..rows).map(|row| row % 2).collect(),
        2,
        &config(),
        &mut provider,
    )
    .unwrap();

    assert_eq!(result.non_finite_importance_count, result.n_iter * 8);
    assert_eq!(result.decisions, [FeatureDecision::Rejected; 3]);
}

#[test]
fn rejects_invalid_forest_settings_and_single_class_targets_before_training() {
    let mut provider = ScriptedImportance {
        calls: 0,
        active_history: Vec::new(),
        missing_oob: false,
        wrong_length: false,
    };
    let mut invalid = config();
    invalid.max_depth = Some(0);
    let rows = 24;
    let values = (0..rows * 3).map(|value| value as f64).collect::<Vec<_>>();
    let matrix = MatrixView::new(&values, rows, 3, 3, 1).unwrap();
    assert_eq!(
        run_boruta(
            matrix.clone(),
            (0..rows).map(|row| row % 2).collect(),
            2,
            &invalid,
            &mut provider,
        )
        .unwrap_err(),
        BorutaError::InvalidConfiguration
    );

    assert_eq!(
        run_boruta(matrix, vec![0; rows], 2, &config(), &mut provider,).unwrap_err(),
        BorutaError::ClassCount
    );
    assert_eq!(provider.calls, 0);
}
