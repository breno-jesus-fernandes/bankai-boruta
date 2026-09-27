use boruta_core::{
    BorutaConfig, ImportanceProvider, ImportanceScale, MatrixView, MaxFeatures,
    PermutationImportanceProvider, PruningMode,
};
use std::sync::Arc;

fn config(scale: ImportanceScale) -> BorutaConfig {
    BorutaConfig {
        n_estimators: 48,
        max_iter: 8,
        alpha: 0.05,
        pruning: PruningMode::Canonical,
        max_features: MaxFeatures::Sqrt,
        max_depth: Some(24),
        n_jobs: 1,
        random_state: 1234,
        importance_scale: scale,
        store_history: false,
    }
}

#[test]
fn native_oob_importance_scores_real_and_virtual_shadow_columns_deterministically() {
    let rows = 96;
    let mut values = Vec::with_capacity(rows * 2);
    let mut labels = Vec::with_capacity(rows);
    for row in 0..rows {
        let label = row % 2;
        values.extend([label as f64, ((row * 19) % 37) as f64]);
        labels.push(label);
    }
    let matrix = MatrixView::new(&values, rows, 2, 2, 1).unwrap();
    let labels = Arc::new(labels);
    let active = [0, 1];
    for scale in [ImportanceScale::Raw, ImportanceScale::ZScore] {
        let cfg = config(scale);
        let mut first_provider = PermutationImportanceProvider;
        let first = first_provider
            .compute(
                matrix.clone(),
                Arc::clone(&labels),
                2,
                &active,
                &cfg,
                cfg.random_state,
            )
            .unwrap();
        let mut second_provider = PermutationImportanceProvider;
        let second = second_provider
            .compute(
                matrix.clone(),
                Arc::clone(&labels),
                2,
                &active,
                &cfg,
                cfg.random_state,
            )
            .unwrap();

        assert!(first.1, "the forest must produce OOB evidence");
        assert_eq!(first.0.len(), 7, "two real features and five shadows");
        assert!(first.0.iter().all(|score| score.is_finite()));
        assert_eq!(first.0, second.0);
    }
}

#[test]
fn parallel_forest_produces_the_same_seeded_importances_as_serial_training() {
    let rows = 120;
    let mut values = Vec::with_capacity(rows * 3);
    let mut labels = Vec::with_capacity(rows);
    for row in 0..rows {
        let label = (row * 7 % 19) > 8;
        values.extend([
            f64::from(label),
            ((row * 13) % 29) as f64,
            ((row * row) % 31) as f64,
        ]);
        labels.push(usize::from(label));
    }
    let matrix = MatrixView::new(&values, rows, 3, 3, 1).unwrap();
    let labels = Arc::new(labels);
    let active = [0, 1, 2];
    let mut serial = config(ImportanceScale::ZScore);
    serial.n_estimators = 40;
    let mut parallel = serial.clone();
    parallel.n_jobs = 4;
    let mut serial_provider = PermutationImportanceProvider;
    let serial_scores = serial_provider
        .compute(
            matrix.clone(),
            Arc::clone(&labels),
            2,
            &active,
            &serial,
            serial.random_state,
        )
        .unwrap()
        .0;
    let mut parallel_provider = PermutationImportanceProvider;
    let parallel_scores = parallel_provider
        .compute(
            matrix.clone(),
            Arc::clone(&labels),
            2,
            &active,
            &parallel,
            parallel.random_state,
        )
        .unwrap()
        .0;
    let mut repeat_provider = PermutationImportanceProvider;
    let repeated_scores = repeat_provider
        .compute(matrix, labels, 2, &active, &parallel, parallel.random_state)
        .unwrap()
        .0;

    assert_eq!(parallel_scores, repeated_scores);
    for (parallel, serial) in parallel_scores.iter().zip(serial_scores) {
        assert!((parallel - serial).abs() < 1e-12);
    }
}
