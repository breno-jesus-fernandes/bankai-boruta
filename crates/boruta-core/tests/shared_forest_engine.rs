use bankai_forest_core::{
    Criterion, DenseInput,
    xrf::{Forest, RfInput, Walk},
};
use boruta_core::{MatrixForestInput, MatrixView};

#[test]
fn boruta_core_can_train_and_read_importance_without_a_python_boundary() {
    let rows = 48;
    let mut values = Vec::with_capacity(rows * 2);
    let mut labels = Vec::with_capacity(rows);
    for row in 0..rows {
        let label = row % 2;
        values.extend([label as f64, ((row * 13) % 19) as f64]);
        labels.push(label);
    }
    let input = DenseInput::training(
        values,
        rows,
        2,
        labels,
        vec![1.0; rows],
        2,
        0.0,
        Criterion::Gini,
        2,
        1,
        0.0,
        None,
    )
    .unwrap();

    assert_eq!(input.feature_count(), 2);
    let forest = Forest::new(&input, 20, 2, false, true, true, 13);
    let importances = forest.importance().collect::<Vec<_>>();

    assert!(!importances.is_empty());
    assert!(
        importances
            .iter()
            .all(|(feature, score)| *feature < 2 && score.is_finite())
    );
}

#[test]
fn borrowed_matrix_input_matches_shared_dense_core_forest_importance() {
    let rows = 80;
    let mut values = Vec::with_capacity(rows * 3);
    let mut labels = Vec::with_capacity(rows);
    for row in 0..rows {
        let label = usize::from((row * 17 % 31) > 15);
        values.extend([
            label as f64,
            ((row * 13) % 23) as f64,
            ((row * 7) % 19) as f64,
        ]);
        labels.push(label);
    }
    let borrowed_view = MatrixView::new(&values, rows, 3, 3, 1).unwrap();
    let borrowed = MatrixForestInput::new(borrowed_view, labels.clone(), 2).unwrap();
    let owned_values = values.clone();
    let owned = DenseInput::training(
        owned_values,
        rows,
        3,
        labels,
        vec![1.0; rows],
        2,
        0.0,
        Criterion::Gini,
        2,
        1,
        0.0,
        None,
    )
    .unwrap();
    let borrowed_forest = Forest::new(&borrowed, 32, 2, false, true, true, 101);
    let owned_forest = Forest::new(&owned, 32, 2, false, true, true, 101);
    let borrowed_importance = borrowed_forest
        .importance()
        .collect::<std::collections::HashMap<_, _>>();
    let owned_importance = owned_forest
        .importance()
        .collect::<std::collections::HashMap<_, _>>();

    assert_eq!(borrowed_importance.len(), owned_importance.len());
    for (feature, expected) in owned_importance {
        let actual = borrowed_importance[&feature];
        assert!(
            (actual - expected).abs() < 1e-12,
            "feature {feature}: {actual} != {expected}"
        );
    }
}

#[test]
fn borrowed_and_owned_forest_importance_match_across_seeds_and_class_counts() {
    for n_classes in [2, 3] {
        for seed in 0..8_u64 {
            let rows = 72;
            let mut values = Vec::with_capacity(rows * 4);
            let mut labels = Vec::with_capacity(rows);
            for row in 0..rows {
                let label = ((row * 5 + seed as usize / 2) % n_classes) as usize;
                values.extend([
                    label as f64,
                    ((row * 7 + seed as usize) % 17) as f64,
                    ((row * 11 + seed as usize * 3) % 23) as f64,
                    ((row * row + seed as usize) % 29) as f64,
                ]);
                labels.push(label);
            }
            let borrowed_view = MatrixView::new(&values, rows, 4, 4, 1).unwrap();
            let borrowed =
                MatrixForestInput::new(borrowed_view, labels.clone(), n_classes).unwrap();
            let owned = DenseInput::training(
                values.clone(),
                rows,
                4,
                labels,
                vec![1.0; rows],
                n_classes,
                0.0,
                Criterion::Gini,
                2,
                1,
                0.0,
                None,
            )
            .unwrap();
            let borrowed_forest = Forest::new(&borrowed, 16, 2, true, true, true, seed);
            let owned_forest = Forest::new(&owned, 16, 2, true, true, true, seed);
            let borrowed_walk = borrowed_forest
                .walk()
                .map(|entry| match entry {
                    Walk::VisitBranch(feature, pivot, score) => (0, feature, pivot, score),
                    Walk::VisitLeaf(vote) => (1, vote, 0.0, 0.0),
                })
                .collect::<Vec<_>>();
            let owned_walk = owned_forest
                .walk()
                .map(|entry| match entry {
                    Walk::VisitBranch(feature, pivot, score) => (0, feature, pivot, score),
                    Walk::VisitLeaf(vote) => (1, vote, 0.0, 0.0),
                })
                .collect::<Vec<_>>();
            for (index, (actual, expected)) in borrowed_walk.iter().zip(&owned_walk).enumerate() {
                assert_eq!(
                    (actual.0, actual.1, actual.2),
                    (expected.0, expected.1, expected.2),
                    "walk structure differs at {index} for classes={n_classes}, seed={seed}"
                );
                assert!(
                    (actual.3 - expected.3).abs() < 1e-12,
                    "walk score differs at {index} for classes={n_classes}, seed={seed}: {} != {}",
                    actual.3,
                    expected.3
                );
            }
            assert_eq!(borrowed_walk.len(), owned_walk.len());
            let borrowed_importance = borrowed_forest
                .importance()
                .collect::<std::collections::HashMap<_, _>>();
            let owned_importance = owned_forest
                .importance()
                .collect::<std::collections::HashMap<_, _>>();

            assert_eq!(borrowed_importance.len(), owned_importance.len());
            for (feature, expected) in owned_importance {
                let actual = borrowed_importance[&feature];
                assert!(
                    (actual - expected).abs() < 1e-12,
                    "classes={n_classes}, seed={seed}, feature={feature}: {actual} != {expected}"
                );
            }
        }
    }
}
