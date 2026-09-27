use bankai_forest_core::xrf::{Forest, RfInput};
use boruta_core::{MatrixForestInput, MatrixView, ShadowSpec};

#[test]
fn borrows_c_order_matrix_without_copying_and_reads_logical_coordinates() {
    let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let view = MatrixView::new(&values, 3, 2, 2, 1).unwrap();

    assert!(view.shares_storage_with(&values));
    assert_eq!(view.shape(), (3, 2));
    assert_eq!(view.get(2, 1), Some(6.0));
}

#[test]
fn reads_fortran_order_matrix_from_its_strides_without_reordering_storage() {
    let values = [1.0, 3.0, 5.0, 2.0, 4.0, 6.0];
    let view = MatrixView::new(&values, 3, 2, 1, 3).unwrap();

    assert!(view.shares_storage_with(&values));
    let mut logical_values = Vec::new();
    for row in 0..3 {
        for column in 0..2 {
            logical_values.push(view.get(row, column).unwrap());
        }
    }
    assert_eq!(logical_values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn virtual_shadows_match_materialized_permuted_columns_and_leave_source_unchanged() {
    let values = [10.0, 1.0, 20.0, 2.0, 30.0, 3.0, 40.0, 4.0];
    let original = values;
    let base = MatrixView::new(&values, 4, 2, 2, 1).unwrap();
    let augmented = base
        .with_virtual_shadows(
            &[1, 0],
            &[
                ShadowSpec {
                    source_feature: 0,
                    row_permutation: vec![2, 0, 3, 1].into(),
                },
                ShadowSpec {
                    source_feature: 1,
                    row_permutation: vec![1, 3, 0, 2].into(),
                },
            ],
        )
        .unwrap();

    assert_eq!(augmented.shape(), (4, 4));
    assert_eq!(augmented.get(2, 0), Some(3.0));
    assert_eq!(augmented.get(0, 1), Some(10.0));
    assert_eq!(augmented.get(0, 2), Some(30.0));
    assert_eq!(augmented.get(3, 2), Some(20.0));
    assert_eq!(augmented.get(0, 3), Some(2.0));
    assert_eq!(augmented.get(3, 3), Some(3.0));
    assert!(augmented.shares_storage_with(&values));
    assert_eq!(values, original);
}

#[test]
fn virtual_shadows_reject_non_permutations_and_unknown_features() {
    let values = [1.0, 2.0, 3.0, 4.0];
    let base = MatrixView::new(&values, 2, 2, 2, 1).unwrap();

    assert!(
        base.with_virtual_shadows(
            &[0],
            &[ShadowSpec {
                source_feature: 0,
                row_permutation: vec![0, 0].into(),
            }],
        )
        .is_err()
    );
    assert!(base.with_virtual_shadows(&[2], &[]).is_err());
    assert!(base.with_virtual_shadows(&[], &[]).is_err());
}

#[test]
fn generates_at_least_five_deterministic_independently_shuffled_shadows_in_rust() {
    let rows = 100;
    let values = (0..rows)
        .flat_map(|row| [row as f64, (row + 1_000) as f64])
        .collect::<Vec<_>>();
    let base = MatrixView::new(&values, rows, 2, 2, 1).unwrap();

    let first = base.with_shuffled_shadows(&[0, 1], 123).unwrap();
    let repeated = base.with_shuffled_shadows(&[0, 1], 123).unwrap();
    let other_seed = base.with_shuffled_shadows(&[0, 1], 124).unwrap();

    assert_eq!(first.shape(), (rows, 7));
    let first_shadow_values = (0..rows)
        .map(|row| first.get(row, 2).unwrap())
        .collect::<Vec<_>>();
    let repeated_shadow_values = (0..rows)
        .map(|row| repeated.get(row, 2).unwrap())
        .collect::<Vec<_>>();
    let other_seed_values = (0..rows)
        .map(|row| other_seed.get(row, 2).unwrap())
        .collect::<Vec<_>>();
    let second_shadow_values = (0..rows)
        .map(|row| first.get(row, 3).unwrap())
        .collect::<Vec<_>>();

    assert_eq!(first_shadow_values, repeated_shadow_values);
    assert_ne!(first_shadow_values, other_seed_values);
    assert_ne!(first_shadow_values, second_shadow_values);
    assert!(first.shares_storage_with(&values));
}

#[test]
fn xrf_trains_one_forest_on_borrowed_original_and_virtual_shadow_columns() {
    let rows = 96;
    let mut values = Vec::with_capacity(rows * 3);
    let mut labels = Vec::with_capacity(rows);
    for row in 0..rows {
        let label = usize::from(row % 2 == 0);
        values.extend([label as f64, ((row * 7) % 29) as f64, (row % 11) as f64]);
        labels.push(label);
    }
    let base = MatrixView::new(&values, rows, 3, 3, 1).unwrap();
    let shadows = base.with_shuffled_shadows(&[0, 1, 2], 91).unwrap();
    let input = MatrixForestInput::new(shadows, labels, 2).unwrap();
    let forest = Forest::new(&input, 32, 2, false, true, true, 97);
    let importances = forest.importance().collect::<Vec<_>>();

    assert_eq!(input.observation_count(), rows);
    assert_eq!(input.feature_count(), 8);
    assert!(input.matrix().shares_storage_with(&values));
    assert!(!importances.is_empty());
    assert!(
        importances
            .iter()
            .all(|(feature, score)| *feature < input.feature_count() && score.is_finite())
    );
}

#[test]
fn borrowed_virtual_matrix_supports_multiclass_forest_training() {
    let rows = 96;
    let mut values = Vec::with_capacity(rows * 2);
    let mut labels = Vec::with_capacity(rows);
    for row in 0..rows {
        let label = row % 3;
        values.extend([label as f64, ((row * 11) % 31) as f64]);
        labels.push(label);
    }
    let base = MatrixView::new(&values, rows, 2, 2, 1).unwrap();
    let augmented = base.with_shuffled_shadows(&[0, 1], 73).unwrap();
    let input = MatrixForestInput::new(augmented, labels, 3).unwrap();
    let forest = Forest::new(&input, 18, 2, false, true, true, 29);

    assert_eq!(input.feature_count(), 7);
    assert!(
        forest
            .importance()
            .all(|(feature, score)| feature < input.feature_count() && score.is_finite())
    );
}

#[test]
fn float32_input_remains_borrowed_while_the_forest_trains() {
    let rows = 48;
    let mut values = Vec::with_capacity(rows * 2);
    let mut labels = Vec::with_capacity(rows);
    for row in 0..rows {
        let label = row % 2;
        values.extend([label as f32, ((row * 9) % 17) as f32]);
        labels.push(label);
    }
    let base = MatrixView::new(&values, rows, 2, 2, 1).unwrap();
    let augmented = base.with_shuffled_shadows(&[0, 1], 81).unwrap();
    let input = MatrixForestInput::new(augmented, labels, 2).unwrap();
    let forest = Forest::new(&input, 16, 2, false, true, true, 101);

    assert!(input.matrix().shares_storage_with(&values));
    assert!(forest.importance().all(|(_, score)| score.is_finite()));
}

#[test]
fn parallel_forest_can_read_a_borrowed_matrix_without_changing_importance() {
    let rows = 72;
    let mut values = Vec::with_capacity(rows * 2);
    let mut labels = Vec::with_capacity(rows);
    for row in 0..rows {
        let label = row % 2;
        values.extend([label as f64, ((row * 11) % 23) as f64]);
        labels.push(label);
    }
    let base = MatrixView::new(&values, rows, 2, 2, 1).unwrap();
    let augmented = base.with_shuffled_shadows(&[0, 1], 83).unwrap();
    let input = MatrixForestInput::new(augmented, labels, 2).unwrap();
    let serial = Forest::new(&input, 20, 2, false, true, true, 103)
        .importance()
        .collect::<std::collections::HashMap<_, _>>();
    let parallel = Forest::new_parallel(&input, 20, 2, false, true, true, 103, 3)
        .importance()
        .collect::<std::collections::HashMap<_, _>>();

    assert_eq!(serial.len(), parallel.len());
    for (feature, expected) in serial {
        assert!((parallel[&feature] - expected).abs() < 1e-12);
    }
}

#[test]
fn virtual_shadow_forest_matches_a_materialized_augmented_matrix() {
    let rows = 72;
    let mut values = Vec::with_capacity(rows * 3);
    let mut labels = Vec::with_capacity(rows);
    for row in 0..rows {
        let label = usize::from(row % 7 < 3);
        values.extend([
            label as f64,
            ((row * 5) % 17) as f64,
            ((row * 11) % 23) as f64,
        ]);
        labels.push(label);
    }
    let base = MatrixView::new(&values, rows, 3, 3, 1).unwrap();
    let virtual_view = base
        .with_virtual_shadows(
            &[0, 1],
            &[
                ShadowSpec {
                    source_feature: 0,
                    row_permutation: (0..rows).rev().collect::<Vec<_>>().into(),
                },
                ShadowSpec {
                    source_feature: 1,
                    row_permutation: (0..rows)
                        .map(|row| (row + 17) % rows)
                        .collect::<Vec<_>>()
                        .into(),
                },
            ],
        )
        .unwrap();
    let mut materialized = Vec::with_capacity(rows * 4);
    for row in 0..rows {
        for feature in 0..4 {
            materialized.push(virtual_view.get(row, feature).unwrap());
        }
    }
    let virtual_input = MatrixForestInput::new(virtual_view, labels.clone(), 2).unwrap();
    let owned_input = MatrixForestInput::new(
        MatrixView::new(&materialized, rows, 4, 4, 1).unwrap(),
        labels,
        2,
    )
    .unwrap();
    let virtual_forest = Forest::new(&virtual_input, 28, 2, false, true, true, 55);
    let owned_forest = Forest::new(&owned_input, 28, 2, false, true, true, 55);
    let virtual_importance = virtual_forest
        .importance()
        .collect::<std::collections::HashMap<_, _>>();
    let owned_importance = owned_forest
        .importance()
        .collect::<std::collections::HashMap<_, _>>();

    assert_eq!(virtual_importance.len(), owned_importance.len());
    for (feature, expected) in owned_importance {
        let actual = virtual_importance[&feature];
        assert!(
            (actual - expected).abs() < 1e-12,
            "feature {feature}: {actual} != {expected}"
        );
    }
}

#[test]
fn matrix_forest_input_rejects_mismatched_or_invalid_targets() {
    let values = [0.0, 1.0, 1.0, 0.0];
    let base = MatrixView::new(&values, 2, 2, 2, 1).unwrap();

    assert!(MatrixForestInput::new(base.clone(), vec![0], 2).is_err());
    assert!(MatrixForestInput::new(base.clone(), vec![0, 1], 1).is_err());
    assert!(MatrixForestInput::new(base, vec![0, 2], 2).is_err());
}

#[test]
fn rejects_shapes_and_strides_that_cannot_address_the_buffer() {
    let values = [1.0, 2.0, 3.0, 4.0];

    assert!(MatrixView::new(&values, 0, 2, 2, 1).is_err());
    assert!(MatrixView::new(&values, 2, 0, 2, 1).is_err());
    assert!(MatrixView::new(&values, 2, 2, 3, 1).is_err());
    assert!(MatrixView::new(&values, usize::MAX, 2, usize::MAX, 1).is_err());
    let values_with_nan = [1.0, f64::NAN, 3.0, 4.0];
    assert!(MatrixView::new(&values_with_nan, 2, 2, 2, 1).is_err());
}
