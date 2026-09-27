use bankai_forest_core::{
    Criterion, DenseInput,
    xrf::{Forest, RfInput},
};

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
    assert!(importances
        .iter()
        .all(|(feature, score)| *feature < 2 && score.is_finite()));
}
