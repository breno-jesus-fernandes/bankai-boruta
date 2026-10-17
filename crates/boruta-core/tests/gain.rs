use boruta_core::gain_importances;
#[test]
fn gain_is_normalized_and_reproducible() {
    let run = |threads, max_features| {
        gain_importances(
            50,
            2,
            (0..50).map(|r| usize::from(r > 24)).collect(),
            10,
            6,
            42,
            16,
            6,
            threads,
            max_features,
            |row, col| {
                if col == 0 {
                    row as f64
                } else {
                    (row % 3) as f64
                }
            },
        )
        .unwrap()
    };
    let a = run(1, 1);
    let b = run(4, 1);
    for (a, b) in a.importances.iter().zip(&b.importances) {
        assert!((a - b).abs() < 1e-12);
    }
    assert!((a.importances.iter().sum::<f64>() - 1.).abs() < 1e-12);
    assert!(a.importances[0] > a.importances[1]);
}
