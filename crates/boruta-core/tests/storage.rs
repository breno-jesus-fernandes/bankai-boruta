use bankai_forest_core::xrf::Forest;
use boruta_core::{FeatureStorage, LogicalMatrix, MatrixForestInput, MatrixView};

#[derive(Clone)]
struct Columns<'a>(&'a [&'a [f64]]);
impl FeatureStorage for Columns<'_> {
    fn shape(&self) -> (usize, usize) {
        (self.0[0].len(), self.0.len())
    }
    fn value(&self, row: usize, column: usize) -> Option<f64> {
        self.0.get(column)?.get(row).copied()
    }
}

#[test]
fn columnar_storage_preserves_shadows_and_seeded_forest() {
    let a = (0..48).map(|i| (i % 2) as f64).collect::<Vec<_>>();
    let b = (0..48).map(|i| (i % 7) as f64).collect::<Vec<_>>();
    let columns = [a.as_slice(), b.as_slice()];
    let dense = a
        .iter()
        .zip(&b)
        .flat_map(|(&a, &b)| [a, b])
        .collect::<Vec<_>>();
    let columnar = LogicalMatrix::from_storage(Columns(&columns))
        .unwrap()
        .with_shuffled_shadows(&[1, 0], 13)
        .unwrap();
    let strided = MatrixView::new(&dense, 48, 2, 2, 1)
        .unwrap()
        .with_shuffled_shadows(&[1, 0], 13)
        .unwrap();
    for row in 0..48 {
        for col in 0..7 {
            assert_eq!(columnar.get(row, col), strided.get(row, col));
        }
    }
    let labels = (0..48).map(|i| i % 2).collect::<Vec<_>>();
    let c = MatrixForestInput::new(columnar, labels.clone(), 2).unwrap();
    let d = MatrixForestInput::new(strided, labels, 2).unwrap();
    let f = Forest::new(&c, 32, 2, true, true, true, 11);
    let g = Forest::new(&d, 32, 2, true, true, true, 11);
    let mut fi = f.importance().collect::<Vec<_>>();
    let mut gi = g.importance().collect::<Vec<_>>();
    fi.sort_by_key(|v| v.0);
    gi.sort_by_key(|v| v.0);
    assert_eq!(fi, gi);
}

#[test]
fn rejects_nonfinite_custom_storage_and_empty_shape() {
    let values = [f64::NAN];
    let columns = [values.as_slice()];
    assert!(LogicalMatrix::from_storage(Columns(&columns)).is_err());
    let values = [];
    let columns = [values.as_slice()];
    assert!(LogicalMatrix::from_storage(Columns(&columns)).is_err());
}
