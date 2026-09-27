//! Safe, shared Arrow storage. FFI ownership is handled by pyo3-arrow/arrow-rs.
use arrow_array::{Array, ArrayRef, Float32Array, Float64Array, RecordBatch};
use arrow_schema::{DataType, SchemaRef};
use boruta_core::FeatureStorage;
use std::collections::HashSet;
use std::sync::Arc;

#[derive(Clone, Debug)]
enum Chunk {
    F32(Float32Array),
    F64(Float64Array),
}
impl Chunk {
    fn value(&self, row: usize) -> f64 {
        match self {
            Self::F32(a) => a.value(row) as f64,
            Self::F64(a) => a.value(row),
        }
    }
    fn buffer(&self) -> (usize, usize) {
        match self {
            Self::F32(a) => (a.values().as_ptr() as usize, a.len()),
            Self::F64(a) => (a.values().as_ptr() as usize, a.len()),
        }
    }
}

#[derive(Clone, Debug)]
struct Column {
    chunks: Vec<Chunk>,
    ends: Vec<usize>,
}
#[derive(Clone, Debug)]
pub(crate) struct ArrowStorage {
    columns: Arc<Vec<Column>>,
    rows: usize,
    pub(crate) names: Vec<String>,
}
impl ArrowStorage {
    pub(crate) fn new(batches: Vec<RecordBatch>, schema: SchemaRef) -> Result<Self, String> {
        for field in schema.fields() {
            if !matches!(field.data_type(), DataType::Float32 | DataType::Float64)
                || field.metadata().contains_key("ARROW:extension:name")
            {
                return Err("Arrow zero_copy requires Float32 or Float64 columns".into());
            }
        }
        let mut columns = schema
            .fields()
            .iter()
            .map(|f| (f.name().clone(), Vec::new()))
            .collect::<Vec<_>>();
        for batch in batches {
            if batch.schema() != schema {
                return Err("Arrow batches must have the same schema".into());
            }
            for (col, array) in columns.iter_mut().zip(batch.columns()) {
                col.1.push(array.clone());
            }
        }
        Self::from_columns(columns)
    }
    pub(crate) fn from_columns(input: Vec<(String, Vec<ArrayRef>)>) -> Result<Self, String> {
        let mut names = Vec::new();
        let mut seen = HashSet::new();
        let mut columns = Vec::new();
        let mut total_rows = None;
        for (name, arrays) in input {
            if !seen.insert(name.clone()) {
                return Err("Arrow feature names must be unique".into());
            }
            let mut chunks = Vec::new();
            let mut ends = Vec::new();
            let mut rows = 0usize;
            for array in arrays {
                if array.null_count() != 0 {
                    return Err(format!("Arrow column {name} contains null values"));
                }
                if array.is_empty() {
                    continue;
                }
                rows = rows
                    .checked_add(array.len())
                    .ok_or("Arrow row count overflow")?;
                let chunk = if let Some(a) = array.as_any().downcast_ref::<Float32Array>() {
                    Chunk::F32(a.clone())
                } else if let Some(a) = array.as_any().downcast_ref::<Float64Array>() {
                    Chunk::F64(a.clone())
                } else {
                    return Err("Arrow zero_copy requires Float32 or Float64 columns".into());
                };
                chunks.push(chunk);
                ends.push(rows);
            }
            if rows == 0 {
                return Err("Arrow input must have at least one row".into());
            }
            if total_rows.is_some_and(|n| n != rows) {
                return Err("Arrow columns have inconsistent row counts".into());
            }
            total_rows = Some(rows);
            names.push(name);
            columns.push(Column { chunks, ends });
        }
        let rows = total_rows.ok_or("Arrow input must have at least one feature")?;
        Ok(Self {
            columns: Arc::new(columns),
            rows,
            names,
        })
    }
    pub(crate) fn buffers(&self) -> Vec<Vec<(usize, usize)>> {
        self.columns
            .iter()
            .map(|column| column.chunks.iter().map(Chunk::buffer).collect())
            .collect()
    }
}
impl FeatureStorage for ArrowStorage {
    fn shape(&self) -> (usize, usize) {
        (self.rows, self.columns.len())
    }
    fn value(&self, row: usize, column: usize) -> Option<f64> {
        if row >= self.rows {
            return None;
        }
        let col = self.columns.get(column)?;
        if col.chunks.len() == 1 {
            return Some(col.chunks[0].value(row));
        }
        let chunk = col.ends.partition_point(|&end| end <= row);
        let start = if chunk == 0 { 0 } else { col.ends[chunk - 1] };
        Some(col.chunks[chunk].value(row - start))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{ArrayRef, Float32Array, Float64Array, Int64Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use boruta_core::{FeatureStorage, LogicalMatrix};
    use std::sync::Arc;

    #[test]
    fn mixed_chunks_offsets_and_clones_share_buffers() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Float32, false),
            Field::new("b", DataType::Float64, false),
        ]));
        let a = Arc::new(Float32Array::from(vec![9., 1., 2., 3.])) as ArrayRef;
        let b = Arc::new(Float64Array::from(vec![8., 4., 5., 6.])) as ArrayRef;
        let first =
            RecordBatch::try_new(schema.clone(), vec![a.slice(1, 1), b.slice(1, 1)]).unwrap();
        let second =
            RecordBatch::try_new(schema.clone(), vec![a.slice(2, 2), b.slice(2, 2)]).unwrap();
        let storage = ArrowStorage::new(vec![first, second], schema).unwrap();
        let expected = a
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap()
            .values()
            .as_ptr()
            .wrapping_add(1) as usize;
        assert_eq!(storage.buffers()[0][0], (expected, 1));
        assert_eq!(storage.shape(), (3, 2));
        assert_eq!(storage.value(1, 0), Some(2.));
        assert_eq!(storage.value(2, 1), Some(6.));
        assert_eq!(storage.value(3, 0), None);
        let clone = storage.clone();
        drop(storage);
        drop(a);
        drop(b);
        assert_eq!(clone.value(0, 1), Some(4.));
        assert_eq!(clone.buffers()[0][0].0, expected);
    }

    #[test]
    fn skips_empty_chunks_and_rejects_inconsistent_column_lengths() {
        let empty = Arc::new(Float64Array::from(Vec::<f64>::new())) as ArrayRef;
        let a = Arc::new(Float64Array::from(vec![1., 2.])) as ArrayRef;
        let b = Arc::new(Float32Array::from(vec![3.])) as ArrayRef;
        let storage =
            ArrowStorage::from_columns(vec![("a".into(), vec![empty.clone(), a.clone(), empty])])
                .unwrap();
        assert_eq!(storage.buffers()[0].len(), 1);
        assert_eq!(storage.value(1, 0), Some(2.));
        assert!(
            ArrowStorage::from_columns(vec![("a".into(), vec![a]), ("b".into(), vec![b])]).is_err()
        );
    }

    #[test]
    fn rejects_nulls_unsupported_types_empty_and_nonfinite() {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Float64, true)]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Float64Array::from(vec![Some(1.), None]))],
        )
        .unwrap();
        assert!(
            ArrowStorage::new(vec![batch], schema.clone())
                .unwrap_err()
                .contains("null")
        );
        assert!(ArrowStorage::new(vec![], schema.clone()).is_err());
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Float64Array::from(vec![f64::INFINITY]))],
        )
        .unwrap();
        assert!(
            LogicalMatrix::from_storage(ArrowStorage::new(vec![batch], schema).unwrap()).is_err()
        );
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(Int64Array::from(vec![1]))])
            .unwrap();
        assert!(
            ArrowStorage::new(vec![batch], schema)
                .unwrap_err()
                .contains("Float32")
        );
    }
}
