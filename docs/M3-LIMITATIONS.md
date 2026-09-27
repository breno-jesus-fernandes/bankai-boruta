# M3 limitation: exact forest parity on tied splits

The borrowed strided matrix adapter and virtual shadows are implemented, and
the focused matrix-view tests cover C/F layouts, f32/f64, deterministic shadow
permutations, and training through XRF. The Python ingress probe also verifies
that supported NumPy arrays are borrowed without copying.

The M3 differential test exposed a remaining mismatch between the borrowed
adapter and `DenseInput` from `bankai-forest-core`: on data with repeated values
and tied Gini gains, a small floating-point difference can select a different
threshold. A later leaf-vote difference follows from that tree divergence.
An epsilon tie rule improves the selected threshold but does not resolve every
case, so it is not acceptable as a fix. The parity test remains active and
failing; M3 is not complete and must not be treated as a release checkpoint.

Before continuing to Boruta orchestration, factor the exact weighted splitter
into the shared core or change the adapter to delegate to that implementation.
Then expand parity coverage to duplicated values, bootstrap masks, multiclass
labels, and OOB importance. Do not weaken or ignore the differential test.

Memory remains bounded by the caller-owned matrix, `O(n * shadows)` row
permutation indices, and per-tree workspaces. The logical matrix view itself
does not materialize shadows or clone the source matrix.
