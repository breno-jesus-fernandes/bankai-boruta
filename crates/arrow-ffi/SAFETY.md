# Arrow FFI boundary

This crate is the sole local unsafe boundary for Arrow ingress. The forest core
and primary PyO3 extension continue to forbid unsafe Rust.

The first prototype used pyo3-arrow 0.19 (compatible with PyO3 0.29). A test with
an unaligned but valid PyArrow values buffer showed that arrow-rs repairs the
alignment by copying. The strict importer therefore needs a preflight before
`from_ffi_and_data_type`; it rejects unaligned values rather than reporting a
copy as zero-copy. arrow-array/arrow-schema are pinned to 59.3.0 in the manifest
and lockfile. No handwritten value-buffer ownership or deallocator is used.

The boundary assumes a conforming Arrow C Stream producer: correctly named
capsules hold valid pointers, callbacks follow the ABI, lengths describe
accessible buffers, and release callbacks own their resources. As with other
native Arrow consumers, capsule names alone cannot validate arbitrary forged
native pointers. Tests use valid producers, including valid unaligned buffers,
plus explicit error callbacks; they never dereference fabricated pointers.

Import validates capsule kind, single consumption, callback presence, numeric
schema, buffer counts, child shape, offset arithmetic and natural alignment.
Only Float32/Float64 and a top-level struct of these types are supported;
extensions, dictionaries, nulls and non-finite values are rejected. Schema,
stream and array cleanup use arrow-rs RAII types on every return path.

The GIL is held while invoking exporters/callbacks. Polars DataFrame exports can
rechunk; importing its individual Series streams preserves independent chunks.
Callbacks finish before training begins. Native workers read immutable owned
Arrow handles; the main fit retains its own handle until all workers join.
Final owner release therefore happens on the calling thread under the GIL.
Do not release the GIL or move destruction into background tasks without a new
audit of Python-backed callbacks and ownership.

Tests cover stream destruction once, source buffer reference counts returning
to baseline on success and partial failure, schema callback failure, buffer
address identity, offsets and independent chunk boundaries. Python tests also
cover reusing a consumed capsule, error propagation, weakrefs, repeated-fit RSS,
and serialized model size. Miri/ASan evidence and platform limits belong in the
M6 validation report; neither is a substitute for the producer contract.
