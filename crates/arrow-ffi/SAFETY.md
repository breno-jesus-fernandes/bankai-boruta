# Arrow FFI boundary

This crate is the sole local unsafe boundary for Arrow ingress. The forest core
and primary PyO3 extension continue to forbid unsafe Rust.

The first prototype used pyo3-arrow 0.19 (compatible with PyO3 0.29). A test with
an unaligned but valid PyArrow values buffer showed that arrow-rs repairs the
alignment by copying. The strict importer therefore needs a preflight before
`from_ffi_and_data_type`; it rejects unaligned values rather than reporting a
copy as zero-copy. arrow-array/arrow-schema are pinned together to 60.0.0 in the
manifests and lockfile, and Dependabot groups Arrow updates.

Arrow 60 makes its Rust stream callback fields private. The importer therefore
uses a local `#[repr(C)]` handle for the published Arrow C Stream ABI, rather
than accessing private arrow-rs fields. A move replaces the producer's handle
with an empty released handle; the consumer calls its producer-supplied release
callback exactly once on drop. It neither reads nor frees `private_data`.
Schemas and value-buffer ownership still use arrow-rs RAII types; no handwritten
value-buffer deallocator is used. This preserves preflight alignment checks
before arrow-rs can repair an unaligned buffer by copying it.

The boundary assumes a conforming Arrow C Stream producer: correctly named
capsules hold valid pointers, callbacks follow the ABI, lengths describe
accessible buffers, and release callbacks own their resources. As with other
native Arrow consumers, capsule names alone cannot validate arbitrary forged
native pointers. Tests use valid producers, including valid unaligned buffers,
plus explicit error callbacks; they never dereference fabricated pointers.

Import validates capsule kind, single consumption, callback presence, numeric
schema, buffer counts, child shape, offset arithmetic and natural alignment.
Only Float32/Float64 and a top-level struct of these types are supported;
extensions, dictionaries, nulls and non-finite values are rejected. Schema
and array cleanup use arrow-rs RAII types on every return path. The local
stream handle runs the producer's release callback on success and all error paths.

The GIL is held while invoking exporters/callbacks. Polars DataFrame exports can
rechunk; importing its individual Series streams preserves independent chunks.
Callbacks finish before training begins. Native workers read immutable owned
Arrow handles; the main fit retains its own handle until all workers join.
Final owner release therefore happens on the calling thread under the GIL.
Do not release the GIL or move destruction into background tasks without a new
audit of Python-backed callbacks and ownership.

Tests cover moved-from exporter handles, missing callbacks, stream destruction
once, source buffer reference counts returning to baseline on success and partial
failure, schema callback failure, buffer address identity, offsets and independent chunk boundaries. Python tests also
cover reusing a consumed capsule, error propagation, weakrefs, repeated-fit RSS,
and serialized model size. Miri/ASan evidence and platform limits belong in the
M6 validation report; neither is a substitute for the producer contract.
