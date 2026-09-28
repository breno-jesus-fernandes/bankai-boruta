"""Arrow ingress contracts: value buffers stay shared, training remains native."""
import gc
import pickle
import weakref

import numpy as np
import pytest

from bankai_boruta import BankaiBoruta, _core

pl = pytest.importorskip('polars')


def frame(chunks=1):
    rng = np.random.default_rng(32)
    y = np.tile([0, 1], 48)
    X = pl.DataFrame({'signal': y.astype('float32'), 'noise': rng.normal(size=y.size)})
    # Different chunk boundaries plus nonzero offsets, mixed f32/f64.
    X = pl.concat([X.slice(i, len(X)//chunks) for i in range(0, len(X), len(X)//chunks)], rechunk=False)
    return X.slice(1, len(X)-2), y[1:-1]


@pytest.mark.parametrize('chunks', [1, 8])
@pytest.mark.parametrize('jobs', [1, 4])
def test_polars_uses_native_loop_without_numpy_conversion(monkeypatch, chunks, jobs):
    X, y = frame(chunks)
    values = np.asarray(X)
    options = dict(n_estimators=24, max_iter=8, random_state=7, store_history=True, n_jobs=jobs)
    expected = BankaiBoruta(**options).fit(values, y)
    def forbidden(*args, **kwargs):
        raise AssertionError('Arrow input was converted to NumPy')
    monkeypatch.setattr(pl.DataFrame, 'to_numpy', forbidden)
    monkeypatch.setattr(pl.DataFrame, '__array__', forbidden)
    calls = []
    original = _core.fit_boruta
    def counted(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(_core, 'fit_boruta', counted)
    actual = BankaiBoruta(**options).fit(X, y)
    assert len(calls) == 1
    np.testing.assert_array_equal(actual.decision_, expected.decision_)
    assert actual.history_ == expected.history_
    assert actual.diagnostics_['copied_bytes'] == 0
    assert actual.feature_names_in_.tolist() == ['signal', 'noise']


def test_arrow_buffer_addresses_match_polars_slices():
    X, _ = frame(1)
    # Series zero-copy exports used only as an independent pointer oracle.
    expected = [s.to_numpy(allow_copy=False).__array_interface__['data'][0] for s in X]
    observed = _core._inspect_arrow_buffers(X)
    assert observed['rows'] == X.height
    assert [col[0][0] for col in observed['buffers']] == expected


@pytest.mark.parametrize('series, message', [
    (pl.Series('bad', [1., None]), 'null'),
    (pl.Series('bad', [1., float('nan')]), 'finite'),
    (pl.Series('bad', [1., float('inf')]), 'finite'),
    (pl.Series('bad', [1, 2]), 'Float32'),
    (pl.Series('bad', ['a', 'b']), 'Float32'),
])
def test_rejects_unsupported_arrow_inputs(series, message):
    with pytest.raises(ValueError, match=message):
        BankaiBoruta(n_estimators=4, max_iter=1).fit(pl.DataFrame(series), [0, 1])


def test_fit_does_not_retain_polars_owner_and_pickle_roundtrip():
    X, y = frame()
    ref = weakref.ref(X)
    fitted = BankaiBoruta(n_estimators=24, max_iter=8, random_state=7).fit(X, y)
    del X
    gc.collect()
    assert ref() is None
    restored = pickle.loads(pickle.dumps(fitted))
    np.testing.assert_array_equal(fitted.decision_, restored.decision_)
    assert restored.feature_names_in_.tolist() == ['signal', 'noise']


def test_external_estimator_materializes_only_active_arrow_columns_per_iteration():
    from sklearn.tree import DecisionTreeClassifier

    X, y = frame(8)
    rows = X.height
    selector = BankaiBoruta(
        estimator=DecisionTreeClassifier(max_depth=3, random_state=5),
        n_estimators=8,
        max_iter=2,
        random_state=6,
        store_history=True,
    ).fit(X, y)

    expected_bytes = sum(
        rows * (active + max(active, 5)) * 8
        for active in selector.diagnostics_["active_features_per_iteration"]
    )
    assert selector.diagnostics_["external_estimator_calls"] == selector.n_iter_
    assert selector.diagnostics_["copied_bytes"] == expected_bytes
    assert selector.feature_names_in_.tolist() == ["signal", "noise"]


def test_failed_refit_clears_previous_fitted_state():
    X, y = frame()
    model = BankaiBoruta(n_estimators=4, max_iter=1).fit(X, y)
    with pytest.raises(ValueError):
        model.fit(pl.DataFrame({'bad': [1., None]}), [0, 1])
    assert not hasattr(model, 'support_')
    assert not hasattr(model, 'feature_names_in_')


def test_unaligned_arrow_is_rejected_instead_of_silently_copied():
    pa = pytest.importorskip('pyarrow')
    raw = np.zeros(1 + 8 * 4, dtype=np.uint8)
    values = np.ndarray((4,), dtype=np.float64, buffer=raw, offset=1)
    values[:] = [0, 1, 0, 1]
    array = pa.Array.from_buffers(pa.float64(), 4, [None, pa.py_buffer(values)])
    table = pa.table({'a': array})
    with pytest.raises(ValueError, match='align'):
        BankaiBoruta(n_estimators=4, max_iter=1).fit(table, [0, 1, 0, 1])


def test_stream_error_after_a_batch_is_propagated():
    pa = pytest.importorskip('pyarrow')
    batch = pa.record_batch({'a': [0., 1., 0., 1.]})
    def batches():
        yield batch
        raise ValueError('deliberate stream failure')
    reader = pa.RecordBatchReader.from_batches(batch.schema, batches())
    with pytest.raises((ValueError, TypeError), match='deliberate stream failure'):
        BankaiBoruta(n_estimators=4, max_iter=1).fit(reader, [0, 1, 0, 1])


def test_differently_chunked_columns_are_not_rechunked_or_copied():
    a = pl.concat([pl.Series('a', [0., 1., 2.]), pl.Series('a', [3., 4., 5., 6.])], rechunk=False)
    b = pl.concat([pl.Series('b', [7., 8.]), pl.Series('b', [9., 10., 11., 12., 13.])], rechunk=False)
    X = pl.DataFrame([a, b]).slice(1, 5)
    originals = []
    for series in X:
        originals.append([(c.to_numpy(allow_copy=False).__array_interface__['data'][0], len(c))
                          for c in series.get_chunks()])
    imported = _core._inspect_arrow_buffers(X)['buffers']
    for col, buffers in enumerate(imported):
        assert sum(length for _, length in buffers) == X.height
        for pointer, length in buffers:
            assert any(start <= pointer and pointer + length*8 <= start+size*8
                       for start, size in originals[col])


def test_capsule_single_consumption_and_invalid_capsule_errors():
    X, y = frame()
    capsule = X.__arrow_c_stream__()
    class Once:
        def __arrow_c_stream__(self, requested_schema=None):
            return capsule
    _core._inspect_arrow_buffers(Once())
    with pytest.raises(ValueError, match='consumed|released'):
        _core._inspect_arrow_buffers(Once())
    class Invalid:
        def __arrow_c_stream__(self, requested_schema=None):
            return object()
    with pytest.raises(TypeError):
        BankaiBoruta(n_estimators=4, max_iter=1).fit(Invalid(), y)


def test_arrow_empty_null_parent_duplicate_and_extension_errors():
    pa = pytest.importorskip('pyarrow')
    from bankai_boruta import _core
    empty = pa.table({'a': pa.array([], type=pa.float64())})
    with pytest.raises(ValueError, match='row'):
        _core._inspect_arrow_buffers(empty)
    duplicates = pa.Table.from_arrays([pa.array([1.]), pa.array([2.])], names=['a', 'a'])
    with pytest.raises(ValueError, match='unique'):
        _core._inspect_arrow_buffers(duplicates)
    schema = pa.schema([pa.field('a', pa.float64(), metadata={'ARROW:extension:name': 'custom'})])
    with pytest.raises(ValueError, match='Float32'):
        _core._inspect_arrow_buffers(pa.Table.from_arrays([pa.array([1.])], schema=schema))


def test_pipeline_cross_validation_and_polars_transform_output():
    from sklearn.pipeline import make_pipeline
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    X, y = frame(8)
    selector = BankaiBoruta(n_estimators=32, max_iter=10, alpha=0.05, random_state=8)
    pipeline = make_pipeline(selector, DecisionTreeClassifier(random_state=2))
    scores = cross_val_score(pipeline, X, y, cv=StratifiedKFold(3, shuffle=True, random_state=4))
    assert min(scores) > 0.9
    assert not hasattr(selector, 'support_')  # CV fits clones within each fold.
    selector.set_output(transform='polars').fit(X, y)
    selected = selector.transform(X)
    assert isinstance(selected, pl.DataFrame)
    assert selected.columns == selector.get_feature_names_out().tolist()
    restored = selector.inverse_transform(np.asarray(selected))
    assert restored.shape == X.shape


def test_serialization_size_is_independent_of_rows_and_joblib_restores(tmp_path):
    import joblib
    sizes = []
    for n in [100, 2000]:
        y = np.tile([0, 1], n//2)
        X = pl.DataFrame({'a': y.astype(float), 'b': np.arange(n, dtype=float)})
        fitted = BankaiBoruta(n_estimators=8, max_iter=8, alpha=0.05, random_state=6).fit(X, y)
        sizes.append(len(pickle.dumps(fitted)))
        path = tmp_path / 'model.joblib'
        joblib.dump(fitted, path)
        restored = joblib.load(path)
        np.testing.assert_array_equal(restored.transform(X), fitted.transform(X))
    assert abs(sizes[1]-sizes[0]) < 256


def test_copy_mode_float_polars_materializes_only_one_matrix(monkeypatch):
    original = _core.fit_boruta
    def counted(*args):
        assert isinstance(args[0], np.ndarray)
        assert args[-3] == 'zero_copy'  # The already materialized buffer must be borrowed.
        assert args[-2] is None  # Native estimator remains the default.
        assert args[-1] == 'auto'
        return original(*args)
    monkeypatch.setattr(_core, 'fit_boruta', counted)
    X, y = frame(8)
    fitted = BankaiBoruta(n_estimators=8, max_iter=2, input_mode='copy', random_state=9).fit(X, y)
    assert fitted.diagnostics_['copied_bytes'] == X.height * X.width * 8


def test_nullable_parent_struct_is_rejected():
    pa = pytest.importorskip('pyarrow')
    parent = pa.StructArray.from_arrays([pa.array([1., 2.])], names=['a'], mask=pa.array([False, True]))
    with pytest.raises(ValueError, match='null'):
        _core._inspect_arrow_buffers(pa.chunked_array([parent]))
