"""Memory representation changes must preserve the a4 numerical contract."""
import numpy as np
import pytest
from bankai_boruta import BankaiBoruta, _boruta_core
from bankai_boruta._estimator import _permutation_dtype
from bankai_random_forest import BankaiRandomForestClassifier


@pytest.fixture(autouse=True)
def a7_reference(monkeypatch):
    # Exact native/adapter parity is the a7 reference contract: its shadows are
    # NumPy permutations. Shadow sessions are covered in test_shadow_session.py.
    monkeypatch.setenv('BANKAI_BORUTA_EXACT_REFERENCE', '1')



def forest(threads=1):
    return BankaiRandomForestClassifier(n_estimators=3, max_depth=3, max_bins=63, n_jobs=threads,
        extra_trees=True, extra_trees_backend='binned')


def equivalent(a, b):
    for name in ('support_', 'support_weak_', 'ranking_'):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    for name in ('importance_history_', 'shadow_threshold_history_'):
        np.testing.assert_allclose(getattr(a, name), getattr(b, name),
                                   rtol=0, atol=1e-14, equal_nan=True)
    assert (a.n_iter_, a.stop_reason_) == (b.n_iter_, b.stop_reason_)


@pytest.mark.parametrize('threads', [1, 4])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('layout', ['C', 'F', 'view'])
@pytest.mark.parametrize('greedy', [False, True])
@pytest.mark.parametrize('classes,features', [(2, 3), (3, 9)])
def test_representations(dtype, layout, greedy, classes, features, threads):
    rng = np.random.RandomState(8)
    X = rng.randint(-2, 3, size=(64, features)).astype(dtype)
    X[:, -1] = 0
    X[::2, -1] = -0.
    if layout == 'F':
        X = np.asfortranarray(X)
    elif layout == 'view':
        X = np.repeat(X, 2, axis=0)[::2, ::-1]
    y = np.arange(len(X)) % classes
    options = dict(n_estimators=3, max_iter=18, random_state=7,
                   greedy=greedy, store_history=True)
    native = BankaiBoruta(forest(threads), **options).fit(X, y)
    adapter = BankaiBoruta(forest(threads), importance_getter='feature_importances_', **options).fit(X, y)
    equivalent(native, adapter)
    equivalent(native, BankaiBoruta(forest(threads), **options).fit(X.astype(np.float64), y))
    assert native.diagnostics_['native_input_dtype'] == np.dtype(dtype).name
    assert native.diagnostics_['input_conversion_bytes'] == 0
    assert native.diagnostics_['permutation_dtype'] == 'uint32'
    assert native.diagnostics_['copied_bytes'] == 0
    assert adapter.diagnostics_['input_conversion_bytes'] == (X.size * 8 if dtype == np.float32 else 0)
    equivalent(adapter, native.fit(X, y))


@pytest.mark.parametrize('rows', [0, 1, 17, 2**32-1, 2**32, 2**32+1])
def test_dtype_boundaries_without_allocation(rows):
    if rows > np.iinfo(np.intp).max:
        with pytest.raises(ValueError):
            _permutation_dtype(rows)
    else:
        assert _permutation_dtype(rows) == np.dtype('uint32' if rows <= 2**32 else 'int64')


def test_addressing_limit():
    with pytest.raises(ValueError):
        _permutation_dtype(np.iinfo(np.intp).max + 1)


@pytest.mark.parametrize('rows', [1, 2, 13, 1000])
def test_shuffle_rng_state(rows):
    a, b = np.random.RandomState(42), np.random.RandomState(42)
    small, large = np.empty(rows, dtype=np.uint32), np.empty(rows, dtype=np.int64)
    for _ in range(13):
        small[:] = np.arange(rows, dtype=np.uint32)
        large[:] = np.arange(rows, dtype=np.int64)
        a.shuffle(small)
        b.shuffle(large)
        np.testing.assert_array_equal(small, large)
    for x, y in zip(a.get_state(), b.get_state()):
        np.testing.assert_equal(x, y)


@pytest.mark.parametrize('values', [np.float32, np.float64])
@pytest.mark.parametrize('indices', [np.uint32, np.int64])
@pytest.mark.parametrize('invalid', ['shape', 'index', 'negative', 'feature', 'nan'])
def test_rust_boundary(values, indices, invalid):
    X = np.ones((4, 2), dtype=values)
    perm = np.arange(4, dtype=indices).reshape(1, 4)
    active = np.array([0], dtype=np.int64)
    if invalid == 'shape': perm = perm[:, :-1]
    if invalid == 'index': perm[0, 0] = 4
    if invalid == 'negative':
        perm[0, 0] = np.iinfo(indices).max if indices == np.uint32 else -1
    if invalid == 'feature': active[0] = 2
    if invalid == 'nan': X[0, 0] = np.nan
    with pytest.raises(ValueError, match='invalid training'):
        _boruta_core.gain_importances(X, np.array([0, 1, 0, 1], dtype=np.int64),
            active, np.array([0], dtype=np.int64), perm, 2, 3, 1, 63, 6, 1, 1)


@pytest.mark.parametrize('values', [np.float32, np.float64])
def test_rust_index_specializations(values):
    X = np.random.RandomState(2).normal(size=(24, 3)).astype(values)
    def run(dtype):
        return _boruta_core.gain_importances(X, np.arange(24, dtype=np.int64) % 2,
            np.array([0, 1, 2], dtype=np.int64), np.array([0], dtype=np.int64),
            np.arange(23, -1, -1, dtype=dtype).reshape(1, 24), 3, 3, 1, 63, 6, 1, 2)[0]
    np.testing.assert_array_equal(run(np.uint32), run(np.int64))


@pytest.mark.parametrize('rows', [0, 1, 2, 3, 31, 32, 33, 623, 624, 625, 100000])
@pytest.mark.parametrize('seed', [0, 7, 42, 2**32-1])
def test_native_shuffle_matches_legacy_stream(rows, seed):
    from bankai_boruta._estimator import _shuffle_permutation
    a, b = np.random.RandomState(seed), np.random.RandomState(seed)
    # Preserve a cached Gaussian and start away from the initial MT boundary.
    a.normal(size=3)
    b.normal(size=3)
    for _ in range(3):
        small = np.arange(rows, dtype=np.uint32)
        large = np.arange(rows, dtype=np.int64)
        _shuffle_permutation(a, small)
        b.shuffle(large)
        np.testing.assert_array_equal(small, large)
        for x, y in zip(a.get_state(), b.get_state()):
            np.testing.assert_equal(x, y)
        assert a.randint(2**31-1) == b.randint(2**31-1)
        assert a.normal() == b.normal()


def test_shuffle_preserves_subclass_behavior():
    from bankai_boruta._estimator import _shuffle_permutation
    class CustomRandomState(np.random.RandomState):
        def shuffle(self, values):
            self.called = True
            super().shuffle(values)
    rng = CustomRandomState(7)
    _shuffle_permutation(rng, np.arange(13, dtype=np.uint32))
    assert rng.called


@pytest.mark.parametrize('bitgen', [np.random.PCG64, np.random.Philox, np.random.SFC64])
def test_shuffle_alternative_randomstate_bit_generators(bitgen):
    from bankai_boruta._estimator import _shuffle_permutation
    a, b = np.random.RandomState(bitgen(7)), np.random.RandomState(bitgen(7))
    for _ in range(3):
        small, large = np.arange(333, dtype=np.uint32), np.arange(333, dtype=np.int64)
        _shuffle_permutation(a, small)
        b.shuffle(large)
        np.testing.assert_array_equal(small, large)
        np.testing.assert_array_equal(a.randint(2**31-1, size=19), b.randint(2**31-1, size=19))


@pytest.mark.parametrize('size,position', [(623, 0), (625, 0), (624, 625)])
def test_shuffle_rejects_invalid_state(size, position):
    with pytest.raises(ValueError, match='invalid MT19937'):
        _boruta_core.shuffle_uint32(np.arange(13, dtype=np.uint32), np.ones(size, dtype=np.uint32), position)
