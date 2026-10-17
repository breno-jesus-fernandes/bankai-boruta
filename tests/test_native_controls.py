import numpy as np
import pytest
from bankai_boruta import BankaiBoruta
from bankai_random_forest import BankaiRandomForestClassifier

@pytest.mark.parametrize('threads', [1, 4])
@pytest.mark.parametrize('features', ['sqrt', .15, 3, None, 'log2'])
@pytest.mark.parametrize('greedy', [False, True])
def test_native_adapter_parity(threads, features, greedy):
    X = np.random.default_rng(100).normal(size=(96, 20))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    base = BankaiRandomForestClassifier(
        n_estimators=5, max_depth=3, max_bins=63, max_features=features,
        n_jobs=threads, extra_trees=True, extra_trees_backend='binned',
    )
    kwargs = dict(n_estimators=5, random_state=42, max_iter=4,
                  greedy=greedy, store_history=True)
    native = BankaiBoruta(base, **kwargs).fit(X, y)
    adapter = BankaiBoruta(base, importance_getter='feature_importances_', **kwargs).fit(X, y)
    assert native.diagnostics_['path'] == 'rust'
    assert adapter.diagnostics_['path'] == 'python'
    assert native.diagnostics_['copied_bytes'] == 0
    assert adapter.diagnostics_['copied_bytes'] > 0
    np.testing.assert_allclose(native.importance_history_, adapter.importance_history_,
                               atol=1e-14, equal_nan=True)
    np.testing.assert_array_equal(native.support_, adapter.support_)
    np.testing.assert_array_equal(native.support_weak_, adapter.support_weak_)
    assert native.n_iter_ == adapter.n_iter_


def test_fractional_sampling_parity_after_classic_rejections():
    X = np.random.default_rng(8).normal(size=(160, 20))
    y = (X[:, 0] > 0).astype(int)
    base = BankaiRandomForestClassifier(
        n_estimators=8, max_depth=3, max_bins=63, max_features=.15,
        n_jobs=2, extra_trees=True, extra_trees_backend='binned',
    )
    kwargs = dict(n_estimators=8, random_state=42, max_iter=20, store_history=True)
    native = BankaiBoruta(base, **kwargs).fit(X, y)
    adapter = BankaiBoruta(base, importance_getter='feature_importances_', **kwargs).fit(X, y)
    # Rejected features disappear, so fractional counts follow changing widths.
    assert np.isnan(native.importance_history_[-1]).sum() > 0
    np.testing.assert_allclose(native.importance_history_, adapter.importance_history_,
                               atol=1e-14, equal_nan=True)
    np.testing.assert_array_equal(native.support_, adapter.support_)


@pytest.mark.parametrize('name,value', [
    ('n_jobs', 0), ('n_jobs', True), ('n_jobs', 1.5), ('n_jobs', '4'),
    ('max_features', 0), ('max_features', 1000), ('max_features', True),
    ('max_features', 1.5), ('max_features', float('nan')), ('max_features', 'invalid'),
])
@pytest.mark.parametrize('getter', ['auto', 'feature_importances_'])
def test_invalid_controls_retain_forest_validation(name, value, getter):
    X = np.random.default_rng(1).normal(size=(24, 6))
    y = np.arange(24) % 2
    params = dict(n_estimators=2, max_depth=3, max_bins=63, extra_trees=True,
                  extra_trees_backend='binned', n_jobs=1)
    params[name] = value
    with pytest.raises(ValueError, match=name):
        BankaiBoruta(BankaiRandomForestClassifier(**params), n_estimators=2,
                     max_iter=2, importance_getter=getter).fit(X, y)


@pytest.mark.parametrize('jobs,expected', [(None, 1), (1, 1), (4, 4), (-1, 8), (-2, 7)])
def test_resolved_controls_cross_native_boundary(monkeypatch, jobs, expected):
    from bankai_boruta import _boruta_core
    monkeypatch.setattr('os.cpu_count', lambda: 8)
    calls = []
    original = _boruta_core.gain_importances

    def record(*args):
        calls.append((args[-2], args[-1]))
        return original(*args)

    monkeypatch.setattr(_boruta_core, 'gain_importances', record)
    X = np.random.default_rng(1).normal(size=(24, 6))
    base = BankaiRandomForestClassifier(n_estimators=2, max_depth=3, max_bins=63,
        max_features=.15, n_jobs=jobs, extra_trees=True, extra_trees_backend='binned')
    result = BankaiBoruta(base, n_estimators=2, max_iter=2).fit(X, np.arange(24) % 2)
    assert result.diagnostics_['path'] == 'rust'
    assert calls == [(expected, 1)]  # floor(.15 * 12 augmented columns)


@pytest.mark.parametrize('threads,features', [(0, 1), (1, 0), (1, 3)])
def test_native_boundary_rejects_invalid_controls(threads, features):
    from bankai_boruta import _boruta_core
    with pytest.raises(ValueError, match='invalid training'):
        _boruta_core.gain_importances(
            np.ones((3, 1)), np.array([0, 1, 0], dtype=np.int64),
            np.array([0], dtype=np.int64), np.array([0], dtype=np.int64),
            np.tile(np.arange(3, dtype=np.int64), (1, 1)),
            2, 3, 1, 63, 6, threads, features)
