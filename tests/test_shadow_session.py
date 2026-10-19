"""Default native exact path: canonical float32 bank plus shuffled shadow values."""
import numpy as np
import pytest
from bankai_boruta import BankaiBoruta, _boruta_core
from bankai_random_forest import BankaiRandomForestClassifier


def data(rows=400, features=12, seed=3):
    X = np.random.default_rng(seed).normal(size=(rows, features)).astype(np.float32)
    y = (X[:, 0] + .7 * X[:, 1] > 0).astype(int)
    return X, y


def fit(X, y, jobs=2, **kwargs):
    base = BankaiRandomForestClassifier(extra_trees=True, extra_trees_backend='exact',
        n_estimators=30, max_depth=5, n_jobs=jobs)
    options = dict(n_estimators=None, max_iter=25, random_state=7, store_history=True)
    options.update(kwargs)
    return BankaiBoruta(base, **options).fit(X, y)


def test_default_exact_path_uses_shadow_session_without_maps():
    X, y = data()
    selector = fit(X, y)
    diagnostics = selector.diagnostics_
    assert diagnostics['path'] == 'rust'
    assert diagnostics['storage_mode'] == 'shadow_session'
    assert diagnostics['materialized_bytes'] == diagnostics['permutation_bytes'] == 0
    assert diagnostics['real_bank_bytes'] == X.nbytes
    assert 0 < diagnostics['shadow_buffer_bytes'] <= X.nbytes
    assert diagnostics['training_storage_rebuilds'] == 0


def test_shadow_session_selects_signal_and_rejects_noise():
    X, y = data()
    selector = fit(X, y, max_iter=60)
    assert selector.support_[:2].all()
    assert not selector.support_[2:].any()
    assert (selector.ranking_[2:] > 1).all()


@pytest.mark.parametrize('jobs', [1, 3, 8])
def test_results_are_deterministic_and_thread_count_invariant(jobs):
    X, y = data()
    reference = fit(X, y, jobs=4)
    other = fit(X, y, jobs=jobs)
    np.testing.assert_array_equal(other.importance_history_, reference.importance_history_)
    for name in ['hits_', 'support_', 'support_weak_', 'ranking_']:
        np.testing.assert_array_equal(getattr(other, name), getattr(reference, name))
    assert other.n_iter_ == reference.n_iter_


@pytest.mark.parametrize('layout', ['C', 'F', 'strided'])
def test_input_row_order_and_layout_do_not_change_results(layout):
    X, y = data()
    reference = fit(X, y)
    order = np.random.default_rng(1).permutation(len(X))
    shuffled = np.array(X[order], order='F' if layout == 'F' else 'C')
    labels = y[order]
    if layout == 'strided':
        shuffled = np.repeat(shuffled, 2, axis=0)[::2]
    other = fit(shuffled, labels)
    np.testing.assert_array_equal(other.importance_history_, reference.importance_history_)
    np.testing.assert_array_equal(other.support_, reference.support_)


def test_successive_fits_reuse_nothing_stale():
    X, y = data()
    selector = fit(X, y)
    first = selector.importance_history_.copy()
    selector.fit(X, y)
    np.testing.assert_array_equal(selector.importance_history_, first)


def test_session_validates_inputs():
    X, y = data(rows=20, features=3)
    labels = y.astype(np.int64)
    with pytest.raises(ValueError):
        _boruta_core.ShadowSession(np.full_like(X, np.nan), labels, 1)
    with pytest.raises(ValueError):
        _boruta_core.ShadowSession(X, labels[:-1], 1)
    with pytest.raises(ValueError):
        _boruta_core.ShadowSession(X, np.zeros_like(labels), 1)
    session = _boruta_core.ShadowSession(X, labels, 1)
    active = np.array([0, 1], dtype=np.int64)
    for bad in (np.array([3], dtype=np.int64), np.array([-1], dtype=np.int64), np.array([], dtype=np.int64)):
        with pytest.raises(ValueError):
            session.gain_importances(active, bad, 1, 3, 3, 5, 6, 1, 1)
    scores, _, _, metrics = session.gain_importances(active, np.array([0, 1, 0, 1, 0], dtype=np.int64), 1, 3, 3, 5, 6, 1, 2)
    assert len(scores) == 7 and np.isclose(sum(scores), 1)
    assert metrics['shadow_buffer_bytes'] >= 5 * len(X) * 4
