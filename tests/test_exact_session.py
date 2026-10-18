"""Frozen a6 materialization and opt-in session consume identical RandomState draws."""
import numpy as np
import pytest
from bankai_boruta import BankaiBoruta, _boruta_core
from bankai_random_forest import BankaiRandomForestClassifier


def fit_pair(monkeypatch, X, y, **controls):
    base = BankaiRandomForestClassifier(extra_trees=True, extra_trees_backend='exact',
        max_depth=4, n_estimators=7, n_jobs=controls.pop('threads', 1), **controls)
    kwargs = dict(n_estimators=None, max_iter=18, random_state=42, store_history=True)
    monkeypatch.delenv('BANKAI_BORUTA_EXACT_SESSION', raising=False)
    reference = BankaiBoruta(base, **kwargs).fit(X, y)
    monkeypatch.setenv('BANKAI_BORUTA_EXACT_SESSION', '1')
    candidate = BankaiBoruta(base, **kwargs).fit(X, y)
    for name in ['hits_', 'support_', 'support_weak_', 'ranking_']:
        np.testing.assert_array_equal(getattr(candidate,name), getattr(reference,name))
    for name in ['importance_history_', 'shadow_threshold_history_']:
        np.testing.assert_allclose(getattr(candidate,name), getattr(reference,name), rtol=1e-12, atol=1e-12, equal_nan=True)
    assert (candidate.n_iter_,candidate.stop_reason_) == (reference.n_iter_,reference.stop_reason_)
    if X.dtype == np.float32:
        assert candidate.diagnostics_['storage_mode'] == 'virtual_session'
        assert candidate.diagnostics_['materialized_bytes'] == 0
        assert candidate.diagnostics_['training_storage_rebuilds'] == 0
    else:
        assert candidate.diagnostics_['path'] == 'python'
    return candidate


@pytest.mark.parametrize('dtype', [np.float32,np.float64])
@pytest.mark.parametrize('layout', ['C','F','strided'])
@pytest.mark.parametrize('classes', [2,3])
@pytest.mark.parametrize('bootstrap', [False,True])
def test_session_fit_parity(monkeypatch, dtype, layout, classes, bootstrap):
    X = np.random.RandomState(8).randint(-2,3,(100,12)).astype(dtype)
    X[:, -1] = 1
    # Distinct f64 values collapsing to the same f32 must retain original ordering.
    X[:, 0] += np.arange(len(X)) * np.finfo(dtype).eps
    X = np.array(X, order='F' if layout == 'F' else 'C')
    if layout == 'strided':
        X = X[::-2,::2]
    y = np.arange(len(X)) % classes
    fit_pair(monkeypatch,X,y,bootstrap=bootstrap)


@pytest.mark.parametrize('controls', [
    dict(threads=4, criterion='entropy', max_features=.5),
    dict(threads=16, criterion='log_loss', min_samples_leaf=3, min_samples_split=6),
    dict(max_samples=.7, min_weight_fraction_leaf=.05, min_impurity_decrease=.0001, max_leaf_nodes=7),
])
def test_native_controls(monkeypatch, controls):
    X = np.random.RandomState(8).normal(size=(160,20)).astype(np.float32)
    result = fit_pair(monkeypatch,X,(X[:,0]>0).astype(int),**controls)
    assert result.diagnostics_['sort_cache_rebuilds'] >= 1


def test_duplicate_shadows_and_full_ties(monkeypatch):
    X = np.zeros((48,2), dtype=np.float32)
    X[:24,0] = 1
    fit_pair(monkeypatch,X,np.arange(48)%2)


def test_cache_invalidation_and_successive_fit(monkeypatch):
    monkeypatch.setenv('BANKAI_BORUTA_EXACT_SESSION','1')
    X = np.random.RandomState(8).normal(size=(160,20)).astype(np.float32)
    result = fit_pair(monkeypatch,X,(X[:,0]>0).astype(int))
    assert result.diagnostics_['sort_cache_rebuilds'] > 1
    first = result.hits_.copy()
    result.fit(X,(X[:,0]>0).astype(int))
    np.testing.assert_array_equal(result.hits_,first)


@pytest.mark.parametrize('dtype',[np.uint32,np.int64])
def test_session_map_validation_isolation_and_snapshot(dtype):
    X = np.arange(40,dtype=np.float64).reshape(10,4)
    y = np.arange(10,dtype=np.int64)%2
    session = _boruta_core.ExactSession(X,y)
    other = _boruta_core.ExactSession(X,y)
    active = np.array([0,1],dtype=np.int64)
    sources = np.array([1,0],dtype=np.int64)
    permutations = np.tile(np.arange(10,dtype=dtype),(2,1))
    args = (active,sources,permutations,3,3,42,None,6,1,2)
    original = session.gain_importances(*args)[0]
    X[:] = -100; y[:] = 0
    np.testing.assert_allclose(session.gain_importances(*args)[0],original,atol=1e-12,rtol=1e-12)
    np.testing.assert_allclose(other.gain_importances(*args)[0],original,atol=1e-12,rtol=1e-12)
    for invalid in [-1,10]:
        bad = permutations.astype(np.int64); bad[0,0]=invalid
        with pytest.raises(ValueError): session.gain_importances(active,sources,bad,*args[3:])
    bad=permutations.copy(); bad[0,0]=1
    with pytest.raises(ValueError,match='permutation'): session.gain_importances(active,sources,bad,*args[3:])
    with pytest.raises(ValueError): session.gain_importances(np.array([4]),sources,*args[2:])
    with pytest.raises(ValueError): session.gain_importances(active,sources,permutations[:1],*args[3:])
    np.testing.assert_allclose(session.gain_importances(*args)[0],original,atol=1e-12,rtol=1e-12)


@pytest.mark.parametrize('X', [np.array([[np.nan],[1.]]),np.array([[np.inf],[1.]]), np.array([[1e300],[1.]])])
def test_invalid_session_values(X):
    with pytest.raises(ValueError): _boruta_core.ExactSession(X,np.array([0,1],dtype=np.int64))


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('layout',['C','F','strided'])
@pytest.mark.parametrize('index_dtype',[np.uint32,np.int64])
def test_internal_original_precision_parity(dtype,layout,index_dtype):
    X=np.random.RandomState(4).randint(-2,3,(64,8)).astype(dtype)
    X[:,0] += np.arange(64)*np.finfo(dtype).eps
    if layout=='F': X=np.asfortranarray(X)
    elif layout=='strided': X=X[::-2,::2]
    y=np.arange(len(X),dtype=np.int64)%3
    rng=np.random.RandomState(42)
    session=_boruta_core.ExactSession(X,y)
    for active in [np.arange(X.shape[1]),np.arange(X.shape[1]),np.array([1,0]),np.arange(X.shape[1])]:
        active=active.astype(np.int64)
        sources=np.tile(active,2)
        permutations=np.array([rng.permutation(len(X)) for _ in sources],dtype=index_dtype)
        settings=(5,4,42,None,6,2,2,'exact','gini')
        expected=_boruta_core.gain_importances(X,y,active,sources,permutations,*settings)
        actual=session.gain_importances(active,sources,permutations,*settings)
        np.testing.assert_allclose(actual[0],expected[0],rtol=1e-12,atol=1e-12)
    assert actual[3]['sort_cache_rebuilds']==3


def test_rng_state_and_round_diagnostics(monkeypatch):
    X=np.random.RandomState(8).normal(size=(80,8)).astype(np.float32)
    y=(X[:,0]>0).astype(int)
    base=BankaiRandomForestClassifier(extra_trees=True,extra_trees_backend='exact',n_estimators=3,max_depth=3,n_jobs=1)
    states=[]
    for enabled in ['0','1']:
        monkeypatch.setenv('BANKAI_BORUTA_EXACT_SESSION',enabled)
        rng=np.random.RandomState(42)
        rng.normal(size=3)
        selector=BankaiBoruta(base,n_estimators=None,max_iter=4,random_state=rng,verbose=2).fit(X,y)
        states.append(rng.get_state())
        rounds=selector.diagnostics_['rounds']
        assert len(rounds)==selector.n_iter_
        assert rounds[-1]['stop_reason']==selector.stop_reason_
        assert all(r['active_width']>0 and len(r['decisions'])==8 for r in rounds)
    assert states[0][0]==states[1][0]
    np.testing.assert_array_equal(states[0][1],states[1][1])
    assert states[0][2:]==states[1][2:]


@pytest.mark.parametrize('rows',[2,31,1000])
def test_owned_shuffle_matches_numpy_and_reuses_buffer(rows):
    X=np.random.RandomState(5).normal(size=(rows,2)).astype(np.float32)
    y=np.arange(rows,dtype=np.int64)%2
    owned=_boruta_core.ExactSession(X,y)
    active=np.array([0,1],dtype=np.int64)
    rng=np.random.RandomState(2077)
    rng.normal(size=3)  # Include the legacy Gaussian cache in state checks.
    for shadows in [6,6,8,6]:
        sources=np.arange(shadows,dtype=np.int64)%2
        state=rng.get_state()
        words,position=owned.shuffle_uint32(state[1],state[2],shadows)
        expected=np.array([rng.permutation(rows) for _ in sources],dtype=np.uint32)
        np.testing.assert_array_equal(words,rng.get_state()[1])
        assert position==rng.get_state()[2]
        settings=(3,3,42,None,6,1,2,'exact')
        a=owned.gain_importances(active,sources,None,*settings)
        b=_boruta_core.gain_importances(X,y,active,sources,expected,*settings)
        np.testing.assert_allclose(a[0],b[0],atol=1e-12,rtol=1e-12)
        assert a[3]['native_permutation_copied_bytes']==0
    state=rng.get_state()
    for words,pos,shadows in [(state[1][:1],state[2],1),(state[1],625,1),(state[1],0,0),(state[1],0,np.iinfo(np.intp).max)]:
        with pytest.raises(ValueError): owned.shuffle_uint32(words,pos,shadows)


def test_custom_randomstate_retains_numpy_shuffle(monkeypatch):
    class CustomRandomState(np.random.RandomState):
        def shuffle(self, values):
            self.shuffles = getattr(self, 'shuffles', 0) + 1
            super().shuffle(values)
    X=np.random.RandomState(8).normal(size=(80,8)).astype(np.float32)
    y=(X[:,0]>0).astype(int)
    base=BankaiRandomForestClassifier(extra_trees=True,extra_trees_backend='exact',n_estimators=3,max_depth=3,n_jobs=1)
    results=[]
    for enabled in ['0','1']:
        monkeypatch.setenv('BANKAI_BORUTA_EXACT_SESSION',enabled)
        rng=CustomRandomState(42)
        selector=BankaiBoruta(base,n_estimators=None,max_iter=4,random_state=rng).fit(X,y)
        results.append(selector)
        assert rng.shuffles==selector.n_iter_*8
    np.testing.assert_array_equal(results[0].hits_,results[1].hits_)
    assert results[1].diagnostics_['native_permutation_copied_bytes']>0
    assert results[1].diagnostics_['permutation_bytes']>0
