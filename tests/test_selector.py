import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from bankai_random_forest import BankaiRandomForestClassifier
from bankai_boruta import BankaiBoruta


class Controlled(ClassifierMixin, BaseEstimator):
    def __init__(self, scores=(2., 0., 1.), random_state=None):
        self.scores = scores
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(y)
        self.feature_importances_ = np.ones(X.shape[1])
        self.feature_importances_[:len(self.scores)] = self.scores
        return self


@pytest.fixture
def data():
    rng = np.random.RandomState(11)
    X = rng.normal(size=(100, 3))
    return X, np.where(X[:, 0] > 0, 'yes', 'no')


@pytest.mark.parametrize('greedy', [False, True])
@pytest.mark.parametrize('two_step', [False, True])
def test_controlled(data, greedy, two_step):
    X, y = data
    base = Controlled()
    model = BankaiBoruta(base, n_estimators=None, random_state=1, greedy=greedy, two_step=two_step, store_history=True).fit(X, y)
    np.testing.assert_array_equal(model.support_, [True, False, False])
    assert not model.support_weak_.any()
    assert model.transform(X).shape == (100, 1)
    assert not hasattr(base, 'classes_')
    assert model.estimator_.classes_.tolist() == ['no', 'yes']
    assert clone(model).estimator.get_params() == model.estimator.get_params()
    assert clone(model).greedy == model.greedy
    if greedy:
        assert model.n_iter_ == model.max_iter_ == 6
        np.testing.assert_array_equal(model.ranking_, [1, 2, 2])


@pytest.mark.parametrize('two_step', [True, False])
@pytest.mark.parametrize('perc', [50, 100])
def test_borutapy_controlled(data, two_step, perc):
    from boruta import BorutaPy
    X, y = data
    # n_estimators is accepted by the reference, so use a forest with a controlled getter.
    class Scripted(RandomForestClassifier):
        def fit(self, X, y):
            self.n_features_in_ = X.shape[1]
            return self
        @property
        def feature_importances_(self):
            return np.r_[2., 0., np.ones(self.n_features_in_ - 2)]
    args = dict(n_estimators=5, max_iter=30, random_state=2, two_step=two_step, perc=perc)
    ours = BankaiBoruta(Scripted(), **args).fit(X, y)
    ref = BorutaPy(Scripted(), **args).fit(X, y)
    for attr in ('support_', 'support_weak_', 'ranking_'):
        np.testing.assert_array_equal(getattr(ours, attr), getattr(ref, attr))


@pytest.mark.parametrize('bonferroni', [True, False])
def test_greedy_reference(data, bonferroni):
    from greedyboruta import GreedyBorutaPy
    class Scripted(RandomForestClassifier):
        def fit(self, X, y):
            self.n_features_in_ = X.shape[1]
            return self
        @property
        def feature_importances_(self):
            return np.r_[2., 0., np.ones(self.n_features_in_ - 2)]
    X, y = data
    args = dict(n_estimators=5, random_state=2, bonferroni=bonferroni)
    ours = BankaiBoruta(Scripted(), greedy=True, **args).fit(X, y)
    ref = GreedyBorutaPy(Scripted(), **args).fit(X, y)
    np.testing.assert_array_equal(ours.support_, ref.support_)
    np.testing.assert_array_equal(ours.ranking_, ref.ranking_)
    assert ours.max_iter_ == ref.max_iter


def test_native_adapter(data):
    X, y = data
    base = BankaiRandomForestClassifier(extra_trees=True, extra_trees_backend='binned', max_bins=63, max_depth=6, n_jobs=-1)
    args = dict(n_estimators=20, greedy=True, random_state=42, store_history=True)
    native = BankaiBoruta(base, **args).fit(X, y)
    adapter = BankaiBoruta(base, importance_getter=lambda e: e.feature_importances_, **args).fit(X, y)
    assert native.diagnostics_['path'] == 'rust'
    np.testing.assert_allclose(native.importance_history_, adapter.importance_history_, atol=1e-12)
    np.testing.assert_array_equal(native.support_, adapter.support_)
    np.testing.assert_array_equal(native.support_, BankaiBoruta(base, **args).fit(X, y).support_)


def test_names_pipeline_refit(data):
    X, y = data
    frame = pd.DataFrame(X, columns=['a', 'b', 'c'])
    selector = BankaiBoruta(Controlled(), n_estimators=None, greedy=True, random_state=1)
    pipeline = make_pipeline(selector, LogisticRegression()).fit(frame, y)
    assert pipeline.predict(frame).shape == y.shape
    assert selector.get_feature_names_out().tolist() == ['a']
    selector.fit(X, y)
    assert not hasattr(selector, 'feature_names_in_')
    with pytest.raises(ValueError):
        selector.fit(X, np.zeros(len(X)))
    assert not hasattr(selector, 'support_')


@pytest.mark.parametrize('p', [1, 2, 3])
def test_few_features_and_no_selection(p):
    X = np.arange(30 * p).reshape(30, p)
    y = np.arange(30) % 3
    s = BankaiBoruta(Controlled(scores=(0.,) * p), n_estimators=None, greedy=True, alpha=np.nextafter(0., 1.)).fit(X, y)
    assert s.max_iter_ == int(np.ceil(np.log2(p) + 1074))
    assert not s.support_.any()
    with pytest.warns(UserWarning):
        assert s.transform(X).shape == (30, 0)


def test_weights_and_constraints_fallback(data):
    X, y = data
    base = BankaiRandomForestClassifier(n_estimators=4, extra_trees=True, extra_trees_backend='binned', max_bins=10, monotonic_cst=[1, 0, 0])
    s = BankaiBoruta(base, n_estimators=None, greedy=True, random_state=1).fit(X, y, sample_weight=np.ones(len(X)))
    assert s.diagnostics_['path'] == 'python'
    assert s.estimator_.monotonic_cst[3:].sum() == 0
    assert base.monotonic_cst == [1, 0, 0]


@pytest.mark.parametrize('scores', [(np.nan,), (np.inf,)])
def test_invalid_importance(data, scores):
    with pytest.raises(ValueError, match='finite vector'):
        BankaiBoruta(Controlled(scores=scores), n_estimators=None).fit(*data)


def test_coef(data):
    s = BankaiBoruta(LogisticRegression(), n_estimators=None, greedy=True, random_state=2).fit(*data)
    assert s.support_.shape == (3,)


@pytest.mark.parametrize('kwargs', [dict(perc=0), dict(alpha=1), dict(n_estimators=0), dict(max_iter=0), dict(greedy=1)])
def test_invalid_params(data, kwargs):
    with pytest.raises(ValueError):
        BankaiBoruta(**kwargs).fit(*data)


@pytest.mark.parametrize('seed', [1, 7, 42])
@pytest.mark.parametrize('trees,depth,bins', [(1, 2, 2), (12, None, 16), (20, 5, 63)])
def test_native_multiclass_and_settings(seed, trees, depth, bins):
    rng = np.random.RandomState(seed)
    # Discrete columns exercise canonical ties, including signed zero.
    X = rng.randint(-2, 3, size=(90, 6)).astype(float)
    X[::2, 0] = -0.
    y = np.array(['a', 'b', 'c'])[np.arange(90) % 3]
    base = BankaiRandomForestClassifier(extra_trees=True, extra_trees_backend='binned', max_bins=bins,
        max_depth=depth, n_jobs=-1, extra_seed=seed)
    kwargs = dict(n_estimators=trees, greedy=True, random_state=seed, store_history=True)
    a = BankaiBoruta(base, **kwargs).fit(X, y)
    b = BankaiBoruta(base, importance_getter=lambda e: e.feature_importances_, **kwargs).fit(X, y)
    np.testing.assert_allclose(a.importance_history_, b.importance_history_, atol=1e-12)
    np.testing.assert_array_equal(a.support_, b.support_)


def test_early_stopping_tentative_and_parameter_preservation(data):
    X, y = data
    s = BankaiBoruta(Controlled(scores=(1.1, 1., 1.)), n_estimators=None,
        max_iter=4, store_history=True, random_state=7).fit(X, y)
    assert s.support_weak_.tolist() == [True, False, False]
    assert not s.support_.any()
    s.set_params(max_iter=100, early_stopping=True, n_iter_no_change=2)
    s.fit(X, y)
    assert s.n_iter_ == 3
    assert s.stop_reason_ == 'early_stopping'
    assert s.max_iter == 100
    s.set_params(store_history=False, greedy=True, max_iter=1)
    s.fit(X, y)
    assert s.n_iter_ == 6 and s.max_iter == 1
    assert not hasattr(s, 'importance_history_')


def test_nested_and_dotted_getter(data):
    base = Controlled()
    selector = BankaiBoruta(base, n_estimators=None, greedy=True, importance_getter='feature_importances_')
    selector.set_params(estimator__scores=(0., 2., 0.))
    selector.fit(*data)
    assert selector.support_.tolist() == [False, True, False]


@pytest.mark.parametrize('kwargs', [dict(class_weight='balanced'), dict(min_samples_leaf=.1),
    dict(bootstrap=False), dict(max_features=None), dict(importance_type='split'), dict(ccp_alpha=.01)])
def test_bankai_parameters_preserved(data, kwargs):
    base = BankaiRandomForestClassifier(n_estimators=3, **kwargs)
    selector = BankaiBoruta(base, n_estimators=None, greedy=True, random_state=2).fit(*data)
    assert selector.diagnostics_['path'] == 'python'
    for key, value in kwargs.items():
        assert selector.estimator_.get_params()[key] == value
    assert not hasattr(base, 'classes_')


def test_one_round_budget_and_weights_rejected(data):
    X, y = data
    s = BankaiBoruta(max_iter=1, random_state=3).fit(X, y)
    assert s.n_iter_ == 0 and s.n_features_ == 0
    with pytest.raises(ValueError):
        BankaiBoruta().fit(X, y, sample_weight=np.zeros(len(X)))
    with pytest.raises(ValueError):
        BankaiBoruta().fit(X, y, sample_weight=-np.ones(len(X)))


def test_regressor_rejected(data):
    from sklearn.ensemble import RandomForestRegressor
    with pytest.raises(ValueError, match='classifier'):
        BankaiBoruta(RandomForestRegressor()).fit(*data)


def test_bad_native_arrays():
    from bankai_boruta import _boruta_core
    with pytest.raises(ValueError):
        _boruta_core.gain_importances(np.ones((3, 1)), np.array([0, 1, 0], dtype=np.int64),
            np.array([0], dtype=np.int64), np.array([2], dtype=np.int64), np.zeros((1, 3), dtype=np.int64), 2, 6, 1, 63, 6)


@pytest.mark.parametrize('perc', [50, 100])
@pytest.mark.parametrize('two_step', [True, False])
@pytest.mark.parametrize('early_stopping', [True, False])
def test_classic_sequences_against_reference(perc, two_step, early_stopping):
    from boruta import BorutaPy
    class Scripted(RandomForestClassifier):
        def fit(self, X, y):
            ids = X[0].astype(int)
            seen = set()
            width = 0
            for value in ids:
                if value in seen:
                    break
                seen.add(value)
                width += 1
            sequence = np.array([[2., -1., .4, .8, 1.5, .2],
                                 [2., -1., 1.6, .7, 1.5, .6],
                                 [2., -1., .2, 1.1, 1.5, .1]])
            iteration = getattr(self, 'calls_', 0)
            self.calls_ = iteration + 1
            self.importances_ = np.r_[sequence[iteration % 3, ids[:width] // 100 - 1],
                                     np.linspace(-.5, 1.5, X.shape[1] - width)]
            return self
        @property
        def feature_importances_(self):
            return self.importances_
    X = np.tile(np.arange(1, 7) * 100., (20, 1))
    y = np.arange(20) % 2
    kwargs = dict(n_estimators=2, max_iter=40, alpha=.1, random_state=3,
        perc=perc, two_step=two_step, early_stopping=early_stopping, n_iter_no_change=8)
    a = BankaiBoruta(Scripted(), **kwargs).fit(X, y)
    b = BorutaPy(Scripted(), **kwargs).fit(X, y)
    for name in ('support_', 'support_weak_', 'ranking_'):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


def test_exact_greedy_round_boundaries():
    from bankai_boruta._estimator import _greedy_rounds
    for p in [1, 3, 5, 31, 1000]:
        alpha = p / 2**16
        assert _greedy_rounds(p, alpha, True) == 16
        assert _greedy_rounds(p, np.nextafter(alpha, 0), True) == 17
        assert _greedy_rounds(p, np.nextafter(alpha, 1), True) == 16
    assert _greedy_rounds(1000, np.nextafter(0., 1.), False) == 1074


def test_pretrained_estimator_remains_unchanged(data):
    X, y = data
    base = RandomForestClassifier(n_estimators=3, random_state=42).fit(X, y)
    importances = base.feature_importances_.copy()
    trees = list(base.estimators_)
    BankaiBoruta(base, n_estimators=None, greedy=True, random_state=7).fit(X, y)
    np.testing.assert_array_equal(base.feature_importances_, importances)
    assert base.estimators_ == trees
    assert base.random_state == 42


@pytest.mark.parametrize('estimator_type', [RandomForestClassifier, BankaiRandomForestClassifier])
def test_warm_start_uses_fresh_shadow_forests(data, estimator_type):
    import warnings
    X, y = data
    kwargs = dict(n_estimators=5, max_depth=3, n_jobs=1)
    selector_kwargs = dict(n_estimators=None, greedy=True, random_state=42, store_history=True)
    with warnings.catch_warnings(record=True) as caught:
        warm = BankaiBoruta(estimator_type(warm_start=True, **kwargs), **selector_kwargs).fit(X, y)
    cold = BankaiBoruta(estimator_type(warm_start=False, **kwargs), **selector_kwargs).fit(X, y)
    assert not any('Warm-start' in str(item.message) for item in caught)
    assert warm.estimator_.warm_start is True
    np.testing.assert_allclose(warm.importance_history_, cold.importance_history_, atol=1e-12)
    np.testing.assert_array_equal(warm.support_, cold.support_)
