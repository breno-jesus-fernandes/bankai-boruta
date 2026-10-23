"""Classification-only Boruta with a clonable scikit-learn interface."""
from math import sqrt
from numbers import Integral, Real
from operator import attrgetter
from time import perf_counter
import os
import warnings

import numpy as np
from scipy.stats import binom, rankdata
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.feature_selection import SelectorMixin
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data, _check_sample_weight


def _permutation_dtype(rows):
    if rows < 0 or rows > np.iinfo(np.intp).max:
        raise ValueError("row count exceeds platform addressing limits")
    return np.dtype(np.uint32 if rows <= 2**32 else np.int64)


def _shuffle_permutation(rng, permutation):
    # Keep arbitrary RandomState subclasses and alternative bit generators on
    # NumPy's own path. For standard MT19937, only specialize the swap width;
    # state, rejection draws, Gaussian cache and shuffle/seed order stay intact.
    if type(rng) is np.random.RandomState and permutation.dtype == np.uint32:
        state = rng.get_state(legacy=False)
        if state['bit_generator'] == 'MT19937':
            from . import _boruta_core
            words, position = _boruta_core.shuffle_uint32(
                permutation, state['state']['key'], state['state']['pos'])
            state['state']['key'] = words
            state['state']['pos'] = position
            rng.set_state(state)
            return
    rng.shuffle(permutation)


def _greedy_rounds(features, alpha, bonferroni):
    # Exact ceiling of log2(p / alpha), including boundaries and subnormals.
    numerator, denominator = float(alpha).as_integer_ratio()
    target = (features if bonferroni else 1) * denominator
    rounds = max(0, target.bit_length() - numerator.bit_length())
    if numerator << rounds < target:
        rounds += 1
    return max(1, rounds)


def _fdr(p, alpha):
    order = np.argsort(p)
    passed = p[order] <= alpha * np.arange(1, len(p) + 1) / len(p)
    if passed.any():
        passed[:np.flatnonzero(passed)[-1] + 1] = True
    result = np.zeros(len(p), dtype=bool)
    result[order] = passed
    return result


class BankaiBoruta(SelectorMixin, BaseEstimator):
    """Boruta classic by default; ``greedy=True`` uses a fixed hit-once campaign.

    ``n_estimators=None`` preserves the estimator's tree count. Greedy ignores
    max_iter, two_step and early stopping; bonferroni controls its round budget.
    Only confirmed features are returned by transform.
    """
    def __init__(self, estimator=None, n_estimators="auto", perc=100, alpha=0.05,
                 two_step=True, max_iter=100, random_state=None, verbose=0,
                 early_stopping=False, n_iter_no_change=20, *, greedy=False,
                 bonferroni=True, importance_getter="auto", store_history=False):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.perc = perc
        self.alpha = alpha
        self.two_step = two_step
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.greedy = greedy
        self.bonferroni = bonferroni
        self.importance_getter = importance_getter
        self.store_history = store_history

    def _validate_parameters(self):
        for name in ("max_iter", "n_iter_no_change", "verbose"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < (0 if name == "verbose" else 1):
                raise ValueError(f"{name} must be a {'non-negative' if name == 'verbose' else 'positive'} integer")
        if not (self.n_estimators is None or self.n_estimators == "auto" or
                isinstance(self.n_estimators, Integral) and not isinstance(self.n_estimators, bool) and self.n_estimators > 0):
            raise ValueError("n_estimators must be 'auto', None or a positive integer")
        for name, upper in (("alpha", 1), ("perc", 100)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or not 0 < value <= upper or (name == "alpha" and value == 1):
                raise ValueError(f"invalid {name}")
        for name in ("two_step", "early_stopping", "greedy", "bonferroni", "store_history"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool")
        if not callable(self.importance_getter) and not (isinstance(self.importance_getter, str) and all(self.importance_getter.split('.'))):
            raise ValueError("importance_getter must be 'auto', an attribute path or callable")

    def fit(self, X, y, sample_weight=None):
        started = perf_counter()
        for name in tuple(vars(self)):
            if name.endswith('_'):
                delattr(self, name)
        self._validate_parameters()
        conversion_started = perf_counter()
        input_dtype = str(X.dtype) if hasattr(X, "dtype") else str(np.asarray(X).dtype)
        original_X = X
        X, y = validate_data(self, X, y, dtype=[np.float64, np.float32])
        conversion_seconds = perf_counter() - conversion_started
        check_classification_targets(y)
        classes, encoded = np.unique(y, return_inverse=True)
        if len(classes) < 2:
            raise ValueError("classification requires at least two classes")
        weights = None if sample_weight is None else _check_sample_weight(sample_weight, X, ensure_non_negative=True)
        if weights is not None and weights.sum() <= 0:
            raise ValueError("sample_weight must have positive total weight")
        from bankai_random_forest import BankaiRandomForestClassifier
        base = self.estimator
        if base is None:
            # Exact Extra Trees split on float32 values, as scikit-learn's trees do;
            # float64 input is cast once so the default forest always runs natively.
            base = BankaiRandomForestClassifier(extra_trees=True, extra_trees_backend="exact",
                    importance_type="gain", max_depth=6, n_jobs=-1)
            if X.dtype != np.float32:
                convert_started = perf_counter()
                X = X.astype(np.float32)
                conversion_seconds += perf_counter() - convert_started
        if not is_classifier(base):
            raise ValueError("estimator must be a classifier")
        self.estimator_ = clone(base)
        params = self.estimator_.get_params()
        if self.n_estimators is not None and 'n_estimators' not in params:
            raise ValueError("estimator has no n_estimators; use n_estimators=None")
        rng = check_random_state(self.random_state)
        p = X.shape[1]
        self.max_iter_ = _greedy_rounds(p, self.alpha, self.bonferroni) if self.greedy else self.max_iter
        # Native eligibility is deliberately conservative and checked against BankaiRF's own validation.
        native, reason = self._native_eligibility(base, params, weights, X)
        if not native and X.dtype != np.float64:
            convert_started = perf_counter()
            X = X.astype(np.float64)
            conversion_seconds += perf_counter() - convert_started
        conversion_bytes = 0 if isinstance(original_X, np.ndarray) and np.shares_memory(X, original_X) else X.nbytes
        permutation_dtype = _permutation_dtype(len(X))
        self.diagnostics_ = dict(path="rust" if native else "python", reason=reason,
            preparation_seconds=0., training_seconds=0., copied_bytes=0,
            copy_seconds=0., input_conversion_seconds=conversion_seconds,
            training_storage_rebuilds=0, input_dtype=input_dtype,
            native_input_dtype=str(X.dtype) if native else None,
            input_conversion_bytes=conversion_bytes,
            permutation_dtype=permutation_dtype.name, permutation_bytes=0)
        # Native exact fits keep one canonical float32 bank and shuffle shadow
        # values in place each round. The a7 materialized path (REFERENCE) and
        # virtual session (SESSION) remain available for parity checks.
        session = shadow_session = None
        exact = native and self._native_settings_['backend'] == 'exact'
        a7_path = self._native_settings_['extra_trees'] and (os.environ.get('BANKAI_BORUTA_EXACT_REFERENCE') == '1'
                                                            or os.environ.get('BANKAI_BORUTA_EXACT_SESSION') == '1') if native else False
        # Binned sessions bin float32 values once; float64 input keeps per-round
        # binning so its edges come from the original float64 values.
        if native and not a7_path and (exact or X.dtype == np.float32):
            from . import _boruta_core
            settings = self._native_settings_
            prep_started = perf_counter()
            binning = {} if exact else dict(bins=settings['bins'], binning_strategy=settings['binning_strategy'],
                                            bin_sample_size=settings['bin_sample_size'], binning_seed=settings['binning_seed'])
            shadow_session = _boruta_core.ShadowSession(X, encoded.astype(np.int64, copy=False), settings['threads'], **binning)
            session_seconds = perf_counter() - prep_started
            self.diagnostics_.update(session_creation_seconds=session_seconds,
                real_bank_bytes=shadow_session.bank_bytes, real_bank_copied_bytes=shadow_session.bank_bytes,
                storage_mode='shadow_session', shuffle_seconds=0., storage_seconds=0.,
                shadow_buffer_bytes=0, materialized_bytes=0, materialized_copied_bytes=0)
            self.diagnostics_['preparation_seconds'] += session_seconds
        elif exact and os.environ.get('BANKAI_BORUTA_EXACT_SESSION') == '1':
            from . import _boruta_core
            prep_started = perf_counter()
            session = _boruta_core.ExactSession(X, encoded.astype(np.int64, copy=False))
            session_seconds = perf_counter() - prep_started
            self.diagnostics_.update(session_creation_seconds=session_seconds,
                real_bank_bytes=X.nbytes, real_bank_copied_bytes=X.nbytes,
                storage_mode='virtual_session', validation_seconds=0., shuffle_seconds=0.,
                sorting_seconds=0., storage_seconds=0., native_permutation_copied_bytes=0,
                materialized_bytes=0, materialized_copied_bytes=0)
            self.diagnostics_['preparation_seconds'] += session_seconds
        else:
            self.diagnostics_['storage_mode'] = 'materialized'
        if self.verbose >= 2:
            self.diagnostics_['rounds'] = []
        decision = np.zeros(p, dtype=int)
        hits = np.zeros(p, dtype=int)
        history, thresholds = [], []
        previous, unchanged = None, 0
        permutations = None
        augmented = None
        session_shuffle = session is not None and type(rng) is np.random.RandomState and permutation_dtype == np.uint32
        row_indices = None if session_shuffle or shadow_session is not None else np.arange(len(X), dtype=permutation_dtype)
        self.diagnostics_['permutation_bytes'] = 0 if row_indices is None else row_indices.nbytes
        encoded = encoded.astype(np.int64, copy=False)
        self.n_iter_ = 0
        self.stop_reason_ = "max_iter"
        rounds = range(1, self.max_iter_ + 1) if self.greedy else range(1, self.max_iter_)
        for iteration in rounds:
            round_started = perf_counter()
            if not self.greedy and not np.any(decision == 0):
                self.stop_reason_ = "decided"
                break
            active = np.arange(p) if self.greedy else np.flatnonzero(decision >= 0)
            sources = active.copy()
            while len(sources) < 5:
                sources = np.tile(sources, 2)
            prep_started = perf_counter()
            if shadow_session is not None:
                shadow_seed = int(rng.randint(np.iinfo(np.int64).max, dtype=np.int64))
            elif session_shuffle:
                state = rng.get_state(legacy=False)
                words, position = session.shuffle_uint32(state['state']['key'], state['state']['pos'], len(sources))
                state['state']['key'] = words
                state['state']['pos'] = position
                rng.set_state(state)
            else:
                if permutations is None or permutations.shape != (len(sources), len(X)):
                    permutations = np.empty((len(sources), len(X)), dtype=permutation_dtype)
                    self.diagnostics_['permutation_bytes'] = max(
                        self.diagnostics_['permutation_bytes'], permutations.nbytes + row_indices.nbytes)
                for permutation in permutations:
                    permutation[:] = row_indices
                    _shuffle_permutation(rng, permutation)
            shuffle_seconds = perf_counter() - prep_started
            self.diagnostics_['preparation_seconds'] += shuffle_seconds
            if shadow_session is None:
                self.diagnostics_['shuffle_seconds'] = self.diagnostics_.get('shuffle_seconds', 0.) + shuffle_seconds
            seed = int(rng.randint(np.iinfo(np.int32).max))
            # A warm-start forest must not retain trees trained on old shadows.
            if params.get('warm_start', False):
                self.estimator_ = clone(base)
            updates = {}
            if self.n_estimators is not None:
                updates['n_estimators'] = max(1, int(100 * sqrt(2 * len(active)) / (params.get('max_depth') or 10))) if self.n_estimators == 'auto' else self.n_estimators
            if 'random_state' in params:
                updates['random_state'] = seed
            self.estimator_.set_params(**updates)
            constraints = params.get('monotonic_cst')
            if constraints is not None:
                constraints = np.asarray(constraints)
                if constraints.shape != (p,):
                    raise ValueError("monotonic_cst must have one value per original feature")
                self.estimator_.set_params(monotonic_cst=np.r_[constraints[active], np.zeros(len(sources), dtype=int)])
            if shadow_session is not None:
                settings = self._native_settings_
                scores, prep, train, metrics = shadow_session.gain_importances(
                    active.astype(np.int64), sources.astype(np.int64), shadow_seed,
                    int(self.estimator_.n_estimators), int(params['max_depth'] or 512),
                    int(check_random_state(seed).randint(np.iinfo(np.int32).max)),
                    int(params['extra_seed']), settings['threads'],
                    self._native_max_features(len(active) + len(sources)),
                    settings['criterion'], settings['min_samples_split'], settings['min_samples_leaf'],
                    settings['min_leaf_weight'], settings['min_impurity_decrease'], settings['max_leaves'],
                    settings['bootstrap'], settings['sample_size'], settings['extra_trees'])
                for name in ('shuffle_seconds', 'storage_seconds'):
                    self.diagnostics_[name] += metrics[name]
                self.diagnostics_['shadow_buffer_bytes'] = max(self.diagnostics_['shadow_buffer_bytes'], int(metrics['shadow_buffer_bytes']))
                self.diagnostics_['preparation_seconds'] += prep
                self.diagnostics_['training_seconds'] += train
            elif native:
                from . import _boruta_core
                settings = self._native_settings_
                gain = _boruta_core.gain_importances if session is None else session.gain_importances
                arrays = (X, encoded) if session is None else ()
                forest = {} if session is not None else dict(extra_trees=settings['extra_trees'])
                result = gain(
                    *arrays, active.astype(np.int64), sources.astype(np.int64), permutations,
                    int(self.estimator_.n_estimators), int(params['max_depth'] or 512),
                    int(check_random_state(seed).randint(np.iinfo(np.int32).max)),
                    settings['bins'], int(params['extra_seed']), settings['threads'],
                    self._native_max_features(len(active) + len(sources)), settings['backend'],
                    settings['criterion'], settings['min_samples_split'], settings['min_samples_leaf'],
                    settings['min_leaf_weight'], settings['min_impurity_decrease'], settings['max_leaves'],
                    settings['bootstrap'], settings['sample_size'], settings['binning_strategy'],
                    settings['bin_sample_size'], settings['binning_seed'], **forest)
                scores, prep, train = result[:3]
                if session is not None:
                    for name, value in result[3].items():
                        if name.endswith('_seconds') or name == 'native_permutation_copied_bytes':
                            self.diagnostics_[name] = self.diagnostics_.get(name, 0) + value
                        else:
                            self.diagnostics_[name] = max(self.diagnostics_.get(name, 0), value)
                else:
                    live_bytes = len(X) * (len(active) + len(sources)) * (1 if settings['bins'] is not None else 4)
                    self.diagnostics_['materialized_bytes'] = max(self.diagnostics_.get('materialized_bytes', 0), live_bytes)
                    self.diagnostics_['materialized_copied_bytes'] = self.diagnostics_.get('materialized_copied_bytes', 0) + live_bytes
                self.diagnostics_['preparation_seconds'] += prep
                self.diagnostics_['training_seconds'] += train
                self.diagnostics_['training_storage_rebuilds'] += int(session is None)
            else:
                prep = perf_counter()
                shape = (len(X), len(active) + len(sources))
                if augmented is None or augmented.shape != shape:
                    augmented = np.empty(shape, dtype=X.dtype)
                for column, source in enumerate(active):
                    augmented[:, column] = X[:, source]
                for shadow, source in enumerate(sources):
                    np.take(X[:, source], permutations[shadow], out=augmented[:, len(active) + shadow])
                self.diagnostics_['copied_bytes'] += augmented.nbytes
                copy_seconds = perf_counter() - prep
                self.diagnostics_['copy_seconds'] += copy_seconds
                self.diagnostics_['preparation_seconds'] += copy_seconds
                train = perf_counter()
                kwargs = {} if weights is None else {'sample_weight': weights}
                self.estimator_.fit(augmented, y, **kwargs)
                scores = self._importances()
                self.diagnostics_['training_seconds'] += perf_counter() - train
            scores = np.asarray(scores, dtype=float)
            if scores.shape != (len(active) + len(sources),) or not np.isfinite(scores).all():
                raise ValueError("importances must be a finite vector with one value per augmented feature")
            threshold = np.percentile(scores[len(active):], self.perc)
            current = np.full(p, np.nan)
            current[active] = scores[:len(active)]
            history.append(current)
            thresholds.append(threshold)
            hits[active] += current[active] > threshold
            if self.greedy:
                decision[hits > 0] = 1
            else:
                tested = np.flatnonzero(decision >= 0)
                accept = binom.sf(hits[tested] - 1, iteration, .5)
                reject = binom.cdf(hits[tested], iteration, .5)
                if self.two_step:
                    accept = _fdr(accept, self.alpha) & (accept <= self.alpha / iteration)
                    reject = _fdr(reject, self.alpha) & (reject <= self.alpha / iteration)
                else:
                    accept = accept <= self.alpha / p
                    reject = reject <= self.alpha / p
                undecided = decision[tested] == 0
                decision[tested[accept & undecided]] = 1
                decision[tested[reject & undecided]] = -1
            self.n_iter_ = iteration
            if self.verbose >= 2:
                self.diagnostics_['rounds'].append(dict(iteration=iteration,
                    active_width=len(active), seed=seed, duration_seconds=perf_counter()-round_started,
                    hits=hits.tolist(), decisions=decision.tolist(), stop_reason=None))
            if self.verbose:
                print(f"Iteration {iteration}/{self.max_iter_}: {(decision == 1).sum()} confirmed")
            if not self.greedy and self.early_stopping:
                unchanged = unchanged + 1 if previous is not None and np.array_equal(previous, decision) else 1
                previous = decision.copy()
                if unchanged > self.n_iter_no_change:
                    self.stop_reason_ = "early_stopping"
                    break
        self.support_ = decision == 1
        self.support_weak_ = np.zeros(p, dtype=bool)
        values = np.asarray(history).reshape(-1, p)
        if not self.greedy and len(history):
            tentative = np.flatnonzero(decision == 0)
            self.support_weak_[tentative] = np.median(values[:, tentative], axis=0) > np.median(thresholds)
        self.ranking_ = np.ones(p, dtype=int)
        self.ranking_[self.support_weak_] = 2
        rejected = np.flatnonzero(~(self.support_ | self.support_weak_))
        if len(rejected):
            ranks = np.ones(len(rejected))
            if not self.greedy and len(history):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    medians = np.nanmedian(rankdata(-values[:, rejected], axis=1, nan_policy='omit'), axis=0)
                ranks = rankdata(medians, nan_policy='omit')
                ranks = np.nan_to_num(ranks, nan=len(rejected))
                ranks -= ranks.min() - 1
            self.ranking_[rejected] = ranks + (2 if self.support_weak_.any() else 1)
        self.n_features_ = int(self.support_.sum())
        self.hits_ = hits
        if self.store_history:
            self.importance_history_ = np.vstack((np.zeros(p), values))
            self.shadow_threshold_history_ = np.asarray(thresholds)
        if self.verbose >= 2 and self.diagnostics_['rounds']:
            self.diagnostics_['rounds'][-1]['stop_reason'] = self.stop_reason_
        self.diagnostics_['fit_seconds'] = perf_counter() - started
        return self

    def _native_max_features(self, width):
        # Reuse the pinned forest's validation and exact resolution semantics.
        self.estimator_.n_features_in_ = width
        return self.estimator_._resolve_max_features()

    def _native_eligibility(self, base, params, weights, X):
        from bankai_random_forest import BankaiRandomForestClassifier
        if type(base) is not BankaiRandomForestClassifier or self.importance_getter != 'auto':
            return False, 'external estimator or custom importance getter'
        # The native Rust path accepts estimator controls explicitly. Options
        # not represented at this boundary retain BankaiRF's Python fit path.
        defaults = BankaiRandomForestClassifier(extra_trees=True, extra_trees_backend='binned', max_bins=63, max_depth=6, n_jobs=-1).get_params()
        allowed = {'n_estimators', 'random_state', 'max_depth', 'max_bins', 'extra_seed', 'n_jobs', 'max_features', 'extra_trees',
                   'extra_trees_backend', 'criterion', 'min_samples_split', 'min_samples_leaf',
                   'min_weight_fraction_leaf', 'min_impurity_decrease', 'max_leaf_nodes',
                   'bootstrap', 'max_samples', 'binning_strategy', 'bin_sample_size', 'binning_seed'}
        if weights is not None or any(type(params[k]) is not type(v) or not np.array_equal(params[k], v) for k, v in defaults.items() if k not in allowed):
            return False, 'configuration outside the validated native subset'
        if self.n_estimators is None and (not isinstance(params['n_estimators'], Integral) or isinstance(params['n_estimators'], bool) or params['n_estimators'] < 1):
            return False, 'tree count requires BankaiRF parameter validation'
        if params['max_depth'] is not None and (not isinstance(params['max_depth'], Integral) or isinstance(params['max_depth'], bool) or params['max_depth'] < 1):
            return False, 'depth requires BankaiRF parameter validation'
        if not isinstance(params['extra_seed'], Integral) or isinstance(params['extra_seed'], bool) or not 0 <= params['extra_seed'] < 2**64:
            return False, 'Extra Trees seed requires BankaiRF parameter validation'
        extra_trees = params['extra_trees']
        if not isinstance(extra_trees, bool):
            return False, 'extra_trees requires a bool'
        # Ordinary Random Forests share both shadow sessions: max_bins selects
        # their histogram mode, as in BankaiRF.
        backend = params['extra_trees_backend'] if extra_trees else ('exact' if params['max_bins'] is None else 'binned')
        if backend not in ('exact', 'binned'):
            return False, 'Extra Trees backend outside native subset'
        if extra_trees and backend == 'exact' and params['max_bins'] is not None:
            return False, 'exact Extra Trees requires max_bins=None'
        if backend == 'exact' and X.dtype != np.float32:
            return False, 'exact native forests currently require float32 input'
        if backend == 'binned' and (not isinstance(params['max_bins'], Integral) or isinstance(params['max_bins'], bool) or not 2 <= params['max_bins'] <= 255):
            return False, 'binned forests require max_bins in [2, 255]'
        if params['criterion'] not in ('gini', 'entropy', 'log_loss'):
            return False, 'criterion outside native subset'
        if not isinstance(params['bootstrap'], bool):
            return False, 'bootstrap requires a bool'
        try:
            max_bins = base._resolve_max_bins()
            max_depth = base._resolve_max_depth()
            max_features = self._native_max_features(X.shape[1])
            min_samples_split = base._resolve_min_samples(params['min_samples_split'], 2, 'min_samples_split', len(X))
            min_samples_leaf = base._resolve_min_samples(params['min_samples_leaf'], 1, 'min_samples_leaf', len(X))
            min_leaf_weight = base._validate_min_weight_fraction_leaf(None, len(X))
            min_impurity_decrease = base._resolve_min_impurity_decrease()
            max_leaves = base._resolve_max_leaf_nodes()
            sample_size = base._resolve_max_samples(len(X))
            threads = base._resolve_n_jobs()
            binning_strategy = base._resolve_binning_strategy()
            bin_sample_size = base._resolve_bin_sample_size()
            binning_seed = base._resolve_binning_seed()
        except (TypeError, ValueError, OverflowError) as exc:
            return False, f'BankaiRF parameter validation requires Python fit: {exc}'
        self._native_settings_ = dict(backend=backend, extra_trees=extra_trees, bins=max_bins, criterion=params['criterion'],
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
            min_leaf_weight=min_leaf_weight, min_impurity_decrease=min_impurity_decrease,
            max_leaves=max_leaves, bootstrap=params['bootstrap'], sample_size=sample_size,
            threads=threads, binning_strategy=binning_strategy, bin_sample_size=bin_sample_size,
            binning_seed=binning_seed, max_depth=max_depth, max_features=max_features)
        kind = f'{backend} Extra Trees' if extra_trees else f'{backend} Random Forest'
        return True, f'{kind} gain; native shadows; estimator controls forwarded'

    def _importances(self):
        getter = self.importance_getter
        if callable(getter):
            return getter(self.estimator_)
        if getter == 'auto':
            if hasattr(self.estimator_, 'feature_importances_'):
                return self.estimator_.feature_importances_
            if hasattr(self.estimator_, 'coef_'):
                coef = np.asarray(self.estimator_.coef_)
                return np.abs(coef) if coef.ndim == 1 else np.linalg.norm(coef, axis=0)
            raise ValueError("estimator exposes neither feature_importances_ nor coef_")
        return attrgetter(getter)(self.estimator_)

    def _get_support_mask(self):
        check_is_fitted(self, 'support_')
        return self.support_
