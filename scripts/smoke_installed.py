"""Run from outside the repository against an installed wheel."""
import numpy as np
from bankai_boruta import BankaiBoruta, __version__

assert __version__ == '0.1.0a8'
for dtype in (np.float32, np.float64):
    X = np.random.RandomState(42).normal(size=(80, 4)).astype(dtype)
    y = np.where(X[:, 0] > 0, 'positive', 'negative')
    s = BankaiBoruta(n_estimators=10, greedy=True, random_state=42).fit(X, y)
    assert s.diagnostics_['path'] == 'rust'
    assert s.diagnostics_['native_input_dtype'] == np.dtype(dtype).name
    assert s.diagnostics_['input_conversion_bytes'] == 0
    assert s.diagnostics_['permutation_dtype'] == 'uint32'
    assert s.transform(X).shape[1] == s.n_features_
import os
from bankai_random_forest import BankaiRandomForestClassifier
os.environ['BANKAI_BORUTA_EXACT_SESSION'] = '1'
X = np.random.RandomState(42).normal(size=(80, 4)).astype(np.float32)
y = (X[:, 0] > 0).astype(int)
base = BankaiRandomForestClassifier(extra_trees=True, extra_trees_backend='exact',
    n_estimators=3, max_depth=3, n_jobs=1)
s = BankaiBoruta(base, n_estimators=None, max_iter=3, random_state=42).fit(X, y)
assert s.diagnostics_['storage_mode'] == 'virtual_session'
assert s.diagnostics_['materialized_bytes'] == 0
assert s.diagnostics_['native_permutation_copied_bytes'] == 0
print('a7 installed wheel binned dtypes and exact session smoke passed')
