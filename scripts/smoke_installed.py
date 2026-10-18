"""Run from outside the repository against an installed wheel."""
import numpy as np
from bankai_boruta import BankaiBoruta, __version__

assert __version__ == '0.1.0a6'
for dtype in (np.float32, np.float64):
    X = np.random.RandomState(42).normal(size=(80, 4)).astype(dtype)
    y = np.where(X[:, 0] > 0, 'positive', 'negative')
    s = BankaiBoruta(n_estimators=10, greedy=True, random_state=42).fit(X, y)
    assert s.diagnostics_['path'] == 'rust'
    assert s.diagnostics_['native_input_dtype'] == np.dtype(dtype).name
    assert s.diagnostics_['input_conversion_bytes'] == 0
    assert s.diagnostics_['permutation_dtype'] == 'uint32'
    assert s.transform(X).shape[1] == s.n_features_
print('a6 installed wheel float32/float64 smoke passed')
