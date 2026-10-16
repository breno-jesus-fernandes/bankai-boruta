"""Run from outside the repository against an installed wheel."""
import numpy as np
from bankai_boruta import BankaiBoruta, __version__
assert __version__ == '0.1.0a3'
rng = np.random.RandomState(42)
X = rng.normal(size=(80, 4))
y = np.where(X[:, 0] > 0, 'positive', 'negative')
s = BankaiBoruta(n_estimators=10, greedy=True, random_state=42).fit(X, y)
assert s.diagnostics_['path'] == 'rust'
assert s.transform(X).shape[1] == s.n_features_
print('a3 wheel smoke passed')
