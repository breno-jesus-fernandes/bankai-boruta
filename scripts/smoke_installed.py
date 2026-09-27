"""Run outside the checkout against an installed wheel/sdist; no pytest required."""
import importlib.util
import pickle
import sys

import numpy as np
from bankai_boruta import BankaiBoruta, _core

assert _core.ENGINE_READY
X = np.column_stack([np.tile([0., 1.], 64), np.arange(128, dtype=float)])
y = X[:, 0].astype(int)
model = BankaiBoruta(n_estimators=32, max_iter=10, alpha=0.05, random_state=4).fit(X, y)
assert model.support_[0]
np.testing.assert_array_equal(pickle.loads(pickle.dumps(model)).transform(X), model.transform(X))
if '--without-polars' in sys.argv:
    assert importlib.util.find_spec('polars') is None
else:
    import joblib
    import polars as pl
    import tempfile
    from pathlib import Path
    columns = pl.DataFrame({'signal': X[:, 0], 'noise': X[:, 1]})
    columns = pl.concat([columns.slice(0, 33), columns.slice(33)], rechunk=False)
    fitted = BankaiBoruta(n_estimators=32, max_iter=10, alpha=0.05, random_state=4).fit(columns, y)
    np.testing.assert_array_equal(fitted.decision_, model.decision_)
    assert fitted.diagnostics_['copied_bytes'] == 0
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'model.joblib'
        joblib.dump(fitted, path)
        restored = joblib.load(path)
        np.testing.assert_array_equal(restored.transform(columns), fitted.transform(columns))
print('installed artifact smoke passed', sys.version.split()[0])
