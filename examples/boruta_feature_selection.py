from sklearn.datasets import make_classification
from bankai_boruta import BankaiBoruta

X, y = make_classification(n_samples=500, n_features=20, random_state=42)
selector = BankaiBoruta(random_state=42, greedy=True)
print(selector.fit_transform(X, y).shape)
print(selector.diagnostics_)
