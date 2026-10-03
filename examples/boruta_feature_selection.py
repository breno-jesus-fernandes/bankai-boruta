"""Select features with Bankai Boruta using Bankai Random Forest."""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from bankai_boruta import BankaiBoruta


def main() -> None:
    # The first 5 features carry signal; the remaining 15 are noise.
    X, y = make_classification(
        n_samples=800,
        n_features=20,
        n_informative=5,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        class_sep=1.2,
        random_state=42,
    )
    feature_names = np.asarray([f"feature_{i:02d}" for i in range(X.shape[1])])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    selector = BankaiBoruta(
        forest_backend="bankai",
        n_estimators=200,
        max_features="sqrt",
        max_iter=100,
        alpha=0.01,
        random_state=42,
        n_jobs=1,
    )
    selector.fit(X_train, y_train)

    # support_ marks confirmed features. support_weak_ marks tentative ones.
    accepted_names = feature_names[selector.support_]
    tentative_names = feature_names[selector.support_weak_]
    print(f"Iterations: {selector.n_iter_} ({selector.stop_reason_})")
    print(f"Accepted features ({len(accepted_names)}): {accepted_names.tolist()}")
    print(f"Tentative features ({len(tentative_names)}): {tentative_names.tolist()}")

    # transform applies the confirmed-feature mask to any compatible matrix.
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    if X_train_selected.shape[1] == 0:
        raise RuntimeError("Boruta did not confirm any features; inspect tentative features")

    from bankai_random_forest import BankaiRandomForestClassifier

    model = BankaiRandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X_train_selected, y_train)
    predictions = model.predict(X_test_selected)
    print(f"Test accuracy with accepted features: {accuracy_score(y_test, predictions):.3f}")


if __name__ == "__main__":
    main()
