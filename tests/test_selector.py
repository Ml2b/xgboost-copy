"""Tests del selector de features."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from features.selector import FeatureSelector, SelectorConfig


def test_selector_keeps_at_least_one_feature() -> None:
    n = 400
    base = pd.Series(range(n), dtype=float)
    X = pd.DataFrame(
        {
            "signal_a": base / n,
            "signal_b": (base / n) + 0.001,
            "noise_a": ((base * 7) % 11) / 10.0,
            "noise_b": ((base * 13) % 17) / 10.0,
        }
    )
    y = (((base % 24) / 24.0) + X["signal_a"] * 0.15 > 0.7).astype(int).to_numpy()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, shuffle=False)

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    selector = FeatureSelector(SelectorConfig(verbose=False, max_features=3))
    result = selector.select(X_train, y_train, X_val, y_val, model, fold_id=1)

    assert result.features_final
    assert len(result.features_final) <= 3
