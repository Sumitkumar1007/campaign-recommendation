from __future__ import annotations

import sys

import numpy as np
import pandas as pd


class ConstantDayModel:
    def __init__(self, encoded_value: int = 0, class_count: int = 1) -> None:
        self.encoded_value = int(encoded_value)
        self.class_count = max(int(class_count), 1)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.encoded_value, dtype=int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probabilities = np.zeros((len(X), self.class_count), dtype=float)
        probabilities[:, self.encoded_value] = 1.0
        return probabilities


def register_legacy_joblib_aliases() -> None:
    """Expose fallback classes on __main__ for old joblib bundles."""
    main_module = sys.modules.get("__main__")
    if main_module is not None and not hasattr(main_module, "ConstantDayModel"):
        setattr(main_module, "ConstantDayModel", ConstantDayModel)
