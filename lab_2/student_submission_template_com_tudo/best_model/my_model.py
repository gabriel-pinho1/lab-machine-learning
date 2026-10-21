"""Structurally valid zero-prediction example; replace it with the final model."""

from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any

import numpy as np
import pandas as pd

from aux import one_day_row


def load_model(model_path: str | Path) -> Any:
    """Load and return the submitted fitted model artifact."""
    with Path(model_path).open("rb") as handle:
        return pickle.load(handle)


def predict(model: Any, history_df: pd.DataFrame, day_features_df: pd.DataFrame) -> np.ndarray:
    """Return the dummy model's constant as a NumPy array with shape (1,).

    history_df contains all observations strictly before the prediction date,
    including past chlorophyll. day_features_df contains exactly the current
    day's exogenous variables and date, but never current-day chlorophyll.

    This example deliberately does not use either DataFrame. Students must
    replace it with their lag construction, saved input preprocessing, and
    prediction procedure. Fitted sklearn.preprocessing input objects are
    allowed; library pipelines and output-processing wrappers are not.
    """

    scaler = model["scaler"]
    kept_features = model["kept_features"]
    clf_ridge = model["model"]

    X_today = one_day_row(history_df, day_features_df)
    
    X_scaled = pd.DataFrame(scaler.transform(X_today), columns=X_today.columns).clip(-3, 3)
    
    X_cut = X_scaled.loc[:, kept_features]
    
    pred_log = clf_ridge.predict(X_cut)
    
    pred_real = np.expm1(pred_log)

    final_pred = np.nan_to_num(pred_real, nan=0.0, posinf=0.0, neginf=0.0)
    
    return np.asarray(final_pred, dtype=float).reshape(1,)

