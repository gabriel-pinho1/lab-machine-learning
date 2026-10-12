"""Auxiliary prediction functions imported by my_model.py."""

from __future__ import annotations

from typing import Any
import pandas as pd

def constant_prediction(model: Any) -> float:
    """Extract the dummy model's constant prediction."""
    return float(model["constant_prediction"])


import pandas as pd

def one_day_row(history: pd.DataFrame, day_features: pd.DataFrame,
                lag_spec: dict[str, list[int]] | None = None) -> pd.DataFrame:
    """Construct exactly one feature row from observed history plus today's exogenous data."""
    if len(day_features) != 1:
        raise ValueError("day_features must contain exactly one row")
    today = day_features.copy()
    if TARGET not in today:
        today[TARGET] = np.nan
    combined = pd.concat([history, today], ignore_index=True, sort=False)
    lagged = build_lagged_frame(combined, lag_spec=lag_spec, require_target=False)
    row = lagged.tail(1).drop(columns=[DATE, TARGET], errors="ignore")
    if len(row) != 1:
        raise ValueError("insufficient history to construct the requested lags")
    return row