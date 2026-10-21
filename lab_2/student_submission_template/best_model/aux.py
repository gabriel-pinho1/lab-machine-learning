"""Auxiliary prediction functions imported by my_model.py."""

from __future__ import annotations

from typing import Any
import pandas as pd
import numpy as np

DATE = "date"
TARGET = "chlorophyll_a_mg_m3"

def build_lagged_frame(raw: pd.DataFrame, lag_spec: dict[str, list[int]] | None = None,
                       require_target: bool = True) -> pd.DataFrame:
    """Return rows indexed by prediction date, never using target lag 0."""
    lag_spec = lag_spec or DEFAULT_LAGS
    if DATE not in raw:
        raise ValueError(f"missing required column: {DATE}")
    if require_target and TARGET not in raw:
        raise ValueError(f"missing required column: {TARGET}")
    frame = raw.copy().sort_values(DATE).reset_index(drop=True)
    columns: dict[str, pd.Series | np.ndarray] = {DATE: pd.to_datetime(frame[DATE])}
    if TARGET in frame:
        columns[TARGET] = frame[TARGET].astype(float)
    for col, lags in lag_spec.items():
        if col not in frame:
            raise ValueError(f"lag specification refers to missing column: {col}")
        for lag in lags:
            if col == TARGET and lag == 0:
                raise ValueError("target lag 0 is forbidden: it leaks the answer")
            columns[f"{col}__lag_{lag}"] = pd.to_numeric(frame[col], errors="coerce").shift(lag)
    day = pd.to_datetime(frame[DATE]).dt.dayofyear.to_numpy()
    columns["annual_sin"] = np.sin(2 * np.pi * day / 365.25)
    columns["annual_cos"] = np.cos(2 * np.pi * day / 365.25)
    out = pd.DataFrame(columns)
    if require_target:
        out = out.dropna(subset=[TARGET])
    max_lag = max(max(v) for v in lag_spec.values())
    return out.iloc[max_lag:].reset_index(drop=True)


high_limit = 15
low_limit = 0

defined_lags = {
    "chlorophyll_a_mg_m3": list(range(1, high_limit)),
    "sst_c": list(range(low_limit, high_limit)),
    "par_umol_m2_s" : list(range(low_limit,high_limit)), 
    "nitrate_umol_l" : list(range(low_limit,high_limit)),
    "wind_speed_m_s" : list(range(low_limit,high_limit)),
    "upwelling_index" : list(range(low_limit,high_limit)),
    "mixed_layer_depth_m" : list(range(low_limit,high_limit)),
    "salinity_psu" : list(range(low_limit,high_limit)),
    "current_speed_m_s" : list(range(low_limit,high_limit)),
    "river_discharge_index" : list(range(low_limit,high_limit)),
    "cloud_fraction" : list(range(low_limit,high_limit)),
    "surface_pressure_hpa" : list(range(low_limit,high_limit)),
    "turbidity_ntu" : list(range(low_limit,high_limit)),
    "nitrate_x_par" : list(range(low_limit,high_limit)),
    "wind_x_upwelling" : list(range(low_limit,high_limit)),
    # O Lasso não eliminou algumas destas features mas o R^2 de validação piorou ligeiramente
    "river_div_turbidity" : list(range(low_limit,high_limit)),
    #"sst_c_x_nitrate" : list(range(low_limit,high_limit))
    #"sst_c": [1, 2, 3],                     # Temperatura de ontem, anteontem e há 3 dias
    #"wind_speed_m_s": [1, 7],               # Vento de ontem e da semana passada (exemplo)
    # Podes adicionar mais variáveis aqui...
    }

def one_day_row(history: pd.DataFrame, day_features: pd.DataFrame,
                lag_spec: dict[str, list[int]] | None = None) -> pd.DataFrame:
    """Construct exactly one feature row from observed history plus today's exogenous data."""
    if len(day_features) != 1:
        raise ValueError("day_features must contain exactly one row")
    today = day_features.copy()
    if TARGET not in today:
        today[TARGET] = np.nan
    combined = pd.concat([history, today], ignore_index=True, sort=False)

    combined[TARGET] = np.log1p(combined[TARGET].astype(float))
    exogenous_cols_mv = ["wind_speed_m_s", "upwelling_index", "current_speed_m_s", "cloud_fraction", "turbidity_ntu"]

    window_size = 3
    
    for feature in (exogenous_cols_mv):
        combined[feature] = combined[feature].rolling(window=window_size, min_periods=1).mean()

    combined["nitrate_x_par"] = combined["nitrate_umol_l"] * combined["par_umol_m2_s"]
    combined["wind_x_upwelling"] = combined["wind_speed_m_s"] * combined["upwelling_index"]
    combined["river_div_turbidity"] = combined["river_discharge_index"] / (combined["turbidity_ntu"] + 1)


    lagged = build_lagged_frame(combined, lag_spec=lag_spec, require_target=False)
    row = lagged.tail(1).drop(columns=[DATE, TARGET], errors="ignore")
    if len(row) != 1:
        raise ValueError("insufficient history to construct the requested lags")
    return row