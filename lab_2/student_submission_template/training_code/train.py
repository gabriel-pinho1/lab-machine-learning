"""Replace this template with the code used to train and save best_model/model.pkl.

The final script should reproduce raw-data loading, chronological splitting,
lag construction, preprocessing, model selection, final fitting, and model
serialization. sklearn.preprocessing tools may be fitted on input features
using history data only and then saved for prediction. Keep all supporting
Python modules in history_code/. Do not use library pipelines, composition
wrappers, imputers, feature selectors, or output transformers. Target
transformation and output post-processing must remain explicit.
"""

from __future__ import annotations
from build_lagged_features import build_lagged_frame, one_day_row
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.parse_args()
    
    args = parser.parse_args()
    
    history_df = pd.read_excel(
        args.data, 
        sheet_name="daily_data"
    )
    DATE = "date"
    TARGET = "chlorophyll_a_mg_m3"
    for p in range (10,15):
        for q in range(8,15):
            Lag_spec = {
            TARGET: list(range(1, p)),
            "sst_c": list(range(0, p)),
            "par_umol_m2_s": list(range(0, q)),
            "nitrate_umol_l": list(range(0, p)),
            "wind_speed_m_s": list(range(0, q)),
            "upwelling_index": list(range(0, p)),
            "mixed_layer_depth_m": list(range(0, q)),
            "salinity_psu": list(range(0, q)),
            "current_speed_m_s": list(range(0, q)),
            "river_discharge_index": list(range(0, q)),
            "cloud_fraction": list(range(0, q)),
            "surface_pressure_hpa": list(range(0, q)),
            "turbidity_ntu": list(range(0, q)),
            }
            lagged_df = build_lagged_frame(history_df, Lag_spec)
            Y = lagged_df["chlorophyll_a_mg_m3"]
            X = lagged_df.drop(columns=["date", "chlorophyll_a_mg_m3"])
            split_idx = -365
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            Y_train, Y_val = Y.iloc[:split_idx], Y.iloc[split_idx:]

            model = LinearRegression()
            model.fit(X_train, Y_train)
            predictions = model.predict(X_val)

            r2 = r2_score(Y_val, predictions)
            mae = mean_absolute_error(Y_val, predictions)

            print(f"Validation R^2: {r2:.4f}")
            print(f"Validation MAE: {mae:.4f} mg/m3")

    X.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()

# linha do terminal pra correr codigo atual:
# python train.py --data ../../data/chlorophyll_student_2015_2023.xlsx --output cleaned_data.csv