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
from build_lagged_features import build_lagged_frame, read_table
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
TARGET = "chlorophyll_a_mg_m3"
NEW_LAGS = {
    TARGET: list(range(1, 15)),
    "sst_c": list(range(0, 15)),
    "par_umol_m2_s": list(range(0, 15)),
    "nitrate_umol_l": list(range(0, 15)),
    "wind_speed_m_s": list(range(0, 15)),
    "upwelling_index": list(range(0, 15)),
    "mixed_layer_depth_m": list(range(0, 15)),
    "salinity_psu": list(range(0, 15)),
    "current_speed_m_s": list(range(0, 15)),
    "river_discharge_index": list(range(0, 15)),
    "cloud_fraction": list(range(0, 15)),
    "surface_pressure_hpa": list(range(0, 15)),
    "turbidity_ntu": list(range(0, 15)),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.parse_args()
    
    args = parser.parse_args()
    
    df_raw = read_table(args.data)
       
    lagged_df = build_lagged_frame(df_raw, NEW_LAGS)
    Y = lagged_df["chlorophyll_a_mg_m3"]
    X = lagged_df.drop(columns=["date", "chlorophyll_a_mg_m3"])
    split_idx = -365
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    Y_train, Y_val = Y.iloc[:split_idx], Y.iloc[split_idx:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns)

    
        
    r2_best = 0
    j_best = 0
    for k in [0.0029]:
        j = k
        model = linear_model.Lasso(alpha=j)
        #model = linear_model.LinearRegression()
        model.fit(X_train_scaled, Y_train)
        predictions = model.predict(X_val_scaled)

        r2 = r2_score(Y_val, predictions)
        mae = mean_absolute_error(Y_val, predictions)

        print(f'alpha={j}')
        print(f"Validation R^2: {r2:.4f}")
        print(f"Validation MAE: {mae:.4f} mg/m3")
        coefficients = pd.Series(model.coef_, index=X_train_scaled.columns)
        zeroed_features = coefficients[coefficients == 0.0]
        kept_features = coefficients[coefficients != 0.0]
        print("--- Features Driven to Zero ---")
        print(zeroed_features)

        print("\n--- Features Kept in the Model ---")
        print(kept_features)    
        if r2 > r2_best:
            r2_best = r2
            j_best = j

    print(f'alpha_best={j_best}')
    print(f"Validation R^2_best: {r2_best:.4f}")
    X_train_scaled.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()

# linha do terminal pra correr codigo atual:
# python train.py --data ../../data/chlorophyll_student_2015_2023.xlsx --output cleaned_data.csv