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
    df_raw["chlorophyll_a_mg_m3"] = np.log1p(df_raw["chlorophyll_a_mg_m3"])
    lagged_df = build_lagged_frame(df_raw, NEW_LAGS)
    Y = lagged_df["chlorophyll_a_mg_m3"]
    X = lagged_df.drop(columns=["date", "chlorophyll_a_mg_m3"])
    raw_columns_names = [col for col in df_raw.columns if col != "date"]
    df_lag_sum = pd.DataFrame()
    new_columns = {}

    for name in raw_columns_names:
        for p in range(1,15):
            if name == "chlorophyll_a_mg_m3":
                
                cols_to_sum = [f"{name}__lag_{i}" for i in range(1, p)]
            else:
                
                cols_to_sum = [f"{name}__lag_{i}" for i in range(0, p)]


            new_columns[f"row_sum_{name}_{p}_days"] = X[cols_to_sum].sum(axis=1)
    df_lag_sum = pd.DataFrame(new_columns)            
    X = pd.concat([X, df_lag_sum], axis = 1)

    split_idx = -365
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    Y_train_log, Y_val_log = Y.iloc[:split_idx], Y.iloc[split_idx:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns)
   

    


    
        
    r2_best = 0
    j_best = 0
    mae_best = 0

    alphas_to_test = np.logspace(-3, 1, 50)

    for j in alphas_to_test:
        
        model = linear_model.Lasso(alpha=j)
        #model = linear_model.LinearRegression()
        model.fit(X_train_scaled, Y_train_log)
        predictions_log = model.predict(X_val_scaled)

        real_predictions = np.expm1(predictions_log)
        Y_val_real = np.expm1(Y_val_log)

        r2 = r2_score(Y_val_real, real_predictions)
        mae = mean_absolute_error(Y_val_real, real_predictions)

        print(f'alpha={j:.4f} | R^2: {r2:.4f} | MAE: {mae:.4f}')
        
        
        if r2 > r2_best:
            r2_best = r2
            j_best = j
            mae_best = mae
            coefficients = pd.Series(model.coef_, index=X_train_scaled.columns)
            
    zeroed_features = coefficients[coefficients == 0.0]
    kept_features = coefficients[coefficients != 0.0]
    print(f'alpha_best={j_best}')
    print(f"Validation R^2_best: {r2_best:.4f}")
    print(f"Validation MAE best: {mae_best:.4f} mg/m3")
    print("--- Features Driven to Zero ---")
    print(zeroed_features)
    print("\n--- Features Kept in the Model ---")
    print(kept_features) 

    X_train_scaled.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()

# linha do terminal pra correr codigo atual:
# python train.py --data ../../data/chlorophyll_student_2015_2023.xlsx --output cleaned_data.csv