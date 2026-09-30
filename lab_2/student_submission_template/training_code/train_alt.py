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
import pickle
import itertools
import copy
import gc
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
TARGET = "chlorophyll_a_mg_m3"
lag_vector = [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15]
NEW_LAGS = {
    TARGET: list(range(1, lag_vector[0])),
    "sst_c": list(range(0, lag_vector[1])),
    "par_umol_m2_s": list(range(0, lag_vector[2])),
    "nitrate_umol_l": list(range(0, lag_vector[3])),
    "wind_speed_m_s": list(range(0, lag_vector[4])),
    "upwelling_index": list(range(0, lag_vector[5])),
    "mixed_layer_depth_m": list(range(0, lag_vector[6])),
    "salinity_psu": list(range(0, lag_vector[7])),
    "current_speed_m_s": list(range(0, lag_vector[8])),
    "river_discharge_index": list(range(0, lag_vector[9])),
    "cloud_fraction": list(range(0, lag_vector[10])),
    "surface_pressure_hpa": list(range(0, lag_vector[11])),
    "turbidity_ntu": list(range(0, lag_vector[12])),
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
    raw_columns_names = [f"{col}__lag_0" for col in df_raw.drop(columns=["date", "chlorophyll_a_mg_m3"]).columns]
    
    raw_columns_names.append("chlorophyll_a_mg_m3__lag_1")

    scaler = StandardScaler()
    df_lagged_copy = lagged_df.drop(columns=["date"]).copy()
    

    new_columns = {}
    for name, name2 in itertools.combinations_with_replacement(raw_columns_names, 2):
        
       new_columns[f"row_mul_{name}_{name2}"] = df_lagged_copy[f"{name}"] * df_lagged_copy[f"{name2}"]

    df_new_features = pd.DataFrame(new_columns)


    lagged_df = pd.concat([lagged_df, df_new_features], axis=1)
    
    Y = lagged_df["chlorophyll_a_mg_m3"]
    X = lagged_df.drop(columns=["date", "chlorophyll_a_mg_m3"])
    raw_columns_names = [col for col in df_raw.columns if col != "date"]
    df_lag_sum = pd.DataFrame()
    new_columns = {}


   
    for idx, name in enumerate(raw_columns_names):
        
        max_lag = lag_vector[idx]
        
        for p in range(1, max_lag):
            if name == "chlorophyll_a_mg_m3":
              
                cols_to_sum = [f"{name}__lag_{i}" for i in range(1, p)]
            else:
                cols_to_sum = [f"{name}__lag_{i}" for i in range(0, p)]

 
            new_columns[f"row_sum_{name}_{p}_days"] = X[cols_to_sum].sum(axis=1)
                
    df_lag_sum = pd.DataFrame(new_columns)            
    X = pd.concat([X, df_lag_sum], axis=1)
    split_idx = -365
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    Y_train_log, Y_val_log = Y.iloc[:split_idx], Y.iloc[split_idx:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns)
   
    del df_raw, df_lagged_copy, lagged_df, df_new_features, df_lag_sum, X, Y
    gc.collect()
    


    
        
    r2_best_lasso = 0
    j_best_lasso = 0
    mae_best_lasso = 0

    alphas_to_test = np.logspace(-3, 1, 50)

    for j in alphas_to_test:
        
        model = linear_model.Lasso(alpha=j)
        model.fit(X_train_scaled, Y_train_log)
        predictions_log = model.predict(X_val_scaled)

        real_predictions = np.expm1(predictions_log)
        Y_val_real = np.expm1(Y_val_log)

        r2 = r2_score(Y_val_real, real_predictions)
        mae = mean_absolute_error(Y_val_real, real_predictions)

        print(f'alpha={j:.4f} | R^2: {r2:.4f} | MAE: {mae:.4f}')
        
        
        if r2 > r2_best_lasso:
            r2_best_lasso = r2
            j_best_lasso = j
            mae_best_lasso = mae
            coefficients_lasso = pd.Series(model.coef_, index=X_train_scaled.columns)
            best_lasso_model = copy.deepcopy(model)
 

    r2_best_ridge = 0
    j_best_ridge = 0
    mae_best_ridge = 0
    alphas_ridge =[1.e-06, 1.e-05, 1.e-04, 1.e-03, 1.e-02, 1.e-01, 1.e+00, 1.e+01,
      1.e+02, 1.e+03, 1.e+04, 1.e+05, 1.e+06]
    
    for j in alphas_ridge:
        
        model = linear_model.Ridge(alpha=j)
        model.fit(X_train_scaled, Y_train_log)
        predictions_log = model.predict(X_val_scaled)

        real_predictions = np.expm1(predictions_log)
        Y_val_real = np.expm1(Y_val_log)

        r2 = r2_score(Y_val_real, real_predictions)
        mae = mean_absolute_error(Y_val_real, real_predictions)

        print(f'alpha={j:.4f} | R^2: {r2:.4f} | MAE: {mae:.4f}')
        
        
        if r2 > r2_best_ridge:
            r2_best_ridge = r2
            j_best_ridge = j
            mae_best_ridge = mae
            coefficients_ridge = pd.Series(model.coef_, index=X_train_scaled.columns)
            best_ridge_model = copy.deepcopy(model)


    r2_best_Elastic = 0
    j_best_Elastic = 0
    mae_best_Elastic = 0
    l1_best_Elastic = 0
    alphas_Elastic = np.logspace(-3, 1, 50) 
    l1_ratios_to_test = [0.05, 0.1, 0.5, 0.7, 0.9, 0.99]
  
    for j in alphas_Elastic:
        for k in l1_ratios_to_test:
        
            model = linear_model.ElasticNet(alpha=j, l1_ratio = k, max_iter=50000)
            model.fit(X_train_scaled, Y_train_log)
            predictions_log = model.predict(X_val_scaled)

            real_predictions = np.expm1(predictions_log)
            Y_val_real = np.expm1(Y_val_log)

            r2 = r2_score(Y_val_real, real_predictions)
            mae = mean_absolute_error(Y_val_real, real_predictions)

            print(f'alpha={j:.4f} | L1 = {k:.4f} | R^2: {r2:.4f} | MAE: {mae:.4f}')
            
            
            if r2 > r2_best_Elastic:
                r2_best_Elastic = r2
                j_best_Elastic = j
                l1_best_Elastic = k
                mae_best_Elastic = mae
                coefficients_Elastic = pd.Series(model.coef_, index=X_train_scaled.columns)
                best_Elastic_model = copy.deepcopy(model)







    zeroed_features_lasso = coefficients_lasso[coefficients_lasso == 0.0]
    kept_features_lasso = coefficients_lasso[coefficients_lasso != 0.0]
    print(f'\n----lasso-----\n')
    print(f'alpha_best={j_best_lasso}')
    print(f"Validation R^2_best: {r2_best_lasso:.4f}")
    print(f"Validation MAE best: {mae_best_lasso:.4f} mg/m3")
    print("--- Features Driven to Zero ---")
    print(zeroed_features_lasso)
    print("\n--- Features Kept in the Model ---")
    print(kept_features_lasso)

    importance_ridge = coefficients_ridge.abs()
    sorted_importance = importance_ridge.sort_values(ascending=False)
    print(f'\n----ridge-----\n')
    print(f'alpha_best={j_best_ridge}')
    print(f"Validation R^2_best: {r2_best_ridge:.4f}")
    print(f"Validation MAE best: {mae_best_ridge:.4f} mg/m3")
    print("\n--- Top 20 Most Important Features ---")
    print(sorted_importance.head(20))

            
    importance_Elastic = coefficients_Elastic.abs()
    sorted_importance_Elastic = importance_Elastic.sort_values(ascending=False)
    zeroed_features_Elastic = coefficients_Elastic[coefficients_Elastic == 0.0]
    print(f'\n----Elastic-----\n')
    print(f'alpha_best={j_best_Elastic}')
    print(f'L1_best={l1_best_Elastic}')
    print(f"Validation R^2_best: {r2_best_Elastic:.4f}")
    print(f"Validation MAE best: {mae_best_Elastic:.4f} mg/m3")
    print("\n--- Top 20 Most Important Features ---")
    print(sorted_importance_Elastic.head(20))
    print("--- Features Driven to Zero ---")
    print(zeroed_features_Elastic)

    artifacts = {
        "lasso": best_lasso_model,
        "ridge": best_ridge_model,
        "elasticnet": best_Elastic_model,
        "scaler": scaler,
        "feature_names": X_train_scaled.columns.tolist()
    }

    # Save models next to the output data file
    model_output_path = args.output.parent / "best_models.pkl"
    with open(model_output_path, "wb") as f:
        pickle.dump(artifacts, f)
        
    print(f"Models and scaler successfully saved to: {model_output_path}")

if __name__ == "__main__":
    main()

# linha do terminal pra correr codigo atual:
# python train_alt.py --data ../../data/chlorophyll_student_2015_2023.xlsx --output models

