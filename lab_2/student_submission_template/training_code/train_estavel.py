"""Replace this template with the code used to train and save best_model/model.pkl.

The final script should reproduce raw-data loading, chronological splitting,
lag construction, preprocessing, model selection, final fitting, and model
serialization. sklearn.preprocessing tools may be fitted on input features
using training data only and then saved for prediction. Keep all supporting
Python modules in training_code/. Do not use library pipelines, composition
wrappers, imputers, feature selectors, or output transformers. Target
transformation and output post-processing must remain explicit.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import gc
import pandas as pd
from scipy.stats import zscore

from build_lagged_features import read_table, build_lagged_frame

from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_absolute_error

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=False, type=Path)
    # Tenho de voltar a meter True no final
    parser.add_argument("--output", required=False, type=Path)

    # Guarda os inputs do terminal nesta variável
    args = parser.parse_args()
 #   raise NotImplementedError("Replace training_code/train.py with your training workflow")


    best_alpha_for_lag_lasso = []
    best_r2_for_lag_lasso = []
    best_alpha_for_lag_ridge = []
    best_r2_for_lag_ridge = []

    # Meter aqui o for
    high_limit_vector = range(4, 10)

    #for high_limit in high_limit_vector:
    high_limit = 7
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
    #"sst_c": [1, 2, 3],                     # Temperatura de ontem, anteontem e há 3 dias
    #"wind_speed_m_s": [1, 7],               # Vento de ontem e da semana passada (exemplo)
    # Podes adicionar mais variáveis aqui...
    }

    df_raw = read_table(args.data)
    df_lagged = build_lagged_frame(df_raw, lag_spec=defined_lags)

    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()

    Y = df_lagged["chlorophyll_a_mg_m3"]
    X = df_lagged.drop(columns = ["date", "chlorophyll_a_mg_m3"])

    #separating dataset into training, validation, and testing
    lower_separation_idx = -3*365
    higher_separation_idx = -1*365

    X_train, X_val, X_test = X.iloc[:lower_separation_idx], X.iloc[lower_separation_idx:higher_separation_idx], X.iloc[higher_separation_idx:]
    Y_train, Y_val, Y_test = Y.iloc[:lower_separation_idx], Y.iloc[lower_separation_idx:higher_separation_idx], Y.iloc[higher_separation_idx:]


    # Aplicar o fit apenas aos dados de treino e validacao
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    X_val_scaled = scaler_X.fit_transform(X_val) 


    # COnverter em panda outra vez para reter o nome das colunas

    X_train_scaled = pd.DataFrame(X_train_scaled, columns= X_train.columns)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns= X_val.columns)    
    X_test_scaled = pd.DataFrame(X_test_scaled, columns= X_test.columns)

    # Estou a definir que todos os valores cujo o resultado do fit.transform estejam para lá de -3 e 3 sejam 
    # limitados a -3 e 3 (porque estatisticamente só os outliers estão fora desse valor)

    for col in X_train_scaled.columns:
            
            if col.startswith("chlorophyll_a_mg_m3__lag_"):
                # 1. Identify where the outliers are
                train_outliers = (X_train_scaled[col] < -3) | (X_train_scaled[col] > 3)
                val_outliers = (X_val_scaled[col] < -3) | (X_val_scaled[col] > 3)
                test_outliers = (X_test_scaled[col] < -3) | (X_test_scaled[col] > 3)

                train_avg_3_above = X_train_scaled[col].rolling(window=3, min_periods=1).mean().shift(1)
                val_avg_3_above = X_val_scaled[col].rolling(window=3, min_periods=1).mean().shift(1)
                test_avg_3_above = X_test_scaled[col].rolling(window=3, min_periods=1).mean().shift(1)

                X_train_scaled.loc[train_outliers, col] = train_avg_3_above[train_outliers]
                X_val_scaled.loc[val_outliers, col] = val_avg_3_above[val_outliers]
                X_test_scaled.loc[test_outliers, col] = test_avg_3_above[test_outliers]
                
                X_train_scaled[col] = X_train_scaled[col].fillna(X_train_scaled[col].clip(-3, 3))
                X_val_scaled[col] = X_val_scaled[col].fillna(X_val_scaled[col].clip(-3, 3)) 
                X_test_scaled[col] = X_test_scaled[col].fillna(X_test_scaled[col].clip(-3, 3))

            else:
                X_train_scaled[col] = X_train_scaled[col].clip(-3, 3)
                X_val_scaled[col] = X_val_scaled[col].clip(-3, 3)
                X_test_scaled[col] = X_test_scaled[col].clip(-3, 3)



    y_train_zscores = zscore(Y_train)
    y_val_zscores = zscore(Y_val)

    train_outliers = (y_train_zscores < -3) | (y_train_zscores > 3)
    val_outliers = (y_val_zscores < -3) | (y_val_zscores > 3)

    train_avg_3_above = Y_train.rolling(window=3, min_periods=1).mean().shift(1)
    val_avg_3_above = Y_val.rolling(window=3, min_periods=1).mean().shift(1)

    Y_train.loc[train_outliers] = train_avg_3_above[train_outliers]
    Y_val.loc[val_outliers] = val_avg_3_above[val_outliers]


    Y_train = Y_train.bfill()
    Y_val = Y_val.bfill()


    #model validation
    vector_length_lasso = 20
    vector_length_ridge = 20

    best_r2_lasso = -float('inf')  
    best_alpha_lasso = None

    alpha_list_lasso = []
    r2_list_lasso = []

    alpha_list_ridge = []
    r2_list_ridge = []

    alpha_lasso_defined = np.logspace(-4, -2, vector_length_lasso)
    alpha_ridge_defined = np.linspace(1e-16, 1e-15, vector_length_ridge)

    for alpha in alpha_lasso_defined:

        clf_lasso = linear_model.Lasso(alpha=alpha, max_iter=150000)
        clf_lasso.fit(X_train_scaled, Y_train)
        

        predicts_lasso = clf_lasso.predict(X_val_scaled)

        r2_lasso = r2_score(Y_val, predicts_lasso)

        if r2_lasso > best_r2_lasso:
            best_r2_lasso = r2_lasso
            best_alpha_lasso = alpha
            mae_best_lasso = mean_absolute_error(Y_val, predicts_lasso)
            coefficients_lasso = pd.Series(clf_lasso.coef_, index=X_train_scaled.columns)


    print(f"The best Lasso model has alpha={best_alpha_lasso} with a R2 of {best_r2_lasso})")

    best_r2_ridge = -float('inf')  
    best_alpha_ridge = None

    for alpha in alpha_ridge_defined:

        clf_ridge = linear_model.Ridge(alpha)
        clf_ridge.fit(X_train_scaled, Y_train)

        predicts_ridge = clf_ridge.predict(X_val_scaled)

        r2_ridge = r2_score(Y_val, predicts_ridge)



        if r2_ridge > best_r2_ridge:
            best_r2_ridge = r2_ridge
            best_alpha_ridge = alpha
            mae_best_ridge = mean_absolute_error(Y_val, predicts_ridge)
            coefficients_ridge = pd.Series(clf_ridge.coef_, index=X_train_scaled.columns)


    #ridge after lasso validation
    kept_features_lasso = coefficients_lasso != 0.0
    
    X_train_cut = X_train_scaled.loc[:, kept_features_lasso]
    X_test_cut =  X_test_scaled.loc[:, kept_features_lasso]
    X_val_cut =  X_val_scaled.loc[:, kept_features_lasso]

    best_r2_ridge_lasso = -float('inf')  
    best_alpha_ridge_lasso = None

    for alpha in alpha_ridge_defined:

            clf_ridge = linear_model.Ridge(alpha)
            clf_ridge.fit(X_train_cut, Y_train)

            predicts_ridge_lasso = clf_ridge.predict(X_val_cut)

    
            r2_ridge_lasso = r2_score(Y_val, predicts_ridge_lasso)
    
 
    
            if r2_ridge_lasso > best_r2_ridge_lasso:
                best_r2_ridge_lasso = r2_ridge_lasso
                best_alpha_ridge_lasso = alpha
                mae_best_ridge_lasso = mean_absolute_error(Y_val, predicts_ridge_lasso)
                coefficients_ridge_lasso = pd.Series(clf_ridge.coef_, index=X_train_cut.columns)


    #testing:

    X_train_full = pd.concat([X_train_scaled, X_val_scaled])
    Y_train_full = pd.concat([Y_train, Y_val])


    clf_lasso = linear_model.Lasso(alpha=best_alpha_lasso, max_iter=150000)
    clf_lasso.fit(X_train_full, Y_train_full)
    predicts_lasso = clf_lasso.predict(X_test_scaled)


    r2_lasso_test = r2_score(Y_test, predicts_lasso)
    mae_lasso_test = mean_absolute_error(Y_test, predicts_lasso)

    clf_ridge = linear_model.Ridge(alpha=best_alpha_ridge, max_iter=150000)
    clf_ridge.fit(X_train_full, Y_train_full)
    predicts_ridge = clf_ridge.predict(X_test_scaled)

    r2_ridge_test = r2_score(Y_test, predicts_ridge)
    mae_ridge_test = mean_absolute_error(Y_test, predicts_ridge)


    X_train_cut_full = pd.concat([X_train_cut, X_val_cut])
  

    clf_ridge = linear_model.Ridge(alpha=best_alpha_ridge_lasso, max_iter=150000)
    clf_ridge.fit(X_train_cut_full, Y_train_full)
    predicts_ridge_lasso = clf_ridge.predict(X_test_cut)

    r2_ridge_lasso_test = r2_score(Y_test, predicts_ridge_lasso)
    mae_ridge_lasso_test = mean_absolute_error(Y_test, predicts_ridge_lasso)

    print(f"validation results: \n")


    zeroed_features_lasso = coefficients_lasso[coefficients_lasso == 0.0]
    kept_features_lasso = coefficients_lasso[coefficients_lasso != 0.0]
    print(f'\n----lasso-----\n')
    print(f'alpha_best={best_alpha_lasso}')
    print(f"Validation R^2_best: {best_r2_lasso:.4f}")
    print(f"Validation MAE best: {mae_best_lasso:.4f} mg/m3")
    print("--- Features Driven to Zero ---")
    print(zeroed_features_lasso)
    print("\n--- Features Kept in the Model ---")
    print(kept_features_lasso)


    importance_ridge = coefficients_ridge.abs()
    sorted_importance = importance_ridge.sort_values(ascending=False)
    print(f'\n----ridge-----\n')
    print(f'alpha_best={best_alpha_ridge}')
    print(f"Validation R^2_best: {best_r2_ridge:.4f}")
    print(f"Validation MAE best: {mae_best_ridge:.4f} mg/m3")
    print("\n--- Top 20 Most Important Features ---")
    print(sorted_importance.head(20))


    importance_ridge_lasso = coefficients_ridge_lasso.abs()
    sorted_importance = importance_ridge_lasso.sort_values(ascending=False)
    print(f'\n----ridge_lasso-----\n')
    print(f'alpha_best={best_alpha_ridge_lasso}')
    print(f"Validation R^2_best: {best_r2_ridge_lasso:.4f}")
    print(f"Validation MAE best: {mae_best_ridge_lasso:.4f} mg/m3")
    print("\n--- Top 20 Most Important Features ---")
    print(sorted_importance.head(20))


    print(f"test results: \n")

    print(f'\n----lasso-----\n')
    print(f"Test R^2: {r2_lasso_test:.4f}")
    print(f"Test MAE: {mae_lasso_test:.4f} mg/m3")

    print(f'\n----ridge-----\n')
    print(f"Test R^2: {r2_ridge_test:.4f}")
    print(f"Test MAE: {mae_ridge_test:.4f} mg/m3")

    print(f'\n----ridge_lasso-----\n')
    print(f"Test R^2: {r2_ridge_lasso_test:.4f}")
    print(f"Test MAE: {mae_ridge_lasso_test:.4f} mg/m3")












    #print(f"The best Ridge model has alpha={best_alpha_ridge} with a R2 of {best_r2_ridge})")


    '''plt.plot(alpha_list_lasso, r2_list_lasso, marker='o', linestyle='-')
    plt.xlabel("Alpha (Lambda)")
    plt.ylabel("R2 Score")
    plt.title(f"Evolution of R2 with alpha for Lasso")
    plt.grid(True)
    plt.savefig(f"grafico_r2_vs_alpha_lasso_lag_{high_limit}.png", dpi=300, bbox_inches='tight')

    plt.figure()
    plt.plot(alpha_list_ridge, r2_list_ridge, marker='o', linestyle='-')
    plt.xlabel("Alpha (Lambda)")
    plt.ylabel("R2 Score")
    plt.title(f"Evolution of R2 with alpha for Ridge")
    plt.grid(True)
    plt.savefig(f"grafico_r2_vs_alpha_ridge_lag_{high_limit}.png", dpi=300, bbox_inches='tight')'''

    '''plt.figure(figsize=(8, 6))
    plt.plot(high_limit_vector,best_r2_for_lag_lasso , marker='o', linestyle='-')
    plt.xlabel("Lag value")
    plt.ylabel("R2")
    plt.title(f"Evolution of best r2 with the lag for Lasso")
    plt.grid(True)
    plt.savefig("grafico_r2_vs_lag_lasso.png", dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.plot(high_limit_vector,best_r2_for_lag_ridge , marker='o', linestyle='-')
    plt.xlabel("Lag value")
    plt.ylabel("R2")
    plt.title(f"Evolution of best r2 with the lag for Ridge")
    plt.grid(True)
    plt.savefig("grafico_r2_vs_lag_ridge.png", dpi=300, bbox_inches='tight')'''

if __name__ == "__main__":
    main()
# Para correr no final
# python train.py --data ../data/chlorophyll_student_2015_2023.xlsx --output ../best_model/model.pkl

# Para correr no entranto
#python train_estavel.py --data ../../data/chlorophyll_student_2015_2023.xlsx 