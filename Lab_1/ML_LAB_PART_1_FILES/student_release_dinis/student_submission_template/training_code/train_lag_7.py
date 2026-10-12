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

import pandas as pd
from scipy.stats import zscore

from build_lagged_features import read_table, build_lagged_frame

from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import r2_score

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
    low_limit = 1

    defined_lags = {
    "chlorophyll_a_mg_m3": list(range(low_limit, high_limit)),
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
    }

    df_raw = read_table(args.data)
    df_raw["chlorophyll_a_mg_m3"] = np.log1p(df_raw["chlorophyll_a_mg_m3"])
    df_lagged = build_lagged_frame(df_raw, lag_spec=defined_lags)

    scaler = StandardScaler()


    Y = df_lagged["chlorophyll_a_mg_m3"]
    X = df_lagged.drop(columns = ["date", "chlorophyll_a_mg_m3"])

    '''raw_columns_names = [col for col in df_raw.columns if col != "date"]
    df_lag_sum = pd.DataFrame()
    new_columns = {}

    for idx, name in enumerate(raw_columns_names):
        
        max_lag = 7
        
        for p in range(1, max_lag):
            if name == "chlorophyll_a_mg_m3":
            
                cols_to_sum = [f"{name}__lag_{i}" for i in range(1, p)]
            else:
                cols_to_sum = [f"{name}__lag_{i}" for i in range(0, p)]


            new_columns[f"row_sum_{name}_{p}_days"] = X[cols_to_sum].sum(axis=1)
                
    df_lag_sum = pd.DataFrame(new_columns) 

    X = pd.concat([X, df_lag_sum], axis=1)'''


    separation_idx = -2*365
    X_train, X_test = X.iloc[:separation_idx], X.iloc[separation_idx:]
    Y_train, Y_test = Y.iloc[:separation_idx], Y.iloc[separation_idx:]


    # Aplicar o fit apenas aos dados de teste
    X_train_scaled = scaler.fit_transform(X_train)

    # COnverter em panda outra vez para reter o nome das colunas
    X_train_scaled_pd = pd.DataFrame(X_train_scaled, columns= X_train.columns)
    X_test_scaled = scaler.transform(X_test)

    # Estou a definir que todos os valores cujo o resultado do fit.transform estejam para lá de -3 e 3 sejam 
    # limitados a -3 e 3 (porque estatisticamente só os outliers estão fora desse valor)
    X_train_scaled = X_train_scaled.clip(-3, 3)
    X_test_scaled = X_test_scaled.clip(-3, 3)
    #X_train_scaled = X_train_scaled.to_csv(args.output, index=False)

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
        clf_lasso = linear_model.Lasso(alpha)
        clf_lasso = linear_model.Lasso(alpha=alpha, max_iter=150000)
        clf_lasso.fit(X_train_scaled, Y_train)
        

        #coeficients = pd.Series(clf_lasso.coef_, index= X_train_scaled.columns)
        #print("Descarded Variables\n")
        #descarded = coeficients[coeficients == 0.0]
        #print(descarded)
        #print("\n")
        #print("Non-Zero Coeficients\n")
        #saved = coeficients[coeficients != 0]
        #print(saved)
        #print("\n")


        predicts_lasso = clf_lasso.predict(X_test_scaled)
        predicts_lasso_true = np.expm1(predicts_lasso)

        Y_test_true = np.expm1(Y_test)

        r2_lasso = r2_score(Y_test_true, predicts_lasso_true)
        alpha_list_lasso.append(alpha)
        r2_list_lasso.append(r2_lasso)


        if r2_lasso > best_r2_lasso:
            best_r2_lasso = r2_lasso
            best_alpha_lasso = alpha
    best_alpha_for_lag_lasso.append(best_alpha_lasso)
    best_r2_for_lag_lasso.append(best_r2_lasso)


    print(f"The best Lasso model has alpha={best_alpha_lasso} with a R2 of {best_r2_lasso})")

    best_r2_ridge = -float('inf')  
    best_alpha_ridge = None

    for alpha in alpha_ridge_defined:

        clf_ridge = linear_model.Ridge(alpha)
        clf_ridge.fit(X_train_scaled, Y_train)

        predicts_ridge = clf_ridge.predict(X_test_scaled)
        predicts_ridge_true = np.expm1(predicts_ridge)

        r2_ridge = r2_score(Y_test_true, predicts_ridge_true)

        alpha_list_ridge.append(alpha)
        r2_list_ridge.append(r2_ridge)

        if r2_ridge > best_r2_ridge:
            best_r2_ridge = r2_ridge
            best_alpha_ridge = alpha

    best_alpha_for_lag_ridge.append(best_alpha_ridge)
    best_r2_for_lag_ridge.append(best_r2_ridge)

    print(f"The best Ridge model has alpha={best_alpha_ridge} with a R2 of {best_r2_ridge})")


    plt.plot(alpha_list_lasso, r2_list_lasso, marker='o', linestyle='-')
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
    plt.savefig(f"grafico_r2_vs_alpha_ridge_lag_{high_limit}.png", dpi=300, bbox_inches='tight')

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
#python train.py --data ../../data/chlorophyll_student_2015_2023.xlsx --output training.csv