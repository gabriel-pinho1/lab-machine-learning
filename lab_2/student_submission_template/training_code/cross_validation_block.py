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
import pickle
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
from sklearn.model_selection import LeaveOneGroupOut

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)

    # Guarda os inputs do terminal nesta variável
    args = parser.parse_args()

    best_alpha_for_lag_lasso = []
    best_r2_for_lag_lasso = []
    best_alpha_for_lag_ridge = []
    best_r2_for_lag_ridge = []

    df_raw = read_table(args.data)

    # Applying the log only to the chlorophyll to smooth, but not eliminate, the spikes
    df_raw["chlorophyll_a_mg_m3"] = np.log1p(df_raw["chlorophyll_a_mg_m3"])

    # Listing only the columns to which a Moving Average will be aplied since they have a lot of daily noise
    exogenous_cols_mv = [col for col in df_raw.columns if col in ["wind_speed_m_s", "upwelling_index", "current_speed_m_s", "cloud_fraction", "turbidity_ntu"]]

    # Moving average days
    window_size = 3

    for feature in (exogenous_cols_mv):
        # The min_periods tells the function to only use the days it has available to performe
        # the moving average, assuring that for the first 2 days there are no problems
        df_raw[feature] = df_raw[feature].rolling(window=window_size, min_periods=1).mean()

    # Adding new columns that relate variables
    df_raw["nitrate_x_par"] = df_raw["nitrate_umol_l"] * df_raw["par_umol_m2_s"]
    df_raw["wind_x_upwelling"] = df_raw["wind_speed_m_s"] * df_raw["upwelling_index"]
    df_raw["river_div_turbidity"] = df_raw["river_discharge_index"] / (df_raw["turbidity_ntu"] + 1)
    df_raw["sst_c_x_nitrate"] = df_raw["sst_c"] * df_raw["nitrate_umol_l"]

    # Defining the lag list accounting for the new columns there were created
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
    }

    df_lagged = build_lagged_frame(df_raw, lag_spec=defined_lags)

    # --- Extract Years for Block CV before dropping the date column ---
    # Convert 'date' to datetime just in case it isn't already formatted properly from read_table
    df_lagged["date"] = pd.to_datetime(df_lagged["date"])
    
    Y = df_lagged["chlorophyll_a_mg_m3"]
    X = df_lagged.drop(columns = ["date", "chlorophyll_a_mg_m3"])

    # Last year of all is the final test
    test_size = 365

    # Split into CV and Test sets
    X_cv, X_test = X.iloc[:-test_size].copy(), X.iloc[-test_size:].copy()
    Y_cv, Y_test = Y.iloc[:-test_size].copy(), Y.iloc[-test_size:].copy()

    # Extract the "groups" (years) for the CV set to use in Block Cross-Validation
    dates_cv = df_lagged["date"].iloc[:-test_size]
    cv_years = dates_cv.dt.year.values

    # Using LeaveOneGroupOut to perform Block Cross-Validation by Year
    logo = LeaveOneGroupOut() 

    vector_length_lasso = 50
    vector_length_ridge = 50
    alpha_lasso_defined = np.logspace(-4, -2, vector_length_lasso)
    alpha_ridge_defined = np.logspace(-2, 4, vector_length_ridge)

    # Starting here the optimization using cross validation
    best_r2_lasso = -float('inf')
    best_alpha_lasso = None

    for alpha in alpha_lasso_defined:
        vector_r2_scores = []

        # logo.split generates splits where one year is left out for validation, rest for training
        for train_index, val_index in logo.split(X_cv, Y_cv, groups=cv_years):
            X_train_fold, X_val_fold = X_cv.iloc[train_index].copy(), X_cv.iloc[val_index].copy()
            Y_train_fold, Y_val_fold = Y_cv.iloc[train_index].copy(), Y_cv.iloc[val_index].copy()
            
            # Scalling the train data
            scaler = StandardScaler()
            X_train_fold_scaled = pd.DataFrame(scaler.fit_transform(X_train_fold), columns=X_train_fold.columns)
            X_val_fold_scaled = pd.DataFrame(scaler.transform(X_val_fold), columns=X_val_fold.columns)
            
            # Clipping the outliers for both test and validation data
            for col in X_train_fold_scaled.columns:
                if col.startswith("chlorophyll_a_mg_m3__lag_"):
                    continue  

                X_train_fold_scaled[col] = X_train_fold_scaled[col].clip(-3, 3)
                X_val_fold_scaled[col] = X_val_fold_scaled[col].clip(-3, 3)
            
            clf_lasso = linear_model.Lasso(alpha=alpha, max_iter=150000)
            clf_lasso.fit(X_train_fold_scaled, Y_train_fold)
            
            # Prever e reverter o logaritmo para avaliar em mg/m3 reais
            preds_lasso_mg = np.expm1(clf_lasso.predict(X_val_fold_scaled))
            Y_val_fold_mg = np.expm1(Y_val_fold)
            
            vector_r2_scores.append(r2_score(Y_val_fold_mg, preds_lasso_mg))

        # Calculates the mean R^2 score for all the yearly blocks for the same alpha
        mean_r2 = np.mean(vector_r2_scores) 

        if mean_r2 > best_r2_lasso:
            best_r2_lasso = mean_r2
            best_alpha_lasso = alpha

    print(f"O melhor Lasso (Block CV) tem alpha={best_alpha_lasso} com R2 médio de {best_r2_lasso:.4f}")

    # After determining the best alpha for all the yearly blocks, we use it to test on all the past data
    scaler_cv = StandardScaler()
    X_cv_scaled = pd.DataFrame(scaler_cv.fit_transform(X_cv), columns=X_cv.columns).clip(-3, 3)

    final_cv_lasso = linear_model.Lasso(alpha = best_alpha_lasso, max_iter = 150000)
    final_cv_lasso.fit(X_cv_scaled, Y_cv)

    preds_lasso_cv_mg = np.expm1(final_cv_lasso.predict(X_cv_scaled))
    mae_best_lasso = mean_absolute_error(np.expm1(Y_cv), preds_lasso_cv_mg)
    coefficients_lasso = pd.Series(final_cv_lasso.coef_, index=X_cv_scaled.columns)
    kept_features_lasso = coefficients_lasso[coefficients_lasso != 0.0].index.tolist()

    # Training ridge on the parameters that lasso did not eliminate
    X_cv_cut = X_cv.loc[:, kept_features_lasso]
    
    best_r2_ridge_lasso = -float('inf')  
    best_alpha_ridge_lasso = None

    for alpha in alpha_ridge_defined:
        fold_r2_scores = []
        
        # Again, use the year groups to run Block CV for Ridge hyperparameter tuning
        for train_index, val_index in logo.split(X_cv_cut, Y_cv, groups=cv_years):
            X_train_fold, X_val_fold = X_cv_cut.iloc[train_index].copy(), X_cv_cut.iloc[val_index].copy()
            Y_train_fold, Y_val_fold = Y_cv.iloc[train_index].copy(), Y_cv.iloc[val_index].copy()
            
            scaler = StandardScaler()
            X_train_fold_scaled = pd.DataFrame(scaler.fit_transform(X_train_fold), columns=X_train_fold.columns)
            X_val_fold_scaled = pd.DataFrame(scaler.transform(X_val_fold), columns=X_val_fold.columns)

            for col in X_train_fold_scaled.columns:
                if col.startswith("chlorophyll_a_mg_m3__lag_"):
                    continue  

                X_train_fold_scaled[col] = X_train_fold_scaled[col].clip(-3, 3)
                X_val_fold_scaled[col] = X_val_fold_scaled[col].clip(-3, 3)
                        
            clf_ridge = linear_model.Ridge(alpha=alpha)
            clf_ridge.fit(X_train_fold_scaled, Y_train_fold)
            
            preds_ridge_mg = np.expm1(clf_ridge.predict(X_val_fold_scaled))
            Y_val_fold_mg = np.expm1(Y_val_fold)
            
            fold_r2_scores.append(r2_score(Y_val_fold_mg, preds_ridge_mg))
            
        mean_r2 = np.mean(fold_r2_scores)
        
        if mean_r2 > best_r2_ridge_lasso:
            best_r2_ridge_lasso = mean_r2
            best_alpha_ridge_lasso = alpha

    # Treinar o Ridge Final no CV Completo para extrair coeficientes finais
    X_cv_cut_scaled = pd.DataFrame(scaler_cv.fit_transform(X_cv_cut), columns=X_cv_cut.columns).clip(-3, 3)
    final_cv_ridge = linear_model.Ridge(alpha=best_alpha_ridge_lasso)
    final_cv_ridge.fit(X_cv_cut_scaled, Y_cv)
    
    preds_ridge_cv_mg = np.expm1(final_cv_ridge.predict(X_cv_cut_scaled))
    mae_best_ridge_lasso = mean_absolute_error(np.expm1(Y_cv), preds_ridge_cv_mg)
    coefficients_ridge_lasso = pd.Series(final_cv_ridge.coef_, index=X_cv_cut_scaled.columns)

    
    #testing:
    scaler_final = StandardScaler()
    X_cv_scaled_final = pd.DataFrame(scaler_final.fit_transform(X_cv), columns=X_cv.columns).clip(-3, 3)
    X_test_scaled = pd.DataFrame(scaler_final.transform(X_test), columns=X_test.columns).clip(-3, 3)

    # Re-treinar o Lasso com o melhor alpha no passado todo
    clf_lasso_test = linear_model.Lasso(alpha=best_alpha_lasso, max_iter=150000)
    clf_lasso_test.fit(X_cv_scaled_final, Y_cv)
    predicts_lasso_test = np.expm1(clf_lasso_test.predict(X_test_scaled))
    
    r2_lasso_test = r2_score(np.expm1(Y_test), predicts_lasso_test)
    mae_lasso_test = mean_absolute_error(np.expm1(Y_test), predicts_lasso_test)

    # Re-treinar o Ridge-Lasso com o melhor alpha no passado TODO
    X_cv_cut_final = X_cv_scaled_final.loc[:, kept_features_lasso]
    X_test_cut_final = X_test_scaled.loc[:, kept_features_lasso]

    clf_ridge_test = linear_model.Ridge(alpha=best_alpha_ridge_lasso, max_iter=150000)
    clf_ridge_test.fit(X_cv_cut_final, Y_cv)
    predicts_ridge_lasso_test = np.expm1(clf_ridge_test.predict(X_test_cut_final))

    r2_ridge_lasso_test = r2_score(np.expm1(Y_test), predicts_ridge_lasso_test)
    mae_ridge_lasso_test = mean_absolute_error(np.expm1(Y_test), predicts_ridge_lasso_test)

    print(f"validation results: \n")

    print_zeroed_lasso = coefficients_lasso[coefficients_lasso == 0.0]
    print_kept_lasso = coefficients_lasso[coefficients_lasso != 0.0]
    
    print(f'\n----lasso-----\n')
    print(f'alpha_best={best_alpha_lasso}')
    print(f"Validation R^2_best: {best_r2_lasso:.4f}")
    print(f"Validation MAE best: {mae_best_lasso:.4f} mg/m3")
    print("--- Features Driven to Zero ---")
    print(print_zeroed_lasso)
    print("\n--- Features Kept in the Model ---")
    print(print_kept_lasso)

    importance_ridge_lasso = coefficients_ridge_lasso.abs()
    sorted_importance = importance_ridge_lasso.sort_values(ascending=False)
    print(f'\n----ridge_lasso-----\n')
    print(f'alpha_best={best_alpha_ridge_lasso}')
    print(f"Validation R^2_best: {best_r2_ridge_lasso:.4f}")
    print(f"Validation MAE best: {mae_best_ridge_lasso:.4f} mg/m3")
    print("\n--- Top 20 Most Important Features ---")
    print(sorted_importance.head(20))

    print("\n")
    print(f"test results: \n")

    print(f'\n----lasso-----\n')
    print(f"Test R^2: {r2_lasso_test:.4f}")
    print(f"Test MAE: {mae_lasso_test:.4f} mg/m3")

    print(f'\n----ridge_lasso-----\n')
    print(f"Test R^2: {r2_ridge_lasso_test:.4f}")
    print(f"Test MAE: {mae_ridge_lasso_test:.4f} mg/m3")

    # ------------------ Plotting Test Predictions vs Real ------------------
    Y_test_real = np.expm1(Y_test)
    
    plt.figure(figsize=(14, 6))
    
    # Subplot 1: Timeline comparison
    plt.subplot(1, 2, 1)
    plt.plot(Y_test_real.values, label='Real Values', color='tab:blue', alpha=0.7)
    plt.plot(predicts_ridge_lasso_test, label='Predictions (Ridge-Lasso)', color='tab:orange', alpha=0.7)
    plt.title("Test Set: Real vs Predicted (Timeline)")
    plt.xlabel("Days in Test Set")
    plt.ylabel("Chlorophyll a (mg/m3)")
    plt.legend()
    
    # Subplot 2: Scatter correlation
    plt.subplot(1, 2, 2)
    plt.scatter(Y_test_real.values, predicts_ridge_lasso_test, alpha=0.5, color='tab:green', s=15)
    
    # Add y=x ideal fit reference line
    max_val = max(Y_test_real.max(), predicts_ridge_lasso_test.max())
    min_val = min(Y_test_real.min(), predicts_ridge_lasso_test.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Fit (y=x)')
    
    plt.title(f"Test Set Correlation (R² = {r2_ridge_lasso_test:.2f})")
    plt.xlabel("Real Chlorophyll a (mg/m3)")
    plt.ylabel("Predicted Chlorophyll a (mg/m3)")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("test_predictions_vs_real.png", dpi=300, bbox_inches='tight')
    plt.show()
    # -----------------------------------------------------------------------

    # Training the model using the obtained parameters on the last year (2023)

    X_total = pd.concat([X_cv, X_test])
    Y_total = pd.concat([Y_cv, Y_test])

    scaler_final = StandardScaler()
    X_final_scaled = pd.DataFrame(scaler_final.fit_transform(X_total), columns= X_total.columns).clip(-3, 3)

    lasso_final = linear_model.Lasso(alpha = best_alpha_lasso, max_iter = 150000)
    lasso_final.fit(X_final_scaled, Y_total)

    coefficients_final = pd.Series(lasso_final.coef_, index = X_final_scaled.columns)
    kept_features_final = coefficients_final[coefficients_final != 0.0].index.tolist()

    X_final_cut = X_final_scaled.loc[:, kept_features_final]
    ridge_final = linear_model.Ridge(alpha = best_alpha_ridge_lasso, max_iter = 150000)
    ridge_final.fit(X_final_cut, Y_total)

    # Saving the final model
    model_artifact = {
    "scaler" : scaler_final,
    "kept_features" : kept_features_final,
    "model": ridge_final
    } 

    with open(args.output, "wb") as f:
        pickle.dump(model_artifact, f)

if __name__ == "__main__":
    main()

    #python cross_validation_block.py --data ../../data/chlorophyll_student_2015_2023.xlsx --output ../best_model/model.pkl