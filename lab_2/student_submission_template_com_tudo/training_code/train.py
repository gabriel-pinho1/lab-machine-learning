
from __future__ import annotations
import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import gc
import pandas as pd
from scipy.stats import zscore

from build_lagged_features import read_table, build_lagged_frame, semi_anual_sinusoidals

from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)


    args = parser.parse_args()


    df_raw = read_table(args.data)

    # Applying the log only to the chlorophyll to smooth, but not eliminate, the spikes
    df_raw["chlorophyll_a_mg_m3"] = np.log1p(df_raw["chlorophyll_a_mg_m3"])

    # Adding new columns that relate variables (teste, depois logo se vê)

    df_raw["nitrate_x_par"] = df_raw["nitrate_umol_l"] * df_raw["par_umol_m2_s"]
    df_raw["wind_x_upwelling"] = df_raw["wind_speed_m_s"] * df_raw["upwelling_index"]
    df_raw["sst_c_x_nitrate"] = df_raw["sst_c"] * df_raw["nitrate_umol_l"]

    # Defining the lag list accounting for the new columns there were created

    #for high_limit in high_limit_vector:
    high_limit = 15
    low_limit = 0


    defined_lags = {
    "chlorophyll_a_mg_m3": list(range(1, high_limit)),
    "sst_c": list(range(low_limit, high_limit)),
    "par_umol_m2_s" : list(range(low_limit, high_limit)), 
    "nitrate_umol_l" : list(range(low_limit, high_limit)),
    "wind_speed_m_s" : list(range(low_limit, 7)),
    "upwelling_index" : list(range(low_limit, high_limit)),
    "mixed_layer_depth_m" : list(range(low_limit, high_limit)),
    "current_speed_m_s" :list(range(low_limit, high_limit)),
    "river_discharge_index" : list(range(low_limit, high_limit)),
    "cloud_fraction" : list(range(low_limit, high_limit)),
    "turbidity_ntu" : list(range(low_limit, high_limit)),
    "nitrate_x_par" : list(range(low_limit, high_limit)),
    "wind_x_upwelling" : list(range(low_limit, high_limit)),
    "sst_c_x_nitrate" : list(range(low_limit,high_limit))
    }

   
    df_lagged = build_lagged_frame(df_raw, lag_spec=defined_lags)
    df_lagged = semi_anual_sinusoidals(df_lagged)

    Y = df_lagged["chlorophyll_a_mg_m3"]
    X = df_lagged.drop(columns = ["date", "chlorophyll_a_mg_m3"])

    

    # Last year of all is the final test
    test_size = 365

    # cv refers to cross validation
    X_cv, X_test = X.iloc[:-test_size].copy(), X.iloc[-test_size:].copy()
    Y_cv, Y_test = Y.iloc[:-test_size].copy(), Y.iloc[-test_size:].copy()


    cv_dates = df_lagged["date"].iloc[:-test_size]
    cv_years = cv_dates.dt.year.values



    # Creating the split for cross validation
    # At least 4 years of training data for the first plit
    # After that, one year is added each time
    validation_years = [2019, 2020, 2021, 2022]
    
    custom_cv_splits = []
    for val_year in validation_years:
        train_idx = np.where(cv_years < val_year)[0]

        val_idx = np.where(cv_years == val_year)[0]
        custom_cv_splits.append((train_idx, val_idx))

    vector_length_lasso = 50
    vector_length_ridge = 50

    alpha_lasso_defined = np.logspace(-4, 0, vector_length_lasso)
    alpha_ridge_defined = np.logspace(-4, 2, vector_length_ridge)


    # Starting here the optimization using cross validation
    best_r2_lasso = -float('inf')
    best_alpha_lasso = None
    alphas_testados = []
    r2_resultados = []
    r2_erros_padrao = []
    features_retidas = []

    for alpha in alpha_lasso_defined:
        vector_r2_scores = []

        for train_index, val_index in custom_cv_splits:
            X_train_fold, X_val_fold = X_cv.iloc[train_index].copy(), X_cv.iloc[val_index].copy()
            
            Y_train_fold, Y_val_fold = Y_cv.iloc[train_index].copy(), Y_cv.iloc[val_index].copy()

            # Scalling the train data
            scaler = StandardScaler()
            X_train_fold_scaled = pd.DataFrame(scaler.fit_transform(X_train_fold), columns=X_train_fold.columns)
            # scalling based on the mean of the test data p
            X_val_fold_scaled = pd.DataFrame(scaler.transform(X_val_fold), columns=X_val_fold.columns)
            
            # Clipping the outliers for both test and validation data
            for col in X_train_fold_scaled.columns:
                if col.startswith("chlorophyll_a_mg_m3__lag_") or col.startswith("annual_sin") or col.startswith("annual_cos"):
                    continue  

                X_train_fold_scaled[col] = X_train_fold_scaled[col].clip(-3, 3)
                X_val_fold_scaled[col] = X_val_fold_scaled[col].clip(-3, 3)
            
            clf_lasso = linear_model.Lasso(alpha=alpha, max_iter=150000)
            clf_lasso.fit(X_train_fold_scaled, Y_train_fold)
            
            # Reverse the log process to calculate the r2 value
            preds_lasso_mg = np.expm1(clf_lasso.predict(X_val_fold_scaled))
            Y_val_fold_mg = np.expm1(Y_val_fold)
            
            vector_r2_scores.append(r2_score(Y_val_fold_mg, preds_lasso_mg))



        # Calculates the mean R^2 score for all the time splits for the same alpha
        mean_r2 = np.mean(vector_r2_scores) 
        se_r2 = np.std(vector_r2_scores) / np.sqrt(len(custom_cv_splits)) # Standard Error

        alphas_testados.append(alpha)
        r2_resultados.append(mean_r2)
        r2_erros_padrao.append(se_r2)
        num_features = np.sum(clf_lasso.coef_ != 0)
        features_retidas.append(num_features)

        if mean_r2 > best_r2_lasso:
            best_r2_lasso = mean_r2
            best_alpha_lasso = alpha

    

    plt.close('all')

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = 'tab:blue'
    ax1.set_xlabel('Alpha (Log Scale)', fontweight='bold')
    ax1.set_ylabel('Mean Validation R²', color=color1, fontweight='bold')
    ax1.plot(alphas_testados, r2_resultados, color=color1, marker='o', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xscale('log') 

    ax2 = ax1.twinx()  

    color2 = 'tab:red'
    ax2.set_ylabel('Number of maintained model features', color=color2, fontweight='bold')  
    ax2.plot(alphas_testados, features_retidas, color=color2, marker='s', linestyle='--', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title('Study of Lasso: R² vs. Model Complexity (Alpha)', fontsize=14)
    plt.tight_layout()
    plt.show()
    plt.savefig("lasso_r2_complexity_study.png", dpi=300, bbox_inches='tight')

    # We perform a test to see the amount of variables that are retained when the largest alpha
    #that is inside the interval of the standard error compared to the alpha that delivers the maximum R^2 is chosen

    max_r2_idx = np.argmax(r2_resultados)
    max_r2 = r2_resultados[max_r2_idx]
    se_at_max_r2 = r2_erros_padrao[max_r2_idx]

    threshold_r2 = max_r2 - se_at_max_r2

    best_parsimonious_alpha = alphas_testados[max_r2_idx]
    
    for i in range(len(alphas_testados) - 1, -1, -1):
        if r2_resultados[i] >= threshold_r2:
            best_parsimonious_alpha = alphas_testados[i]
            break 

    print(f"R2 Max: {max_r2:.4f}")
    print(f"The best parcimounious model has alpha={best_parsimonious_alpha} with an average R2 of {r2_resultados[alphas_testados.index(best_parsimonious_alpha)]:.4f}")


    scaler_cv = StandardScaler()
    X_cv_scaled = pd.DataFrame(scaler_cv.fit_transform(X_cv), columns=X_cv.columns).clip(-3, 3)

    final_cv_lasso = linear_model.Lasso(alpha = best_parsimonious_alpha, max_iter = 150000)
    final_cv_lasso.fit(X_cv_scaled, Y_cv)

    preds_lasso_cv_mg = np.expm1(final_cv_lasso.predict(X_cv_scaled))
    mae_best_lasso = mean_absolute_error(np.expm1(Y_cv), preds_lasso_cv_mg)
    coefficients_lasso = pd.Series(final_cv_lasso.coef_, index=X_cv_scaled.columns)

    kept_features_lasso = coefficients_lasso[coefficients_lasso != 0.0].index.tolist()
    
    print(f"Number of features kept by Lasso: {len(kept_features_lasso)}")



    print("\n" + "="*50)
    print("Variance and Suficiency Analysis of the Model")
    print("="*50)

    parsimonious_lasso = linear_model.Lasso(alpha=best_parsimonious_alpha, max_iter=150000)
    
    print(f"\nTestin Parcimounious model (Alpha={best_parsimonious_alpha:.4f}) year by year:")
    
    r2_por_ano = []
    mae_por_ano = []
    
    for idx, (train_index, val_index) in enumerate(custom_cv_splits):
        ano_validacao = validation_years[idx]
        
        X_t, X_v = X_cv.iloc[train_index].copy(), X_cv.iloc[val_index].copy()
        Y_t, Y_v = Y_cv.iloc[train_index].copy(), Y_cv.iloc[val_index].copy()


        scaler_diag = StandardScaler()
        X_t_scaled = pd.DataFrame(scaler_diag.fit_transform(X_t), columns=X_t.columns).clip(-3, 3)
        X_v_scaled = pd.DataFrame(scaler_diag.transform(X_v), columns=X_v.columns).clip(-3, 3)
        

        parsimonious_lasso.fit(X_t_scaled, Y_t)
        preds_mg = np.expm1(parsimonious_lasso.predict(X_v_scaled))
        Y_v_mg = np.expm1(Y_v)
        

        fold_r2 = r2_score(Y_v_mg, preds_mg)
        fold_mae = mean_absolute_error(Y_v_mg, preds_mg)
        
        r2_por_ano.append(fold_r2)
        mae_por_ano.append(fold_mae)
        
        print(f" -> Year {ano_validacao}: R2 = {fold_r2:.3f} | MAE = {fold_mae:.3f} mg/m3")

    print("\n--- Variance analysis ---")
    print(f"Standart deeviation of R2: {np.std(r2_por_ano):.3f}")
    print(f"Max difference betwwen years (R2): {max(r2_por_ano) - min(r2_por_ano):.3f}")
    
    print("\n--- Suficiency Analysis (Trade-off Complexidade vs R2) ---")
    
    max_lasso = linear_model.Lasso(alpha=best_alpha_lasso, max_iter=150000)

    X_cv_scaled_diag = pd.DataFrame(StandardScaler().fit_transform(X_cv), columns=X_cv.columns).clip(-3, 3)
    
    max_lasso.fit(X_cv_scaled_diag, Y_cv)
    features_modelo_complexo = np.sum(max_lasso.coef_ != 0)
    
    parsimonious_lasso.fit(X_cv_scaled_diag, Y_cv)
    features_modelo_simples = np.sum(parsimonious_lasso.coef_ != 0)

    coefs_parc = pd.Series(parsimonious_lasso.coef_, index=X_cv.columns)
    features_parcimonious_model_kept = coefs_parc[coefs_parc != 0.0].index.tolist()

    features_nomes = X_cv.columns[parsimonious_lasso.coef_ != 0].tolist()
    
    r2_simples = np.mean(r2_por_ano)
    r2_complexo = max_r2 
    
    print(f" (Max R2) : {features_modelo_complexo} features -> R2 = {r2_complexo:.4f}")
    print(f" Parcimonious Model    : {features_modelo_simples} features -> R2 = {r2_simples:.4f}")
    print(f"Simplification Cost   : Lost {r2_complexo - r2_simples:.4f}  R2 and removed {(features_modelo_complexo - features_modelo_simples)} features.")
    
    print("\n Kept features after Lasso:")
    for feat in features_nomes:
        print(f" - {feat}")
    print("="*50 + "\n")


    
    # Training ridge on the parameters that lasso did not eliminate after the parcimonious alpha is chosen
    X_cv_cut = X_cv.loc[:, features_parcimonious_model_kept]
    
    best_r2_ridge_lasso = -float('inf')  
    best_alpha_ridge_lasso = None

    for alpha in alpha_ridge_defined:
        fold_r2_scores = []
        
        for train_index, val_index in custom_cv_splits:
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

    # training the final Ridge on all the CV data
    X_cv_cut_scaled = pd.DataFrame(scaler_cv.fit_transform(X_cv_cut), columns=X_cv_cut.columns).clip(-3, 3)
    final_cv_ridge = linear_model.Ridge(alpha=best_alpha_ridge_lasso)
    final_cv_ridge.fit(X_cv_cut_scaled, Y_cv)
    
    preds_ridge_cv_mg = np.expm1(final_cv_ridge.predict(X_cv_cut_scaled))
    mae_best_ridge_lasso = mean_absolute_error(np.expm1(Y_cv), preds_ridge_cv_mg)
    coefficients_ridge_lasso = pd.Series(final_cv_ridge.coef_, index=X_cv_cut_scaled.columns)


    
    # Final testing on the 2023 data

    scaler_final = StandardScaler()
    X_cv_scaled_final = pd.DataFrame(scaler_final.fit_transform(X_cv), columns=X_cv.columns).clip(-3, 3)
    X_test_scaled = pd.DataFrame(scaler_final.transform(X_test), columns=X_test.columns).clip(-3, 3)

    # Re-train Lasso with the parsimonious coeficient
    clf_lasso_test = linear_model.Lasso(alpha=best_parsimonious_alpha, max_iter=150000)
    clf_lasso_test.fit(X_cv_scaled_final, Y_cv)
    predicts_lasso_test = np.expm1(clf_lasso_test.predict(X_test_scaled))
    
    r2_lasso_test = r2_score(np.expm1(Y_test), predicts_lasso_test)
    mae_lasso_test = mean_absolute_error(np.expm1(Y_test), predicts_lasso_test)

    # Re-train Ridge with the parsimonious coeficient
    X_cv_cut_final = X_cv_scaled_final.loc[:, features_parcimonious_model_kept]
    X_test_cut_final = X_test_scaled.loc[:, features_parcimonious_model_kept]

    clf_ridge_test = linear_model.Ridge(alpha=best_alpha_ridge_lasso, max_iter=150000)
    clf_ridge_test.fit(X_cv_cut_final, Y_cv)
    predicts_ridge_lasso_test = np.expm1(clf_ridge_test.predict(X_test_cut_final))

    r2_ridge_lasso_test = r2_score(np.expm1(Y_test), predicts_ridge_lasso_test)
    mae_ridge_lasso_test = mean_absolute_error(np.expm1(Y_test), predicts_ridge_lasso_test)

    print(f"validation results of Ridge Model run on features kept by Lasso: \n")
    

    importance_ridge_lasso = coefficients_ridge_lasso.abs()
    sorted_importance = importance_ridge_lasso.sort_values(ascending=False)
    print(f'\n----ridge_lasso-----\n')

    print("\n")
    print(f"test results: \n")

    print(f'\n----lasso-----\n')
    print(f"Test R^2: {r2_lasso_test:.4f}")
    print(f"Test MAE: {mae_lasso_test:.4f} mg/m3")

    print(f'\n----ridge_lasso-----\n')
    print(f"Test R^2: {r2_ridge_lasso_test:.4f}")
    print(f"Test MAE: {mae_ridge_lasso_test:.4f} mg/m3")
    print("\n--- Ridge final weighs ---")
    print(sorted_importance)

    # Plotting test predictions against the true data of 2023
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

    # Final training on all the available data 

    X_total = pd.concat([X_cv, X_test])
    Y_total = pd.concat([Y_cv, Y_test])

    scaler_final = StandardScaler()
    X_final_scaled = pd.DataFrame(scaler_final.fit_transform(X_total), columns= X_total.columns).clip(-3, 3)

    # Comentar daqui
    '''lasso_final = linear_model.Lasso(alpha = best_parsimonious_alpha, max_iter = 150000)
    lasso_final.fit(X_final_scaled, Y_total)

    coefficients_final = pd.Series(lasso_final.coef_, index = X_final_scaled.columns)
    kept_features_final = coefficients_final[coefficients_final != 0.0].index.tolist()'''
    # Até aqui

    X_final_cut = X_final_scaled.loc[:, features_parcimonious_model_kept]
    #X_final_cut = X_final_scaled.loc[:, kept_features_final]

    ridge_final = linear_model.Ridge(alpha = best_alpha_ridge_lasso, max_iter = 150000)
    ridge_final.fit(X_final_cut, Y_total)

    # Saving the final model
    model_artifact = {
    "scaler" : scaler_final,
    #"kept_features" : kept_features_final,
    "kept_features" : features_parcimonious_model_kept,
    "model": ridge_final
    } 

    with open(args.output, "wb") as f:
        pickle.dump(model_artifact, f)


if __name__ == "__main__":
    main()
