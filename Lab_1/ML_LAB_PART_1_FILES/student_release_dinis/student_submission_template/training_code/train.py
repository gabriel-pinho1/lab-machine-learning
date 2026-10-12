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
    parser.add_argument("--output", required=False, type=Path)

    args = parser.parse_args()

    low_limit_vector = range(0, 4)
    high_limit_vector = range(5, 9)

    defined_lags = {
        "chlorophyll_a_mg_m3": list(range(1, 7)),
        "sst_c": list(range(1, 7)),
        "par_umol_m2_s" : list(range(1,7)), 
        "nitrate_umol_l" : list(range(1,7)),
        "wind_speed_m_s" : list(range(1,7)),
        "upwelling_index" : list(range(1,7)),
        "mixed_layer_depth_m" : list(range(1,7)),
        "salinity_psu" : list(range(1,7)),
        "current_speed_m_s" : list(range(1,7)),
        "river_discharge_index" : list(range(1,7)),
        "cloud_fraction" : list(range(1,7)),
        "surface_pressure_hpa" : list(range(1,7)),
        "turbidity_ntu" : list(range(1,7)),
    }

    df_raw = read_table(args.data)
    
    for key in defined_lags:
        x_low = []
        y_high = []
        z_r2 = [] 
        
        for low_limit in low_limit_vector:
            for high_limit in high_limit_vector:
                if key == "chlorophyll_a_mg_m3" and low_limit == 0:
                    continue
                
                defined_lags[key] = list(range(low_limit, high_limit))

                df_raw_temp = df_raw.copy()
                df_raw_temp["chlorophyll_a_mg_m3"] = np.log1p(df_raw_temp["chlorophyll_a_mg_m3"])
                
                df_lagged = build_lagged_frame(df_raw_temp, lag_spec=defined_lags)

                scaler = StandardScaler()

                Y = df_lagged["chlorophyll_a_mg_m3"]
                X = df_lagged.drop(columns = ["date", "chlorophyll_a_mg_m3"])

                separation_idx = -2*365
                X_train, X_test = X.iloc[:separation_idx], X.iloc[separation_idx:]
                Y_train, Y_test = Y.iloc[:separation_idx], Y.iloc[separation_idx:]

                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                X_train_scaled = X_train_scaled.clip(-3, 3)
                X_test_scaled = X_test_scaled.clip(-3, 3)

                vector_length_lasso = 20
                best_r2_lasso = -float('inf')  
                alpha_lasso_defined = np.logspace(-4, -2, vector_length_lasso)

                for alpha in alpha_lasso_defined:
                    clf_lasso = linear_model.Lasso(alpha=alpha, max_iter=150000)
                    clf_lasso.fit(X_train_scaled, Y_train)
                    
                    predicts_lasso = clf_lasso.predict(X_test_scaled)
                    predicts_lasso_true = np.expm1(predicts_lasso)
                    Y_test_true = np.expm1(Y_test)

                    r2_lasso = r2_score(Y_test_true, predicts_lasso_true)

                    if r2_lasso > best_r2_lasso:
                        best_r2_lasso = r2_lasso

                x_low.append(low_limit)
                y_high.append(high_limit)
                z_r2.append(best_r2_lasso)

        df_heatmap = pd.DataFrame({
            'low_limit': x_low,
            'high_limit': y_high,
            'R2': z_r2
        })
        
        heatmap_data = df_heatmap.pivot(index='high_limit', columns='low_limit', values='R2')
        heatmap_data = heatmap_data.sort_index(ascending=False)

        fig, ax = plt.subplots(figsize=(8, 6))
        
        cax = ax.imshow(heatmap_data.values, cmap="viridis", aspect="auto")
        
        ax.set_xticks(np.arange(len(heatmap_data.columns)))
        ax.set_yticks(np.arange(len(heatmap_data.index)))
        ax.set_xticklabels(heatmap_data.columns)
        ax.set_yticklabels(heatmap_data.index)
        
        for i in range(len(heatmap_data.index)):
            for j in range(len(heatmap_data.columns)):
                val = heatmap_data.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.4f}", ha="center", va="center", color="white", fontsize=9)

        cbar = fig.colorbar(cax, ax=ax)
        cbar.set_label('R2 Score')
        
        plt.title(f"Heatmap de R2 (Lasso) - {key}")
        plt.xlabel("Low Limit")
        plt.ylabel("High Limit")
        plt.tight_layout()
        
        filename = f"heatmap_r2_{key}.png"
        plt.savefig(filename, dpi=300)
        plt.close()

if __name__ == "__main__":
    main()