
from __future__ import annotations
from build_lagged_features import build_lagged_frame, read_table
import argparse
from pathlib import Path
import itertools
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

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
    
    # You had parser.parse_args() twice in a row; you only need it once
    args = parser.parse_args()
    
    df_raw = read_table(args.data)
    df_raw["chlorophyll_a_mg_m3"] = np.log1p(df_raw["chlorophyll_a_mg_m3"])
    lagged_df = build_lagged_frame(df_raw)
    raw_columns_names = [f"{col}__lag_0" for col in df_raw.drop(columns=["date", "chlorophyll_a_mg_m3"]).columns]

    raw_columns_names.append("chlorophyll_a_mg_m3__lag_1")


    scaler = StandardScaler()
    df_lagged_copy = lagged_df.drop(columns=["date"]).copy()
    

    new_columns = {}
    for name, name2 in itertools.combinations_with_replacement(raw_columns_names, 2):
        
       new_columns[f"row_mul_{name}_{name2}"] = df_lagged_copy[f"{name}"] * df_lagged_copy[f"{name2}"]

    df_new_features = pd.DataFrame(new_columns)
    second_order_columns = df_new_features.columns.tolist()
    df_lagged_copy = pd.concat([df_lagged_copy, df_new_features], axis=1)
    df_lagged_scaled = scaler.fit_transform(df_lagged_copy)
    
    df_lagged_scaled = pd.DataFrame(df_lagged_scaled, columns=df_lagged_copy.columns)
    df_lagged_scaled = pd.concat([lagged_df['date'], df_lagged_scaled], axis = 1)
    

    scatter_dir = args.output / "scatter_plots"
    line_dir = args.output / "line_plots"
    
    scatter_dir.mkdir(parents=True, exist_ok=True)
    line_dir.mkdir(parents=True, exist_ok=True)
    

    
    for name in second_order_columns:

        x =  df_lagged_scaled['date']
        y1 = df_lagged_scaled[name]
        y2 = df_lagged_scaled['chlorophyll_a_mg_m3']
        corr_value = y1.corr(y2)
        if abs(corr_value) >= 0.5:
            plt.plot(x, y1, label = str(name))
            plt.plot(x, y2, label ="log(chlorophyll_a_mg_m3)")
            plt.title(f"{name} vs log(chlorophyll_a_mg_m3)\nCorrelation: {corr_value:.2f}")
            plt.xlabel("date")
            plt.ylabel("Normalized y values")
            plt.legend()

            safe_name = name.replace("/", "_") 
            
            plt.savefig(line_dir / f"lines_{safe_name}.png")
            plt.close() 
                


if __name__ == "__main__":
    main()

# linha do terminal pra correr codigo atual:
# python plots_second.py --data ../../data/chlorophyll_student_2015_2023.xlsx --output plots_second