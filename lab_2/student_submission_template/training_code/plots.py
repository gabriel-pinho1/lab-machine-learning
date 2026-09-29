from __future__ import annotations
from build_lagged_features import build_lagged_frame, read_table
import argparse
from pathlib import Path

# Add this to prevent matplotlib from trying to open windows (useful for scripts)
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    
    # You had parser.parse_args() twice in a row; you only need it once
    args = parser.parse_args()
    
    df_raw = read_table(args.data)
    lagged_df = build_lagged_frame(df_raw)
    Y = lagged_df["chlorophyll_a_mg_m3"]
    X = lagged_df.drop(columns=["date", "chlorophyll_a_mg_m3"])
    split_idx = -365
    X_train = X.iloc[:split_idx]
    Y_train = Y.iloc[:split_idx]
    lagged_columns_names = X_train.columns.tolist()
    inputs = lagged_columns_names
    
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    df_raw_copy = lagged_df.drop(columns=["date"]).copy()
    df_raw_scaled = scaler.fit_transform(df_raw_copy)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    df_raw_scaled = pd.DataFrame(df_raw_scaled, columns=df_raw_copy.columns)
    raw_collumn_names = df_raw_scaled.columns.tolist()
    df_raw_scaled = pd.concat([lagged_df['date'], df_raw_scaled], axis = 1)
    

    scatter_dir = args.output / "scatter_plots"
    line_dir = args.output / "line_plots"
    
    scatter_dir.mkdir(parents=True, exist_ok=True)
    line_dir.mkdir(parents=True, exist_ok=True)
    
    '''  for name in inputs:
        x = X_train_scaled[name]
        y = Y_train.values

        plt.scatter(x, y)
        plt.title(f"Scatter Plot: {name}")
        plt.xlabel(f"{name}")
        plt.ylabel("Y Values")

        safe_name = name.replace("/", "_") 
        
        plt.savefig(scatter_dir / f"scatter_{safe_name}.png")
        plt.close() 
        
    print("yay2") '''
    
    
    for name in raw_collumn_names:

        x =  df_raw_scaled['date']
        y1 = df_raw_scaled[name]
        y2 = df_raw_scaled['chlorophyll_a_mg_m3']
        corr_value = y1.corr(y2)
        plt.plot(x, y1, label = str(name))
        plt.plot(x, y2, label ="chlorophyll_a_mg_m3")
        plt.title(f"{name} vs chlorophyll_a_mg_m3\nCorrelation: {corr_value:.2f}")
        plt.xlabel("date")
        plt.ylabel("Normalized y values")
        plt.legend()

        safe_name = name.replace("/", "_") 
        
        plt.savefig(line_dir / f"lines_{safe_name}.png")
        plt.close() 
                


if __name__ == "__main__":
    main()

# linha do terminal pra correr codigo atual:
# python plots.py --data ../../data/chlorophyll_student_2015_2023.xlsx --output scatter_plots