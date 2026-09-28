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
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    Y_train, Y_val = Y.iloc[:split_idx], Y.iloc[split_idx:]
    
    column_names = X_train.columns.tolist()
    inputs = column_names
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns)
    
    print("yay")
    
    # CRITICAL FIX: Create the output directory if it doesn't exist
    args.output.mkdir(parents=True, exist_ok=True)
    
    for name in inputs:
        x = X_train_scaled[name]
        y = Y_train.values

        plt.scatter(x, y)
        plt.title(f"Scatter Plot: {name}")
        plt.xlabel("X Values")
        plt.ylabel("Y Values")

        safe_name = name.replace("/", "_") 
        
        # Now this will successfully save because the directory exists
        plt.savefig(args.output / f"scatter_{safe_name}.png")
        plt.close() # Free up memory
        
    print("yay2")

if __name__ == "__main__":
    main()

# linha do terminal pra correr codigo atual:
# python plots.py --data ../../data/chlorophyll_student_2015_2023.xlsx --output scatter_plots