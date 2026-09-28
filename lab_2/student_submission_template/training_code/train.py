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

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.parse_args()
    
    args = parser.parse_args()
    
    df_history = pd.read_excel(
        args.data, 
        sheet_name="daily_data"
    )
    
    df_history["date"] = pd.to_datetime(df_history["date"])
    df_history = df_history.sort_values("date").reset_index(drop=True)

    df_X = pd.DataFrame()
    p_max = 15
    q_max = 15
    column_names = df_history.columns.tolist()
    exogenous_variables = column_names[2:]
    new_lags = {}
    Y = df_history["chlorophyll_a_mg_m3"]
    for p in range(1, p_max + 1):

        new_lags[f'chlorophyll_lag_{p}'] = df_history["chlorophyll_a_mg_m3"].shift(p)

    for name in exogenous_variables:
        for q in range (0,q_max):

            new_lags[f'{name}_lag_{q}'] = df_history[f'{name}'].shift(q)


    new_lags["s_year"] = np.sin(2 * np.pi * df_history["date"].dt.dayofyear / 365.25)
    new_lags["c_year"] = np.cos(2 * np.pi * df_history["date"].dt.dayofyear / 365.25)
    df_X = pd.DataFrame(new_lags)
    df_X = df_X.fillna(0)
 


    df_X.to_csv(args.output, index=False) # temporario senao n corre 

if __name__ == "__main__":
    main()

# linha do terminal pra correr codigo atual:
# python train.py --data ../../data/chlorophyll_student_2015_2023.xlsx --output cleaned_data.csv