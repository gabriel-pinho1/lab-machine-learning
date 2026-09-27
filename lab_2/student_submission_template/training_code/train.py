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

import argparse
from pathlib import Path
import pandas as pd

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.parse_args()
    
    args = parser.parse_args()
    
    df = pd.read_excel(
        args.data, 
        sheet_name="daily_data"
    )
    
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    df.to_csv(args.output, index=False) # temporario senao n corre 

if __name__ == "__main__":
    main()

# linha do terminal pra correr codigo atual:
# python train.py --data ../../data/chlorophyll_student_2015_2023.xlsx --output cleaned_data.csv