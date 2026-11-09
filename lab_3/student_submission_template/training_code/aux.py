from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime
import csv
import numpy as np
import pandas as pd


DATE = "Date"
DEFAULT_REGISTRY_PATH = Path("experiment_registry.csv")
def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_excel(path, sheet_name="Sheet1") if path.suffix.lower() == ".xlsx" else pd.read_csv(path)
    df[DATE] = pd.to_datetime(df[DATE])
    return df.sort_values(DATE).reset_index(drop=True)


def log_experiment(
    
    model_type: str,
    preprocessing_tag: str,
    val_auc: float,
    best_params: dict,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> None:
    #Appends model metrics, hyperparameters, and preprocessing labels to a registry CSV.
    registry_path = Path(registry_path)
    file_exists = registry_path.exists()

    fieldnames = [
        "timestamp",
        "model_type",
        "preprocessing_pipeline",
        "val_roc_auc",
        "best_hyperparameters",
    ]

    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": model_type,
        "preprocessing_pipeline": preprocessing_tag,
        "val_roc_auc": f"{val_auc:.4f}",
        "best_hyperparameters": str(best_params),
    }

    with open(registry_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)

    print(f"Experiment logged to: {registry_path}")