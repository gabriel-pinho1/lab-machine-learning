from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime
import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import SplineTransformer

DATE = "Date"
DEFAULT_REGISTRY_PATH = Path("experiment_registry.csv")
def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_excel(path, sheet_name="Sheet1") if path.suffix.lower() == ".xlsx" else pd.read_csv(path)
    df[DATE] = pd.to_datetime(df[DATE])
    return df.sort_values(DATE).reset_index(drop=True)

def periodic_doy_spline( df: pd.DataFrame, n_knots: int = 13, degree: int = 3) -> pd.DataFrame:
   
    df = df.copy()
    doy = df["Date"].dt.dayofyear.to_numpy()

    # Map DOY onto a fixed [0, 1] annual phase so knots are deterministic across CV folds
    phase = ((doy - 1) / 365.25).reshape(-1, 1)
    knots = np.linspace(0.0, 1.0, n_knots).reshape(-1, 1)

    spline = SplineTransformer(
        knots=knots,
        degree=degree,
        extrapolation="periodic",
        include_bias=False,
    )
    basis = spline.fit_transform(phase)

    spline_cols = [f"doy_spline_{i}" for i in range(basis.shape[1])]
    df[spline_cols] = basis

    

    return df

def month_sine(df: pd.DataFrame):
    df = df.copy()

    df["Month_sine"] = np.sin(2*np.pi*df["Month"]/12)
    df["Month_cos"] = np.cos(2*np.pi*df["Month"]/12)

    df = df.drop(columns=["Month"])
    
    return df


def cluster_ydoy_interactions(df: pd.DataFrame):
  df = df.copy()
  for c in range(6):
    df[f"cluster_{c}_ydoy_sin"] = df[f"cluster_{c}"] * df["YDOY_sin"]
    df[f"cluster_{c}_ydoy_cos"] = df[f"cluster_{c}"] * df["YDOY"]
  return df

def biannual_sine(df: pd.DataFrame):
    df = df.copy()
    doy = df["Date"].dt.dayofyear
    df["bianual_sin"] = np.sin(4 * np.pi * doy / 365.25)
    df["Bianual_cos"] = np.cos(4 * np.pi * doy / 365.25)

    
    
    return df

def trimester_sine(df: pd.DataFrame):
    df = df.copy()
    doy = df["Date"].dt.dayofyear
    df["timester_sin"] = np.sin(8 * np.pi * doy / 365.25)
    df["trimester_cos"] = np.cos(8 * np.pi * doy / 365.25)
    return df



def flag_chl(df: pd.DataFrame) -> pd.DataFrame:
  df = df.copy()
  df["CHL_flag"] = df["CHL"].notna().astype(float)
  return df

def log_chl(df: pd.DataFrame) -> pd.DataFrame:
  df = df.copy()
  df["CHL"] = np.log1p(df["CHL"])
  return df

def SST_UI_SST_Square(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SST_UI_inter"] = df["SST"] * df["UI"]
    df["SST_squared"] = df["SST"] ** 2
    return df



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

def annual_harmonics(df: pd.DataFrame, ks=(2, 3)):
    df = df.copy()
    doy = df["Date"].dt.dayofyear
    for k in ks:
        df[f"h{k}_sin"] = np.sin(2 * np.pi * k * doy / 365.25)
        df[f"h{k}_cos"] = np.cos(2 * np.pi * k * doy / 365.25)
    return df

def cluster_season_interactions(df: pd.DataFrame, clusters=(0, 4, 5)):
    df = df.copy()
    doy = df["Date"].dt.dayofyear
    s = np.sin(2 * np.pi * doy / 365.25)
    c = np.cos(2 * np.pi * doy / 365.25)
    for k in clusters:
        m = (df["spatial_cluster"] == k).astype(float)
        df[f"int_c{k}_sin"] = m * s
        df[f"int_c{k}_cos"] = m * c
    return df


def ui_split(df: pd.DataFrame):
    df = df.copy()
    df["UI_pos"] = df["UI"].clip(lower=0)
    df["UI_neg"] = (-df["UI"]).clip(lower=0)
    return df

def phys_splines(
    df: pd.DataFrame, drop_raw: bool = True
) -> pd.DataFrame:
  df = df.copy()

  knots = np.array([
      [-2.0, 11.0],
      [0.04, 14.5],
      [2.0, 21.0],
  ])
  spline = SplineTransformer(
      n_knots=3,
      degree=2,
      knots=knots,
      include_bias=False,
  )
  sp_cols = ["UI_sp_0", "UI_sp_1", "UI_sp_2", "SST_sp_0", "SST_sp_1", "SST_sp_2"]
  df[sp_cols] = spline.fit_transform(df[["UI", "SST"]].to_numpy())
  if drop_raw:
    df = df.drop(columns=["UI", "SST"])
  return df