from __future__ import annotations

import argparse
from pathlib import Path
import pickle

from aux import log_experiment, month_sine, read_table, trimester_sine, SST_UI_SST_Square, annual_harmonics, cluster_season_interactions, ui_split
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler



def prepare_split(X: pd.DataFrame,Y: pd.Series,train_idx: np.ndarray,val_idx: np.ndarray,num_cols: list[str],binary_cols: list[str],):
  X_tr, Y_tr = X.iloc[train_idx].copy(), Y.iloc[train_idx].copy()
  X_val, Y_val = X.iloc[val_idx].copy(), Y.iloc[val_idx].copy()

  num_scaler = StandardScaler()
  X_tr_num = num_scaler.fit_transform(X_tr[num_cols])
  X_val_num = num_scaler.transform(X_val[num_cols])

  chl_cols = [c for c in ["CHL", "CHL_uncertainty"] if c in X_tr.columns]
  if chl_cols:
    chl_scaler = StandardScaler()
    X_tr_chl = np.nan_to_num(chl_scaler.fit_transform(X_tr[chl_cols]), nan=0.0)
    X_val_chl = np.nan_to_num(chl_scaler.transform(X_val[chl_cols]), nan=0.0)

    X_tr_mat = np.hstack([X_tr_chl, X_tr_num, X_tr[binary_cols].values])
    X_val_mat = np.hstack([X_val_chl, X_val_num, X_val[binary_cols].values])
  else:
    X_tr_mat = np.hstack([X_tr_num, X_tr[binary_cols].values])
    X_val_mat = np.hstack([X_val_num, X_val[binary_cols].values])

  return X_tr_mat, Y_tr.values, X_val_mat, Y_val.values


def tune_hyperparameters(folds: list, param_grid: list):
  best_params, best_score, best_details = None, -1.0, {}

  for params in param_grid:
    scores = {}
    for year, X_tr, Y_tr, X_va, Y_va in folds:
      model = LogisticRegression(**params, max_iter=2000, random_state=42)
      model.fit(X_tr, Y_tr)
      scores[year] = roc_auc_score(Y_va, model.predict_proba(X_va)[:, 1])

    mean_score = np.mean(list(scores.values()))
    if mean_score > best_score:
      best_score, best_params, best_details = mean_score, params, scores

  return best_params, best_score, best_details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    df = read_table("../../data/classification_task1_train.csv")
    df = month_sine(df)
    #df = trimester_sine(df)
    #df = SST_UI_SST_Square(df)
    #df = annual_harmonics(df)
    #df = cluster_season_interactions(df)
    df = ui_split(df)

    for c in range(6):
        df[f"cluster_{c}"] = (df["spatial_cluster"] == c).astype(float)

    for s in ["Winter", "Spring", "Summer", "Fall"]:
        df[f"season_{s}"] = (df["Season"] == s).astype(float)

    doy = df["Date"].dt.dayofyear
    df["YDOY_sin"] = np.sin(2 * np.pi * doy / 365.25)

    Y_all = df["HAB"]
    X_all = df.drop(
        columns=["CHL", "CHL_uncertainty","Date", "HAB", "Year", "Day", "Season", "spatial_cluster", "UI"]
    )
    years = df["Date"].dt.year.values

    PREPROCESSING_TAG = (
        "Drop CHL + drop Year + drop Day + "
        "replace Season + replace Month + replace cluster +"
        " trimester sine + " "UI divided and without normal UI"  
    )

    binary_cols = [c for c in X_all.columns if c.startswith(("cluster_", "season_"))]
    num_cols = [
        c for c in X_all.columns if c not in binary_cols + ["CHL", "CHL_uncertainty"]
    ]

    # Defining the parameter grid
    param_grid = [
        {"penalty": pen, "solver": solv, "C": c, "class_weight": cw}
        for cw in [None, "balanced"]
        for c in np.logspace(-4, 3, 50)
        for pen, solv in [("l2", "lbfgs"), ("l1", "liblinear")]
    ] + [
        {"penalty": None, "solver": "lbfgs", "class_weight": cw}
        for cw in [None, "balanced"]
    ]

    # Inner loop tuning (cross validation)
    val_years = [2016, 2018, 2019]
    inner_folds = [
        (
            yr,
            *prepare_split(
                X_all,
                Y_all,
                np.where(years < yr)[0],
                np.where(years == yr)[0],
                num_cols,
                binary_cols,
            ),
        )
        for yr in val_years
    ]

    best_params, mean_auc, fold_aucs = tune_hyperparameters(
        inner_folds, param_grid
    )

    print("Cross Validation Best Results")
    print(f"Best Hyperparameters: {best_params}")
    for yr, auc in fold_aucs.items():
        print(f"  Year {yr}: AUC = {auc:.4f}")
        print(
            f"Validation average AUC: {mean_auc:.4f} (+/-"
            f" {np.std(list(fold_aucs.values())):.4f})\n"
        )


    test_year = 2020
    X_tr_test, Y_tr_test, X_te, Y_te = prepare_split(
        X_all,
        Y_all,
        np.where(years < test_year)[0],
        np.where(years == test_year)[0],
        num_cols,
        binary_cols,
    )

    final_test_model = LogisticRegression(
        **best_params, max_iter=2000, random_state=42
    )
    final_test_model.fit(X_tr_test, Y_tr_test)
    test_auc = roc_auc_score(Y_te, final_test_model.predict_proba(X_te)[:, 1])

    print(f"====== AVALIAÇÃO EXTERNA (TESTE {test_year}) ======")
    print(f"AUC Teste {test_year}: {test_auc:.4f}")

    log_experiment(
        model_type="LogisticRegressor",
        preprocessing_tag=PREPROCESSING_TAG,
        val_auc=mean_auc,
        best_params=best_params,
    )

    # Refit scaler and model on the full dataset (including 2020) before saving
    X_full = X_all
    final_scaler = StandardScaler()
    X_full_mat = np.hstack(
        [final_scaler.fit_transform(X_full[num_cols]), X_full[binary_cols].values]
    )

    final_model = LogisticRegression(**best_params, max_iter=2000, random_state=42)
    final_model.fit(X_full_mat, Y_all.values)

    # Save the fitted model, scaler, and column metadata
    model_artifact = {
        "model": final_model,
        "scaler": final_scaler,
        "cols": num_cols + binary_cols,
        "num_cols": num_cols,
        "binary_cols": binary_cols,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(model_artifact, f)


if __name__ == "__main__":
  main()

#python train_LR.py --data ../../data/classification_task1_train.csv --output ../best_model/model_LR.pkl