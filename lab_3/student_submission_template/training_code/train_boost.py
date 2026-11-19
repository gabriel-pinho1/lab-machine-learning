from __future__ import annotations

import argparse
from pathlib import Path
import pickle

from aux import log_experiment, read_table
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid

CAT_COLS = ["Season", "spatial_cluster"]


def prepare_split(
    X: pd.DataFrame,
    Y: pd.Series,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
):
  return (
      X.iloc[train_idx].copy(),
      Y.iloc[train_idx].values,
      X.iloc[val_idx].copy(),
      Y.iloc[val_idx].values,
  )


def _evaluate_params(params: dict, folds: list):
  val_scores = {}
  train_scores = {}

  for year, X_tr, Y_tr, X_va, Y_va in folds:
    model = HistGradientBoostingClassifier(
        **params,
        categorical_features=CAT_COLS,
        random_state=42,
    )
    model.fit(X_tr, Y_tr)

    val_scores[year] = roc_auc_score(Y_va, model.predict_proba(X_va)[:, 1])
    train_scores[year] = roc_auc_score(Y_tr, model.predict_proba(X_tr)[:, 1])

  mean_val_score = float(np.mean(list(val_scores.values())))
  mean_train_score = float(np.mean(list(train_scores.values())))

  return (
      params,
      mean_val_score,
      {
          "val_scores": val_scores,
          "train_scores": train_scores,
          "mean_train": mean_train_score,
      },
  )


def tune_hyperparameters_boost(folds: list, param_grid: list, n_jobs: int = -1):
  results = Parallel(n_jobs=n_jobs)(
      delayed(_evaluate_params)(params, folds) for params in param_grid
  )
  best_params, best_score, best_details = max(results, key=lambda r: r[1])
  return best_params, best_score, best_details


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--data", required=True, type=Path)
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()

  df = read_table(args.data)

 

  Y_all = df["HAB"]
  X_all = df.drop(columns=["Date", "HAB", "Year", "Month", "Day", "YDOY"])
  years = df["Date"].dt.year.values

  PREPROCESSING_TAG = (
      "native CHL NaNs + native Season & cluster + YDOY sin/cos + "
      "drop Year/Month/Day + drop "
  )

  param_grid = list(
    ParameterGrid({
    "max_iter": [150, 250, 350],
    "learning_rate": [0.01, 0.02, 0.03],
    "max_leaf_nodes": [4, 6, 8],
    "max_depth": [3],
    "min_samples_leaf": [40, 60, 80],
    "l2_regularization": [5.0, 20.0, 50.0],
    "max_features": [0.5, 0.7, 0.9],
    "max_bins": [32, 64, 128],
})
)

  val_years = [2016, 2018, 2019]
  inner_folds = [
      (
          yr,
          *prepare_split(
              X_all,
              Y_all,
              np.where(years < yr)[0],
              np.where(years == yr)[0],
          ),
      )
      for yr in val_years
  ]

  best_params, mean_val_auc, best_details = tune_hyperparameters_boost(
      inner_folds, param_grid
  )

  print("Cross Validation results")
  print(f"Best hyperparameters: {best_params}")
  print(
      f"Average AUC training: {best_details['mean_train']:.4f} (to avoid"
      " memorization)"
  )
  print(
      f"Average validation AUC: {mean_val_auc:.4f} (+/-"
      f" {np.std(list(best_details['val_scores'].values())):.4f})"
  )

  print("\nDetails by validation year:")
  for yr, auc in best_details["val_scores"].items():
    train_auc = best_details["train_scores"][yr]
    print(f"  Year {yr} | Train AUC: {train_auc:.4f} -> Val AUC: {auc:.4f}")

  test_year = 2020
  X_tr_test, Y_tr_test, X_te, Y_te = prepare_split(
      X_all,
      Y_all,
      np.where(years < test_year)[0],
      np.where(years == test_year)[0],
  )

  final_test_model = HistGradientBoostingClassifier(
      **best_params,
      categorical_features=CAT_COLS,
      random_state=42,
  )
  final_test_model.fit(X_tr_test, Y_tr_test)
  test_auc = roc_auc_score(Y_te, final_test_model.predict_proba(X_te)[:, 1])

  print(f"\nTest year {test_year}")
  print(f"AUC Test {test_year}: {test_auc:.4f}")

  log_experiment(
      model_type="HistGradientBoostingClassifier",
      preprocessing_tag=PREPROCESSING_TAG,
      val_auc=mean_val_auc,
      best_params=best_params,
  )

  # Refit on the full dataset (including 2020) before saving
  final_model = HistGradientBoostingClassifier(
      **best_params,
      categorical_features=CAT_COLS,
      random_state=42,
  )
  final_model.fit(X_all, Y_all.values)

  model_artifact = {
      "model": final_model,
      "scaler": None,
      "chl_scaler": None,
      "cols": list(X_all.columns),
      "cat_cols": CAT_COLS,
  }

  args.output.parent.mkdir(parents=True, exist_ok=True)
  with open(args.output, "wb") as f:
    pickle.dump(model_artifact, f)


if __name__ == "__main__":
  main()

# python train_boost.py --data ../../data/classification_task1_train.csv --output ../best_model/model_boost.pkl