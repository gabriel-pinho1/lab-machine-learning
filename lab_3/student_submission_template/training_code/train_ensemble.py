from __future__ import annotations

import argparse
from pathlib import Path
import pickle

from aux import log_experiment, read_table, flag_chl, log_chl
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from sklearn.ensemble import BaggingClassifier


def build_mlp_ensemble(
    best_params: dict, n_models: int = 10, n_jobs: int = -1
):
  base_mlp = MLPClassifier(**best_params, max_iter=100000)
  return BaggingClassifier(
      estimator=base_mlp,
      n_estimators=n_models,
      bootstrap=False,
      n_jobs=n_jobs,
      random_state=42,
  )





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





def _evaluate_params(params: dict, folds: list, tune_seeds=(42, 0, 1)):
  val_scores = {}
  train_scores = {}

  for year, X_tr, Y_tr, X_va, Y_va in folds:
    val_preds = []
    train_preds = []
    for seed in tune_seeds:
      model = MLPClassifier(**params, max_iter=100000, random_state=seed)
      model.fit(X_tr, Y_tr)
      val_preds.append(model.predict_proba(X_va)[:, 1])
      train_preds.append(model.predict_proba(X_tr)[:, 1])

    val_scores[year] = roc_auc_score(Y_va, np.mean(val_preds, axis=0))
    train_scores[year] = roc_auc_score(Y_tr, np.mean(train_preds, axis=0))

  mean_val_score = float(np.mean(list(val_scores.values())))
  mean_train_score = float(np.mean(list(train_scores.values())))

  return (params, mean_val_score,{"val_scores": val_scores,"train_scores": train_scores,"mean_train": mean_train_score,},)

def evaluate_ensemble_cv(folds: list, best_params: dict, n_models: int = 10):
  val_scores = {}
  train_scores = {}

  for year, X_tr, Y_tr, X_va, Y_va in folds:
    ensemble = build_mlp_ensemble(best_params, n_models=n_models, n_jobs=-1)
    ensemble.fit(X_tr, Y_tr)

    val_scores[year] = roc_auc_score(Y_va, ensemble.predict_proba(X_va)[:, 1])
    train_scores[year] = roc_auc_score(Y_tr, ensemble.predict_proba(X_tr)[:, 1])

  mean_val = float(np.mean(list(val_scores.values())))
  mean_train = float(np.mean(list(train_scores.values())))
  individual_aucs = [
    roc_auc_score(Y_va, est.predict_proba(X_va)[:, 1])
    for est in ensemble.estimators_
    ]
  print(
       f"Year {year} individual seed AUCs: {[round(a, 4) for a in individual_aucs]}"
    )
  return mean_val, {
      "val_scores": val_scores,
      "train_scores": train_scores,
      "mean_train": mean_train,
  }

def tune_hyperparameters_nn(folds: list, param_grid: list, n_jobs: int = -1):
  results = Parallel(n_jobs=n_jobs)(
      delayed(_evaluate_params)(params, folds) for params in param_grid
  )

  # max() preserves first-seen tie-breaking, matching your original loop
  best_params, best_score, best_details = max(results, key=lambda r: r[1])
  return best_params, best_score, best_details


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--data", required=True, type=Path)
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()

  df = read_table(args.data)

  for c in range(6):
    df[f"cluster_{c}"] = (df["spatial_cluster"] == c).astype(float)

  for s in ["Winter", "Spring", "Summer", "Fall"]:
    df[f"season_{s}"] = (df["Season"] == s).astype(float)

  doy = df["Date"].dt.dayofyear
  df["YDOY_sin"] = np.sin(2 * np.pi * doy / 365.25)

  df = flag_chl(df)  

  Y_all = df["HAB"]
  # Add "CHL", "CHL_uncertainty" here if dropping them
  X_all = df.drop(columns=[ "Date", "HAB", "Year","Month", "Day","Season","spatial_cluster",])
  
  years = df["Date"].dt.year.values

  PREPROCESSING_TAG = (
      "flag CHL + log CHL + drop Year + drop Month + drop Day + "
      "replace Season + replace cluster"
  )

  binary_cols = [ c for c in X_all.columns if c.startswith(("cluster_", "season_", "CHL_flag"))]
  num_cols = [c for c in X_all.columns if c not in binary_cols + ["CHL", "CHL_uncertainty"]]

  param_grid = list(
      ParameterGrid({
          "hidden_layer_sizes": [(4,), (8,), (16,), (32,), (32, 16), (8, 4)],
          "activation": ["relu", "tanh"],
          "solver": ["adam", "lbfgs"],
          "alpha": [0.001, 0.01, 0.1, 1.0, 10],
          
      })
  )

  val_years = [2016, 2018, 2019]
  inner_folds = [  (yr,*prepare_split( X_all, Y_all, np.where(years < yr)[0],  np.where(years == yr)[0], num_cols, binary_cols,),)
      for yr in val_years
  ]

  best_params, single_val_auc, single_details = tune_hyperparameters_nn(
      inner_folds, param_grid
  )

  N_MODELS = 10
  mean_val_auc, best_details = evaluate_ensemble_cv(
      inner_folds, best_params, n_models=N_MODELS
  )

  print("Cross Validation results")
  print(f"Best hyperparameters: {best_params}")
  print(
      f"Single MLP validation AUC: {single_val_auc:.4f} (+/-"
      f" {np.std(list(single_details['val_scores'].values())):.4f})"
  )
  print(
      f"Ensemble ({N_MODELS}x) average training AUC:"
      f" {best_details['mean_train']:.4f}"
  )
  print(
      f"Ensemble ({N_MODELS}x) average validation AUC: {mean_val_auc:.4f} (+/-"
      f" {np.std(list(best_details['val_scores'].values())):.4f})"
  )

  print("\nDetails by validation year (Ensemble):")
  for yr, auc in best_details["val_scores"].items():
    train_auc = best_details["train_scores"][yr]
    print(f"  Year {yr} | Train AUC: {train_auc:.4f} -> Val AUC: {auc:.4f}")



  test_year = 2020
  X_tr_test, Y_tr_test, X_te, Y_te = prepare_split(
      X_all,
      Y_all,
      np.where(years < test_year)[0],
      np.where(years == test_year)[0],
      num_cols,
      binary_cols,
  )

  final_test_model = build_mlp_ensemble(best_params, n_models=N_MODELS)
  final_test_model.fit(X_tr_test, Y_tr_test)
  test_auc = roc_auc_score(Y_te, final_test_model.predict_proba(X_te)[:, 1])

  print(f"\nTest year {test_year}")
  print(f"AUC Test {test_year}: {test_auc:.4f}")

  log_experiment(
      model_type=f"MLPClassifier_Ensemble_{N_MODELS}",
      preprocessing_tag=PREPROCESSING_TAG,
      val_auc=mean_val_auc,
      best_params=best_params,
  )

  # Refit scalers and ensemble on the full dataset (including 2020) before saving
  final_scaler = StandardScaler()
  X_full_num = final_scaler.fit_transform(X_all[num_cols])

  chl_cols = [c for c in ["CHL", "CHL_uncertainty"] if c in X_all.columns]
  if chl_cols:
    final_chl_scaler = StandardScaler()
    X_full_chl = np.nan_to_num(
        final_chl_scaler.fit_transform(X_all[chl_cols]), nan=0.0
    )
    X_full_mat = np.hstack([X_full_chl, X_full_num, X_all[binary_cols].values])
  else:
    final_chl_scaler = None
    X_full_mat = np.hstack([X_full_num, X_all[binary_cols].values])

  final_model = build_mlp_ensemble(best_params, n_models=N_MODELS)
  final_model.fit(X_full_mat, Y_all.values)

  model_artifact = {
      "model": final_model,
      "scaler": final_scaler,
      "chl_scaler": final_chl_scaler,
      "cols": chl_cols + num_cols + binary_cols,
      "num_cols": num_cols,
      "binary_cols": binary_cols,
      "chl_cols": chl_cols,
  }

  args.output.parent.mkdir(parents=True, exist_ok=True)
  with open(args.output, "wb") as f:
    pickle.dump(model_artifact, f)


if __name__ == "__main__":
  main()

# python train_ensemble.py --data ../../data/classification_task1_train.csv --output ../best_model/model_ensemble.pkl