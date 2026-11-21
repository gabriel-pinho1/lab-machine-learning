from __future__ import annotations

import argparse
from pathlib import Path
import pickle

from aux import log_experiment, month_sine, read_table, trimester_sine
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

BEST_LR_PARAMS = {
    "penalty": "l1",
    "solver": "liblinear",
    "C": 0.1,
    "class_weight": "balanced",
}

BEST_HGB_PARAMS = {
    "l2_regularization": 100.0,
    "learning_rate": 0.01,
    "max_bins": 32,
    "max_depth": 2,
    "max_features": 0.7,
    "max_iter": 200,
    "max_leaf_nodes": 4,
    "min_samples_leaf": 100
}


HGB_CAT_COLS = ["Season", "spatial_cluster"]


def build_feature_sets(df: pd.DataFrame):

  # 1. HGB features
  hgb_drop = ["Date", "HAB", "Year", "Month", "Day", "YDOY"]
  df_hgb = df.copy()
  df_hgb["abs_UI"] = np.abs(df_hgb["UI"])
  df_hgb["chl_snr"] = df_hgb["CHL"] / (df_hgb["CHL_uncertainty"] + 1e-6)
  doy = df_hgb["Date"].dt.dayofyear
  df_hgb["Doy"] =  doy 
  X_hgb = df_hgb.drop(columns=hgb_drop, errors="ignore").copy()
  hgb_cols = list(X_hgb.columns)

  # 2. LR features
  df_lr = month_sine(df.copy())
  df_lr = trimester_sine(df_lr)

  for c in range(6):
    df_lr[f"cluster_{c}"] = (df_lr["spatial_cluster"] == c).astype(float)

  for s in ["Winter", "Spring", "Summer", "Fall"]:
    df_lr[f"season_{s}"] = (df_lr["Season"] == s).astype(float)

  doy = df_lr["Date"].dt.dayofyear
  df_lr["YDOY_sin"] = np.sin(2 * np.pi * doy / 365.25)

  lr_drop = [
      "CHL",
      "CHL_uncertainty",
      "Date",
      "HAB",
      "Year",
      "Day",
      "Season",
      "spatial_cluster",
      
  ]
  X_lr = df_lr.drop(columns=lr_drop, errors="ignore").copy()

  binary_cols = [c for c in X_lr.columns if c.startswith(("cluster_", "season_"))]
  num_cols = [c for c in X_lr.columns if c not in binary_cols]

  return X_lr, num_cols, binary_cols, X_hgb, hgb_cols


def transform_lr_split(
    X_tr: pd.DataFrame,
    X_va: pd.DataFrame,
    num_cols: list[str],
    binary_cols: list[str],
):
  scaler = StandardScaler()
  X_tr_mat = np.hstack(
      [scaler.fit_transform(X_tr[num_cols]), X_tr[binary_cols].values]
  )
  X_va_mat = np.hstack(
      [scaler.transform(X_va[num_cols]), X_va[binary_cols].values]
  )
  return X_tr_mat, X_va_mat, scaler


def apply_blend(p_lr: np.ndarray, p_hgb: np.ndarray, w_lr: float, mode: str):
  if mode == "rank" and len(p_lr) > 1:
    p_lr = rankdata(p_lr) / len(p_lr)
    p_hgb = rankdata(p_hgb) / len(p_hgb)
  #return w_lr * p_lr + (1.0 - w_lr) * p_hgb
  return w_lr * (p_lr ** 2) + (1 - w_lr) * (p_hgb ** 2)


def predict_blend(artifact: dict, df_test: pd.DataFrame) -> np.ndarray:
  """Generates blended probabilities on unseen data using the saved artifact."""
  X_lr, _, _, X_hgb, _ = build_feature_sets(df_test)

  X_lr_mat = np.hstack([
      artifact["lr_scaler"].transform(X_lr[artifact["lr_num_cols"]]),
      X_lr[artifact["lr_binary_cols"]].values,
  ])
  X_hgb_df = X_hgb[artifact["hgb_cols"]]

  p_lr = artifact["lr_model"].predict_proba(X_lr_mat)[:, 1]
  p_hgb = artifact["hgb_model"].predict_proba(X_hgb_df)[:, 1]
  return apply_blend(p_lr, p_hgb, artifact["w_lr"], artifact["blend_mode"])


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--data", required=True, type=Path)
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()

  df = read_table(args.data)
  Y_all = df["HAB"].values
  years = df["Date"].dt.year.values

  X_lr_all, num_cols, binary_cols, X_hgb_all, hgb_cols = build_feature_sets(df)

  # 1. Generate leak-free out-of-fold validation predictions on [2016, 2018, 2019]
  val_years = [2016, 2018, 2019]
  lr_oof, hgb_oof, y_va_dict = {}, {}, {}
  lr_scores, hgb_scores = {}, {}

  for yr in val_years:
    tr_idx = np.where(years < yr)[0]
    va_idx = np.where(years == yr)[0]
    y_va_dict[yr] = Y_all[va_idx]

    X_tr_lr, X_va_lr, _ = transform_lr_split(
        X_lr_all.iloc[tr_idx], X_lr_all.iloc[va_idx], num_cols, binary_cols
    )
    lr_model = LogisticRegression(
        **BEST_LR_PARAMS, max_iter=2000, random_state=42
    ).fit(X_tr_lr, Y_all[tr_idx])

    hgb_model = HistGradientBoostingClassifier(
        **BEST_HGB_PARAMS, categorical_features=HGB_CAT_COLS, random_state=42
    ).fit(X_hgb_all.iloc[tr_idx], Y_all[tr_idx])

    lr_oof[yr] = lr_model.predict_proba(X_va_lr)[:, 1]
    hgb_oof[yr] = hgb_model.predict_proba(X_hgb_all.iloc[va_idx])[:, 1]

    lr_scores[yr] = roc_auc_score(y_va_dict[yr], lr_oof[yr])
    hgb_scores[yr] = roc_auc_score(y_va_dict[yr], hgb_oof[yr])

  print(
      f"LR  CV AUC: {np.mean(list(lr_scores.values())):.4f} | "
      + ", ".join(f"{yr}: {auc:.4f}" for yr, auc in lr_scores.items())
  )
  print(
      f"HGB CV AUC: {np.mean(list(hgb_scores.values())):.4f} | "
      + ", ".join(f"{yr}: {auc:.4f}" for yr, auc in hgb_scores.items())
      + "\n"
  )

  blend_candidates = []
  for w_lr in np.round(np.linspace(0.0, 1.0, 11), 2):
    for mode in ["raw", "rank"]:
      metrics_va = []
      for yr in val_years:
        p_blend = apply_blend(lr_oof[yr], hgb_oof[yr], float(w_lr), mode)
        metrics_va.append({
            "auc": roc_auc_score(y_va_dict[yr], p_blend),
            "ap": average_precision_score(y_va_dict[yr], p_blend),
            "brier": brier_score_loss(y_va_dict[yr], p_blend)
        })
      
      mean_auc = np.mean([m["auc"] for m in metrics_va])
      mean_ap = np.mean([m["ap"] for m in metrics_va])
      mean_brier = np.mean([m["brier"] for m in metrics_va])
      
      blend_candidates.append((mode, float(w_lr), float(mean_auc), float(mean_ap), float(mean_brier), metrics_va))

  best_mode, best_w_lr, best_blend_auc, best_blend_ap, best_blend_brier, best_blend_folds = max(
      blend_candidates, key=lambda r: r[2]
  )
  w_hgb = round(1.0 - best_w_lr, 1)

  print(
      f"\nWinning Blend -> mode={best_mode}, w_lr={best_w_lr:.1f}, w_hgb={w_hgb:.1f}\n"
      f"  CV AUC:   {best_blend_auc:.4f} (+/- {np.std([m['auc'] for m in best_blend_folds]):.4f})\n"
      f"  CV AP:    {best_blend_ap:.4f} (+/- {np.std([m['ap'] for m in best_blend_folds]):.4f})\n"
      f"  CV Brier: {best_blend_brier:.4f} (+/- {np.std([m['brier'] for m in best_blend_folds]):.4f})"
  )
  
  test_year = 2020
  tr_idx = np.where(years < test_year)[0]
  te_idx = np.where(years == test_year)[0]

  X_tr_lr, X_te_lr, _ = transform_lr_split(
      X_lr_all.iloc[tr_idx], X_lr_all.iloc[te_idx], num_cols, binary_cols
  )
  lr_test_model = LogisticRegression(
      **BEST_LR_PARAMS, max_iter=2000, random_state=42
  ).fit(X_tr_lr, Y_all[tr_idx])

  hgb_test_model = HistGradientBoostingClassifier(
      **BEST_HGB_PARAMS, categorical_features=HGB_CAT_COLS, random_state=42
  ).fit(X_hgb_all.iloc[tr_idx], Y_all[tr_idx])

  p_te_lr = lr_test_model.predict_proba(X_te_lr)[:, 1]
  p_te_hgb = hgb_test_model.predict_proba(X_hgb_all.iloc[te_idx])[:, 1]
  p_te_blend = apply_blend(p_te_lr, p_te_hgb, best_w_lr, best_mode)
  
  test_auc = roc_auc_score(Y_all[te_idx], p_te_blend)
  test_ap = average_precision_score(Y_all[te_idx], p_te_blend)
  test_brier = brier_score_loss(Y_all[te_idx], p_te_blend)

  print(f"\n====== EXTERNAL HOLDOUT EVALUATION (TEST {test_year}) ======")
  print(f"AUC Test {test_year}:   {test_auc:.4f}")
  print(f"AP Test {test_year}:    {test_ap:.4f}")
  print(f"Brier Test {test_year}: {test_brier:.4f}")

  PREPROCESSING_TAG = (
      "LR(drop CHL, cyclical+onehot + only trimester functions) + HGB(native CHL NaNs, native Season &"
      " cluster, drop YDOY)"
  )
  log_experiment(
      model_type=f"Blend_LR_{best_w_lr:.1f}_HGB_{w_hgb:.1f}_{best_mode}",
      preprocessing_tag=PREPROCESSING_TAG,
      val_auc=best_blend_auc,
      best_params={
          "w_lr": best_w_lr,
          "w_hgb": w_hgb,
          "blend_mode": best_mode,
          "holdout_2020_auc": round(float(test_auc), 4),
          "lr": BEST_LR_PARAMS,
          "hgb": BEST_HGB_PARAMS,
      },
  )

  final_scaler = StandardScaler()
  X_full_lr_mat = np.hstack([
      final_scaler.fit_transform(X_lr_all[num_cols]),
      X_lr_all[binary_cols].values,
  ])
  final_lr = LogisticRegression(
      **BEST_LR_PARAMS, max_iter=2000, random_state=42
  ).fit(X_full_lr_mat, Y_all)

  final_hgb = HistGradientBoostingClassifier(
      **BEST_HGB_PARAMS, categorical_features=HGB_CAT_COLS, random_state=42
  ).fit(X_hgb_all, Y_all)

  model_artifact = {
      "lr_model": final_lr,
      "hgb_model": final_hgb,
      "w_lr": best_w_lr,
      "blend_mode": best_mode,
      "lr_scaler": final_scaler,
      "lr_num_cols": num_cols,
      "lr_binary_cols": binary_cols,
      "hgb_cols": hgb_cols,
      "hgb_cat_cols": HGB_CAT_COLS,
  }

  args.output.parent.mkdir(parents=True, exist_ok=True)
  with open(args.output, "wb") as f:
    pickle.dump(model_artifact, f)


if __name__ == "__main__":
  main()

# python train_blend.py --data ../../data/classification_task1_train.csv --output ../best_model/model_blend.pkl 