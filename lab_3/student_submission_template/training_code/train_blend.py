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
from sklearn.metrics import roc_auc_score
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
    "max_depth": 3,
    "max_features": 0.6,
    "max_iter": 200,
    "max_leaf_nodes": 5,
    "min_samples_leaf": 100,
}

HGB_CAT_COLS = ["Season", "spatial_cluster"]


def build_feature_sets(df: pd.DataFrame):

  # 1. HGB features
  hgb_drop = ["Date", "HAB", "Year", "Month", "Day", "YDOY"]
  X_hgb = df.drop(columns=hgb_drop, errors="ignore").copy()
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
  return w_lr * p_lr + (1.0 - w_lr) * p_hgb


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

  # 2. Sweep blend weights across raw probabilities and rank averages on CV
  print("=== Blend Weight Sweep on CV (2016 / 2018 / 2019) ===")
  print(
      "w_lr | Raw Prob Mean AUC (2016/2018/2019) | Rank Avg Mean AUC"
      " (2016/2018/2019)"
  )
  print("-" * 78)

  blend_candidates = []
  for w_lr in np.round(np.linspace(0.0, 1.0, 11), 2):
    mode_aucs = {}
    for mode in ["raw", "rank"]:
      aucs = [
          roc_auc_score(
              y_va_dict[yr],
              apply_blend(lr_oof[yr], hgb_oof[yr], float(w_lr), mode),
          )
          for yr in val_years
      ]
      mode_aucs[mode] = aucs
      blend_candidates.append((mode, float(w_lr), float(np.mean(aucs)), aucs))

    r_aucs = mode_aucs["raw"]
    k_aucs = mode_aucs["rank"]
    print(
        f"{w_lr:4.1f} | {np.mean(r_aucs):.4f}"
        f" ({r_aucs[0]:.3f}/{r_aucs[1]:.3f}/{r_aucs[2]:.3f}) |"
        f" {np.mean(k_aucs):.4f}"
        f" ({k_aucs[0]:.3f}/{k_aucs[1]:.3f}/{k_aucs[2]:.3f})"
    )

  best_mode, best_w_lr, best_blend_auc, best_blend_folds = max(
      blend_candidates, key=lambda r: r[2]
  )
  w_hgb = round(1.0 - best_w_lr, 1)

  print(
      f"\nWinning Blend -> mode={best_mode}, w_lr={best_w_lr:.1f},"
      f" w_hgb={w_hgb:.1f}, CV AUC={best_blend_auc:.4f} (+/-"
      f" {np.std(best_blend_folds):.4f})"
  )

  # 3. Single-shot holdout evaluation on 2020 (no individual model leaking)
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

  print(f"\n====== EXTERNAL HOLDOUT EVALUATION (TEST {test_year}) ======")
  print(f"AUC Test {test_year} (Winning Blend): {test_auc:.4f}")

  # 4. Log experiment to registry
  PREPROCESSING_TAG = (
      "LR(drop CHL, cyclical+onehot) + HGB(native CHL NaNs, native Season &"
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

  # 5. Refit both models together on the full dataset (2015-2020) and serialize
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