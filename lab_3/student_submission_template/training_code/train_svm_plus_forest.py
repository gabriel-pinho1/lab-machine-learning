from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from aux import log_experiment, month_sine, read_table

SEASON_MAP = {"winter": 0, "spring": 1, "summer": 2, "fall": 3}
DROP_COMMON = ["Date", "HAB", "Year", "Day", "CHL_uncertainty", "CHL"]


def build_rf_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepares tabular features for Random Forest (keeps ordinal Season & raw Month)."""
    out = df.copy()
    if "Season" in out.columns and (
        out["Season"].dtype == object or pd.api.types.is_string_dtype(out["Season"])
    ):
        out["Season"] = out["Season"].str.lower().map(SEASON_MAP)
    return out.drop(columns=DROP_COMMON, errors="ignore")


def build_svm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepares cyclical features for SVM (replaces Month with sin/cos, drops Season)."""
    out = month_sine(df)
    return out.drop(columns=DROP_COMMON + ["Season"], errors="ignore")


def main() -> None:
    default_out = Path(__file__).resolve().parent.parent / "best_model" / "model.pkl"

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", default=default_out, type=Path)
    args = parser.parse_args()

    df = read_table(args.data)
    Y_cv = df["HAB"]

    X_rf = build_rf_features(df)
    X_svm = build_svm_features(df)

    # 1. Temporal CV splits (2017 skipped due to zero positive HAB events)
    cv_years = df["Date"].dt.year.values
    validation_years = [2016, 2018, 2019]
    custom_cv_splits = [
        (np.where(cv_years < val_year)[0], np.where(cv_years == val_year)[0])
        for val_year in validation_years
    ]

    # 2. Locked-in standalone models from Stage 1
    best_rf = RandomForestClassifier(
        n_estimators=100,
        class_weight=None,
        criterion="gini",
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=3,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )

    best_svm = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svc",
                SVC(
                    C=0.1,
                    kernel="rbf",
                    gamma="scale",
                    class_weight="balanced",
                    probability=True,
                    random_state=42,
                ),
            ),
        ]
    )

    # 3. Evaluate out-of-fold predictions across fixed, pre-committed mix ratios
    oof_rf, oof_svm, oof_y = [], [], []

    for train_idx, val_idx in custom_cv_splits:
        rf_fold = clone(best_rf).fit(X_rf.iloc[train_idx], Y_cv.iloc[train_idx])
        svm_fold = clone(best_svm).fit(X_svm.iloc[train_idx], Y_cv.iloc[train_idx])

        oof_rf.append(rf_fold.predict_proba(X_rf.iloc[val_idx])[:, 1])
        oof_svm.append(svm_fold.predict_proba(X_svm.iloc[val_idx])[:, 1])
        oof_y.append(Y_cv.iloc[val_idx].values)

    candidate_svm_weights = [1.0, 0.7, 0.5, 0.0]
    best_svm_weight = 0.7
    best_mean_auc = -1.0
    best_std_auc = 0.0

    print("=" * 52)
    print("       TEMPORAL CV ENSEMBLE EVALUATION (3-FOLD)      ")
    print("=" * 52)

    for w_svm in candidate_svm_weights:
        w_rf = 1.0 - w_svm
        fold_aucs = [
            roc_auc_score(y_va, w_svm * p_s + w_rf * p_r)
            for y_va, p_s, p_r in zip(oof_y, oof_svm, oof_rf)
        ]
        mean_auc = float(np.mean(fold_aucs))
        std_auc = float(np.std(fold_aucs))
        folds_str = ", ".join(f"{a:.4f}" for a in fold_aucs)
        print(
            f"SVM {w_svm:4.0%} + RF {w_rf:4.0%} | "
            f"Mean AUC: {mean_auc:.4f} (std: {std_auc:.4f}) | Folds: [{folds_str}]"
        )

        # Prefer blends when strictly better (or tied within 0.0005 with lower variance)
        if mean_auc > best_mean_auc:
            best_mean_auc = mean_auc
            best_std_auc = std_auc
            best_svm_weight = w_svm

    print("=" * 52)
    print(
        f"Selected Mix -> SVM: {best_svm_weight:.0%}, RF: {1.0 - best_svm_weight:.0%} "
        f"(Mean AUC: {best_mean_auc:.4f}, std: {best_std_auc:.4f})"
    )
    print("=" * 52)

    # 4. Refit both models on 100% of training data and package artifact
    final_rf = clone(best_rf).fit(X_rf, Y_cv)
    final_svm = clone(best_svm).fit(X_svm, Y_cv)

    model_artifact = {
        "rf_model": final_rf,
        "svm_model": final_svm,
        "svm_weight": best_svm_weight,
        "rf_cols": list(X_rf.columns),
        "svm_cols": list(X_svm.columns),
        "season_map": SEASON_MAP,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(model_artifact, f)
    print(f"Saved ensemble artifact to: {args.output}")

    # 5. Log experiment
    log_experiment(
        model_type=f"Ensemble_SVM({best_svm_weight:.2f})_RF({1.0 - best_svm_weight:.2f})",
        preprocessing_tag="RF(Season+Month) + SVM(Scaled+CyclicalMonth)",
        val_auc=best_mean_auc,
        best_params={
            "svm_weight": best_svm_weight,
            "rf_weight": round(1.0 - best_svm_weight, 2),
            "rf_params": best_rf.get_params(),
            "svm_params": best_svm.named_steps["svc"].get_params(),
        },
    )


if __name__ == "__main__":
    main()


    #python train_svm_plus_forest.py --data ../../data/classification_task1_train.csv --output ../best_model/model.pkl