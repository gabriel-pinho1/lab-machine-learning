from __future__ import annotations

import argparse
from pathlib import Path
import pickle

from aux import log_experiment, read_table
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import ParameterGrid

CAT_COLS = ["Season", "spatial_cluster"]

def prepare_split(X: pd.DataFrame, Y: pd.Series, train_idx: np.ndarray, val_idx: np.ndarray,):
    return (X.iloc[train_idx].copy(),
    
        Y.iloc[train_idx].values,
    
        X.iloc[val_idx].copy(),
    
        Y.iloc[val_idx].values)



def _evaluate_params(params: dict, folds: list):
    val_metrics = {}
    train_scores = {}

    for year, X_tr, Y_tr, X_va, Y_va in folds:
        model = HistGradientBoostingClassifier(
            **params,
            categorical_features=CAT_COLS,
            random_state=42,
        )
        model.fit(X_tr, Y_tr)

        y_prob_va = model.predict_proba(X_va)[:, 1]
        y_prob_tr = model.predict_proba(X_tr)[:, 1]

        val_metrics[year] = {
            "auc": roc_auc_score(Y_va, y_prob_va),
            "ap": average_precision_score(Y_va, y_prob_va),
            "brier": brier_score_loss(Y_va, y_prob_va)
        }
        train_scores[year] = roc_auc_score(Y_tr, y_prob_tr)

    mean_val_auc = float(np.mean([m["auc"] for m in val_metrics.values()]))
    mean_train_score = float(np.mean(list(train_scores.values())))

    return (
        params,
        mean_val_auc,
        {
            "val_metrics": val_metrics,
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
    #X_all = df.drop(columns=["Date", "HAB", "Year", "Month", "Day", "YDOY"])
    X_all = df.drop(columns=["Date", "HAB", "Year", "Day", "Month", "YDOY"])
    years = df["Date"].dt.year.values

    PREPROCESSING_TAG = (
        "native CHL NaNs + native Season & cluster + YDOY sin/cos + "
        "drop Year/Month/Day + drop +" "New evaluation metrics + " "new_param_grid_values"
    )

    param_grid = list(
        ParameterGrid({
            "max_iter": [150, 250, 350],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_leaf_nodes": [15, 31, 63],
            "max_depth": [3, 5, 7],
            "min_samples_leaf": [20, 40, 60],
            "l2_regularization": [0.0, 10.0, 50.0],
            "max_features": [0.5, 0.7, 0.9],
            "max_bins": [32, 64, 128],
            "class_weight": ['balanced']
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

    print("\n" + "="*50)
    print("Cross Validation results")
    print(f"Best hyperparameters: {best_params}")
    print(f"Average AUC training: {best_details['mean_train']:.4f}")

    mean_ap = np.mean([m['ap'] for m in best_details["val_metrics"].values()])
    mean_brier = np.mean([m['brier'] for m in best_details["val_metrics"].values()])

    print(
        f"Average validation AUC:   {mean_val_auc:.4f} (+/- {np.std([m['auc'] for m in best_details['val_metrics'].values()]):.4f})\n"
        f"Average validation AP:    {mean_ap:.4f} (+/- {np.std([m['ap'] for m in best_details['val_metrics'].values()]):.4f})\n"
        f"Average validation Brier: {mean_brier:.4f} (+/- {np.std([m['brier'] for m in best_details['val_metrics'].values()]):.4f})"
    )

    print("\nDetails by validation year:")
    for yr, metrics in best_details["val_metrics"].items():
        train_auc = best_details["train_scores"][yr]
        print(f"  Year {yr} | Train AUC: {train_auc:.4f} -> Val AUC: {metrics['auc']:.4f} | AP: {metrics['ap']:.4f} | Brier: {metrics['brier']:.4f}")

    # --- TESTE 2020 ---
    test_year = 2020
    X_tr_test, Y_tr_test, X_te, Y_te = prepare_split(
        X_all, Y_all, np.where(years < test_year)[0], np.where(years == test_year)[0]
    )

    final_test_model = HistGradientBoostingClassifier(
        **best_params, categorical_features=CAT_COLS, random_state=42
    )
    final_test_model.fit(X_tr_test, Y_tr_test)

    test_prob = final_test_model.predict_proba(X_te)[:, 1]
    print(f"\n====== AVALIAÇÃO EXTERNA (TESTE {test_year}) ======")
    print(f"AUC Teste {test_year}:   {roc_auc_score(Y_te, test_prob):.4f}")
    print(f"AP Teste {test_year}:    {average_precision_score(Y_te, test_prob):.4f}")
    print(f"Brier Teste {test_year}: {brier_score_loss(Y_te, test_prob):.4f}\n")

    log_experiment(
        model_type="HistGradientBoostingClassifier",
        preprocessing_tag=PREPROCESSING_TAG,
        val_auc=mean_val_auc,
        best_params=best_params,
    )

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

#python train_boost.py --data ../../data/classification_task1_train.csv --output ../best_model/model_boost.pkl 