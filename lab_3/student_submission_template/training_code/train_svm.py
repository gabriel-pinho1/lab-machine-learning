from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from aux import log_experiment, month_sine, read_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    df = read_table(args.data)
    df = month_sine(df)

    Y_cv = df["HAB"]
    # Keep unscaled features here; scale inside each fold to prevent leakage
    X_cv = df.drop(
        columns=["Date", "HAB", "Year", "Day", "CHL_uncertainty", "CHL", "Season"]
    )

    PREPROCESSING_TAG = (
        "drop CHL + drop CHL_uncertainty + drop Year + drop Day + drop Season + replace Month"
    )

    # Build temporal CV splits
    cv_years = df["Date"].dt.year.values
    validation_years = [2016, 2018, 2019]  # 2017 has no true positives -> AUC undefined
    test_years = [2017, 2019, 2020]
    custom_cv_splits = []
    for val_year, test_year in zip(validation_years, test_years):
        train_idx = np.where(cv_years < val_year)[0]
        val_idx = np.where(cv_years == val_year)[0]
        test_idx = np.where(cv_years == test_year)[0]
        custom_cv_splits.append((train_idx, val_idx, test_idx))

    C_vec = [0.1, 1, 10, 100]
    gamma_vec = ["scale", 0.001, 0.01, 0.1, 1]
    kernel_vec = ["rbf", "linear"]
    class_weight_vec = ["balanced", None]
    random_state = 42

    mean_best = -1.0
    C_best = None
    gamma_best = None
    kernel_best = None
    class_weight_best = None

    # SVM hyperparameter grid search
    for C in C_vec:
        for kernel in kernel_vec:
            active_gammas = ["scale"] if kernel == "linear" else gamma_vec
            for gamma in active_gammas:
                for class_weight in class_weight_vec:
                    fold_auc = []  # Reset for every hyperparameter combination

                    for train_index, val_index, _ in custom_cv_splits:
                        X_train_fold = X_cv.iloc[train_index].copy()
                        X_val_fold = X_cv.iloc[val_index].copy()
                        Y_train_fold = Y_cv.iloc[train_index].copy()
                        Y_val_fold = Y_cv.iloc[val_index].copy()

                        # Fit scaler strictly on training fold, transform validation fold
                        scaler = StandardScaler()
                        X_train_scaled = pd.DataFrame(
                            scaler.fit_transform(X_train_fold),
                            columns=X_train_fold.columns,
                        )
                        X_val_scaled = pd.DataFrame(
                            scaler.transform(X_val_fold),
                            columns=X_val_fold.columns,
                        )

                        svm_model = SVC(
                            kernel=kernel,
                            gamma=gamma,
                            class_weight=class_weight,
                            C=C,
                            random_state=random_state,
                            probability=True,
                        )

                        svm_model.fit(X_train_scaled, Y_train_fold)
                        y_prob = svm_model.predict_proba(X_val_scaled)[:, 1]
                        auc = roc_auc_score(Y_val_fold, y_prob)
                        if not np.isnan(auc):
                            fold_auc.append(auc)

                    mean_score = float(np.mean(fold_auc))

                    if mean_score > mean_best:
                        mean_best = mean_score
                        C_best = C
                        gamma_best = gamma
                        kernel_best = kernel
                        class_weight_best = class_weight

    print("=" * 40)
    print("     BEST HYPERPARAMETER SEARCH RESULTS     ")
    print("=" * 40)
    print(f"Best Validation ROC-AUC Score: {mean_best:.4f}\n")
    print("Optimal Hyperparameters:")
    print(f"C                  : {C_best}")
    print(f"gamma              : {gamma_best}")
    print(f"kernel             : {kernel_best}")
    print(f"class_weight       : {class_weight_best}")
    print("=" * 40)

    best_hyperparams = {
        "kernel": kernel_best,
        "gamma": gamma_best,
        "class_weight": class_weight_best,
        "C": C_best,
        "probability": True,
        "random_state": random_state,
    }

    # Fit final scaler and final SVM on 100% of training data
    final_scaler = StandardScaler()
    X_cv_scaled = pd.DataFrame(
        final_scaler.fit_transform(X_cv),
        columns=X_cv.columns,
    )

    final_model = SVC(
        kernel=kernel_best,
        gamma=gamma_best,
        class_weight=class_weight_best,
        C=C_best,
        probability=True,
        random_state=random_state,
    )
    final_model.fit(X_cv_scaled, Y_cv)

    log_experiment(
        model_type="SVM",
        preprocessing_tag=PREPROCESSING_TAG,
        val_auc=mean_best,
        best_params=best_hyperparams,
    )

    # Save both the fitted SVC and the fitted StandardScaler
    model_artifact = {
        "model": final_model,
        "scaler": final_scaler,
        "cols": list(X_cv.columns),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(model_artifact, f)


if __name__ == "__main__":
    main()


#python train_svm.py --data ../../data/classification_task1_train.csv --output ../best_model/model_svm.pkl