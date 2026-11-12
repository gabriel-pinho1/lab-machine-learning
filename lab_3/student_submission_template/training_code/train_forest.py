
from __future__ import annotations
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import gc
import pandas as pd
from scipy.stats import zscore

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from aux import read_table, log_experiment, month_sine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    
    args = parser.parse_args()

    df = read_table(args.data)
    # Apply mapping
    season_map = {'winter': 0, 'spring': 1, 'summer': 2, 'fall': 3}
    df['Season'] = df['Season'].astype(str).str.lower().map(season_map)
   
    
    
    Y_cv = df["HAB"]
    X_cv = df.drop(columns = ["Date", "HAB","Year","Day", "CHL_uncertainty", "CHL"])

    PREPROCESSING_TAG = "drop CHL + drop CHL_uncertainty + drop Year + drop Day"
    # TAG structure e.g. "drop Variable + Add Variable + what was done to andle missing data etc"

    # 2. Build CV splits dynamically using dates in X_cv
    cv_dates = df["Date"]
    cv_years = cv_dates.dt.year.values

    validation_years = [2016,2018,2019] # 2017 has no true positives therefore AUC is not defined
    test_years = [2017,2019,2020]
    custom_cv_splits = []
    for val_year, test_year in zip(validation_years, test_years):
        train_idx = np.where(cv_years < val_year)[0]

        val_idx = np.where(cv_years == val_year)[0]
        test_idx = np.where(cv_years == test_year)[0]
        custom_cv_splits.append((train_idx, val_idx, test_idx))
    
    
    class_weight_vec = ["balanced", None]
    tree_depth_vec = [3,5,10,None]
    min_sample_split_vec = [2, 5]
    min_samples_leaf_vec =  [1, 3]
    
    random_state = 42

    mean_best = -1.0
    criterion_best = None
    leaf_best = None
    split_best = None
    depth_best = None
    weight_best = None

    #forest hyperparameter cv
    for weight in class_weight_vec:

        for sample_leaf in min_samples_leaf_vec:
            for sample_split in min_sample_split_vec:
                for depth in tree_depth_vec:
                    fold_auc = []
                    for train_index, val_index, test_idx in custom_cv_splits:
                        X_train_fold, X_val_fold = X_cv.iloc[train_index].copy(), X_cv.iloc[val_index].copy()
                        
                        Y_train_fold, Y_val_fold = Y_cv.iloc[train_index].copy(), Y_cv.iloc[val_index].copy()
                        

                        forest_model = clf = RandomForestClassifier(
                                            n_estimators= 100,
                                            class_weight= weight,
                                            criterion= "gini",         # 'gini' ou 'entropy'
                                            max_depth=depth,              # Limita a profundidade máxima a 5 níveis
                                            min_samples_split=sample_split,     # Um nó precisa de ter pelo menos 10 amostras para ser dividido
                                            min_samples_leaf=sample_leaf,       # Cada folha deve ter pelo menos 4 amostras
                                            max_features='sqrt',      # Utiliza sqrt(n_features) em cada divisão
                                            n_jobs=-1,                # Usa mais cores do CPU
                                            random_state=random_state           # Garante a reprodução dos mesmos resultados
                                        )
                        forest_model.fit(X_train_fold, Y_train_fold)
                        y_prob = forest_model.predict_proba(X_val_fold)[:, 1]
                        auc = roc_auc_score(Y_val_fold, y_prob)
                        if not np.isnan(auc):
                            fold_auc.append(auc)
                
                    mean_score = np.mean(fold_auc)

                    if mean_score > mean_best:
                        mean_best = mean_score
                        depth_best = depth
                        criterion_best = "gini"
                        leaf_best = sample_leaf
                        split_best = sample_split
                        weight_best = weight

    #posterior forest size iteration




    print("=" * 40)
    print("     BEST HYPERPARAMETER SEARCH RESULTS     ")
    print("=" * 40)
    print(f"Best Validation mean ROC-AUC Score: {mean_best:.4f}\n")
    print("Optimal Hyperparameters:")
    print(f"   criterion:         '{criterion_best}'")
    print(f"   class_weight:      {weight_best}")
    print(f"   max_depth:         {depth_best}")
    print(f"   min_samples_leaf:  {leaf_best}")
    print(f"   min_samples_split: {split_best}")
    print("=" * 40)

 



    # Posterior forest size iteration across ALL validation years
    tree_steps = range(10, 701, 10)
    fold_curves = []

    for train_index, val_index, _ in custom_cv_splits:
        X_tr, X_va = X_cv.iloc[train_index], X_cv.iloc[val_index]
        Y_tr, Y_va = Y_cv.iloc[train_index], Y_cv.iloc[val_index]

        big_model = RandomForestClassifier(
            n_estimators=700,
            n_jobs=-1,
            class_weight=weight_best,
            criterion=criterion_best,
            max_depth=depth_best,
            min_samples_split=split_best,
            min_samples_leaf=leaf_best,
            max_features="sqrt",
            random_state=random_state,
        )
        big_model.fit(X_tr, Y_tr)

        all_tree_probs = np.array(
            [tree.predict_proba(X_va.values)[:, 1] for tree in big_model.estimators_]
        )

        fold_scores = [
            roc_auc_score(Y_va, np.mean(all_tree_probs[:k], axis=0))
            for k in tree_steps
        ]
        fold_curves.append(fold_scores)

    # Mean ROC-AUC curve across 2016, 2018, and 2019
    mean_auc_curve = np.mean(fold_curves, axis=0)
    auc_700 = mean_auc_curve[-1]
    window = 6  # Current step k + the next 5 tested steps
    threshold = 0.995 * auc_700
    diminished_returns = tree_steps[-1]  # Fallback to 700 if plateau isn't sustained

    for i in range(len(tree_steps) - window + 1):
        if all(score >= threshold for score in mean_auc_curve[i : i + window]):
            diminished_returns = tree_steps[i]
            break

    print("=" * 40)
    print(f" Full 700-Tree Mean ROC-AUC:       {auc_700:.4f}")
    print(f" Diminishing Returns Tree Cutoff:  {diminished_returns}")
    print("=" * 40)
        



    final_model = RandomForestClassifier(   # model with a lot of trees to determine diminishing returns
    n_estimators= diminished_returns,
    n_jobs = -1,
    class_weight = weight_best,
    criterion=criterion_best,
    max_depth=depth_best,
    min_samples_split=split_best,
    min_samples_leaf=leaf_best,
    max_features="sqrt",
    random_state=random_state,
        )
    final_model.fit(X_cv, Y_cv)


    model_artifact = {
    "model": final_model
    } 

    with open(args.output, "wb") as f:
        pickle.dump(model_artifact, f)

        best_hyperparams = {
        "n_estimators" : diminished_returns,
        "class_weight" : weight_best,
        "criterion":criterion_best,
        "max_depth":depth_best,
        "min_samples_split":split_best,
        "min_samples_leaf": leaf_best,
        "max_features":"sqrt",
        "random_state" :random_state,
        }

    log_experiment(
            
            model_type="RandomForestClassifier",
            preprocessing_tag=PREPROCESSING_TAG,
            val_auc=mean_best,
            best_params=best_hyperparams,
        
        )



if __name__ == "__main__":
    main()


#python train_forest.py --data ../../data/classification_task1_train.csv --output ../best_model/model_forest.pkl