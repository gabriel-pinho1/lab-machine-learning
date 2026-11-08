
from __future__ import annotations
import pickle
import numpy as np
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
from aux import read_table


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
    X_cv = df.drop(columns = ["Date", "HAB"])


    

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
    scaler = StandardScaler()
    tree_depth_vec = [3,5,10,None]
    min_sample_split_vec = [2, 5, 10]
    min_samples_leaf_vec =  [1, 2, 4]
    criterion_vec = ['gini', 'entropy']

    mean_best = -1.0
    criterion_best = None
    leaf_best = None
    split_best = None
    depth_best = None

    #tree hyperparameter cv
    for sample_leaf in min_samples_leaf_vec:
        for sample_split in min_sample_split_vec:
            for criterion in criterion_vec:
                for depth in tree_depth_vec:
                    fold_auc = []
                    for train_index, val_index, test_index in custom_cv_splits:
                        X_train_fold, X_val_fold = X_cv.iloc[train_index].copy(), X_cv.iloc[val_index].copy()
                        
                        Y_train_fold, Y_val_fold = Y_cv.iloc[train_index].copy(), Y_cv.iloc[val_index].copy()
                        

                        tree_model = clf = DecisionTreeClassifier(
                                            criterion= criterion,         # 'gini' ou 'entropy'
                                            max_depth=depth,              # Limita a profundidade máxima a 5 níveis
                                            min_samples_split=sample_split,     # Um nó precisa de ter pelo menos 10 amostras para ser dividido
                                            min_samples_leaf=sample_leaf,       # Cada folha deve ter pelo menos 4 amostras
                                            max_features='sqrt',      # Utiliza sqrt(n_features) em cada divisão
                                            random_state=42           # Garante a reprodução dos mesmos resultados
                                        )
                        tree_model.fit(X_train_fold, Y_train_fold)
                        y_prob = tree_model.predict_proba(X_val_fold)[:, 1]
                        auc = roc_auc_score(Y_val_fold, y_prob)
                        if not np.isnan(auc):
                            fold_auc.append(auc)
                   
                    mean_score = np.mean(fold_auc)

                    if mean_score > mean_best:
                        mean_best = mean_score
                        depth_best = depth
                        criterion_best = criterion
                        leaf_best = sample_leaf
                        split_best = sample_split

    print("=" * 40)
    print("     BEST HYPERPARAMETER SEARCH RESULTS     ")
    print("=" * 40)
    print(f"Best Validation ROC-AUC Score: {mean_best:.4f}\n")
    print("Optimal Hyperparameters:")
    print(f"   criterion:         '{criterion_best}'")
    print(f"   max_depth:         {depth_best}")
    print(f"   min_samples_leaf:  {leaf_best}")
    print(f"   min_samples_split: {split_best}")
    print("=" * 40)

 



    '''scaler = StandardScaler()
                        X_train_fold_scaled = pd.DataFrame(scaler.fit_transform(X_train_fold), columns=X_train_fold.columns)
                        # scalling based on the mean of the test data p
                        X_val_fold_scaled = pd.DataFrame(scaler.transform(X_val_fold), columns=X_val_fold.columns)'''

















    final_model = 0


    model_artifact = {
    "model": final_model
    } 

    with open(args.output, "wb") as f:
        pickle.dump(model_artifact, f)






if __name__ == "__main__":
    main()


#python train.py --data ../../data/classification_task1_train.csv --output ../best_model/model.pkl