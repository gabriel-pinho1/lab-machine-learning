import pickle
from pathlib import Path
import numpy as np
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

# 1. Define your locked-in winning models from Stage 1
# best_rf = RandomForestClassifier(...)
# best_svm = Pipeline([("scaler", StandardScaler()), ("svc", SVC(probability=True, ...))])

rf_fold_aucs, svm_fold_aucs, blend_fold_aucs = [], [], []

for train_index, val_index, _ in custom_cv_splits:
    X_tr_rf, X_va_rf = X_rf.iloc[train_index], X_rf.iloc[val_index]
    X_tr_svm, X_va_svm = X_svm.iloc[train_index], X_svm.iloc[val_index]
    Y_tr, Y_va = Y_cv.iloc[train_index], Y_cv.iloc[val_index]

    rf_fold = clone(best_rf).fit(X_tr_rf, Y_tr)
    svm_fold = clone(best_svm).fit(X_tr_svm, Y_tr)

    p_rf = rf_fold.predict_proba(X_va_rf)[:, 1]
    p_svm = svm_fold.predict_proba(X_va_svm)[:, 1]
    p_blend = 0.5 * p_rf + 0.5 * p_svm

    rf_fold_aucs.append(roc_auc_score(Y_va, p_rf))
    svm_fold_aucs.append(roc_auc_score(Y_va, p_svm))
    blend_fold_aucs.append(roc_auc_score(Y_va, p_blend))

print(f"RF 3-Fold Mean AUC:    {np.mean(rf_fold_aucs):.4f} (std: {np.std(rf_fold_aucs):.4f})")
print(f"SVM 3-Fold Mean AUC:   {np.mean(svm_fold_aucs):.4f} (std: {np.std(svm_fold_aucs):.4f})")
print(f"Blend 3-Fold Mean AUC: {np.mean(blend_fold_aucs):.4f} (std: {np.std(blend_fold_aucs):.4f})")


final_rf = clone(best_rf).fit(X_rf, Y_cv)
final_svm = clone(best_svm).fit(X_svm, Y_cv)

artifact = {
    "rf_model": final_rf,
    "svm_model": final_svm,
    "rf_cols": list(X_rf.columns),
    "svm_cols": list(X_svm.columns),
}

out_path = Path(__file__).resolve().parent.parent / "best_model" / "model.pkl"
with open(out_path, "wb") as f:
    pickle.dump(artifact, f)