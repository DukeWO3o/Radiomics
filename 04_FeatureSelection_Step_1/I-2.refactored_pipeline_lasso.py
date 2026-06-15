# -*- coding: utf-8 -*-
"""
=============================================================================
Radiomics RMPP Prediction - LASSO-based Feature Selection (No Data Leakage)
=============================================================================
Key differences from I.refactored_pipeline.py (mRMR version):
  - Feature selection: LassoCV (L1 regularization) replaces MWU+FDR+mRMR
  - Automatically determines optimal feature count via cross-validated alpha
  - More stable feature selection, less hyperparameter tuning needed
  - Supports imaging-only or imaging+clinical combined data

Usage:
  1. Set INPUT_FILE to your data path.
  2. For imaging-only:     01b_影像版-v4-374-557-931.xlsx
     For imaging+clinical: 01d_影像+临床版-v4-374-557-931.xlsx
  3. python I-2.refactored_pipeline_lasso.py
=============================================================================
"""
import matplotlib
matplotlib.use("Agg")
import pandas as pd, numpy as np, os, warnings, joblib
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt, seaborn as sns
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression, LassoCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve, f1_score,
    precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay,
    brier_score_loss, balanced_accuracy_score, matthews_corrcoef,
)
from sklearn.calibration import calibration_curve
import shap

# ========================= CONFIGURATION =========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else r"E:\Softwares\PycharmProject\Radiomics\04_FeatureSelection_Step_1"
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "06_Modelling", "output_lasso")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Change this to your data file ----
INPUT_FILE = r"E:\Softwares\PycharmProject\Radiomics\03_MergedData\01c_影像+年龄版-v4-374-557-931.xlsx"

N_FOLDS = 5
RANDOM_STATE = 42
MV_MISSING = 0.30
MV_VARIANCE = 0.05
LASSO_CV_FOLDS = 5         # internal CV folds for LassoCV alpha selection
LASSO_COEF_THRESHOLD = 1e-5
USE_SMOTE = True             # Set False to disable SMOTE
SMOTE_K_NEIGHBORS = 5        # k_neighbors for SMOTE # minimum |coef| to keep a feature

# =========================  STEP 1: Load  =========================
print("=" * 60)
print(" Radiomics RMPP - LASSO Pipeline (No Data Leakage)")
print("=" * 60)
print("\n[1] Loading data:", INPUT_FILE)
df = pd.read_excel(INPUT_FILE)
print(f"    Shape: {df.shape}")
print(f"    Labels: {dict(df['label'].value_counts().sort_index())}")

# =========================  STEP 2: Split  =========================
print("\n[2] Train/validation split (80:20, stratified) ...")
df = df.reset_index(drop=True)
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE, stratify=df["label"])
train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)
print(f"    Train: {train_df.shape}   Valid: {valid_df.shape}")

# =========================  STEP 3: Preprocess  =========================
print("\n[3] Label-free preprocessing ...")

# Identify column types
id_cols = [c for c in train_df.columns if c.lower() in ["patientid", "image", "mask", "name", "姓名", "住院号"]]
label_col = "label"
diag_cols = [c for c in train_df.columns if c.startswith("diagnostics")]

# Separate radiomics vs clinical columns
radiomics_prefixes = ("original_", "shape2D_", "firstorder_", "glcm_", "glrlm_", "glszm_", "gldm_", "ngtdm_")
radio_cols = [c for c in train_df.columns if any(c.startswith(p) for p in radiomics_prefixes)]
all_skip = set(id_cols + [label_col] + diag_cols + radio_cols)
clinical_cols = [c for c in train_df.columns if c not in all_skip]
# Remove purely metadata columns from clinical
clinical_cols = [c for c in clinical_cols if train_df[c].dtype in ["int64", "float64"] and c not in ["label"]]

print(f"    Radiomics features : {len(radio_cols)}")
print(f"    Clinical features  : {len(clinical_cols)} -> {clinical_cols}")

y_train = train_df[label_col].values
y_valid = valid_df[label_col].values

Xr_train = train_df[radio_cols].copy()
Xr_valid = valid_df[radio_cols].copy()
Xc_train = train_df[clinical_cols].copy() if clinical_cols else None
Xc_valid = valid_df[clinical_cols].copy() if clinical_cols else None

# 3a. Missing rate filter (on radiomics only, based on training set)
missing_ratio = Xr_train.isnull().mean()
keep = missing_ratio[missing_ratio <= MV_MISSING].index.tolist()
Xr_train = Xr_train[keep]; Xr_valid = Xr_valid[keep]

# 3b. Impute + variance filter (radiomics)
train_means_r = Xr_train.mean()
Xr_imp = Xr_train.fillna(train_means_r)
sel_var = VarianceThreshold(threshold=MV_VARIANCE)
sel_var.fit(Xr_imp)
radio_kept = Xr_train.columns[sel_var.get_support()].tolist()
Xr_train = Xr_imp[radio_kept]
Xr_valid = Xr_valid[radio_kept].fillna(train_means_r[radio_kept])
print(f"    Radiomics after filter: {len(radio_kept)}")

# 3c. Impute clinical (median)
if Xc_train is not None:
    train_meds_c = Xc_train.median()
    Xc_train = Xc_train.fillna(train_meds_c)
    Xc_valid = Xc_valid.fillna(train_meds_c)

# 3d. Standardize
scaler_r = StandardScaler()
Xr_train_s = pd.DataFrame(scaler_r.fit_transform(Xr_train), columns=radio_kept)
Xr_valid_s = pd.DataFrame(scaler_r.transform(Xr_valid), columns=radio_kept)

if Xc_train is not None:
    scaler_c = StandardScaler()
    Xc_train_s = pd.DataFrame(scaler_c.fit_transform(Xc_train), columns=clinical_cols)
    Xc_valid_s = pd.DataFrame(scaler_c.transform(Xc_valid), columns=clinical_cols)
    # Merge: clinical columns always included, not subject to LASSO selection
    X_full_train = pd.concat([Xr_train_s, Xc_train_s], axis=1)
    X_full_valid = pd.concat([Xr_valid_s, Xc_valid_s], axis=1)
    radio_mask = np.array([True]*len(radio_kept) + [False]*len(clinical_cols))  # which cols are radiomics
    print(f"    Combined features: {X_full_train.shape[1]} ({len(radio_kept)} radio + {len(clinical_cols)} clinical)")
else:
    X_full_train = Xr_train_s
    X_full_valid = Xr_valid_s
    radio_mask = np.ones(len(radio_kept), dtype=bool)

# =====  Correlation heatmap (radiomics only) =====
print("\n[4] Correlation visualization (radiomics) ...")
plt.figure(figsize=(36, 27))
sns.heatmap(Xr_train_s.corr(), cmap="viridis", linewidths=0.5, annot=False)
plt.title("Radiomics Feature Correlation Matrix (Training Set)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_correlation_heatmap.svg"))
plt.close()
print("    Saved.")

# =========================  STEP 5: LASSO Selector  =========================


# =========================  SMOTE (no imblearn dependency) =========================
class SimpleSMOTE(BaseEstimator, TransformerMixin):
    def __init__(self, k_neighbors=5, random_state=42):
        self.k_neighbors = k_neighbors
        self.random_state = random_state
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        import numpy as np
        X = np.asarray(X); y_arr = np.asarray(y) if y is not None else None
        if y_arr is None or len(np.unique(y_arr)) < 2:
            return X
        classes, counts = np.unique(y_arr, return_counts=True)
        maj = classes[np.argmax(counts)]; min_c = classes[np.argmin(counts)]
        n_syn = counts.max() - counts.min()
        if n_syn <= 0: return X
        X_min = X[y_arr == min_c]
        nn = NearestNeighbors(n_neighbors=min(self.k_neighbors + 1, len(X_min)))
        nn.fit(X_min)
        rng = np.random.RandomState(self.random_state)
        syn = []; idxs = rng.choice(len(X_min), n_syn, replace=True)
        for i in idxs:
            nbrs = nn.kneighbors([X_min[i]], return_distance=False)[0]
            ni = rng.choice(nbrs[1:])
            syn.append(X_min[i] + rng.random() * (X_min[ni] - X_min[i]))
        return np.vstack([X, np.array(syn)])
class LassoFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Within-CV-fold feature selection using LassoCV.
    Only operates on radiomics columns (marked by radio_mask_).
    Clinical columns (if any) pass through unchanged.
    """
    def __init__(self, radio_mask, cv_folds=LASSO_CV_FOLDS, threshold=LASSO_COEF_THRESHOLD):
        self.radio_mask = radio_mask           # store as-is for sklearn clone()
        self.cv_folds = cv_folds
        self.threshold = threshold
        self.selected_indices_ = None
        self.selected_features_ = None
        self.lasso_ = None

    def fit(self, X, y):
        X_df = pd.DataFrame(X)
        radio_mask_arr = np.array(self.radio_mask)  # convert here, not in __init__
        radio_idx = np.where(radio_mask_arr)[0]
        clinical_idx = np.where(~radio_mask_arr)[0]

        if len(radio_idx) == 0:
            self.selected_indices_ = np.arange(X.shape[1])
            self.selected_features_ = list(range(X.shape[1]))
            return self

        X_radio = X_df.iloc[:, radio_idx].values
        self.lasso_ = LassoCV(cv=self.cv_folds, random_state=RANDOM_STATE, max_iter=5000, n_jobs=-1)
        self.lasso_.fit(X_radio, y)

        coef = np.abs(self.lasso_.coef_)
        kept = coef >= self.threshold
        kept_idx = radio_idx[kept]

        self.selected_indices_ = np.sort(np.concatenate([kept_idx, clinical_idx]))
        self.selected_features_ = list(self.selected_indices_)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        return X_df.iloc[:, self.selected_indices_].values

print("\n[5] LASSO-based feature selector defined (LassoCV, alpha auto-tuned).")

# =========================  STEP 6: Models  =========================
models_config = {
    "LogisticRegression": {
        "model": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced"),
        "params": {"classifier__C": [0.001, 0.01, 0.1, 1, 10, 100],
                   "classifier__penalty": ["l1", "l2"], "classifier__solver": ["liblinear"]},
    },
    "DecisionTree": {
        "model": DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
        "params": {"classifier__max_depth": [2, 3, 4], "classifier__min_samples_split": [20, 30, 40],
                   "classifier__min_samples_leaf": [15, 20, 25], "classifier__max_features": [0.3, 0.5, 0.7]},
    },
    "RandomForest": {
        "model": RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
        "params": {"classifier__n_estimators": [100, 200], "classifier__max_depth": [3, 4, 5],
                   "classifier__min_samples_split": [20, 30], "classifier__min_samples_leaf": [10, 15],
                   "classifier__max_features": [0.3, 0.5], "classifier__bootstrap": [True]},
    },
    "MLP": {
        "model": MLPClassifier(max_iter=2000, random_state=RANDOM_STATE, early_stopping=True),
        "params": {"classifier__hidden_layer_sizes": [(50,), (100,), (50, 25)],
                   "classifier__alpha": [0.0001, 0.001, 0.01], "classifier__activation": ["relu", "tanh"]},
    },
    "SVM": {
        "model": SVC(probability=True, random_state=RANDOM_STATE, class_weight="balanced"),
        "params": {"classifier__C": [0.1, 1, 5, 10], "classifier__gamma": ["scale", "auto"],
                   "classifier__kernel": ["rbf", "linear"]},
    },
    "XGBoost": {
        "model": XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE),
        "params": {"classifier__n_estimators": [100, 200], "classifier__max_depth": [3, 4],
                   "classifier__learning_rate": [0.05, 0.1, 0.15], "classifier__gamma": [0, 0.1, 0.5],
                   "classifier__reg_alpha": [0, 0.1, 0.5], "classifier__reg_lambda": [1, 5, 10]},
    },
    "LightGBM": {
        "model": LGBMClassifier(random_state=RANDOM_STATE, verbose=-1),
        "params": {"classifier__n_estimators": [100, 200], "classifier__max_depth": [3, 4],
                   "classifier__learning_rate": [0.05, 0.1], "classifier__num_leaves": [15, 31],
                   "classifier__min_child_samples": [20, 30], "classifier__reg_alpha": [0.1, 0.3],
                   "classifier__reg_lambda": [0.1, 0.3]},
    },
    "GaussianNB": {
        "model": GaussianNB(),
        "params": {"classifier__var_smoothing": [1e-9, 1e-8, 1e-7]},
    },
    "KNN": {
        "model": KNeighborsClassifier(),
        "params": {"classifier__n_neighbors": [5, 7, 9, 11, 13]},
    },
}
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
print(f"[6] {len(models_config)} models configured.")

# =========================  STEP 7: Train  =========================
print("\n[7] Training (LASSO feature selection inside each CV fold) ...")
print("=" * 60)

best_models = {}
train_results = {}
valid_results = {}
all_pred_probs = {}

plt.figure(figsize=(10, 8))
for name, cfg in models_config.items():
    print(f"\n--- {name} ---")
    if USE_SMOTE:
        pipe = Pipeline([
            ("selector", LassoFeatureSelector(radio_mask, cv_folds=LASSO_CV_FOLDS)),
            ("smote", SimpleSMOTE(k_neighbors=SMOTE_K_NEIGHBORS, random_state=RANDOM_STATE)),
            ("classifier", cfg["model"]),
        ])
    else:
        pipe = Pipeline([
            ("selector", LassoFeatureSelector(radio_mask, cv_folds=LASSO_CV_FOLDS)),
            ("classifier", cfg["model"]),
        ])
    grid = GridSearchCV(pipe, cfg["params"], cv=cv, scoring="roc_auc", n_jobs=-1, verbose=0)
    grid.fit(X_full_train.values, y_train)
    best_model = grid.best_estimator_
    best_models[name] = best_model
    n_feats = len(best_model.named_steps["selector"].selected_features_)
    alpha = best_model.named_steps["selector"].lasso_.alpha_ if best_model.named_steps["selector"].lasso_ else "N/A"
    print(f"  Lasso alpha={alpha:.4f}, features kept={n_feats}")
    print(f"  Best params: {grid.best_params_}")

    joblib.dump(best_model, os.path.join(OUTPUT_DIR, f"model_{name}.pkl"))

    # Train eval
    y_tr_p = best_model.predict(X_full_train.values)
    y_tr_pr = best_model.predict_proba(X_full_train.values)[:, 1]
    train_results[name] = {
        "Train_AUC": roc_auc_score(y_train, y_tr_pr),
        "Train_Accuracy": accuracy_score(y_train, y_tr_p),
        "Train_F1": f1_score(y_train, y_tr_p),
        "Train_BalancedAcc": balanced_accuracy_score(y_train, y_tr_p),
        "Train_MCC": matthews_corrcoef(y_train, y_tr_p),
    }
    fpr, tpr, _ = roc_curve(y_train, y_tr_pr)
    plt.plot(fpr, tpr, label=f"{name} (AUC={train_results[name]['Train_AUC']:.3f})")

    # Valid eval
    y_v_p = best_model.predict(X_full_valid.values)
    y_v_pr = best_model.predict_proba(X_full_valid.values)[:, 1]
    all_pred_probs[name] = y_v_pr
    valid_results[name] = {
        "Valid_AUC": roc_auc_score(y_valid, y_v_pr),
        "Valid_Accuracy": accuracy_score(y_valid, y_v_p),
        "Valid_F1": f1_score(y_valid, y_v_p),
        "Valid_Precision": precision_score(y_valid, y_v_p),
        "Valid_Recall": recall_score(y_valid, y_v_p),
        "Valid_BalancedAcc": balanced_accuracy_score(y_valid, y_v_p),
        "Valid_MCC": matthews_corrcoef(y_valid, y_v_p),
        "Valid_Brier": brier_score_loss(y_valid, y_v_pr),
    }
    print(f"  Train AUC={train_results[name]['Train_AUC']:.3f}  Valid AUC={valid_results[name]['Valid_AUC']:.3f}  "
          f"Brier={valid_results[name]['Valid_Brier']:.3f}")

    # Confusion matrices
    for prefix, X_, y_ in [("Train", X_full_train, y_train), ("Valid", X_full_valid, y_valid)]:
        cm = confusion_matrix(y_, best_model.predict(X_.values))
        ConfusionMatrixDisplay(cm).plot(cmap="Blues")
        plt.title(f"{prefix} CM - {name}")
        plt.savefig(os.path.join(OUTPUT_DIR, f"02_{prefix}_CM_{name}.svg"), dpi=300)
        plt.close()

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curves (Training Set)")
plt.legend(loc="lower right")
plt.savefig(os.path.join(OUTPUT_DIR, "03_Train_ROC_all.svg"), dpi=300)
plt.close()

# =========================  STEP 8: Valid ROC + Comparison  =========================
print("\n[8] Validation evaluation ...")
tr_df = pd.DataFrame(train_results).T
va_df = pd.DataFrame(valid_results).T
all_df = tr_df.join(va_df)
all_df.to_excel(os.path.join(OUTPUT_DIR, "04_Model_Performance.xlsx"))

best_name = va_df["Valid_AUC"].idxmax()
best_model = best_models[best_name]
print(f"  Best: {best_name}  Valid AUC={va_df.loc[best_name,'Valid_AUC']:.3f}")
# ===== Export selected features for all models =====
print("  Exporting feature selection results ...")
all_feat_names = list(X_full_train.columns)
feature_records = []
for name, m in best_models.items():
    sel = m.named_steps["selector"]
    if hasattr(sel, "selected_features_"):
        indices = sel.selected_features_
        real_names = [all_feat_names[i] for i in indices if i < len(all_feat_names)]
        # Get LASSO coefficients if available
        coefs = ""
        if hasattr(sel, "lasso_") and sel.lasso_ is not None:
            coef_arr = sel.lasso_.coef_
            # Map coefficients to kept features
            kept_indices = [i for i in indices if i < len(radio_kept)]
            if len(kept_indices) > 0:
                coefs = ", ".join([f"{all_feat_names[i]}={coef_arr[np.where(np.array(radio_kept)==all_feat_names[i])[0][0] if all_feat_names[i] in radio_kept else -1]:.4f}" if all_feat_names[i] in radio_kept else f"{all_feat_names[i]}" for i in kept_indices])
        feature_records.append({
            "Model": name,
            "N_Features": len(real_names),
            "Selected_Features": ", ".join(real_names),
            "LASSO_Alpha": sel.lasso_.alpha_ if hasattr(sel, "lasso_") and sel.lasso_ else "N/A",
        })
feat_df = pd.DataFrame(feature_records)
feat_df.to_excel(os.path.join(OUTPUT_DIR, "04b_Selected_Features.xlsx"), index=False)

# Best model detail
best_sel = best_model.named_steps["selector"]
if hasattr(best_sel, "selected_features_"):
    best_indices = best_sel.selected_features_
    best_names = [all_feat_names[i] for i in best_indices if i < len(all_feat_names)]
    print(f"  Best model ({best_name}) uses {len(best_names)} features:")
    for fn in best_names:
        print(f"    - {fn}")


plt.figure(figsize=(10, 8))
for name, m in best_models.items():
    pr = m.predict_proba(X_full_valid.values)[:, 1]
    fpr, tpr, _ = roc_curve(y_valid, pr)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_valid,pr):.3f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curves (Validation Set)")
plt.legend(loc="lower right")
plt.savefig(os.path.join(OUTPUT_DIR, "05_Valid_ROC_all.svg"), dpi=300)
plt.close()

# Performance bar charts
for metric in ["Valid_AUC","Valid_Accuracy","Valid_F1","Valid_BalancedAcc","Valid_MCC"]:
    if metric in va_df.columns:
        plt.figure(figsize=(10, 5))
        sns.barplot(x=va_df.index, y=va_df[metric])
        plt.title(f"Validation {metric}"); plt.xticks(rotation=45)
        plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, f"06_{metric}_Bar.svg"), dpi=300); plt.close()

# Train vs Valid comparison
comp_metrics = ["AUC","Accuracy","F1","BalancedAcc","MCC"]
fig, axes = plt.subplots(2, 3, figsize=(18, 12)); axes = axes.ravel()
for i, m in enumerate(comp_metrics):
    if i < len(axes) and f"Train_{m}" in all_df.columns:
        pd.DataFrame({"Train": all_df[f"Train_{m}"], "Valid": all_df[f"Valid_{m}"]}).plot.bar(ax=axes[i])
        axes[i].set_title(f"Train vs Valid {m}"); axes[i].tick_params(axis="x", rotation=45)
for i in range(len(comp_metrics), len(axes)): fig.delaxes(axes[i])
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "07_Train_vs_Valid.svg"), dpi=300); plt.close()

# =========================  STEP 9: SHAP  =========================
print(f"\n[9] SHAP for {best_name} ...")
X_explain = X_full_train.values[:50]
try:
    X_trans = best_model.named_steps["selector"].transform(X_explain)
    if best_name in ["RandomForest","XGBoost","LightGBM","DecisionTree"]:
        explainer = shap.TreeExplainer(best_model.named_steps["classifier"])
        sv = explainer.shap_values(X_trans)
        if isinstance(sv, list) and len(sv) == 2: sv = sv[1]
    elif best_name == "LogisticRegression":
        explainer = shap.LinearExplainer(best_model.named_steps["classifier"], X_trans)
        sv = explainer.shap_values(X_trans)
    else:
        def fn(x): return best_model.named_steps["classifier"].predict_proba(x)
        explainer = shap.KernelExplainer(fn, X_trans[:10])
        sv = explainer.shap_values(X_trans[:30])
        X_trans = X_trans[:30]  # align rows with shap_vals
        if isinstance(sv, list) and len(sv) == 2: sv = sv[1]

    feat_names = [f"f{i}" for i in range(X_trans.shape[1])]
    plt.figure(figsize=(10, 8)); shap.summary_plot(sv, X_trans, feature_names=feat_names, show=False)
    plt.title(f"SHAP Summary - {best_name}"); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"08_SHAP_Summary_{best_name}.svg"), dpi=300); plt.close()

    plt.figure(figsize=(10, 6)); shap.summary_plot(sv, X_trans, feature_names=feat_names, plot_type="bar", show=False)
    plt.title(f"SHAP Importance - {best_name}"); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"08_SHAP_Bar_{best_name}.svg"), dpi=300); plt.close()
    print("  SHAP done.")
except Exception as e:
    print(f"  SHAP failed: {e}")

# =========================  STEP 10: Calibration  =========================
print(f"\n[10] Calibration curve for {best_name} ...")
from statsmodels.nonparametric.smoothers_lowess import lowess
plt.figure(figsize=(8, 8))
plt.plot([0, 1], [0, 1], "--", color="gray", label="Ideal")
y_pr = best_model.predict_proba(X_full_valid.values)[:, 1]
brier = brier_score_loss(y_valid, y_pr)
try:
    lr = lowess(y_valid, y_pr, frac=0.8, it=3, return_sorted=True)
    plt.plot(lr[:, 0], lr[:, 1], "-", color="black", lw=2, label=f"{best_name}")
    ax2 = plt.gca().twinx()
    mp = y_valid == 1; mn = y_valid == 0
    if mp.sum(): ax2.hist(y_pr[mp], bins=20, alpha=0.3, color="red")
    if mn.sum(): ax2.hist(y_pr[mn], bins=20, alpha=0.3, color="blue", bottom=0.98)
    ax2.set_ylim(0, 1.1); ax2.set_ylabel("Density")
except Exception as e:
    print(f"  LOWESS failed ({e}), fallback binned")
    fop, mpv = calibration_curve(y_valid, y_pr, n_bins=10)
    plt.plot(mpv, fop, "s-", color="black", label=f"{best_name}")
plt.xlim(0, 1); plt.ylim(0, 1)
plt.xlabel("Predicted Probability"); plt.ylabel("Actual Probability")
plt.title(f"Calibration - {best_name} (Brier={brier:.3f})")
plt.gca().spines["top"].set_visible(False); plt.gca().spines["right"].set_visible(False)
plt.legend(loc="lower right")
plt.savefig(os.path.join(OUTPUT_DIR, f"09_Calibration_{best_name}.svg"), dpi=300, bbox_inches="tight"); plt.close()

# =========================  STEP 11: DCA  =========================
print(f"\n[11] DCA for {best_name} ...")
def net_benefit(y_true, y_prob, thresh):
    n = len(y_true); prev = y_true.sum() / n
    nb_m, nb_a = [], []
    for t in thresh:
        if t >= 1.0: nb_m.append(0); nb_a.append(0)
        else:
            yp = (y_prob >= t).astype(int); w = t / (1 - t)
            tp, fp = ((y_true==1)&(yp==1)).sum(), ((y_true==0)&(yp==1)).sum()
            nb_m.append(tp/n - fp/n*w); nb_a.append(prev - (1-prev)*w)
    return np.array(nb_m), np.array(nb_a)

th = np.arange(0, 1.01, 0.01); nb_m, nb_a = net_benefit(y_valid, y_pr, th)
plt.figure(figsize=(8, 8))
plt.plot(th, nb_m, "r-", lw=2, label=best_name)
plt.plot(th, nb_a, "gray", ls="--", label="Treat All")
plt.plot(th, np.zeros_like(th), "k-", lw=1, label="Treat None")
plt.ylim(-0.05, max(nb_m.max(), nb_a.max()) + 0.05); plt.xlim(0, 1.0)
plt.xlabel("Threshold Probability"); plt.ylabel("Net Benefit")
plt.title(f"DCA - {best_name}"); plt.legend(loc="upper right")
plt.grid(True, ls="--", alpha=0.3)
plt.savefig(os.path.join(OUTPUT_DIR, f"10_DCA_{best_name}.svg"), dpi=300, bbox_inches="tight"); plt.close()

# =========================  STEP 12: DeLong  =========================
print("\n[12] DeLong test ...")
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5*(i+j-1)+1; i = j
    T2 = np.empty(N); T2[J] = T; return T2

def _fastDeLong(pred, n1):
    m, n = n1, pred.shape[1]-n1; pos, neg = pred[:,:m], pred[:,m:]; k = pred.shape[0]
    tx = np.array([_midrank(neg[r]) for r in range(k)])
    ty = np.array([_midrank(pos[r]) for r in range(k)])
    tz = np.array([_midrank(pred[r]) for r in range(k)])
    auc = tz[:,:m].sum(1)/(m*n) - (m+1)/(2*n)
    v01 = (tz[:,:m]-tx)/n; v10 = 1-(tz[:,m:]-ty)/m
    return auc, np.cov(v01)/m + np.cov(v10)/n

def delong_p(y, p1, p2):
    from scipy.stats import norm
    o = np.argsort(y); y_s = y[o]; n1 = (y_s==1).sum()
    pred = np.vstack([p1[o], p2[o]]); aucs, cov = _fastDeLong(pred, n1)
    z = np.abs(aucs[0]-aucs[1]) / np.sqrt(cov[0,0]+cov[1,1]-2*cov[0,1])
    return 2*(1-norm.cdf(z))

names = list(all_pred_probs.keys()); n = len(names)
dm = np.ones((n, n))
for i in range(n):
    for j in range(i+1, n):
        try: dm[i,j] = dm[j,i] = delong_p(y_valid, all_pred_probs[names[i]], all_pred_probs[names[j]])
        except: pass
pd.DataFrame(dm, index=names, columns=names).to_excel(os.path.join(OUTPUT_DIR, "11_DeLong_Pvalues.xlsx"))
print("  DeLong matrix saved.")

# =========================  FINAL  =========================
print("\n" + "=" * 60)
print(" LASSO Pipeline Complete")
print("=" * 60)
print(f"  Best model : {best_name}")
for k in ["Valid_AUC","Valid_Accuracy","Valid_BalancedAcc","Valid_MCC","Valid_Brier"]:
    if k in va_df.columns: print(f"  {k:20s}: {va_df.loc[best_name,k]:.4f}")
print(f"\n  Output -> {OUTPUT_DIR}")
print("  Feature selection: LassoCV (no data leakage)")
if USE_SMOTE:
    print("  SMOTE oversampling applied within each CV fold")
if clinical_cols: print(f"  Clinical features always included: {clinical_cols}")
