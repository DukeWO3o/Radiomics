# -*- coding: utf-8 -*-
# =============================================================================
# Radiomics RMPP Prediction - App-Ready Clinical Pipeline
#
# Adapted from "I.refactored_pipeline_larger_fonts.py" so that the exported
# model bundle is directly consumable by the Streamlit app in
# 08_RMPP_Prediction_App.
#
# Key changes vs. the refactored pipeline:
#   1. Deployment signature is FIXED: MWU + FDR + mRMR are performed ONCE on
#      the internal TRAINING set (not inside each CV fold). The app needs one
#      single feature order shared by all 9 models.
#   2. All four clinical variables collected by the app (age, CRP, WBC, LDH)
#      are always included in the signature.
#   3. Both radiomics and clinical features are standardized with dedicated
#      scalers so distance-based models see a common scale.
#   4. Models are exported as plain sklearn estimators, together with the
#      StandardScaler, imputation means and a model_config.json, matching
#      app.py expectations (04_StandardScaler.pkl, 04_Imputation_Means.pkl,
#      06_<name>_best_model.pkl).
# =============================================================================
import json
import os
import warnings
import joblib

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import pymrmr

from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve, f1_score,
    precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay,
    brier_score_loss, balanced_accuracy_score, matthews_corrcoef
)
from sklearn.calibration import calibration_curve

import shap
from scipy.stats import mannwhitneyu
from statsmodels.sandbox.stats.multicomp import multipletests

# ============================ Paths & config ============================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, '06_Modelling', 'output_app')
os.makedirs(OUTPUT_DIR, exist_ok=True)

APP_MODELS_DIR = os.path.join(PROJECT_ROOT, '08_RMPP_Prediction_App', 'models')
os.makedirs(APP_MODELS_DIR, exist_ok=True)

INPUT_FILE = r"E:\Softwares\PycharmProject\Radiomics\03_MergedData\02d_ICC_filtered_影像+临床版-v4-374-557-931.xlsx"

N_FOLDS = 5
RANDOM_STATE = 42
MRMR_K = 9
P_THRESHOLD = 0.05
MV_MISSING = 0.30
MV_VARIANCE = 0.05
USE_SMOTE = False
SMOTE_K_NEIGHBORS = 5

# Clinical variables collected by the app at prediction time.
DEPLOY_CLINICAL_FEATURES = ['age', 'CRP', 'WBC', 'LDH']
# No clinical variables are excluded: all four are required by the app.
EXCLUDE_CLINICAL_FEATURES = []

MODEL_NAMES = ["LogisticRegression", "DecisionTree", "RandomForest", "MLP",
               "SVM", "XGBoost", "LightGBM", "GaussianNB", "KNN"]

print('=' * 60)
print(' Radiomics RMPP Prediction - App-Ready Clinical Pipeline')
print(' Fixed signature (selected on internal training set only)')
print('=' * 60)


def sanitize_columns(df):
    df = df.copy()
    df.columns = [c.replace('[', '').replace(']', '').replace('<', '')
                  .replace('>', '').replace(':', '_') for c in df.columns]
    return df


def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float64)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=np.float64)
    T2[J] = T
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    neg = predictions_sorted_transposed[:, :n]
    pos = predictions_sorted_transposed[:, n:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty([k, n]); ty = np.empty([k, m]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = compute_midrank(neg[r, :])
        ty[r, :] = compute_midrank(pos[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = np.sum(tz[:, n:], axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, n:] - ty[:, :]) / n
    v10 = 1.0 - (tz[:, :n] - tx[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def calc_pvalue(aucs, sigma):
    from scipy.stats import norm
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(l @ sigma @ l.T)
    return 2 * (1 - norm.cdf(z))[0, 0]


def delong_roc_test(y_true, prob1, prob2):
    y_sorted = np.array(y_true)
    order = np.argsort(y_sorted)
    prob1_sorted = prob1[order]
    prob2_sorted = prob2[order]
    y_sorted = y_sorted[order]
    n_pos = np.sum(y_sorted == 1)
    pred = np.vstack([prob1_sorted, prob2_sorted])
    aucs, cov = fastDeLong(pred, n_pos)
    return calc_pvalue(aucs, cov)


def calc_net_benefit(y_true, y_prob, thresholds):
    n = len(y_true)
    prevalence = np.sum(y_true) / n
    nb_model = []
    nb_all = []
    for t in thresholds:
        if t >= 1.0:
            nb_model.append(0)
            nb_all.append(0)
        else:
            y_pred = (y_prob >= t).astype(int)
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            w = t / (1 - t)
            nb_model.append(tp / n - fp / n * w)
            nb_all.append(prevalence - (1 - prevalence) * w)
    return np.array(nb_model), np.array(nb_all)


# ============================ 1. Load & split ============================
print('\n[1] Loading data...')
df = pd.read_excel(INPUT_FILE).reset_index(drop=True)
print(f'  Raw shape: {df.shape}')

print('\n[2] Splitting train/validation (80:20, stratified)...')
train_df, valid_df = train_test_split(
    df, test_size=0.2, random_state=RANDOM_STATE, stratify=df['label'])
train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)
print(f'  Train: {train_df.shape}, Valid: {valid_df.shape}')
print(f'  Train labels: {dict(train_df["label"].value_counts())}')

y_train = train_df['label'].values
y_valid = valid_df['label'].values

# ============================ 2. Preprocessing ============================
print('\n[3] Preprocessing (label-free, fitted on train only)...')
id_cols = [c for c in train_df.columns if c.lower() in ['patientid', 'image', 'mask']]
diag_cols = [c for c in train_df.columns if c.startswith('diagnostics')]
drop_cols = id_cols + diag_cols + ['label']
X_train_full = train_df.drop(columns=drop_cols, errors='ignore')
X_valid_full = valid_df.drop(columns=drop_cols, errors='ignore')

clinical_cols = [c for c in DEPLOY_CLINICAL_FEATURES if c in X_train_full.columns]
exclude_clinical_cols = [c for c in EXCLUDE_CLINICAL_FEATURES if c in X_train_full.columns]
feature_cols = list(X_train_full.columns)
radiomics_cols = [c for c in feature_cols if c not in clinical_cols and c not in exclude_clinical_cols]
print(f'  Clinical kept (raw): {clinical_cols}')
print(f'  Clinical excluded (unavailable at deployment): {exclude_clinical_cols}')
print(f'  Radiomics candidate features: {len(radiomics_cols)}')

# Missing filter (>30%) and variance filter (<0.05) on radiomics only (train)
missing_ratio = X_train_full[radiomics_cols].isnull().mean()
radiomics_keep = missing_ratio[missing_ratio <= MV_MISSING].index.tolist()
train_rad = X_train_full[radiomics_keep].copy()
valid_rad = X_valid_full[radiomics_keep].copy()

impute_values = train_rad.mean()

train_rad_imp = train_rad.fillna(impute_values)
selector_var = VarianceThreshold(threshold=MV_VARIANCE)
selector_var.fit(train_rad_imp)
support_mask = selector_var.get_support()
radiomics_all = [radiomics_keep[i] for i in range(len(radiomics_keep)) if support_mask[i]]

train_rad_imp = train_rad_imp[radiomics_all]
valid_rad_imp = valid_rad[radiomics_all].fillna(impute_values[radiomics_all])
print(f'  After missing+variance filter: {len(radiomics_all)} radiomics features')

# Standardize radiomics only; clinical features are standardized separately below. Fit on a DataFrame so
# feature_names_in_ is populated (required by the Streamlit app).
scaler = StandardScaler()
scaler.fit(train_rad_imp)
train_rad_scaled = pd.DataFrame(scaler.transform(train_rad_imp),
                                columns=radiomics_all, index=train_rad_imp.index)
valid_rad_scaled = pd.DataFrame(scaler.transform(valid_rad_imp),
                                columns=radiomics_all, index=valid_rad_imp.index)
print('  Radiomics standardization done.')


# ============================ 3. Feature selection (fixed signature) ============================
print('\n[4] Feature selection (MWU + FDR + mRMR on the training set)...')


def global_select_features(X_scaled, y, k=MRMR_K, p_threshold=P_THRESHOLD):
    results = []
    for col in X_scaled.columns:
        g0 = X_scaled.loc[y == 0, col]
        g1 = X_scaled.loc[y == 1, col]
        if len(g0) > 1 and len(g1) > 1:
            _, p = mannwhitneyu(g0, g1, alternative='two-sided')
        else:
            p = 1.0
        results.append({'feature': col, 'p_raw': p})
    res_df = pd.DataFrame(results).dropna()
    _, p_adj, _, _ = multipletests(res_df['p_raw'], method='fdr_bh')
    res_df['p_adj'] = p_adj
    sig_feats = res_df[res_df['p_adj'] < p_threshold]['feature'].tolist()
    if len(sig_feats) == 0:
        print('  WARNING: no feature significant after FDR, using top 5 by raw p')
        sig_feats = res_df.nsmallest(5, 'p_raw')['feature'].tolist()

    n_select = min(k, len(sig_feats))
    mrmr_df = X_scaled[sig_feats].copy()
    mrmr_df['label'] = y
    try:
        mrmr_result = pymrmr.mRMR(mrmr_df, 'MIQ', n_select)
        selected = [f for f in mrmr_result if f != 'label']
    except Exception:
        selected = sig_feats[:n_select]
    return selected, res_df.set_index('feature')['p_adj'].to_dict()


selected_radiomics, p_adj_dict = global_select_features(train_rad_scaled, y_train)
print(f'  Selected radiomics ({len(selected_radiomics)}): {selected_radiomics}')

signature = clinical_cols + selected_radiomics
print(f'  Deployment signature ({len(signature)}): {signature}')

# Clinical features (age/CRP/WBC/LDH): impute missing values with training-set
# means, then z-score standardize with a dedicated scaler so the four clinical
# inputs share the same scale as the standardized radiomics features.
clinical_train = train_df[clinical_cols].reset_index(drop=True)
clinical_valid = valid_df[clinical_cols].reset_index(drop=True)
clinical_impute = clinical_train.mean()
if clinical_train.isnull().any().any():
    clinical_train = clinical_train.fillna(clinical_impute)
    clinical_valid = clinical_valid.fillna(clinical_impute)

clinical_scaler = StandardScaler()
clinical_scaler.fit(clinical_train)
clinical_train_scaled = pd.DataFrame(
    clinical_scaler.transform(clinical_train), columns=clinical_cols,
    index=clinical_train.index)
clinical_valid_scaled = pd.DataFrame(
    clinical_scaler.transform(clinical_valid), columns=clinical_cols,
    index=clinical_valid.index)
print('  Clinical standardization done.')

X_train_final = pd.concat([
    clinical_train_scaled, train_rad_scaled[selected_radiomics].reset_index(drop=True)], axis=1)
X_valid_final = pd.concat([
    clinical_valid_scaled, valid_rad_scaled[selected_radiomics].reset_index(drop=True)], axis=1)

X_train_final = sanitize_columns(X_train_final)
X_valid_final = sanitize_columns(X_valid_final)
signature = list(X_train_final.columns)
print(f'  Final training matrix: {X_train_final.shape}')

# ============================ 4. Model definitions ============================
print('\n[5] Model definitions (tuned hyperparameters)...')

scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
print(f'  scale_pos_weight: {scale_pos_weight:.3f}')

models_config = {
    'LogisticRegression': {
        'model': LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, class_weight='balanced'),
        'params': {'C': [0.001, 0.01, 0.1, 1, 10], 'penalty': ['l2'], 'solver': ['lbfgs']}
    },
    'DecisionTree': {
        'model': DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight='balanced'),
        'params': {'max_depth': [2, 3, 4], 'min_samples_leaf': [20, 30, 50, 80]}
    },
    'RandomForest': {
        'model': RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced'),
        'params': {'n_estimators': [100, 300], 'max_depth': [3, 4, 5],
                   'min_samples_leaf': [10, 20, 40], 'max_features': ['sqrt', 0.3]}
    },
    'MLP': {
        'model': MLPClassifier(max_iter=2000, random_state=RANDOM_STATE, early_stopping=True),
        'params': {'hidden_layer_sizes': [(16,), (32,), (16, 8), (32, 16)],
                   'alpha': [0.01, 0.1, 1], 'activation': ['relu']}
    },
    'SVM': {
        'model': SVC(probability=True, random_state=RANDOM_STATE, class_weight='balanced'),
        'params': {'C': [0.01, 0.1, 1, 10], 'gamma': ['scale', 0.001, 0.01, 0.1],
                   'kernel': ['rbf']}
    },
    'XGBoost': {
        'model': XGBClassifier(eval_metric='logloss', random_state=RANDOM_STATE,
                               scale_pos_weight=scale_pos_weight),
        'params': {'n_estimators': [100, 300], 'max_depth': [2, 3],
                   'learning_rate': [0.01, 0.05, 0.1], 'reg_lambda': [1, 10],
                   'min_child_weight': [1, 5], 'subsample': [0.7, 0.8]}
    },
    'LightGBM': {
        'model': LGBMClassifier(random_state=RANDOM_STATE, verbose=-1,
                                scale_pos_weight=scale_pos_weight),
        'params': {'num_leaves': [5, 15, 31], 'max_depth': [2, 3, 4],
                   'learning_rate': [0.01, 0.05], 'min_child_samples': [20, 50, 100],
                   'reg_lambda': [0.1, 1, 5]}
    },
    'GaussianNB': {
        'model': GaussianNB(),
        'params': {'var_smoothing': list(np.logspace(-10, -5, 12))}
    },
    'KNN': {
        'model': KNeighborsClassifier(),
        'params': {'n_neighbors': [5, 7, 9, 11, 15, 21], 'weights': ['uniform', 'distance']}
    }
}

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# ============================ 5. Training ============================
print('\n[6] Training 9 models on the fixed signature...')
print('=' * 60)

best_models = {}
best_params_all = {}
train_results = {}
valid_results = {}
all_pred_probs = {}

plt.figure(figsize=(10, 8))

for name in MODEL_NAMES:
    config = models_config[name]
    print(f'\n--- {name} ---')

    grid = GridSearchCV(config['model'], config['params'], cv=cv, scoring='roc_auc',
                        n_jobs=-1, verbose=0, error_score='raise')
    grid.fit(X_train_final, y_train)
    best_model = grid.best_estimator_
    best_models[name] = best_model
    best_params_all[name] = grid.best_params_
    print(f'  Best params: {grid.best_params_}')

    y_train_pred = best_model.predict(X_train_final)
    y_train_prob = best_model.predict_proba(X_train_final)[:, 1]
    y_valid_pred = best_model.predict(X_valid_final)
    y_valid_prob = best_model.predict_proba(X_valid_final)[:, 1]
    all_pred_probs[name] = y_valid_prob

    train_results[name] = {
        'Train_Accuracy': accuracy_score(y_train, y_train_pred),
        'Train_AUC': roc_auc_score(y_train, y_train_prob),
        'Train_F1': f1_score(y_train, y_train_pred),
        'Train_Precision': precision_score(y_train, y_train_pred),
        'Train_Recall': recall_score(y_train, y_train_pred),
        'Train_BalancedAcc': balanced_accuracy_score(y_train, y_train_pred),
        'Train_MCC': matthews_corrcoef(y_train, y_train_pred),
    }
    valid_results[name] = {
        'Valid_Accuracy': accuracy_score(y_valid, y_valid_pred),
        'Valid_AUC': roc_auc_score(y_valid, y_valid_prob),
        'Valid_F1': f1_score(y_valid, y_valid_pred),
        'Valid_Precision': precision_score(y_valid, y_valid_pred),
        'Valid_Recall': recall_score(y_valid, y_valid_pred),
        'Valid_BalancedAcc': balanced_accuracy_score(y_valid, y_valid_pred),
        'Valid_MCC': matthews_corrcoef(y_valid, y_valid_pred),
        'Valid_Brier': brier_score_loss(y_valid, y_valid_prob),
    }

    fpr, tpr, _ = roc_curve(y_train, y_train_prob)
    plt.plot(fpr, tpr, label=f'{name} (AUC={train_results[name]["Train_AUC"]:.3f})')

    cm = confusion_matrix(y_train, y_train_pred)
    ConfusionMatrixDisplay(cm).plot(cmap='Blues')
    plt.title(f'Train Confusion Matrix - {name}')
    plt.savefig(os.path.join(OUTPUT_DIR, f'02_Train_CM_{name}.svg'), dpi=300)
    plt.close()

    cm = confusion_matrix(y_valid, y_valid_pred)
    ConfusionMatrixDisplay(cm).plot(cmap='Blues')
    plt.title(f'Valid Confusion Matrix - {name}')
    plt.savefig(os.path.join(OUTPUT_DIR, f'02_Valid_CM_{name}.svg'), dpi=300)
    plt.close()

    print(f'  Train AUC={train_results[name]["Train_AUC"]:.3f}, '
          f'Valid AUC={valid_results[name]["Valid_AUC"]:.3f}, '
          f'Valid Brier={valid_results[name]["Valid_Brier"]:.3f}')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('ROC Curves (Training Set)', fontsize=18)
plt.tick_params(labelsize=14)
plt.legend(loc='lower right', fontsize=13)
plt.savefig(os.path.join(OUTPUT_DIR, '03_Train_ROC_all.svg'), dpi=300)
plt.close()

# ============================ 6. Performance table ============================
print('\n[7] Validation evaluation & visualization...')
train_df_res = pd.DataFrame(train_results).T
valid_df_res = pd.DataFrame(valid_results).T
all_df = train_df_res.join(valid_df_res)
all_df.to_excel(os.path.join(OUTPUT_DIR, '04_Model_Performance.xlsx'))
print('  Performance table saved.')

best_name = valid_df_res['Valid_AUC'].idxmax()
best_model = best_models[best_name]
print(f'  Best model: {best_name} (Valid AUC={valid_df_res.loc[best_name, "Valid_AUC"]:.3f})')

feature_records = [{'Model': name, 'N_Features': len(signature),
                    'Selected_Features': ', '.join(signature)} for name in MODEL_NAMES]
pd.DataFrame(feature_records).to_excel(os.path.join(OUTPUT_DIR, '04b_Selected_Features.xlsx'), index=False)
print('  - 04b_Selected_Features.xlsx')

params_records = []
for name in MODEL_NAMES:
    record = {'Model': name, 'Valid_AUC': valid_df_res.loc[name, 'Valid_AUC']}
    for k, v in best_params_all[name].items():
        record[k] = str(v)
    params_records.append(record)
pd.DataFrame(params_records).to_excel(os.path.join(OUTPUT_DIR, '04c_Best_Hyperparameters.xlsx'), index=False)
print('  - 04c_Best_Hyperparameters.xlsx')

# Validation ROC
plt.figure(figsize=(10, 8))
for name in MODEL_NAMES:
    y_prob = best_models[name].predict_proba(X_valid_final)[:, 1]
    auc = roc_auc_score(y_valid, y_prob)
    fpr, tpr, _ = roc_curve(y_valid, y_prob)
    plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('ROC Curves (Validation Set)', fontsize=18)
plt.tick_params(labelsize=14)
plt.legend(loc='lower right', fontsize=13)
plt.savefig(os.path.join(OUTPUT_DIR, '05_Valid_ROC_all.svg'), dpi=300)
plt.close()

# Performance bars
valid_metrics = ['Valid_Accuracy', 'Valid_F1', 'Valid_Precision', 'Valid_Recall',
                 'Valid_AUC', 'Valid_BalancedAcc', 'Valid_MCC']
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
axes = axes.ravel()
for i, metric in enumerate(valid_metrics):
    if i < len(axes) and metric in valid_df_res.columns:
        sns.barplot(x=valid_df_res.index, y=valid_df_res[metric], ax=axes[i])
        axes[i].set_title(f'Validation {metric}')
        axes[i].tick_params(axis='x', rotation=45)
for i in range(len(valid_metrics), len(axes)):
    fig.delaxes(axes[i])
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '06_Valid_Performance_Bar.svg'), dpi=300)
plt.close()

comparison_metrics = ['Accuracy', 'F1', 'Precision', 'Recall', 'AUC']
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()
for i, metric in enumerate(comparison_metrics):
    if i < len(axes):
        comp = pd.DataFrame({'Train': all_df[f'Train_{metric}'], 'Validation': all_df[f'Valid_{metric}']})
        comp.plot(kind='bar', ax=axes[i])
        axes[i].set_title(f'Train vs Valid {metric}')
        axes[i].tick_params(axis='x', rotation=45)
for i in range(len(comparison_metrics), len(axes)):
    fig.delaxes(axes[i])
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '07_Train_vs_Valid_Bar.svg'), dpi=300)
plt.close()
print('  ROC + performance comparison plots saved.')

# ============================ 7. SHAP (best model) ============================
print(f'\n[8] SHAP analysis for best model: {best_name}...')
X_explain = X_train_final.values[:50]

if best_name in ['RandomForest', 'XGBoost', 'LightGBM', 'DecisionTree']:
    explainer = shap.TreeExplainer(best_model)
    shap_vals = explainer.shap_values(X_explain)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1] if len(shap_vals) == 2 else shap_vals[0]
elif best_name == 'LogisticRegression':
    explainer = shap.LinearExplainer(best_model, X_explain)
    shap_vals = explainer.shap_values(X_explain)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1] if len(shap_vals) == 2 else shap_vals[0]
else:
    explainer = shap.KernelExplainer(best_model.predict_proba, X_explain[:10])
    shap_vals = explainer.shap_values(X_explain[:30])
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1] if len(shap_vals) == 2 else shap_vals[0]
    X_explain = X_explain[:30]

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_vals, X_explain, feature_names=signature, show=False)
plt.title(f'SHAP Summary - {best_name}')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f'08_SHAP_Summary_{best_name}.svg'), dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_vals, X_explain, feature_names=signature, plot_type='bar', show=False)
plt.title(f'SHAP Feature Importance - {best_name}')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f'08_SHAP_Bar_{best_name}.svg'), dpi=300)
plt.close()
print('  SHAP plots saved.')

# ============================ 8. Bootstrap CI ============================
print('\n[9] Bootstrap 95% CI for validation AUC (1000 iterations)...')
np.random.seed(RANDOM_STATE)
bootstrap_results = {}
for name in MODEL_NAMES:
    y_prob = best_models[name].predict_proba(X_valid_final)[:, 1]
    aucs = []
    n_val = len(y_valid)
    for _ in range(1000):
        idx = np.random.choice(n_val, n_val, replace=True)
        try:
            aucs.append(roc_auc_score(y_valid[idx], y_prob[idx]))
        except ValueError:
            aucs.append(np.nan)
    aucs = np.array(aucs)
    aucs = aucs[~np.isnan(aucs)]
    ci_low = np.percentile(aucs, 2.5)
    ci_high = np.percentile(aucs, 97.5)
    bootstrap_results[name] = {'AUC_95CI_Low': ci_low, 'AUC_95CI_High': ci_high}
    print(f'  {name:20s} AUC={valid_df_res.loc[name, "Valid_AUC"]:.4f}  '
          f'95% CI=[{ci_low:.4f}, {ci_high:.4f}]')

ci_df = pd.DataFrame(bootstrap_results).T
valid_df_res = valid_df_res.join(ci_df)
all_df = train_df_res.join(valid_df_res)
all_df.to_excel(os.path.join(OUTPUT_DIR, '04_Model_Performance.xlsx'), index=True)
print('  Bootstrap CI saved.')

# ============================ 9. Calibration ============================
print(f'\n[10] Calibration curve for best model: {best_name}...')
from statsmodels.nonparametric.smoothers_lowess import lowess

y_prob_best = best_model.predict_proba(X_valid_final)[:, 1]
brier = brier_score_loss(y_valid, y_prob_best)

plt.figure(figsize=(8, 8))
plt.plot([0, 1], [0, 1], '--', color='gray', label='Ideal')
try:
    lowess_res = lowess(y_valid, y_prob_best, frac=0.8, it=3, return_sorted=True)
    plt.plot(lowess_res[:, 0], lowess_res[:, 1], '-', color='black', linewidth=2,
             label=f'Calibration ({best_name})')
except Exception:
    fop, mpv = calibration_curve(y_valid, y_prob_best, n_bins=10)
    plt.plot(mpv, fop, 's-', color='black', label='Calibration (binned)')
plt.xlim(0, 1); plt.ylim(0, 1)
plt.xlabel('Predicted Probability', fontsize=16)
plt.ylabel('Actual Probability', fontsize=16)
plt.title(f'Calibration Curve - {best_name} (Brier={brier:.3f})', fontsize=18)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=14)
plt.legend(loc='lower right', fontsize=13)
plt.savefig(os.path.join(OUTPUT_DIR, f'09_Calibration_{best_name}.svg'), dpi=300, bbox_inches='tight')
plt.close()
print(f'  Calibration curve saved (Brier={brier:.3f}).')

# ============================ 10. DCA ============================
print(f'\n[11] Decision Curve Analysis for best model: {best_name}...')
thresholds = np.arange(0, 1.01, 0.01)
nb_model, nb_all = calc_net_benefit(y_valid, y_prob_best, thresholds)

plt.figure(figsize=(8, 8))
plt.plot(thresholds, nb_model, 'r-', linewidth=2, label=best_name)
plt.plot(thresholds, nb_all, 'gray', linestyle='--', label='Treat All')
plt.plot(thresholds, np.zeros_like(thresholds), 'k-', linewidth=1, label='Treat None')
y_max = max(np.max(nb_model), np.max(nb_all)) + 0.05
plt.ylim(-0.05, y_max); plt.xlim(0, 1.0)
plt.xlabel('Threshold Probability', fontsize=16)
plt.ylabel('Net Benefit', fontsize=16)
plt.title(f'Decision Curve Analysis - {best_name}', fontsize=18)
plt.tick_params(labelsize=14)
plt.legend(loc='upper right', fontsize=13)
plt.grid(True, linestyle='--', alpha=0.3)
plt.savefig(os.path.join(OUTPUT_DIR, f'10_DCA_{best_name}.svg'), dpi=300, bbox_inches='tight')
plt.close()
print('  DCA plot saved.')

# ============================ 11. DeLong pairwise ============================
print('\n[12] DeLong test for AUC pairwise comparison...')
model_names = MODEL_NAMES
n_models = len(model_names)
delong_matrix = np.ones((n_models, n_models))
for i in range(n_models):
    for j in range(i + 1, n_models):
        try:
            p = delong_roc_test(y_valid, all_pred_probs[model_names[i]], all_pred_probs[model_names[j]])
            delong_matrix[i, j] = p
            delong_matrix[j, i] = p
        except Exception:
            delong_matrix[i, j] = 1.0
            delong_matrix[j, i] = 1.0

delong_df = pd.DataFrame(delong_matrix, index=model_names, columns=model_names)
delong_df.to_excel(os.path.join(OUTPUT_DIR, '11_DeLong_Pvalues.xlsx'))

p_flat, p_positions = [], []
for i in range(n_models):
    for j in range(i + 1, n_models):
        p_flat.append(delong_matrix[i, j])
        p_positions.append((i, j))
if len(p_flat) > 0:
    _, p_corrected, _, _ = multipletests(p_flat, method='fdr_bh')
    delong_corrected = np.ones((n_models, n_models))
    for (i, j), pc in zip(p_positions, p_corrected):
        delong_corrected[i, j] = pc
        delong_corrected[j, i] = pc
    pd.DataFrame(delong_corrected, index=model_names, columns=model_names).to_excel(
        os.path.join(OUTPUT_DIR, '11_DeLong_Pvalues_FDRcorrected.xlsx'))
print('  DeLong p-value matrices saved.')

# ============================ 12. Export app bundle ============================
print('\n[13] Exporting app-compatible model bundle...')

joblib.dump(impute_values[radiomics_all], os.path.join(APP_MODELS_DIR, '04_Imputation_Means.pkl'))
joblib.dump(scaler, os.path.join(APP_MODELS_DIR, '04_StandardScaler.pkl'))
joblib.dump(clinical_scaler, os.path.join(APP_MODELS_DIR, '04_ClinicalScaler.pkl'))
for name in MODEL_NAMES:
    joblib.dump(best_models[name], os.path.join(APP_MODELS_DIR, f'06_{name}_best_model.pkl'))

auc_weights = {name: float(valid_df_res.loc[name, 'Valid_AUC']) for name in MODEL_NAMES}

# Youden threshold for the best model on the internal validation set
fpr_best, tpr_best, thr_best = roc_curve(y_valid, y_prob_best)
youden_idx = np.argmax(tpr_best - fpr_best)
youden_threshold = float(thr_best[youden_idx])

model_config = {
    'version': 'app-ready-v1',
    'model_names': MODEL_NAMES,
    'clinical_features': clinical_cols,
    'excluded_clinical_features': exclude_clinical_cols,
    'clinical_imputation': {c: float(v) for c, v in clinical_impute.items()},
    'clinical_scaler_file': '04_ClinicalScaler.pkl',
    'clinical_scaler_feature_names': clinical_cols,
    'selected_radiomics': selected_radiomics,
    'final_feature_order': signature,
    'scaler_feature_names': radiomics_all,
    'auc_weights': auc_weights,
    'threshold': 0.5,
    'youden_threshold': youden_threshold,
    'best_model': best_name,
    'mrmr_k': MRMR_K,
    'p_threshold': P_THRESHOLD,
    'random_state': RANDOM_STATE,
    'n_folds': N_FOLDS,
    'input_file': INPUT_FILE,
}
with open(os.path.join(APP_MODELS_DIR, 'model_config.json'), 'w', encoding='utf-8') as f:
    json.dump(model_config, f, ensure_ascii=False, indent=2)
print(f'  Bundle exported to: {APP_MODELS_DIR}')
print(f'  Signature ({len(signature)}): {signature}')
print(f'  AUC weights: {auc_weights}')
print(f'  Youden threshold (best model): {youden_threshold:.4f}')

# ============================ Final summary ============================
print('\n' + '=' * 60)
print(' FINAL SUMMARY')
print('=' * 60)
print(f'\nBest model: {best_name}')
print(f'  Valid AUC:      {valid_df_res.loc[best_name, "Valid_AUC"]:.4f}')
print(f'  Valid Accuracy:  {valid_df_res.loc[best_name, "Valid_Accuracy"]:.4f}')
print(f'  Valid Precision: {valid_df_res.loc[best_name, "Valid_Precision"]:.4f}')
print(f'  Valid Recall:    {valid_df_res.loc[best_name, "Valid_Recall"]:.4f}')
print(f'  Valid F1:        {valid_df_res.loc[best_name, "Valid_F1"]:.4f}')
print(f'  Balanced Acc:    {valid_df_res.loc[best_name, "Valid_BalancedAcc"]:.4f}')
print(f'  MCC:             {valid_df_res.loc[best_name, "Valid_MCC"]:.4f}')
print(f'  Brier Score:     {valid_df_res.loc[best_name, "Valid_Brier"]:.4f}')
print(f'\nPaper outputs: {OUTPUT_DIR}')
print(f'App bundle:    {APP_MODELS_DIR}')
print('\nNote:')
print('  - Feature selection is performed ONCE on the internal training set,')
print('    giving a single fixed signature required by the clinical app.')
print('  - Age / CRP / WBC / LDH and radiomics are both z-score standardized.')