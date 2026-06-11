# -*- coding: utf-8 -*-
# =============================================================================
# Radiomics RMPP Prediction - No Data Leakage Pipeline
# Feature selection (MWU+FDR+mRMR) performed WITHIN each CV fold
# Saved to: 04_FeatureSelection_Step_1
# =============================================================================
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import os, warnings, joblib
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

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

# ========================= ·������ =========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else r'E:\Softwares\PycharmProject\Radiomics\04_FeatureSelection_Step_1'
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_FILE = r"E:\Softwares\PycharmProject\Radiomics\03_MergedData\01b_影像版-v4-374-557-931.xlsx"
N_FOLDS = 5
RANDOM_STATE = 42
MRMR_K = 9
P_THRESHOLD = 0.05
MV_MISSING = 0.30
MV_VARIANCE = 0.05

print('='*60)
print(' Radiomics RMPP Prediction - NO Data Leakage Pipeline')
print(' Feature selection (MWU+FDR+mRMR) INSIDE each CV fold')
print('='*60)

# ========================= ����1: �������� =========================
print('\n[1] Loading data...')
df = pd.read_excel(INPUT_FILE)
print(f'  Raw shape: {df.shape}')

# ========================= ����2: ѵ��/��֤������ (8:2) =========================
print('\n[2] Splitting train/validation (80:20, stratified)...')
df = df.reset_index(drop=True)
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE, stratify=df['label'])
train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)
print(f'  Train: {train_df.shape}, Valid: {valid_df.shape}')
print(f'  Train labels: {dict(train_df["label"].value_counts())}')

# ========================= ����3: Ԥɸѡ (��ʹ�ñ�ǩ����й¶) =========================
print('\n[3] Preprocessing (label-free, no leakage)...')
id_cols = [c for c in train_df.columns if c.lower() in ['patientid', 'image', 'mask', 'age']]
y_train = train_df['label'].values
y_valid = valid_df['label'].values

# ��������
X_train_full = train_df.drop(columns=id_cols + ['label'], errors='ignore')
X_valid_full = valid_df.drop(columns=id_cols + ['label'], errors='ignore')

# 3a. �Ƴ�diagnostics��
diag_cols = [c for c in X_train_full.columns if c.startswith('diagnostics')]
X_train_full = X_train_full.drop(columns=diag_cols, errors='ignore')
X_valid_full = X_valid_full.drop(columns=diag_cols, errors='ignore')

# 3b. ȱʧ�� > 30% (����ѵ����)
missing_ratio = X_train_full.isnull().mean()
keep_cols = missing_ratio[missing_ratio <= MV_MISSING].index.tolist()
X_train_full = X_train_full[keep_cols]
X_valid_full = X_valid_full[keep_cols]
print(f'  After missing filter: {X_train_full.shape[1]} features')

# 3c. ���� < 0.05 (����ѵ����)
train_means = X_train_full.mean()
X_train_imputed = X_train_full.fillna(train_means)
selector_var = VarianceThreshold(threshold=MV_VARIANCE)
selector_var.fit(X_train_imputed)
selected_mask = selector_var.get_support()
feature_names_all = X_train_full.columns[selected_mask].tolist()
X_train_clean = X_train_imputed[feature_names_all].copy()
X_valid_clean = X_valid_full[feature_names_all].fillna(train_means[feature_names_all]).copy()
print(f'  After variance filter: {len(feature_names_all)} features')

# 3d. ��׼�� (����ѵ����)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_clean), columns=feature_names_all)
X_valid_scaled = pd.DataFrame(scaler.transform(X_valid_clean), columns=feature_names_all)
print('  Standardization done.')

# ========================= ����4: ��������Կ��ӻ� =========================
print('\n[4] Correlation visualization...')
plt.figure(figsize=(36, 27))
sns.heatmap(X_train_scaled.corr(), cmap='viridis', linewidths=0.5, annot=False)
plt.title('Feature Correlation Matrix (Training Set)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '01_correlation_heatmap.svg'))
plt.close()

fig = sns.clustermap(X_train_scaled.corr(), method='average', metric='euclidean',
                      cmap='viridis', figsize=(30, 30), dendrogram_ratio=(0.1, 0.1),
                      annot=False, linewidths=0.15, linecolor='white')
plt.savefig(os.path.join(OUTPUT_DIR, '01_correlation_clustermap.svg'), dpi=300)
plt.close()
print('  Saved correlation plots.')

print('\nReady for CV-internal feature selection + modeling.')
print(f'  Candidate features (input to CV): {len(feature_names_all)}')
# ========================= ����5: CV�ڲ�����ѡ���� =========================
class CVFeatureSelector(BaseEstimator, TransformerMixin):
    '''
    Within-CV-fold feature selection:
    1. Mann-Whitney U test per feature
    2. FDR (Benjamini-Hochberg) correction
    3. mRMR for final K features
    '''
    def __init__(self, k=MRMR_K, p_threshold=P_THRESHOLD):
        self.k = k
        self.p_threshold = p_threshold
        self.selected_features_ = None
        self.p_values_ = None

    def fit(self, X, y):
        X_df = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])
        y_series = pd.Series(y)

        # Step A: Mann-Whitney U
        results = []
        for col in X_df.columns:
            g0 = X_df.loc[y_series == 0, col]
            g1 = X_df.loc[y_series == 1, col]
            if len(g0) > 1 and len(g1) > 1:
                _, p = mannwhitneyu(g0, g1, alternative='two-sided')
                results.append({'feature': col, 'p_raw': p})
            else:
                results.append({'feature': col, 'p_raw': 1.0})
        res_df = pd.DataFrame(results).dropna()

        # Step B: FDR correction
        _, p_adj, _, _ = multipletests(res_df['p_raw'], method='fdr_bh')
        res_df['p_adj'] = p_adj
        sig_feats = res_df[res_df['p_adj'] < self.p_threshold]['feature'].tolist()

        if len(sig_feats) == 0:
            print('  [CV fold] WARNING: No significant features after FDR, using top 5 by raw p')
            sig_feats = res_df.nsmallest(5, 'p_raw')['feature'].tolist()

        # Step C: mRMR
        n_select = min(self.k, len(sig_feats))
        X_sig = X_df[sig_feats].copy()
        X_sig['label'] = y_series.values
        try:
            mrmr_result = pymrmr.mRMR(X_sig, 'MIQ', n_select)
            self.selected_features_ = [f for f in mrmr_result if f != 'label']
        except Exception:
            self.selected_features_ = sig_feats[:n_select]

        self.p_values_ = res_df.set_index('feature')['p_adj'].to_dict()
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])
        return X_df[self.selected_features_].values

print('[5] CV-internal feature selector defined.')

# ========================= ����6: ģ�Ͷ��� =========================
models_config = {
    'LogisticRegression': {
        'model': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced'),
        'params': {'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
                   'classifier__penalty': ['l1', 'l2'], 'classifier__solver': ['liblinear']}
    },
    'DecisionTree': {
        'model': DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight='balanced'),
        'params': {'classifier__max_depth': [2, 3, 4], 'classifier__min_samples_split': [20, 30, 40],
                   'classifier__min_samples_leaf': [15, 20, 25], 'classifier__max_features': [0.3, 0.5, 0.7]}
    },
    'RandomForest': {
        'model': RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced'),
        'params': {'classifier__n_estimators': [100, 200], 'classifier__max_depth': [3, 4, 5],
                   'classifier__min_samples_split': [20, 30], 'classifier__min_samples_leaf': [10, 15],
                   'classifier__max_features': [0.3, 0.5], 'classifier__bootstrap': [True]}
    },
    'MLP': {
        'model': MLPClassifier(max_iter=2000, random_state=RANDOM_STATE, early_stopping=True),
        'params': {'classifier__hidden_layer_sizes': [(50,), (100,), (50, 25)],
                   'classifier__alpha': [0.0001, 0.001, 0.01], 'classifier__activation': ['relu', 'tanh']}
    },
    'SVM': {
        'model': SVC(probability=True, random_state=RANDOM_STATE, class_weight='balanced'),
        'params': {'classifier__C': [0.1, 1, 5, 10], 'classifier__gamma': ['scale', 'auto'],
                   'classifier__kernel': ['rbf', 'linear']}
    },
    'XGBoost': {
        'model': XGBClassifier(eval_metric='logloss', random_state=RANDOM_STATE),
        'params': {'classifier__n_estimators': [100, 200], 'classifier__max_depth': [3, 4],
                   'classifier__learning_rate': [0.05, 0.1, 0.15], 'classifier__gamma': [0, 0.1, 0.5],
                   'classifier__reg_alpha': [0, 0.1, 0.5], 'classifier__reg_lambda': [1, 5, 10]}
    },
    'LightGBM': {
        'model': LGBMClassifier(random_state=RANDOM_STATE, verbose=-1),
        'params': {'classifier__n_estimators': [100, 200], 'classifier__max_depth': [3, 4],
                   'classifier__learning_rate': [0.05, 0.1], 'classifier__num_leaves': [15, 31],
                   'classifier__min_child_samples': [20, 30], 'classifier__reg_alpha': [0.1, 0.3],
                   'classifier__reg_lambda': [0.1, 0.3]}
    },
    'GaussianNB': {
        'model': GaussianNB(),
        'params': {'classifier__var_smoothing': [1e-9, 1e-8, 1e-7]}
    },
    'KNN': {
        'model': KNeighborsClassifier(),
        'params': {'classifier__n_neighbors': [5, 7, 9, 11, 13]}
    }
}

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
print(f'[6] 9 models configured with {N_FOLDS}-fold CV.')
# ========================= ����7: ѵ������ģ�� (CV�ڲ�����ѡ��) =========================
print('\n[7] Training 9 models (feature selection INSIDE each CV fold)...')
print('=' * 60)

best_models = {}
train_results = {}
valid_results = {}
all_pred_probs = {}  # for DeLong test

# ѵ����ROCͼ
plt.figure(figsize=(10, 8))

for name, config in models_config.items():
    print(f'\n--- {name} ---')

    # ����Pipeline: ����ѡ�� -> ������
    pipeline = Pipeline([
        ('selector', CVFeatureSelector(k=MRMR_K)),
        ('classifier', config['model'])
    ])

    # GridSearchCV
    grid = GridSearchCV(pipeline, config['params'], cv=cv, scoring='roc_auc',
                        n_jobs=-1, verbose=0, error_score='raise')
    grid.fit(X_train_scaled.values, y_train)
    best_model = grid.best_estimator_
    best_models[name] = best_model
    print(f'  Best params: {grid.best_params_}')
    print(f'  Selected features: {best_model.named_steps["selector"].selected_features_}')

    joblib.dump(best_model, os.path.join(OUTPUT_DIR, f'model_{name}.pkl'))

    # ѵ��������
    y_train_pred = best_model.predict(X_train_scaled.values)
    y_train_prob = best_model.predict_proba(X_train_scaled.values)[:, 1]

    train_acc = accuracy_score(y_train, y_train_pred)
    train_auc = roc_auc_score(y_train, y_train_prob)
    train_f1 = f1_score(y_train, y_train_pred)
    train_prec = precision_score(y_train, y_train_pred)
    train_rec = recall_score(y_train, y_train_pred)
    train_bal = balanced_accuracy_score(y_train, y_train_pred)
    train_mcc = matthews_corrcoef(y_train, y_train_pred)

    train_results[name] = {'Train_Accuracy': train_acc, 'Train_AUC': train_auc,
                           'Train_F1': train_f1, 'Train_Precision': train_prec,
                           'Train_Recall': train_rec, 'Train_BalancedAcc': train_bal,
                           'Train_MCC': train_mcc}

    # ѵ����ROC����
    fpr, tpr, _ = roc_curve(y_train, y_train_prob)
    plt.plot(fpr, tpr, label=f'{name} (AUC={train_auc:.3f})')

    # ѵ������������
    cm = confusion_matrix(y_train, y_train_pred)
    ConfusionMatrixDisplay(cm).plot(cmap='Blues')
    plt.title(f'Train Confusion Matrix - {name}')
    plt.savefig(os.path.join(OUTPUT_DIR, f'02_Train_CM_{name}.svg'), dpi=300)
    plt.close()

    # ��֤������
    y_valid_pred = best_model.predict(X_valid_scaled.values)
    y_valid_prob = best_model.predict_proba(X_valid_scaled.values)[:, 1]
    all_pred_probs[name] = y_valid_prob

    valid_acc = accuracy_score(y_valid, y_valid_pred)
    valid_auc = roc_auc_score(y_valid, y_valid_prob)
    valid_f1 = f1_score(y_valid, y_valid_pred)
    valid_prec = precision_score(y_valid, y_valid_pred)
    valid_rec = recall_score(y_valid, y_valid_pred)
    valid_bal = balanced_accuracy_score(y_valid, y_valid_pred)
    valid_mcc = matthews_corrcoef(y_valid, y_valid_pred)
    valid_brier = brier_score_loss(y_valid, y_valid_prob)

    valid_results[name] = {'Valid_Accuracy': valid_acc, 'Valid_AUC': valid_auc,
                           'Valid_F1': valid_f1, 'Valid_Precision': valid_prec,
                           'Valid_Recall': valid_rec, 'Valid_BalancedAcc': valid_bal,
                           'Valid_MCC': valid_mcc, 'Valid_Brier': valid_brier}

    # ��֤����������
    cm = confusion_matrix(y_valid, y_valid_pred)
    ConfusionMatrixDisplay(cm).plot(cmap='Blues')
    plt.title(f'Valid Confusion Matrix - {name}')
    plt.savefig(os.path.join(OUTPUT_DIR, f'02_Valid_CM_{name}.svg'), dpi=300)
    plt.close()

    print(f'  Train AUC={train_auc:.3f}, Valid AUC={valid_auc:.3f}, Valid Brier={valid_brier:.3f}')

# ����ѵ����ROCͼ
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC Curves (Training Set)')
plt.legend(loc='lower right')
plt.savefig(os.path.join(OUTPUT_DIR, '03_Train_ROC_all.svg'), dpi=300)
plt.close()
# ========================= ����8: ��֤��ROC + ���ܶԱ� =========================
print('\n[8] Validation set evaluation & visualization...')

train_df_res = pd.DataFrame(train_results).T
valid_df_res = pd.DataFrame(valid_results).T
all_df = train_df_res.join(valid_df_res)
all_df.to_excel(os.path.join(OUTPUT_DIR, '04_Model_Performance.xlsx'))
print('  Performance table saved.')

# ȷ�����ģ�� (������֤��AUC)
best_name = valid_df_res['Valid_AUC'].idxmax()
best_model = best_models[best_name]
print(f'  Best model: {best_name} (Valid AUC={valid_df_res.loc[best_name, "Valid_AUC"]:.3f})')

# ��֤��ROCͼ
plt.figure(figsize=(10, 8))
for name, model in best_models.items():
    y_prob = model.predict_proba(X_valid_scaled.values)[:, 1]
    auc = roc_auc_score(y_valid, y_prob)
    fpr, tpr, _ = roc_curve(y_valid, y_prob)
    plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC Curves (Validation Set)')
plt.legend(loc='lower right')
plt.savefig(os.path.join(OUTPUT_DIR, '05_Valid_ROC_all.svg'), dpi=300)
plt.close()

# ���ܶԱ���״ͼ
valid_metrics = ['Valid_Accuracy', 'Valid_F1', 'Valid_Precision', 'Valid_Recall', 'Valid_AUC', 'Valid_BalancedAcc', 'Valid_MCC']
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

# Train vs Valid �Ա�
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
# ========================= ����9: SHAP���� (���ģ��) =========================
print(f'\n[9] SHAP analysis for best model: {best_name}...')

X_explain = X_train_scaled.values[:50]
feature_names_arr = np.array(feature_names_all)

if best_name in ['RandomForest', 'XGBoost', 'LightGBM', 'DecisionTree']:
    explainer = shap.TreeExplainer(best_model.named_steps['classifier'])
    # Get transformed features for SHAP
    X_transformed = best_model.named_steps['selector'].transform(X_explain)
    sel_features = best_model.named_steps['selector'].selected_features_
    shap_vals = explainer.shap_values(X_transformed)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1] if len(shap_vals) == 2 else shap_vals[0]
elif best_name == 'LogisticRegression':
    X_transformed = best_model.named_steps['selector'].transform(X_explain)
    sel_features = best_model.named_steps['selector'].selected_features_
    explainer = shap.LinearExplainer(best_model.named_steps['classifier'], X_transformed)
    shap_vals = explainer.shap_values(X_transformed)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1] if len(shap_vals) == 2 else shap_vals[0]
else:
    # KernelExplainer fallback
    X_transformed = best_model.named_steps['selector'].transform(X_explain)
    sel_features = best_model.named_steps['selector'].selected_features_
    def pred_fn(x):
        return best_model.named_steps['classifier'].predict_proba(x)
    explainer = shap.KernelExplainer(pred_fn, X_transformed[:10])
    shap_vals = explainer.shap_values(X_transformed[:30])
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1] if len(shap_vals) == 2 else shap_vals[0]

# SHAP Summary Plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_vals, X_transformed, feature_names=sel_features, show=False)
plt.title(f'SHAP Summary - {best_name}')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f'08_SHAP_Summary_{best_name}.svg'), dpi=300)
plt.close()

# SHAP Bar Plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_vals, X_transformed, feature_names=sel_features, plot_type='bar', show=False)
plt.title(f'SHAP Feature Importance - {best_name}')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f'08_SHAP_Bar_{best_name}.svg'), dpi=300)
plt.close()
print('  SHAP plots saved.')
# ========================= ����10: У׼���� =========================
print(f'\n[10] Calibration curve for best model: {best_name}...')

from statsmodels.nonparametric.smoothers_lowess import lowess

plt.figure(figsize=(8, 8))
plt.plot([0, 1], [0, 1], '--', color='gray', label='Ideal')

y_prob = best_model.predict_proba(X_valid_scaled.values)[:, 1]
brier = brier_score_loss(y_valid, y_prob)

# LOWESS ƽ��У׼
try:
    lowess_res = lowess(y_valid, y_prob, frac=0.8, it=3, return_sorted=True)
    plt.plot(lowess_res[:, 0], lowess_res[:, 1], '-', color='black', linewidth=2, label=f'Calibration ({best_name})')

    # �ܶȷֲ�
    ax2 = plt.gca().twinx()
    mask_pos = y_valid == 1
    mask_neg = y_valid == 0
    if np.sum(mask_pos) > 0:
        ax2.hist(y_prob[mask_pos], bins=20, alpha=0.3, color='red', label='RMPP(+)')
    if np.sum(mask_neg) > 0:
        ax2.hist(y_prob[mask_neg], bins=20, alpha=0.3, bottom=0.98, color='blue', label='RMPP(-)')
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel('Density')
except Exception as e:
    print(f'  LOWESS failed: {e}, using standard calibration')
    fop, mpv = calibration_curve(y_valid, y_prob, n_bins=10)
    plt.plot(mpv, fop, 's-', color='black', label='Calibration (binned)')

plt.xlim(0, 1); plt.ylim(0, 1)
plt.xlabel('Predicted Probability'); plt.ylabel('Actual Probability')
plt.title(f'Calibration Curve - {best_name} (Brier={brier:.3f})')
ax = plt.gca()
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.legend(loc='lower right')
plt.savefig(os.path.join(OUTPUT_DIR, f'09_Calibration_{best_name}.svg'), dpi=300, bbox_inches='tight')
plt.close()
print(f'  Calibration curve saved (Brier={brier:.3f}).')
# ========================= ����11: �������߷��� (DCA) =========================
print(f'\n[11] Decision Curve Analysis for best model: {best_name}...')

def calc_net_benefit(y_true, y_prob, thresholds):
    n = len(y_true)
    prevalence = np.sum(y_true) / n
    nb_model = []
    nb_all = []
    for t in thresholds:
        if t >= 1.0:
            nb_model.append(0); nb_all.append(0)
        else:
            y_pred = (y_prob >= t).astype(int)
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            w = t / (1 - t)
            nb_model.append(tp / n - fp / n * w)
            nb_all.append(prevalence - (1 - prevalence) * w)
    return np.array(nb_model), np.array(nb_all)

thresholds = np.arange(0, 1.01, 0.01)
nb_model, nb_all = calc_net_benefit(y_valid, y_prob, thresholds)

plt.figure(figsize=(8, 8))
plt.plot(thresholds, nb_model, 'r-', linewidth=2, label=best_name)
plt.plot(thresholds, nb_all, 'gray', linestyle='--', label='Treat All')
plt.plot(thresholds, np.zeros_like(thresholds), 'k-', linewidth=1, label='Treat None')
y_max = max(np.max(nb_model), np.max(nb_all)) + 0.05
plt.ylim(-0.05, y_max); plt.xlim(0, 1.0)
plt.xlabel('Threshold Probability'); plt.ylabel('Net Benefit')
plt.title(f'Decision Curve Analysis - {best_name}')
plt.legend(loc='upper right'); plt.grid(True, linestyle='--', alpha=0.3)
plt.savefig(os.path.join(OUTPUT_DIR, f'10_DCA_{best_name}.svg'), dpi=300, bbox_inches='tight')
plt.close()
print('  DCA plot saved.')
# ========================= ����12: DeLong���� =========================
print('\n[12] DeLong test for AUC pairwise comparison...')

def compute_midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x)
    T = np.zeros(N, dtype=np.float64); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1; i = j
    T2 = np.empty(N, dtype=np.float64); T2[J] = T
    return T2

def fastDeLong(predictions_sorted_transposed, label_1_count):
    m = label_1_count; n = predictions_sorted_transposed.shape[1] - m
    pos = predictions_sorted_transposed[:, :m]; neg = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = compute_midrank(neg[r, :])
        ty[r, :] = compute_midrank(pos[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = np.sum(tz[:, :m], axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01); sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov

def calc_pvalue(aucs, sigma):
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(l @ sigma @ l.T)
    from scipy.stats import norm
    return 2 * (1 - norm.cdf(z))[0, 0]

def delong_roc_test(y_true, prob1, prob2):
    y_sorted = np.array(y_true)
    order = np.argsort(y_sorted)
    prob1_sorted = prob1[order]; prob2_sorted = prob2[order]
    y_sorted = y_sorted[order]
    n_pos = np.sum(y_sorted == 1)
    pred = np.vstack([prob1_sorted, prob2_sorted])
    aucs, cov = fastDeLong(pred, n_pos)
    return calc_pvalue(aucs, cov)

# ����ģ�������Ƚ�
model_names = list(all_pred_probs.keys())
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
print('  DeLong p-value matrix saved.')

# ========================= ����13: ���ջ��� =========================
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

print(f'\nAll results saved to: {OUTPUT_DIR}')
print('\nKey design difference from original:')
print('  - Mann-Whitney U + FDR + mRMR performed WITHIN each CV fold')
print('  - No data leakage: feature selection uses only training-fold data')
print('  - All 9 models compared fairly with identical CV splits')
