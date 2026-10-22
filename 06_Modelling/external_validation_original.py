# -*- coding: utf-8 -*-
# =============================================================================
# External validation for the RMPP prediction model (original CV pipeline)
#
# Loads the bundle produced by the ORIGINAL modeling script
# 04_FeatureSelection_Step_1/I.refactored_pipeline_larger_fonts.py:
#   - model_<name>.pkl        (sklearn Pipeline: CVFeatureSelector -> classifier)
#   - 04_StandardScaler.pkl
#   - 04_Imputation_Means.pkl
#   - model_config.json
#
# Each pipeline performs its own feature selection internally, so the external
# cohort only needs to provide the same pre-filtered feature columns.
#
# Produces paper-ready figures and tables: ROC, confusion matrices,
# calibration curve, decision curve analysis, bootstrap 95% CI for AUC,
# DeLong pairwise comparisons, and a summary Excel file.
# =============================================================================
import json
import os
import warnings

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve, f1_score,
    precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay,
    brier_score_loss, balanced_accuracy_score, matthews_corrcoef
)
from sklearn.calibration import calibration_curve
import pymrmr
from scipy.stats import mannwhitneyu
from statsmodels.sandbox.stats.multicomp import multipletests

# ============================ Paths & config ============================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

BUNDLE_DIR = os.path.join(PROJECT_ROOT, '06_Modelling', 'output')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, '06_Modelling', 'external_validation_original_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Independent external cohort (labeled). Must contain the same 'label',
# radiomics and clinical columns as the internal data.
EXTERNAL_FILE = r"E:\Softwares\PycharmProject\Radiomics\03_MergedData\EXTERNAL_影像+临床版.xlsx"

# If the external radiomics columns carry a prefix (e.g. 'original_'),
# set this to the prefix so it is stripped before matching.
STRIP_FEATURE_PREFIX = None

RANDOM_STATE = 42
MRMR_K = 9
P_THRESHOLD = 0.05


def sanitize_columns(df):
    df = df.copy()
    df.columns = [c.replace('[', '').replace(']', '').replace('<', '')
                  .replace('>', '').replace(':', '_') for c in df.columns]
    return df


class CVFeatureSelector(BaseEstimator, TransformerMixin):
    '''
    Within-CV-fold feature selection. Copied from the modeling script so
    joblib can unpickle the saved sklearn Pipelines (they reference this class).
    '''
    def __init__(self, k=MRMR_K, p_threshold=P_THRESHOLD, protected_indices=None):
        self.k = k
        self.p_threshold = p_threshold
        self.protected_indices = protected_indices if protected_indices else []
        self.selected_features_ = None
        self.p_values_ = None

    def fit(self, X, y):
        all_cols = [f'f{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=all_cols)
        y_series = pd.Series(y)

        protected_cols = [all_cols[i] for i in self.protected_indices if i < len(all_cols)]
        candidate_mask = [i for i in range(X.shape[1]) if i not in self.protected_indices]
        candidate_cols = [all_cols[i] for i in candidate_mask]

        results = []
        for col in candidate_cols:
            g0 = X_df.loc[y_series == 0, col]
            g1 = X_df.loc[y_series == 1, col]
            if len(g0) > 1 and len(g1) > 1:
                _, p = mannwhitneyu(g0, g1, alternative='two-sided')
                results.append({'feature': col, 'p_raw': p})
            else:
                results.append({'feature': col, 'p_raw': 1.0})
        res_df = pd.DataFrame(results).dropna()

        _, p_adj, _, _ = multipletests(res_df['p_raw'], method='fdr_bh')
        res_df['p_adj'] = p_adj
        sig_feats = res_df[res_df['p_adj'] < self.p_threshold]['feature'].tolist()

        if len(sig_feats) == 0:
            sig_feats = res_df.nsmallest(5, 'p_raw')['feature'].tolist()

        n_select = min(self.k, len(sig_feats))
        X_sig = X_df[sig_feats].copy()
        X_sig['label'] = y_series.values
        try:
            mrmr_result = pymrmr.mRMR(X_sig, 'MIQ', n_select)
            mrmr_selected = [f for f in mrmr_result if f != 'label']
        except Exception:
            mrmr_selected = sig_feats[:n_select]

        self.selected_features_ = protected_cols + mrmr_selected
        self.p_values_ = res_df.set_index('feature')['p_adj'].to_dict()
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])
        return X_df[self.selected_features_].values


class SimpleSMOTE(BaseEstimator, TransformerMixin):
    '''Minimal SMOTE implementation (copied for joblib unpickling).'''
    def __init__(self, k_neighbors=5, random_state=42):
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = np.asarray(X)
        y_arr = np.asarray(y) if y is not None else None
        if y_arr is None:
            return X
        classes, counts = np.unique(y_arr, return_counts=True)
        if len(classes) < 2:
            return X
        majority_class = classes[np.argmax(counts)]
        minority_class = classes[np.argmin(counts)]
        n_majority = counts.max()
        n_minority = counts.min()
        n_synthetic = n_majority - n_minority
        if n_synthetic <= 0:
            return X
        X_min = X[y_arr == minority_class]
        nn = NearestNeighbors(n_neighbors=min(self.k_neighbors + 1, n_minority))
        nn.fit(X_min)
        rng = np.random.RandomState(self.random_state)
        synthetic = []
        indices = rng.choice(n_minority, n_synthetic, replace=True)
        for i in indices:
            neighbors = nn.kneighbors([X_min[i]], return_distance=False)[0]
            nn_idx = rng.choice(neighbors[1:])
            diff = X_min[nn_idx] - X_min[i]
            gap = rng.random()
            synthetic.append(X_min[i] + gap * diff)
        X_synthetic = np.array(synthetic)
        y_synthetic = np.full(n_synthetic, minority_class)
        self.y_resampled_ = np.hstack([y_arr, y_synthetic])
        return np.vstack([X, X_synthetic])

    def fit_resample(self, X, y):
        self.fit(X, y)
        return self.transform(X, y), self.y_resampled_


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


def classification_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Sensitivity': recall_score(y_true, y_pred),
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        'PPV': precision_score(y_true, y_pred),
        'NPV': tn / (tn + fn) if (tn + fn) > 0 else np.nan,
        'F1': f1_score(y_true, y_pred),
        'MCC': matthews_corrcoef(y_true, y_pred),
        'BalancedAcc': balanced_accuracy_score(y_true, y_pred),
    }


# ============================ 1. Load bundle ============================
print('=' * 60)
print(' External validation of the RMPP prediction model (original CV pipeline)')
print('=' * 60)

config_path = os.path.join(BUNDLE_DIR, 'model_config.json')
with open(config_path, 'r', encoding='utf-8') as f:
    model_config = json.load(f)

model_names = model_config['model_names']
feature_order = model_config['feature_order']
internal_auc = model_config.get('internal_valid_auc', {})
threshold = float(model_config.get('threshold', 0.5))
youden_threshold = float(model_config.get('youden_threshold', 0.5))
print(f'  Feature order ({len(feature_order)} columns) loaded.')

scaler = joblib.load(os.path.join(BUNDLE_DIR, '04_StandardScaler.pkl'))
impute_values = joblib.load(os.path.join(BUNDLE_DIR, '04_Imputation_Means.pkl'))
models = {name: joblib.load(os.path.join(BUNDLE_DIR, f'model_{name}.pkl'))
          for name in model_names}
print('  Model bundle loaded.')

# ============================ 2. Load external data ============================
print(f'\n[1] Loading external data: {EXTERNAL_FILE}')
ext_df = pd.read_excel(EXTERNAL_FILE).reset_index(drop=True)
print(f'  External shape: {ext_df.shape}')
if 'label' not in ext_df.columns:
    raise ValueError("External data must contain a 'label' column.")
y_ext = ext_df['label'].values
print(f'  External labels: {dict(pd.Series(y_ext).value_counts())}')

id_cols = [c for c in ext_df.columns if c.lower() in ['patientid', 'image', 'mask']]
diag_cols = [c for c in ext_df.columns if c.startswith('diagnostics')]
drop_cols = id_cols + diag_cols + ['label']
X_ext = ext_df.drop(columns=drop_cols, errors='ignore')

# ============================ 3. Align + preprocess ============================
print('\n[2] Aligning external features to the training feature space...')

if STRIP_FEATURE_PREFIX:
    X_ext = X_ext.rename(columns={c: c.replace(STRIP_FEATURE_PREFIX, '', 1)
                                  for c in X_ext.columns if c.startswith(STRIP_FEATURE_PREFIX)})
X_ext = sanitize_columns(X_ext)

missing_cols = [c for c in feature_order if c not in X_ext.columns]
if missing_cols:
    print(f'  WARNING: {len(missing_cols)} columns absent in external data -> imputed with train mean')

X_ext_aligned = X_ext.reindex(columns=feature_order).fillna(impute_values)
X_ext_final = pd.DataFrame(scaler.transform(X_ext_aligned), columns=feature_order)
print(f'  External prediction matrix: {X_ext_final.shape}')

# ============================ 4. Predictions ============================
print('\n[3] Predicting with all models...')
ext_probs = {name: models[name].predict_proba(X_ext_final)[:, 1] for name in model_names}

# ============================ 5. Metrics ============================
print('\n[4] Computing metrics...')
ext_results = {}
for name in model_names:
    prob = ext_probs[name]
    pred = (prob >= threshold).astype(int)
    metrics = classification_metrics(y_ext, pred)
    metrics['AUC'] = roc_auc_score(y_ext, prob)
    metrics['Brier'] = brier_score_loss(y_ext, prob)
    ext_results[name] = metrics

ext_df_res = pd.DataFrame(ext_results).T
ext_df_res['Internal_Valid_AUC'] = [internal_auc.get(n, np.nan) for n in model_names]
ext_df_res = ext_df_res[['AUC', 'Internal_Valid_AUC', 'Accuracy', 'Sensitivity',
                         'Specificity', 'PPV', 'NPV', 'F1', 'MCC', 'BalancedAcc', 'Brier']]
ext_df_res.to_excel(os.path.join(OUTPUT_DIR, 'EX_Model_Performance.xlsx'))
print('  External performance table saved.')

# ============================ 6. ROC ============================
print('\n[5] ROC curves...')
plt.figure(figsize=(10, 8))
for name in model_names:
    fpr, tpr, _ = roc_curve(y_ext, ext_probs[name])
    auc = ext_df_res.loc[name, 'AUC']
    plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('ROC Curves (External Validation Set)', fontsize=18)
plt.tick_params(labelsize=14)
plt.legend(loc='lower right', fontsize=13)
plt.savefig(os.path.join(OUTPUT_DIR, 'EX_ROC_all.svg'), dpi=300)
plt.close()

# ============================ 7. Confusion matrices ============================
print('\n[6] Confusion matrices...')
for name in model_names:
    pred = (ext_probs[name] >= threshold).astype(int)
    cm = confusion_matrix(y_ext, pred)
    ConfusionMatrixDisplay(cm).plot(cmap='Blues')
    plt.title(f'External Confusion Matrix - {name}')
    plt.savefig(os.path.join(OUTPUT_DIR, f'EX_ConfusionMatrix_{name}.svg'), dpi=300)
    plt.close()
print('  Confusion matrices saved.')

# ============================ 8. Bootstrap CI for external AUC ============================
print('\n[7] Bootstrap 95% CI for external AUC (1000 iterations)...')
np.random.seed(RANDOM_STATE)
boot_results = {}
n_ext = len(y_ext)
for name in model_names:
    prob = ext_probs[name]
    aucs = []
    for _ in range(1000):
        idx = np.random.choice(n_ext, n_ext, replace=True)
        try:
            aucs.append(roc_auc_score(y_ext[idx], prob[idx]))
        except ValueError:
            aucs.append(np.nan)
    aucs = np.array(aucs)
    aucs = aucs[~np.isnan(aucs)]
    ci_low = np.percentile(aucs, 2.5)
    ci_high = np.percentile(aucs, 97.5)
    boot_results[name] = {'AUC_95CI_Low': ci_low, 'AUC_95CI_High': ci_high}
    print(f'  {name:20s} AUC={ext_df_res.loc[name, "AUC"]:.4f}  '
          f'95% CI=[{ci_low:.4f}, {ci_high:.4f}]')

ci_df = pd.DataFrame(boot_results).T
ext_df_res = ext_df_res.join(ci_df)
ext_df_res.to_excel(os.path.join(OUTPUT_DIR, 'EX_Model_Performance.xlsx'))
print('  Bootstrap CI saved.')

# ============================ 9. Calibration (best model) ============================
best_name = ext_df_res['AUC'].idxmax()
best_prob = ext_probs[best_name]
print(f'\n[8] Calibration curve for best external model: {best_name}...')
from statsmodels.nonparametric.smoothers_lowess import lowess

brier = brier_score_loss(y_ext, best_prob)
plt.figure(figsize=(8, 8))
plt.plot([0, 1], [0, 1], '--', color='gray', label='Ideal')
try:
    lowess_res = lowess(y_ext, best_prob, frac=0.8, it=3, return_sorted=True)
    plt.plot(lowess_res[:, 0], lowess_res[:, 1], '-', color='black', linewidth=2,
             label=f'Calibration ({best_name})')
except Exception:
    fop, mpv = calibration_curve(y_ext, best_prob, n_bins=10)
    plt.plot(mpv, fop, 's-', color='black', label='Calibration (binned)')
plt.xlim(0, 1); plt.ylim(0, 1)
plt.xlabel('Predicted Probability', fontsize=16)
plt.ylabel('Actual Probability', fontsize=16)
plt.title(f'External Calibration Curve - {best_name} (Brier={brier:.3f})', fontsize=18)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=14)
plt.legend(loc='lower right', fontsize=13)
plt.savefig(os.path.join(OUTPUT_DIR, f'EX_Calibration_{best_name}.svg'), dpi=300, bbox_inches='tight')
plt.close()
print(f'  Calibration curve saved (Brier={brier:.3f}).')

# ============================ 10. DCA (best model) ============================
print(f'\n[9] Decision Curve Analysis for best external model: {best_name}...')
thresholds = np.arange(0, 1.01, 0.01)
nb_model, nb_all = calc_net_benefit(y_ext, best_prob, thresholds)

plt.figure(figsize=(8, 8))
plt.plot(thresholds, nb_model, 'r-', linewidth=2, label=best_name)
plt.plot(thresholds, nb_all, 'gray', linestyle='--', label='Treat All')
plt.plot(thresholds, np.zeros_like(thresholds), 'k-', linewidth=1, label='Treat None')
y_max = max(np.max(nb_model), np.max(nb_all)) + 0.05
plt.ylim(-0.05, y_max); plt.xlim(0, 1.0)
plt.xlabel('Threshold Probability', fontsize=16)
plt.ylabel('Net Benefit', fontsize=16)
plt.title(f'External Decision Curve Analysis - {best_name}', fontsize=18)
plt.tick_params(labelsize=14)
plt.legend(loc='upper right', fontsize=13)
plt.grid(True, linestyle='--', alpha=0.3)
plt.savefig(os.path.join(OUTPUT_DIR, f'EX_DCA_{best_name}.svg'), dpi=300, bbox_inches='tight')
plt.close()
print('  DCA plot saved.')

# ============================ 11. DeLong pairwise ============================
print('\n[10] DeLong test for AUC pairwise comparison (external)...')
n_models = len(model_names)
delong_matrix = np.ones((n_models, n_models))
for i in range(n_models):
    for j in range(i + 1, n_models):
        try:
            p = delong_roc_test(y_ext, ext_probs[model_names[i]], ext_probs[model_names[j]])
            delong_matrix[i, j] = p
            delong_matrix[j, i] = p
        except Exception:
            delong_matrix[i, j] = 1.0
            delong_matrix[j, i] = 1.0
pd.DataFrame(delong_matrix, index=model_names, columns=model_names).to_excel(
    os.path.join(OUTPUT_DIR, 'EX_DeLong_Pvalues.xlsx'))
print('  DeLong p-value matrix saved.')

# ============================ 12. Summary ============================
print('\n' + '=' * 60)
print(' EXTERNAL VALIDATION SUMMARY')
print('=' * 60)
for name in model_names:
    print(f'  {name:20s} AUC={ext_df_res.loc[name, "AUC"]:.4f} '
          f'(internal {ext_df_res.loc[name, "Internal_Valid_AUC"]:.4f}), '
          f'Acc={ext_df_res.loc[name, "Accuracy"]:.4f}, '
          f'F1={ext_df_res.loc[name, "F1"]:.4f}')
print(f'\nBest external model: {best_name} '
      f'(AUC={ext_df_res.loc[best_name, "AUC"]:.4f})')
print(f'Threshold used for classification metrics: {threshold}')
print(f'Youden threshold (from internal validation): {youden_threshold:.4f}')
print(f'\nAll results saved to: {OUTPUT_DIR}')