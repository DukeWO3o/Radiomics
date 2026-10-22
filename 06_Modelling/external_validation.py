# -*- coding: utf-8 -*-
# =============================================================================
# External validation for the RMPP prediction model
#
# Loads the app-ready model bundle produced by
# 04_FeatureSelection_Step_1/I.refactored_pipeline_app.py and evaluates it on
# an independent, LABELED external cohort. The feature extraction of the
# external cohort must follow the same preprocessing / segmentation /
# pyradiomics pipeline as the internal cohort.
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

from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve, f1_score,
    precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay,
    brier_score_loss, balanced_accuracy_score, matthews_corrcoef
)
from sklearn.calibration import calibration_curve

# ============================ Paths & config ============================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

APP_MODELS_DIR = os.path.join(PROJECT_ROOT, '08_RMPP_Prediction_App', 'models')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, '06_Modelling', 'external_validation_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Independent external cohort (labeled). Must contain the same 'label',
# radiomics and clinical columns as the internal data.
EXTERNAL_FILE = r"E:\Softwares\PycharmProject\Radiomics\03_MergedData\EXTERNAL_连云港_影像+临床版-107-8-115.xlsx"

# If the external radiomics columns carry a prefix (e.g. 'original_'),
# set this to the prefix so it is stripped before matching.
STRIP_FEATURE_PREFIX = None

RANDOM_STATE = 42


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
print(' External validation of the RMPP prediction model')
print('=' * 60)

config_path = os.path.join(APP_MODELS_DIR, 'model_config.json')
with open(config_path, 'r', encoding='utf-8') as f:
    model_config = json.load(f)

# Load scalers/models first; the deployment signature is derived from whatever
# bundle format is present (cv_internal_pipeline vs app pipeline).
scaler = joblib.load(os.path.join(APP_MODELS_DIR, '04_StandardScaler.pkl'))
impute_values = joblib.load(os.path.join(APP_MODELS_DIR, '04_Imputation_Means.pkl'))
clinical_scaler_path = os.path.join(APP_MODELS_DIR, '04_ClinicalScaler.pkl')
clinical_scaler = joblib.load(clinical_scaler_path) if os.path.exists(clinical_scaler_path) else None

model_names = model_config['model_names']
models = {name: joblib.load(os.path.join(APP_MODELS_DIR, f'06_{name}_best_model.pkl'))
          for name in model_names}
print('  Model bundle loaded.')

# --- resolve the deployment signature from the bundle ---
clinical_features = (model_config.get('clinical_features')
                     or list(model_config.get('protected_features') or ['age', 'CRP', 'WBC', 'LDH']))

if model_config.get('model_selected_features'):
    per_model_features = {n: model_config['model_selected_features'][n] for n in model_names}
else:
    shared_order = list(model_config['final_feature_order'])
    per_model_features = {n: shared_order for n in model_names}
final_feature_order = per_model_features[model_names[0]]

scaler_feature_names_cfg = model_config.get('scaler_feature_names')
unified_scaling = not scaler_feature_names_cfg or all(
    c in list(scaler.feature_names_in_) for c in clinical_features)
scaler_feature_names = list(scaler.feature_names_in_) if unified_scaling else list(scaler_feature_names_cfg)
selected_radiomics = model_config.get('selected_radiomics') or [
    f for f in final_feature_order if f not in clinical_features]

auc_weights = model_config.get('auc_weights') or model_config.get('internal_valid_auc') or {}
threshold = float(model_config.get('threshold', 0.5))
youden_threshold = float(model_config.get('youden_threshold', 0.5))
print(f'  Signature ({len(final_feature_order)}): {final_feature_order}')
print(f'  Unified scaling (clinical inside StandardScaler): {unified_scaling}')

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

# rename external columns if they carry a prefix, then align to the training space
rad_cols_raw = {c: (c.replace(STRIP_FEATURE_PREFIX, '', 1)
                    if STRIP_FEATURE_PREFIX and c.startswith(STRIP_FEATURE_PREFIX)
                    else c) for c in X_ext.columns}
X_ext_renamed = X_ext.rename(columns=rad_cols_raw)

if unified_scaling:
    # The single StandardScaler covers radiomics + clinical together
    # (I.refactored_pipeline_larger_fonts.py format).
    X_aligned = X_ext_renamed.reindex(columns=scaler_feature_names).fillna(impute_values)
    missing_cols = [c for c in scaler_feature_names if c not in X_ext_renamed.columns]
    if missing_cols:
        print(f'  WARNING: {len(missing_cols)} columns absent in external data -> imputed with train mean')
    X_ext_scaled = pd.DataFrame(scaler.transform(X_aligned), columns=scaler_feature_names)
    print(f'  Scaled full matrix: {X_ext_scaled.shape}')
else:
    # Radiomics-only scaler + optional dedicated clinical scaler (app.py format).
    X_rad = X_ext_renamed.reindex(columns=scaler_feature_names).fillna(impute_values)
    missing_cols = [c for c in scaler_feature_names if c not in X_ext_renamed.columns]
    if missing_cols:
        print(f'  WARNING: {len(missing_cols)} radiomics columns absent in external data -> imputed with train mean')
    X_ext_rad_scaled = pd.DataFrame(scaler.transform(X_rad), columns=scaler_feature_names)

    missing_clinical = [c for c in clinical_features if c not in ext_df.columns]
    if missing_clinical:
        raise ValueError(f"External data missing clinical features: {missing_clinical}")
    clinical_ext = ext_df[clinical_features].reset_index(drop=True)
    clinical_imputation = model_config.get('clinical_imputation', {})
    if clinical_ext.isnull().any().any():
        if clinical_imputation:
            print('  WARNING: missing clinical values imputed with training-set mean')
            clinical_ext = clinical_ext.fillna(pd.Series(clinical_imputation))
        else:
            print('  WARNING: missing clinical values imputed with column mean')
            clinical_ext = clinical_ext.fillna(clinical_ext.mean())
    if clinical_scaler is not None:
        clinical_scaler_cols = list(clinical_scaler.feature_names_in_)
        clinical_ext = clinical_ext.reindex(columns=clinical_scaler_cols)
        clinical_ext_scaled = pd.DataFrame(clinical_scaler.transform(clinical_ext),
                                           columns=clinical_scaler_cols)
        print('  Clinical standardization applied.')
    else:
        clinical_ext_scaled = clinical_ext
        print('  WARNING: 04_ClinicalScaler.pkl not found; clinical features used raw.')

# ============================ 4. Predictions ============================
print('\n[3] Predicting with all models...')
ext_probs = {}
for name in model_names:
    feats = per_model_features[name]
    if unified_scaling:
        X_m = X_ext_scaled[feats]
    else:
        X_m = pd.concat([clinical_ext_scaled.reset_index(drop=True),
                         X_ext_rad_scaled[selected_radiomics].reset_index(drop=True)], axis=1)
        X_m = sanitize_columns(X_m)[feats]
    ext_probs[name] = models[name].predict_proba(X_m)[:, 1]
    print(f'  {name:20s} ({X_m.shape[0]} x {X_m.shape[1]})')

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
ext_df_res['Internal_Valid_AUC'] = [auc_weights.get(n, np.nan) for n in model_names]
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