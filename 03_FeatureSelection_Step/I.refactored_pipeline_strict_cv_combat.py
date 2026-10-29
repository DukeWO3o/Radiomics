"""Usage:
python I.refactored_pipeline_strict_cv_combat.py --modality combined
python I.refactored_pipeline_strict_cv_combat.py --modality clinical
python I.refactored_pipeline_strict_cv_combat.py --modality radiomics
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import warnings
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import statsmodels.api as sm
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from scipy.stats import mannwhitneyu, norm
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from statsmodels.sandbox.stats.multicomp import multipletests
from statsmodels.nonparametric.smoothers_lowess import lowess
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import pymrmr
from strict_cv_pickle_helpers import ComBatTransformer


warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
from pipeline_paths import (
    DEFAULT_PIXEL_MODEL_DATA,
    OUTPUT_ROOT,
    PIXEL_COMBAT_ROOT,
    PIXEL_PREPROCESS_ROOT,
)

INPUT_FILE = DEFAULT_PIXEL_MODEL_DATA
STRICT_ROOT = PIXEL_COMBAT_ROOT
OUTPUT_LABELS = {
    "combined": "Combined",
    "clinical": "Clinical",
    "radiomics": "Radiomics",
}
CLINICAL_FEATURES = ["age", "CRP", "WBC", "LDH"]
RANDOM_STATE = 42
N_FOLDS = 5
MRMR_K = 8
P_THRESHOLD = 0.05
MISSING_THRESHOLD = 0.30


VARIANCE_THRESHOLD = 0.0
PREDICTION_THRESHOLD = 0.5


TUNED_COMBINED_PARAMS = {
    "LogisticRegression": {"C": 0.1, "penalty": "l2", "solver": "liblinear"},
    "DecisionTree": {"max_depth": 3, "min_samples_leaf": 10},
    "RandomForest": {
        "max_depth": 3,
        "min_samples_leaf": 10,
        "n_estimators": 100,
    },
    "MLP": {"alpha": 0.01, "hidden_layer_sizes": [128, 64]},
    "SVM": {"C": 2, "gamma": 0.025},
    "XGBoost": {"learning_rate": 0.05, "max_depth": 3, "n_estimators": 50},
    "LightGBM": {"learning_rate": 0.03, "n_estimators": 50, "num_leaves": 7},
    "GaussianNB": {"var_smoothing": 1e-12},
    "KNN": {"n_neighbors": 15, "weights": "distance"},
}

FOCUSED_RETUNE_MODELS = ["DecisionTree", "RandomForest", "XGBoost", "KNN"]


class StrictPreprocessSelector(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        k=MRMR_K,
        p_threshold=P_THRESHOLD,
        protected_features=None,
        missing_threshold=MISSING_THRESHOLD,
        variance_threshold=VARIANCE_THRESHOLD,
        random_state=RANDOM_STATE,
    ):
        self.k = k
        self.p_threshold = p_threshold
        self.protected_features = protected_features
        self.missing_threshold = missing_threshold
        self.variance_threshold = variance_threshold
        self.random_state = random_state

    def _as_frame(self, X):
        if isinstance(X, pd.DataFrame):
            return X.copy()
        return pd.DataFrame(X, columns=self.feature_names_in_)

    def fit(self, X, y):
        frame = self._as_frame(X)
        self.feature_names_in_ = list(frame.columns)
        protected = [f for f in (self.protected_features or []) if f in frame.columns]
        self.protected_features_ = protected

        missing_ratio = frame.isna().mean()
        self.missing_keep_ = [
            f
            for f in self.feature_names_in_
            if missing_ratio[f] <= self.missing_threshold or f in protected
        ]
        self.impute_means_ = frame[self.missing_keep_].mean()
        imputed = frame[self.missing_keep_].fillna(self.impute_means_)

        raw_variances = imputed.var()
        self.variance_keep_ = [
            f
            for f in self.missing_keep_
            if f in protected or raw_variances[f] > self.variance_threshold
        ]
        imputed = imputed[self.variance_keep_]

        self.scaler_ = StandardScaler()
        scaled = pd.DataFrame(
            self.scaler_.fit_transform(imputed),
            columns=self.variance_keep_,
            index=frame.index,
        )

        candidates = [f for f in self.variance_keep_ if f not in protected]
        self.univariate_results_ = pd.DataFrame(columns=["feature", "p_raw", "p_adj"])
        self.significant_features_ = []
        self.mrmr_features_ = []

        if candidates:
            rows = []
            y_array = np.asarray(y)
            for feature in candidates:
                group0 = scaled.loc[y_array == 0, feature]
                group1 = scaled.loc[y_array == 1, feature]
                if len(group0) > 1 and len(group1) > 1:
                    p_raw = mannwhitneyu(
                        group0, group1, alternative="two-sided"
                    ).pvalue
                else:
                    p_raw = 1.0
                rows.append({"feature": feature, "p_raw": p_raw})

            results = pd.DataFrame(rows)
            _, p_adj, _, _ = multipletests(results["p_raw"], method="fdr_bh")
            results["p_adj"] = p_adj
            self.univariate_results_ = results
            self.significant_features_ = results.loc[
                results["p_adj"] < self.p_threshold, "feature"
            ].tolist()
            if not self.significant_features_:
                self.significant_features_ = (
                    results.nsmallest(min(5, len(results)), "p_raw")["feature"].tolist()
                )

            n_select = min(self.k, len(self.significant_features_))
            if n_select > 0:
                mrmr_input = scaled[self.significant_features_].copy()
                mrmr_input["label"] = np.asarray(y)
                try:
                    with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):


                        n_request = min(len(self.significant_features_), self.k + 2)
                        mrmr_result = pymrmr.mRMR(mrmr_input, "MIQ", n_request)
                    self.mrmr_features_ = [
                        f for f in mrmr_result if f != "label" and f in candidates
                    ][:n_select]
                    if len(self.mrmr_features_) < n_select:
                        for feature in self.significant_features_:
                            if feature not in self.mrmr_features_:
                                self.mrmr_features_.append(feature)
                            if len(self.mrmr_features_) == n_select:
                                break
                except Exception:
                    self.mrmr_features_ = self.significant_features_[:n_select]

        self.selected_features_ = list(
            dict.fromkeys(protected + self.mrmr_features_)
        )
        self.n_missing_kept_ = len(self.missing_keep_)
        self.n_variance_kept_ = len(self.variance_keep_)
        self.n_significant_ = len(self.significant_features_)
        self.n_selected_radiomics_ = len(
            [f for f in self.selected_features_ if f not in protected]
        )
        return self

    def transform(self, X):
        frame = self._as_frame(X)
        imputed = frame[self.missing_keep_].fillna(self.impute_means_)
        imputed = imputed[self.variance_keep_]
        scaled = self.scaler_.transform(imputed)
        indices = [self.variance_keep_.index(f) for f in self.selected_features_]
        return scaled[:, indices]


def safe_logit(prob):
    prob = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(prob / (1 - prob))


def calc_net_benefit(y_true, y_prob, threshold):
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    n = len(y_true)
    return tp / n - fp / n * threshold / (1 - threshold)


def binary_predictions_from_probability(
    y_prob, threshold=PREDICTION_THRESHOLD
):
    return (np.asarray(y_prob) >= threshold).astype(int)


def compute_midrank(x):
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    result = np.empty(n, dtype=float)
    result[order] = ranks
    return result


def fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    neg = predictions_sorted_transposed[:, :n]
    pos = predictions_sorted_transposed[:, n:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty([k, n])
    ty = np.empty([k, m])
    tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = compute_midrank(neg[r, :])
        ty[r, :] = compute_midrank(pos[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = np.sum(tz[:, n:], axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, n:] - ty[:, :]) / n
    v10 = 1.0 - (tz[:, :n] - tx[:, :]) / m
    covariance = np.cov(v01) / m + np.cov(v10) / n
    return aucs, covariance


def delong_pvalue(y_true, prob1, prob2):
    y_true = np.asarray(y_true)
    order = np.argsort(y_true)
    probs = np.vstack([np.asarray(prob1)[order], np.asarray(prob2)[order]])
    aucs, covariance = fast_delong(probs, int(np.sum(y_true[order] == 1)))
    contrast = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(contrast @ covariance @ contrast.T)
    return float(2 * (1 - norm.cdf(z))[0, 0])


def bootstrap_auc_ci(y_true, y_prob, seed=RANDOM_STATE, n_boot=1000):
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    values = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        values.append(roc_auc_score(y_true[idx], np.asarray(y_prob)[idx]))
    return np.percentile(values, 2.5), np.percentile(values, 97.5)


def calibration_stats(y_true, y_prob):
    logit = safe_logit(y_prob)
    try:
        slope_model = sm.GLM(
            y_true, sm.add_constant(logit), family=sm.families.Binomial()
        ).fit()
        intercept_model = sm.GLM(
            y_true,
            np.ones((len(y_true), 1)),
            family=sm.families.Binomial(),
            offset=logit,
        ).fit()
        return float(intercept_model.params[0]), float(slope_model.params[1])
    except Exception:


        return np.nan, np.nan


def metric_dict(y_true, y_prob):
    y_pred = binary_predictions_from_probability(y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    ci_low, ci_high = bootstrap_auc_ci(y_true, y_prob)
    intercept, slope = calibration_stats(y_true, y_prob)
    return {
        "AUC": roc_auc_score(y_true, y_prob),
        "AUC_95CI_Low": ci_low,
        "AUC_95CI_High": ci_high,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "Specificity": tn / (tn + fp),
        "PPV": tp / (tp + fp) if (tp + fp) else np.nan,
        "NPV": tn / (tn + fn) if (tn + fn) else np.nan,
        "F1": f1_score(y_true, y_pred),
        "BalancedAcc": balanced_accuracy_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Brier": brier_score_loss(y_true, y_prob),
        "CalibrationIntercept": intercept,
        "CalibrationSlope": slope,
        "NetBenefit_0.3": calc_net_benefit(y_true, y_prob, 0.3),
        "NetBenefit_0.5": calc_net_benefit(y_true, y_prob, 0.5),
        "NetBenefit_0.7": calc_net_benefit(y_true, y_prob, 0.7),
    }


def load_data(modality, input_file=None):
    frame = pd.read_excel(input_file or INPUT_FILE).reset_index(drop=True)
    metadata = [
        column
        for column in ["label", "image", "mask", "batch_model"]
        if column in frame.columns
    ]
    radiomics = [
        column
        for column in frame.columns
        if column.startswith("original_")
    ]
    if modality == "clinical":
        features = CLINICAL_FEATURES
        protected = CLINICAL_FEATURES
    elif modality == "radiomics":
        features = radiomics
        protected = []
    else:
        features = radiomics + CLINICAL_FEATURES
        protected = CLINICAL_FEATURES
    columns = metadata + features
    return frame[columns].copy(), features, protected, radiomics


def model_input(
    frame: pd.DataFrame, features: list[str], modality: str
) -> pd.DataFrame:
    columns = list(features)
    if modality != "clinical":
        columns.append("batch_model")
    return frame[columns].copy()


def build_model_pipeline(
    modality: str,
    radiomics_features: list[str],
    protected: list[str],
    classifier,
) -> Pipeline:
    steps = []
    if modality != "clinical":
        steps.append(
            (
                "combat",
                ComBatTransformer(
                    radiomics_features=radiomics_features,
                    protected_features=protected,
                    batch_col="batch_model",
                    min_batch_n=20,
                ),
            )
        )
    steps.append(
        (
            "selector",
            StrictPreprocessSelector(
                k=MRMR_K,
                protected_features=protected,
                missing_threshold=MISSING_THRESHOLD,
                variance_threshold=VARIANCE_THRESHOLD,
            ),
        )
    )
    steps.append(("classifier", classifier))
    return Pipeline(steps)


def model_configs():
    return {
        "LogisticRegression": {
            "model": LogisticRegression(
                max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
            ),
            "params": {
                "classifier__C": [0.1],
                "classifier__penalty": ["l2"],
                "classifier__solver": ["liblinear"],
            },
        },
        "DecisionTree": {
            "model": DecisionTreeClassifier(
                random_state=RANDOM_STATE, class_weight="balanced"
            ),
            "params": {
                "classifier__max_depth": [4],
                "classifier__min_samples_split": [40],
                "classifier__min_samples_leaf": [15],
                "classifier__max_features": [0.5],
            },
        },
        "RandomForest": {
            "model": RandomForestClassifier(
                random_state=RANDOM_STATE, class_weight="balanced"
            ),
            "params": {
                "classifier__n_estimators": [85],
                "classifier__max_depth": [3],
                "classifier__min_samples_split": [30],
                "classifier__min_samples_leaf": [10],
                "classifier__max_features": [0.5],
                "classifier__bootstrap": [True],
            },
        },
        "MLP": {
            "model": MLPClassifier(
                max_iter=2000, random_state=RANDOM_STATE, early_stopping=True
            ),
            "params": {
                "classifier__hidden_layer_sizes": [(128, 64, 32)],
                "classifier__alpha": [0.01],
                "classifier__activation": ["relu"],
            },
        },
        "SVM": {
            "model": SVC(
                probability=True, random_state=RANDOM_STATE, class_weight="balanced"
            ),
            "params": {
                "classifier__C": [2],
                "classifier__gamma": [0.025],
                "classifier__kernel": ["rbf"],
            },
        },
        "XGBoost": {
            "model": XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE),
            "params": {
                "classifier__n_estimators": [100],
                "classifier__max_depth": [2],
                "classifier__learning_rate": [0.1],
                "classifier__gamma": [0],
                "classifier__reg_alpha": [2.0],
                "classifier__reg_lambda": [10],
            },
        },
        "LightGBM": {
            "model": LGBMClassifier(
                random_state=RANDOM_STATE,
                verbose=-1,
                subsample_freq=1,
            ),
            "params": {
                "classifier__n_estimators": [95],
                "classifier__max_depth": [3],
                "classifier__learning_rate": [0.05],
                "classifier__num_leaves": [5],
                "classifier__min_child_samples": [100],
                "classifier__reg_lambda": [0.6],
            },
        },
        "GaussianNB": {
            "model": GaussianNB(),
            "params": {"classifier__var_smoothing": [1e-10]},
        },
        "KNN": {
            "model": KNeighborsClassifier(),
            "params": {"classifier__n_neighbors": [7]},
        },
    }


def focused_tuning_grids():
    decision_tree_tuples = [
        (2, 20, 40, None, 0.0),
        (2, 30, 60, None, 0.001),
        (2, 40, 80, 0.5, 0.001),
        (3, 10, 40, None, 0.0),
        (3, 20, 60, None, 0.001),
        (3, 30, 80, 0.5, 0.001),
        (3, 40, 100, 0.7, 0.001),
        (4, 10, 40, None, 0.0),
        (4, 20, 60, None, 0.001),
        (4, 30, 80, 0.5, 0.001),
        (4, 40, 100, 0.7, 0.002),
        (5, 20, 40, None, 0.001),
        (5, 30, 60, 0.5, 0.001),
        (5, 40, 80, 0.7, 0.002),
        (2, 10, 40, 0.5, 0.0),
        (3, 15, 50, 0.7, 0.001),
        (4, 15, 50, 0.5, 0.001),
        (5, 25, 60, 0.5, 0.002),
        (2, 25, 50, 0.7, 0.001),
        (3, 25, 50, None, 0.002),
    ]
    decision_tree = [
        {
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "min_samples_split": min_samples_split,
            "max_features": max_features,
            "ccp_alpha": ccp_alpha,
        }
        for (
            max_depth,
            min_samples_leaf,
            min_samples_split,
            max_features,
            ccp_alpha,
        ) in decision_tree_tuples
    ]

    random_forest = []
    for max_depth in [2, 3, 4]:
        for min_samples_leaf in [10, 20, 30, 40]:
            for max_features in ["sqrt", 0.5]:
                random_forest.append(
                    {
                        "n_estimators": 300,
                        "max_depth": max_depth,
                        "min_samples_leaf": min_samples_leaf,
                        "max_features": max_features,
                        "max_samples": 0.8,
                        "bootstrap": True,
                    }
                )

    xgb_keys = [
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "min_child_weight",
        "reg_alpha",
        "reg_lambda",
        "scale_pos_weight",
    ]
    xgb_tuples = [
        (30, 2, 0.03, 0.7, 0.5, 5, 2, 10, 0.67),
        (30, 2, 0.05, 0.8, 0.5, 10, 2, 10, 0.67),
        (30, 3, 0.02, 0.7, 0.5, 10, 5, 20, 0.67),
        (30, 3, 0.03, 0.8, 0.8, 20, 5, 20, 1.0),
        (50, 2, 0.02, 0.7, 0.5, 5, 1, 10, 1.0),
        (50, 2, 0.03, 0.8, 0.5, 10, 2, 10, 0.67),
        (50, 2, 0.05, 0.9, 0.8, 20, 5, 20, 1.0),
        (50, 3, 0.02, 0.7, 0.5, 10, 2, 20, 1.0),
        (50, 3, 0.03, 0.8, 0.8, 20, 5, 20, 0.67),
        (50, 3, 0.05, 0.9, 0.5, 10, 2, 10, 1.0),
        (80, 2, 0.02, 0.7, 0.8, 5, 1, 10, 0.67),
        (80, 2, 0.03, 0.8, 0.5, 10, 2, 20, 1.0),
        (80, 2, 0.05, 0.9, 0.8, 20, 5, 20, 0.67),
        (80, 3, 0.02, 0.7, 0.5, 10, 2, 10, 1.0),
        (80, 3, 0.03, 0.8, 0.8, 20, 5, 20, 1.0),
        (80, 3, 0.05, 0.9, 0.5, 20, 5, 20, 0.67),
        (30, 2, 0.02, 0.9, 0.8, 20, 5, 20, 1.0),
        (50, 2, 0.02, 0.7, 0.8, 5, 5, 20, 0.67),
        (80, 2, 0.03, 0.7, 0.5, 20, 5, 20, 1.0),
        (50, 3, 0.02, 0.9, 0.5, 5, 1, 10, 1.0),
        (30, 2, 0.03, 0.7, 0.8, 10, 5, 20, 1.0),
        (50, 2, 0.03, 0.9, 0.5, 20, 5, 20, 1.0),
        (80, 2, 0.02, 0.9, 0.8, 10, 2, 20, 0.67),
        (80, 3, 0.03, 0.7, 0.5, 20, 5, 20, 1.0),
    ]
    xgboost = [dict(zip(xgb_keys, values)) for values in xgb_tuples]

    knn = []
    for n_neighbors in [11, 15, 21, 25, 31, 41]:
        for weights in ["uniform", "distance"]:
            for metric in ["euclidean", "manhattan"]:
                knn.append(
                    {
                        "n_neighbors": n_neighbors,
                        "weights": weights,
                        "metric": metric,
                    }
                )

    return {
        "DecisionTree": decision_tree,
        "RandomForest": random_forest,
        "XGBoost": xgboost,
        "KNN": knn,
    }


def build_focused_fold_data(
    X_train,
    y_train,
    protected,
    radiomics_features,
    modality,
    cv,
):
    fold_data = []
    for fold_id, (train_idx, valid_idx) in enumerate(
        cv.split(X_train, y_train), start=1
    ):
        combat = None
        if modality != "clinical":
            combat = ComBatTransformer(
                radiomics_features=radiomics_features,
                protected_features=protected,
                batch_col="batch_model",
                min_batch_n=20,
            )
            combat.fit(X_train.iloc[train_idx], y_train[train_idx])
        train_features = (
            combat.transform(X_train.iloc[train_idx])
            if combat is not None
            else X_train.iloc[train_idx]
        )
        valid_features = (
            combat.transform(X_train.iloc[valid_idx])
            if combat is not None
            else X_train.iloc[valid_idx]
        )
        selector = StrictPreprocessSelector(
            k=MRMR_K,
            protected_features=protected,
            missing_threshold=MISSING_THRESHOLD,
            variance_threshold=VARIANCE_THRESHOLD,
        )
        selector.fit(train_features, y_train[train_idx])
        fold_data.append(
            {
                "fold": fold_id,
                "x_train": selector.transform(train_features),
                "y_train": y_train[train_idx],
                "x_valid": selector.transform(valid_features),
                "y_valid": y_train[valid_idx],
            }
        )
    return fold_data


def score_focused_trial(base_model, params, fold_data):
    train_aucs = []
    cv_aucs = []
    train_briers = []
    cv_briers = []
    record = {
        "Mean_Train_AUC": np.nan,
        "Mean_CV_AUC": np.nan,
        "Train_CV_Gap": np.nan,
        "SD_CV_AUC": np.nan,
        "Mean_Train_Brier": np.nan,
        "Mean_CV_Brier": np.nan,
        "Parameters": json.dumps(params, ensure_ascii=False),
    }
    for fold in fold_data:
        model = clone(base_model).set_params(**params)
        model.fit(fold["x_train"], fold["y_train"])
        train_prob = model.predict_proba(fold["x_train"])[:, 1]
        valid_prob = model.predict_proba(fold["x_valid"])[:, 1]
        train_aucs.append(roc_auc_score(fold["y_train"], train_prob))
        cv_aucs.append(roc_auc_score(fold["y_valid"], valid_prob))
        train_briers.append(brier_score_loss(fold["y_train"], train_prob))
        cv_briers.append(brier_score_loss(fold["y_valid"], valid_prob))
    record["Mean_Train_AUC"] = float(np.mean(train_aucs))
    record["Mean_CV_AUC"] = float(np.mean(cv_aucs))
    record["Train_CV_Gap"] = record["Mean_Train_AUC"] - record["Mean_CV_AUC"]
    record["SD_CV_AUC"] = float(np.std(cv_aucs, ddof=1))
    record["Mean_Train_Brier"] = float(np.mean(train_briers))
    record["Mean_CV_Brier"] = float(np.mean(cv_briers))
    return record


def run_focused_four_model_retuning(
    X_train,
    X_valid,
    y_train,
    y_valid,
    protected,
    radiomics_features,
    modality,
    cv,
    base_configs,
    best_models,
    best_params,
    selection_scores,
    train_probs,
    valid_probs,
    train_results,
    valid_results,
    output_dir,
):
    grids = focused_tuning_grids()
    fold_data = build_focused_fold_data(
        X_train,
        y_train,
        protected,
        radiomics_features,
        modality,
        cv,
    )
    trials = []

    for model_name in FOCUSED_RETUNE_MODELS:
        print(f"Focused retuning {model_name}: {len(grids[model_name])} trials")
        base_model = base_configs[model_name]["model"]
        for trial_index, params in enumerate(grids[model_name], start=1):
            record = score_focused_trial(base_model, params, fold_data)
            record["Model"] = model_name
            record["Trial"] = trial_index
            trials.append(record)
            print(
                f"  {trial_index:02d}: CV AUC={record['Mean_CV_AUC']:.4f}, "
                f"gap={record['Train_CV_Gap']:.4f}"
            )

    trials_df = pd.DataFrame(trials)
    full_combat = None
    if modality != "clinical":
        full_combat = ComBatTransformer(
            radiomics_features=radiomics_features,
            protected_features=protected,
            batch_col="batch_model",
            min_batch_n=20,
        )
        full_combat.fit(X_train, y_train)
        X_train_combat = full_combat.transform(X_train)
        X_valid_combat = full_combat.transform(X_valid)
    else:
        X_train_combat = X_train
        X_valid_combat = X_valid
    full_selector = StrictPreprocessSelector(
        k=MRMR_K,
        protected_features=protected,
        missing_threshold=MISSING_THRESHOLD,
        variance_threshold=VARIANCE_THRESHOLD,
    )
    full_selector.fit(X_train_combat, y_train)
    X_train_final = full_selector.transform(X_train_combat)

    comparison_rows = []
    fresh_configs = model_configs()
    for model_name in FOCUSED_RETUNE_MODELS:
        old_train = train_results[model_name]
        old_valid = valid_results[model_name]
        current_params = {
            key.replace("classifier__", ""): value
            for key, value in best_params[model_name].items()
        }
        current_cv = score_focused_trial(
            fresh_configs[model_name]["model"],
            current_params,
            fold_data,
        )
        old_cv = float(current_cv["Mean_CV_AUC"])
        candidates = trials_df.loc[trials_df["Model"] == model_name].copy()


        eligible = candidates.loc[
            (candidates["Mean_CV_AUC"] >= old_cv + 0.002)
            & (candidates["Train_CV_Gap"] <= 0.12)
        ].copy()

        accept_new = not eligible.empty
        if accept_new:
            best_cv = float(eligible["Mean_CV_AUC"].max())
            cv_band = eligible.loc[
                eligible["Mean_CV_AUC"] >= best_cv - 0.01
            ].copy()
            cv_band["Training_Only_Score"] = (
                cv_band["Mean_CV_AUC"]
                - 0.5 * cv_band["Train_CV_Gap"].clip(lower=0)
            )
            best = cv_band.sort_values(
                "Training_Only_Score", ascending=False
            ).iloc[0]
            params = json.loads(best["Parameters"])
            classifier = clone(base_configs[model_name]["model"]).set_params(
                **params
            )
            classifier.fit(X_train_final, y_train)
            pipeline_steps = []
            if full_combat is not None:
                pipeline_steps.append(
                    ("combat", copy.deepcopy(full_combat))
                )
            pipeline_steps.extend(
                [
                    ("selector", copy.deepcopy(full_selector)),
                    ("classifier", classifier),
                ]
            )
            focused_pipeline = Pipeline(pipeline_steps)
            best_models[model_name] = focused_pipeline
            joblib.dump(
                focused_pipeline, output_dir / f"model_{model_name}.pkl"
            )
            best_params[model_name] = {
                f"classifier__{key}": value
                for key, value in params.items()
            }
            selection_scores[model_name] = float(best["Mean_CV_AUC"])
            train_probs[model_name] = focused_pipeline.predict_proba(X_train)[
                :, 1
            ]
            train_results[model_name] = metric_dict(
                y_train, train_probs[model_name]
            )
            selected_cv = float(best["Mean_CV_AUC"])
            selected_gap = float(best["Train_CV_Gap"])
            selected_params = params
            selected_source = "retuned"
        else:
            selected_cv = old_cv
            selected_gap = float(current_cv["Train_CV_Gap"])
            selected_params = current_params
            selected_source = "current"

        comparison_rows.append(
            {
                "Model": model_name,
                "Old_CV_AUC": old_cv,
                "Selected_CV_AUC": selected_cv,
                "Delta_CV_AUC": selected_cv - old_cv,
                "Selected_Train_CV_Gap": selected_gap,
                "Old_Train_AUC": old_train["AUC"],
                "Old_Valid_AUC": old_valid["AUC"],
                "Old_Train_Valid_Gap": old_train["AUC"] - old_valid["AUC"],
                "Selected_Train_AUC": train_results[model_name]["AUC"],
                "Selected_Valid_AUC": np.nan,
                "Selected_Train_Valid_Gap": np.nan,
                "Old_Valid_Brier": old_valid["Brier"],
                "Selected_Valid_Brier": np.nan,
                "Accept_New": bool(accept_new),
                "Selected_Source": selected_source,
                "Selected_Parameters": json.dumps(
                    selected_params, ensure_ascii=False
                ),
            }
        )


    X_valid_final = full_selector.transform(X_valid_combat)
    for model_name in FOCUSED_RETUNE_MODELS:
        valid_probs[model_name] = best_models[model_name].predict_proba(
            X_valid
        )[:, 1]
        valid_results[model_name] = metric_dict(
            y_valid, valid_probs[model_name]
        )

    comparison = pd.DataFrame(comparison_rows)
    comparison["Selected_Valid_AUC"] = comparison["Model"].map(
        {name: valid_results[name]["AUC"] for name in FOCUSED_RETUNE_MODELS}
    )
    comparison["Selected_Valid_Brier"] = comparison["Model"].map(
        {name: valid_results[name]["Brier"] for name in FOCUSED_RETUNE_MODELS}
    )
    comparison["Selected_Train_Valid_Gap"] = (
        comparison["Selected_Train_AUC"] - comparison["Selected_Valid_AUC"]
    )


    validated_trials = []
    for record in trials:
        params = json.loads(record["Parameters"])
        model = clone(base_configs[record["Model"]]["model"]).set_params(**params)
        model.fit(X_train_final, y_train)
        train_prob = model.predict_proba(X_train_final)[:, 1]
        valid_prob = model.predict_proba(X_valid_final)[:, 1]
        validated_trials.append(
            {
                **record,
                "Full_Train_AUC": roc_auc_score(y_train, train_prob),
                "Valid_AUC": roc_auc_score(y_valid, valid_prob),
                "Train_Valid_Gap": roc_auc_score(y_train, train_prob)
                - roc_auc_score(y_valid, valid_prob),
                "Valid_Brier": brier_score_loss(y_valid, valid_prob),
            }
        )
    validated_df = pd.DataFrame(validated_trials)

    trials_df.to_excel(
        output_dir / "13_Focused_Retuning_Trials.xlsx", index=False
    )
    validated_df.to_excel(
        output_dir / "13_Focused_Retuning_Trials_with_Validation.xlsx",
        index=False,
    )
    comparison.to_excel(
        output_dir / "13_Focused_Retuning_Comparison.xlsx", index=False
    )
    return comparison


def icc_a1(ratings):
    n, k = ratings.shape
    grand_mean = ratings.values.mean()
    sst = np.sum((ratings.values - grand_mean) ** 2)
    ssr = k * np.sum((ratings.mean(axis=1) - grand_mean) ** 2)
    ssc = n * np.sum((ratings.mean(axis=0) - grand_mean) ** 2)
    sse = max(sst - ssr - ssc, 0.0)
    msr = ssr / (n - 1) if n > 1 else 0.0
    msc = ssc / (k - 1) if k > 1 else 0.0
    mse = sse / ((n - 1) * (k - 1)) if (n - 1) * (k - 1) > 0 else 0.0
    denominator = msr + (k - 1) * mse + k * (msc - mse) / n
    return (msr - mse) / denominator if denominator else np.nan


def write_icc_audit(valid_df, radiomics_features, output_dir):
    sample_dir = OUTPUT_ROOT / "Reproducibility"
    icc = pd.read_excel(
        PIXEL_PREPROCESS_ROOT / "03_ICC_results_pixel_spacing_497.xlsx"
    )
    sample = pd.read_excel(
        sample_dir / "sampled_50_cases_radiomics.xlsx",
        usecols=["image"],
    )
    valid_keys = set(
        valid_df["image"]
        .astype(str)
        .str.replace(".png", "", regex=False)
        .str.lower()
    )
    sample_keys = set(
        sample["image"]
        .astype(str)
        .str.replace(".png", "", regex=False)
        .str.lower()
    )
    overlap_keys = sample_keys & valid_keys
    full_set = set(icc.loc[icc["pass_ICC"], "feature"])
    audit = {
        "icc_sample_n": int(len(icc)),
        "validation_n": int(len(valid_df)),
        "overlap_n": int(len(overlap_keys)),
        "overlap_images": sorted(overlap_keys),
        "full_sample_pass_n": len(full_set),
        "current_model_radiomics_n": len(radiomics_features),
        "current_set_matches_audit": set(radiomics_features) == full_set,
        "note": "ICC filtering used transformed R1/R2 masks and was not fitted using internal-validation outcomes.",
    }
    (output_dir / "00_ICC_Audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return audit


def plot_correlation_and_roc(
    selected_matrix, selected_names, train_probs, valid_probs, y_train, y_valid, output_dir
):
    corr = pd.DataFrame(selected_matrix, columns=selected_names).corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="viridis", linewidths=0.4)
    plt.title("Selected Feature Correlation")
    plt.tight_layout()
    plt.savefig(output_dir / "01_correlation_heatmap.svg", dpi=300)
    plt.close()

    fig = sns.clustermap(
        corr, cmap="viridis", linewidths=0.2, figsize=(10, 10)
    )
    fig.savefig(output_dir / "01_correlation_clustermap.svg", dpi=300)
    plt.close()

    for label, probabilities, y_true, filename in [
        ("Training", train_probs, y_train, "03_Train_ROC_all.svg"),
        ("Internal validation", valid_probs, y_valid, "05_Valid_ROC_all.svg"),
    ]:
        plt.figure(figsize=(8, 6))
        for name, prob in probabilities.items():
            fpr, tpr, _ = roc_curve(y_true, prob)
            auc = roc_auc_score(y_true, prob)
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate", fontsize=16)
        plt.ylabel("True Positive Rate", fontsize=16)
        plt.title(f"ROC Curves ({label})", fontsize=18)
        plt.legend(fontsize=13, loc="lower right")
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=300)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modality",
        choices=["combined", "clinical", "radiomics"],
        default="combined",
    )
    parser.add_argument(
        "--use-tuned",
        action="store_true",
        help="Use the retained combined-model baseline parameters.",
    )
    parser.add_argument(
        "--skip-focused-retune",
        action="store_true",
        help="Skip the built-in focused retuning of DecisionTree, RandomForest, XGBoost, and KNN.",
    )
    parser.add_argument(
        "--output-root",
        default=str(STRICT_ROOT),
        help="Optional root directory for model outputs.",
    )
    parser.add_argument(
        "--input-file",
        default=str(INPUT_FILE),
        help="Optional branch-specific modelling workbook.",
    )
    parser.add_argument(
        "--parameter-file",
        default=None,
        help=(
            "JSON mapping model names to fixed parameters produced by "
            "N.retune_imager_spacing_branches.py."
        ),
    )
    args = parser.parse_args()
    if args.use_tuned and args.parameter_file:
        parser.error("--use-tuned and --parameter-file are mutually exclusive")
    modality = args.modality
    output_dir = Path(args.output_root) / OUTPUT_LABELS[modality]
    output_dir.mkdir(parents=True, exist_ok=True)

    data, features, protected, radiomics_features = load_data(
        modality,
        args.input_file,
    )
    train_df, valid_df = train_test_split(
        data,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=data["label"],
    )
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    X_train = model_input(train_df, features, modality)
    X_valid = model_input(valid_df, features, modality)
    y_train = train_df["label"].to_numpy()
    y_valid = valid_df["label"].to_numpy()

    split_manifest = pd.concat(
        [
            train_df[["image", "label", "batch_model"]].assign(
                Dataset="Train"
            ),
            valid_df[["image", "label", "batch_model"]].assign(
                Dataset="Validation"
            ),
        ],
        ignore_index=True,
    )
    split_manifest.to_excel(output_dir / "00_Data_Split_Manifest.xlsx", index=False)
    icc_audit = write_icc_audit(valid_df, radiomics_features, output_dir)

    cv = StratifiedKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    configs = model_configs()
    fixed_parameter_source = None
    if args.parameter_file:
        fixed_parameter_source = Path(args.parameter_file).resolve()
        fixed_parameters = json.loads(
            fixed_parameter_source.read_text(encoding="utf-8")
        )
        missing_models = sorted(set(configs) - set(fixed_parameters))
        if missing_models:
            raise ValueError(
                f"Parameter file is missing models: {missing_models}"
            )
        for model_name, params in fixed_parameters.items():
            if model_name not in configs:
                continue
            estimator_params = dict(params)
            if model_name == "MLP" and isinstance(
                estimator_params.get("hidden_layer_sizes"),
                list,
            ):
                estimator_params["hidden_layer_sizes"] = tuple(
                    estimator_params["hidden_layer_sizes"]
                )
            configs[model_name]["params"] = {
                f"classifier__{key}": [value]
                for key, value in estimator_params.items()
            }
    if args.use_tuned and modality == "combined":
        for model_name, params in TUNED_COMBINED_PARAMS.items():
            configs[model_name]["model"].set_params(**params)


            for key, value in params.items():
                configs[model_name]["params"][f"classifier__{key}"] = [value]
    best_models = {}
    best_params = {}
    train_probs = {}
    valid_probs = {}
    train_results = {}
    valid_results = {}
    selection_scores = {}

    for name, config in configs.items():
        pipeline = build_model_pipeline(
            modality=modality,
            radiomics_features=radiomics_features,
            protected=protected,
            classifier=config["model"],
        )
        grid = GridSearchCV(
            pipeline,
            config["params"],
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            error_score="raise",
        )
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        best_models[name] = model
        best_params[name] = grid.best_params_
        selection_scores[name] = float(grid.best_score_)
        joblib.dump(model, output_dir / f"model_{name}.pkl")

        train_prob = model.predict_proba(X_train)[:, 1]
        valid_prob = model.predict_proba(X_valid)[:, 1]
        train_probs[name] = train_prob
        valid_probs[name] = valid_prob
        train_results[name] = metric_dict(y_train, train_prob)
        valid_results[name] = metric_dict(y_valid, valid_prob)
        print(
            f"{name:20s} train AUC={train_results[name]['AUC']:.3f} "
            f"valid AUC={valid_results[name]['AUC']:.3f}"
        )

    if (
        modality == "combined"
        and not args.skip_focused_retune
        and fixed_parameter_source is None
    ):
        focused_comparison = run_focused_four_model_retuning(
            X_train=X_train,
            X_valid=X_valid,
            y_train=y_train,
            y_valid=y_valid,
            protected=protected,
            radiomics_features=radiomics_features,
            modality=modality,
            cv=cv,
            base_configs=configs,
            best_models=best_models,
            best_params=best_params,
            selection_scores=selection_scores,
            train_probs=train_probs,
            valid_probs=valid_probs,
            train_results=train_results,
            valid_results=valid_results,
            output_dir=output_dir,
        )
        print("\nFocused retuning comparison:")
        print(
            focused_comparison[
                [
                    "Model",
                    "Old_CV_AUC",
                    "Selected_CV_AUC",
                    "Selected_Train_CV_Gap",
                    "Old_Valid_AUC",
                    "Selected_Valid_AUC",
                    "Selected_Source",
                ]
            ].to_string(index=False)
        )

    train_perf = pd.DataFrame(train_results).T.add_prefix("Train_")
    valid_perf = pd.DataFrame(valid_results).T.add_prefix("Valid_")
    performance = train_perf.join(valid_perf)
    performance["Selection_CV_AUC"] = pd.Series(selection_scores)
    performance.to_excel(output_dir / "04_Model_Performance.xlsx")


    best_name = max(selection_scores, key=selection_scores.get)
    best_model = best_models[best_name]
    final_selector = best_model.named_steps["selector"]
    final_combat = best_model.named_steps.get("combat")
    final_features = list(final_selector.selected_features_)


    fold_records = []
    fold_feature_sets = []
    for fold, (train_idx, _) in enumerate(cv.split(X_train, y_train), start=1):
        combat = None
        if final_combat is not None:
            combat = ComBatTransformer(
                radiomics_features=radiomics_features,
                protected_features=protected,
                batch_col="batch_model",
                min_batch_n=20,
            )
            combat.fit(X_train.iloc[train_idx], y_train[train_idx])
            fold_train = combat.transform(X_train.iloc[train_idx])
        else:
            fold_train = X_train.iloc[train_idx]
        selector = StrictPreprocessSelector(
            k=MRMR_K,
            protected_features=protected,
            missing_threshold=MISSING_THRESHOLD,
            variance_threshold=VARIANCE_THRESHOLD,
        )
        selector.fit(fold_train, y_train[train_idx])
        fold_feature_sets.append(set(selector.selected_features_))
        fold_records.append(
            {
                "Fold": fold,
                "n_training": len(train_idx),
                "n_input_features": len(features),
                "n_missing_kept": selector.n_missing_kept_,
                "n_variance_kept": selector.n_variance_kept_,
                "n_significant_univariate": selector.n_significant_,
                "n_selected_radiomics": selector.n_selected_radiomics_,
                "selected_features": "; ".join(selector.selected_features_),
            }
        )
    fold_details = pd.DataFrame(fold_records)
    fold_details.to_excel(output_dir / "03_Fold_Preprocessing_and_Selection.xlsx", index=False)

    frequency_rows = []
    for feature in features:
        count = sum(feature in selected for selected in fold_feature_sets)
        frequency_rows.append(
            {
                "Feature": feature,
                "Selected_in_folds": count,
                "Frequency": count / N_FOLDS,
                "Protected": feature in protected,
                "In_final_model": feature in final_features,
            }
        )
    frequency = pd.DataFrame(frequency_rows).sort_values(
        ["Selected_in_folds", "Feature"], ascending=[False, True]
    )
    frequency.to_excel(output_dir / "03_Selection_Frequency.xlsx", index=False)

    feature_records = []
    for name, model in best_models.items():
        selected = list(model.named_steps["selector"].selected_features_)
        feature_records.append(
            {
                "Model": name,
                "N_Features": len(selected),
                "Selected_Features": ", ".join(selected),
            }
        )
    pd.DataFrame(feature_records).to_excel(
        output_dir / "04b_Selected_Features.xlsx", index=False
    )

    pd.DataFrame(
        [
            {"Model": name, **{k: str(v) for k, v in params.items()}}
            for name, params in best_params.items()
        ]
    ).to_excel(output_dir / "04c_Best_Hyperparameters.xlsx", index=False)


    train_for_selector = (
        final_combat.transform(X_train)
        if final_combat is not None
        else X_train
    )
    selected_matrix = final_selector.transform(train_for_selector)
    plot_correlation_and_roc(
        selected_matrix,
        final_features,
        train_probs,
        valid_probs,
        y_train,
        y_valid,
        output_dir,
    )


    for name, model in best_models.items():
        for label, X, y in [
            ("Train", X_train, y_train),
            ("Valid", X_valid, y_valid),
        ]:
            probability = model.predict_proba(X)[:, 1]
            pred = binary_predictions_from_probability(probability)
            cm = confusion_matrix(y, pred, labels=[0, 1])
            reference = train_results[name] if label == "Train" else valid_results[name]
            tn, fp, fn, tp = cm.ravel()
            matrix_recall = tp / (tp + fn)
            matrix_specificity = tn / (tn + fp)
            if not (
                np.isclose(matrix_recall, reference["Recall"])
                and np.isclose(matrix_specificity, reference["Specificity"])
            ):
                raise RuntimeError(
                    f"{label} confusion matrix for {name} does not match "
                    "the probability-threshold performance metrics."
                )
            from sklearn.metrics import ConfusionMatrixDisplay
            ConfusionMatrixDisplay(cm).plot(cmap="Blues")
            plt.title(f"{label} Confusion Matrix - {name}")
            plt.savefig(output_dir / f"02_{label}_CM_{name}.svg", dpi=300)
            plt.close()


    valid_metrics = [
        "Valid_Accuracy",
        "Valid_F1",
        "Valid_Precision",
        "Valid_Recall",
        "Valid_AUC",
        "Valid_BalancedAcc",
        "Valid_MCC",
    ]
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.ravel()
    for i, metric in enumerate(valid_metrics):
        sns.barplot(x=valid_perf.index, y=valid_perf[metric], ax=axes[i])
        axes[i].set_title(f"Validation {metric}")
        axes[i].tick_params(axis="x", rotation=45)
    for i in range(len(valid_metrics), len(axes)):
        fig.delaxes(axes[i])
    plt.tight_layout()
    plt.savefig(output_dir / "06_Valid_Performance_Bar.svg", dpi=300)
    plt.close()

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    for i, metric in enumerate(["Accuracy", "F1", "Precision", "Recall", "AUC"]):
        comp = pd.DataFrame(
            {
                "Train": performance[f"Train_{metric}"],
                "Validation": performance[f"Valid_{metric}"],
            }
        )
        comp.plot(kind="bar", ax=axes[i])
        axes[i].set_title(f"Train vs Valid {metric}")
        axes[i].tick_params(axis="x", rotation=45)
    fig.delaxes(axes[5])
    plt.tight_layout()
    plt.savefig(output_dir / "07_Train_vs_Valid_Bar.svg", dpi=300)
    plt.close()


    explain_input = (
        final_combat.transform(X_train.iloc[:50])
        if final_combat is not None
        else X_train.iloc[:50]
    )
    X_explain = final_selector.transform(explain_input)
    if best_name in ["RandomForest", "XGBoost", "LightGBM", "DecisionTree"]:
        explainer = shap.TreeExplainer(best_model.named_steps["classifier"])
        shap_values = explainer.shap_values(X_explain)
    elif best_name == "LogisticRegression":
        explainer = shap.LinearExplainer(
            best_model.named_steps["classifier"], X_explain
        )
        shap_values = explainer.shap_values(X_explain)
    else:
        explainer = shap.KernelExplainer(
            best_model.named_steps["classifier"].predict_proba, X_explain[:10]
        )
        shap_values = explainer.shap_values(X_explain[:30])
        X_explain = X_explain[:30]
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_explain, feature_names=final_features, show=False
    )
    plt.title(f"SHAP Summary - {best_name}")
    plt.tight_layout()
    plt.savefig(output_dir / f"08_SHAP_Summary_{best_name}.svg", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_explain,
        feature_names=final_features,
        plot_type="bar",
        show=False,
    )
    plt.title(f"SHAP Feature Importance - {best_name}")
    plt.tight_layout()
    plt.savefig(output_dir / f"08_SHAP_Bar_{best_name}.svg", dpi=300)
    plt.close()


    y_prob = valid_probs[best_name]
    brier = brier_score_loss(y_valid, y_prob)
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Ideal")
    lowess_result = lowess(y_valid, y_prob, frac=0.8, it=3, return_sorted=True)
    plt.plot(
        lowess_result[:, 0],
        lowess_result[:, 1],
        "-",
        color="black",
        linewidth=2,
        label=f"Calibration ({best_name})",
    )
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Predicted Probability", fontsize=16)
    plt.ylabel("Actual Probability", fontsize=16)
    plt.title(f"Calibration Curve - {best_name} (Brier={brier:.3f})", fontsize=18)
    plt.legend(loc="lower right")
    plt.savefig(output_dir / f"09_Calibration_{best_name}.svg", dpi=300)
    plt.close()

    thresholds = np.arange(0, 1.01, 0.01)
    prevalence = float(np.mean(y_valid))
    nb_model = []
    nb_all = []
    for threshold in thresholds:
        if threshold >= 1.0:
            nb_model.append(0.0)
            nb_all.append(0.0)
            continue
        predicted = (y_prob >= threshold).astype(int)
        true_positive = np.sum((y_valid == 1) & (predicted == 1))
        false_positive = np.sum((y_valid == 0) & (predicted == 1))
        weight = threshold / (1 - threshold)
        nb_model.append(
            true_positive / len(y_valid)
            - false_positive / len(y_valid) * weight
        )
        nb_all.append(prevalence - (1 - prevalence) * weight)
    nb_model = np.asarray(nb_model)
    nb_all = np.asarray(nb_all)
    y_max = max(float(np.max(nb_model)), float(np.max(nb_all))) + 0.05

    plt.figure(figsize=(8, 8))
    plt.plot(thresholds, nb_model, "r-", linewidth=2, label=best_name)
    plt.plot(thresholds, nb_all, "gray", linestyle="--", label="Treat All")
    plt.plot(thresholds, np.zeros_like(thresholds), "k-", label="Treat None")
    plt.ylim(-0.05, y_max)
    plt.xlim(0, 1.0)
    plt.xlabel("Threshold Probability", fontsize=16)
    plt.ylabel("Net Benefit", fontsize=16)
    plt.title(f"Decision Curve Analysis - {best_name}", fontsize=18)
    plt.tick_params(labelsize=14)
    plt.legend(loc="upper right", fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.savefig(
        output_dir / f"10_DCA_{best_name}.svg",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


    names = list(valid_probs)
    p_matrix = np.ones((len(names), len(names)))
    raw_pairs = []
    positions = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            p_value = delong_pvalue(
                y_valid, valid_probs[names[i]], valid_probs[names[j]]
            )
            p_matrix[i, j] = p_value
            p_matrix[j, i] = p_value
            raw_pairs.append(p_value)
            positions.append((i, j))
    _, adjusted, _, _ = multipletests(raw_pairs, method="fdr_bh")
    adjusted_matrix = np.ones_like(p_matrix)
    for (i, j), value in zip(positions, adjusted):
        adjusted_matrix[i, j] = value
        adjusted_matrix[j, i] = value
    pd.DataFrame(p_matrix, index=names, columns=names).to_excel(
        output_dir / "11_DeLong_Pvalues.xlsx"
    )
    adjusted_df = pd.DataFrame(adjusted_matrix, index=names, columns=names)
    adjusted_df.to_excel(output_dir / "11_DeLong_Pvalues_FDRcorrected.xlsx")

    linkage_matrix = linkage(
        pdist(adjusted_matrix, metric="correlation"), method="average"
    )
    for filename, vmax, annotate in [
        ("12_DeLong_Heatmap.svg", 1.0, False),
        ("12_DeLong_Heatmap_zoom.svg", 0.25, False),
        ("12_DeLong_Heatmap_full.svg", 0.25, True),
    ]:
        annotation = None
        if annotate:
            annotation = np.empty_like(adjusted_matrix, dtype=object)
            for i in range(len(names)):
                for j in range(len(names)):
                    value = adjusted_matrix[i, j]
                    text = f"{value:.3f}"
                    if i != j and value < 0.001:
                        text += " ***"
                    elif i != j and value < 0.01:
                        text += " **"
                    elif i != j and value < 0.05:
                        text += " *"
                    annotation[i, j] = text
        grid = sns.clustermap(
            adjusted_matrix,
            row_linkage=linkage_matrix,
            col_linkage=linkage_matrix,
            cmap="viridis",
            vmin=0,
            vmax=vmax,
            xticklabels=names,
            yticklabels=names,
            annot=annotation,
            fmt="" if annotate else None,
            linewidths=0.5,
            linecolor="white",
            figsize=(9, 9),
            cbar_kws={"label": "FDR-corrected p-value"},
        )
        grid.fig.suptitle("DeLong Test", fontsize=20)
        grid.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()


    final_selector.univariate_results_.to_excel(
        output_dir / "03_Final_Training_Univariate_FDR.xlsx", index=False
    )
    preprocessing_order = []
    if modality != "clinical":
        preprocessing_order.append(
            "ComBat: batch-model harmonization fitted on each inner training "
            "fold; batches with fewer than 20 training samples were passed through"
        )
    preprocessing_order += [
        "Missing-rate filter: fitted on each inner training fold",
        "Mean imputation: means fitted on each inner training fold",
        "Constant-variance filter: threshold 0.0 in raw units, fitted within fold",
        "Z-score standardization: fitted within each inner training fold",
        "Mann-Whitney U + Benjamini-Hochberg FDR: fitted within fold",
    ]
    if modality != "clinical":
        preprocessing_order.append(
            "mRMR: fitted within fold, K=8 radiomics features"
        )
    preprocessing_order.append("Classifier: fitted within the same fold")

    pipeline_flow = {
        "pipeline": "strict_cv_internal",
        "modality": modality,
        "split": {
            "train_n": int(len(train_df)),
            "validation_n": int(len(valid_df)),
            "random_state": RANDOM_STATE,
            "stratified": True,
        },
        "order": preprocessing_order,
        "variance_threshold_note": "The previous 0.05 threshold after Z-score scaling was removed because it is not meaningful after standardization; only constant features are removed before scaling.",
        "validation_usage": "The 20% internal validation set was not used for preprocessing, feature selection, hyperparameter tuning, focused-retuning candidate ranking, or model replacement. All models were fixed before validation evaluation.",
        "model_selection": (
            "Fixed parameters were selected using training-only five-fold CV "
            "from the recorded full parameter search; the internal validation "
            "set was evaluated only after parameters were locked."
            if fixed_parameter_source
            else "Initial and focused classifiers are selected using training-only five-fold CV."
        ),
        "use_tuned_parameters": bool(args.use_tuned and modality == "combined"),
        "fixed_parameter_file": (
            str(fixed_parameter_source)
            if fixed_parameter_source is not None
            else None
        ),
        "focused_retuning": {
            "enabled": bool(
                modality == "combined"
                and not args.skip_focused_retune
                and fixed_parameter_source is None
            ),
            "models": FOCUSED_RETUNE_MODELS,
            "trials_file": "13_Focused_Retuning_Trials.xlsx",
            "validation_trials_file": "13_Focused_Retuning_Trials_with_Validation.xlsx",
            "comparison_file": "13_Focused_Retuning_Comparison.xlsx",
        },
        "selection_frequency_file": "03_Selection_Frequency.xlsx",
        "fold_details_file": "03_Fold_Preprocessing_and_Selection.xlsx",
        "icc_audit": icc_audit,
        "mrmr_applied": modality != "clinical",
        "mrmr_k": MRMR_K if modality != "clinical" else None,
        "combat": {
            "enabled": modality != "clinical",
            "batch_column": "batch_model",
            "minimum_training_batch_n": 20,
            "eligible_batches": (
                final_combat.eligible_batches_
                if final_combat is not None
                else []
            ),
            "passthrough_batches": (
                final_combat.passthrough_batches_
                if final_combat is not None
                else []
            ),
        },
        "final_features": final_features,
        "final_feature_source": "selector refitted on the full training set after hyperparameter selection",
    }
    (output_dir / "00_Pipeline_Flow.json").write_text(
        json.dumps(pipeline_flow, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    model_config = {
        "modality": modality,
        "pipeline_type": "strict_cv_internal",
        "model_names": list(best_models.keys()),
        "feature_order": features,
        "protected_features": protected,
        "model_selected_features": {
            name: list(model.named_steps["selector"].selected_features_)
            for name, model in best_models.items()
        },
        "selection_criterion": (
            "Fixed parameters from 14_Tuning_Best_Parameters.json, selected "
            "by training-only five-fold CV; validation metrics are "
            "post-selection only."
            if fixed_parameter_source
            else "Focused candidates selected by training-only CV AUC and train-CV gap; validation metrics are post-selection only."
        ),
        "selection_cv_auc": selection_scores,
        "best_model": best_name,
        "use_tuned_parameters": bool(args.use_tuned and modality == "combined"),
        "focused_retuning_enabled": bool(
            modality == "combined"
            and not args.skip_focused_retune
            and fixed_parameter_source is None
        ),
        "fixed_parameter_file": (
            str(fixed_parameter_source)
            if fixed_parameter_source is not None
            else None
        ),
        "focused_retuning_models": FOCUSED_RETUNE_MODELS,
        "final_features": final_features,
        "mrmr_applied": modality != "clinical",
        "mrmr_k": MRMR_K if modality != "clinical" else None,
        "required_input_columns": (
            features + ["batch_model"]
            if modality != "clinical"
            else features
        ),
        "combat_batch_column": (
            "batch_model" if modality != "clinical" else None
        ),
        "combat_eligible_batches": (
            final_combat.eligible_batches_
            if final_combat is not None
            else []
        ),
        "combat_passthrough_batches": (
            final_combat.passthrough_batches_
            if final_combat is not None
            else []
        ),
        "random_state": RANDOM_STATE,
        "n_folds": N_FOLDS,
        "icc_audit": icc_audit,
        "selection_frequency_file": "03_Selection_Frequency.xlsx",
        "fold_details_file": "03_Fold_Preprocessing_and_Selection.xlsx",
    }
    (output_dir / "model_config.json").write_text(
        json.dumps(model_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    joblib.dump(
        train_df[["image", "label", "batch_model"]],
        output_dir / "00_Train_Patient_Manifest.pkl",
    )
    joblib.dump(
        valid_df[["image", "label", "batch_model"]],
        output_dir / "00_Validation_Patient_Manifest.pkl",
    )
    print(f"\nBest model: {best_name}")
    print(f"Final features: {final_features}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
