from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import os

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, norm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from statsmodels.sandbox.stats.multicomp import multipletests


MRMR_K = 8
P_THRESHOLD = 0.05
MISSING_THRESHOLD = 0.30
VARIANCE_THRESHOLD = 0.0
RANDOM_STATE = 42


class ComBatTransformer(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        radiomics_features=None,
        protected_features=None,
        batch_col="batch_model",
        min_batch_n=20,
        eb=True,
        parametric=True,
    ):
        self.radiomics_features = radiomics_features
        self.protected_features = protected_features
        self.batch_col = batch_col
        self.min_batch_n = min_batch_n
        self.eb = eb
        self.parametric = parametric

    def _as_frame(self, X):
        if isinstance(X, pd.DataFrame):
            return X.copy()
        return pd.DataFrame(X, columns=self.feature_names_in_)

    def fit(self, X, y=None):
        from neuroCombat import neuroCombat

        frame = self._as_frame(X)
        self.feature_names_in_ = list(frame.columns)
        if self.batch_col not in frame:
            raise ValueError(f"Missing batch column: {self.batch_col}")
        self.radiomics_features_ = [
            feature
            for feature in (self.radiomics_features or [])
            if feature in frame.columns
        ]
        if not self.radiomics_features_:
            raise ValueError("No radiomics features supplied to ComBat")
        self.protected_features_ = [
            feature
            for feature in (self.protected_features or [])
            if feature in frame.columns
        ]
        self.output_features_ = [
            feature
            for feature in self.feature_names_in_
            if feature != self.batch_col
        ]

        batch = frame[self.batch_col].astype(str)
        counts = batch.value_counts()
        self.training_batch_counts_ = {
            str(name): int(value) for name, value in counts.items()
        }
        self.eligible_batches_ = sorted(
            name for name, value in counts.items() if value >= self.min_batch_n
        )
        self.passthrough_batches_ = sorted(
            name for name, value in counts.items() if value < self.min_batch_n
        )
        eligible = batch.isin(self.eligible_batches_).to_numpy()
        self.harmonized_n_ = int(eligible.sum())
        self.passthrough_n_ = int((~eligible).sum())
        if self.harmonized_n_ == 0:
            self.estimates_ = None
            return self
        if len(self.eligible_batches_) < 2:
            self.estimates_ = None
            self.eligible_batches_ = []
            self.harmonized_n_ = 0
            self.passthrough_n_ = len(frame)
            return self

        combat_data = frame.loc[
            eligible, self.radiomics_features_
        ].astype(float).T
        covars = pd.DataFrame({"batch": batch.loc[eligible].to_numpy()})
        result = neuroCombat(
            dat=combat_data,
            covars=covars,
            batch_col="batch",
            eb=self.eb,
            parametric=self.parametric,
        )
        self.estimates_ = result["estimates"]
        return self

    def transform(self, X):
        from neuroCombat import neuroCombatFromTraining

        frame = self._as_frame(X)
        if self.estimates_ is None:
            return frame[self.output_features_].copy()
        batch = frame[self.batch_col].astype(str)
        known_batches = {str(value) for value in self.estimates_["batches"]}
        harmonize = batch.isin(known_batches).to_numpy()
        output = frame.copy()
        output[self.radiomics_features_] = output[
            self.radiomics_features_
        ].astype(float)
        self.last_transform_harmonized_n_ = int(harmonize.sum())
        self.last_transform_passthrough_n_ = int((~harmonize).sum())
        if harmonize.any():
            combat_data = frame.loc[
                harmonize, self.radiomics_features_
            ].astype(float).T
            result = neuroCombatFromTraining(
                dat=combat_data,
                batch=batch.loc[harmonize].to_numpy(),
                estimates=self.estimates_,
            )
            output.loc[
                harmonize, self.radiomics_features_
            ] = result["data"].T
        return output[self.output_features_].copy()


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
        import pymrmr

        frame = self._as_frame(X)
        self.feature_names_in_ = list(frame.columns)
        protected = [
            feature
            for feature in (self.protected_features or [])
            if feature in frame.columns
        ]
        self.protected_features_ = protected

        missing_ratio = frame.isna().mean()
        self.missing_keep_ = [
            feature
            for feature in self.feature_names_in_
            if missing_ratio[feature] <= self.missing_threshold
            or feature in protected
        ]
        self.impute_means_ = frame[self.missing_keep_].mean()
        imputed = frame[self.missing_keep_].fillna(self.impute_means_)

        raw_variances = imputed.var()
        self.variance_keep_ = [
            feature
            for feature in self.missing_keep_
            if feature in protected
            or raw_variances[feature] > self.variance_threshold
        ]
        imputed = imputed[self.variance_keep_]

        self.scaler_ = StandardScaler()
        scaled = pd.DataFrame(
            self.scaler_.fit_transform(imputed),
            columns=self.variance_keep_,
            index=frame.index,
        )

        candidates = [
            feature
            for feature in self.variance_keep_
            if feature not in protected
        ]
        self.univariate_results_ = pd.DataFrame(
            columns=["feature", "p_raw", "p_adj"]
        )
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
                        group0,
                        group1,
                        alternative="two-sided",
                    ).pvalue
                else:
                    p_raw = 1.0
                rows.append({"feature": feature, "p_raw": p_raw})

            results = pd.DataFrame(rows)
            _, p_adj, _, _ = multipletests(
                results["p_raw"],
                method="fdr_bh",
            )
            results["p_adj"] = p_adj
            self.univariate_results_ = results
            self.significant_features_ = results.loc[
                results["p_adj"] < self.p_threshold,
                "feature",
            ].tolist()
            if not self.significant_features_:
                self.significant_features_ = results.nsmallest(
                    min(5, len(results)),
                    "p_raw",
                )["feature"].tolist()

            n_select = min(self.k, len(self.significant_features_))
            if n_select > 0:
                mrmr_input = scaled[self.significant_features_].copy()
                mrmr_input["label"] = np.asarray(y)
                try:
                    with open(
                        os.devnull,
                        "w",
                    ) as devnull, redirect_stdout(
                        devnull
                    ), redirect_stderr(
                        devnull
                    ):
                        n_request = min(
                            len(self.significant_features_),
                            self.k + 2,
                        )
                        mrmr_result = pymrmr.mRMR(
                            mrmr_input,
                            "MIQ",
                            n_request,
                        )
                    self.mrmr_features_ = [
                        feature
                        for feature in mrmr_result
                        if feature != "label"
                        and feature in candidates
                    ][:n_select]
                    if len(self.mrmr_features_) < n_select:
                        for feature in self.significant_features_:
                            if feature not in self.mrmr_features_:
                                self.mrmr_features_.append(feature)
                            if len(self.mrmr_features_) == n_select:
                                break
                except Exception:
                    self.mrmr_features_ = self.significant_features_[
                        :n_select
                    ]

        self.selected_features_ = list(
            dict.fromkeys(protected + self.mrmr_features_)
        )
        self.n_missing_kept_ = len(self.missing_keep_)
        self.n_variance_kept_ = len(self.variance_keep_)
        self.n_significant_ = len(self.significant_features_)
        self.n_selected_radiomics_ = len(
            [
                feature
                for feature in self.selected_features_
                if feature not in protected
            ]
        )
        return self

    def transform(self, X):
        frame = self._as_frame(X)
        imputed = frame[self.missing_keep_].fillna(self.impute_means_)
        imputed = imputed[self.variance_keep_]
        scaled = self.scaler_.transform(imputed)
        indices = [
            self.variance_keep_.index(feature)
            for feature in self.selected_features_
        ]
        return scaled[:, indices]


def compute_midrank(values):
    order = np.argsort(values)
    sorted_values = values[order]
    n = len(values)
    ranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    result = np.empty(n, dtype=float)
    result[order] = ranks
    return result


def fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    negative = predictions_sorted_transposed[:, :n]
    positive = predictions_sorted_transposed[:, n:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty([k, n])
    ty = np.empty([k, m])
    tz = np.empty([k, m + n])
    for row in range(k):
        tx[row, :] = compute_midrank(negative[row, :])
        ty[row, :] = compute_midrank(positive[row, :])
        tz[row, :] = compute_midrank(
            predictions_sorted_transposed[row, :]
        )
    aucs = (
        np.sum(tz[:, n:], axis=1) / (m * n)
        - (m + 1.0) / (2.0 * n)
    )
    v01 = (tz[:, n:] - ty[:, :]) / n
    v10 = 1.0 - (tz[:, :n] - tx[:, :]) / m
    covariance = np.cov(v01) / m + np.cov(v10) / n
    return aucs, covariance


def delong_pvalue(y_true, prob1, prob2):
    y_true = np.asarray(y_true)
    order = np.argsort(y_true)
    probabilities = np.vstack(
        [
            np.asarray(prob1)[order],
            np.asarray(prob2)[order],
        ]
    )
    aucs, covariance = fast_delong(
        probabilities,
        int(np.sum(y_true[order] == 1)),
    )
    contrast = np.array([[1, -1]])
    z_value = np.abs(np.diff(aucs)) / np.sqrt(
        contrast @ covariance @ contrast.T
    )
    return float(2 * (1 - norm.cdf(z_value))[0, 0])
