from __future__ import annotations

import json

import numpy as np
from sklearn.model_selection import ParameterGrid, ParameterSampler


RANDOM_STATE = 42


TUNING_SPACES = {
    "LogisticRegression": {
        "C": [0.03, 0.1, 0.3, 1.0, 3.0],
        "penalty": ["l1", "l2", "elasticnet"],
        "solver": ["liblinear", "saga", "newton-cg"],
        "l1_ratio": [0.2, 0.5, 0.8],
    },
    "DecisionTree": {
        "max_depth": [2, 3, 4, 5, None],
        "min_samples_split": [10, 20, 40, 80, 120],
        "min_samples_leaf": [5, 10, 20, 40, 60],
        "max_features": [None, "sqrt", 0.5, 0.7],
        "ccp_alpha": [0.0, 0.0005, 0.001, 0.002, 0.005],
    },
    "RandomForest": {
        "n_estimators": [100, 200, 300, 400],
        "max_depth": [2, 3, 4, 5, None],
        "min_samples_split": [10, 20, 40, 80],
        "min_samples_leaf": [5, 10, 20, 40],
        "max_features": ["sqrt", 0.3, 0.5, 0.7],
        "max_samples": [0.6, 0.8, 1.0],
    },
    "MLP": {
        "hidden_layer_sizes": [
            (64,),
            (128,),
            (128, 64),
            (128, 64, 32),
            (256, 128, 64),
        ],
        "alpha": [0.00001, 0.0001, 0.001, 0.01, 0.1],
        "activation": ["tanh", "relu", "logistic"],
        "learning_rate_init": [0.0005, 0.001, 0.003],
        "batch_size": [32, 64, 128],
    },
    "SVM": {
        "C": [0.1, 0.5, 1.0, 2.0, 5.0],
        "gamma": ["scale", 0.01, 0.025, 0.05, 0.1],
        "kernel": ["rbf", "linear", "poly"],
        "degree": [2, 3, 4],
    },
    "XGBoost": {
        "n_estimators": [50, 100, 150, 200, 300],
        "max_depth": [2, 3, 4, 5, 6],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.5, 0.7, 1.0],
        "min_child_weight": [1, 5, 10, 20, 40],
        "reg_alpha": [0.0, 0.1, 1.0, 5.0, 10.0],
        "reg_lambda": [0.1, 1.0, 5.0, 10.0, 20.0],
        "gamma": [0.0, 0.1, 0.5, 1.0],
        "scale_pos_weight": [0.67, 1.0, 1.5],
    },
    "LightGBM": {
        "n_estimators": [50, 100, 150, 200, 300],
        "max_depth": [2, 3, 4, 5, 6],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "num_leaves": [5, 7, 15, 31],
        "min_child_samples": [20, 50, 100, 150],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.5, 0.7, 1.0],
        "reg_lambda": [0.1, 0.6, 5.0, 10.0],
    },
    "GaussianNB": {
        "var_smoothing": [1e-12, 1e-10, 1e-8, 1e-6, 1e-4],
    },
    "KNN": {
        "n_neighbors": [5, 11, 21, 31, 41],
        "metric": ["euclidean", "manhattan", "chebyshev"],
    },
}

TARGET_TRIAL_COUNTS = {
    "LogisticRegression": 30,
    "DecisionTree": 60,
    "RandomForest": 60,
    "MLP": 50,
    "SVM": 60,
    "XGBoost": 80,
    "LightGBM": 70,
    "GaussianNB": 5,
    "KNN": 15,
}

DEFAULT_PARAMETERS = {
    "LogisticRegression": {
        "C": 0.3,
        "penalty": "l2",
        "solver": "saga",
        "l1_ratio": 0.5,
    },
    "DecisionTree": {
        "max_depth": 3,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "max_features": None,
        "ccp_alpha": 0.001,
    },
    "RandomForest": {
        "n_estimators": 200,
        "max_depth": 3,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "max_features": "sqrt",
        "max_samples": 0.8,
    },
    "MLP": {
        "hidden_layer_sizes": (128, 64),
        "alpha": 0.001,
        "activation": "relu",
        "learning_rate_init": 0.001,
        "batch_size": 64,
    },
    "SVM": {
        "C": 1.0,
        "gamma": "scale",
        "kernel": "rbf",
        "degree": 3,
    },
    "XGBoost": {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "min_child_weight": 10,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "gamma": 0.1,
        "scale_pos_weight": 1.0,
    },
    "LightGBM": {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.03,
        "num_leaves": 7,
        "min_child_samples": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_lambda": 0.6,
    },
    "GaussianNB": {"var_smoothing": 1e-10},
    "KNN": {"n_neighbors": 21, "metric": "euclidean"},
}


def _value_key(value) -> str:
    return json.dumps(normalise_value(value), sort_keys=True)


def normalise_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [normalise_value(item) for item in value]
    if isinstance(value, list):
        return [normalise_value(item) for item in value]
    return value


def normalise_parameters(parameters: dict) -> dict:
    return {
        key: normalise_value(value)
        for key, value in parameters.items()
    }


def estimator_parameters(parameters: dict) -> dict:
    result = dict(parameters)
    hidden = result.get("hidden_layer_sizes")
    if isinstance(hidden, list):
        result["hidden_layer_sizes"] = tuple(hidden)
    return result


def _valid_logistic_candidate(candidate: dict) -> bool:
    solver = candidate["solver"]
    penalty = candidate["penalty"]
    if solver == "liblinear":
        return penalty in {"l1", "l2"}
    if solver == "saga":
        return penalty in {"l1", "l2", "elasticnet"}
    if solver == "newton-cg":
        return penalty == "l2"
    return False


def _valid_candidate(model_name: str, candidate: dict) -> bool:
    if model_name == "LogisticRegression":
        return _valid_logistic_candidate(candidate)
    return True


def _coverage_candidates(model_name: str) -> list[dict]:
    default = DEFAULT_PARAMETERS[model_name]
    candidates = [normalise_parameters(default)]
    for parameter, values in TUNING_SPACES[model_name].items():
        for value in values:
            candidate = dict(default)
            candidate[parameter] = value
            candidate = normalise_parameters(candidate)
            if _valid_candidate(model_name, candidate):
                candidates.append(candidate)
    return candidates


def sample_parameter_candidates(
    model_name: str,
    random_state: int = RANDOM_STATE,
) -> list[dict]:
    space = TUNING_SPACES[model_name]
    target = TARGET_TRIAL_COUNTS[model_name]
    candidates = []
    seen = set()

    for candidate in _coverage_candidates(model_name):
        key = _value_key(candidate)
        if key not in seen:
            seen.add(key)
            candidates.append(candidate)

    sampler = ParameterSampler(
        space,
        n_iter=target * 3,
        random_state=random_state,
    )
    for candidate in sampler:
        candidate = normalise_parameters(candidate)
        if not _valid_candidate(model_name, candidate):
            continue
        key = _value_key(candidate)
        if key not in seen:
            seen.add(key)
            candidates.append(candidate)
        if len(candidates) >= target:
            break

    if len(candidates) < target:
        for candidate in ParameterGrid(space):
            candidate = normalise_parameters(candidate)
            if not _valid_candidate(model_name, candidate):
                continue
            key = _value_key(candidate)
            if key not in seen:
                seen.add(key)
                candidates.append(candidate)
            if len(candidates) >= target:
                break
    return candidates[:target]


def parameter_space_records() -> list[dict]:
    rows = []
    for model_name, space in TUNING_SPACES.items():
        candidates = sample_parameter_candidates(model_name)
        for parameter, values in space.items():
            sampled_values = {
                _value_key(candidate[parameter])
                for candidate in candidates
            }
            rows.append(
                {
                    "Model": model_name,
                    "Parameter": parameter,
                    "Allowed_Values": json.dumps(
                        [normalise_value(value) for value in values],
                        ensure_ascii=False,
                    ),
                    "N_Allowed_Values": len(values),
                    "N_Sampled_Values": len(sampled_values),
                    "All_Allowed_Values_Sampled": (
                        len(sampled_values) == len(values)
                    ),
                    "Target_Candidate_Count": TARGET_TRIAL_COUNTS[model_name],
                }
            )
    return rows


def validate_parameter_spaces() -> None:
    for model_name, space in TUNING_SPACES.items():
        for parameter, values in space.items():
            if not 3 <= len(values) <= 5:
                raise ValueError(
                    f"{model_name}.{parameter} has {len(values)} values; "
                    "expected 3-5"
                )
        candidates = sample_parameter_candidates(model_name)
        for parameter, values in space.items():
            sampled = {_value_key(candidate[parameter]) for candidate in candidates}
            if len(sampled) != len(values):
                raise ValueError(
                    f"{model_name}.{parameter} did not cover all configured "
                    "values in the sampled candidates"
                )
