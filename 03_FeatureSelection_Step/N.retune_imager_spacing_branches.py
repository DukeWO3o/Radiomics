from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

import imager_spacing_tuning as tuning


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
from pipeline_paths import (
    DEFAULT_PIXEL_MODEL_DATA,
    PIXEL_COMBAT_ROOT,
    PIXEL_NO_COMBAT_ROOT,
)

COMBAT_SCRIPT = SCRIPT_DIR / "I.refactored_pipeline_strict_cv_combat.py"
NO_COMBAT_SCRIPT = (
    SCRIPT_DIR / "I.refactored_pipeline_strict_cv_imager_no_combat.py"
)
PIXEL_DATA_FILE = DEFAULT_PIXEL_MODEL_DATA
RANDOM_STATE = 42
N_FOLDS = 5
MAX_TRAIN_CV_GAP = 0.12
CV_BAND = 0.01

BRANCHES = {
    "pixel_no_combat": {
        "script": NO_COMBAT_SCRIPT,
        "root": PIXEL_NO_COMBAT_ROOT,
        "combat_enabled": False,
        "input_file": PIXEL_DATA_FILE,
    },
    "pixel_combat": {
        "script": COMBAT_SCRIPT,
        "root": PIXEL_COMBAT_ROOT,
        "combat_enabled": True,
        "input_file": PIXEL_DATA_FILE,
    },
}


def load_branch_module(name: str):
    settings = BRANCHES[name]
    module_name = f"_imager_spacing_tuning_{name}"
    spec = importlib.util.spec_from_file_location(
        module_name,
        settings["script"],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load branch script: {settings['script']}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def build_fold_data(
    module,
    X_train,
    y_train,
    protected,
    radiomics_features,
    combat_enabled,
):
    cv = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    folds = []
    for fold_id, (train_index, valid_index) in enumerate(
        cv.split(X_train, y_train),
        start=1,
    ):
        x_train_fold = X_train.iloc[train_index]
        x_valid_fold = X_train.iloc[valid_index]
        combat = None
        if combat_enabled:
            combat = module.ComBatTransformer(
                radiomics_features=radiomics_features,
                protected_features=protected,
                batch_col="batch_model",
                min_batch_n=20,
            )
            combat.fit(x_train_fold, y_train[train_index])
            x_train_fold = combat.transform(x_train_fold)
            x_valid_fold = combat.transform(x_valid_fold)

        selector = module.StrictPreprocessSelector(
            k=module.MRMR_K,
            protected_features=protected,
            missing_threshold=module.MISSING_THRESHOLD,
            variance_threshold=module.VARIANCE_THRESHOLD,
        )
        selector.fit(x_train_fold, y_train[train_index])
        folds.append(
            {
                "fold": fold_id,
                "x_train": selector.transform(x_train_fold),
                "y_train": y_train[train_index],
                "x_valid": selector.transform(x_valid_fold),
                "y_valid": y_train[valid_index],
                "selected_features": list(selector.selected_features_),
            }
        )
    return folds


def branch_model_input(module, frame, features, combat_enabled):
    if hasattr(module, "model_input"):
        return module.model_input(frame, features, "combined")
    columns = list(features)
    if combat_enabled:
        columns.append("batch_model")
    return frame[columns].copy()


def build_branch_pipeline(
    module,
    radiomics_features,
    protected,
    classifier,
    combat_enabled,
):
    if hasattr(module, "build_model_pipeline"):
        return module.build_model_pipeline(
            modality="combined",
            radiomics_features=radiomics_features,
            protected=protected,
            classifier=classifier,
        )
    return Pipeline(
        [
            (
                "selector",
                module.StrictPreprocessSelector(
                    k=module.MRMR_K,
                    protected_features=protected,
                    missing_threshold=module.MISSING_THRESHOLD,
                    variance_threshold=module.VARIANCE_THRESHOLD,
                ),
            ),
            ("classifier", classifier),
        ]
    )


def prepare_estimator(model_name: str, estimator):
    estimator = clone(estimator)
    if model_name == "RandomForest":
        estimator.set_params(bootstrap=True)
    elif model_name == "LightGBM":
        estimator.set_params(subsample_freq=1)
    return estimator


def score_candidate(
    model_name: str,
    base_estimator,
    parameters: dict,
    folds: list[dict],
) -> dict:
    train_aucs = []
    cv_aucs = []
    train_brier = []
    cv_brier = []
    errors = []
    for fold in folds:
        try:
            estimator = prepare_estimator(model_name, base_estimator)
            estimator.set_params(**tuning.estimator_parameters(parameters))
            estimator.fit(fold["x_train"], fold["y_train"])
            train_probability = estimator.predict_proba(fold["x_train"])[:, 1]
            valid_probability = estimator.predict_proba(fold["x_valid"])[:, 1]
            train_aucs.append(
                roc_auc_score(fold["y_train"], train_probability)
            )
            cv_aucs.append(
                roc_auc_score(fold["y_valid"], valid_probability)
            )
            train_brier.append(
                brier_score_loss(fold["y_train"], train_probability)
            )
            cv_brier.append(
                brier_score_loss(fold["y_valid"], valid_probability)
            )
        except Exception as error:
            errors.append(f"fold {fold['fold']}: {type(error).__name__}: {error}")

    if errors:
        return {
            "Status": "Failed",
            "Mean_Train_AUC": np.nan,
            "Mean_CV_AUC": np.nan,
            "Train_CV_Gap": np.nan,
            "SD_CV_AUC": np.nan,
            "Mean_Train_Brier": np.nan,
            "Mean_CV_Brier": np.nan,
            "Error": " | ".join(errors),
        }

    mean_train_auc = float(np.mean(train_aucs))
    mean_cv_auc = float(np.mean(cv_aucs))
    return {
        "Status": "Success",
        "Mean_Train_AUC": mean_train_auc,
        "Mean_CV_AUC": mean_cv_auc,
        "Train_CV_Gap": mean_train_auc - mean_cv_auc,
        "SD_CV_AUC": float(np.std(cv_aucs, ddof=1)),
        "Mean_Train_Brier": float(np.mean(train_brier)),
        "Mean_CV_Brier": float(np.mean(cv_brier)),
        "Error": "",
    }


def select_best_trial(trials: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    successful = trials.loc[trials["Status"] == "Success"].copy()
    if successful.empty:
        raise RuntimeError("No successful tuning trials")

    eligible = successful.loc[
        successful["Train_CV_Gap"] <= MAX_TRAIN_CV_GAP
    ].copy()
    if eligible.empty:
        eligible = successful.copy()

    best_cv_auc = float(eligible["Mean_CV_AUC"].max())
    band = eligible.loc[
        eligible["Mean_CV_AUC"] >= best_cv_auc - CV_BAND
    ].copy()
    band["Training_Only_Score"] = (
        band["Mean_CV_AUC"]
        - 0.5 * band["Train_CV_Gap"].clip(lower=0)
        - 0.01 * band["Mean_CV_Brier"]
    )
    best = band.sort_values(
        [
            "Training_Only_Score",
            "Mean_CV_AUC",
            "Mean_CV_Brier",
        ],
        ascending=[False, False, True],
    ).iloc[0]
    return best, band


def prune_inactive_parameters(model_name: str, parameters: dict) -> dict:
    selected = dict(parameters)
    if model_name == "LogisticRegression" and selected.get("penalty") != "elasticnet":
        selected.pop("l1_ratio", None)
    if model_name == "SVM" and selected.get("kernel") != "poly":
        selected.pop("degree", None)
    return selected


def evaluate_selected_pipeline(
    module,
    branch_name: str,
    model_name: str,
    parameters: dict,
    base_configs: dict,
    X_train,
    X_valid,
    y_train,
    y_valid,
    radiomics_features,
    protected,
) -> dict:
    estimator = prepare_estimator(
        model_name,
        base_configs[model_name]["model"],
    ).set_params(**tuning.estimator_parameters(parameters))
    pipeline = build_branch_pipeline(
        module=module,
        radiomics_features=radiomics_features,
        protected=protected,
        classifier=estimator,
        combat_enabled=branch_name == "combat",
    )
    pipeline.fit(X_train, y_train)
    train_probability = pipeline.predict_proba(X_train)[:, 1]
    valid_probability = pipeline.predict_proba(X_valid)[:, 1]
    return {
        "Branch": branch_name,
        "Model": model_name,
        "Selected_Parameters": json.dumps(
            parameters,
            ensure_ascii=False,
            sort_keys=True,
        ),
        "Full_Train_AUC": float(roc_auc_score(y_train, train_probability)),
        "Valid_AUC_Post_Selection_Only": float(
            roc_auc_score(y_valid, valid_probability)
        ),
        "Full_Train_Brier": float(
            brier_score_loss(y_train, train_probability)
        ),
        "Valid_Brier_Post_Selection_Only": float(
            brier_score_loss(y_valid, valid_probability)
        ),
    }


def run_branch(
    branch_name: str,
    models: list[str],
    force: bool,
) -> None:
    settings = BRANCHES[branch_name]
    output_dir = settings["root"]
    cache_dir = output_dir / "14_Tuning_Cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    module = load_branch_module(branch_name)
    module.INPUT_FILE = Path(settings["input_file"])
    tuning.validate_parameter_spaces()

    space_file = output_dir / "14_Tuning_Parameter_Space.xlsx"
    pd.DataFrame(tuning.parameter_space_records()).to_excel(
        space_file,
        index=False,
    )

    data, features, protected, radiomics_features = module.load_data(
        "combined"
    )
    train_df, valid_df = train_test_split(
        data,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=data["label"],
    )
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    X_train = branch_model_input(
        module,
        train_df,
        features,
        settings["combat_enabled"],
    )
    X_valid = branch_model_input(
        module,
        valid_df,
        features,
        settings["combat_enabled"],
    )
    y_train = train_df["label"].to_numpy()
    y_valid = valid_df["label"].to_numpy()

    print(
        f"{branch_name}: building fold-internal transformed data "
        f"for {len(train_df)} training / {len(valid_df)} validation cases"
    )
    folds = build_fold_data(
        module=module,
        X_train=X_train,
        y_train=y_train,
        protected=protected,
        radiomics_features=radiomics_features,
        combat_enabled=settings["combat_enabled"],
    )
    base_configs = module.model_configs()

    best_rows = []
    for model_name in models:
        cache_file = cache_dir / f"{model_name}_trials.csv"
        if cache_file.exists() and not force:
            trials = pd.read_csv(cache_file)
            print(f"{branch_name} {model_name}: loaded {len(trials)} cached trials")
        else:
            candidates = tuning.sample_parameter_candidates(model_name)
            print(
                f"{branch_name} {model_name}: "
                f"{len(candidates)} candidates x {N_FOLDS} folds"
            )
            records = []
            base_estimator = base_configs[model_name]["model"]
            for trial_id, parameters in enumerate(candidates, start=1):
                record = score_candidate(
                    model_name=model_name,
                    base_estimator=base_estimator,
                    parameters=parameters,
                    folds=folds,
                )
                records.append(
                    {
                        "Branch": branch_name,
                        "Model": model_name,
                        "Trial": trial_id,
                        "Parameters": json.dumps(
                            parameters,
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        **record,
                    }
                )
                print(
                    f"  {trial_id:03d}/{len(candidates):03d} "
                    f"status={record['Status']} "
                    f"CV_AUC={record['Mean_CV_AUC']:.4f} "
                    f"gap={record['Train_CV_Gap']:.4f}"
                )
            trials = pd.DataFrame(records)
            trials.to_csv(cache_file, index=False, encoding="utf-8-sig")

        best, _ = select_best_trial(trials)
        parameters = prune_inactive_parameters(
            model_name,
            json.loads(best["Parameters"]),
        )
        validation = evaluate_selected_pipeline(
            module=module,
            branch_name=branch_name,
            model_name=model_name,
            parameters=parameters,
            base_configs=base_configs,
            X_train=X_train,
            X_valid=X_valid,
            y_train=y_train,
            y_valid=y_valid,
            radiomics_features=radiomics_features,
            protected=protected,
        )
        best_rows.append(
            {
                "Branch": branch_name,
                "Model": model_name,
                "Selected_Parameters": json.dumps(
                    parameters,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "Selected_Trial": int(best["Trial"]),
                "N_Candidates": int(len(trials)),
                "Selected_CV_AUC": float(best["Mean_CV_AUC"]),
                "Selected_Train_AUC_CV_FOLDS": float(
                    best["Mean_Train_AUC"]
                ),
                "Selected_Train_CV_Gap": float(best["Train_CV_Gap"]),
                "Selected_SD_CV_AUC": float(best["SD_CV_AUC"]),
                "Selected_CV_Brier": float(best["Mean_CV_Brier"]),
                **validation,
            }
        )

    all_trials = pd.concat(
        [
            pd.read_csv(cache_dir / f"{model_name}_trials.csv")
            for model_name in models
        ],
        ignore_index=True,
    )
    all_trials.to_excel(
        output_dir / "14_Tuning_All_Trials.xlsx",
        index=False,
    )
    best_frame = pd.DataFrame(best_rows)
    best_frame.to_excel(
        output_dir / "14_Tuning_Best_Parameters.xlsx",
        index=False,
    )
    best_payload = {
        row["Model"]: json.loads(row["Selected_Parameters"])
        for row in best_rows
    }
    (output_dir / "14_Tuning_Best_Parameters.json").write_text(
        json.dumps(best_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "branch": branch_name,
        "combat_enabled": bool(settings["combat_enabled"]),
        "cohort_n": int(len(data)),
        "train_n": int(len(train_df)),
        "validation_n": int(len(valid_df)),
        "random_state": RANDOM_STATE,
        "n_folds": N_FOLDS,
        "candidate_generation": (
            "Deterministic coverage candidates plus ParameterSampler "
            "within the recorded discrete spaces."
        ),
        "parameter_value_rule": "Each tuned parameter has 3-5 allowed values.",
        "selection_rule": (
            "Training-only five-fold CV. Candidates with train-CV AUC gap "
            f"> {MAX_TRAIN_CV_GAP} are excluded when eligible candidates "
            f"exist. Within {CV_BAND} of the best CV AUC, rank by "
            "Mean_CV_AUC - 0.5*gap - 0.01*Brier."
        ),
        "validation_usage": (
            "The locked 20% internal-validation set is evaluated only after "
            "the best parameters are selected."
        ),
        "models": best_rows,
        "files": {
            "parameter_space": str(space_file),
            "all_trials": str(output_dir / "14_Tuning_All_Trials.xlsx"),
            "best_parameters_xlsx": str(
                output_dir / "14_Tuning_Best_Parameters.xlsx"
            ),
            "best_parameters_json": str(
                output_dir / "14_Tuning_Best_Parameters.json"
            ),
        },
    }
    (output_dir / "14_Tuning_Summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(best_frame.to_string(index=False))
    print(f"{branch_name} tuning outputs: {output_dir}")


def main() -> None:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(
        description="Retune both PixelSpacing branches with locked validation."
    )
    parser.add_argument(
        "--branch",
        choices=[
            "pixel_both",
            "pixel_combat",
            "pixel_no_combat",
        ],
        default="pixel_both",
    )
    parser.add_argument(
        "--models",
        default=",".join(tuning.TUNING_SPACES),
        help="Comma-separated subset of classifiers.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute cached model trials.",
    )
    args = parser.parse_args()

    models = [value.strip() for value in args.models.split(",") if value.strip()]
    unknown = sorted(set(models) - set(tuning.TUNING_SPACES))
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")
    branch_aliases = {
        "pixel_both": ["pixel_no_combat", "pixel_combat"],
    }
    branches = branch_aliases.get(args.branch, [args.branch])
    for branch_name in branches:
        run_branch(branch_name, models, args.force)


if __name__ == "__main__":
    main()
