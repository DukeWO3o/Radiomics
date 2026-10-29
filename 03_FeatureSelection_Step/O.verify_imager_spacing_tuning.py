from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from pipeline_paths import (
    PIXEL_COMBAT_ROOT,
    PIXEL_NO_COMBAT_ROOT,
    PIXEL_SUMMARY_ROOT,
)

BRANCHES = {
    "pixel_no_combat": PIXEL_NO_COMBAT_ROOT,
    "pixel_combat": PIXEL_COMBAT_ROOT,
}


def compare_parameter(expected, actual) -> bool:
    if isinstance(expected, list):
        expected = tuple(expected)
    if isinstance(expected, float) and np.isnan(expected):
        return bool(pd.isna(actual))
    return expected == actual


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from strict_cv_pickle_helpers import (
        ComBatTransformer,
        StrictPreprocessSelector,
    )

    setattr(sys.modules["__main__"], "StrictPreprocessSelector", StrictPreprocessSelector)
    setattr(sys.modules["__main__"], "ComBatTransformer", ComBatTransformer)

    errors = []
    results = {}
    for branch, root in BRANCHES.items():
        branch_errors = []
        model_dir = root / "\u8054\u5408\u6a21\u578b"
        space = pd.read_excel(root / "14_Tuning_Parameter_Space.xlsx")
        trials = pd.read_excel(root / "14_Tuning_All_Trials.xlsx")
        best = pd.read_excel(root / "14_Tuning_Best_Parameters.xlsx")
        best_json = json.loads(
            (root / "14_Tuning_Best_Parameters.json").read_text(
                encoding="utf-8"
            )
        )
        performance = pd.read_excel(
            model_dir / "04_Model_Performance.xlsx",
            index_col=0,
        )

        if not space["N_Allowed_Values"].between(3, 5).all():
            branch_errors.append(f"{branch}: parameter value count outside 3-5")
        if not space["All_Allowed_Values_Sampled"].all():
            branch_errors.append(f"{branch}: not all configured values were sampled")
        if (trials["Status"] != "Success").any():
            branch_errors.append(f"{branch}: failed tuning trials were found")
        if any(
            column.lower().startswith("valid")
            for column in trials.columns
        ):
            branch_errors.append(
                f"{branch}: validation metrics leaked into all-trial file"
            )

        branch_rows = []
        for row in best.itertuples(index=False):
            model_name = row.Model
            expected = best_json[model_name]
            model = joblib.load(model_dir / f"model_{model_name}.pkl")
            classifier = model.named_steps["classifier"]
            model_parameters_match = True
            for parameter, value in expected.items():
                actual = classifier.get_params()[parameter]
                if not compare_parameter(value, actual):
                    model_parameters_match = False
                    branch_errors.append(
                        f"{branch}/{model_name}: {parameter} "
                        f"expected={value!r}, actual={actual!r}"
                    )

            cv_delta = abs(
                float(performance.loc[model_name, "Selection_CV_AUC"])
                - float(row.Selected_CV_AUC)
            )
            valid_delta = abs(
                float(performance.loc[model_name, "Valid_AUC"])
                - float(row.Valid_AUC_Post_Selection_Only)
            )
            if cv_delta > 1e-3:
                branch_errors.append(f"{branch}/{model_name}: CV AUC mismatch")
            if valid_delta > 1e-10:
                branch_errors.append(
                    f"{branch}/{model_name}: validation AUC mismatch"
                )
            branch_rows.append(
                {
                    "Model": model_name,
                    "N_Parameters": int(len(expected)),
                    "N_Trials": int(
                        (trials["Model"] == model_name).sum()
                    ),
                    "Selected_CV_AUC": float(row.Selected_CV_AUC),
                    "Valid_AUC_Post_Selection_Only": float(
                        row.Valid_AUC_Post_Selection_Only
                    ),
                    "CV_Delta_Final_vs_Tuning": cv_delta,
                    "Valid_Delta_Final_vs_Tuning": valid_delta,
                    "Pickle_Parameters_Match": model_parameters_match,
                }
            )

        errors.extend(branch_errors)
        result = {
            "branch": branch,
            "status": "passed" if not branch_errors else "failed",
            "errors": branch_errors,
            "parameter_rows": int(len(space)),
            "min_allowed_values": int(space["N_Allowed_Values"].min()),
            "max_allowed_values": int(space["N_Allowed_Values"].max()),
            "all_parameter_values_sampled": bool(
                space["All_Allowed_Values_Sampled"].all()
            ),
            "trial_rows": int(len(trials)),
            "successful_trials": int((trials["Status"] == "Success").sum()),
            "validation_absent_from_all_trials": not any(
                column.lower().startswith("valid")
                for column in trials.columns
            ),
            "models": branch_rows,
        }
        results[branch] = result
        (root / "15_Tuning_Verification.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    payload = {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "branches": results,
    }
    verification_path = (
        PIXEL_SUMMARY_ROOT / "15_All_Branches_Tuning_Verification.json"
    )
    verification_path.parent.mkdir(parents=True, exist_ok=True)
    verification_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
