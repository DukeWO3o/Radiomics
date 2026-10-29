from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from pipeline_paths import (
    PIXEL_COMBAT_ROOT,
    PIXEL_NO_COMBAT_ROOT,
    PIXEL_PREPROCESS_ROOT,
    PIXEL_SUMMARY_ROOT,
)

DATA_DIR = PIXEL_PREPROCESS_ROOT
BRANCHES = {
    "PixelSpacing497_NoComBat": PIXEL_NO_COMBAT_ROOT,
    "PixelSpacing497_ComBat": PIXEL_COMBAT_ROOT,
}
OUTPUT_XLSX = PIXEL_SUMMARY_ROOT / "06_PixelSpacing497_Model_Summary.xlsx"
OUTPUT_JSON = PIXEL_SUMMARY_ROOT / "06_PixelSpacing497_Model_Summary.json"


def main() -> None:
    OUTPUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    e_summary = json.loads(
        (DATA_DIR / "01_E_processing_summary.json").read_text(
            encoding="utf-8"
        )
    )
    icc_summary = json.loads(
        (DATA_DIR / "03_ICC_summary_pixel_spacing_497.json").read_text(
            encoding="utf-8"
        )
    )
    overview = pd.DataFrame(
        [
            ["Cohort N", len(pd.read_csv(
                DATA_DIR / "01_E_processing_manifest.csv"
            ))],
            ["RMPP N", 303],
            ["non-RMPP N", 194],
            ["Training N", 397],
            ["Locked validation N", 100],
            ["Spacing source", e_summary["spacing_source"]],
            ["Radiomics features extracted", 102],
            ["Radiomics features with ICC > 0.75", icc_summary["pass_n"]],
            ["ICC paired cases", icc_summary["sample_n"]],
            ["ICC median", icc_summary["median_icc"]],
        ],
        columns=["Item", "Value"],
    )

    best_rows = []
    all_rows = []
    for branch_name, root in BRANCHES.items():
        model_dir = root / "\u8054\u5408\u6a21\u578b"
        config = json.loads(
            (model_dir / "model_config.json").read_text(encoding="utf-8")
        )
        performance = pd.read_excel(
            model_dir / "04_Model_Performance.xlsx",
            index_col=0,
        )
        best_model = config["best_model"]
        row = performance.loc[best_model]
        best_rows.append(
            {
                "Branch": branch_name,
                "Best_Model_By_Training_CV": best_model,
                "Training_CV_AUC": float(row["Selection_CV_AUC"]),
                "Valid_AUC": float(row["Valid_AUC"]),
                "Valid_AUC_95CI_Low": float(row["Valid_AUC_95CI_Low"]),
                "Valid_AUC_95CI_High": float(row["Valid_AUC_95CI_High"]),
                "Valid_Recall": float(row["Valid_Recall"]),
                "Valid_Specificity": float(row["Valid_Specificity"]),
                "Valid_Brier": float(row["Valid_Brier"]),
            }
        )
        for model_name, metrics in performance.iterrows():
            all_rows.append(
                {
                    "Branch": branch_name,
                    "Model": model_name,
                    "Training_CV_AUC": float(
                        metrics["Selection_CV_AUC"]
                    ),
                    "Valid_AUC": float(metrics["Valid_AUC"]),
                    "Valid_AUC_95CI_Low": float(
                        metrics["Valid_AUC_95CI_Low"]
                    ),
                    "Valid_AUC_95CI_High": float(
                        metrics["Valid_AUC_95CI_High"]
                    ),
                    "Valid_Recall": float(metrics["Valid_Recall"]),
                    "Valid_Specificity": float(
                        metrics["Valid_Specificity"]
                    ),
                    "Valid_Brier": float(metrics["Valid_Brier"]),
                    "Selected_Best_Model": model_name == best_model,
                }
            )

    best_frame = pd.DataFrame(best_rows)
    all_frame = pd.DataFrame(all_rows)

    helper_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(helper_dir))
    from strict_cv_pickle_helpers import (
        ComBatTransformer,
        StrictPreprocessSelector,
        fast_delong,
    )

    setattr(sys.modules["__main__"], "StrictPreprocessSelector", StrictPreprocessSelector)
    setattr(sys.modules["__main__"], "ComBatTransformer", ComBatTransformer)
    data = pd.read_excel(
        DATA_DIR / "04_ICC_filtered_pixel_spacing_497.xlsx"
    )
    svm_predictions = {}
    validation_labels = None
    for branch_name, root in BRANCHES.items():
        model_dir = root / "\u8054\u5408\u6a21\u578b"
        config = json.loads(
            (model_dir / "model_config.json").read_text(encoding="utf-8")
        )
        manifest = joblib.load(
            model_dir / "00_Validation_Patient_Manifest.pkl"
        )
        keys = ["image", "label"]
        if "batch_model" in manifest.columns:
            keys.append("batch_model")
        validation = data.merge(manifest, on=keys, how="inner")
        columns = config.get(
            "required_input_columns",
            config["feature_order"],
        )
        model = joblib.load(model_dir / "model_SVM.pkl")
        svm_predictions[branch_name] = model.predict_proba(
            validation[columns]
        )[:, 1]
        current_labels = validation["label"].to_numpy()
        if validation_labels is None:
            validation_labels = current_labels
        elif not np.array_equal(validation_labels, current_labels):
            raise ValueError("Branch validation labels are not aligned")

    prediction_matrix = np.vstack(
        [
            svm_predictions["PixelSpacing497_NoComBat"],
            svm_predictions["PixelSpacing497_ComBat"],
        ]
    )
    order = np.argsort(validation_labels)
    aucs, covariance = fast_delong(
        prediction_matrix[:, order],
        int(np.sum(validation_labels[order] == 1)),
    )
    contrast = np.array([[1, -1]])
    z_value = np.abs(np.diff(aucs)) / np.sqrt(
        contrast @ covariance @ contrast.T
    )
    delong_p = float(2 * (1 - norm.cdf(z_value))[0, 0])
    svm_comparison = pd.DataFrame(
        [
            {
                "Branch": "PixelSpacing497_NoComBat",
                "Valid_AUC": float(
                    roc_auc_score(
                        validation_labels,
                        svm_predictions["PixelSpacing497_NoComBat"],
                    )
                ),
            },
            {
                "Branch": "PixelSpacing497_ComBat",
                "Valid_AUC": float(
                    roc_auc_score(
                        validation_labels,
                        svm_predictions["PixelSpacing497_ComBat"],
                    )
                ),
            },
        ]
    )
    svm_comparison["Paired_DeLong_P_ComBat_vs_NoComBat"] = [
        np.nan,
        delong_p,
    ]
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        overview.to_excel(writer, sheet_name="Overview", index=False)
        best_frame.to_excel(writer, sheet_name="Best_Models", index=False)
        all_frame.to_excel(writer, sheet_name="All_Models", index=False)
        svm_comparison.to_excel(
            writer,
            sheet_name="Same_SVM_DeLong",
            index=False,
        )

    payload = {
        "cohort_n": 497,
        "rmpp_n": 303,
        "non_rmpp_n": 194,
        "train_n": 397,
        "validation_n": 100,
        "spacing_source": "PixelSpacing",
        "radiomics_features_extracted": 102,
        "icc_passed_features": icc_summary["pass_n"],
        "icc_sample_n": icc_summary["sample_n"],
        "best_models": best_rows,
        "same_svm_delong_p": delong_p,
    }
    OUTPUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
