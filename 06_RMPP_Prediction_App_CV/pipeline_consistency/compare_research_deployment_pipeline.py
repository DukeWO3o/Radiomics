from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import cohen_kappa_score, roc_auc_score


BASE_DIR = Path(__file__).resolve().parent
APP_DIR = BASE_DIR.parent
PROJECT_DIR = APP_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))
from pipeline_paths import APP_DEPLOYMENT_OUTPUT_ROOT, DEFAULT_MODEL_DATA_931

U_NET_METRICS = (
    APP_DIR / "unet_validation_output" / "01_Test_Case_Metrics.csv"
)
MODEL_CONFIG = APP_DIR / "models" / "model_config.json"
SVM_MODEL = APP_DIR / "models" / "model_SVM.pkl"
RADIOMICS_CONFIG = APP_DIR / "config" / "radiomics_config.yaml"
RESEARCH_RADIOMICS_CONFIG = (
    PROJECT_DIR / "01_PreProcess" / "radiomics_config.yaml"
)
REFERENCE_WORKBOOK = DEFAULT_MODEL_DATA_931
OUTPUT_DIR = APP_DEPLOYMENT_OUTPUT_ROOT
CACHE_DIR = OUTPUT_DIR / "cache"

CLINICAL_FEATURES = ["age", "CRP", "WBC", "LDH"]
THRESHOLD = 0.5


class StrictPreprocessSelector:

    def __init__(
        self,
        k=8,
        p_threshold=0.05,
        protected_features=None,
        missing_threshold=0.30,
        variance_threshold=0.0,
        random_state=42,
    ):
        self.k = k
        self.p_threshold = p_threshold
        self.protected_features = protected_features
        self.missing_threshold = missing_threshold
        self.variance_threshold = variance_threshold
        self.random_state = random_state

    def _as_frame(self, x):
        if isinstance(x, pd.DataFrame):
            return x.copy()
        return pd.DataFrame(x, columns=self.feature_names_in_)

    def transform(self, x):
        frame = self._as_frame(x)
        imputed = frame[self.missing_keep_].fillna(self.impute_means_)
        imputed = imputed[self.variance_keep_]
        scaled = self.scaler_.transform(imputed)
        indices = [self.variance_keep_.index(f) for f in self.selected_features_]
        return scaled[:, indices]


def set_pickle_runtime() -> None:
    setattr(sys.modules["__main__"], "StrictPreprocessSelector", StrictPreprocessSelector)


def sanitize_filename(value: str) -> str:
    return Path(str(value)).name


def calculate_icc_2_1(x, y) -> float:
    values = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
    values = values[np.isfinite(values).all(axis=1)]
    n, k = values.shape
    if n < 2 or k != 2:
        return math.nan

    grand_mean = values.mean()
    row_means = values.mean(axis=1)
    column_means = values.mean(axis=0)
    ss_total = ((values - grand_mean) ** 2).sum()
    ss_rows = k * ((row_means - grand_mean) ** 2).sum()
    ss_columns = n * ((column_means - grand_mean) ** 2).sum()
    ss_error = ss_total - ss_rows - ss_columns

    ms_rows = ss_rows / (n - 1)
    ms_columns = ss_columns / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    denominator = (
        ms_rows
        + (k - 1) * ms_error
        + (k * (ms_columns - ms_error) / n)
    )
    if denominator == 0:
        return 1.0 if np.allclose(values[:, 0], values[:, 1]) else math.nan
    return float((ms_rows - ms_error) / denominator)


def agreement_metrics(x, y) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if not len(x):
        return {
            "N": 0,
            "Mean_X": math.nan,
            "Mean_Y": math.nan,
            "Mean_Difference_Y_minus_X": math.nan,
            "SD_Difference": math.nan,
            "Bland_Altman_Lower": math.nan,
            "Bland_Altman_Upper": math.nan,
            "Mean_Absolute_Difference": math.nan,
            "Median_Absolute_Difference": math.nan,
            "Pearson_r": math.nan,
            "Spearman_rho": math.nan,
            "ICC_2_1": math.nan,
        }

    difference = y - x
    abs_difference = np.abs(difference)
    diff_sd = float(np.std(difference, ddof=1)) if len(difference) > 1 else math.nan
    nonzero = np.abs(x) > np.finfo(float).eps
    relative_difference = np.full_like(abs_difference, np.nan, dtype=float)
    relative_difference[nonzero] = (
        abs_difference[nonzero] / np.abs(x[nonzero]) * 100.0
    )
    scale = max(float(np.std(x, ddof=1)), float(np.std(y, ddof=1)))
    normalized_abs_difference = (
        abs_difference / scale if scale > 0 else np.zeros_like(abs_difference)
    )
    return {
        "N": int(len(x)),
        "Mean_X": float(np.mean(x)),
        "Mean_Y": float(np.mean(y)),
        "Mean_Difference_Y_minus_X": float(np.mean(difference)),
        "SD_Difference": diff_sd,
        "Bland_Altman_Lower": float(np.mean(difference) - 1.96 * diff_sd),
        "Bland_Altman_Upper": float(np.mean(difference) + 1.96 * diff_sd),
        "Mean_Absolute_Difference": float(np.mean(abs_difference)),
        "Median_Absolute_Difference": float(np.median(abs_difference)),
        "Mean_Relative_Absolute_Difference_Percent": float(
            np.nanmean(relative_difference)
        ),
        "Median_Relative_Absolute_Difference_Percent": float(
            np.nanmedian(relative_difference)
        ),
        "Mean_Normalized_Absolute_Difference": float(
            np.mean(normalized_abs_difference)
        ),
        "Pearson_r": float(pd.Series(x).corr(pd.Series(y), method="pearson")),
        "Spearman_rho": float(pd.Series(x).corr(pd.Series(y), method="spearman")),
        "ICC_2_1": calculate_icc_2_1(x, y),
    }


def probability_metrics(x, y, label=None) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    metrics = agreement_metrics(x, y)
    predicted_x = (x >= THRESHOLD).astype(int)
    predicted_y = (y >= THRESHOLD).astype(int)
    metrics.update(
        {
            "Classification_Agreement": float(np.mean(predicted_x == predicted_y)),
            "Classification_Disagreement": float(np.mean(predicted_x != predicted_y)),
            "Cohen_Kappa": float(cohen_kappa_score(predicted_x, predicted_y)),
            "Reference_Positive_N": int(predicted_x.sum()),
            "Deployment_Positive_N": int(predicted_y.sum()),
            "Both_Positive_N": int(((predicted_x == 1) & (predicted_y == 1)).sum()),
            "Both_Negative_N": int(((predicted_x == 0) & (predicted_y == 0)).sum()),
            "Reference_Positive_Deployment_Negative_N": int(
                ((predicted_x == 1) & (predicted_y == 0)).sum()
            ),
            "Reference_Negative_Deployment_Positive_N": int(
                ((predicted_x == 0) & (predicted_y == 1)).sum()
            ),
        }
    )
    if label is not None:
        finite_x = np.isfinite(x)
        finite_y = np.isfinite(y)
        finite_label = np.asarray(label)[finite_x | finite_y]
        if len(np.unique(finite_label)) == 2:
            metrics["Reference_AUC"] = (
                float(roc_auc_score(np.asarray(label)[finite_x], x[finite_x]))
                if finite_x.any()
                else math.nan
            )
            metrics["Deployment_AUC"] = (
                float(roc_auc_score(np.asarray(label)[finite_y], y[finite_y]))
                if finite_y.any()
                else math.nan
            )
    return metrics


def build_research_extractor():
    from radiomics import featureextractor

    return featureextractor.RadiomicsFeatureExtractor(
        str(RESEARCH_RADIOMICS_CONFIG)
    )


def build_app_extractor():
    from radiomics import featureextractor

    return featureextractor.RadiomicsFeatureExtractor(str(RADIOMICS_CONFIG))


def assert_aligned_extractor_settings(research_extractor, app_extractor) -> None:

    keys = sorted(
        set(research_extractor.settings) | set(app_extractor.settings)
    )
    differences = {
        key: (
            research_extractor.settings.get(key),
            app_extractor.settings.get(key),
        )
        for key in keys
        if research_extractor.settings.get(key)
        != app_extractor.settings.get(key)
    }
    if differences:
        details = json.dumps(differences, ensure_ascii=False, default=str)
        raise RuntimeError(
            "Research and deployment PyRadiomics settings differ: " + details
        )


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_selected_features(
    extractor,
    image_path: Path,
    mask_path: Path,
    u_net_mask_path: Path | None,
    selected_features: list[str],
) -> dict:
    import SimpleITK as sitk

    image = sitk.ReadImage(str(image_path))
    if u_net_mask_path is not None:
        output_size = (image.GetWidth(), image.GetHeight())
        predicted = Image.open(u_net_mask_path).convert("L").resize(
            output_size, resample=Image.Resampling.NEAREST
        )
        mask_array = (np.asarray(predicted) > 0).astype(np.uint8)
        mask = sitk.GetImageFromArray(mask_array)
        mask.CopyInformation(image)
    else:
        mask = sitk.ReadImage(str(mask_path))
        if (
            mask.GetWidth() != image.GetWidth()
            or mask.GetHeight() != image.GetHeight()
        ):
            raise ValueError(
                f"Image-mask geometry mismatch: {image_path.name}, "
                f"{image.GetSize()} vs {mask.GetSize()}"
            )

    result = extractor.execute(image, mask)
    features = {
        key.replace("original_", ""): float(value)
        for key, value in result.items()
        if "diagnostics" not in key
    }
    return {
        feature: features.get(feature, math.nan)
        for feature in selected_features
    }


def prepare_case_table() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    with open(MODEL_CONFIG, "r", encoding="utf-8") as handle:
        model_config = json.load(handle)
    selected = model_config["model_selected_features"]["SVM"]
    radiomics_features = [feature for feature in selected if feature not in CLINICAL_FEATURES]
    if len(radiomics_features) != 8:
        raise ValueError(
            f"Expected 8 final radiomics features, found {len(radiomics_features)}"
        )

    reference = pd.read_excel(REFERENCE_WORKBOOK, sheet_name="Sheet1")
    reference["Image"] = reference["image"].map(sanitize_filename)
    reference["Label"] = reference["label"].astype(int)
    reference = reference.drop_duplicates(["Label", "Image"], keep="first").set_index(
        ["Label", "Image"]
    )

    u_net_cases = pd.read_csv(U_NET_METRICS, encoding="utf-8-sig")
    u_net_cases["Image"] = u_net_cases["Image"].map(sanitize_filename)
    u_net_cases["Label"] = u_net_cases["Label"].astype(int)
    anonymized_ids = {
        image_name: f"UNET-{index:03d}"
        for index, image_name in enumerate(sorted(set(u_net_cases["Image"])), start=1)
    }
    u_net_cases["Case_ID"] = u_net_cases["Image"].map(anonymized_ids)

    reference_keys = set(reference.index)
    u_net_keys = set(zip(u_net_cases["Label"], u_net_cases["Image"]))
    missing_keys = sorted(u_net_keys - reference_keys)
    excluded = u_net_cases[
        [
            (label, image) in missing_keys
            for label, image in zip(u_net_cases["Label"], u_net_cases["Image"])
        ]
    ].copy()
    excluded["Exclusion_Reason"] = (
        "Not present in the final 931-case modelling workbook; no paired "
        "research-pipeline radiomics reference was available."
    )
    excluded = excluded[
        ["Case_ID", "Dice", "HD95_mm", "Exclusion_Reason"]
    ].copy()
    u_net_cases = u_net_cases[
        [
            (label, image) not in missing_keys
            for label, image in zip(u_net_cases["Label"], u_net_cases["Image"])
        ]
    ].copy()
    if u_net_cases.empty:
        raise ValueError("No U-Net test cases overlap the final modelling cohort.")

    case_table = u_net_cases.merge(
        reference.reset_index(),
        on=["Label", "Image"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_reference"),
    )
    for feature in selected:
        if feature not in case_table.columns:
            raise ValueError(f"Reference feature missing: {feature}")
    return case_table, excluded, radiomics_features


def extract_case_features(
    case_table: pd.DataFrame,
    radiomics_features: list[str],
    force: bool,
) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "paired_feature_cache.csv"
    if cache_path.exists() and not force:
        cached = pd.read_csv(cache_path, encoding="utf-8-sig")
        cache_updated = False
        if "Case_ID" not in cached.columns and "Image" in cached.columns:
            id_map = dict(zip(case_table["Image"], case_table["Case_ID"]))
            cached["Case_ID"] = cached["Image"].map(id_map)
            cached = cached.drop(columns=["Image"])
            cache_updated = True
        required = set(case_table["Case_ID"])
        if "Case_ID" in cached.columns and required.issubset(set(cached["Case_ID"])):
            if cache_updated:
                cached.to_csv(cache_path, index=False, encoding="utf-8-sig")
            return cached[cached["Case_ID"].isin(required)].copy()


    app_extractor = build_app_extractor()
    research_extractor = build_research_extractor()
    assert_aligned_extractor_settings(research_extractor, app_extractor)
    rows = []
    start_time = time.perf_counter()

    for index, row in case_table.iterrows():
        image_path = Path(row["Image_Path"])
        expert_mask_path = Path(row["Mask_Path"])
        u_net_mask_path = Path(row["Prediction_Mask_Path"])
        record = {"Case_ID": row["Case_ID"]}

        extraction_start = time.perf_counter()
        expert_app = extract_selected_features(
            app_extractor,
            image_path,
            expert_mask_path,
            None,
            radiomics_features,
        )
        u_net_research = extract_selected_features(
            research_extractor,
            image_path,
            expert_mask_path,
            u_net_mask_path,
            radiomics_features,
        )
        u_net_app = extract_selected_features(
            app_extractor,
            image_path,
            expert_mask_path,
            u_net_mask_path,
            radiomics_features,
        )
        for feature in radiomics_features:
            record[f"Expert_AppConfig__{feature}"] = expert_app[feature]
            record[f"UNet_ResearchConfig__{feature}"] = u_net_research[feature]
            record[f"UNet_AppConfig__{feature}"] = u_net_app[feature]
        record["Feature_Extraction_ms"] = (
            time.perf_counter() - extraction_start
        ) * 1000.0
        rows.append(record)

        if (index + 1) % 10 == 0 or index == len(case_table) - 1:
            pd.DataFrame(rows).to_csv(
                cache_path,
                index=False,
                encoding="utf-8-sig",
            )
            elapsed = time.perf_counter() - start_time
            print(
                f"Extracted {index + 1}/{len(case_table)} cases "
                f"({elapsed:.1f} s elapsed)"
            )

    return pd.DataFrame(rows)


def make_model_input(
    case_row: pd.Series,
    feature_names: list[str],
    radiomics_values: dict,
    feature_order: list[str],
) -> pd.DataFrame:
    clinical = {feature: float(case_row[feature]) for feature in CLINICAL_FEATURES}
    full_row = {
        feature: clinical.get(feature, radiomics_values.get(feature, math.nan))
        for feature in feature_order
    }
    return pd.DataFrame([full_row], columns=feature_order).astype(float)


def build_case_outputs(
    case_table: pd.DataFrame,
    extracted: pd.DataFrame,
    excluded: pd.DataFrame,
    radiomics_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    set_pickle_runtime()
    model = joblib.load(str(SVM_MODEL))
    with open(MODEL_CONFIG, "r", encoding="utf-8") as handle:
        model_config = json.load(handle)
    feature_order = model_config["feature_order"]

    paired = case_table.merge(
        extracted,
        on="Case_ID",
        how="left",
        validate="one_to_one",
    )
    case_rows = []
    for _, row in paired.iterrows():
        research_features = {
            feature: float(row[feature]) for feature in radiomics_features
        }
        expert_app_features = {
            feature: float(row[f"Expert_AppConfig__{feature}"])
            for feature in radiomics_features
        }
        u_net_research_features = {
            feature: float(row[f"UNet_ResearchConfig__{feature}"])
            for feature in radiomics_features
        }
        u_net_app_features = {
            feature: float(row[f"UNet_AppConfig__{feature}"])
            for feature in radiomics_features
        }

        inputs = {
            "Research_Reference": make_model_input(
                row, radiomics_features, research_features, feature_order
            ),
            "ExpertMask_AppConfig": make_model_input(
                row, radiomics_features, expert_app_features, feature_order
            ),
            "UNetMask_ResearchConfig": make_model_input(
                row, radiomics_features, u_net_research_features, feature_order
            ),
            "Deployment_Emulation": make_model_input(
                row, radiomics_features, u_net_app_features, feature_order
            ),
        }
        probabilities = {
            name: float(model.predict_proba(values)[0, 1])
            for name, values in inputs.items()
        }
        case_output = {
            "Case_ID": row["Case_ID"],
            "Label": int(row["Label"]),
            "Dice": float(row["Dice"]),
            "HD95_mm": float(row["HD95_mm"]),
            "Age": float(row["age"]),
            "CRP": float(row["CRP"]),
            "WBC": float(row["WBC"]),
            "LDH": float(row["LDH"]),
            **probabilities,
        }
        for feature in radiomics_features:
            case_output[f"Research_Reference__{feature}"] = research_features[feature]
            case_output[f"ExpertMask_AppConfig__{feature}"] = (
                expert_app_features[feature]
            )
            case_output[f"UNetMask_ResearchConfig__{feature}"] = (
                u_net_research_features[feature]
            )
            case_output[f"Deployment_Emulation__{feature}"] = u_net_app_features[feature]
        case_rows.append(case_output)

    cases = pd.DataFrame(case_rows)
    comparisons = [
        (
            "Overall_Research_vs_Deployment",
            "Research_Reference",
            "Deployment_Emulation",
        ),
        (
            "ExtractionConfig_ExpertMask_ResearchConfig_vs_AppConfig",
            "Research_Reference",
            "ExpertMask_AppConfig",
        ),
        (
            "Segmentation_ResearchConfig_ExpertMask_vs_UNetMask",
            "Research_Reference",
            "UNetMask_ResearchConfig",
        ),
        (
            "Segmentation_AppConfig_ExpertMask_vs_UNetMask",
            "ExpertMask_AppConfig",
            "Deployment_Emulation",
        ),
    ]

    feature_rows = []
    for comparison, reference_name, deployment_name in comparisons:
        for feature in radiomics_features:
            reference_values = cases[f"{reference_name}__{feature}"]
            deployment_values = cases[f"{deployment_name}__{feature}"]
            feature_rows.append(
                {
                    "Comparison": comparison,
                    "Feature": feature,
                    **agreement_metrics(reference_values, deployment_values),
                }
            )
    feature_agreement = pd.DataFrame(feature_rows)

    probability_rows = []
    for comparison, reference_name, deployment_name in comparisons:
        probability_rows.append(
            {
                "Comparison": comparison,
                "Reference": reference_name,
                "Evaluation": deployment_name,
                **probability_metrics(
                    cases[reference_name],
                    cases[deployment_name],
                    cases["Label"].to_numpy(),
                ),
            }
        )
    probability_agreement = pd.DataFrame(probability_rows)
    summary = probability_agreement[
        probability_agreement["Comparison"] == "Overall_Research_vs_Deployment"
    ].iloc[0].to_dict()
    summary.update(
        {
            "N_Cases": int(len(cases)),
            "N_U_Net_Test_Cases": int(len(cases) + len(excluded)),
            "N_Excluded_No_Research_Reference": int(len(excluded)),
            "Final_Radiomics_Feature_N": int(len(radiomics_features)),
            "Final_Clinical_Feature_N": int(len(CLINICAL_FEATURES)),
            "Model": "SVM RBF",
            "Threshold": THRESHOLD,
            "Research_Radiomics_Config_SHA256": file_digest(
                RESEARCH_RADIOMICS_CONFIG
            ),
            "Deployment_Radiomics_Config_SHA256": file_digest(
                RADIOMICS_CONFIG
            ),
            "Overall_Feature_ICC_Median": float(
                feature_agreement.loc[
                    feature_agreement["Comparison"]
                    == "Overall_Research_vs_Deployment",
                    "ICC_2_1",
                ].median()
            ),
            "Overall_Feature_ICC_Min": float(
                feature_agreement.loc[
                    feature_agreement["Comparison"]
                    == "Overall_Research_vs_Deployment",
                    "ICC_2_1",
                ].min()
            ),
        }
    )
    return cases, feature_agreement, summary, probability_agreement


def format_workbook(path: Path) -> None:
    from openpyxl import load_workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

    workbook = load_workbook(path)
    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(name="Arial", size=10, bold=True, color="FFFFFF")
    body_font = Font(name="Arial", size=10)
    thin = Side(style="thin", color="D9D9D9")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    for sheet in workbook.worksheets:
        for cell in sheet[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(
                horizontal="center", vertical="center", wrap_text=True
            )
            cell.border = border
        for row in sheet.iter_rows(min_row=2):
            for cell in row:
                cell.font = body_font
                cell.alignment = Alignment(vertical="center", wrap_text=True)
                cell.border = border
        for column in sheet.columns:
            values = [
                str(cell.value) if cell.value is not None else ""
                for cell in column
            ]
            width = min(max(len(value) for value in values) + 2, 48)
            sheet.column_dimensions[column[0].column_letter].width = max(width, 12)
        sheet.freeze_panes = "A2"
        sheet.sheet_view.showGridLines = False
    workbook.save(path)


def write_outputs(
    cases: pd.DataFrame,
    feature_agreement: pd.DataFrame,
    probability_agreement: pd.DataFrame,
    excluded: pd.DataFrame,
    summary: dict,
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    workbook_path = OUTPUT_DIR / "Research_vs_Deployment_Pipeline_Consistency.xlsx"
    readme_rows = [
        {
            "Item": "Purpose",
            "Description": (
                "Paired comparison of the research pipeline and the deployed app "
                "pipeline on the "
                f"{summary['N_U_Net_Test_Cases']} independent U-Net test cases."
            ),
        },
        {
            "Item": "Research_Reference",
            "Description": (
                "Final 8 radiomics features stored in the research modelling "
                "workbook; expert mask and research extraction settings "
                "(normalize=false)."
            ),
        },
        {
            "Item": "Radiomics_Configuration_Alignment",
            "Description": (
                "The analysis loads the original research and deployment YAML files "
                "separately and fails before extraction if their effective "
                "PyRadiomics settings differ."
            ),
        },
        {
            "Item": "ExpertMask_AppConfig",
            "Description": (
                "Expert mask with the app radiomics settings "
                "(normalize=false, normalizeScale=1; aligned with research)."
            ),
        },
        {
            "Item": "UNetMask_ResearchConfig",
            "Description": (
                "U-Net mask upsampled from 512 to 1024 with nearest-neighbour "
                "interpolation, using research extraction settings."
            ),
        },
        {
            "Item": "Deployment_Emulation",
            "Description": (
                "U-Net mask upsampled from 512 to 1024 with nearest-neighbour "
                "interpolation, using app extraction settings."
            ),
        },
        {
            "Item": "Clinical_Inputs",
            "Description": (
                "Age, CRP, WBC, and LDH were fixed to the same values across "
                "all paired comparisons."
            ),
        },
        {
            "Item": "Probability",
            "Description": (
                "Predicted probability from the saved final SVM pipeline. "
                "Positive class threshold was 0.5."
            ),
        },
        {
            "Item": "Statistics",
            "Description": (
                "ICC(2,1) absolute agreement, Bland-Altman bias and limits, "
                "Pearson/Spearman correlation, and threshold classification "
                "agreement."
            ),
        },
        {
            "Item": "Excluded_Cases",
            "Description": (
                "U-Net test images absent from the final 931-case modelling "
                "workbook were excluded because no paired research reference "
                "could be constructed without redefining the study cohort."
            ),
        },
    ]
    readme = pd.DataFrame(readme_rows)
    summary_frame = pd.DataFrame(
        [{"Metric": key, "Value": value} for key, value in summary.items()]
    )
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        readme.to_excel(writer, sheet_name="README", index=False)
        summary_frame.to_excel(writer, sheet_name="Summary", index=False)
        probability_agreement.to_excel(
            writer, sheet_name="Probability_Agreement", index=False
        )
        feature_agreement.to_excel(
            writer, sheet_name="Feature_Agreement", index=False
        )
        cases.to_excel(writer, sheet_name="Case_Level_Results", index=False)
        excluded.to_excel(writer, sheet_name="Excluded_Cases", index=False)
    format_workbook(workbook_path)

    cases.to_csv(OUTPUT_DIR / "Case_Level_Results.csv", index=False, encoding="utf-8-sig")
    feature_agreement.to_csv(
        OUTPUT_DIR / "Feature_Agreement.csv", index=False, encoding="utf-8-sig"
    )
    probability_agreement.to_csv(
        OUTPUT_DIR / "Probability_Agreement.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (OUTPUT_DIR / "Summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return workbook_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare research and deployed RMPP pipelines using paired expert "
            "and U-Net masks."
        )
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract features even if the paired feature cache exists.",
    )
    args = parser.parse_args()

    case_table, excluded, radiomics_features = prepare_case_table()
    print(
        f"Prepared {len(case_table)} cases with "
        f"{len(radiomics_features)} final radiomics features; "
        f"excluded {len(excluded)} cases without a research reference."
    )
    extracted = extract_case_features(case_table, radiomics_features, args.force)
    cases, feature_agreement, summary, probability_agreement = build_case_outputs(
        case_table,
        extracted,
        excluded,
        radiomics_features,
    )
    workbook_path = write_outputs(
        cases,
        feature_agreement,
        probability_agreement,
        excluded,
        summary,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Workbook: {workbook_path}")


if __name__ == "__main__":
    main()
