from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from PIL import Image
from radiomics import featureextractor


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_ROOT))

from pipeline_paths import (
    DEFAULT_CLINICAL_931,
    DEFAULT_PIXEL_MODEL_DATA,
    OUTPUT_ROOT,
    PIXEL_PREPROCESS_ROOT,
)

COHORT_FILE = DEFAULT_CLINICAL_931
OUTPUT_DIR = PIXEL_PREPROCESS_ROOT
MANIFEST_FILE = OUTPUT_DIR / "01_E_processing_manifest.csv"
CONFIG_FILE = PIPELINE_ROOT / "01_PreProcess" / "radiomics_config.yaml"
ICC_SAMPLE_DIR = OUTPUT_ROOT / "Reproducibility"
R2_MASK_ROOT = ICC_SAMPLE_DIR / "ROI_review_50cases"
R2_TRANSFORMED_ROOT = PIXEL_PREPROCESS_ROOT / "R2_transformed_masks"
E_SCRIPT = PIPELINE_ROOT / "01_PreProcess" / "E.DCM图像标准化_ImagerSpacing.py"
ICC_THRESHOLD = 0.75


def load_e_module():
    spec = importlib.util.spec_from_file_location(
        "_imager_spacing_e_module", E_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load E script: {E_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


E = load_e_module()
logging.getLogger("radiomics").setLevel(logging.ERROR)


def read_with_geometry(image_path: Path, mask_path: Path, row):
    image = sitk.ReadImage(str(image_path))
    mask = sitk.ReadImage(str(mask_path))
    if image.GetSize() != mask.GetSize():
        raise ValueError(
            f"Image/mask size mismatch: {image_path.name}: "
            f"{image.GetSize()} vs {mask.GetSize()}"
        )
    spacing = (
        float(row["final_spacing_x_mm"]),
        float(row["final_spacing_y_mm"]),
    )
    origin = (
        float(row["final_origin_x_mm"]),
        float(row["final_origin_y_mm"]),
    )
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    mask.CopyInformation(image)
    return image, mask


def extract_features(
    image_path: Path,
    mask_path: Path,
    row,
    extractor,
) -> dict:
    image, mask = read_with_geometry(image_path, mask_path, row)
    result = extractor.execute(image, mask)
    return {
        key: value
        for key, value in result.items()
        if not key.startswith("diagnostics_")
    }


def output_paths_for_row(row):
    return Path(row["new_png_path"]), Path(row["new_mask_path"])


def parse_censored_numeric(series: pd.Series) -> pd.Series:
    values = (
        series.astype(str)
        .str.replace("＜", "<", regex=False)
        .str.strip()
        .replace({"nan": np.nan, "None": np.nan, "": np.nan})
    )
    censored = values.str.startswith("<", na=False)
    numeric = pd.to_numeric(values, errors="coerce")
    thresholds = pd.to_numeric(
        values.where(censored).str.replace("<", "", regex=False),
        errors="coerce",
    )
    numeric.loc[censored] = thresholds.loc[censored] / math.sqrt(2)
    return numeric


def transform_r2_masks(
    manifest: pd.DataFrame,
    sample: pd.DataFrame,
    r2_transformed_root: Path,
):
    e_module = E
    transformed_rows = []
    for _, sample_row in sample.iterrows():
        matches = manifest.loc[
            (manifest["label"] == int(sample_row["label"]))
            & (manifest["image"] == sample_row["image"])
        ]
        if len(matches) != 1:
            raise ValueError(f"Cannot uniquely find manifest row: {sample_row.image}")
        row = matches.iloc[0]
        group_name = "RMPP" if int(row["label"]) == 1 else "nonRMPP"
        old_mask_path = (
            R2_MASK_ROOT
            / group_name
            / "02_Mask_R2"
            / row["image"]
        )
        if not old_mask_path.exists():
            raise FileNotFoundError(old_mask_path)

        old_mask = np.asarray(
            Image.open(old_mask_path).convert("L")
        ).copy()
        if old_mask.shape != (1024, 1024):
            raise ValueError(
                f"R2 mask must be 1024 x 1024: {old_mask_path} {old_mask.shape}"
            )
        dicom = pydicom.dcmread(row["dicom_path"], stop_before_pixels=True)
        imager_spacing = e_module.spacing_pair(dicom.ImagerPixelSpacing)
        old_geometry = e_module.legacy_geometry(dicom, imager_spacing)
        new_geometry = {
            "final_spacing_x_mm": float(row["final_spacing_x_mm"]),
            "final_spacing_y_mm": float(row["final_spacing_y_mm"]),
            "final_origin_x_mm": float(row["final_origin_x_mm"]),
            "final_origin_y_mm": float(row["final_origin_y_mm"]),
        }
        new_mask = e_module.transform_mask(
            (old_mask > 0).astype(np.uint8),
            old_geometry,
            new_geometry,
        )
        output_dir = (
            r2_transformed_root / group_name / "03_Binarized"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / row["image"]
        Image.fromarray((new_mask > 0).astype(np.uint8)).save(output_path)
        transformed_rows.append(
            {
                "image": row["image"],
                "label": int(row["label"]),
                "source_r2_mask": str(old_mask_path),
                "transformed_r2_mask": str(output_path),
                "r2_mask_foreground_fraction": float(new_mask.mean()),
            }
        )
    return pd.DataFrame(transformed_rows)


def icc_a1(ratings: pd.DataFrame) -> float:
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


def calculate_icc(
    manifest: pd.DataFrame,
    r1_features: pd.DataFrame,
    r2_features: pd.DataFrame,
) -> pd.DataFrame:
    merged = r1_features.merge(
        r2_features,
        on=["label", "image"],
        suffixes=("_R1", "_R2"),
        validate="one_to_one",
    )
    feature_names = [
        column.replace("original_", "")
        for column in r1_features.columns
        if column.startswith("original_")
    ]
    rows = []
    for feature in feature_names:
        r1_column = f"original_{feature}_R1"
        r2_column = f"original_{feature}_R2"
        if r1_column not in merged or r2_column not in merged:
            continue
        ratings = merged[[r1_column, r2_column]].dropna()
        if len(ratings) < 10:
            continue
        rows.append(
            {
                "feature": feature,
                "ICC_A1": icc_a1(ratings),
                "n_pairs": len(ratings),
            }
        )
    icc = pd.DataFrame(rows).sort_values(
        "ICC_A1", ascending=False
    ).reset_index(drop=True)
    icc["pass_ICC"] = icc["ICC_A1"] > ICC_THRESHOLD
    return icc


def build_analysis_data(
    cohort: pd.DataFrame,
    manifest: pd.DataFrame,
    features: pd.DataFrame,
    icc: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_columns = [
        column for column in features.columns if column.startswith("original_")
    ]
    passed = icc.loc[icc["pass_ICC"], "feature"].tolist()
    passed_columns = [f"original_{feature}" for feature in passed]

    metadata_columns = [
        "label",
        "image",
        "姓名",
        "住院号",
        "性别",
        "age",
        "CRP",
        "WBC",
        "LDH",
    ]
    present_metadata = [
        column for column in metadata_columns if column in cohort.columns
    ]
    cohort_meta = cohort[present_metadata].copy()
    for column in ["age", "CRP", "WBC", "LDH"]:
        if column in cohort_meta.columns:
            cohort_meta[column] = parse_censored_numeric(
                cohort_meta[column]
            )
    manifest_fields = manifest[
        [
            "label",
            "image",
            "manufacturer",
            "manufacturer_model",
            "imager_row_spacing_mm",
            "imager_column_spacing_mm",
            "final_spacing_x_mm",
            "final_spacing_y_mm",
            "final_origin_x_mm",
            "final_origin_y_mm",
        ]
    ].copy()
    manifest_fields["batch_model"] = (
        manifest_fields["manufacturer"].astype(str)
        + " | "
        + manifest_fields["manufacturer_model"].astype(str)
    )
    features_small = features[["label", "image"] + passed_columns].copy()
    merged = (
        cohort_meta.merge(
            manifest_fields,
            on=["label", "image"],
            how="left",
            validate="one_to_one",
        )
        .merge(
            features_small,
            on=["label", "image"],
            how="left",
            validate="one_to_one",
        )
    )
    if merged[passed_columns].isna().any().any():
        missing = merged.loc[
            merged[passed_columns].isna().any(axis=1), ["label", "image"]
        ]
        raise ValueError(f"Missing radiomics features after merge: {missing}")
    full_features = cohort_meta.merge(
        manifest_fields,
        on=["label", "image"],
        how="left",
        validate="one_to_one",
    ).merge(
        features[["label", "image"] + feature_columns],
        on=["label", "image"],
        how="left",
        validate="one_to_one",
    )
    return merged, full_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--force-r1",
        action="store_true",
        help="Re-extract all R1 features even if the R1 workbook exists.",
    )
    parser.add_argument(
        "--force-r2",
        action="store_true",
        help="Re-extract all R2 features even if the R2 workbook exists.",
    )
    parser.add_argument(
        "--audit-dir",
        default=str(OUTPUT_DIR),
        help="Directory containing the E manifest and receiving branch outputs.",
    )
    parser.add_argument(
        "--branch-slug",
        default="imager_spacing",
        help="Filename slug for branch-specific feature and ICC outputs.",
    )
    parser.add_argument(
        "--r2-root",
        default=str(R2_TRANSFORMED_ROOT),
        help="Directory for transformed R2 masks.",
    )
    args = parser.parse_args()

    output_dir = Path(args.audit_dir)
    r2_transformed_root = Path(args.r2_root)
    default_branch = args.branch_slug == "imager_spacing"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(output_dir / "01_E_processing_manifest.csv")
    cohort = pd.read_excel(COHORT_FILE)
    cohort = cohort.loc[cohort["label"].isin([0, 1])].copy()
    cohort["label"] = cohort["label"].astype(int)
    cohort = cohort.merge(
        manifest[["label", "image"]],
        on=["label", "image"],
        how="inner",
        validate="one_to_one",
    ).reset_index(drop=True)
    if len(cohort) != len(manifest):
        raise ValueError(
            f"Clinical cohort rows {len(cohort)} do not match "
            f"manifest rows {len(manifest)}"
        )
    extractor = featureextractor.RadiomicsFeatureExtractor(str(CONFIG_FILE))

    r1_suffix = "_pilot" if args.limit else ""
    r1_output = output_dir / (
        f"02_R1_features_imager_spacing{r1_suffix}.xlsx"
        if default_branch
        else f"02_R1_features_{args.branch_slug}{r1_suffix}.xlsx"
    )
    rows = manifest if args.limit is None else manifest.head(args.limit)
    if r1_output.exists() and not args.force_r1:
        print(f"Loading existing R1 features: {r1_output}")
        r1_features = pd.read_excel(r1_output)
    else:
        feature_rows = []
        for index, row in rows.iterrows():
            image_path, mask_path = output_paths_for_row(row)
            features = extract_features(image_path, mask_path, row, extractor)
            feature_rows.append(
                {
                    "label": int(row["label"]),
                    "image": row["image"],
                    **features,
                }
            )
            print(
                f"[R1 {index + 1:03d}/{len(rows):03d}] {row['image']}"
            )
        r1_features = pd.DataFrame(feature_rows)
    r1_features.to_excel(
        r1_output,
        index=False,
    )
    if args.limit:
        return

    sample = pd.read_excel(
        ICC_SAMPLE_DIR / "sampled_50_cases_radiomics.xlsx",
        usecols=["label", "image"],
    )
    sample["label"] = sample["label"].astype(int)
    sample = sample.merge(
        manifest[["label", "image"]],
        on=["label", "image"],
        how="inner",
        validate="one_to_one",
    ).reset_index(drop=True)
    if len(sample) < 10:
        raise ValueError(
            f"Only {len(sample)} ICC sample cases occur in this branch"
        )
    r2_output = output_dir / (
        "02_R2_features_imager_spacing.xlsx"
        if default_branch
        else f"02_R2_features_{args.branch_slug}.xlsx"
    )
    if r2_output.exists() and not args.force_r2:
        print(f"Loading existing R2 features: {r2_output}")
        r2_features = pd.read_excel(r2_output)
    else:
        transformed_r2 = transform_r2_masks(
            manifest,
            sample,
            r2_transformed_root,
        )
        transformed_r2.to_excel(
            output_dir / "02_R2_transformed_masks_manifest.xlsx",
            index=False,
        )
        r2_rows = []
        for index, sample_row in sample.iterrows():
            manifest_row = manifest.loc[
                (manifest["label"] == int(sample_row["label"]))
                & (manifest["image"] == sample_row["image"])
            ].iloc[0]
            group_name = "RMPP" if int(sample_row["label"]) == 1 else "nonRMPP"
            r2_mask = (
                r2_transformed_root
                / group_name
                / "03_Binarized"
                / sample_row["image"]
            )
            image_path = Path(manifest_row["new_png_path"])
            features = extract_features(
                image_path, r2_mask, manifest_row, extractor
            )
            r2_rows.append(
                {
                    "label": int(sample_row["label"]),
                    "image": sample_row["image"],
                    **features,
                }
            )
            print(
                f"[R2 {index + 1:03d}/{len(sample):03d}] "
                f"{sample_row['image']}"
            )
        r2_features = pd.DataFrame(r2_rows)
        r2_features.to_excel(r2_output, index=False)

    icc = calculate_icc(manifest, r1_features, r2_features)
    icc_output = output_dir / (
        "03_ICC_results_imager_spacing.xlsx"
        if default_branch
        else f"03_ICC_results_{args.branch_slug}.xlsx"
    )
    icc.to_excel(
        icc_output,
        index=False,
    )
    icc_summary = {
        "sample_n": int(len(sample)),
        "feature_n": int(len(icc)),
        "pass_n": int(icc["pass_ICC"].sum()),
        "pass_percent": float(icc["pass_ICC"].mean() * 100),
        "mean_icc": float(icc["ICC_A1"].mean()),
        "median_icc": float(icc["ICC_A1"].median()),
        "min_icc": float(icc["ICC_A1"].min()),
        "max_icc": float(icc["ICC_A1"].max()),
        "threshold": ICC_THRESHOLD,
    }
    icc_summary_output = output_dir / (
        "03_ICC_summary_imager_spacing.json"
        if default_branch
        else f"03_ICC_summary_{args.branch_slug}.json"
    )
    icc_summary_output.write_text(
        json.dumps(icc_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    analysis, full_features = build_analysis_data(
        cohort, manifest, r1_features, icc
    )
    analysis_output = DEFAULT_PIXEL_MODEL_DATA
    analysis_output.parent.mkdir(parents=True, exist_ok=True)
    full_features_output = output_dir / (
        "04_ImagerSpacing_all_features_931.xlsx"
        if default_branch
        else f"04_all_features_{args.branch_slug}.xlsx"
    )
    analysis.to_excel(analysis_output, index=False)
    full_features.to_excel(full_features_output, index=False)
    print(json.dumps(icc_summary, ensure_ascii=False, indent=2))
    print(
        f"Saved analysis dataset: {analysis_output}"
    )


if __name__ == "__main__":
    main()
