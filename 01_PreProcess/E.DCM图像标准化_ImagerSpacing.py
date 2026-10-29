from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from pypinyin import lazy_pinyin


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_ROOT))

from pipeline_paths import (
    DEFAULT_CLINICAL_931,
    IMAGE_ROOT,
    OUTPUT_ROOT,
    PIXEL_PREPROCESS_ROOT,
)

COHORT_FILE = DEFAULT_CLINICAL_931
GROUPS = {
    1: IMAGE_ROOT / "RMPP",
    0: IMAGE_ROOT / "nonRMPP",
}
OUTPUT_NAME = "05_ImagerSpacing"
OUTPUT_NAMES = {
    "imager": "05_ImagerSpacing",
    "pixel": "06_PixelSpacing",
}
TARGET_SPACING_MM = 0.5
TARGET_SIZE = 1024


def apply_window(image: np.ndarray, center: float, width: float) -> np.ndarray:
    image = np.clip(image, center - width / 2, center + width / 2)
    image -= center - width / 2
    image /= width
    image *= 255.0
    return image.astype(np.uint8)


def normalize_by_percentile(
    image: np.ndarray, low: float = 1, high: float = 99
) -> np.ndarray:
    image = np.clip(image, np.percentile(image, low), np.percentile(image, high))
    image -= image.min()
    span = float(image.max() - image.min())
    image = image / (span if span > 0 else 1.0) * 255.0
    return image.astype(np.uint8)


def window_intensity(image: np.ndarray, dataset) -> np.ndarray:
    if "WindowCenter" in dataset and "WindowWidth" in dataset:
        center = dataset.WindowCenter
        width = dataset.WindowWidth
        if isinstance(center, pydicom.multival.MultiValue):
            center = center[0]
        if isinstance(width, pydicom.multival.MultiValue):
            width = width[0]
        return apply_window(image.astype(np.float32), float(center), float(width))
    return normalize_by_percentile(image.astype(np.float32))


def should_invert_image(dataset, image: np.ndarray, threshold: int = 128) -> bool:
    photometric = str(
        getattr(dataset, "PhotometricInterpretation", "")
    ).strip().upper()
    lut_shape = str(getattr(dataset, "PresentationLUTShape", "")).strip().upper()
    if photometric == "MONOCHROME1":
        return True
    if photometric == "MONOCHROME2":
        return False
    if lut_shape == "INVERSE":
        return True
    if lut_shape == "IDENTITY":
        return False
    border = np.concatenate(
        [
            image[:2].ravel(),
            image[-2:].ravel(),
            image[:, :2].ravel(),
            image[:, -2:].ravel(),
        ]
    )
    return float(np.median(border)) > threshold


def spacing_pair(value) -> tuple[float, float]:
    if value is None:
        raise ValueError("Required spacing field is missing")
    row, column = float(value[0]), float(value[1])
    if row <= 0 or column <= 0:
        raise ValueError(f"Invalid spacing values: {value}")
    return row, column


def resample_by_spacing(
    image: np.ndarray, source_row: float, source_column: float
) -> np.ndarray:
    factors = (
        source_row / TARGET_SPACING_MM,
        source_column / TARGET_SPACING_MM,
    )
    new_h = int(image.shape[0] * factors[0])
    new_w = int(image.shape[1] * factors[1])
    if new_h <= 0 or new_w <= 0:
        raise ValueError(f"Invalid resampled size: {new_h} x {new_w}")
    return cv2.resize(
        image, (new_w, new_h), interpolation=cv2.INTER_AREA
    )


def resize_with_padding(image: np.ndarray) -> tuple[np.ndarray, dict]:
    old_h, old_w = image.shape[:2]
    scale = min(TARGET_SIZE / old_w, TARGET_SIZE / old_h)
    new_w = int(old_w * scale)
    new_h = int(old_h * scale)
    resized = cv2.resize(
        image, (new_w, new_h), interpolation=cv2.INTER_AREA
    )
    canvas = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)
    top = (TARGET_SIZE - new_h) // 2
    left = (TARGET_SIZE - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas, {
        "content_width": new_w,
        "content_height": new_h,
        "padding_left": left,
        "padding_top": top,
    }


def normalized_pinyin(value: str) -> str:
    return "".join(lazy_pinyin(str(value))).lower().replace(" ", "")


def image_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def array_digest_from_path(path: Path) -> str:
    return image_hash(read_gray(path))


def find_existing_source(
    group_root: Path,
    other_group_root: Path,
    subfolder: str,
    image_name: str,
) -> Path:
    primary = group_root / subfolder / image_name
    if primary.exists():
        return primary
    fallback = other_group_root / subfolder / image_name
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"Source not found in either group: {subfolder}/{image_name}"
    )


def legacy_standardized_image(dataset) -> np.ndarray:
    image = window_intensity(dataset.pixel_array, dataset)
    if "PixelSpacing" in dataset:
        row, column = spacing_pair(dataset.PixelSpacing)
        image = resample_by_spacing(image, row, column)
    if should_invert_image(dataset, image):
        image = 255 - image
    image, _ = resize_with_padding(image)
    return image


def new_standardized_image(
    dataset,
    spacing_source: str = "imager",
) -> tuple[np.ndarray, dict]:
    image = window_intensity(dataset.pixel_array, dataset)
    attribute = (
        "ImagerPixelSpacing"
        if spacing_source == "imager"
        else "PixelSpacing"
    )
    row, column = spacing_pair(getattr(dataset, attribute, None))
    resampled = resample_by_spacing(image, row, column)
    if should_invert_image(dataset, resampled):
        resampled = 255 - resampled
    image, geometry = resize_with_padding(resampled)

    original_h, original_w = dataset.pixel_array.shape[:2]
    final_spacing_x = column * original_w / geometry["content_width"]
    final_spacing_y = row * original_h / geometry["content_height"]
    geometry.update(
        {
            "source_row_spacing_mm": row,
            "source_column_spacing_mm": column,
            "final_spacing_x_mm": final_spacing_x,
            "final_spacing_y_mm": final_spacing_y,
            "final_origin_x_mm": -geometry["padding_left"] * final_spacing_x,
            "final_origin_y_mm": -geometry["padding_top"] * final_spacing_y,
            "resampled_width": resampled.shape[1],
            "resampled_height": resampled.shape[0],
        }
    )
    return image, geometry


def legacy_geometry(dataset, imager_spacing: tuple[float, float]) -> dict:
    original_h, original_w = int(dataset.Rows), int(dataset.Columns)
    if "PixelSpacing" in dataset:
        old_row, old_column = spacing_pair(dataset.PixelSpacing)
        old_intermediate_h = int(original_h * old_row / TARGET_SPACING_MM)
        old_intermediate_w = int(original_w * old_column / TARGET_SPACING_MM)
    else:
        old_row, old_column = imager_spacing
        old_intermediate_h = original_h
        old_intermediate_w = original_w
    scale = min(
        TARGET_SIZE / old_intermediate_w,
        TARGET_SIZE / old_intermediate_h,
    )
    old_content_w = int(old_intermediate_w * scale)
    old_content_h = int(old_intermediate_h * scale)
    old_spacing_x = old_column * original_w / old_content_w
    old_spacing_y = old_row * original_h / old_content_h
    left = (TARGET_SIZE - old_content_w) // 2
    top = (TARGET_SIZE - old_content_h) // 2
    return {
        "old_spacing_x_mm": old_spacing_x,
        "old_spacing_y_mm": old_spacing_y,
        "old_origin_x_mm": -left * old_spacing_x,
        "old_origin_y_mm": -top * old_spacing_y,
        "old_content_width": old_content_w,
        "old_content_height": old_content_h,
        "old_padding_left": left,
        "old_padding_top": top,
    }


def transform_mask(
    old_mask: np.ndarray,
    old_geometry: dict,
    new_geometry: dict,
) -> np.ndarray:
    if old_mask.shape != (TARGET_SIZE, TARGET_SIZE):
        raise ValueError(f"Old mask must be 1024 x 1024, got {old_mask.shape}")
    old_image = sitk.GetImageFromArray((old_mask > 0).astype(np.uint8))
    old_image.SetSpacing(
        (
            old_geometry["old_spacing_x_mm"],
            old_geometry["old_spacing_y_mm"],
        )
    )
    old_image.SetOrigin(
        (
            old_geometry["old_origin_x_mm"],
            old_geometry["old_origin_y_mm"],
        )
    )
    resampled = sitk.Resample(
        old_image,
        (TARGET_SIZE, TARGET_SIZE),
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        (
            new_geometry["final_origin_x_mm"],
            new_geometry["final_origin_y_mm"],
        ),
        (
            new_geometry["final_spacing_x_mm"],
            new_geometry["final_spacing_y_mm"],
        ),
        old_image.GetDirection(),
        0,
        sitk.sitkUInt8,
    )
    return (sitk.GetArrayFromImage(resampled) > 0).astype(np.uint8)


def build_mapping(cohort: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dicom_records = []
    diagnostics = []
    for source_label, group_root in GROUPS.items():
        decoded = 0
        failed = 0
        for path in sorted((group_root / "00_DCM").glob("*.dcm")):
            try:
                dataset = pydicom.dcmread(path)
                legacy_image = legacy_standardized_image(dataset)
            except Exception as error:
                failed += 1
                diagnostics.append(
                    {
                        "label": source_label,
                        "dicom_name": path.name,
                        "status": "legacy_decode_failed",
                        "detail": str(error),
                    }
                )
                continue
            dicom_records.append(
                {
                    "path": path,
                    "source_label": source_label,
                    "stem": path.stem,
                    "pinyin": normalized_pinyin(path.stem),
                    "legacy_hash": image_hash(legacy_image),
                }
            )
            decoded += 1
        diagnostics.append(
            {
                "label": source_label,
                "status": "inventory",
                "dicom_decoded": decoded,
                "dicom_failed": failed,
            }
        )

    dicoms_by_hash = defaultdict(list)
    dicoms_by_name = defaultdict(list)
    dicoms_by_pinyin = defaultdict(list)
    for record in dicom_records:
        dicoms_by_hash[record["legacy_hash"]].append(record)
        dicoms_by_name[record["stem"]].append(record)
        dicoms_by_pinyin[record["pinyin"]].append(record)

    assigned_paths = set()
    records = []
    for _, row in cohort.iterrows():
        label = int(row["label"])
        group_root = GROUPS[label]
        other_group_root = GROUPS[1 - label]
        image_name = row["image"]
        image_path = find_existing_source(
            group_root, other_group_root, "01_PNG", image_name
        )
        mask_path = find_existing_source(
            group_root, other_group_root, "03_Binarized", image_name
        )
        current_digest = array_digest_from_path(image_path)

        candidates = [
            record
            for record in dicoms_by_hash.get(current_digest, [])
            if record["path"] not in assigned_paths
        ]
        match_method = "legacy_image_hash"
        if len(candidates) != 1:
            candidates = [
                record
                for record in dicoms_by_name.get(
                    str(row["姓名"]).strip(), []
                )
                if record["path"] not in assigned_paths
            ]
            match_method = "chinese_name"
        if len(candidates) != 1:
            candidates = [
                record
                for record in dicoms_by_pinyin.get(
                    Path(image_name).stem.lower().split("_")[0], []
                )
                if record["path"] not in assigned_paths
            ]
            match_method = "pinyin_fallback"
        if len(candidates) > 1:
            preferred = [
                record
                for record in candidates
                if record["source_label"] == label
            ]
            if len(preferred) == 1:
                candidates = preferred
        if len(candidates) != 1:
            raise ValueError(
                f"Cannot uniquely map label={label} image={image_name} "
                f"name={row['姓名']} candidates="
                f"{[(r['path'].name, r['source_label']) for r in candidates]}"
            )
        matched = candidates[0]
        dicom_path = matched["path"]
        assigned_paths.add(dicom_path)

        dataset = pydicom.dcmread(dicom_path, stop_before_pixels=True)
        imager_spacing = spacing_pair(
            getattr(dataset, "ImagerPixelSpacing", None)
        )
        records.append(
            {
                "label": label,
                "image": image_name,
                "name": row["姓名"],
                "hospital_id": row["住院号"],
                "dicom_path": str(dicom_path),
                "dicom_source_label": int(matched["source_label"]),
                "source_png_path": str(image_path),
                "source_mask_path": str(mask_path),
                "manufacturer": str(
                    getattr(dataset, "Manufacturer", "Not recorded")
                ).strip(' "'),
                "manufacturer_model": str(
                    getattr(
                        dataset,
                        "ManufacturerModelName",
                        "Not recorded",
                    )
                ).strip(' "'),
                "imager_row_spacing_mm": imager_spacing[0],
                "imager_column_spacing_mm": imager_spacing[1],
                "pixel_spacing_available": "PixelSpacing" in dataset,
                "pixel_row_spacing_mm": (
                    spacing_pair(dataset.PixelSpacing)[0]
                    if "PixelSpacing" in dataset
                    else np.nan
                ),
                "pixel_column_spacing_mm": (
                    spacing_pair(dataset.PixelSpacing)[1]
                    if "PixelSpacing" in dataset
                    else np.nan
                ),
                "match_method": match_method,
                "legacy_hash": matched["legacy_hash"],
            }
        )

    mapping = pd.DataFrame(records).sort_values(
        ["label", "image"], kind="stable"
    ).reset_index(drop=True)
    if len(mapping) != len(cohort):
        raise ValueError(f"Mapped {len(mapping)} rows, expected {len(cohort)}")
    diagnostics.append(
        {
            "label": "all",
            "status": "mapping_complete",
            "mapped_rows": len(mapping),
            "assigned_dicom_files": len(assigned_paths),
            "cross_source_rows": int(
                (mapping["label"] != mapping["dicom_source_label"]).sum()
            ),
        }
    )
    return mapping, pd.DataFrame(diagnostics)


def process_group(
    mapping: pd.DataFrame,
    cohort: pd.DataFrame,
    limit: int | None,
    output_name: str,
    spacing_source: str,
) -> pd.DataFrame:
    results = []
    rows = mapping if limit is None else mapping.head(limit)
    for _, row in rows.iterrows():
        label = int(row["label"])
        group_root = GROUPS[label]
        output_root = group_root / output_name
        png_out = output_root / "01_PNG"
        mask_out = output_root / "02_Binarized"
        png_out.mkdir(parents=True, exist_ok=True)
        mask_out.mkdir(parents=True, exist_ok=True)

        dataset = pydicom.dcmread(row["dicom_path"])
        new_image, new_geometry = new_standardized_image(
            dataset,
            spacing_source=spacing_source,
        )
        imager_spacing = spacing_pair(dataset.ImagerPixelSpacing)
        old_geometry = legacy_geometry(dataset, imager_spacing)
        old_mask = read_gray(Path(row["source_mask_path"]))
        old_mask = (old_mask > 0).astype(np.uint8)
        new_mask = transform_mask(old_mask, old_geometry, new_geometry)

        output_name = Path(row["image"]).name
        cv2.imwrite(str(png_out / output_name), new_image)
        cv2.imwrite(str(mask_out / output_name), new_mask)

        results.append(
            {
                **row.to_dict(),
                **old_geometry,
                **new_geometry,
                "new_image_hash": image_hash(new_image),
                "old_mask_foreground_fraction": float(old_mask.mean()),
                "new_mask_foreground_fraction": float(new_mask.mean()),
                "new_png_path": str(png_out / output_name),
                "new_mask_path": str(mask_out / output_name),
            }
        )
        print(
            f"[{len(results):03d}/{len(rows):03d}] "
            f"{output_name} -> {output_name}"
        )
    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit-mapping",
        action="store_true",
        help="Only map DICOM files to cohort images; do not write new PNGs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N mapped cases, for a pilot run.",
    )
    parser.add_argument(
        "--spacing-source",
        choices=["imager", "pixel"],
        default="imager",
        help="DICOM spacing field used for the new image branch.",
    )
    parser.add_argument(
        "--require-pixel-spacing",
        action="store_true",
        help="Restrict processing to cases with DICOM PixelSpacing.",
    )
    args = parser.parse_args()

    cohort = pd.read_excel(COHORT_FILE)
    cohort = cohort.loc[cohort["label"].isin([0, 1])].copy()
    cohort["label"] = cohort["label"].astype(int)
    cohort["image"] = cohort["image"].astype(str)
    cohort = cohort.reset_index(drop=True)
    if len(cohort) != 931:
        raise ValueError(f"Expected 931 binary-cohort rows, got {len(cohort)}")

    mapping, diagnostics = build_mapping(cohort)
    if args.spacing_source == "pixel":
        mapping = mapping.loc[mapping["pixel_spacing_available"]].copy()
    elif args.require_pixel_spacing:
        mapping = mapping.loc[mapping["pixel_spacing_available"]].copy()
    mapping = mapping.reset_index(drop=True)
    if mapping.empty:
        raise ValueError("No cases remain after spacing-source filtering")

    output_name = OUTPUT_NAMES[args.spacing_source]
    branch_slug = (
        "imager_spacing"
        if args.spacing_source == "imager"
        else "pixel_spacing_497"
    )
    audit_dir = (
        OUTPUT_ROOT / "ImagerSpacingPreprocessing"
        if args.spacing_source == "imager"
        else PIXEL_PREPROCESS_ROOT
    )
    audit_dir.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(
        audit_dir / "00_DICOM_mapping_internal.csv",
        index=False,
    )
    diagnostics.to_csv(
        audit_dir / "00_DICOM_mapping_diagnostics.csv", index=False
    )
    print(mapping["match_method"].value_counts().to_string())
    print(diagnostics.to_string(index=False))
    if args.audit_mapping:
        return

    processed = process_group(
        mapping,
        cohort,
        args.limit,
        output_name=output_name,
        spacing_source=args.spacing_source,
    )
    manifest_path = audit_dir / (
        "01_E_processing_manifest_pilot.csv"
        if args.limit
        else "01_E_processing_manifest.csv"
    )
    processed.to_csv(manifest_path, index=False)
    summary = {
        "cohort_n": int(len(cohort)),
        "processed_n": int(len(processed)),
        "eligible_n": int(len(mapping)),
        "target_spacing_mm": TARGET_SPACING_MM,
        "target_size": TARGET_SIZE,
        "spacing_source": (
            "ImagerPixelSpacing"
            if args.spacing_source == "imager"
            else "PixelSpacing"
        ),
        "branch_slug": branch_slug,
        "output_dirs": {
            str(label): str(root / output_name)
            for label, root in GROUPS.items()
        },
        "mask_method": (
            "Existing reviewed binary masks transformed to the new grid with "
            "nearest-neighbor physical-coordinate resampling."
        ),
    }
    (audit_dir / "01_E_processing_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
