from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_ROOT))
from pipeline_paths import DEFAULT_CLINICAL_931, IMAGE_ROOT, OUTPUT_ROOT

COHORT = DEFAULT_CLINICAL_931
DICOM_DIRS = {
    1: IMAGE_ROOT / "RMPP" / "00_DCM",
    0: IMAGE_ROOT / "nonRMPP" / "00_DCM",
}
OUTPUT_DIR = OUTPUT_ROOT / "Preprocessing"


def spacing_pair(value):
    if value is None:
        return None
    try:
        row, column = (float(value[0]), float(value[1]))
    except (TypeError, ValueError, IndexError):
        return None
    if row <= 0 or column <= 0:
        return None
    return row, column


def find_candidate(rows_by_name, name):
    candidates = rows_by_name.get(str(name).strip(), [])
    if len(candidates) == 1:
        return candidates[0], None
    if len(candidates) > 1:
        return None, f"ambiguous name ({len(candidates)} files)"
    return None, "file not found"


def main() -> None:
    cohort = pd.read_excel(COHORT, usecols=["label", "image", "姓名", "住院号"])
    records = []
    unmatched = []

    for label, directory in DICOM_DIRS.items():
        files_by_name = {}
        for path in directory.glob("*.dcm"):
            files_by_name.setdefault(path.stem, []).append(path)
        subset = cohort[cohort["label"] == label]
        for _, row in subset.iterrows():
            path, reason = find_candidate(files_by_name, row["姓名"])
            if path is None:
                unmatched.append(
                    {
                        "label": label,
                        "image": row["image"],
                        "name": row["姓名"],
                        "reason": reason,
                    }
                )
                continue
            dataset = pydicom.dcmread(path, stop_before_pixels=True)
            pixel = spacing_pair(getattr(dataset, "PixelSpacing", None))
            imager = spacing_pair(getattr(dataset, "ImagerPixelSpacing", None))
            records.append(
                {
                    "label": int(label),
                    "image": row["image"],
                    "hospital_id": row["住院号"],
                    "manufacturer": str(
                        getattr(dataset, "Manufacturer", "Not recorded")
                    ).strip(' "'),
                    "model": str(
                        getattr(dataset, "ManufacturerModelName", "Not recorded")
                    ).strip(' "'),
                    "pixel_row": pixel[0] if pixel else np.nan,
                    "pixel_column": pixel[1] if pixel else np.nan,
                    "imager_row": imager[0] if imager else np.nan,
                    "imager_column": imager[1] if imager else np.nan,
                }
            )

    data = pd.DataFrame(records)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_DIR / "DICOM_spacing_records.csv", index=False)
    pd.DataFrame(unmatched).to_csv(
        OUTPUT_DIR / "DICOM_spacing_unmatched.csv",
        index=False,
    )
    paired = data.dropna(subset=["pixel_row", "imager_row"]).copy()
    paired["row_ratio"] = paired["pixel_row"] / paired["imager_row"]
    paired["column_ratio"] = paired["pixel_column"] / paired["imager_column"]
    paired["mean_ratio"] = paired[["row_ratio", "column_ratio"]].mean(axis=1)
    paired["mean_abs_difference_mm"] = paired[
        ["pixel_row", "pixel_column", "imager_row", "imager_column"]
    ].apply(
        lambda row: (
            abs(row["pixel_row"] - row["imager_row"])
            + abs(row["pixel_column"] - row["imager_column"])
        )
        / 2,
        axis=1,
    )

    print(f"cohort={len(cohort)} mapped={len(data)} unmatched={len(unmatched)}")
    print(f"PixelSpacing available={data['pixel_row'].notna().sum()}")
    print(f"ImagerPixelSpacing available={data['imager_row'].notna().sum()}")
    if unmatched:
        print("\nUnmatched:")
        print(pd.DataFrame(unmatched).to_string(index=False))

    print("\nAvailability by manufacturer:")
    by_vendor = (
        data.assign(
            pixel_available=data["pixel_row"].notna(),
            imager_available=data["imager_row"].notna(),
        )
        .groupby("manufacturer")
        .agg(
            n=("image", "size"),
            pixel_available=("pixel_available", "sum"),
            imager_available=("imager_available", "sum"),
        )
        .sort_values("n", ascending=False)
    )
    by_vendor["pixel_percent"] = by_vendor["pixel_available"] / by_vendor["n"] * 100
    by_vendor["imager_percent"] = (
        by_vendor["imager_available"] / by_vendor["n"] * 100
    )
    print(by_vendor.to_string(float_format=lambda value: f"{value:.2f}"))

    print("\nPaired fields:")
    print(f"n={len(paired)}")
    print(
        "exact_equal="
        f"{np.isclose(paired['pixel_row'], paired['imager_row']).sum()}/"
        f"{len(paired)}"
    )
    for column in ["row_ratio", "column_ratio", "mean_ratio", "mean_abs_difference_mm"]:
        values = paired[column].replace([np.inf, -np.inf], np.nan).dropna()
        print(
            f"{column}: median={values.median():.4f}, "
            f"IQR={values.quantile(0.25):.4f}-{values.quantile(0.75):.4f}, "
            f"range={values.min():.4f}-{values.max():.4f}"
        )

    print("\nPaired differences by manufacturer:")
    by_vendor_pair = (
        paired.groupby("manufacturer")
        .agg(
            n=("image", "size"),
            exact_equal=(
                "mean_ratio",
                lambda values: np.isclose(values, 1.0).sum(),
            ),
            median_ratio=("mean_ratio", "median"),
            min_ratio=("mean_ratio", "min"),
            max_ratio=("mean_ratio", "max"),
            median_abs_difference_mm=("mean_abs_difference_mm", "median"),
        )
        .sort_values("n", ascending=False)
    )
    print(by_vendor_pair.to_string(float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
