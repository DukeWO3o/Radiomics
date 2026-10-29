import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_ROOT))
from pipeline_paths import DEFAULT_CLINICAL_931, DEFAULT_MODEL_DATA_931, OUTPUT_ROOT

REPRO_DIR = OUTPUT_ROOT / "Reproducibility"
R1 = REPRO_DIR / "sampled_50_cases_radiomics.xlsx"
R2_N = REPRO_DIR / "ROI_review_50cases" / "nonRMPP-2.csv"
R2_P = REPRO_DIR / "ROI_review_50cases" / "RMPP-2.csv"
FULL = DEFAULT_CLINICAL_931
OUT_F = REPRO_DIR
MODEL_DATA_OUT = DEFAULT_MODEL_DATA_931

ICC_THRESHOLD = 0.75


print(">>> 读取 R1 数据...")
r1 = pd.read_excel(R1)
print(f"    R1 shape: {r1.shape}")

print(">>> 读取 R2 数据 (nonRMPP + RMPP)...")
r2_n = pd.read_csv(R2_N)
r2_p = pd.read_csv(R2_P)
r2 = pd.concat([r2_n, r2_p], ignore_index=True)
print(f"    R2 shape: {r2.shape}")


print(">>> 按 image 列对齐 R1 与 R2 ...")
r1["image_key"] = r1["image"].str.replace(".png", "", regex=False).str.lower()
r2["image_key"] = r2["image"].str.replace(".png", "", regex=False).str.lower()

merged = r1.merge(r2, on="image_key", how="inner")
print(f"    匹配成功: {len(merged)} 例")
assert len(merged) == 50, f"ERROR: 期望50例, 实际匹配{len(merged)}例"


r2_feat_cols = [c for c in r2.columns
                if c.startswith("original_")
                and not c.startswith("diagnostics_")]


common_feats = []
for r2_col in r2_feat_cols:
    r1_col = r2_col[len("original_"):]
    if r1_col in r1.columns:
        common_feats.append((r1_col, r2_col))

print(f"    共同影像组学特征数: {len(common_feats)}")


print(">>> 计算 ICC(A,1) ...")

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
    denom = msr + (k - 1) * mse + k * (msc - mse) / n
    if denom == 0:
        return np.nan
    return (msr - mse) / denom

icc_results = []
for r1_col, r2_col in common_feats:
    ratings = merged[[r1_col, r2_col]].dropna()
    if len(ratings) < 10:
        continue
    icc_val = icc_a1(ratings)
    icc_results.append({
        "feature": r1_col,
        "ICC_A1": icc_val,
        "n_pairs": len(ratings),
    })

icc_df = pd.DataFrame(icc_results).sort_values("ICC_A1", ascending=False)
icc_df["pass_ICC"] = icc_df["ICC_A1"] > ICC_THRESHOLD
n_pass = int(icc_df["pass_ICC"].sum())
n_total = len(icc_df)
print(f"    ICC > 0.75 的特征数: {n_pass} / {n_total}")
print(f"    ICC 均值: {icc_df['ICC_A1'].mean():.4f}  |  中位数: {icc_df['ICC_A1'].median():.4f}")
print(f"    ICC 范围: [{icc_df['ICC_A1'].min():.4f}, {icc_df['ICC_A1'].max():.4f}]")


icc_out = OUT_F / "02_ICC_results.xlsx"
icc_df.to_excel(icc_out, index=False)
print(f">>> ICC详细结果已保存: {icc_out}")


lines = []
lines.append("=" * 60)
lines.append("ICC 特征可重复性分析")
lines.append("=" * 60)
lines.append(f"匹配病例数: {len(merged)}")
lines.append(f"分析特征数: {n_total}")
lines.append(f"ICC > 0.75 特征数: {n_pass}")
lines.append(f"ICC > 0.75 特征占比: {n_pass/n_total*100:.1f}%")
lines.append(f"ICC 均值: {icc_df['ICC_A1'].mean():.4f}")
lines.append(f"ICC 中位数: {icc_df['ICC_A1'].median():.4f}")
lines.append(f"ICC 范围: [{icc_df['ICC_A1'].min():.4f}, {icc_df['ICC_A1'].max():.4f}]")
lines.append("")

if n_pass > 0:
    lines.append("ICC > 0.75 的特征:")
    lines.append("-" * 40)
    for _, row in icc_df[icc_df["pass_ICC"]].iterrows():
        lines.append(f"  {row['feature']:50s}  ICC = {row['ICC_A1']:.4f}")
    lines.append("")

failed = icc_df[~icc_df["pass_ICC"]]
if len(failed) > 0:
    lines.append("ICC <= 0.75 的特征 (已剔除):")
    lines.append("-" * 40)
    for _, row in failed.iterrows():
        lines.append(f"  {row['feature']:50s}  ICC = {row['ICC_A1']:.4f}")

summary_txt = "\n".join(lines)
summary_path = OUT_F / "02_ICC_summary.txt"
summary_path.write_text(summary_txt, encoding="utf-8")
print(f">>> ICC汇总已保存: {summary_path}")


print(">>> 读取全量数据...")
full = pd.read_excel(FULL)
print(f"    全量数据 shape: {full.shape}")


meta_cols = ["label", "image", "姓名"]
diag_cols = [c for c in full.columns if c.startswith("diagnostics_")]

all_rf_names = [r1_c for r1_c, _ in common_feats]

pass_rf_names = icc_df[icc_df["pass_ICC"]]["feature"].tolist()

fail_rf_names = [f for f in all_rf_names if f not in pass_rf_names]


rf_present = [c for c in pass_rf_names if c in full.columns]
rf_missing = [c for c in pass_rf_names if c not in full.columns]
fail_rf_in_full = [c for c in fail_rf_names if c in full.columns]


non_rf_cols = meta_cols + diag_cols + [
    c for c in full.columns
    if c not in meta_cols
    and not c.startswith("diagnostics_")
    and c not in all_rf_names
]
non_rf_cols = list(dict.fromkeys(non_rf_cols))

print(f"    保留的影像组学特征: {len(rf_present)}")
print(f"    剔除的影像组学特征: {len(fail_rf_in_full)} ({', '.join(fail_rf_in_full)})")
if rf_missing:
    print(f"    WARNING: 全量缺失的通过特征: {rf_missing}")

final = full[non_rf_cols + rf_present].copy()

binary_mask = final["label"].isin([0, 1])
n_nonrmpp = int((final["label"] == 0).sum())
n_rmpp = int((final["label"] == 1).sum())
n_binary = int(binary_mask.sum())

print(f"\n{'='*50}")
print("最终数据集")
print(f"{'='*50}")
print(f"总行数: {len(final)}")
print(f"二分类有效行数: {n_binary} (非RMPP={n_nonrmpp}, RMPP={n_rmpp})")
print(f"总列数: {len(final.columns)}")
print(f"  - 元数据列: {len(meta_cols)}")
print(f"  - Diagnostics列: {len(diag_cols)}")
print(f"  - 临床变量列: {len(non_rf_cols) - len(meta_cols) - len(diag_cols)}")
print(f"  - 影像组学特征列: {len(rf_present)}")

out_name = f"02_ICC_filtered_临床版-v5-{n_nonrmpp}-{n_rmpp}-{n_binary}.xlsx"
out_path = MODEL_DATA_OUT
final.to_excel(out_path, index=False)
print(f"\n>>> 最终数据已保存: {out_path}")
print("\n*** 全部完成! ***")
