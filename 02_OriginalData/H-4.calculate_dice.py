# -*- coding: utf-8 -*-
u'''
=============================================================================
Dice Coefficient Calculation between Two Radiologists' ROI Masks
=============================================================================
For each of the 50 sampled cases:
  - R1 mask:  02_Mask/{pinyin}.png         (original / radiologist 1)
  - R2 mask:  02_Mask - 对照/{Chinese}.png  (control / radiologist 2)

Output:
  - dice_results.xlsx   per-case Dice scores
  - dice_summary.txt    aggregate statistics
=============================================================================
'''
import os, sys
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ======================== CONFIGURATION ========================
MAPPING_FILE = r'E:\Softwares\PycharmProject\Radiomics\02_OriginalData\02_ICC_sample\sampled_50_cases_file_mapping.xlsx'
ROI_DIR      = r'E:\Softwares\PycharmProject\Radiomics\02_OriginalData\02_ICC_sample\ROI_review_50cases'
OUTPUT_DIR   = r'E:\Softwares\PycharmProject\Radiomics\02_OriginalData\02_ICC_sample'

SUBDIR_R1 = '02_Mask'
SUBDIR_R2 = '02_Mask - 对照'
LABEL_MAP = {0: 'DCM', 1: 'DCM-RMPP'}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================================================
# 1. Load mapping
# ================================================================
print('=' * 60)
print(' Dice Coefficient: Inter-Rater ROI Agreement')
print('=' * 60)

mapping = pd.read_excel(MAPPING_FILE)
print(f'\n[1] Loaded {len(mapping)} cases from mapping file')

# ================================================================
# 2. Calculate Dice per case
# ================================================================
print('\n[2] Calculating Dice coefficients ...')

results = []
errors = []

for idx, row in mapping.iterrows():
    pinyin = os.path.splitext(row['image'])[0]
    chinese = row['姓名']
    lbl = int(row['label'])
    group = LABEL_MAP[lbl]

    r1_path = os.path.join(ROI_DIR, group, SUBDIR_R1, pinyin + '.png')
    r2_path = os.path.join(ROI_DIR, group, SUBDIR_R2, chinese + '.png')

    r1_ok = os.path.isfile(r1_path)
    r2_ok = os.path.isfile(r2_path)

    if not (r1_ok and r2_ok):
        missing = []
        if not r1_ok: missing.append('R1')
        if not r2_ok: missing.append('R2')
        errors.append((pinyin, chinese, missing))
        results.append({
            'pinyin': pinyin, 'chinese': chinese, 'label': lbl,
            'group': group, 'dice': np.nan, 'r1_pixels': np.nan,
            'r2_pixels': np.nan, 'intersection': np.nan
        })
        continue

    # Load masks
    mask1 = np.array(Image.open(r1_path)).astype(bool)
    mask2 = np.array(Image.open(r2_path)).astype(bool)

    # Verify same shape
    if mask1.shape != mask2.shape:
        print(f'  WARNING: shape mismatch for {pinyin}: '
              f'{mask1.shape} vs {mask2.shape}')
        errors.append((pinyin, chinese, ['shape_mismatch']))
        results.append({
            'pinyin': pinyin, 'chinese': chinese, 'label': lbl,
            'group': group, 'dice': np.nan, 'r1_pixels': np.nan,
            'r2_pixels': np.nan, 'intersection': np.nan
        })
        continue

    # Dice = 2 * TP / (2*TP + FP + FN) = 2 * |A∩B| / (|A| + |B|)
    intersection = np.logical_and(mask1, mask2).sum()
    r1_px = mask1.sum()
    r2_px = mask2.sum()
    dice = 2.0 * intersection / (r1_px + r2_px) if (r1_px + r2_px) > 0 else 1.0

    results.append({
        'pinyin': pinyin, 'chinese': chinese, 'label': lbl,
        'group': group, 'dice': round(dice, 6),
        'r1_pixels': int(r1_px), 'r2_pixels': int(r2_px),
        'intersection': int(intersection)
    })

    if (idx + 1) % 10 == 0:
        print(f'    Processed {idx+1}/{len(mapping)} ...')

df = pd.DataFrame(results)
print(f'    Done. {len(df)} cases, {len(errors)} errors')

# ================================================================
# 3. Summary statistics
# ================================================================
print('\n[3] Summary statistics ...')

valid = df.dropna(subset=['dice'])
n_valid = len(valid)
n_total = len(df)

summary = {
    'n_cases': n_total,
    'n_valid': n_valid,
    'n_error': len(errors),
    'dice_mean': valid['dice'].mean(),
    'dice_median': valid['dice'].median(),
    'dice_std': valid['dice'].std(),
    'dice_min': valid['dice'].min(),
    'dice_max': valid['dice'].max(),
}

# By label
by_label = valid.groupby('label')['dice'].agg(['mean', 'std', 'min', 'max', 'count'])

print(f'    Valid cases : {n_valid}/{n_total}')
print(f'    Dice mean   : {summary["dice_mean"]:.4f}')
print(f'    Dice median : {summary["dice_median"]:.4f}')
print(f'    Dice std    : {summary["dice_std"]:.4f}')
print(f'    Dice range  : [{summary["dice_min"]:.4f}, {summary["dice_max"]:.4f}]')
print()
print('    By label:')
print(by_label.to_string())

# ================================================================
# 4. Save results
# ================================================================
print('\n[4] Saving results ...')

# 4a. Excel
dice_out = os.path.join(OUTPUT_DIR, 'dice_results.xlsx')
df.to_excel(dice_out, index=False)
print(f'    {dice_out}')

# 4b. Histogram
if n_valid > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Overall histogram
    axes[0].hist(valid['dice'], bins=15, color='steelblue', edgecolor='white',
                 alpha=0.85)
    axes[0].axvline(valid['dice'].mean(), color='red', linestyle='--',
                    linewidth=1.5, label=f"Mean={valid['dice'].mean():.3f}")
    axes[0].set_xlabel('Dice Coefficient')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Dice Distribution (n={n_valid})')
    axes[0].legend()

    # By label
    for lbl, color, name in [(0, '#2196F3', 'non-RMPP'), (1, '#F44336', 'RMPP')]:
        sub = valid[valid['label'] == lbl]['dice']
        if len(sub) > 0:
            axes[1].hist(sub, bins=12, alpha=0.5, color=color, label=name)
    axes[1].set_xlabel('Dice Coefficient')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Dice by Group')
    axes[1].legend()

    plt.tight_layout()
    hist_out = os.path.join(OUTPUT_DIR, 'dice_histogram.png')
    plt.savefig(hist_out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'    {hist_out}')

# 4c. Text report
report_lines = [
    '=' * 60,
    'Dice Coefficient Report - Inter-Rater ROI Agreement',
    '=' * 60,
    '',
    'R1 (original):  ' + SUBDIR_R1,
    'R2 (control):   ' + SUBDIR_R2,
    '',
    f'Total cases:     {n_total}',
    f'Valid pairs:     {n_valid}',
    f'Errors:          {len(errors)}',
    '',
    f'Dice mean   = {summary["dice_mean"]:.4f}',
    f'Dice median = {summary["dice_median"]:.4f}',
    f'Dice std    = {summary["dice_std"]:.4f}',
    f'Dice min    = {summary["dice_min"]:.4f}',
    f'Dice max    = {summary["dice_max"]:.4f}',
    '',
    'By label:',
    by_label.to_string(),
    '',
]

if errors:
    report_lines.append('Errors:')
    for p, c, m in errors:
        report_lines.append(f'  {p} ({c}): missing {m}')

report_lines += [
    '',
    'Interpretation:',
    '  Dice > 0.90 : excellent agreement',
    '  Dice 0.80-0.90 : good agreement',
    '  Dice 0.70-0.80 : moderate agreement',
    '  Dice < 0.70 : poor agreement',
    '',
    'If mean Dice > 0.85-0.90, ROI segmentation is reproducible.',
    '',
]

report_out = os.path.join(OUTPUT_DIR, 'dice_summary.txt')
with open(report_out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))
print(f'    {report_out}')

print()
print('\n'.join(report_lines))
print('Done!')
