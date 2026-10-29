u"""Usage:
python H-3.copy_sampled_files.py
"""
import os, sys, shutil
import pandas as pd
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_ROOT))
from pipeline_paths import IMAGE_ROOT, OUTPUT_ROOT

MAPPING_FILE = OUTPUT_ROOT / "Reproducibility" / "sampled_50_cases_file_mapping.xlsx"
SOURCE_ROOT = IMAGE_ROOT
OUTPUT_DIR = OUTPUT_ROOT / "Reproducibility" / "ROI_review_50cases"

SUBFOLDERS = ['00_DCM', '01_PNG', '02_Mask', '03_Binarized']
LABEL_MAP  = {0: 'nonRMPP', 1: 'RMPP'}


print('=' * 60)
print('[1] Loading file mapping ...')
mapping = pd.read_excel(MAPPING_FILE)
print(f'    {len(mapping)} cases')


print()
print('[2] Scanning source directories ...')

pinyin_index = {}
chinese_index = {}

for group_dir in ['nonRMPP', 'RMPP']:
    for sub in SUBFOLDERS:
        dirpath = os.path.join(SOURCE_ROOT, group_dir, sub)
        if not os.path.isdir(dirpath):
            print(f'    WARNING: not found -> {dirpath}')
            continue
        for fname in os.listdir(dirpath):
            full = os.path.join(dirpath, fname)
            base = os.path.splitext(fname)[0]

            if sub in ('01_PNG', '03_Binarized'):
                pinyin_index.setdefault(base, {})[sub] = full
            else:
                chinese_index.setdefault(base, {})[sub] = full

print(f'    Pinyin index  : {len(pinyin_index)} entries')
print(f'    Chinese index : {len(chinese_index)} entries')


print()
print('[3] Creating output folder structure ...')

for group_dir in LABEL_MAP.values():
    for sub in SUBFOLDERS:
        os.makedirs(os.path.join(OUTPUT_DIR, group_dir, sub), exist_ok=True)

print(f'    {OUTPUT_DIR}')
for group_dir in LABEL_MAP.values():
    print(f'      {group_dir}/')
    for sub in SUBFOLDERS:
        print(f'        {sub}/')


print()
print('[4] Matching & copying files ...')

stats = {'copied': 0, 'missing': 0, 'skipped': 0}
missing_list = []

for _, row in mapping.iterrows():
    img_file = row['image']
    pinyin = os.path.splitext(img_file)[0]
    chinese = row['姓名']
    lbl = int(row['label'])
    group_dir = LABEL_MAP[lbl]


    for sub in ['01_PNG', '03_Binarized']:
        src = pinyin_index.get(pinyin, {}).get(sub, '')
        if src and os.path.isfile(src):
            dst_dir = os.path.join(OUTPUT_DIR, group_dir, sub)
            dst = os.path.join(dst_dir, os.path.basename(src))
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                stats['copied'] += 1
            else:
                stats['skipped'] += 1
        else:
            stats['missing'] += 1
            missing_list.append((img_file, chinese, group_dir, sub))


    for sub in ['00_DCM', '02_Mask']:
        src = chinese_index.get(chinese, {}).get(sub, '')
        if src and os.path.isfile(src):
            dst_dir = os.path.join(OUTPUT_DIR, group_dir, sub)
            dst = os.path.join(dst_dir, os.path.basename(src))
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                stats['copied'] += 1
            else:
                stats['skipped'] += 1
        else:
            stats['missing'] += 1
            missing_list.append((img_file, chinese, group_dir, sub))

total = len(mapping) * 4
print(f'    Copied : {stats["copied"]}/{total}')
print(f'    Skipped: {stats["skipped"]}/{total}  (already exists)')
print(f'    Missing: {stats["missing"]}/{total}')

if missing_list:
    print()
    print('    Missing files:')
    for img, name, grp, sub in missing_list[:20]:
        print(f'      {grp}/{sub}/  <-  {img}  ({name})')
    if len(missing_list) > 20:
        print(f'      ... and {len(missing_list)-20} more')


print()
print('[5] Final folder contents:')

for group_dir in LABEL_MAP.values():
    for sub in SUBFOLDERS:
        d = os.path.join(OUTPUT_DIR, group_dir, sub)
        n = len(os.listdir(d)) if os.path.isdir(d) else 0
        print(f'    {group_dir}/{sub}/  ->  {n} files')

print()
print('Done!')
