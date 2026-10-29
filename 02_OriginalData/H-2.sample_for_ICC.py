import os, sys, shutil
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import pandas as pd


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_ROOT))
from pipeline_paths import DEFAULT_SAMPLING_COHORT, IMAGE_ROOT, OUTPUT_ROOT

RADIOMICS_FILE = DEFAULT_SAMPLING_COHORT
SOURCE_ROOT = IMAGE_ROOT
OUTPUT_DIR = OUTPUT_ROOT / "Reproducibility"

N_SAMPLE    = 50
RANDOM_SEED = 42

LABEL_COL  = 'label'
IMAGE_COL  = 'image'
NAME_COL   = '姓名'

SUBFOLDERS = {
    '00_DCM':       '.dcm',
    '01_PNG':       '.png',
    '02_Mask':      '.png',
    '03_Binarized': '.png',
}

OUTPUT_DIR_ROI = os.path.join(OUTPUT_DIR, 'ROI_review_50cases')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR_ROI, exist_ok=True)


print('=' * 60)
print('[1] Loading clinical+radiomics data ...')
df_raw = pd.read_excel(RADIOMICS_FILE)
df = df_raw[df_raw[LABEL_COL].isin([0, 1])].copy()
df[LABEL_COL] = df[LABEL_COL].astype(int)
print(f'    Total: {len(df_raw)} -> binary: {len(df)}  '
      f'(dropped {len(df_raw)-len(df)} non-binary)')
for lbl, cnt in sorted(df[LABEL_COL].value_counts().items()):
    print(f'      label={lbl}: {cnt}')

for col in [LABEL_COL, IMAGE_COL, NAME_COL]:
    if col not in df.columns:
        raise KeyError(f'Column "{col}" not found!')


print()
print('[2] Stratified sampling ({0} binary -> {1}) ...'.format(len(df), N_SAMPLE))

stratum_sizes = df[LABEL_COL].value_counts()
stratum_n = {}
for lbl, cnt in stratum_sizes.items():
    stratum_n[lbl] = max(1, round(N_SAMPLE * cnt / len(df)))
diff = N_SAMPLE - sum(stratum_n.values())
if diff != 0:
    largest_lbl = max(stratum_n, key=stratum_n.get)
    stratum_n[largest_lbl] += diff

sampled_parts = []
for lbl, n in stratum_n.items():
    sampled_parts.append(df[df[LABEL_COL] == lbl].sample(n=n, random_state=RANDOM_SEED))
sampled_df = pd.concat(sampled_parts, ignore_index=True)

for lbl, cnt in sampled_df[LABEL_COL].value_counts().sort_index().items():
    print(f'      label={lbl}: {cnt}')


print()
print('[3] Scanning source directories ...')

pinyin_index = {}
chinese_index = {}

for group_dir in ['nonRMPP', 'RMPP']:
    for sub, ext in SUBFOLDERS.items():
        dirpath = os.path.join(SOURCE_ROOT, group_dir, sub)
        if not os.path.isdir(dirpath):
            continue
        for fname in os.listdir(dirpath):
            if not fname.lower().endswith(ext):
                continue
            full = os.path.join(dirpath, fname)
            base = os.path.splitext(fname)[0]

            if sub in ('01_PNG', '03_Binarized'):
                pinyin_index.setdefault(base, {})[sub] = full
            else:
                chinese_index.setdefault(base, {})[sub] = full

print(f'    Pinyin index  : {len(pinyin_index)} entries')
print(f'    Chinese index : {len(chinese_index)} entries')


print()
print('[4] Matching & copying source files ...')

results = []
found = 0
missing = 0
copied = 0

for _, row in sampled_df.iterrows():
    img_file = row[IMAGE_COL]
    pinyin = os.path.splitext(img_file)[0]
    chinese = row[NAME_COL]
    lbl = row[LABEL_COL]
    label_name = 'RMPP' if lbl == 1 else 'nonRMPP'


    case_folder = os.path.join(OUTPUT_DIR_ROI,
                               '{0}_{1}'.format(label_name, pinyin))
    os.makedirs(case_folder, exist_ok=True)

    rec = {'image': img_file, 'label': lbl, '姓名': chinese,
           'case_folder': case_folder}


    for sub in ['01_PNG', '03_Binarized']:
        src = pinyin_index.get(pinyin, {}).get(sub, '')
        col = 'source_{0}'.format(sub)
        if src and os.path.isfile(src):
            rec[col] = src
            found += 1

            dst = os.path.join(case_folder, os.path.basename(src))
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
            copied += 1
        else:
            rec[col] = 'NOT_FOUND'
            missing += 1


    for sub in ['00_DCM', '02_Mask']:
        src = chinese_index.get(chinese, {}).get(sub, '')
        col = 'source_{0}'.format(sub)
        if src and os.path.isfile(src):
            rec[col] = src
            found += 1
            dst = os.path.join(case_folder, os.path.basename(src))
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
            copied += 1
        else:
            rec[col] = 'NOT_FOUND'
            missing += 1

    results.append(rec)

result_df = pd.DataFrame(results)
total_slots = len(results) * 4
print(f'    Matched: {found}/{total_slots}  |  Copied: {copied} files')
print(f'    Missing: {missing}/{total_slots}')


print()
print('[5] Exporting ...')


radiomics_out = os.path.join(OUTPUT_DIR, 'sampled_50_cases_radiomics.xlsx')
sampled_df.to_excel(radiomics_out, index=False)
print('    {0}'.format(radiomics_out))


mapping_out = os.path.join(OUTPUT_DIR, 'sampled_50_cases_file_mapping.xlsx')
result_df.to_excel(mapping_out, index=False)
print('    {0}'.format(mapping_out))


missing_cases = result_df[
    ['source_{0}'.format(s) for s in SUBFOLDERS]
].eq('NOT_FOUND').any(axis=1)
n_missing = missing_cases.sum()

case_folders_created = len(os.listdir(OUTPUT_DIR_ROI))

report_lines = [
    '=' * 60,
    'ROI Reproducibility Sampling Report',
    '=' * 60,
    '',
    'Source data : ' + str(RADIOMICS_FILE),
    'Source files: ' + str(SOURCE_ROOT),
    'Binary cases: {0} (non-RMPP={1}, RMPP={2})'.format(
        len(df), stratum_sizes.get(0, 0), stratum_sizes.get(1, 0)),
    'Sampled     : {0} (non-RMPP={1}, RMPP={2}, seed={3})'.format(
        len(sampled_df), stratum_n.get(0, 0), stratum_n.get(1, 0), RANDOM_SEED),
    '',
    'File matching & copying:',
    '  Matched: {0}/{1}  |  Copied: {2} files'.format(found, total_slots, copied),
    '  Missing: {0}/{1}  ({2} cases with >=1 missing)'.format(missing, total_slots, n_missing),
    '',
    'Copied to: ' + str(OUTPUT_DIR_ROI),
    '  Folder structure:  {label}_{pinyin}/',
    '    e.g.  RMPP_anzining/',
    '          00_DCM/安紫宁.dcm     (if DCM matched)',
    '          01_PNG/anzining.png   (if PNG matched)',
    '          02_Mask/安紫宁.png    (if Mask matched)',
    '          03_Binarized/anzining.png',
    '  Case folders created: {0}'.format(case_folders_created),
    '',
]


if n_missing > 0:
    report_lines.append('Cases with missing source files:')
    for _, row in result_df[missing_cases].iterrows():
        missing_subs = [s for s in SUBFOLDERS
                        if row.get('source_{0}'.format(s)) == 'NOT_FOUND']
        report_lines.append(
            '  {0} ({1}): missing {2}'.format(
                row['image'], row['姓名'], missing_subs))
    report_lines.append('')

report_lines += [
    'Next steps:',
    '  1. Have a 2nd radiologist independently delineate ROIs',
    '     on the 50 CXR images in: ' + str(OUTPUT_DIR_ROI),
    '  2. Export new masks + run PyRadiomics feature extraction.',
    '  3. Calculate Dice coefficient between original & new masks.',
    '  4. Calculate ICC (two-way random, single rater) per feature.',
    '  5. Retain only features with ICC > 0.75 for modeling.',
    '  6. Report in manuscript: mean ICC, range, proportion >0.75.',
    '',
]

report_out = os.path.join(OUTPUT_DIR, 'sampling_report.txt')
with open(report_out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))
print('    {0}'.format(report_out))

print()
print('\n'.join(report_lines))
print('Done!')
