# -*- coding: utf-8 -*-
# =============================================================================
# Merge the external validation cohort (Lianyungang) into the SAME style as the
# internal merged dataset:
#   03_MergedData/02d_ICC_filtered_影像+临床版-v4-374-557-931.xlsx
#
# Output column layout (identical to the internal file):
#   label | image | diagnostics_* (internal naming) | <120 ICC-filtered
#   radiomics features in internal order> | age | CRP | WBC | LDH
#
# Join key between the radiomics table and the clinical summary:
#   pinyin(姓名) stored in the radiomics "image" filename, e.g.
#     anyixin.png            -> 安依芯  (exact pinyin match)
#     fanziang3yue099983.png -> 范子昂3月 (pinyin + suffix digits)
#     zhuyinuo0106507.png    -> 朱依诺 (2 visits -> matched by row order)
#
# Dependencies: openpyxl, pypinyin (same env as the modelling scripts).
# =============================================================================
import os
import re
import openpyxl
from pypinyin import lazy_pinyin, Style

# ------------------------------ config ------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

RADIOMICS_FILE = os.path.join(SCRIPT_DIR, 'Lianyungang_original', '连云港_radiomics_features.xlsx')
CLINICAL_FILE = os.path.join(SCRIPT_DIR, 'Lianyungang_original', '连云港临床汇总.xlsx')
OUTPUT_FILE = os.path.join(PROJECT_ROOT, '03_MergedData',
                           'EXTERNAL_连云港_影像+临床版-107-8-115.xlsx')

RADIOMICS_SHEET = 'Lianyungang_RMPP-radiomics_feat'
CLINICAL_SHEET = 'Sheet2'

INTERNAL_REFERENCE = os.path.join(
    PROJECT_ROOT, '03_MergedData', '02d_ICC_filtered_影像+临床版-v4-374-557-931.xlsx')

# clinical columns to carry over
CLINICAL_MAP = {
    'label': 'RMPP（0/1）',
    'age_years': '年龄',
    'age_months': '月龄',
    'CRP': 'CRP（mg/L）',
    'WBC': 'WBC(10^9/L)',
    'LDH': 'LDH',
}
HOSPITAL_ID_COL = '住院号'
NAME_COL = '姓名'

# ------------------------------ helpers ------------------------------


def read_sheet(path, sheet_name):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[sheet_name]
    rows = ws.iter_rows(values_only=True)
    headers = [str(c) if c is not None else '' for c in next(rows)]
    data = []
    for r in rows:
        if all(v is None for v in r):
            continue
        data.append(dict(zip(headers, r)))
    return headers, data


def to_float(v):
    if v is None:
        return None
    s = str(v).strip().replace(',', '')
    if s == '' or s.lower() in ('nan', 'none', 'null'):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def pinyin_of(name):
    return ''.join(lazy_pinyin(str(name), style=Style.NORMAL)).lower()


def split_image_key(image_base):
    """Return (pinyin_prefix, trailing_digits_or_None)."""
    m = re.match(r'^(.*?)(\d+)$', image_base)
    if m:
        return m.group(1).lower(), m.group(2)
    return image_base.lower(), None


def norm_id(v):
    if v is None:
        return None
    s = str(v).strip()
    if s == '':
        return None
    try:
        return str(int(s))
    except ValueError:
        return s


# ------------------------------ load data ------------------------------
print('Reading clinical summary ...')
_, clin_rows = read_sheet(CLINICAL_FILE, CLINICAL_SHEET)
print(f'  clinical rows: {len(clin_rows)}')

clin_by_py = {}
for c in clin_rows:
    py = pinyin_of(c.get(NAME_COL, ''))
    clin_by_py.setdefault(py, []).append(c)

print('Reading radiomics features ...')
_, rad_rows = read_sheet(RADIOMICS_FILE, RADIOMICS_SHEET)
print(f'  radiomics rows: {len(rad_rows)}')

# ------------------------------ join ------------------------------
print('\nJoining by pinyin(姓名) [+ 住院号 suffix] ...')
matched = []          # (rad_row, clin_row)
unmatched = []
for r in rad_rows:
    img = str(r.get('image', ''))
    if not img:
        unmatched.append(('no-image', r))
        continue
    base = os.path.splitext(os.path.basename(img))[0]
    prefix, id_suffix = split_image_key(base)
    cands = clin_by_py.get(prefix, [])

    hit = None
    if id_suffix is not None:
        for c in cands:
            if norm_id(c.get(HOSPITAL_ID_COL)) == norm_id(id_suffix):
                hit = c
                break
    if hit is None and len(cands) == 1:
        hit = cands[0]
    if hit is not None:
        matched.append((r, hit))
        continue

    # ambiguous (e.g. 朱依诺 has 2 visits): assign by row order of the two files
    if len(cands) > 1:
        pos_in_rad = len(matched) + len(unmatched)
        clin_ambig = sorted(clin_rows.index(c) for c in cands)
        # order the candidates by their position in the clinical file
        ordered = sorted(cands, key=lambda c: clin_rows.index(c))
        assigned = ordered[pos_in_rad % len(ordered)]
        matched.append((r, assigned))
        print(f'  NOTE: ambiguous name "{prefix}" -> assigned by row order '
              f'{r.get("image")} <-> 住院号 {assigned.get(HOSPITAL_ID_COL)}')
        continue

    unmatched.append((img, prefix))

print(f'  matched: {len(matched)} / {len(rad_rows)}')
if unmatched:
    print('  UNMATCHED:')
    for u in unmatched:
        print('   ', u)

# label agreement (radiomics label vs clinical RMPP)
mismatch = 0
missing_label = 0
for r, c in matched:
    rl = r.get('label')
    cl = to_float(c.get(CLINICAL_MAP['label']))
    if rl is None:
        missing_label += 1
        continue
    if cl is not None and int(rl) != int(cl):
        mismatch += 1
        print(f'  LABEL MISMATCH image={r.get("image")} rad={rl} clin={cl}')
print(f'  label mismatches: {mismatch}  missing rad labels: {missing_label}')

# ------------------------------ internal column template ------------------------------
print('\nBuilding merged table (internal column template) ...')
int_headers, _ = read_sheet(INTERNAL_REFERENCE, 'Sheet1')
out_cols = [c for c in int_headers]
print(f'  internal template columns: {len(out_cols)}')

rad_headers, _ = read_sheet(RADIOMICS_FILE, RADIOMICS_SHEET)
rad_col_set = set(rad_headers)

# map external diagnostics names -> internal naming (Image-/Mask- -> -original_)
diag_map = {}
for c in rad_headers:
    if c.startswith('diagnostics_Image-') and not c.startswith('diagnostics_Image-interpolated_'):
        diag_map[c] = 'diagnostics_Image-original_' + c[len('diagnostics_Image-'):]
    elif c.startswith('diagnostics_Mask-') and not c.startswith('diagnostics_Mask-interpolated_'):
        diag_map[c] = 'diagnostics_Mask-original_' + c[len('diagnostics_Mask-'):]
# reverse map: internal name -> external source column
diag_map_rev = {v: k for k, v in diag_map.items()}

# check template columns are all available
missing_cols = [c for c in out_cols
                if c not in ('label', 'image', 'age', 'CRP', 'WBC', 'LDH')
                and c not in diag_map.values()
                and c not in rad_col_set]
print(f'  template columns absent from external radiomics: {len(missing_cols)}')
if missing_cols:
    print('   ', missing_cols[:30])

# ------------------------------ write output ------------------------------
wb_out = openpyxl.Workbook()
ws_out = wb_out.active
ws_out.title = 'Sheet1'
ws_out.append(out_cols)

n_null_clin = {'age': 0, 'CRP': 0, 'WBC': 0, 'LDH': 0}
for r, c in matched:
    row = {}
    for col in out_cols:
        if col == 'label':
            row[col] = r.get('label')
            if row[col] is None:
                row[col] = int(to_float(c.get(CLINICAL_MAP['label'])))
        elif col == 'image':
            row[col] = r.get('image')
        elif col == 'age':
            y = to_float(c.get(CLINICAL_MAP['age_years']))
            m = to_float(c.get(CLINICAL_MAP['age_months']))
            row[col] = round(y + (m if m is not None else 0.0) / 12.0, 4) if y is not None else None
        elif col == 'CRP':
            row[col] = to_float(c.get(CLINICAL_MAP['CRP']))
        elif col == 'WBC':
            row[col] = to_float(c.get(CLINICAL_MAP['WBC']))
        elif col == 'LDH':
            row[col] = to_float(c.get(CLINICAL_MAP['LDH']))
        elif col.startswith('diagnostics'):
            src = diag_map_rev.get(col, col)
            row[col] = r.get(src)
        else:
            row[col] = r.get(col)
        if col in ('age', 'CRP', 'WBC', 'LDH') and row[col] is None:
            n_null_clin[col] += 1
    ws_out.append([row[col] for col in out_cols])

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
wb_out.save(OUTPUT_FILE)
print(f'  saved: {OUTPUT_FILE}')
print(f'  rows written: {len(matched)}  columns: {len(out_cols)}')
print(f'  null clinical values: {n_null_clin}')

print('\nDone.')
