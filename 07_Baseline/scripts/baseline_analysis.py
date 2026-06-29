import pandas as pd
import pyreadstat
import numpy as np
from scipy import stats
from scipy.stats import shapiro, ttest_ind, mannwhitneyu, chi2_contingency, fisher_exact
import warnings
warnings.filterwarnings('ignore')

path = r'E:\\OneDrive\\学习\\影像组学\\小论文附件\\数据分析 v2\\全临床基线931.sav'
df, meta = pyreadstat.read_sav(path)

g0 = df[df['label'] == 0]
g1 = df[df['label'] == 1]
n0, n1 = len(g0), len(g1)
print(f'nonRMPP: n={n0}, RMPP: n={n1}')

def norm_test(data):
    clean = data.dropna()
    if len(clean) < 3 or len(clean) > 5000:
        return None, None, None
    try:
        stat, p = shapiro(clean)
        return stat, p, p >= 0.05
    except:
        return None, None, None

variables = [
    (4, 'Age', 'c'), (5, 'Sex', 'cat'), (6, 'Weight', 'c'),
    (7, 'Cough', 'cat'), (8, 'Fever', 'cat'),
    (9, 'PreadmissionFeverDuration', 'c'), (10, 'LengthOfStay', 'c'),
    (11, 'PleuralReaction', 'cat'), (12, 'PleuralEffusion', 'cat'),
    (13, 'PulmonaryConsolidation', 'cat'),
    (15, 'CRP', 'c'), (16, 'WBC', 'c'), (17, 'NEUT', 'c'),
    (18, 'LYMPH', 'c'), (19, 'HB', 'c'), (20, 'PLT', 'c'),
    (21, 'ALT', 'c'), (22, 'AST', 'c'), (23, 'LDH', 'c'),
    (24, 'CK', 'c'), (25, 'CKMB', 'c'),
    (26, 'PT', 'c'), (27, 'APTT', 'c'), (28, 'DDimer', 'c'),
    (29, 'IL-5', 'c'), (30, 'IFN-a', 'c'), (31, 'IL-2', 'c'),
    (32, 'IL-6', 'c'), (33, 'IL-1b', 'c'), (34, 'IL-10', 'c'),
    (35, 'IFN-g', 'c'), (36, 'IL-8', 'c'), (37, 'IL-17', 'c'),
    (38, 'IL-4', 'c'), (39, 'IL-12P70', 'c'), (40, 'TNF-a', 'c'),
    (41, 'Bronchoscopy', 'cat'), (42, 'ICU', 'cat'), (43, 'Oxygen', 'cat'),
    (44, 'SMPP', 'cat'), (45, 'PE', 'cat'), (46, 'NP', 'cat'),
]

results = []

for idx, ename, vtype in variables:
    col = df.columns[idx]
    
    if vtype == 'c':
        v0 = g0[col].dropna()
        v1 = g1[col].dropna()
        n0v, n1v = len(v0), len(v1)
        
        _, p0, norm0 = norm_test(v0)
        _, p1, norm1 = norm_test(v1)
        
        if norm0 and norm1 and norm0 is not None:
            stat_t, p_val = ttest_ind(v0, v1, equal_var=False)
            test_method = 't-test'
            fmt0 = '{:.2f} +/- {:.2f}'.format(v0.mean(), v0.std())
            fmt1 = '{:.2f} +/- {:.2f}'.format(v1.mean(), v1.std())
            dist = 'normal'
        else:
            stat_t, p_val = mannwhitneyu(v0, v1, alternative='two-sided')
            test_method = 'Mann-Whitney'
            q0_25, q0_75 = np.percentile(v0, [25, 75])
            q1_25, q1_75 = np.percentile(v1, [25, 75])
            fmt0 = '{:.2f} ({:.2f}-{:.2f})'.format(v0.median(), q0_25, q0_75)
            fmt1 = '{:.2f} ({:.2f}-{:.2f})'.format(v1.median(), q1_25, q1_75)
            dist = 'non-normal'
        
        p_str = '{:.4f}'.format(p_val) if p_val >= 0.001 else '<0.001'
        if p_val < 0.001:
            sig = '***'
        elif p_val < 0.01:
            sig = '**'
        elif p_val < 0.05:
            sig = '*'
        else:
            sig = ''
        
        results.append({
            'Variable': ename,
            'nonRMPP': fmt0,
            'RMPP': fmt1,
            'n_nonRMPP': n0v,
            'n_RMPP': n1v,
            'Test': test_method,
            'P': p_str,
            'Sig': sig,
            'Distribution': dist,
            'p_norm0': '{:.4f}'.format(p0) if p0 else 'N/A',
            'p_norm1': '{:.4f}'.format(p1) if p1 else 'N/A',
        })
    
    elif vtype == 'cat':
        v0 = g0[col].dropna()
        v1 = g1[col].dropna()
        v0 = v0[v0 < 9]
        v1 = v1[v1 < 9]
        
        if len(v0) == 0 and len(v1) == 0:
            continue
        
        all_vals = sorted(set(int(x) for x in set(v0.unique()) | set(v1.unique())))
        
        sub_results = []
        for val in all_vals:
            c0 = (v0 == val).sum()
            c1 = (v1 == val).sum()
            pct0 = c0 / len(v0) * 100 if len(v0) > 0 else 0
            pct1 = c1 / len(v1) * 100 if len(v1) > 0 else 0
            sub_results.append({
                'Variable': '  {}= {}'.format(ename, val),
                'nonRMPP': '{} ({:.1f}%)'.format(c0, pct0),
                'RMPP': '{} ({:.1f}%)'.format(c1, pct1),
                'n_nonRMPP': len(v0),
                'n_RMPP': len(v1),
                'Test': '',
                'P': '',
                'Sig': '',
                'Distribution': '',
                'p_norm0': '',
                'p_norm1': '',
            })
        
        if len(all_vals) >= 2:
            ct = pd.crosstab(
                pd.concat([pd.Series(v0.values), pd.Series(v1.values)]),
                [0]*len(v0) + [1]*len(v1)
            )
            _, p_chi, _, expected = chi2_contingency(ct)
            use_fisher = (expected < 5).any()
            
            if use_fisher and ct.shape == (2, 2):
                try:
                    _, p_val = fisher_exact(ct)
                    test_method = 'Fisher'
                except:
                    p_val = p_chi
                    test_method = 'Chi-square'
            else:
                p_val = p_chi
                test_method = 'Chi-square'
            
            p_str = '{:.4f}'.format(p_val) if p_val >= 0.001 else '<0.001'
            if p_val < 0.001:
                sig = '***'
            elif p_val < 0.01:
                sig = '**'
            elif p_val < 0.05:
                sig = '*'
            else:
                sig = ''
            sub_results[-1]['Test'] = test_method
            sub_results[-1]['P'] = p_str
            sub_results[-1]['Sig'] = sig
        
        results.extend(sub_results)

print()
print('=' * 110)
print('TABLE 1. Baseline Characteristics of Study Population')
print('=' * 110)
header = '{:<30} {:<28} {:<28} {:<15} {:<10} {:<6}'.format(
    'Variable', 'nonRMPP (n={})'.format(n0), 'RMPP (n={})'.format(n1),
    'Test', 'P', 'Sig')
print(header)
print('-' * 110)

for r in results:
    line = '{:<30} {:<28} {:<28} {:<15} {:<10} {:<6}'.format(
        r['Variable'], r['nonRMPP'], r['RMPP'],
        r['Test'], r['P'], r['Sig'])
    print(line)

print('-' * 110)
print('Continuous: mean +/- SD (normal) or median (IQR) (non-normal); Categorical: n (%)')
print('* p<0.05, ** p<0.01, *** p<0.001')

# Save to Excel
result_rows = []
for r in results:
    result_rows.append({
        'Variable': r['Variable'].strip(),
        'nonRMPP (n={})'.format(n0): r['nonRMPP'],
        'RMPP (n={})'.format(n1): r['RMPP'],
        'Test': r['Test'],
        'P_value': r['P'],
        'Significance': r['Sig'],
    })
result_df = pd.DataFrame(result_rows)
out_path = r'E:\\Softwares\\PycharmProject\\Radiomics\\Table1_Baseline_Characteristics.xlsx'
result_df.to_excel(out_path, index=False)
print()
print('Table 1 saved to: Table1_Baseline_Characteristics.xlsx')
