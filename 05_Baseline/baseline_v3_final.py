import pandas as pd, pyreadstat, numpy as np, os
import sys
from pathlib import Path
from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency, fisher_exact, levene
import warnings
warnings.filterwarnings("ignore")

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_ROOT))
from pipeline_paths import BASELINE_OUTPUT_ROOT, CLINICAL_ROOT, DEFAULT_CLINICAL_SAV

sav_path = (
    DEFAULT_CLINICAL_SAV
    if DEFAULT_CLINICAL_SAV.exists()
    else sorted(CLINICAL_ROOT.glob("*.sav"))[0]
)

df, meta = pyreadstat.read_sav(sav_path)
g0 = df[df["label"] == 0]
g1 = df[df["label"] == 1]
n0, n1 = len(g0), len(g1)
print("nonRMPP:", n0, "RMPP:", n1)


variables = [
    (4, "Age", "normal"),
    (5, "Sex", "cat"),
    (12, "Weight", "normal"),
    (6, "Cough", "cat"),
    (7, "Fever", "cat"),
    (13, "PreadmissionFeverDuration", "normal"),
    (14, "LengthOfStay", "normal"),
    (8, "PleuralReaction", "cat"),
    (9, "PleuralEffusion", "cat"),
    (10, "PulmonaryConsolidation", "cat"),
    (11, "BronchiolitisObliterans", "cat"),
    (15, "CRP", "nonnormal"),
    (16, "WBC", "normal"),
    (17, "NEUT", "nonnormal"),
    (18, "LYMPH", "nonnormal"),
    (19, "HB", "normal"),
    (20, "PLT", "normal"),
    (21, "ALT", "nonnormal"),
    (22, "AST", "nonnormal"),
    (23, "LDH", "nonnormal"),
    (24, "CK", "nonnormal"),
    (25, "CKMB", "nonnormal"),
    (26, "PT", "normal"),
    (27, "APTT", "normal"),
    (28, "DDimer", "nonnormal"),
    (29, "IL-5", "nonnormal"),
    (30, "IFN-a", "nonnormal"),
    (31, "IL-2", "nonnormal"),
    (32, "IL-6", "nonnormal"),
    (33, "IL-1b", "nonnormal"),
    (34, "IL-10", "nonnormal"),
    (35, "IFN-g", "nonnormal"),
    (36, "IL-8", "nonnormal"),
    (37, "IL-17", "nonnormal"),
    (38, "IL-4", "nonnormal"),
    (39, "IL-12P70", "nonnormal"),
    (40, "TNF-a", "nonnormal"),
    (41, "Bronchoscopy", "cat"),
    (42, "ICU", "cat"),
    (43, "Oxygen", "cat"),
    (44, "SMPP", "cat"),
    (45, "PE", "cat"),
    (46, "NP", "cat"),
]

def fmt_p(p_val):
    if p_val < 0.001: return "<0.001", "***"
    elif p_val < 0.01: return "{:.4f}".format(p_val), "**"
    elif p_val < 0.05: return "{:.4f}".format(p_val), "*"
    else: return "{:.4f}".format(p_val), ""

results = []

for idx, ename, dist in variables:
    col = df.columns[idx]

    if dist == "normal":
        v0 = g0[col].dropna()
        v1 = g1[col].dropna()
        lev_stat, lev_p = levene(v0, v1)
        equal_var = lev_p >= 0.05
        if equal_var:
            stat_t, p_val = ttest_ind(v0, v1, equal_var=True)
            test_method = "t-test"
        else:
            stat_t, p_val = ttest_ind(v0, v1, equal_var=False)
            test_method = "Welch t-test"
        p_str, sig = fmt_p(p_val)
        results.append({
            "Variable": ename,
            "nonRMPP": "{:.2f} +/- {:.2f}".format(v0.mean(), v0.std()),
            "RMPP": "{:.2f} +/- {:.2f}".format(v1.mean(), v1.std()),
            "Test": test_method, "P": p_str, "Sig": sig,
            "Levene_P": "{:.4f}".format(lev_p),
            "Var": "unequal" if lev_p < 0.05 else "equal",
        })

    elif dist == "nonnormal":
        v0 = g0[col].dropna()
        v1 = g1[col].dropna()
        stat_t, p_val = mannwhitneyu(v0, v1, alternative="two-sided")
        p_str, sig = fmt_p(p_val)
        q0_25, q0_75 = np.percentile(v0, [25, 75])
        q1_25, q1_75 = np.percentile(v1, [25, 75])
        results.append({
            "Variable": ename,
            "nonRMPP": "{:.2f} ({:.2f}-{:.2f})".format(v0.median(), q0_25, q0_75),
            "RMPP": "{:.2f} ({:.2f}-{:.2f})".format(v1.median(), q1_25, q1_75),
            "Test": "Mann-Whitney", "P": p_str, "Sig": sig,
            "Levene_P": "-", "Var": "-",
        })

    elif dist == "cat":
        v0 = g0[col].dropna(); v1 = g1[col].dropna()
        v0 = v0[v0 < 9]; v1 = v1[v1 < 9]
        all_vals = sorted(set(int(x) for x in set(v0.unique()) | set(v1.unique())))
        sub_results = []
        for val in all_vals:
            c0 = (v0 == val).sum(); c1 = (v1 == val).sum()
            pct0 = c0/len(v0)*100 if len(v0)>0 else 0
            pct1 = c1/len(v1)*100 if len(v1)>0 else 0
            sub_results.append({
                "Variable": "  {}={}".format(ename, val),
                "nonRMPP": "{} ({:.1f}%)".format(c0, pct0),
                "RMPP": "{} ({:.1f}%)".format(c1, pct1),
                "Test": "", "P": "", "Sig": "", "Levene_P": "", "Var": "",
            })
        if len(all_vals) >= 2:
            ct = pd.crosstab(
                pd.concat([pd.Series(v0.values), pd.Series(v1.values)]),
                [0]*len(v0)+[1]*len(v1))
            _, p_chi, _, expected = chi2_contingency(ct)
            use_fisher = (expected < 5).any()
            if use_fisher and ct.shape==(2,2):
                try: _, p_val = fisher_exact(ct); test_method="Fisher"
                except: p_val=p_chi; test_method="Chi-square"
            else: p_val=p_chi; test_method="Chi-square"
            p_str, sig = fmt_p(p_val)
            sub_results[-1]["Test"]=test_method; sub_results[-1]["P"]=p_str; sub_results[-1]["Sig"]=sig
        results.extend(sub_results)


result_rows = []
for r in results:
    result_rows.append({
        "Variable": r["Variable"].strip(),
        "nonRMPP (n={})".format(n0): r["nonRMPP"],
        "RMPP (n={})".format(n1): r["RMPP"],
        "Test": r["Test"], "P_value": r["P"], "Significance": r["Sig"],
        "Levene_P": r["Levene_P"], "Variance": r["Var"],
    })
result_df = pd.DataFrame(result_rows)
out_path = BASELINE_OUTPUT_ROOT / "Table1_Baseline_Characteristics.xlsx"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
result_df.to_excel(out_path, index=False)
print("Saved:", out_path)
