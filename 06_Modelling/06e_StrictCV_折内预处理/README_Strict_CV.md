# Strict CV-internal preprocessing pipeline

Script: `04_FeatureSelection_Step_1/I.refactored_pipeline_strict_cv.py`

## Why this version exists

The earlier scripts fitted missingness filtering, mean imputation, variance
filtering, and Z-score standardization on the complete training set before
cross-validation. Feature selection was performed inside each CV fold, but the
preprocessing distribution still included the assessment folds. This version
moves every data-dependent preprocessing step into the corresponding inner
training fold.

## Exact computational flow

1. Load the 98 ICC-filtered radiomics features and the four clinical variables
   (`age`, `CRP`, `WBC`, `LDH`).
2. Split the cohort 80:20 into training and internal validation sets using
   stratified sampling with random seed 42.
3. Freeze the validation set. It is not used for imputation, variance filtering,
   scaling, univariate testing, mRMR, classifier fitting, focused-retuning
   candidate ranking, model replacement, or threshold selection.
4. Within each inner training fold:
   1. Remove features with missing rate > 30% (protected clinical variables are
      retained).
   2. Impute missing values using means from that training fold only.
   3. Remove constant features using a variance threshold of 0.0 in raw units.
   4. Fit Z-score standardization on that training fold only.
   5. Run Mann-Whitney U tests on the standardized training fold, followed by
      Benjamini-Hochberg FDR correction at alpha = 0.05.
   6. Run mRMR on the significant candidate features and retain up to eight
      radiomics features.
   7. Fit the classifier on the same training fold.
5. Select the initial classifier by mean training-set cross-validated AUC from
   `GridSearchCV`.
6. For combined models, run the built-in focused retuning for Decision Tree,
   Random Forest, XGBoost, and KNN. This evaluates 20, 24, 24, and 24 parameter
   configurations, respectively.
7. Refit each selected pipeline on the complete training set to obtain the final
   preprocessing parameters, final feature list, and classifier.
8. Lock all models and thresholds. Only then evaluate the frozen 20% internal
   validation set once.

## Variance threshold

The previous `0.05` variance threshold was removed. A threshold of `0.05` after
Z-score standardization is not meaningful because standardized feature
variances are near 1. The new pipeline uses a threshold of `0.0` on the raw
feature scale, which only removes constant features before scaling.

## Feature reporting

- `03_Fold_Preprocessing_and_Selection.xlsx`: number of features remaining
  after missingness and variance filtering, number of significant univariate
  features, and the exact features selected in each fold.
- `03_Selection_Frequency.xlsx`: number of folds (out of five) in which each
  feature was selected, plus whether it entered the final model.
- `04b_Selected_Features.xlsx`: final feature set used by each classifier after
  the full-training refit.
- `model_config.json` and `00_Pipeline_Flow.json`: exact pipeline order and
  final feature list.

The per-fold mRMR request is eight radiomics features. The final model uses the
feature set from the full-training refit, not the union of the five fold-specific
sets.

## ICC audit

The 50-case R1/R2 reproducibility sample overlaps with six cases in the current
187-case internal validation set. Recomputing ICC after excluding those six
cases still yielded the same 98-feature set as the full 50-case audit. This audit
is saved in `00_ICC_Audit.json`; it is a fixed measurement-reliability audit and
was not used to fit the validation outcomes.

## Model-selection rule

Initial classifier selection uses mean training-set CV AUC. Focused candidate
selection also uses training-only CV and excludes configurations with a
train-CV gap greater than 0.12. Among eligible configurations within 0.01 CV AUC
of the best, the selected score is:

`Training_CV_AUC - 0.5 * Train_CV_Gap`

The internal validation set is not used in this score or in the decision to
replace the current model. Its metrics are calculated only after all models are
fixed.

## Built-in focused retuning

The focused retuning is part of
`I.refactored_pipeline_strict_cv.py`; no separate tuning script is required.

- `13_Focused_Retuning_Trials.xlsx`: all training-CV trials.
- `13_Focused_Retuning_Trials_with_Validation.xlsx`: post-selection audit
  only. These validation metrics do not affect candidate ranking.
- `13_Focused_Retuning_Comparison.xlsx`: retained current versus retuned model.

Final selected parameters:

- Decision Tree: retuned.
- Random Forest: current parameters retained.
- XGBoost: current parameters retained.
- KNN: retuned to 21 neighbors, uniform weights, and Euclidean distance.

## Output folders

- `联合模型`: combined clinical + radiomics, regenerated with the best
  parameters from the sequential tuning run (`--use-tuned`)
- `联合模型/13_Focused_Retuning_*.xlsx`: focused four-model trials and final
  current-versus-retuned comparison.
- `联合模型_默认参数备份`: the strict-CV combined run before applying tuned
  parameters
- `仅临床变量`: clinical-only, strict-CV default parameters
- `仅影像组学`: radiomics-only, strict-CV default parameters

The final model comparison workbook is `模型间比较_调参后.xlsx`. It includes the
updated nine combined classifiers, the four-model focused-retuning trials, the
current-versus-retuned comparison, final feature frequencies, and the
within-combined DeLong matrix.
