# RMPP Strict-CV Prediction App

This folder contains the strict cross-validation version of the RMPP chest-radiograph classification app.

## Model bundle

The files in `models/` are copied from:

`06_Modelling/06e_StrictCV_折内预处理/联合模型`

Each `model_<classifier>.pkl` is a complete scikit-learn Pipeline containing:

1. The fitted `StrictPreprocessSelector`
2. The fitted classifier

Missing-value imputation, Z-score standardization, and radiomics feature selection are therefore performed inside each loaded model. The legacy files `04_Imputation_Means.pkl`, `04_StandardScaler.pkl`, and `04_ClinicalScaler.pkl` are not used by this app.

The model selected by mean training-set five-fold cross-validation AUC is SVM. The frozen internal validation results are stored in the training output directory and are not used to select the model in the app.

## Run

Use the project's Pyradiomics environment with the required DLL directories on `PATH`:

```powershell
$env:PATH = "E:\Softwares\Anaconda\envs\Pyradiomics;E:\Softwares\Anaconda\envs\Pyradiomics\Library\mingw-w64\bin;E:\Softwares\Anaconda\envs\Pyradiomics\Library\usr\bin;E:\Softwares\Anaconda\envs\Pyradiomics\Library\bin;E:\Softwares\Anaconda\envs\Pyradiomics\Scripts;E:\Softwares\Anaconda\envs\Pyradiomics\bin;" + $env:PATH
E:\Softwares\Anaconda\envs\Pyradiomics\python.exe -m streamlit run app.py
```

## Runtime inputs

The app accepts:

1. A chest-radiograph DICOM file
2. Age in years and months
3. CRP, WBC, and LDH

The output is a research-prototype RMPP classification probability. It is not intended for autonomous diagnosis or treatment decisions.
