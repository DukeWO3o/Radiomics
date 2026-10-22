# RMPP 胸片影像组学项目脚本说明

最后更新：2026-10-21

本文件说明项目中各类脚本的用途、当前状态和主要输出位置。当前论文使用的主流程是：

```text
胸片预处理
  -> PyRadiomics 特征提取
  -> 分割可重复性与 ICC 筛选
  -> 折内预处理和特征选择
  -> 单次五折严格交叉验证
  -> 5 x 5 repeated stratified CV
  -> 年龄/ROI 相关性分析、DeLong 比较和补充材料
  -> 研究原型 App 或外部验证
```

项目中有较多早期版本和对照脚本。除非明确需要复现历史结果，否则应优先使用下文标注为“当前主流程”的脚本。

## 1. 当前论文主流程

| 阶段 | 当前脚本或数据 | 主要输出 |
|---|---|---|
| 图像预处理 | `01_PreProcess/A-H*.py` | `01_PNG`、`02_Mask`、`03_Binarized` |
| 特征提取 | `01_PreProcess/H.pyradiomics.py` | 影像组学特征 CSV |
| 分割重复性 | `02_OriginalData/H-2` 至 `H-5` | Dice、ICC 和 ICC 筛选后数据 |
| 建模输入 | `03_MergedData/02d_ICC_filtered_影像+临床版-v4-374-557-931.xlsx` | 931 例，557 RMPP、374 non-RMPP |
| 主建模 | `04_FeatureSelection_Step_1/I.refactored_pipeline_strict_cv.py` | `06_Modelling/06e_StrictCV_折内预处理/*` |
| 重复 CV | `04_FeatureSelection_Step_1/I-5.repeated_5x5_cv_strict.py` | `06e_StrictCV_折内预处理/09_5x5_Repeated_CV` |
| 年龄/ROI 相关性 | `04_FeatureSelection_Step_1/J.correlation_age_ROI.py` | `06e_StrictCV_折内预处理/08_Correlation_Age_ROI` |
| 外部验证 | `06_Modelling/external_validation.py` | `06_Modelling/external_validation_output` |
| 当前 App | `08_RMPP_Prediction_App_CV/app.py` | Streamlit 研究原型 |

当前固定数据划分为：

- 训练集：744 例
- 内部验证集：187 例
- 划分比例：80:20，患者层面分层随机划分
- 随机种子：42
- 内部验证集不参与插补、标准化、特征选择、调参或模型替换

## 2. 运行环境

原建模流程使用 Python 3.7、scikit-learn 1.0.2，并依赖：

```text
pandas
numpy
scipy
scikit-learn
statsmodels
matplotlib
seaborn
shap
joblib
pymrmr
xgboost
lightgbm
openpyxl
```

本机 `Pyradiomics` Conda 环境中安装了 `pymrmr`。如果直接使用系统 Python，可能因为缺少 `pymrmr` 或相关 DLL 而无法运行。

建议在 PyCharm 中选择对应解释器：

```text
E:\Softwares\Anaconda\envs\Pyradiomics\python.exe
```

## 3. 原始影像预处理：`01_PreProcess`

这些脚本按字母顺序执行，但并不是每个数据集都必须运行全部步骤。

| 脚本 | 作用 |
|---|---|
| `A.dat转txt.py` | 批量将 `.dat` 文件改名为 `.txt`。 |
| `B.利用2023姓名表重命名.py` | 根据 2023 年姓名表校正病例文件夹名称。 |
| `C.重命名DCM或JPG.py` | 统一 DICOM 或 JPG 文件命名。 |
| `D.利用2023RMPP筛选RMPP.py` | 根据 RMPP 名单筛选目标病例。 |
| `E.DCM图像标准化.py` | 读取 DICOM，调整窗宽窗位并输出 1024 x 1024 PNG。 |
| `E.DCM图像标准化-不反转.py` | `E` 的替代版本，不执行灰度反转；同一次分析中只能选择一个版本。 |
| `F.binarization.py` | 将 ROI mask 二值化并输出到 `03_Binarized`。 |
| `G.名字转换为拼音.py` | 中文姓名转拼音，解决 image/mask 同名对齐问题。 |
| `H.pyradiomics.py` | 使用 `01_PNG + 03_Binarized + radiomics_config.yaml` 批量提取 PyRadiomics 特征。 |
| `radiomics_config.yaml` | PyRadiomics 特征提取配置。 |

注意：

- `E.DCM图像标准化.py` 和 `E.DCM图像标准化-不反转.py` 是互斥方案。
- 第二位勾画者 R2 的特征提取通常需要复制 `H.pyradiomics.py` 后修改输入和输出路径。

## 4. 分割与特征可重复性：`02_OriginalData`

| 脚本 | 作用 |
|---|---|
| `H-2.sample_for_ICC.py` | 分层抽取 50 例用于第二位医师重复勾画，并生成文件映射表。 |
| `H-3.copy_sampled_files.py` | 将抽取病例的原始影像和 mask 复制给第二位勾画者。 |
| `H-4.calculate_dice.py` | 计算 R1/R2 分割的 Dice 系数并输出 `dice_results.xlsx`。 |
| `H-5.calculate_ICC.py` | 计算 R1/R2 特征 ICC，保留 ICC > 0.75 的特征，形成 ICC 筛选后的数据集。 |

论文中使用的结果：

- 50 例重复勾画
- 平均 Dice：0.946
- 102 个提取特征中保留 98 个
- 最终建模数据：`02d_ICC_filtered_影像+临床版-v4-374-557-931.xlsx`

## 5. 数据构建辅助脚本

| 脚本 | 状态 | 作用 |
|---|---|---|
| `00_DataSetSample/merge_external_dataset.py` | 外部验证数据准备 | 将连云港外部队列的影像组学表和临床表合并成与内部数据相同的列结构。 |
| `03_MergedData/I-1+2.数据集划分+特征初筛+标准化+显著性分析 Delta-副本.py` | 历史版本 | 早期数据划分、特征初筛、标准化和显著性分析。当前主流程已把相关步骤放入折内 Pipeline。 |
| `03_MergedData/I-3.特征选择并导出Delta-副本.py` | 历史版本 | 早期独立特征选择脚本。当前流程不再单独使用。 |

## 6. 建模脚本：`04_FeatureSelection_Step_1`

| 脚本 | 状态 | 作用 |
|---|---|---|
| `I.refactored_pipeline_strict_cv.py` | **当前主建模脚本** | 当前单次五折严格 CV。缺失率过滤、均值插补、常量特征过滤、Z-score、MWU/FDR、mRMR 和分类器均在各训练折内完成；输出九个模型、校准、DCA、SHAP、DeLong、最终特征和模型配置。 |
| `I-5.repeated_5x5_cv_strict.py` | **当前重复验证脚本** | 复用严格 CV 的折内处理逻辑，固定原最终参数，在原 744 例训练集内进行 5 x 5 repeated stratified CV。仅评估折划分稳定性，不重新调参，不使用内部验证集。 |
| `J.correlation_age_ROI.py` | **当前补充分析脚本** | 计算最终影像组学特征与年龄、ROI 面积的相关性，并进行 bootstrap 置信区间和 BH-FDR 校正；输出 Online Resource 8。 |
| `I.refactored_pipeline_modalities.py` | 历史对照 | 分别运行仅临床和仅影像组学模型，输出到旧 `06d...` 目录。当前三模型比较以严格 CV 的 `仅临床变量`、`仅影像组学` 和 `联合模型` 输出为准。 |
| `I.refactored_pipeline_larger_fonts.py` | 历史主流程 | 旧版 `06d` 联合模型流程，曾选择 LightGBM，并与当前 SVM 严格 CV 结果不同。不得与当前结果混用。 |
| `I.refactored_pipeline_app.py` | 旧版部署导出 | 从旧版建模流程导出 App 使用的 scaler、插补均值和分类器到 `06_Modelling/output_app` 与旧 App 的 `models`。该脚本使用旧的部署签名，不等同于当前严格折内流程。 |
| `I-2.refactored_pipeline_lasso.py` | 对照方案 | 使用 LASSO 替代 MWU/FDR + mRMR，输出到历史 `output_lasso` 目录，不作为当前结果。 |
| `I-1+2.数据集划分+特征初筛+标准化+显著性分析 Delta-副本.py` | 历史版本 | `03_MergedData` 中同名脚本的副本。 |
| `I.refactored_pipeline.py` | 历史版本 | 较早的完整建模流程。 |
| `调参用.py` | 历史版本 | 早期调参流程，不应作为最终分析入口。 |

### 6.1 当前严格 CV 的特征选择顺序

```text
缺失率 > 30% 的特征删除
-> 当前训练折内均值插补
-> 原始尺度常量特征过滤（阈值 0.0）
-> 当前训练折内 Z-score
-> Mann-Whitney U + Benjamini-Hochberg FDR
-> mRMR 保留 8 个影像组学特征
-> 拼接 age、CRP、WBC、LDH
-> 分类器拟合
```

注意：

- 不再使用 Z-score 后的 0.05 方差阈值。
- 最终模型使用完整训练集重新拟合得到的特征，而不是各折特征并集或多数投票。
- 最终参数见 `Online_Resource_5_strictCV_model_performance.xlsx` 的 `Best_Hyperparameters`。
- 历史完整搜索网格未保留，因此当前只报告最终参数。

### 6.2 当前建模输出

```text
06_Modelling/
  06e_StrictCV_折内预处理/
    仅临床变量/
    仅影像组学/
    联合模型/
    08_Correlation_Age_ROI/
    09_5x5_Repeated_CV/
```

主要文件：

| 文件 | 含义 |
|---|---|
| `04_Model_Performance.xlsx` | 九个模型的训练和验证性能。 |
| `04b_Selected_Features.xlsx` | 各模型的特征选择结果。 |
| `04c_Best_Hyperparameters.xlsx` | 本轮严格 CV 的最终参数。 |
| `09_Calibration_SVM.svg` | SVM 校准曲线。 |
| `10_DCA_SVM.svg` | SVM 决策曲线。 |
| `11_DeLong_Pvalues_FDRcorrected.xlsx` | 模型间 DeLong 检验及 FDR 校正结果。 |
| `model_*.pkl` | 可用于后续评估或部署的完整 Pipeline。 |
| `model_config.json` | 特征顺序、最终特征、模型名和选择信息。 |

`09_5x5_Repeated_CV` 中的主要文件：

| 文件 | 含义 |
|---|---|
| `01_5x5_Repeated_CV_Results.xlsx` | 汇总、225 条模型 x 折结果、重复均值、折内特征选择和最终参数。 |
| `02_Fold_AUC_Distribution.svg` | 25 个折的 AUC 分布箱线图。 |
| `03_Repeated_AUC_By_Run.svg` | 5 次重复 CV 的 AUC 折线图。 |
| `04_Summary.csv` | 九个模型的重复 CV 汇总。 |
| `05_Fold_AUCs.csv` | 逐折 AUC、Brier、accuracy、recall 和 specificity。 |
| `06_Repeated_CV_Config.json` | 数据划分、随机种子、参数和运行信息。 |

## 7. 历史两步建模目录：`05_FeatureSelection_Step_2`

`05b`、`05c`、`05d` 以及其中 Lasso/mRMR 的 v2、v2.1、v2.2、v2.3 脚本属于早期分步建模流程：

- `05b`：仅影像组学
- `05c`：影像组学 + 年龄
- `05d`：影像组学 + 临床变量
- `v2.3-校准曲线` 是早期带校准曲线的版本

这些脚本依赖早期导出的 `train_mrmr_selected.xlsx` 和 `val_mrmr_selected.xlsx`，已被 `I.refactored_pipeline_strict_cv.py` 取代。除复现历史结果外，不应使用。

## 8. 模型比较、外部验证和基线表

| 脚本 | 状态 | 作用 |
|---|---|---|
| `06_Modelling/06d_.../12_DeLong_Heatmap.py` | 历史图表脚本 | 读取旧 DeLong 结果并绘制聚类热图。当前 DeLong 热图由严格 CV 流程直接生成。 |
| `06_Modelling/external_validation.py` | 当前外部验证脚本 | 加载 App 模型包，在带标签的外部队列上计算 ROC、混淆矩阵、校准、DCA、bootstrap CI 和 DeLong。当前外部队列验证功能仍在完善。 |
| `06_Modelling/external_validation_original.py` | 历史版本 | 外部验证脚本的早期实现。 |
| `07_Baseline/scripts/baseline_v3_final.py` | **当前基线表脚本** | 读取 `.sav`，按指定分布假设和检验方法生成 Table 1。 |
| `07_Baseline/scripts/baseline_analysis.py` | 历史版本 | 早期基线表分析。 |
| `07_Baseline/scripts/baseline_v2.py` | 历史版本 | 基线表第二版。 |
| `07_Baseline/scripts/_gen.py` | 辅助脚本 | 早期自动生成或整理分析的辅助代码。 |

## 9. Web 研究原型：`08_RMPP_Prediction_App*`

| 脚本 | 状态 | 作用 |
|---|---|---|
| `08_RMPP_Prediction_App_CV/app.py` | **当前推荐 App** | 读取当前严格 CV 或冻结流程导出的 `model_*.pkl`、`model_config.json`，完成 DICOM 标准化、U-Net 分割、PyRadiomics 提取、模型预测和风险展示。 |
| `08_RMPP_Prediction_App_CV/train_unet.py` | 当前分段模型训练 | 训练用于自动肺部分割的 U-Net。 |
| `08_RMPP_Prediction_App_CV/unet_model.py` | 当前模型结构 | 定义 `SimpleUNet` 网络结构。 |
| `08_RMPP_Prediction_App_CV/deploy/run_app.py` | 当前部署入口 | 打包或部署环境中的 Streamlit 启动入口。 |
| `08_RMPP_Prediction_App/app.py` | 旧版 App | 使用旧的 `06_<name>_best_model.pkl` 和旧预处理配置，仅用于历史复现。 |
| `08_RMPP_Prediction_App/train_unet.py` | 旧版分段训练 | 旧 App 对应的 U-Net 训练脚本。 |
| `08_RMPP_Prediction_App/unet_model.py` | 旧版模型结构 | 旧 App 的 U-Net 网络定义。 |

当前 App 仍是研究原型，不能表述为已经完成外部验证或可以直接用于床旁决策。

## 10. 投稿文档辅助脚本

| 脚本 | 作用 |
|---|---|
| `_build_pediatric_radiology_submission.py` | 从原始 EJP 稿件和附件构建 Pediatric Radiology 投稿包，包括稿件、图片、补充材料目录和格式整理。 |
| `_build_pediatric_radiology_cover_letter.py` | 生成 Pediatric Radiology Cover Letter 的 DOCX 初稿。 |

这两个脚本只用于文档排版和投稿包构建，不参与统计分析。

## 11. 建议运行顺序

### 仅复现当前论文主要统计结果

```powershell
# 当前单次五折严格 CV
python .\04_FeatureSelection_Step_1\I.refactored_pipeline_strict_cv.py --modality combined

# 5 x 5 repeated stratified CV
python .\04_FeatureSelection_Step_1\I-5.repeated_5x5_cv_strict.py --modality combined

# 最终特征与年龄、ROI 面积相关性
python .\04_FeatureSelection_Step_1\J.correlation_age_ROI.py
```

需要确保运行的是安装有 `pymrmr` 的 Python 环境。

### 从原始影像开始重新建立队列

```text
01_PreProcess/A
-> B
-> C
-> D
-> E（选择其中一个版本）
-> F
-> G
-> H
-> H-2
-> H-3
-> H-4
-> H-5
-> I.refactored_pipeline_strict_cv.py
-> I-5.repeated_5x5_cv_strict.py
```

## 12. 版本使用原则

1. 当前论文的结果以 `06e_StrictCV_折内预处理`、`09_5x5_Repeated_CV` 和 `Pediatric Radiology/Supplementary` 中的文件为准。
2. 不要混用 `06d` 旧版 LightGBM 结果、旧图或旧 App 模型与当前 SVM 严格 CV 结果。
3. 不要将历史搜索网格未保留解释为未进行调参；当前只报告通过训练集 CV 选出的最终参数。
4. 任何模型参数、数据划分、随机种子、特征选择流程或输出结果发生变化时，应同步更新正文、表格、图片和 Supplementary。

## 13. 代码仓库与数据可用性

- 公开代码仓库：`https://github.com/DukeWO3o/RMPP`，仅包含脚本、图表与汇总结果
- 患者级数据（胸片、分割掩膜、影像组学特征表、临床表格、数据划分清单）不进入版本库，由 `.gitignore` 拦截
- 模型文件（`*.pkl`）不公开：KNN 的邻居集合与 SVM 的支持向量会内嵌训练样本
- 数据可向通讯作者申请，需通过伦理审批并签署数据使用协议
