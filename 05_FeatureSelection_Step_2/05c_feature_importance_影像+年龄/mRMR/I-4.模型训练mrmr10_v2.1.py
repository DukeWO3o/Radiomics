# ================== 模块 4：机器学习建模与验证 ==================
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier  # 决策树
from sklearn.naive_bayes import GaussianNB  # 朴素贝叶斯

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier  # LightGBM

from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve, f1_score,
    precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay,
    brier_score_loss
)
from sklearn.calibration import calibration_curve

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取已标准化的训练集和验证集
print("正在加载数据...")
train_data = pd.read_excel("F:/2023/Pyradiomics结果/第五次尝试/05c_feature_importance_影像+年龄/mRMR/train_mrmr_影像+年龄10特征.xlsx")
valid_data = pd.read_excel("F:/2023/Pyradiomics结果/第五次尝试/05c_feature_importance_影像+年龄/mRMR/val_mrmr_影像+年龄10特征.xlsx")

X_train = train_data.drop(columns=['label'])
y_train = train_data['label']

X_valid = valid_data.drop(columns=['label'])
y_valid = valid_data['label']

print(f"训练集大小: {X_train.shape}")
print(f"验证集大小: {X_valid.shape}")
print("训练集标签分布:")
print(y_train.value_counts())

# 计算正负样本比例用于LightGBM和XGBoost
scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])

# === 特征名称安全处理 ===
def sanitize_names(df):
    """清理列名中的特殊字符，避免树模型报错"""
    cols = df.columns
    new_cols = []
    for col in cols:
        new_col = col.replace('[', '').replace(']', '').replace('<', '').replace('>', '')
        new_col = new_col.replace(':', '_').replace(',', '_').replace('(', '_').replace(')', '_')
        # 避免中文，此处仅处理特殊符号
        new_cols.append(new_col)
    df.columns = new_cols
    return df

X_train = sanitize_names(X_train)
X_valid = sanitize_names(X_valid)
print("✅ 特征名称已安全处理。")

# 2. 定义模型和参数网格用于调参
models = {
    "LogisticRegression": {
        # 使用 saga 求解器支持 ElasticNet (L1+L2)，这对高维特征筛选非常有效
        "model": LogisticRegression(max_iter=3000, random_state=42, class_weight='balanced', solver='saga'),
        "params": {
            'C': [0.01, 0.05, 0.1, 0.5, 1, 5],  # 重点关注较小的 C (强正则化)
            'penalty': ['l1', 'l2', 'elasticnet'],
            'l1_ratio': [0.5]  # 仅当 penalty='elasticnet' 时使用
        }
    },
    "DecisionTree": {
        "model": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        "params": {
            'max_depth': [2, 3],  # 决策树极易过拟合，严格限制深度
            'min_samples_split': [10, 20],
            'min_samples_leaf': [5, 10],  # 强制叶子节点包含更多样本
            'max_features': [0.5, 0.7, 'sqrt'] # 限制分裂时使用的特征数
        }
    },
    "RandomForest": {
        "model": RandomForestClassifier(random_state=42, class_weight='balanced'),
        "params": {
            'n_estimators': [100, 300],
            'max_depth': [2, 3, 4],  # 降低深度
            'min_samples_split': [10, 20],
            'min_samples_leaf': [3, 5, 10], # 关键参数：防止由于个别噪声样本导致的过拟合
            'max_features': ['sqrt', 0.3],  # 减少特征数量，增加树之间的差异性
            'bootstrap': [True]
        }
    },
    "MLP": {
        # 增加 alpha (L2正则化) 防止权重过大
        "model": MLPClassifier(max_iter=2000, random_state=42, early_stopping=True, n_iter_no_change=20),
        "params": {
            'hidden_layer_sizes': [(30,), (50,), (30, 15)], # 影像组学数据通常不需要太深的网络
            'alpha': [0.01, 0.1, 1], # 强正则化
            'learning_rate_init': [0.001, 0.01],
            'activation': ['relu', 'tanh']
        }
    },
    "SVM": {
        "model": SVC(probability=True, random_state=42, class_weight='balanced', cache_size=500),
        "params": {
            'C': [13, 14, 15, 16, 17, 20],
            'gamma': ['scale', 0.001, 0.004, 0.005, 0.006, 0.01, 0.05, 0.1], # 手动指定 gamma 有时比 scale 更好
            'kernel': ['rbf']
        }
    },
    "XGBoost": {
        # 增加 subsample 和 colsample_bytree 进行随机采样
        "model": XGBClassifier(eval_metric="logloss", random_state=42, use_label_encoder=False,
                               scale_pos_weight=scale_pos_weight), # 保持类别权重
        "params": {
            'n_estimators': [200, 300], # 增加树的数量
            'max_depth': [2, 3], # 浅树
            'learning_rate': [0.01, 0.03], # 慢学习（需配合大 n_estimators）
            'gamma': [0.1], # 增加分裂所需的最小损失减少量
            'reg_alpha': [0.1, 1], # L1 正则化 (特征筛选)
            'reg_lambda': [5, 10], # L2 正则化
            'subsample': [0.7, 0.8], # 行采样：每次只用 70%-80% 的样本建树
            'colsample_bytree': [0.8] # 列采样：每次只用 60%-80% 的特征
        }
    },
    "LightGBM": {
        "model": LGBMClassifier(random_state=42, verbose=-1, scale_pos_weight=scale_pos_weight),
        "params": {
            'n_estimators': [200, 300],
            'learning_rate': [0.01, 0.05],
            'num_leaves': [7, 15], # 重要：max_depth=3 时 leaves 建议 < 8
            'max_depth': [3, 4],
            'min_child_samples': [10, 20], # 防止过拟合的核心参数
            'reg_alpha': [1],
            'reg_lambda': [5],
            'subsample': [0.8], # 启用 bagging
            'subsample_freq': [1],   # 必须设置频率，subsample 才会生效
            'colsample_bytree': [0.8]
        }
    },
    "GaussianNB": {
        "model": GaussianNB(),
        "params": {
            'var_smoothing': [1e-9, 1e-8, 1e-5, 1e-3] # 扩大平滑范围
        }
    },
    "KNN": {
        "model": KNeighborsClassifier(),
        "params": {
            'n_neighbors': [5, 9, 13, 15], # 增加 K 值，使边界更平滑
            'weights': ['uniform', 'distance'],
            'p': [1, 2] # 1=曼哈顿距离, 2=欧氏距离
        }
    }
}

# 3. 交叉验证设置
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 4. 训练模型 & 调参
best_models = {}
train_results = {}
valid_results = {}

plt.figure(figsize=(8, 6))

for name, config in models.items():
    print(f"\n=== 训练并调参模型：{name} ===")

    grid_search = GridSearchCV(
        config["model"],
        config["params"],
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print(f"最佳参数: {grid_search.best_params_}")

    best_models[name] = best_model

    # 保存模型
    joblib.dump(best_model, f"F:/2023/Pyradiomics结果/第五次尝试/06_model/06_{name}_best_model.pkl")

    # 训练集评估
    y_train_pred = best_model.predict(X_train)
    y_train_prob = best_model.predict_proba(X_train)[:, 1] if hasattr(best_model, "predict_proba") else None

    # 计算各种评估指标
    train_acc = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)

    if y_train_prob is not None:
        train_auc = roc_auc_score(y_train, y_train_prob)
        # ROC 曲线
        fpr, tpr, _ = roc_curve(y_train, y_train_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={train_auc:.3f})")
    else:
        train_auc = None

    # 交叉验证评估
    cv_acc = cross_val_score(best_model, X_train, y_train, cv=cv, scoring="accuracy").mean()
    cv_auc = cross_val_score(best_model, X_train, y_train, cv=cv,
                             scoring="roc_auc").mean() if y_train_prob is not None else None

    train_results[name] = {
        "Train_Accuracy": train_acc,
        "Train_F1": train_f1,
        "Train_Precision": train_precision,
        "Train_Recall": train_recall,
        "Train_AUC": train_auc,
        "CV_Accuracy": cv_acc,
        "CV_AUC": cv_auc
    }

    # 验证集评估
    y_valid_pred = best_model.predict(X_valid)
    y_valid_prob = best_model.predict_proba(X_valid)[:, 1] if hasattr(best_model, "predict_proba") else None

    valid_acc = accuracy_score(y_valid, y_valid_pred)
    valid_f1 = f1_score(y_valid, y_valid_pred)
    valid_precision = precision_score(y_valid, y_valid_pred)
    valid_recall = recall_score(y_valid, y_valid_pred)
    if y_valid_prob is not None:
        valid_auc = roc_auc_score(y_valid, y_valid_prob)
        # 计算 Brier Score
        valid_brier = brier_score_loss(y_valid, y_valid_prob)
    else:
        valid_auc = None
        valid_brier = None

    valid_results[name] = {
        "Valid_Accuracy": valid_acc,
        "Valid_F1": valid_f1,
        "Valid_Precision": valid_precision,
        "Valid_Recall": valid_recall,
        "Valid_AUC": valid_auc,
        "Valid_Brier_Score": valid_brier
    }

    # 训练集混淆矩阵
    cm_train = confusion_matrix(y_train, y_train_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_train)
    disp.plot(cmap="Blues")
    plt.title(f"Train Confusion Matrix - {name}")
    plt.savefig(f"F:/2023/Pyradiomics结果/第五次尝试/06_model/06_Train_ConfusionMatrix_{name}.svg", dpi=300,format="svg")
    plt.close()

    # 验证集混淆矩阵
    cm_valid = confusion_matrix(y_valid, y_valid_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_valid)
    disp.plot(cmap="Blues")
    plt.title(f"Valid Confusion Matrix - {name}")
    plt.savefig(f"F:/2023/Pyradiomics结果/第五次尝试/06_model/06_Valid_ConfusionMatrix_{name}.svg", dpi=300,format="svg")
    plt.close()

# 绘制训练集 ROC 曲线

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves of Models (Train Set)")
plt.legend()
plt.savefig("F:/2023/Pyradiomics结果/第五次尝试/06_model/06_Train_ROC_curves.svg", dpi=300,format="svg")
plt.close()

# 5. 模型性能对比
train_results_df = pd.DataFrame(train_results).T
valid_results_df = pd.DataFrame(valid_results).T

# 合并训练和验证结果
all_results_df = train_results_df.join(valid_results_df)
all_results_df.to_excel("F:/2023/Pyradiomics结果/第五次尝试/06_model/06_Model_Performance.xlsx")

# 绘制训练集性能对比图
train_metrics = ['Train_Accuracy', 'Train_F1', 'Train_Precision', 'Train_Recall', 'Train_AUC', 'CV_Accuracy', 'CV_AUC']
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
axes = axes.ravel()

for i, metric in enumerate(train_metrics):
    if i < len(axes) and metric in train_results_df.columns:
        sns.barplot(x=train_results_df.index, y=train_results_df[metric], ax=axes[i])
        axes[i].set_title(f"Train {metric} Comparison")
        axes[i].set_ylabel(metric)
        axes[i].tick_params(axis='x', rotation=45)

# 移除多余的子图
for i in range(len(train_metrics), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.savefig("F:/2023/Pyradiomics结果/第五次尝试/06_model/06_Train_Performance_Comparison.svg", dpi=300,format="svg")
plt.close()

# 绘制验证集性能对比图
valid_metrics = ['Valid_Accuracy', 'Valid_F1', 'Valid_Precision', 'Valid_Recall', 'Valid_AUC', 'Valid_Brier_Score']
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for i, metric in enumerate(valid_metrics):
    if i < len(axes) and metric in valid_results_df.columns:
        sns.barplot(x=valid_results_df.index, y=valid_results_df[metric], ax=axes[i])
        axes[i].set_title(f"Validation {metric} Comparison")
        axes[i].set_ylabel(metric)
        axes[i].tick_params(axis='x', rotation=45)

# 移除多余的子图
for i in range(len(valid_metrics), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.savefig("F:/2023/Pyradiomics结果/第五次尝试/06_model/06_Validation_Performance_Comparison.svg", dpi=300,format="svg")
plt.close()

# 6. SHAP 值解释 (以最佳模型为例)
# 确定最佳模型（基于验证集AUC）
if 'Valid_AUC' in valid_results_df.columns:
    best_model_name = valid_results_df['Valid_AUC'].idxmax()
else:
    best_model_name = valid_results_df.index[0]

print(f"\n=== 对最佳模型 {best_model_name} 进行 SHAP 解释 ===")

best_model = best_models[best_model_name]

# 准备解释数据
explain_data = X_train.iloc[:50]  # 使用部分训练数据加快计算
feature_names = X_train.columns

# 计算 SHAP 值
if hasattr(best_model, 'predict_proba'):
    try:
        # 为树模型创建解释器
        if best_model_name in ["RandomForest", "XGBoost", "LightGBM"]:  # 添加LightGBM
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(explain_data)

            # 明确使用正类的 SHAP 值
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values_class1 = shap_values[1]
            else:
                shap_values_class1 = shap_values

            # 摘要图
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, explain_data, feature_names=feature_names, show=False)
            plt.title(f"SHAP Summary Plot - {best_model_name}")
            plt.tight_layout()
            plt.savefig(f"F:/2023/Pyradiomics结果/第五次尝试/06_model/06_SHAP_Summary_{best_model_name}.svg", dpi=300,format="svg")
            plt.close()

            # 特征重要性图
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, explain_data, feature_names=feature_names, plot_type="bar", show=False)
            plt.title(f"SHAP Feature Importance - {best_model_name}")
            plt.tight_layout()
            plt.savefig(f"F:/2023/Pyradiomics结果/第五次尝试/06_model/06_SHAP_Importance_{best_model_name}.svg", dpi=300,format="svg")
            plt.close()

        # 为线性模型创建解释器
        elif best_model_name == "LogisticRegression":
            explainer = shap.LinearExplainer(best_model, explain_data)
            shap_values = explainer.shap_values(explain_data)

            # 摘要图
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, explain_data, feature_names=feature_names, show=False)
            plt.title(f"SHAP Summary Plot - {best_model_name}")
            plt.tight_layout()
            plt.savefig(f"F:/2023/Pyradiomics结果/第五次尝试/06_model/06_SHAP_Summary_{best_model_name}.svg", dpi=300,format="svg")
            plt.close()

        # 为其他模型创建解释器（使用KernelExplainer）
        else:
            # 使用KernelExplainer作为通用解释器
            explainer = shap.KernelExplainer(best_model.predict_proba, explain_data)
            shap_values = explainer.shap_values(explain_data)

            # 摘要图（二分类问题，取第1类的SHAP值）
            if len(shap_values) == 2:  # 二分类问题
                shap_values_class1 = shap_values[1]
            else:
                shap_values_class1 = shap_values

            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values_class1, explain_data, feature_names=feature_names, show=False)
            plt.title(f"SHAP Summary Plot - {best_model_name}")
            plt.tight_layout()
            plt.savefig(f"F:/2023/Pyradiomics结果/第五次尝试/06_model/06_SHAP_Summary_{best_model_name}.svg", dpi=300,format="svg")
            plt.close()

        print(f"✅ SHAP 分析完成并保存")
    except Exception as e:
        print(f"❌ SHAP 分析出错: {e}")

# 7. 输出最佳模型和参数
best_model_info = {
    "best_model": best_model_name,
    "parameters": best_model.get_params(),
    "train_performance": train_results_df.loc[best_model_name].to_dict(),
    "validation_performance": valid_results_df.loc[best_model_name].to_dict()
}

best_model_info_df = pd.DataFrame([best_model_info])
best_model_info_df.to_excel("F:/2023/Pyradiomics结果/第五次尝试/06_model/06_Best_Model_Info.xlsx", index=False)

# 8. 绘制训练集与验证集性能对比
comparison_metrics = ['Accuracy', 'F1', 'Precision', 'Recall', 'AUC']
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for i, metric in enumerate(comparison_metrics):
    if i < len(axes):
        train_metric = f"Train_{metric}"
        valid_metric = f"Valid_{metric}"

        if train_metric in all_results_df.columns and valid_metric in all_results_df.columns:
            # 创建对比数据
            comparison_data = pd.DataFrame({
                'Train': all_results_df[train_metric],
                'Validation': all_results_df[valid_metric]
            })

            comparison_data.plot(kind='bar', ax=axes[i])
            axes[i].set_title(f"Train vs Validation {metric} Comparison")
            axes[i].set_ylabel(metric)
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].legend()

# 移除多余的子图
for i in range(len(comparison_metrics), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.savefig("F:/2023/Pyradiomics结果/第五次尝试/06_model/06_Train_vs_Validation_Comparison.svg", dpi=300,format="svg")
plt.close()

# 绘制验证集 ROC 曲线
plt.figure(figsize=(8, 6))
# 遍历所有已训练的最佳模型
for name, best_model in best_models.items():
    if hasattr(best_model, "predict_proba"):
        try:
            y_valid_prob = best_model.predict_proba(X_valid)[:, 1]
            valid_auc = roc_auc_score(y_valid, y_valid_prob)
            fpr, tpr, _ = roc_curve(y_valid, y_valid_prob)

            plt.plot(fpr, tpr, label=f"{name} (Valid AUC={valid_auc:.3f})")
        except Exception:
            # 某些模型可能在验证集上predict_proba失败
            pass

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves of Models (Validation Set)")
plt.legend(loc="lower right")

# 保存验证集 ROC 曲线图
plt.savefig("F:/2023/Pyradiomics结果/第五次尝试/06_model/06_Valid_ROC_curves.svg", dpi=300,format="svg")
plt.close()
print("✅ 验证集 ROC 曲线图已保存。")

# ================== 步骤 9：绘制校准曲线 ==================
# 需要先安装 statsmodels: pip install statsmodels
from statsmodels.nonparametric.smoothers_lowess import lowess

print("\n=== 绘制校准曲线 ===")
plt.figure(figsize=(8, 8))

# 设置绘图风格
plt.style.use('default')
plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 绘制 Ideal 线 (对角虚线)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1, label="Ideal")

# 2. 获取最佳模型数据
best_model = best_models[best_model_name]

if hasattr(best_model, "predict_proba"):
    try:
        # 获取验证集预测概率 (Numpy array)
        y_valid_prob = best_model.predict_proba(X_valid)[:, 1]

        # --- 关键修复 1: 数据类型转换 ---
        # 将 Pandas Series 转换为 Numpy array，避免索引报错
        y_valid_np = y_valid.to_numpy()

        # --- 关键修复 2: Lowess 平滑 ---
        # 使用纯 Numpy 数组进行计算
        smooth_result = lowess(y_valid_np, y_valid_prob, frac=0.4, it=0)
        x_smooth = smooth_result[:, 0]
        y_smooth = smooth_result[:, 1]

        # 获取 Brier Score
        brier_score = valid_results_df.loc[best_model_name, 'Valid_Brier_Score']

        # 绘制平滑校准曲线
        plt.plot(x_smooth, y_smooth, color='black', linewidth=2, linestyle='-',
                 label=f"Apparent (Brier={brier_score:.3f})")

        # --- 关键修复 3: Rug Plot 语法更新 ---
        # 创建布尔掩码 (Mask)，确保是纯布尔数组
        mask_neg = (y_valid_np == 0)
        mask_pos = (y_valid_np == 1)

        # 底部竖线 (负样本): 使用 x= 参数替代 axis='x'
        # data 必须是一维数组
        sns.rugplot(x=y_valid_prob[mask_neg], height=0.03, color='black', alpha=0.4, linewidth=1)

        # 顶部竖线 (正样本): 手动绘制在顶部
        plt.scatter(y_valid_prob[mask_pos], [0.98] * np.sum(mask_pos),
                    marker='|', color='black', s=100, alpha=0.4, linewidth=1)

    except Exception as e:
        import traceback

        print(f"绘制平滑曲线时出错: {e}")
        traceback.print_exc()
        # 出错时的备选方案
        fop, mpv = calibration_curve(y_valid, y_valid_prob, n_bins=10)
        plt.plot(mpv, fop, "s-", color='black', label="Calibration (Fallback)")

else:
    print(f"模型 {best_model_name} 不支持 predict_proba")

# 3. 美化图表
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("Predicted Probability", fontsize=12)
plt.ylabel("Actual Probability", fontsize=12)
plt.title(f"Calibration Curve ({best_model_name})", fontsize=14)

# 去除顶部和右侧边框
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 图例设置
plt.legend(loc="lower right", frameon=False, fontsize=10)

# 保存图片
output_path = f"F:/2023/Pyradiomics结果/第五次尝试/06_model/06_Valid_Calibration_Smooth_{best_model_name}.svg"
plt.savefig(output_path, dpi=300, format="svg", bbox_inches='tight')
plt.close()
print(f"✅ 校准曲线已保存至: {output_path}")

# ================== 步骤 10：绘制决策曲线 (DCA) ==================
print(f"\n=== 正在绘制最佳模型 ({best_model_name}) 的决策曲线 (DCA) ===")


def calculate_net_benefit(y_true, y_prob, thresholds):
    net_benefits = []
    # 计算全治疗 (Treat All) 的净收益
    # Treat All 意味着所有样本都被预测为阳性 (TP + FP = Total)
    prevalence = np.sum(y_true) / len(y_true)

    treat_all_benefits = []

    for thresh in thresholds:
        # 模型净收益计算
        # 只有当概率 > 阈值时，模型预测为阳性
        y_pred = (y_prob >= thresh).astype(int)

        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        n = len(y_true)

        # Net Benefit 公式: (TP / N) - (FP / N) * (pt / (1 - pt))
        if thresh == 1.0:
            nb = 0
        else:
            weight = thresh / (1 - thresh)
            nb = (tp / n) - (fp / n) * weight
        net_benefits.append(nb)

        # Treat All 净收益计算 (针对当前阈值)
        # TP_all = 所有阳性样本, FP_all = 所有阴性样本
        if thresh == 1.0:
            nb_all = 0
        else:
            nb_all = prevalence - (1 - prevalence) * (thresh / (1 - thresh))
        treat_all_benefits.append(nb_all)

    return np.array(net_benefits), np.array(treat_all_benefits)


# 准备数据
if hasattr(best_model, "predict_proba"):
    thresholds = np.arange(0, 1.01, 0.01)  # 阈值从 0 到 1
    y_valid_prob = best_model.predict_proba(X_valid)[:, 1]

    # 计算净收益
    nb_model, nb_all = calculate_net_benefit(y_valid, y_valid_prob, thresholds)

    # 开始绘图
    plt.figure(figsize=(8, 8))

    # 1. 模型曲线
    plt.plot(thresholds, nb_model, color='red', linewidth=2, label=best_model_name)

    # 2. Treat All 曲线
    plt.plot(thresholds, nb_all, color='gray', linestyle='--', label='Treat All')

    # 3. Treat None 曲线 (y=0)
    plt.plot(thresholds, np.zeros_like(thresholds), color='black', linewidth=1, label='Treat None')

    # 优化图形显示范围 (通常只关注 y > -0.05 的部分)
    y_min = -0.05
    y_max = max(np.max(nb_model), np.max(nb_all)) + 0.05
    plt.ylim(y_min, y_max)
    plt.xlim(0, 1.0)

    plt.xlabel('Threshold Probability', fontsize=12)
    plt.ylabel('Net Benefit', fontsize=12)
    plt.title(f'Decision Curve Analysis - {best_model_name}', fontsize=14)
    plt.legend(loc="upper right")
    plt.grid(True, linestyle='--', alpha=0.3)

    # 保存 DCA 图
    dca_output_path = f"F:/2023/Pyradiomics结果/第五次尝试/06_model/06_Valid_DCA_{best_model_name}.svg"
    plt.savefig(dca_output_path, dpi=300, format="svg", bbox_inches='tight')
    plt.close()
    print(f"✅ 决策曲线 (DCA) 已保存至: {dca_output_path}")

print("\n✅ 所有模型训练、调参、评估、校准及DCA分析完成!")
print(f"最佳模型: {best_model_name}")
print("详细结果已保存至指定目录")