# ================== 模块 4：机器学习建模与验证 ==================
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import os
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

# === 路径配置（基于脚本自身位置） ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "06_Modelling", "06_model")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 读取已标准化的训练集和验证集
print("正在加载数据...")
train_data = pd.read_excel(os.path.join(SCRIPT_DIR, "train_mrmr_selected.xlsx"))
valid_data = pd.read_excel(os.path.join(SCRIPT_DIR, "val_mrmr_selected.xlsx"))

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
            'reg_alpha': [0.1, 1],
            'reg_lambda': [0.1, 1],
            'subsample': [0.7, 0.8], # 启用 bagging
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

plt.figure(figsize=(10, 8))

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
    joblib.dump(best_model, os.path.join(OUTPUT_DIR, f"06_{name}_best_model.pkl"))

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
    plt.savefig(os.path.join(OUTPUT_DIR, f"06_Train_ConfusionMatrix_{name}.svg"), dpi=300,format="svg")
    plt.close()

    # 验证集混淆矩阵
    cm_valid = confusion_matrix(y_valid, y_valid_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_valid)
    disp.plot(cmap="Blues")
    plt.title(f"Valid Confusion Matrix - {name}")
    plt.savefig(os.path.join(OUTPUT_DIR, f"06_Valid_ConfusionMatrix_{name}.svg"), dpi=300,format="svg")
    plt.close()

# 绘制训练集 ROC 曲线
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves of Models (Train Set)")
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "06_Train_ROC_curves.svg"), dpi=300,format="svg")
plt.close()

# ================== 步骤 5：模型性能对比与 DeLong 检验 ==================
import scipy.stats
from scipy import stats

# --- 1. 定义 DeLong 检验函数 (Fast DeLong Algorithm) ---
def compute_midrank(x):
    """计算中位数秩，用于 DeLong 算法"""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float64)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=np.float64)
    T2[J] = T
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count):
    """快速 DeLong 协方差计算"""
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float64)
    ty = np.empty([k, n], dtype=np.float64)
    tz = np.empty([k, m + n], dtype=np.float64)

    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def calc_pvalue(aucs, sigma):
    """计算 P 值"""
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return 2 * scipy.stats.norm.sf(np.abs(z))


def delong_roc_test(y_true, y_pred1, y_pred2):
    """
    执行 DeLong 检验
    :param y_true: 真实标签
    :param y_pred1: 模型1的预测概率
    :param y_pred2: 模型2的预测概率
    :return: p-value
    """
    order = (-y_true).argsort()
    label_1_count = int(y_true.sum())
    predictions_sorted_transposed = np.vstack((y_pred1, y_pred2))[:, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    return calc_pvalue(aucs, delongcov)


# --- 2. 确定最佳模型并计算 P-value ---
print("\n=== 正在进行 DeLong 统计检验 ===")

# 找出验证集 AUC 最高的模型作为 "Reference" (最佳模型)
valid_aucs = {name: res['Valid_AUC'] for name, res in valid_results.items() if res['Valid_AUC'] is not None}
best_model_name = max(valid_aucs, key=valid_aucs.get)
print(f"最佳模型 (Reference) 是: {best_model_name} (AUC={valid_aucs[best_model_name]:.4f})")

# 获取最佳模型的预测概率
best_model_prob = best_models[best_model_name].predict_proba(X_valid)[:, 1]
y_valid_np = y_valid.to_numpy() if hasattr(y_valid, 'to_numpy') else y_valid

p_values = {}
for name, model in best_models.items():
    if name == best_model_name:
        p_values[name] = np.nan  # 最佳模型自己对自己不需要计算
        continue

    if hasattr(model, "predict_proba"):
        try:
            curr_prob = model.predict_proba(X_valid)[:, 1]
            # 计算 DeLong P-value
            p_val = delong_roc_test(y_valid_np, best_model_prob, curr_prob)
            p_values[name] = p_val[0][0]  # 提取标量值
        except Exception as e:
            print(f"无法计算 {name} 的 P-value: {e}")
            p_values[name] = np.nan
    else:
        p_values[name] = np.nan

# --- 3. 构建新样式表格 ---
print("\n=== 构建最终结果表格 ===")

final_table_rows = []

# 定义想要展示的列顺序
metrics_order = ['Accuracy', 'F1', 'Precision', 'Recall', 'AUC']

for name in models.keys():  # 保持原始定义的顺序
    if name not in train_results: continue

    # 1. 训练集行
    train_row = {
        'Model': name,
        'Dataset': 'Train',
        'P-value (vs Best)': ''  # 训练集通常不看这个，或者留空
    }
    for m in metrics_order:
        train_row[m] = train_results[name].get(f'Train_{m}', np.nan)
    final_table_rows.append(train_row)

    # 2. 验证集行
    valid_row = {
        'Model': '',  # 留空，实现合并单元格的视觉效果
        'Dataset': 'Validation',
        'P-value (vs Best)': p_values.get(name, np.nan)
    }
    # 特殊处理：如果是最佳模型，P值标为 "-" 或 "Reference"
    if name == best_model_name:
        valid_row['P-value (vs Best)'] = 'Reference'

    for m in metrics_order:
        valid_row[m] = valid_results[name].get(f'Valid_{m}', np.nan)
    final_table_rows.append(valid_row)

# 转换为 DataFrame
df_final = pd.DataFrame(final_table_rows)

# 调整列顺序
cols = ['Model', 'Dataset'] + metrics_order + ['P-value (vs Best)']
df_final = df_final[cols]


# 格式化数字 (保留3位小数，P值若非常小则用科学计数法)
def format_val(x):
    if isinstance(x, float):
        return f"{x:.4f}"
    return x


def format_p(x):
    if x == 'Reference': return x
    if isinstance(x, float):
        if np.isnan(x): return ""
        if x < 0.001: return "<0.001"
        return f"{x:.4f}"
    return x


# 应用格式化 (仅用于显示/保存，不影响计算)
df_display = df_final.copy()
for col in metrics_order:
    df_display[col] = df_display[col].apply(format_val)
df_display['P-value (vs Best)'] = df_display['P-value (vs Best)'].apply(format_p)

# 保存
output_file = os.path.join(OUTPUT_DIR, "06_Model_Performance_NewStyle.xlsx")
df_display.to_excel(output_file, index=False)
print(f"✅ 新样式表格已保存至: {output_file}")

# --- 4. 绘制所有模型 ROC 曲线 (含 P 值) ---
plt.figure(figsize=(10, 8))
plt.plot([0, 1], [0, 1], "k:", label="Chance")

# 绘制最佳模型
fpr, tpr, _ = roc_curve(y_valid, best_model_prob)
plt.plot(fpr, tpr, label=f"{best_model_name} (AUC={valid_aucs[best_model_name]:.3f})", linewidth=2.5, color='red',
         zorder=10)

# 绘制其他模型
for name, model in best_models.items():
    if name == best_model_name: continue
    if hasattr(model, "predict_proba"):
        try:
            prob = model.predict_proba(X_valid)[:, 1]
            auc = roc_auc_score(y_valid, prob)
            p_val = p_values.get(name, np.nan)
            p_text = f"p={p_val:.3f}" if p_val >= 0.001 else "p<0.001"

            fpr, tpr, _ = roc_curve(y_valid, prob)
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f}, {p_text})", linewidth=1, alpha=0.7)
        except:
            pass

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curves comparison (Reference: {best_model_name})")
plt.legend(loc="lower right", fontsize=9)
plt.savefig(os.path.join(OUTPUT_DIR, "06_Valid_ROC_with_Pvalues.svg"), dpi=300, format="svg")
plt.close()
print("✅ 带 P-value 的 ROC 曲线已保存。")

# ================== 步骤 9：绘制 R 语言 rms 风格校准曲线 ==================
from sklearn.calibration import CalibratedClassifierCV
from statsmodels.nonparametric.smoothers_lowess import lowess
import numpy as np
import matplotlib.pyplot as plt

print("\n=== 开始绘制 R 语言 rms 风格校准曲线 ===")

for name, model in best_models.items():
    print(f"正在处理模型: {name}...")

    try:
        # 获取原始未校准的预测概率 (Apparent)
        y_valid_prob_apparent = model.predict_proba(X_valid)[:, 1]

        # 1. 二次校准 (获取 Bias-corrected 概率)
        calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv=5)
        calibrated_model.fit(X_train, y_train)
        y_valid_prob_calib = calibrated_model.predict_proba(X_valid)[:, 1]

        # 转换为 Numpy 数组防止报错
        y_valid_np = y_valid.to_numpy() if hasattr(y_valid, 'to_numpy') else y_valid

        # 2. 使用 Lowess 进行非参数平滑拟合
        # frac=0.6: 控制平滑程度，越大越平滑；it=0: 加快计算速度
        smooth_apparent = lowess(y_valid_np, y_valid_prob_apparent, frac=0.6, it=0)
        smooth_calib = lowess(y_valid_np, y_valid_prob_calib, frac=0.6, it=0)

        # 提取平滑后的 x 和 y 坐标
        x_app, y_app = smooth_apparent[:, 0], smooth_apparent[:, 1]
        x_cal, y_cal = smooth_calib[:, 0], smooth_calib[:, 1]

        # 3. 开始仿 R 语言 rms 绘图
        fig, ax = plt.subplots(figsize=(6, 6))

        # 强制设置极简背景
        plt.style.use('default')
        plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei']  # 英文优先 Arial，符合学术规范
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')

        # [1] 绘制 Ideal 线 (完美的对角线，点划线)
        ax.plot([0, 1], [0, 1], linestyle='-.', color='black', linewidth=1.2)

        # [2] 绘制 Apparent 线 (点线)
        ax.plot(x_app, y_app, linestyle=':', color='black', linewidth=1.5)

        # [3] 绘制 Bias-corrected 线 (实线)
        ax.plot(x_cal, y_cal, linestyle='-', color='black', linewidth=1.5)

        # 4. 坐标轴与刻度美化
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])

        # 设置坐标轴刻度朝内 (R语言默认风格)
        ax.tick_params(axis='both', direction='in', length=5, width=1, colors='black',
                       top=False, right=False, labelsize=11)

        # 设置边框 (只保留左边和下边，或者全保留但细一点)
        for spine in ['top', 'right', 'bottom', 'left']:
            ax.spines[spine].set_linewidth(1)
            ax.spines[spine].set_color('black')

        # 标签设置
        ax.set_xlabel("Predicted Probability", fontsize=12, labelpad=8)
        ax.set_ylabel("Actual Probability", fontsize=12, labelpad=8)

        # 如果需要图标题，可以取消下方注释
        # ax.set_title(f"Calibration - {name}", fontsize=14, pad=15)

        # 5. 绘制 Rug Plot (分布密度线)
        # 在顶部边缘画小竖线
        ax.scatter(y_valid_prob_apparent, np.full_like(y_valid_prob_apparent, 1.0),
                   marker='|', color='black', s=40, alpha=0.3, linewidth=0.8)

        # 6. 自定义图例 (完全复刻位置和样式)
        # 手动创建线条对象用于图例
        from matplotlib.lines import Line2D

        custom_lines = [
            Line2D([0], [0], color='black', linestyle=':', lw=1.5),
            Line2D([0], [0], color='black', linestyle='-', lw=1.5),
            Line2D([0], [0], color='black', linestyle='-.', lw=1.2)
        ]

        # 将图例放在右下角偏上的位置，取消外边框
        ax.legend(custom_lines, ['Apparent', 'Bias-corrected', 'Ideal'],
                  loc=(0.65, 0.15), frameon=False, fontsize=11, handlelength=3)

        # 保存图表
        output_path = os.path.join(OUTPUT_DIR, f"06_Valid_Calibration_RMS_{name}.svg")
        plt.savefig(output_path, dpi=300, format="svg", bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)  # 显式关闭 figure
        print(f"  ✅ 已保存: 06_Valid_Calibration_CI_{name}.svg")

    except Exception as e:
        print(f"  ❌ 处理模型 {name} 时出错: {e}")
        plt.close('all')

print("\n✅ 所有模型的 rms 风格校准曲线绘制完成！")
# ================== 步骤 10：绘制决策曲线 (DCA) - 使用原始未校准概率 ==================
import numpy as np
import matplotlib.pyplot as plt

print(f"\n=== 正在绘制最佳模型 ({best_model_name}) 的决策曲线 (DCA) ===")


def calculate_net_benefit(y_true, y_prob, thresholds):
    """计算模型和 Treat All 的净收益 (Net Benefit)"""
    net_benefits = []
    treat_all_benefits = []
    prevalence = np.sum(y_true) / len(y_true)

    for thresh in thresholds:
        # 模型净收益计算
        y_pred = (y_prob >= thresh).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        n = len(y_true)

        if thresh == 1.0:
            nb = 0
            nb_all = 0
        else:
            weight = thresh / (1 - thresh)
            nb = (tp / n) - (fp / n) * weight
            nb_all = prevalence - (1 - prevalence) * weight

        net_benefits.append(nb)
        treat_all_benefits.append(nb_all)

    return np.array(net_benefits), np.array(treat_all_benefits)


# 1. 获取最佳模型的【原始未校准】预测概率
best_model = best_models[best_model_name]

if hasattr(best_model, "predict_proba"):
    # 直接从 best_model 获取概率，不经过 CalibratedClassifierCV
    y_valid_prob_raw = best_model.predict_proba(X_valid)[:, 1]
    y_valid_np = y_valid.to_numpy() if hasattr(y_valid, 'to_numpy') else y_valid

    # 设定阈值范围 (0.01 到 0.99)
    thresholds = np.arange(0.01, 1.00, 0.01)

    # 计算净收益
    nb_model, nb_all = calculate_net_benefit(y_valid_np, y_valid_prob_raw, thresholds)

    # 2. 开始绘图
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.style.use('default')
    plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei']  # 优先英文无衬线字体
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # 绘制模型收益线 (红色粗实线)
    ax.plot(thresholds, nb_model, color='firebrick', linewidth=2.5, label=f'{best_model_name} Model')

    # 绘制 Treat All 收益线 (灰色虚线)
    ax.plot(thresholds, nb_all, color='gray', linestyle='--', linewidth=1.5, label='Treat All')

    # 绘制 Treat None 收益线 (水平黑线 y=0)
    ax.plot(thresholds, np.zeros_like(thresholds), color='black', linewidth=1.5, label='Treat None')

    # 3. 优化图形显示范围 (重要)
    # 医学 DCA 通常聚焦于 0~0.8 的阈值区间，因为 >0.8 的概率极少作为临床干预阈值
    ax.set_xlim([0.0, 0.8])

    # Y轴下限设为稍微低于 0，上限设为 Treat All 的最高点加上一点余量
    y_min = -0.05
    y_max = max(np.max(nb_model), np.max(nb_all)) + 0.05
    ax.set_ylim([y_min, y_max])

    # 4. 美化图表细节
    ax.set_xlabel('Threshold Probability', fontsize=12, labelpad=8)
    ax.set_ylabel('Net Benefit', fontsize=12, labelpad=8)

    # 刻度朝内，经典学术风格
    ax.tick_params(axis='both', direction='in', length=5, width=1, colors='black', top=False, right=False)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)  # 去除上、右边框
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_linewidth(1.2)
        ax.spines[spine].set_color('black')

    # 添加网格线以增加可读性
    ax.grid(True, linestyle=':', alpha=0.6)

    # 图例设置在右上角
    ax.legend(loc="upper right", frameon=False, fontsize=11)

    # 5. 保存 DCA 图
    dca_output_path = os.path.join(OUTPUT_DIR, f"06_Valid_DCA_Raw_{best_model_name}.svg")
    plt.savefig(dca_output_path, dpi=300, format="svg", bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"✅ 最佳模型原始概率 DCA 决策曲线已保存至: {dca_output_path}")

else:
    print(f"模型 {best_model_name} 不支持 predict_proba，无法绘制决策曲线。")

print("\n✅ 所有模型训练、调参、评估、校准及DCA分析完成!")
print(f"最佳模型: {best_model_name}")
print("详细结果已保存至指定目录")