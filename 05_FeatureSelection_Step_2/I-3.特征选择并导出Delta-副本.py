import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
import pymrmr

# === 参数设置 ===
# Lasso 阈值：系数小于此值的特征被视为不重要
LASSO_COEF_THRESHOLD = 1e-4
# mRMR 选择的最大特征数（您之前设置为 min(21, X_train.shape[1])）
MRMR_N_FEATURES = 15 # 建议选择一个明确的、较小的数量，如 10 个

# === 步骤 1：加载训练集和验证集===
train_file = r"F:\2023\Pyradiomics结果\第五次尝试\04d_Processing_Results_影像+临床\影像+临床_train_data.xlsx"
val_file = r"F:\2023\Pyradiomics结果\第五次尝试\04d_Processing_Results_影像+临床\影像+临床_valid_data.xlsx"

train_df = pd.read_excel(train_file)
val_df = pd.read_excel(val_file)

# 假设所有非 ID/Label 的列都是特征
id_cols = [col for col in train_df.columns if col.lower() in ['patientid', 'image', 'mask', 'age']]
feature_cols = [col for col in train_df.columns if col not in id_cols + ['label']]

X_train = train_df[feature_cols].copy()
y_train = train_df['label']
X_val = val_df[feature_cols].copy()
y_val = val_df['label']

print(f"当前特征数量: {X_train.shape[1]}")

# === 创建输出目录 ===
output_dir = r"F:\2023\Pyradiomics结果\第五次尝试\05_feature_importance"
os.makedirs(output_dir, exist_ok=True)

# 为每种方法创建子目录
methods = ['Lasso', 'mRMR']
for method in methods:
    os.makedirs(os.path.join(output_dir, method), exist_ok=True)

# ---------------- Lasso (L1 正则化) ----------------
# LassoCV 自动进行交叉验证选择最优 alpha (正则化强度)
lasso = LassoCV(cv=5, random_state=42, max_iter=500000, n_jobs=-1).fit(X_train, y_train)

# 修正 1: 计算特征的绝对值系数作为重要性
lasso_importance = np.abs(lasso.coef_)
lasso_result = pd.DataFrame({
    'Feature': X_train.columns,
    'Lasso_Importance': lasso_importance # 命名更清晰
}).sort_values(by="Lasso_Importance", ascending=False)

# 修正 2: 选择系数绝对值大于阈值的特征 (排除微小的非零系数)
lasso_selected_features = lasso_result[lasso_result['Lasso_Importance'] >= LASSO_COEF_THRESHOLD]['Feature'].tolist()

# 保存筛选后的数据集
if lasso_selected_features:
    # 确保ID列和标签一起保存
    train_lasso = pd.concat([train_df[id_cols + ['label']], X_train[lasso_selected_features]], axis=1)
    val_lasso = pd.concat([val_df[id_cols + ['label']], X_val[lasso_selected_features]], axis=1)

    train_lasso.to_excel(os.path.join(output_dir, 'Lasso', 'train_lasso_selected.xlsx'), index=False)
    val_lasso.to_excel(os.path.join(output_dir, 'Lasso', 'val_lasso_selected.xlsx'), index=False)

lasso_result.to_excel(os.path.join(output_dir, 'Lasso', '05_lasso_importance.xlsx'), index=False)
print("✅ Lasso 特征排序完成，最优 Alpha:", lasso.alpha_)
print(f"✅ Lasso 筛选后特征数量 (Importance > {LASSO_COEF_THRESHOLD}): {len(lasso_selected_features)}")

# ---------------- mRMR (Minimum Redundancy Maximum Relevance) ----------------
# 修正 3: mRMR 的输入数据结构和特征数量
mrmr_data = X_train.join(y_train.rename('label')) # 确保标签列名为 'label'
n_features_to_select = min(MRMR_N_FEATURES, X_train.shape[1])

# mRMR 函数返回按重要性排序的特征列表
mrmr_result_list = pymrmr.mRMR(mrmr_data, 'MIQ', n_features_to_select)

# mRMR选择的特征
mrmr_selected_features = [feat for feat in mrmr_result_list if feat != 'label']

# 修正 4: 结果DataFrame和排名分数
mrmr_result_df = pd.DataFrame({
    'Feature': mrmr_selected_features,
    # 排名：第1名重要性最高，得分为 n
    'mRMR_Importance': list(range(len(mrmr_selected_features), 0, -1))
})

# 保存筛选后的数据集
if mrmr_selected_features:
    train_mrmr = pd.concat([train_df[id_cols + ['label']], X_train[mrmr_selected_features]], axis=1)
    val_mrmr = pd.concat([val_df[id_cols + ['label']], X_val[mrmr_selected_features]], axis=1)

    train_mrmr.to_excel(os.path.join(output_dir, 'mRMR', 'train_mrmr_selected.xlsx'), index=False)
    val_mrmr.to_excel(os.path.join(output_dir, 'mRMR', 'val_mrmr_selected.xlsx'), index=False)

mrmr_result_df.to_excel(os.path.join(output_dir, 'mRMR', '05_mrmr_importance.xlsx'), index=False)
print(f"✅ mRMR 特征排序完成，筛选后特征数量: {len(mrmr_selected_features)}")


# ---------------- 合并并标准化 ----------------
# 修正 5: 避免列名覆盖，使用清晰的列名进行 merge
all_results = pd.DataFrame({'Feature': feature_cols})

all_results = all_results.merge(lasso_result[['Feature', 'Lasso_Importance']], on="Feature", how="left")
all_results = all_results.merge(mrmr_result_df[['Feature', 'mRMR_Importance']], on="Feature", how="left")

# 填充NaN (未被该方法选中的特征重要性为 0)
all_results = all_results.fillna(0)

# 标准化处理 (Min-Max 到 0-1 区间)
scaler = MinMaxScaler()
importance_cols = ['Lasso_Importance', 'mRMR_Importance']
all_results[importance_cols] = scaler.fit_transform(all_results[importance_cols])

# 修正 6: 保持列名清晰
all_results.columns = ["Feature", "Lasso", "mRMR"]

# 保存合并结果
all_results.to_excel(os.path.join(output_dir, "05_all_methods_normalized.xlsx"), index=False)
print("✅ 合并特征重要性结果已保存 (all_methods_normalized.xlsx)")


# ---------------- 可视化 ----------------
# 修正 7: 调整图形尺寸和标签
# 只绘制被至少一个方法选中的特征
plot_df = all_results[(all_results['Lasso'] > 0) | (all_results['mRMR'] > 0)].copy()

plt.figure(figsize=(15, 7)) # 增大画布
plot_df.set_index('Feature').plot(kind='bar', figsize=(15, 7), ax=plt.gca()) # ax=plt.gca() 确保绘图到当前 figure

plt.ylabel("Normalized Importance (0-1)", fontsize=12)
plt.title("Feature Importance Comparison (Lasso vs. mRMR)", fontsize=14)
plt.xlabel("Feature Name", fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10) # 旋转 X 轴标签
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "05_feature_importance_comparison.png"), dpi=300)
plt.close()

print("✅ 可视化完成，保存为 feature_importance_comparison.png")