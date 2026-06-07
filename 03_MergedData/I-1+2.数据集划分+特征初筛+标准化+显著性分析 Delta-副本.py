# ================== 优化后的完整流程（增加多重比较和可视化） ==================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from scipy.stats import mannwhitneyu
from statsmodels.sandbox.stats.multicomp import multipletests  # 引入多重比较校正
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# 定义全局输出目录和输入文件路径，避免硬编码
INPUT_FILE = r"F:\2023\Pyradiomics结果\第五次尝试\01d_影像+临床版-v4-374-557-931.xlsx"
BASE_OUTPUT_DIR = r"F:\2023\Pyradiomics结果\第五次尝试"
OUTPUT_STEP_FOLDER = os.path.join(BASE_OUTPUT_DIR, '04_Processing_Results')
P_VALUE_THRESHOLD = 0.05  # 显著性水平


def correct_workflow_optimized():
    # 确保输出目录存在
    os.makedirs(OUTPUT_STEP_FOLDER, exist_ok=True)

    # === 步骤1：加载原始数据 ===
    df = pd.read_excel(INPUT_FILE)
    print(f"原始数据形状: {df.shape}")

    # === 步骤2：首先划分训练集和验证集 (8:2) ===
    df = df.reset_index(drop=True)
    train_df, valid_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label']
    )

    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)

    print(f"训练集大小: {train_df.shape}, 验证集大小: {valid_df.shape}")

    # 3.1 识别ID列和标签
    y_train = train_df['label']
    y_valid = valid_df['label']

    id_cols = [col for col in train_df.columns if col.lower() in ['patientid', 'image', 'mask', 'age']]

    X_train_full = train_df.drop(columns=id_cols + ['label'], errors='ignore')

    # 确保验证集只包含特征列
    X_valid_full = valid_df.drop(columns=id_cols + ['label'], errors='ignore')

    # === 步骤3：在训练集上进行特征初筛 ===

    # 3.2 丢弃diagnostics特征
    diagnostics_cols = [col for col in X_train_full.columns if col.startswith('diagnostics')]
    X_train = X_train_full.drop(columns=diagnostics_cols, errors='ignore')

    # 3.3 删除缺失率 > 30% 的特征（仅在训练集上计算）
    missing_ratio = X_train.isnull().mean()
    features_to_keep_missing = missing_ratio[missing_ratio <= 0.3].index.tolist()
    X_train = X_train[features_to_keep_missing]
    print(f"训练集缺失率筛选后特征数: {X_train.shape[1]}")

    # 获取训练集特征的均值用于填充（Imputation Strategy）
    train_mean_imputation = X_train.mean()

    # 3.4 低方差过滤（仅在训练集上计算）
    selector = VarianceThreshold(threshold=0.05)
    # 使用训练集均值填充NaN值
    X_train_filled_for_selection = X_train.fillna(train_mean_imputation)

    # 保存用于拟合 VarianceThreshold 的特征列名
    cols_used_for_variance_fit = X_train_filled_for_selection.columns

    selector.fit(X_train_filled_for_selection)

    # 获取最终选择的特征列名
    selected_features = X_train.columns[selector.get_support()].tolist()
    X_train_final_selected = X_train[selected_features]
    print(f"训练集低方差筛选后特征数: {X_train_final_selected.shape[1]}")

    # 可视化：特征间相关性（在填充后/标准化前）
    plt.figure(figsize=(36, 27))
    sns.heatmap(X_train_filled_for_selection[selected_features].corr(), cmap='viridis', linewidths=.5, annot=False)
    plt.tight_layout()
    plt.title('Feature Correlation Matrix (Selected Features, Training Set)')
    plt.savefig(os.path.join(OUTPUT_STEP_FOLDER, '3_correlation_matrix.svg'))
    plt.close()
    sns.clustermap(
        X_train_filled_for_selection[selected_features].corr(),
        method='average',  # 指定聚类方法，默认是 'average' (平均链接)
        metric='euclidean',  # 指定距离度量方法，默认是 'euclidean' (欧氏距离)
        cmap='viridis',  # 指定颜色映射，与普通热力图类似
        figsize=(30, 30),  # 设置图幅大小，确保可读性
        dendrogram_ratio=(0.1, 0.1),
        annot=False,  # 一般在特征过多时不显示数值
        linewidths=0.15,  # 增加方块之间的线宽
        linecolor='white'
    )
    plt.title('Feature Correlation Matrix')
    plt.savefig(os.path.join(OUTPUT_STEP_FOLDER, '3_correlation_clustermatrix.svg'))
    plt.close()


    fig = sns.clustermap(
        X_train_filled_for_selection[selected_features].corr(),
        method='average',
        metric='euclidean',
        cmap='viridis',
        figsize=(30, 30),
        dendrogram_ratio=(0.1, 0.1),
        annot=False,
        linewidths=0.15,
        linecolor='white',
        cbar=False,
    )
    # --- 隐藏颜色条刻度的代码 ---
    fig.ax_cbar.set_xticklabels([])
    fig.ax_cbar.set_yticklabels([])
    # 设置图表标题
    #fig.fig.suptitle('Feature Correlation Matrix', fontsize=30)
    # 调整主热力图刻度标签字体大小
    plt.setp(fig.ax_heatmap.get_xticklabels(), fontsize=8, ha='right')
    plt.setp(fig.ax_heatmap.get_yticklabels(), fontsize=8)
    fig.fig.tight_layout()
    # 保存并关闭
    plt.savefig(os.path.join(OUTPUT_STEP_FOLDER, '3_correlation_clustermatrix_no_cbar_ticks.svg'), dpi=300)
    plt.close()

    print("聚类热力图已生成，颜色条刻度已隐藏。")

    # === 步骤4：将特征选择应用到验证集并处理缺失值 ===
    # 验证集应用训练集的特征选择结果
    X_valid_selected = X_valid_full[selected_features].copy()

    # 使用训练集的均值填充验证集的NaN值
    X_valid_filled = X_valid_selected.fillna(train_mean_imputation)

    # === 步骤5：在训练集上进行标准化（Standardization） ===
    # StandardScaler 仅在训练集上 fit
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_filled_for_selection[selected_features])
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=selected_features)

    # 使用训练集的标准化参数转换验证集 (transform only)
    X_valid_scaled = scaler.transform(X_valid_filled)
    X_valid_scaled = pd.DataFrame(X_valid_scaled, columns=selected_features)

    # 保存标准化器和填充均值（用于外部验证）
    joblib.dump(scaler, os.path.join(OUTPUT_STEP_FOLDER, '4_StandardScaler.pkl'))
    joblib.dump(train_mean_imputation, os.path.join(OUTPUT_STEP_FOLDER, '4_Imputation_Means.pkl'))

    # === 步骤6：在训练集上进行显著性分析（单因素） ===
    results = []

    # 确保索引一致
    y_train_reset = y_train.reset_index(drop=True)
    X_train_scaled = X_train_scaled.reset_index(drop=True)

    for col in selected_features:
        try:
            # 确保分组数据来自标准化后的训练集
            group0 = X_train_scaled.loc[y_train_reset == 0, col]
            group1 = X_train_scaled.loc[y_train_reset == 1, col]

            if len(group0) > 1 and len(group1) > 1:  # 确保每组至少有两个样本
                stat, p = mannwhitneyu(group0, group1, alternative='two-sided')
                results.append({'Feature': col, 'p_value_raw': p})
            else:
                results.append({'Feature': col, 'p_value_raw': np.nan})
        except Exception as e:
            # print(f"处理特征 {col} 时出错: {e}")
            results.append({'Feature': col, 'p_value_raw': np.nan})

    results_df = pd.DataFrame(results).dropna(subset=['p_value_raw'])

    # *********** 多重比较校正 (Benjamini-Hochberg/FDR) ***********
    reject, p_adjusted, _, _ = multipletests(results_df['p_value_raw'], method='fdr_bh')
    results_df['p_value_adjusted'] = p_adjusted

    results_df = results_df.sort_values(by="p_value_adjusted", ascending=True).reset_index(drop=True)
    results_df['Significant'] = results_df['p_value_adjusted'] < P_VALUE_THRESHOLD

    # 选择显著特征
    significant_features = results_df[results_df['Significant']]['Feature'].tolist()
    print(f"经 FDR 校正后，显著特征数量: {len(significant_features)}")

    # 可视化：显著特征的箱线图
    if significant_features:
        # 准备绘图数据
        plot_df = X_train_scaled[significant_features].copy()
        plot_df['Label'] = y_train_reset

        n_plots = len(significant_features)
        cols = 3
        rows = int(np.ceil(n_plots / cols))

        plt.figure(figsize=(5 * cols, 4 * rows))
        for i, feature in enumerate(significant_features):
            plt.subplot(rows, cols, i + 1)
            sns.boxplot(x='Label', y=feature, data=plot_df)
            p_adj = results_df[results_df['Feature'] == feature]['p_value_adjusted'].values[0]
            plt.title(f'{feature}\n(Adj P={p_adj:.3e})')

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_STEP_FOLDER, '6_significant_features_boxplots.svg'))
        plt.close()

    # === 步骤7：准备最终数据集 ===
    # 训练集（仅显著特征）
    final_train = pd.concat([
        train_df[id_cols].reset_index(drop=True),
        y_train_reset,
        X_train_scaled[significant_features]
    ], axis=1)

    # 验证集（仅显著特征）
    final_valid = pd.concat([
        valid_df[id_cols].reset_index(drop=True),
        y_valid.reset_index(drop=True),
        X_valid_scaled[significant_features].reset_index(drop=True)  # 确保索引正确对齐
    ], axis=1)

    # 步骤8：保存结果
    # 收集特征选择信息
    # 使用拟合时的列名来索引 selector 的支持布尔数组
    variance_cols_supported = cols_used_for_variance_fit[selector.get_support()]

    # 提取通过低方差筛选的特征对应的方差
    variance_data = {}
    for i, col in enumerate(cols_used_for_variance_fit):
        if selector.get_support()[i]:
            # 只有被选中的特征，才记录其方差
            variance_data[col] = selector.variances_[i]

    # 构建最终的 feature_selection_info DataFrame
    feature_selection_info = pd.DataFrame({
        'feature': selected_features,
        # 缺失率和显著性信息应以 selected_features 为基准进行匹配
        'missing_ratio_train': [missing_ratio.get(feat, np.nan) for feat in selected_features],
        'variance_train': [variance_data.get(feat, np.nan) for feat in selected_features],

        # 合并显著性信息
        'p_value_raw': [results_df[results_df['Feature'] == feat]['p_value_raw'].values[0] if feat in results_df[
            'Feature'].values else np.nan for feat in selected_features],
        'p_value_adjusted': [
            results_df[results_df['Feature'] == feat]['p_value_adjusted'].values[0] if feat in results_df[
                'Feature'].values else np.nan for feat in selected_features],
        'significant_fdr': [feat in significant_features for feat in selected_features]
    })

    final_train.to_excel(os.path.join(OUTPUT_STEP_FOLDER, 'final_train_data.xlsx'), index=False)
    final_valid.to_excel(os.path.join(OUTPUT_STEP_FOLDER, 'final_valid_data.xlsx'), index=False)
    results_df.to_excel(os.path.join(OUTPUT_STEP_FOLDER, 'significance_analysis_fdr.xlsx'), index=False)
    feature_selection_info.to_excel(os.path.join(OUTPUT_STEP_FOLDER, 'feature_selection_summary.xlsx'), index=False)

    print("✅ 优化后的流程完成！结果已保存到:")
    print(f"   {OUTPUT_STEP_FOLDER}")
    print(f"   显著训练集: {final_train.shape}")
    print(f"   显著验证集: {final_valid.shape}")
    print(f"   最终显著特征数: {len(significant_features)}")

    return final_train, final_valid, scaler, significant_features


# 运行优化后的流程
try:
    final_train, final_valid, scaler, sig_features = correct_workflow_optimized()
    print("✅ 流程执行成功！")
except Exception as e:
    print(f"❌ 流程执行失败: {e}")
    import traceback

    traceback.print_exc()