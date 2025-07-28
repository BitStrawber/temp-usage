# visualize_results.py
# (合并版: 三联相关性图 + 五种分布图)

import pandas as pd
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm


# ==============================================================================
# 绘图函数
# ==============================================================================

def plot_correlation_comparison(df, output_dir):
    """
    生成三联相关性对比图。
    """
    print("\n--- 任务 1: 生成三联相关性对比图 ---")

    # 准备数据
    if '是否优化成功' in df.columns:
        df['Optimization_Successful_EN'] = df['是否优化成功'].map(
            {'是': 'Yes', '否': 'No', '分析跳过': 'Skipped'}).fillna('Skipped')
    else:
        # 如果列不存在，创建一个默认列以避免绘图错误
        df['Optimization_Successful_EN'] = 'N/A'

    # 定义绘图配置 (适应新的列名)
    plot_configs = [
        {'title': 'Original Sequence Score vs. Experimental', 'y_col': 'predicted_score_mutated_sequence', 'hue': None,
         'point_color': '#A6C8FF'},
        {'title': 'Base Model Generated Score vs. Experimental', 'y_col': 'predicted_score_predicted_sequence',
         'hue': None, 'point_color': '#FFB6C1'},
        {'title': 'Trained Model Generated Score vs. Experimental',
         'y_col': 'predicted_score_predicted_sequence_trained', 'hue': 'Optimization_Successful_EN',
         'point_color': None}
    ]

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.suptitle('Correlation Analysis: Predicted Score vs. Experimental DMS Score', fontsize=18, y=1.0)

    for i, config in enumerate(plot_configs):
        if config['y_col'] in df.columns and 'DMS_score' in df.columns:
            plot_single_correlation(axes[i], df, 'DMS_score', config['y_col'], config['title'],
                                    hue_col=config['hue'], point_color=config['point_color'])
        else:
            missing = [col for col in ['DMS_score', config['y_col']] if col not in df.columns]
            axes[i].text(0.5, 0.5, f"Missing columns:\n{', '.join(missing)}", ha='center', va='center', color='red')
            axes[i].set_title(config['title'], fontsize=14, pad=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = os.path.join(output_dir, "1_correlation_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 三联对比图已成功保存至: {output_path}")
    plt.close(fig)


def plot_single_correlation(ax, df, x_col, y_col, title, hue_col=None, point_color=None):
    """
    (辅助函数) 绘制单个相关性散点图。
    """
    subset = df[[x_col, y_col]].dropna()
    if len(subset) < 2:
        ax.text(0.5, 0.5, 'Not enough data', horizontalalignment='center', verticalalignment='center')
        ax.set_title(title, fontsize=12)
        return
    pearson_r, _ = pearsonr(subset[x_col], subset[y_col])
    spearman_r, _ = spearmanr(subset[x_col], subset[y_col])

    palette = {'Yes': '#2ca02c', 'No': '#d62728', 'Skipped': '#7f7f7f', 'N/A': '#7f7f7f'}

    if hue_col and hue_col in df.columns:
        sns.scatterplot(x=x_col, y=y_col, data=df, ax=ax, hue=hue_col, palette=palette, alpha=0.8, s=15, edgecolor=None)
        ax.legend(title='Optimization Successful?', loc='lower right')
    else:
        sns.scatterplot(x=x_col, y=y_col, data=df, ax=ax, color=point_color, alpha=0.8, s=15, edgecolor=None)

    sns.regplot(x=x_col, y=y_col, data=subset, ax=ax, scatter=False, color='#e41a1c', ci=None)

    plot_text = f'Pearson r = {pearson_r:.3f}\nSpearman ρ = {spearman_r:.3f}'
    ax.text(0.05, 0.95, plot_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel('DMS Score (Experimental)', fontsize=11)
    ax.set_ylabel('Predicted Score (Model)', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.6)


def plot_score_distributions(df, output_dir):
    """
    生成五种不同的分数分布图。
    """
    print("\n--- 任务 2: 生成分数分布图 ---")

    # 定义列映射 (适应新的列名)
    columns_map = {
        'predicted_score_mutated_sequence': 'Original Sequence Score',
        'predicted_score_predicted_sequence': 'Base Model Generated Score',
        'predicted_score_predicted_sequence_trained': 'Trained Model Generated Score'
    }

    # 筛选出实际存在的列
    value_vars_original = [col for col in columns_map.keys() if col in df.columns]
    if not value_vars_original:
        print("⚠️ 警告: 找不到任何预测分数相关的列，已跳过分布图生成。")
        return

    # "融化"DataFrame为长格式
    df_melted = df.melt(
        value_vars=value_vars_original,
        var_name='Score Type (Original)',
        value_name='Predicted Score'
    )
    df_melted['Score Type'] = df_melted['Score Type (Original)'].map(columns_map)

    # 设置主题
    sns.set_theme(style="whitegrid", palette="viridis")

    # --- 图 1: KDE 图 ---
    plt.figure(figsize=(12, 7))
    sns.kdeplot(data=df_melted, x='Predicted Score', hue='Score Type', fill=True, alpha=0.5, linewidth=2.5)
    plt.title('Score Distribution Comparison (KDE Plot)', fontsize=16, pad=20)
    plt.xlabel('Predicted Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.tight_layout()
    output_path = os.path.join(output_dir, '2_kde_plot.png')
    plt.savefig(output_path, dpi=300)
    print(f"✅ 已生成 KDE 图至: {output_path}")
    plt.close()

    # --- 图 2: 直方图 ---
    plt.figure(figsize=(12, 7))
    sns.histplot(data=df_melted, x='Predicted Score', hue='Score Type', kde=True, stat='density', common_norm=False,
                 element='step', fill=False)
    plt.title('Score Distribution Comparison (Histogram)', fontsize=16, pad=20)
    plt.xlabel('Predicted Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.tight_layout()
    output_path = os.path.join(output_dir, '3_histogram_plot.png')
    plt.savefig(output_path, dpi=300)
    print(f"✅ 已生成直方图至: {output_path}")
    plt.close()

    # --- 图 3: 箱形图 ---
    plt.figure(figsize=(10, 8))
    sns.boxplot(data=df_melted, x='Score Type', y='Predicted Score')
    plt.title('Score Statistics Comparison (Box Plot)', fontsize=16, pad=20)
    plt.xlabel('')
    plt.ylabel('Predicted Score', fontsize=12)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    output_path = os.path.join(output_dir, '4_box_plot.png')
    plt.savefig(output_path, dpi=300)
    print(f"✅ 已生成箱形图至: {output_path}")
    plt.close()

    # --- 图 4: 小提琴图 ---
    plt.figure(figsize=(10, 8))
    sns.violinplot(data=df_melted, x='Score Type', y='Predicted Score')
    plt.title('Score Density and Statistics (Violin Plot)', fontsize=16, pad=20)
    plt.xlabel('')
    plt.ylabel('Predicted Score', fontsize=12)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    output_path = os.path.join(output_dir, '5_violin_plot.png')
    plt.savefig(output_path, dpi=300)
    print(f"✅ 已生成小提琴图至: {output_path}")
    plt.close()

    # --- 图 5: ECDF 图 ---
    plt.figure(figsize=(12, 7))
    sns.ecdfplot(data=df_melted, x='Predicted Score', hue='Score Type', linewidth=2.5)
    plt.title('Cumulative Score Distribution (ECDF Plot)', fontsize=16, pad=20)
    plt.xlabel('Predicted Score', fontsize=12)
    plt.ylabel('Proportion of Data', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    output_path = os.path.join(output_dir, '6_ecdf_plot.png')
    plt.savefig(output_path, dpi=300)
    print(f"✅ 已生成 ECDF 图至: {output_path}")
    plt.close()


# ==============================================================================
# 主函数
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="生成蛋白质工程结果的多种可视化图表。")
    # --- 核心参数简化为目录 ---
    parser.add_argument("--input_dir", type=str, required=True,
                        help="包含 'prediction_results_scored.csv' 文件的输入目录。")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="所有可视化图表的输出目录。")

    args = parser.parse_args()

    # --- 1. 路径设置与数据加载 ---
    os.makedirs(args.output_dir, exist_ok=True)
    results_file_path = os.path.join(args.input_dir, "prediction_results_scored.csv")

    if not os.path.exists(results_file_path):
        print(f"❌ 错误: 找不到结果文件 '{results_file_path}'。脚本将退出。")
        return

    try:
        df = pd.read_csv(results_file_path)
        df = df[~df[df.columns[0]].astype(str).str.contains("--- 统计结果 ---", na=False)]
        score_cols = [col for col in df.columns if 'score' in col]
        for col in score_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 为了分布图，需要dropna来确保数据一致性
        df.dropna(subset=[
            'predicted_score_mutated_sequence',
            'predicted_score_predicted_sequence',
            'predicted_score_predicted_sequence_trained'
        ], how='any', inplace=True)

        if df.empty:
            print("❌ 错误: 清理后没有剩余数据可供可视化。请检查CSV文件内容。")
            return

    except Exception as e:
        print(f"❌ 错误: 加载或处理文件 '{results_file_path}' 时失败: {e}")
        return

    print("数据加载和预处理成功。")

    # --- 2. 任务 1: 生成三联相关性对比图 ---
    plot_correlation_comparison(df.copy(), args.output_dir)  # 使用副本以防修改原df

    # --- 3. 任务 2: 生成五种分数分布图 ---
    plot_score_distributions(df.copy(), args.output_dir)

    print(f"\n🎉 所有可视化图表已成功保存到目录: {args.output_dir}")


if __name__ == "__main__":
    main()

    python visualize_pipeline.py \
    --input_dir "../蛋白质数据/A0A192B1T2_9HIV1_Haddox_2018/" \
    --output_dir "../蛋白质数据/A0A192B1T2_9HIV1_Haddox_2018/可视化结果/"


python step4打分结果可视化.py \
  --results_file "../蛋白质数据/A0A192B1T2_9HIV1_Haddox_2018/prediction_results_scored.csv" \
  --output_dir "../蛋白质数据/A0A192B1T2_9HIV1_Haddox_2018/可视化结果/"