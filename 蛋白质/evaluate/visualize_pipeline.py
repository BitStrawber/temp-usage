# -*- coding: utf-8 -*-

import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr


# --- 修改：移除了全局中文字体设置 ---
# plt.rcParams['font.sans-serif'] = ['SimHei'] # 已删除
# plt.rcParams['axes.unicode_minus'] = False    # 已删除

def plot_correlation(ax, df, x_col, y_col, title, hue_col=None):
    """在指定的matplotlib轴上绘制散点图和相关性系数。"""
    subset = df[[x_col, y_col]].dropna()

    if len(subset) < 2:
        ax.text(0.5, 0.5, 'Not enough data', horizontalalignment='center', verticalalignment='center')
        ax.set_title(title, fontsize=12)
        return

    pearson_r, _ = pearsonr(subset[x_col], subset[y_col])
    spearman_r, _ = spearmanr(subset[x_col], subset[y_col])

    sns.scatterplot(
        x=x_col, y=y_col, data=df, ax=ax, hue=hue_col,
        palette={'是': '#2ca02c', '否': '#ff7f0e', '分析跳过': '#7f7f7f'} if hue_col else None,
        alpha=0.6, s=15, edgecolor=None
    )
    sns.regplot(
        x=x_col, y=y_col, data=subset, ax=ax, scatter=False, color='#e41a1c'
    )

    plot_text = f'Pearson r = {pearson_r:.3f}\nSpearman ρ = {spearman_r:.3f}'
    ax.text(0.05, 0.95, plot_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

    # --- 修改：将所有绘图相关的标签和标题改为英文 ---
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel('DMS Score (Experimental)', fontsize=11)
    ax.set_ylabel('Predicted Score (Model)', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.6)
    if hue_col:
        # 图例标题也改为英文
        ax.legend(title='Optimization Successful?')


def main():
    parser = argparse.ArgumentParser(description="可视化比较回归模型对不同序列的预测分数与真实DMS分数的相关性。")
    parser.add_argument("--results_file", type=str, required=True,
                        help="由run_pipeline.py生成的包含所有预测分数的CSV文件路径。")
    parser.add_argument("--output_plot", type=str, required=True, help="生成的对比图的保存路径。")

    args = parser.parse_args()

    # 1. 加载数据 - 日志保持中文
    try:
        df = pd.read_csv(args.results_file)
        df = df[~df[df.columns[0]].astype(str).str.contains("--- 统计结果 ---", na=False)]
        for col in df.columns:
            if 'score' in col:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    except FileNotFoundError as e:
        print(f"[错误] 无法找到文件: {e}")
        return

    print("数据加载和预处理成功。开始生成可视化图表...")

    # 2. 定义绘图配置 - 标题改为英文
    plot_configs = [
        {'title': 'Original Sequence Score vs. Experimental', 'y_col': 'predicted_score_mutated_sequence', 'hue': None},
        {'title': 'Base Model Sequence Score vs. Experimental', 'y_col': 'predicted_score_predicted_sequence_base',
         'hue': None},
        {'title': 'Finetuned Model Sequence Score vs. Experimental',
         'y_col': 'predicted_score_predicted_sequence_finetuned', 'hue': '是否优化成功'}
    ]

    # 3. 创建可视化图表 - 主标题改为英文
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle('Correlation Analysis: Predicted Score vs. Experimental DMS Score', fontsize=18, y=1.02)

    for i, config in enumerate(plot_configs):
        if config['y_col'] in df.columns and 'DMS_score' in df.columns:
            plot_correlation(axes[i], df, 'DMS_score', config['y_col'], config['title'], hue_col=config['hue'])
        else:
            axes[i].text(0.5, 0.5, f"Missing columns:\n'DMS_score' or\n'{config['y_col']}'",
                         ha='center', va='center', color='red')
            axes[i].set_title(config['title'], fontsize=14, pad=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # 4. 保存图表 - 日志保持中文
    output_dir = os.path.dirname(args.output_plot)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(args.output_plot, dpi=300, bbox_inches='tight')
    print(f"✅ 可视化图表已成功保存至: {args.output_plot}")


if __name__ == "__main__":
    main()

    python visualize_pipeline.py \
    --results_file "../../../总结果/(stability)SC6A4_HUMAN_Young_2021/prediction_results_scored.csv" \
    --output_plot "../../../总结果/可视化结果/(stability)SC6A4_HUMAN_Young_2021.png"

