# -*- coding: utf-8 -*-

import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# 设置matplotlib以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


def calculate_accuracy(row):
    """计算单个序列的预测准确率。"""
    original_seq = list(row['mutated_sequence'])
    masked_seq = list(row['masked_sequence'])
    predicted_seq = list(row['predicted_sequence'])

    masked_indices = [i for i, char in enumerate(masked_seq) if char == '<mask>']

    if not masked_indices:
        return 1.0  # 如果没有mask，则准确率为100%

    correct_predictions = 0
    for i in masked_indices:
        if original_seq[i] == predicted_seq[i]:
            correct_predictions += 1

    return correct_predictions / len(masked_indices)


def plot_correlation(ax, df, x_col, y_col, title):
    """在指定的matplotlib轴上绘制散点图和相关性系数。"""
    # 计算皮尔逊相关系数
    r, p = pearsonr(df[x_col], df[y_col])

    # 绘制带有回归线的散点图
    sns.regplot(
        x=x_col,
        y=y_col,
        data=df,
        ax=ax,
        scatter_kws={'alpha': 0.5, 's': 15},  # 设置散点透明度和大小
        line_kws={'color': 'red'}  # 设置回归线颜色
    )

    ax.set_title(f'{title}\n(Pearson r = {r:.3f})', fontsize=12)
    ax.set_xlabel('DMS Score (实验值)', fontsize=10)
    ax.set_ylabel('预测准确率', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)


def main():
    parser = argparse.ArgumentParser(description="比较不同模型预测结果与DMS分数的相关性。")
    parser.add_argument("--original_results", type=str, required=True, help="原始模型（基础+微调）的预测结果CSV文件路径。")
    parser.add_argument("--extended_results", type=str, required=True,
                        help="使用扩展数据训练后模型的新预测结果CSV文件路径。")
    parser.add_argument("--output_plot", type=str, required=True, help="生成的对比图的保存路径。")

    args = parser.parse_args()

    # --- 1. 加载数据 ---
    try:
        df_orig = pd.read_csv(args.original_results)
        df_ext = pd.read_csv(args.extended_results)
    except FileNotFoundError as e:
        print(f"[错误] 无法找到文件: {e}")
        return

    print("数据加载成功。开始计算预测准确率...")

    # --- 2. 计算每个模型的预测准确率 ---
    # 为每个模型的预测序列计算准确率
    df_orig['base_accuracy'] = df_orig.rename(columns={'predicted_sequence_base': 'predicted_sequence'}).apply(
        calculate_accuracy, axis=1)
    df_orig['original_finetuned_accuracy'] = df_orig.rename(
        columns={'predicted_sequence_finetuned': 'predicted_sequence'}).apply(calculate_accuracy, axis=1)
    df_ext['extended_finetuned_accuracy'] = df_ext.rename(
        columns={'predicted_sequence_finetuned': 'predicted_sequence'}).apply(calculate_accuracy, axis=1)

    # --- 3. 合并数据以便绘图 ---
    # 使用 'mutant' 或 'masked_sequence' 作为合并的键
    merge_key = 'mutant' if 'mutant' in df_orig.columns else 'masked_sequence'
    combined_df = pd.merge(
        df_orig[['DMS_score', merge_key, 'base_accuracy', 'original_finetuned_accuracy']],
        df_ext[[merge_key, 'extended_finetuned_accuracy']],
        on=merge_key
    )

    print("准确率计算完成。开始生成可视化图表...")

    # --- 4. 创建可视化图表 ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('模型预测准确率 vs. 实验DMS分数', fontsize=16, y=1.02)

    # 绘制基础模型的结果
    plot_correlation(axes[0], combined_df, 'DMS_score', 'base_accuracy', '基础模型')

    # 绘制原始微调模型的结果
    plot_correlation(axes[1], combined_df, 'DMS_score', 'original_finetuned_accuracy', '原始微调模型')

    # 绘制扩展数据微调模型的结果
    plot_correlation(axes[2], combined_df, 'DMS_score', 'extended_finetuned_accuracy', '扩展数据微调模型')

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整布局以适应主标题

    # --- 5. 保存图表 ---
    plt.savefig(args.output_plot, dpi=300, bbox_inches='tight')
    print(f"可视化图表已成功保存至: {args.output_plot}")

    # 显示图表 (如果您在支持GUI的环境中运行)
    # plt.show()


if __name__ == "__main__":
    # 命令行运行示例:
    # python visualize_results.py \
    #   --original_results "autodl-tmp/总结果/(activity)A0A247D711_LISMN_Stadelmann_2021/prediction_results_from_low_scores.csv" \
    #   --extended_results "autodl-tmp/总结果/((activity)A0A247D711_LISMN_Stadelmann_2021_extended)/prediction_results_scored.csv" \
    #   --output_plot "autodl-tmp/总结果/A0A247D711_LISMN_comparison.png"
    main()