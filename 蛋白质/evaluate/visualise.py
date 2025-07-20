# visualize.py

import pandas as pd
import numpy as np
import argparse
import os
import joblib  # 用于加载 scikit-learn 模型
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# 确保 utils.py 在同一个目录下或在 Python 路径中
# 从共享工具文件中导入函数
try:
    from utils import check_device, load_esm2_model, get_protein_embeddings_batch
except ImportError:
    print("错误: 无法导入 'utils.py'。请确保它与 visualize.py 在同一目录下。")
    exit(1)


def load_visualization_data(csv_path, sequence_col, score_col):
    """从CSV文件中加载用于可视化的数据。"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"错误: 数据文件 '{csv_path}' 未找到。")

    print(f"从 {csv_path} 加载数据...")
    df = pd.read_csv(csv_path)

    # 检查必需的列是否存在
    if sequence_col not in df.columns or score_col not in df.columns:
        raise ValueError(f"错误: CSV文件必须包含指定的序列列 '{sequence_col}' 和分数列 '{score_col}'。")

    # 移除包含缺失值的行
    df.dropna(subset=[sequence_col, score_col], inplace=True)
    if df.empty:
        raise ValueError("错误: 在移除缺失值后，没有有效的数据。")

    print(f"成功加载 {len(df)} 条有效样本用于可视化。")
    return df


def plot_predictions(y_true, y_pred, output_path):
    """将模型的预测值与真实值进行比较并可视化。"""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print("\n--- 整体性能评估 ---")
    print(f"  R² (决定系数): {r2:.4f}")
    print(f"  RMSE (均方根误差): {rmse:.4f}")

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(8, 8))

    sns.scatterplot(x=y_true, y=y_pred, alpha=0.7, color='darkgreen', label='预测值')
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='理想情况 (y=x)')

    plt.xlabel('真实的DMS分数', fontsize=12)
    plt.ylabel('预测的DMS分数', fontsize=12)
    plt.title(f'模型预测 vs. 真实分数 (R² = {r2:.4f})', fontsize=14)
    plt.legend()
    plt.tight_layout()

    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        plt.savefig(output_path, dpi=300)
        print(f"\n✅ 可视化图表已保存到: {output_path}")
    except Exception as e:
        print(f"❌ 保存图表时出错: {e}")


def main():
    parser = argparse.ArgumentParser(description="使用已训练的模型对蛋白质序列进行打分并可视化结果。")
    parser.add_argument("--csv_path", type=str, default="./DMS_ProteinGym_substitutions/A4GRB6_PSEAI_Chen_2020.csv", help="包含蛋白质序列和分数的CSV文件路径。")
    parser.add_argument("--model_path", type=str, default="./scoring_model2.pth",
                        help="已训练好的模型文件路径 (例如 'scoring_model.joblib')。")
    parser.add_argument("--figure_output_path", type=str, default="prediction_visualization2.png",
                        help="保存可视化图表的路径。")
    parser.add_argument("--sequence_column", type=str, default="mutated_sequence",
                        help="CSV文件中包含蛋白质序列的列名。")
    parser.add_argument("--score_column", type=str, default="DMS_score", help="CSV文件中包含真实分数的列名。")
    parser.add_argument("--model_name", type=str, default="esm2_t12_35M_UR50D", help="用于生成嵌入向量的ESM-2模型名称。")
    parser.add_argument("--repr_layer", type=int, default=12, help="从ESM模型的哪一层提取表征。")
    parser.add_argument("--batch_size", type=int, default=16, help="生成嵌入向量时的批次大小。")

    args = parser.parse_args()

    # 1. 加载数据
    df = load_visualization_data(args.csv_path, args.sequence_column, args.score_column)

    # 2. 初始化 ESM-2 模型
    device = check_device()
    model_esm, alphabet, batch_converter = load_esm2_model(args.model_name, device)

    # 3. 为所有序列生成嵌入向量
    print("\n--- 正在生成蛋白质嵌入向量... ---")
    sequences = df[args.sequence_column].tolist()
    embeddings = get_protein_embeddings_batch(
        sequences, model_esm, alphabet, batch_converter, device, args.repr_layer, args.batch_size
    )

    # 4. 加载训练好的打分模型
    print(f"\n--- 正在从 {args.model_path} 加载模型... ---")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"错误: 模型文件 '{args.model_path}' 未找到。")
    try:
        scoring_model = joblib.load(args.model_path)
        print("✅ 模型加载成功。")
    except Exception as e:
        raise IOError(f"加载模型文件时出错: {e}")

    # 5. 使用模型进行预测
    print("\n--- 正在使用加载的模型进行预测... ---")
    predicted_scores = scoring_model.predict(embeddings)
    print("✅ 预测完成。")

    # 6. 可视化结果
    true_scores = df[args.score_column].values
    plot_predictions(true_scores, predicted_scores, args.figure_output_path)


if __name__ == "__main__":
    main()