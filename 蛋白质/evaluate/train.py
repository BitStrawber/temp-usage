# train.py

import pandas as pd
import numpy as np
import argparse
import os
import joblib  # 用于保存和加载 scikit-learn 模型
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# 从共享工具文件中导入函数
from utils import check_device, load_esm2_model, get_protein_embeddings_batch


def load_training_data(csv_path, sequence_col, score_col):
    """从CSV文件中加载训练数据并进行基本检查。"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"错误: 训练文件 '{csv_path}' 未找到。")

    print(f"从 {csv_path} 加载训练数据...")
    df = pd.read_csv(csv_path)

    # 检查必需的列是否存在
    if sequence_col not in df.columns or score_col not in df.columns:
        raise ValueError(f"错误: CSV文件必须包含指定的序列列 '{sequence_col}' 和分数列 '{score_col}'。")

    # 移除包含缺失值的行
    df.dropna(subset=[sequence_col, score_col], inplace=True)
    if df.empty:
        raise ValueError("错误: 在移除缺失值后，没有有效的训练数据。")

    print(f"成功加载 {len(df)} 条有效训练样本。")
    return df


def train_and_evaluate(X, y, test_size=0.2, random_state=42):
    """训练线性回归模型并评估其性能。"""
    print("\n--- 模型训练与评估 ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_test_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    print(f"\n模型在测试集上的性能:")
    print(f"  R² (决定系数): {test_r2:.4f}")
    print(f"  RMSE (均方根误差): {test_rmse:.4f}")

    return model, (y_test, y_test_pred)


def plot_evaluation(results, output_path):
    """将模型在测试集上的表现可视化。"""
    y_test, y_test_pred = results

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(8, 8))

    sns.scatterplot(x=y_test, y=y_test_pred, alpha=0.7, color='royalblue', label='预测值')
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='理想情况 (y=x)')

    plt.xlabel('真实的DMS分数', fontsize=12)
    plt.ylabel('预测的DMS分数', fontsize=12)
    plt.title(f'模型性能评估 (R² = {r2_score(y_test, y_test_pred):.4f})', fontsize=14)
    plt.legend()
    plt.tight_layout()

    try:
        plt.savefig(output_path, dpi=300)
        print(f"\n✅ 评估图表已保存到: {output_path}")
    except Exception as e:
        print(f"❌ 保存图表时出错: {e}")


def main():
    parser = argparse.ArgumentParser(description="使用蛋白质序列和分数训练一个打分模型。")
    parser.add_argument("--csv_path", type=str, required=True, help="包含训练数据的CSV文件路径。")
    parser.add_argument("--sequence_column", type=str, default="mutated_sequence",
                        help="CSV文件中包含蛋白质序列的列名。")
    parser.add_argument("--score_column", type=str, default="DMS_score", help="CSV文件中包含分数的列名。")
    parser.add_argument("--model_output_path", type=str, default="scoring_model2.pth",
                        help="保存训练好的模型权重的文件路径。")
    parser.add_argument("--figure_output_path", type=str, default="training_evaluation.png",
                        help="保存模型评估图表的路径。")
    parser.add_argument("--model_name", type=str, default="esm2_t12_35M_UR50D", help="用于生成嵌入向量的ESM-2模型名称。")
    parser.add_argument("--repr_layer", type=int, default=12, help="从ESM模型的哪一层提取表征。")
    parser.add_argument("--batch_size", type=int, default=16, help="生成嵌入向量时的批次大小。")

    args = parser.parse_args()

    # 1. 初始化和加载
    device = check_device()
    df = load_training_data(args.csv_path, args.sequence_column, args.score_column)
    model_esm, alphabet, batch_converter = load_esm2_model(args.model_name, device)

    # 2. 生成嵌入向量
    sequences = df[args.sequence_column].tolist()
    embeddings = get_protein_embeddings_batch(
        sequences, model_esm, alphabet, batch_converter, device, args.repr_layer, args.batch_size
    )

    # 3. 训练模型
    scores = df[args.score_column].values
    trained_model, eval_results = train_and_evaluate(embeddings, scores)

    # 4. 保存训练好的模型
    try:
        # 获取输出路径的目录部分
        output_dir = os.path.dirname(args.model_output_path)

        # 只有当目录名不为空时，才尝试创建目录
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"已创建目录: {output_dir}")
        joblib.dump(trained_model, args.model_output_path)
        print(f"✅ 模型已成功保存到: {args.model_output_path}")
    except Exception as e:
        print(f"❌ 保存模型时出错: {e}")

    # 5. 可视化评估结果
    plot_evaluation(eval_results, args.figure_output_path)


if __name__ == "__main__":
    main()