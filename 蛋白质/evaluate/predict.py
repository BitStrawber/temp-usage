# predict.py

import pandas as pd
import numpy as np
import argparse
import os
import joblib

# 从共享工具文件中导入函数
from utils import check_device, load_esm2_model, get_protein_embeddings_batch


def load_prediction_data(csv_path):
    """从CSV文件中加载待预测的数据。"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ 错误: 待预测文件 '{csv_path}' 未找到。")

    print(f"从 {csv_path} 加载待预测序列...")
    df = pd.read_csv(csv_path)
    print(f"✅ 成功加载 {len(df)} 行数据。")
    return df


def load_scoring_model(model_path):
    """加载之前训练好的打分模型。"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ 错误: 模型文件 '{model_path}' 未找到。请先运行 train.py 进行训练。")

    print(f"正在从 {model_path} 加载打分模型...")
    model = joblib.load(model_path)
    print("✅ 模型加载成功。")
    return model


def analyze_and_annotate(df, trained_seq_col, gen_seq_col, original_score_col):
    """
    分析预测分数，对满足特定优化条件的序列进行标注和统计。
    条件: 微调模型预测分 > 原始模型预测分 > 原始序列真实分
    """
    print("\n--- 正在进行优化分析和标注 ---")

    # 1. 定义用于比较的列名
    # 请注意：列名是根据主函数中的命名规则 'predicted_score_' + sequence_column_name 生成的
    score_col_trained = f"predicted_score_{trained_seq_col}"
    score_col_gen = f"predicted_score_{gen_seq_col}"

    # 2. 检查所有需要的列是否存在于DataFrame中
    required_cols = [score_col_trained, score_col_gen, original_score_col]
    if not all(col in df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in df.columns]
        print(f"⚠️ 警告: 缺少进行分析所需的列: {missing_cols}。无法进行优化分析。")
        print("   请确保您已对以下序列列进行了打分：", [trained_seq_col, gen_seq_col])
        return df, 0  # 返回原始DataFrame和0计数

    # 3. 定义布尔条件
    condition = (df[score_col_trained] > df[score_col_gen]) & \
                (df[score_col_gen] > df[original_score_col])

    # 4. 根据条件创建新列 '是否优化成功'
    df['是否优化成功'] = np.where(condition, '是', '否')

    # 5. 统计成功的数量
    improvement_count = df['是否优化成功'].value_counts().get('是', 0)

    print(f"分析完成。新列 '是否优化成功' 已添加。")
    print(f"📈 优化成功 (微调分 > 生成分 > 原始分) 的序列数量: {improvement_count}")

    return df, improvement_count


def main():
    parser = argparse.ArgumentParser(
        description="使用预训练模型对新的蛋白质序列进行打分，并进行优化分析。",
        formatter_class=argparse.RawTextHelpFormatter  # 保持帮助信息格式
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="./predict_result_3.csv",
        help="包含待打分蛋白质序列的CSV文件路径。"
    )
    parser.add_argument(
        "--sequence_columns",
        type=str,
        nargs='+',  # 接受一个或多个值
        required=True,
        help="需要进行打分的列名列表，用空格分隔。\n"
             "例如: --sequence_columns mutated_sequence predicted_sequence_trained predicted_sequence"
    )
    parser.add_argument(
        "--model_input_path",
        type=str,
        default="scoring_model.pth",
        help="预训练打分模型权重的文件路径。"
    )
    parser.add_argument(
        "--output_csv_path",
        type=str,
        default="predicted_scores_with_analysis_3.csv",
        help="保存打分和分析结果的CSV文件路径。"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="esm2_t12_35M_UR50D",
        help="用于生成嵌入向量的ESM-2模型名称（必须与训练时使用的模型一致）。"
    )
    parser.add_argument(
        "--repr_layer",
        type=int,
        default=12,
        help="从ESM模型的哪一层提取表征（必须与训练时一致）。"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="生成嵌入向量时的批次大小。"
    )

    args = parser.parse_args()

    # 1. 初始化和加载
    device = check_device()
    df = load_prediction_data(args.csv_path)
    scoring_model = load_scoring_model(args.model_input_path)
    esm_model, alphabet, batch_converter = load_esm2_model(args.model_name, device)

    # 2. 循环为每个指定的序列列生成嵌入向量并预测分数
    for col in args.sequence_columns:
        print(f"\n--- 正在处理列: '{col}' ---")

        if col not in df.columns:
            print(f"⚠️ 警告: 指定的列 '{col}' 在CSV文件中不存在，已跳过。")
            continue

        sequences = df[col].dropna().tolist()
        if not sequences:
            print(f"ℹ️ 列 '{col}' 中没有有效序列，跳过。")
            continue

        embeddings = get_protein_embeddings_batch(
            sequences, esm_model, alphabet, batch_converter, device, args.repr_layer, args.batch_size
        )

        predictions = scoring_model.predict(embeddings)
        pred_series = pd.Series(predictions, index=df[col].dropna().index)

        output_col_name = f"predicted_score_{col}"
        df[output_col_name] = pred_series
        df[output_col_name] = df[output_col_name].round(9)
        print(f"✅ 已生成预测分数并添加到新列 '{output_col_name}'。")

    # 3. 执行分析和标注
    # 根据您的CSV文件，列名是固定的
    df_analyzed, count = analyze_and_annotate(
        df=df,
        trained_seq_col="predicted_sequence_trained",
        gen_seq_col="predicted_sequence",
        original_score_col="DMS_score"
    )

    # 4. 保存最终结果
    try:
        # 获取输出路径的目录部分
        output_dir = os.path.dirname(args.output_csv_path)

        # 只有当目录名不为空时，才尝试创建目录
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"已创建目录: {output_dir}")
        df_analyzed.to_csv(args.output_csv_path, index=False, float_format='%.9f')
        print(f"\n🎉 所有打分和分析完成！结果已保存到: {args.output_csv_path}")
        print("\n最终结果预览:")
        # 预览时显示关键列
        preview_cols = list(args.sequence_columns) + \
                       [f"predicted_score_{col}" for col in args.sequence_columns] + \
                       ['DMS_score', '是否优化成功']
        # 确保预览的列都存在
        preview_cols = [c for c in preview_cols if c in df_analyzed.columns]
        print(df_analyzed[preview_cols].head().to_string())
    except Exception as e:
        print(f"❌ 保存结果CSV时出错: {e}")


if __name__ == "__main__":
    main()