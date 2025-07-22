# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import argparse
import re
import tempfile
from tqdm import tqdm
import joblib

# 导入核心库
import torch
from sklearn.linear_model import RidgeCV
from transformers import AutoTokenizer, AutoModelForMaskedLM


# ------------------------------------------------------------------------------
# 假设您有一个 utils.py 文件包含以下函数。
# 如果没有，请将这些辅助函数放在此脚本的顶部或一个名为 utils.py 的文件中。
# ------------------------------------------------------------------------------

def check_device():
    """检查可用的计算设备（优先使用GPU）。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ 使用设备: {device}")
    return device


def load_esm2_model(model_name, device):
    """加载ESM-2模型和分词器。"""
    print(f"正在从 '{model_name}' 加载 ESM-2 模型...")
    # ESM模型在加载时可能会显示关于position_embeddings的警告，这是正常的，可以忽略
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("✅ ESM-2 模型加载成功。")
    # --- 修改：只返回两个值 ---
    return model, tokenizer


def get_protein_embeddings_batch(sequences, model, tokenizer, device, repr_layer, batch_size):
    """批量生成蛋白质序列的嵌入向量。"""
    embeddings = []

    # 按照批次处理序列以节省内存
    for i in tqdm(range(0, len(sequences), batch_size), desc="生成嵌入向量"):
        batch_seqs = sequences[i:i + batch_size]

        # 使用tokenizer进行编码
        inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # 安全检查：确保请求的层存在
        num_layers = len(outputs.hidden_states)
        if not (0 <= repr_layer < num_layers):
            raise ValueError(
                f"错误: 请求的层 '{repr_layer}' 超出模型范围。 "
                f"此模型共有 {num_layers} 个层 (有效索引为 0 到 {num_layers - 1})。"
            )

        hidden_states = outputs.hidden_states[repr_layer]

        # 对每个序列的嵌入向量取平均值（去除padding的影响）
        for j, seq in enumerate(batch_seqs):
            seq_len = inputs['attention_mask'][j].sum().item()
            seq_embedding = hidden_states[j, 1:seq_len - 1].mean(dim=0).cpu().numpy()
            embeddings.append(seq_embedding)

    return np.array(embeddings)


# ==============================================================================
# 集成模型训练逻辑 (嵌入向量 + 回归模型)
# ==============================================================================

def train_model(data_path: str, model_output_path: str, base_model: str, repr_layer: int, batch_size: int):
    """
    使用ESM-2生成嵌入向量，并训练一个回归模型来预测分数。
    """
    print("=" * 80)
    print(f"开始训练打分模型 (嵌入+回归)...")
    print(f" -> 训练数据: {data_path}")
    print(f" -> ESM-2模型: {base_model} (使用第 {repr_layer} 层)")
    print(f" -> 模型将保存至: {model_output_path}")

    # 1. 初始化设备和ESM-2模型
    device = check_device()
    # --- 修改：接收两个返回值 ---
    esm_model, tokenizer = load_esm2_model(base_model, device)

    # 2. 加载训练数据
    df_train = pd.read_csv(data_path)
    sequences = df_train['sequence'].tolist()
    labels = df_train['label'].values
    print(f"共 {len(sequences)} 条序列用于训练。")

    # 3. 生成蛋白质嵌入向量
    # --- 修改：移除 batch_converter 参数 ---
    embeddings = get_protein_embeddings_batch(
        sequences, esm_model, tokenizer, device, repr_layer, batch_size
    )

    # 4. 训练回归模型
    print("嵌入向量生成完毕，开始训练回归模型...")
    scoring_model = RidgeCV(alphas=np.logspace(-3, 3, 10), cv=5)
    scoring_model.fit(embeddings, labels)
    print(f"回归模型训练完成。交叉验证选出的最佳alpha值为: {scoring_model.alpha_:.4f}")

    # 5. 保存训练好的回归模型
    joblib.dump(scoring_model, model_output_path)
    print(f"✅ 模型已成功保存至: {model_output_path}")
    print("=" * 80)

# ==============================================================================
# 集成数据打分逻辑 (嵌入向量 + 回归模型)
# ==============================================================================

def score_data(model_path: str, base_model: str, repr_layer: int, batch_size: int, data_to_score_path: str,
               output_file_path: str):
    """
    使用训练好的回归模型，为新的蛋白质序列进行打分。
    """
    print("=" * 80)
    print(f"开始使用回归模型为多个序列列进行打分...")
    print(f" -> 打分模型: {model_path}")
    print(f" -> ESM-2模型: {base_model} (使用第 {repr_layer} 层)")
    print(f" -> 待打分数据: {data_to_score_path}")

    # 1. 初始化和加载
    device = check_device()
    df = pd.read_csv(data_to_score_path)
    scoring_model = joblib.load(model_path)
    # --- 修改：接收两个返回值 (此行已正确，但为保持一致性而展示) ---
    esm_model, tokenizer = load_esm2_model(base_model, device)

    # 2. 定义需要被打分的序列列
    sequence_columns_to_score = [
        'mutated_sequence',
        'predicted_sequence_finetuned',
        'predicted_sequence_base'
    ]

    for col in sequence_columns_to_score:
        print(f"\n--- 正在处理列: '{col}' ---")

        if col not in df.columns:
            print(f"⚠️ 警告: 指定的列 '{col}' 在CSV文件中不存在，已跳过。")
            continue

        sequences_to_score = df[col].dropna().tolist()
        if not sequences_to_score:
            print(f"ℹ️ 列 '{col}' 中没有有效序列，跳过。")
            continue

        # --- 修改：移除 batch_converter 参数 ---
        embeddings = get_protein_embeddings_batch(
            sequences_to_score, esm_model, tokenizer, device, repr_layer, batch_size
        )

        print("嵌入向量生成完毕，开始预测分数...")
        predictions = scoring_model.predict(embeddings)

        pred_series = pd.Series(predictions, index=df[col].dropna().index)

        output_col_name = f"predicted_score_{col}"
        df[output_col_name] = pred_series
        df[output_col_name] = df[output_col_name].round(9)
        print(f"✅ 已生成预测分数并添加到新列 '{output_col_name}'。")

    # ... 后续分析和保存部分无需修改 ...
    print("\n--- 正在进行优化分析和标注 ---")
    score_finetuned = 'predicted_score_predicted_sequence_finetuned'
    score_base = 'predicted_score_predicted_sequence_base'
    original_score_col = 'DMS_score'
    required_score_cols = [score_finetuned, score_base, original_score_col]

    improvement_count = 0
    if all(col in df.columns for col in required_score_cols):
        condition = (df[score_finetuned] > df[score_base]) & \
                    (df[score_base] > df[original_score_col])
        df['是否优化成功'] = np.where(condition, '是', '否')
        improvement_count = df['是否优化成功'].value_counts().get('是', 0)
        print(f"分析完成。新列 '是否优化成功' 已添加。")
        print(f"📈 优化成功 (微调分 > 基础分 > 原始DMS分) 的序列数量: {improvement_count}")
    else:
        missing_cols = [col for col in required_score_cols if col not in df.columns]
        print(f"⚠️ 警告: 缺少进行分析所需的列: {missing_cols}。已跳过优化分析。")
        df['是否优化成功'] = '分析跳过'

    summary_data = {col: '' for col in df.columns}
    summary_data[df.columns[0]] = "--- 统计结果 ---"
    if '是否优化成功' in df.columns and improvement_count > 0:
        summary_data['是否优化成功'] = f"总计优化成功: {improvement_count}"

    summary_df = pd.DataFrame([summary_data])
    results_df = pd.concat([df, summary_df], ignore_index=True)

    results_df.to_csv(output_file_path, index=False, float_format='%.9f')
    print(f"\n✅ 打分和分析完成！结果已保存到: {output_file_path}")
    print("=" * 80)

# ==============================================================================
# 自动化流程的核心逻辑
# ==============================================================================

def extract_dms_id_from_path(path: str) -> str:
    """从文件路径中提取DMS_id。"""
    match = re.search(r'\((.*?)\)([^/]+)', path)
    if match:
        return match.group(2).strip()
    else:
        try:
            return os.path.basename(os.path.dirname(path))
        except Exception:
            raise ValueError(f"无法从路径 {path} 中提取DMS_id。")


def find_and_combine_data(target_dms_id: str, metadata_df: pd.DataFrame, dms_data_dir: str) -> pd.DataFrame:
    """根据元数据查找并合并数据集用于训练。"""
    target_protein_metadata = metadata_df[metadata_df['DMS_id'] == target_dms_id]
    if target_protein_metadata.empty:
        raise ValueError(f"DMS_id '{target_dms_id}' 在元数据文件中未找到。")

    target_assay = target_protein_metadata['selection_assay'].iloc[0]
    target_title = target_protein_metadata['title'].iloc[0]

    print(f"\n目标蛋白质 '{target_dms_id}' 的属性:")
    print(f" -> selection_assay: {target_assay}")
    print(f" -> title: {target_title}")

    matching_proteins = metadata_df[
        (metadata_df['selection_assay'] == target_assay) &
        (metadata_df['title'] == target_title)
        ]

    print(f"\n找到 {len(matching_proteins)} 个匹配的数据集用于数据扩展:")

    combined_df_list = []
    for _, row in matching_proteins.iterrows():
        dms_file = row['DMS_filename']
        file_path = os.path.join(dms_data_dir, dms_file)
        print(f" -> 加载数据: {dms_file}")

        if not os.path.exists(file_path):
            print(f"    [警告] 文件不存在，已跳过: {file_path}")
            continue
        try:
            df = pd.read_csv(file_path)
            if 'mutated_sequence' in df.columns and 'DMS_score' in df.columns:
                combined_df_list.append(df[['mutated_sequence', 'DMS_score']])
            else:
                print(f"    [警告] 文件 {dms_file} 缺少 'mutated_sequence' 或 'DMS_score' 列，已跳过。")
        except Exception as e:
            print(f"    [错误] 处理文件 {file_path} 时出错: {e}。已跳过。")

    if not combined_df_list:
        raise RuntimeError("未能加载任何数据来构建扩展训练集。")

    combined_df = pd.concat(combined_df_list, ignore_index=True)
    print(f"\n数据集合并完成，共 {len(combined_df)} 条记录用于训练。")
    return combined_df


def main():
    parser = argparse.ArgumentParser(description="使用扩展数据自动化蛋白质打分流程。")
    parser.add_argument("--target_data_path", type=str, required=True, help="待打分的目标文件路径。")
    parser.add_argument("--metadata_file", type=str, required=True, help="元数据CSV文件路径 ('工作簿1.csv')。")
    parser.add_argument("--dms_data_dir", type=str, required=True, help="包含所有DMS原始数据的CSV文件的目录。")
    parser.add_argument("--output_base_dir", type=str, required=True, help="保存最终结果的基础目录。")

    parser.add_argument("--base_model", type=str, default="../../../esm2_model_local",
                        help="用于生成嵌入向量的ESM-2模型路径。")
    parser.add_argument("--repr_layer", type=int, default=12,
                        help="从ESM-2模型的哪一层提取表征。对于33层模型，33是最后一层。")
    parser.add_argument("--batch_size", type=int, default=16, help="生成嵌入向量时的批处理大小。")
    args = parser.parse_args()

    target_dms_id = extract_dms_id_from_path(args.target_data_path)

    original_folder_name = os.path.basename(os.path.dirname(args.target_data_path))
    result_folder_name = f"{original_folder_name}"
    final_output_dir = os.path.join(args.output_base_dir, result_folder_name)
    os.makedirs(final_output_dir, exist_ok=True)

    print(f"开始处理目标蛋白质: {target_dms_id}")
    print(f"结果将保存至: {final_output_dir}")

    try:
        metadata_df = pd.read_csv(args.metadata_file)
        combined_training_data = find_and_combine_data(target_dms_id, metadata_df, args.dms_data_dir)
    except Exception as e:
        print(f"[错误] 数据准备阶段失败: {e}")
        return

    # 使用临时目录管理中间文件
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_data_path = os.path.join(temp_dir, "training_data.csv")

        training_df = combined_training_data[['mutated_sequence', 'DMS_score']].rename(
            columns={'mutated_sequence': 'sequence', 'DMS_score': 'label'}
        )
        training_df.to_csv(temp_data_path, index=False)

        temp_model_path = os.path.join(temp_dir, "scoring_model.joblib")

        # 训练回归模型
        train_model(
            data_path=temp_data_path,
            model_output_path=temp_model_path,
            base_model=args.base_model,
            repr_layer=args.repr_layer,
            batch_size=args.batch_size
        )

        final_output_file = os.path.join(final_output_dir, "prediction_results_scored.csv")

        # 使用回归模型进行打分
        score_data(
            model_path=temp_model_path,
            base_model=args.base_model,
            repr_layer=args.repr_layer,
            batch_size=args.batch_size,
            data_to_score_path=args.target_data_path,
            output_file_path=final_output_file
        )

    print("\n流程处理成功完成！")
    print(f"最终打分结果已保存在: {final_output_file}")


if __name__ == "__main__":

    main()

    python run_pipeline.py \
    --target_data_path "../../../总结果/(yeast growth)DLG4_HUMAN_Faure_2021/prediction_results_from_low_scores.csv" \
    --metadata_file "../../../DMS_substitutions.csv" \
    --dms_data_dir "../../../蛋白质数据/" \
    --output_base_dir "../../../总结果/"

autodl-tmp/总结果/(yeast growth)DLG4_HUMAN_Faure_2021


