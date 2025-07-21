# -*- coding: utf-8 -*-

import pandas as pd
import os
import argparse
import re
import tempfile
from tqdm import tqdm

# 导入您原始脚本中使用的核心库
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    pipeline,
)


# ==============================================================================
# 集成模型训练逻辑 (源自您的 train.py)
# ==============================================================================

def train_model(data_path: str, model_output_dir: str, base_model: str, epochs: int, batch_size: int):
    """
    在蛋白质序列上训练/微调一个语言模型。
    此函数会自动检测并使用CUDA（如果可用），否则回退到CPU。

    Args:
        data_path (str): 合并后的训练数据CSV文件路径。
        model_output_dir (str): 用于保存微调后模型的目录。
        base_model (str): Hugging Face上的基础模型名称 (例如, 'facebook/esm2_t33_650M_UR50D')。
        epochs (int): 训练的轮数。
        batch_size (int): 每个设备的训练批次大小。
    """
    print("=" * 80)
    print(f"开始模型微调...")
    print(f" -> 训练数据: {data_path}")
    print(f" -> 基础模型: {base_model}")
    print(f" -> 训练配置: {epochs} 个 epochs, 批大小为 {batch_size}")
    print(f" -> 模型将保存至: {model_output_dir}")

    # --- 新增：设备检测逻辑 ---
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    print(f"检测到可用设备: {device.upper()}")
    if not use_cuda:
        print("[警告] 未检测到CUDA GPU，将使用CPU进行训练。这可能会非常缓慢。")
    # --- 结束新增部分 ---

    # 1. 加载分词器 (tokenizer) 和基础模型
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForMaskedLM.from_pretrained(base_model)

    # 2. 加载并处理数据集
    raw_datasets = load_dataset('csv', data_files={'train': data_path})

    def tokenize_function(examples):
        return tokenizer(examples["sequence"], truncation=True, padding="max_length", max_length=512)

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=["sequence", "label"]
    )

    # 3. 设置数据整理器 (Data Collator)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=0.15
    )

    # 4. 设置训练参数
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
        logging_steps=500,
        # --- 修改：根据检测结果明确告知Trainer是否使用CUDA ---
        no_cuda=not use_cuda
    )

    # 5. 初始化并运行训练器 (Trainer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        data_collator=data_collator,
    )

    print("开始训练...")
    trainer.train()
    trainer.save_model(model_output_dir)
    print("模型微调完成。")
    print("=" * 80)


# ==============================================================================
# 集成数据打分逻辑 (源自您的 predict.py)
# ==============================================================================

def score_data(model_path: str, base_model_path: str, data_to_score_path: str, output_file_path: str):
    """
    使用微调模型和基础模型对含有mask的序列进行打分（预测）。

    Args:
        model_path (str): 微调后模型的目录路径。
        base_model_path (str): 基础模型的路径或名称。
        data_to_score_path (str): 包含待打分序列的CSV文件路径。
        output_file_path (str): 保存最终预测结果的文件路径。
    """
    print("=" * 80)
    print(f"开始数据打分...")
    print(f" -> 微调模型: {model_path}")
    print(f" -> 基础模型: {base_model_path}")
    print(f" -> 待打分数据: {data_to_score_path}")

    # 自动检测是否有可用的GPU
    device = 0 if torch.cuda.is_available() else -1
    print(f"使用设备: {'cuda:0' if device == 0 else 'cpu'}")

    # 1. 加载微调模型和基础模型的pipeline
    pipe_finetuned = pipeline('fill-mask', model=model_path, device=device)
    pipe_base = pipeline('fill-mask', model=base_model_path, device=device)

    # 2. 加载待打分的数据
    df = pd.read_csv(data_to_score_path)

    results = []

    # 3. 遍历每一行数据进行预测
    for index, row in tqdm(df.iterrows(), total=len(df), desc="正在打分序列"):
        masked_sequence = row['masked_sequence']

        # 使用微调模型预测
        outputs_finetuned = pipe_finetuned(masked_sequence)
        temp_seq_finetuned = list(masked_sequence)
        # 将mask位置替换为预测概率最高的氨基酸
        for output in outputs_finetuned:
            token = output['token_str']
            temp_seq_finetuned[output['start']] = token
        predicted_sequence_finetuned = "".join(temp_seq_finetuned)

        # 使用基础模型预测
        outputs_base = pipe_base(masked_sequence)
        temp_seq_base = list(masked_sequence)
        for output in outputs_base:
            token = output['token_str']
            temp_seq_base[output['start']] = token
        predicted_sequence_base = "".join(temp_seq_base)

        # 将预测结果添加到原始行数据中
        result_row = row.to_dict()
        result_row['predicted_sequence_finetuned'] = predicted_sequence_finetuned
        result_row['predicted_sequence_base'] = predicted_sequence_base
        results.append(result_row)

    # 4. 保存包含预测结果的新CSV文件
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file_path, index=False)

    print(f"数据打分完成，结果已保存至 {output_file_path}")
    print("=" * 80)


# ==============================================================================
# 自动化流程的核心逻辑
# ==============================================================================

def extract_dms_id_from_path(path: str) -> str:
    """
    从文件路径中提取DMS_id。
    例如: '.../(activity)A0A247D711_LISMN_Stadelmann_2021/file.csv'
    返回: 'A0A247D711_LISMN_Stadelmann_2021'
    """
    # 正则表达式匹配括号内容和紧随其后的蛋白质ID
    match = re.search(r'\((.*?)\)([^/]+)', path)
    if match:
        # group(2) 是我们需要的蛋白质ID部分
        return match.group(2).strip()
    else:
        # 如果正则不匹配，尝试使用目录名作为备选方案
        try:
            return os.path.basename(os.path.dirname(path))
        except Exception:
            raise ValueError(f"无法从路径 {path} 中提取DMS_id。")


def find_and_combine_data(target_dms_id: str, metadata_df: pd.DataFrame, dms_data_dir: str) -> pd.DataFrame:
    """
    根据元数据查找并合并用于训练的数据集。

    Args:
        target_dms_id (str): 目标蛋白质的DMS_id。
        metadata_df (pd.DataFrame): 包含所有实验元数据的DataFrame。
        dms_data_dir (str): 存储所有DMS数据集CSV文件的目录。

    Returns:
        pd.DataFrame: 一个包含所有匹配数据集的合并后的DataFrame。
    """
    # 1. 查找目标蛋白质的元数据
    target_protein_metadata = metadata_df[metadata_df['DMS_id'] == target_dms_id]
    if target_protein_metadata.empty:
        raise ValueError(f"DMS_id '{target_dms_id}' 在元数据文件中未找到。")

    target_assay = target_protein_metadata['selection_assay'].iloc[0]
    target_title = target_protein_metadata['title'].iloc[0]

    print(f"\n目标蛋白质 '{target_dms_id}' 的属性:")
    print(f" -> selection_assay: {target_assay}")
    print(f" -> title: {target_title}")

    # 2. 查找所有具有相同 selection_assay 和 title 的蛋白质
    matching_proteins = metadata_df[
        (metadata_df['selection_assay'] == target_assay) &
        (metadata_df['title'] == target_title)
        ]

    print(f"\n找到 {len(matching_proteins)} 个匹配的数据集用于数据扩展:")

    # 3. 加载并合并这些数据集
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
            # 确保文件包含必需的列
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
    parser.add_argument("--target_data_path", type=str, required=True,
                        help="待打分的目标文件路径，例如 '.../(activity)PROTEIN_ID/file.csv'。")
    parser.add_argument("--metadata_file", type=str, required=True, help="元数据CSV文件路径 ('工作簿1.csv')。")
    parser.add_argument("--dms_data_dir", type=str, required=True, help="包含所有DMS原始数据的CSV文件的目录。")
    parser.add_argument("--output_base_dir", type=str, required=True, help="保存最终结果的基础目录。")

    # 添加模型训练相关的参数
    parser.add_argument("--base_model", type=str, default="facebook/esm2_t33_650M_UR50D", help="用于微调的基础模型。")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数。")
    parser.add_argument("--batch_size", type=int, default=4, help="训练批处理大小。")

    args = parser.parse_args()

    # --- 步骤 1: 初始化并从路径中提取ID ---
    target_dms_id = extract_dms_id_from_path(args.target_data_path)

    # 创建一个以 "(DMS_id_extended)" 命名的子文件夹来存放本次运行的结果
    result_folder_name = f"({os.path.basename(os.path.dirname(args.target_data_path))}_extended)"
    final_output_dir = os.path.join(args.output_base_dir, result_folder_name)
    os.makedirs(final_output_dir, exist_ok=True)

    print(f"开始处理目标蛋白质: {target_dms_id}")
    print(f"结果将保存至: {final_output_dir}")

    # --- 步骤 2: 数据扩展 ---
    try:
        metadata_df = pd.read_csv(args.metadata_file)
        combined_training_data = find_and_combine_data(target_dms_id, metadata_df, args.dms_data_dir)
    except FileNotFoundError:
        print(f"[错误] 元数据文件未找到: {args.metadata_file}")
        return
    except Exception as e:
        print(f"[错误] 数据准备阶段失败: {e}")
        return

    # --- 步骤 3 & 4: 训练和打分 ---
    # 使用临时目录来管理中间文件，脚本执行完毕后会自动删除
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_data_path = os.path.join(temp_dir, "training_data_for_model.csv")

        # 准备训练数据：重命名列以匹配训练函数的要求
        training_df = combined_training_data[['mutated_sequence', 'DMS_score']].rename(
            columns={'mutated_sequence': 'sequence', 'DMS_score': 'label'}
        )
        training_df.to_csv(temp_data_path, index=False)

        temp_model_dir = os.path.join(temp_dir, "fine_tuned_model")

        # 调用训练函数
        train_model(
            data_path=temp_data_path,
            model_output_dir=temp_model_dir,
            base_model=args.base_model,
            epochs=args.epochs,
            batch_size=args.batch_size
        )

        # 调用打分函数
        final_output_file = os.path.join(final_output_dir, "prediction_results_scored.csv")

        score_data(
            model_path=temp_model_dir,
            base_model_path=args.base_model,
            data_to_score_path=args.target_data_path,
            output_file_path=final_output_file
        )

    print("\n流程处理成功完成！")
    print(f"最终打分结果已保存在: {final_output_file}")


if __name__ == "__main__":
    # 命令行运行示例:
    # python run_pipeline.py \
    #   --target_data_path "autodl-tmp/总结果/(activity)A0A247D711_LISMN_Stadelmann_2021/prediction_results_from_low_scores.csv" \
    #   --metadata_file "path/to/你的/工作簿1.csv" \
    #   --dms_data_dir "path/to/your/dms_data_directory/" \
    #   --output_base_dir "autodl-tmp/总结果/"
    main()