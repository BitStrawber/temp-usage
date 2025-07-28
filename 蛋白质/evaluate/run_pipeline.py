import pandas as pd
import numpy as np
import os
import argparse
import joblib
from tqdm import tqdm
import torch
from sklearn.linear_model import RidgeCV
from transformers import AutoTokenizer, AutoModelForMaskedLM
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# 辅助函数 (保持不变)
# ==============================================================================
# ... (check_device, load_esm2_model, get_protein_embeddings_batch 函数无需修改)
def check_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ 使用设备: {device}")
    return device

def load_esm2_model(model_name, device):
    print(f"正在从 '{model_name}' 加 लोड ESM-2 模型...")
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("✅ ESM-2 模型加载成功。")
    return model, tokenizer

def get_protein_embeddings_batch(sequences, model, tokenizer, device, repr_layer, batch_size):
    embeddings = []
    model.eval()
    for i in tqdm(range(0, len(sequences), batch_size), desc="生成嵌入向量"):
        batch_seqs = sequences[i:i + batch_size]
        inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[repr_layer]
        for j in range(len(batch_seqs)):
            seq_len = inputs['attention_mask'][j].sum().item()
            seq_embedding = hidden_states[j, 1:seq_len - 1].mean(dim=0).cpu().numpy()
            embeddings.append(seq_embedding)
    return np.array(embeddings)
# ==============================================================================
# --- 修改: 模型性能可视化函数 ---
# ==============================================================================

def plot_training_performance(y_train, y_pred_train, y_test, y_pred_test, output_path):
    """
    (扩展版) 绘制 2x2 面板，全面评估模型性能，包括残差图。
    """
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    print(f"\n--- 模型性能评估 ---")
    print(f"  训练集 R²: {train_r2:.4f}")
    print(f"  测试集 R²: {test_r2:.4f}")

    plt.style.use('seaborn-v0_8-whitegrid')
    # --- 修改: 创建一个 2x2 的子图布局 ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('Scoring Model Performance Evaluation', fontsize=18)

    # --- 图 1: 训练集预测 vs. 真实 ---
    sns.scatterplot(x=y_train, y=y_pred_train, alpha=0.6, color='blue', ax=axes[0, 0], s=20)
    min_val_train = min(y_train.min(), y_pred_train.min())
    max_val_train = max(y_train.max(), y_pred_train.max())
    axes[0, 0].plot([min_val_train, max_val_train], [min_val_train, max_val_train], 'r--', lw=2)
    axes[0, 0].set_xlabel('True Score (Training Set)', fontsize=12)
    axes[0, 0].set_ylabel('Predicted Score', fontsize=12)
    axes[0, 0].set_title(f'Performance on Training Set (R² = {train_r2:.4f})', fontsize=14)
    axes[0, 0].grid(True)

    # --- 图 2: 测试集预测 vs. 真实 ---
    sns.scatterplot(x=y_test, y=y_pred_test, alpha=0.6, color='green', ax=axes[0, 1], s=20)
    min_val_test = min(y_test.min(), y_pred_test.min())
    max_val_test = max(y_test.max(), y_pred_test.max())
    axes[0, 1].plot([min_val_test, max_val_test], [min_val_test, max_val_test], 'r--', lw=2)
    axes[0, 1].set_xlabel('True Score (Test Set)', fontsize=12)
    axes[0, 1].set_ylabel('Predicted Score', fontsize=12)
    axes[0, 1].set_title(f'Performance on Test Set (R² = {test_r2:.4f})', fontsize=14)
    axes[0, 1].grid(True)

    # --- 图 3 (新增): 训练集残差图 ---
    train_residuals = y_train - y_pred_train
    sns.scatterplot(x=y_pred_train, y=train_residuals, alpha=0.6, color='blue', ax=axes[1, 0], s=20)
    axes[1, 0].axhline(y=0, color='r', linestyle='--')
    axes[1, 0].set_xlabel('Predicted Score (Training Set)', fontsize=12)
    axes[1, 0].set_ylabel('Residuals (True - Predicted)', fontsize=12)
    axes[1, 0].set_title('Residuals on Training Set', fontsize=14)
    axes[1, 0].grid(True)

    # --- 图 4 (新增): 测试集残差图 ---
    test_residuals = y_test - y_pred_test
    sns.scatterplot(x=y_pred_test, y=test_residuals, alpha=0.6, color='green', ax=axes[1, 1], s=20)
    axes[1, 1].axhline(y=0, color='r', linestyle='--')
    axes[1, 1].set_xlabel('Predicted Score (Test Set)', fontsize=12)
    axes[1, 1].set_ylabel('Residuals (True - Predicted)', fontsize=12)
    axes[1, 1].set_title('Residuals on Test Set', fontsize=14)
    axes[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(output_path, dpi=300)
    print(f"\n✅ 模型性能图表 (4-panel) 已保存至: {output_path}")
    plt.close(fig)

# ==============================================================================
# --- 修改: 集成模型训练与评估逻辑 ---
# ==============================================================================
def train_model(data_path: str, model_output_path: str, base_model: str, repr_layer: int, batch_size: int):
    """
    (修改版) 使用ESM-2生成嵌入向量，划分数据，训练回归模型，
    评估其在训练集和测试集上的性能，并生成可视化图表。
    """
    print("=" * 80)
    print(f"开始训练和评估打分模型...")
    print(f" -> 训练数据: {data_path}")
    print(f" -> ESM-2模型: {base_model} (使用第 {repr_layer} 层)")

    device = check_device()
    esm_model, tokenizer = load_esm2_model(base_model, device)
    df_train_full = pd.read_csv(data_path)

    sequences_full = df_train_full['sequence'].tolist()
    labels_full = df_train_full['label'].values
    print(f"共 {len(sequences_full)} 条序列用于训练和测试。")
    embeddings_full = get_protein_embeddings_batch(
        sequences_full, esm_model, tokenizer, device, repr_layer, batch_size
    )

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings_full, labels_full, test_size=0.2, random_state=42
    )
    print(f"\n数据已划分为: {len(X_train)} (训练) / {len(X_test)} (测试) 样本。")

    print("开始在训练集上训练回归模型...")
    scoring_model = RidgeCV(alphas=np.logspace(-3, 3, 10), cv=5)
    scoring_model.fit(X_train, y_train)
    print(f"回归模型训练完成。交叉验证选出的最佳alpha值为: {scoring_model.alpha_:.4f}")

    # --- 在训练集和测试集上进行预测以供评估 ---
    y_pred_train = scoring_model.predict(X_train)
    y_pred_test = scoring_model.predict(X_test)

    # --- 调用双图可视化函数 ---
    performance_plot_path = model_output_path.replace(".joblib", "_performance.png")
    plot_training_performance(y_train, y_pred_train, y_test, y_pred_test, performance_plot_path)

    joblib.dump(scoring_model, model_output_path)
    print(f"\n✅ 模型已成功保存至: {model_output_path}")
    print("=" * 80)

# ... (score_data, extract_dms_id_from_path, find_and_combine_data, 和 main 函数保持不变)
# ... (这些函数的内容与您上一版代码完全相同)
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
        'predicted_sequence_trained',
        'predicted_sequence'
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
    score_trained = 'predicted_score_predicted_sequence_trained'
    score_gen = 'predicted_score_predicted_sequence'
    original_score_col = 'DMS_score'
    required_score_cols = [score_trained, score_gen, original_score_col]

    improvement_count = 0
    if all(col in df.columns for col in required_score_cols):
        condition = (df[score_trained] > df[score_gen]) & \
                    (df[score_gen] > df[original_score_col])
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

def extract_dms_id_from_path(path: str) -> str:
    """从文件路径中提取DMS_id。"""
    try:
        return os.path.basename(os.path.dirname(path))
    except Exception:
        raise ValueError(f"无法从路径 {path} 中提取DMS_id。")

def find_and_combine_data(target_dms_id: str, dms_data_dir: str, metadata_file_path: str = None) -> pd.DataFrame:
    """
    (净化版) 根据是否提供元数据文件路径，选择数据扩展策略。
    移除了大部分调试信息，保留核心流程日志。
    """
    combined_df_list = []

    # --- 模式 1: 提供了元数据文件，进行大规模扩展 ---
    if metadata_file_path:
        print("\n--- 模式: 使用元数据进行数据集扩展 ---")
        if not os.path.exists(metadata_file_path):
             raise FileNotFoundError(f"指定的元数据文件不存在: {metadata_file_path}")

        try:
            metadata_df = pd.read_csv(metadata_file_path)

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
            print(f"\n在元数据中找到 {len(matching_proteins)} 个匹配的数据集:")

            for _, row in matching_proteins.iterrows():
                dms_id = row['DMS_id']
                dms_filename = row['DMS_filename']
                file_path = os.path.join(dms_data_dir, dms_id, dms_filename)
                print(f" -> 正在查找数据: {file_path}")

                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    if 'mutated_sequence' in df.columns and 'DMS_score' in df.columns:
                        combined_df_list.append(df[['mutated_sequence', 'DMS_score']])
                    else:
                        print(f"    [警告] 文件 {dms_filename} 缺少必需列。")
                else:
                    print(f"    [警告] 文件不存在，已跳过。")
        except Exception as e:
            # 保留顶层的错误捕获，以便在筛选或读取过程中出错时仍能提供信息
            print(f"❌ 在处理元数据时发生错误: {e}")
            raise e

    # --- 模式 2: 未提供元数据，只使用自身数据 ---
    else:
        print(f"\n--- 模式: 仅使用目标 '{target_dms_id}' 自身的数据进行训练 ---")
        self_data_filename = f"{target_dms_id}.csv"
        file_path = os.path.join(dms_data_dir, target_dms_id, self_data_filename)
        print(f" -> 正在查找自身数据: {file_path}")

        file_found = False
        if os.path.exists(file_path):
            file_found = True
        else:
            alt_filename = "DMS_substitutions.csv"
            file_path = os.path.join(dms_data_dir, target_dms_id, alt_filename)
            print(f" -> 自身数据文件未找到，尝试备用名称: {file_path}")
            if os.path.exists(file_path):
                file_found = True

        if file_found:
            try:
                df = pd.read_csv(file_path)
                if 'mutated_sequence' in df.columns and 'DMS_score' in df.columns:
                    combined_df_list.append(df[['mutated_sequence', 'DMS_score']])
                else:
                    print(f"    [警告] 文件缺少必需列。")
            except Exception as e:
                print(f"    [错误] 处理文件 {file_path} 时出错: {e}。")
        else:
            print(f"    [警告] 自身数据文件和备用文件均不存在，已跳过。")


    if not combined_df_list:
        raise RuntimeError("未能加载任何有效数据来构建训练集。")

    combined_df = pd.concat(combined_df_list, ignore_index=True)
    print(f"\n数据集合并完成，共 {len(combined_df)} 条记录用于训练。")
    return combined_df

def main():
    parser = argparse.ArgumentParser(description="使用灵活的数据源自动化蛋白质打分流程。")
    parser.add_argument("--target_data_path", type=str, required=True, help="待打分的目标文件路径。")
    parser.add_argument("--metadata_file", type=str, default=None, help="(可选) 元数据CSV文件路径。如果提供，将进行数据集扩展。")
    parser.add_argument("--dms_data_dir", type=str, required=True, help="包含所有DMS数据子目录的根目录。")
    parser.add_argument("--output_base_dir", type=str, required=True, help="保存最终结果的基础目录。")
    parser.add_argument("--base_model", type=str, default="../esm2_model_local", help="ESM-2模型路径。")
    parser.add_argument("--repr_layer", type=int, default=12, help="ESM-2提取表征的层。")
    parser.add_argument("--batch_size", type=int, default=16, help="批处理大小。")
    args = parser.parse_args()

    target_dms_id = extract_dms_id_from_path(args.target_data_path)
    final_output_dir = os.path.join(args.output_base_dir, target_dms_id)
    os.makedirs(final_output_dir, exist_ok=True)

    print(f"开始处理目标蛋白质: {target_dms_id}")
    print(f"结果将保存至: {final_output_dir}")

    try:
        # 直接将文件路径传递给函数，让函数内部处理 None 的情况
        combined_training_data = find_and_combine_data(
            target_dms_id=target_dms_id,
            dms_data_dir=args.dms_data_dir,
            metadata_file_path=args.metadata_file
        )
    except Exception as e:
        print(f"[错误] 数据准备阶段失败: {e}")
        return

    # 永久保存模型的逻辑
    training_data_path = os.path.join(final_output_dir, "temp_training_data.csv")
    training_df = combined_training_data[['mutated_sequence', 'DMS_score']].rename(
        columns={'mutated_sequence': 'sequence', 'DMS_score': 'label'}
    )
    training_df.to_csv(training_data_path, index=False)

    model_output_path = os.path.join(final_output_dir, "scoring_model.joblib")

    train_model(
        data_path=training_data_path,
        model_output_path=model_output_path,
        base_model=args.base_model,
        repr_layer=args.repr_layer,
        batch_size=args.batch_size
    )

    final_output_file = os.path.join(final_output_dir, "prediction_results_scored.csv")

    score_data(
        model_path=model_output_path,
        base_model=args.base_model,
        repr_layer=args.repr_layer,
        batch_size=args.batch_size,
        data_to_score_path=args.target_data_path,
        output_file_path=final_output_file
    )

    try:
        os.remove(training_data_path)
        print(f"已清理临时训练数据文件: {training_data_path}")
    except OSError as e:
        print(f"清理临时文件时出错: {e}")

    print("\n流程处理成功完成！")
    print(f"最终打分结果已保存在: {final_output_file}")

if __name__ == "__main__":
    main()

    python run_pipeline.py \
    --target_data_path "../蛋白质数据/A0A192B1T2_9HIV1_Haddox_2018/prediction_results.csv" \
    --dms_data_dir "../蛋白质数据/" \
    --output_base_dir "../蛋白质数据/" \
    --metadata_file "./DMS_substitutions.csv"

