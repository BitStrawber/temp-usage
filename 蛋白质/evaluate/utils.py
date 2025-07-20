# utils.py

# ----------------- 导入所需库 -----------------
import torch
import esm
import numpy as np
from tqdm import tqdm
import os
import warnings

# 忽略不必要的警告信息
warnings.filterwarnings('ignore')

# ----------------- 函数定义 -----------------

def check_device():
    """检查可用的计算设备 (GPU或CPU) 并返回。"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用的计算设备: {device}")
    return device


def load_esm2_model(model_name="esm2_t12_35M_UR50D", device='cpu'):
    """
    加载指定名称的ESM-2模型、词典和批次转换器。
    利用 torch.hub 的本地缓存机制，避免重复下载。
    """
    print(f"正在加载 {model_name} 模型...")
    try:
        model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        batch_converter = alphabet.get_batch_converter()
        model.eval()
        model.to(device)
        print("✅ 模型加载成功。")
        return model, alphabet, batch_converter
    except Exception as e:
        print(f"❌ 错误: 加载ESM模型 '{model_name}' 失败。原因: {e}")
        print("\n🚨 请确保已正确安装最新版 ESM 库: `pip install fair-esm`")
        print(f"   ESM 会自动尝试下载模型。如果失败，请检查网络连接或手动将模型文件 '{model_name}.pt' 放置于缓存目录。")
        exit(1)


def get_protein_embeddings_batch(sequences, model, alphabet, batch_converter, device, layer=12, batch_size=32):
    """
    使用ESM-2模型，分批次地为蛋白质序列生成嵌入向量(embeddings)。
    """
    if not sequences or all(s is None or s == '' for s in sequences):
        print("ℹ️ 提示: 序列列表为空，将返回空嵌入向量数组。")
        return np.array([])

    print(f"正在为 {len(sequences)} 条序列生成嵌入向量...")
    all_embeddings = []

    for i in tqdm(range(0, len(sequences), batch_size), desc="正在分批处理序列"):
        batch_sequences = sequences[i:i + batch_size]
        data = [(f"protein_{j}", seq) for j, seq in enumerate(batch_sequences)]

        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[layer], return_contacts=False)

        token_representations = results["representations"][layer]
        batch_embeddings = []
        for j, tokens_len in enumerate(batch_lens):
            # 对每个序列的表征进行平均池化，忽略起始(BOS)和结束(EOS)标记
            seq_repr = token_representations[j, 1:tokens_len - 1].mean(0)
            batch_embeddings.append(seq_repr.cpu().numpy())

        all_embeddings.extend(batch_embeddings)

    return np.array(all_embeddings)