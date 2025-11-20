# train_protonet.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def preprocess_data(file_path):
    """
    加载、清洗并预处理网络流量数据集。
    (最终版本 - 包含了您指定的、非常详尽的列移除列表)
    """
    print("--- 步骤 1: 开始数据预处理 ---")
    
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except Exception as e:
        print(f"❌ 加载文件时发生错误: {e}")
        return None, None, None

    print(f"原始数据集维度: {df.shape}")

    target_column = 'Attack Type'
    if target_column not in df.columns:
        print(f"❌ 错误: 目标列 '{target_column}' 不在CSV文件中。")
        return None, None, None

    # --- 缺失值处理 ---
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=[target_column], inplace=True)
    
    if df.shape[0] == 0:
        print(f"❌ 错误: 数据集变为空。")
        return None, None, None
    print(f"删除标签缺失值后维度: {df.shape}")
    
    # 编码目标标签
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[target_column])
    print(f"标签 '{target_column}' 已被编码为 {len(np.unique(y))} 个类别。")
    print("类别映射:", {i: c for i, c in enumerate(label_encoder.classes_)})
    
    # --- [核心修改] 智能特征工程与选择 ---
    
    # ==================================================================
    # |              >>>>>> 以下是您指定的代码块 <<<<<<                  |
    # ==================================================================

    # --- 1. 定义要移除的列 ---
    columns_to_drop = [
        # 标识符和高基数特征
        'SrcId', 'SrcAddr', 'DstAddr', 'SrcMac', 'DstMac', 'SrcOui', 'DstOui',
        'sIpId', 'dIpId', 
        
        # 之前分析出的所有强相关特征 (根据您的要求添加)
        'IdleTime', 'AckDat', 'DstTCPBase', 'SrcTCPBase', 
        'TcpRtt', 'dHops', 'sHops', 'dTtl', 'Seq', 'Rank', 'Offset', 
        
        
        # 冗余或直接相关的特征
        'RunTime', 'Label', 'Attack Tool', 'attack_cat',
        
        # 原始时间戳
        'StartTime', 'LastTime'
    ]
    
    # 过滤掉数据集中不存在的列名，避免出错
    existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
    
    # 同样要确保目标列也被排除在特征之外
    if target_column in existing_cols_to_drop:
        existing_cols_to_drop.remove(target_column)
    if target_column in df.columns:
         df.drop(columns=[target_column], inplace=True)

    print(f"\n计划移除 {len(existing_cols_to_drop)} 个特征...")
    df.drop(columns=existing_cols_to_drop, inplace=True, errors='ignore')
    print("特征移除完成。")
    
    # ==================================================================
    # |              >>>>>> 代码块集成结束 <<<<<<                       |
    # ==================================================================

    # 2. 将时间戳等 object 类型的列强制转换为数值，无法转换的变为 NaN
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 3. 现在所有列都应该是数值类型了，我们可以用0来填充所有剩余的NaN
    df.fillna(0, inplace=True)
    
    print(f"所有特征已转换为数值类型。")

    # 4. 现在所有列都是数值型，可以直接进行归一化
    scaler = StandardScaler()
    X = scaler.fit_transform(df)

    # 确保最终数组类型正确
    X = X.astype(np.float32)
    
    print(f"\n预处理完成。最终特征矩阵维度: {X.shape}")
    print("----------------------------------\n")
    return X, y, label_encoder



# train_protonet.py (接上文)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler, Dataset

class PrototypicalNetMLP(nn.Module):
    """
    用于表格数据的 MLP 嵌入网络。 (加深版)
    """
    def __init__(self, input_dim, embedding_dim=64):
        super(PrototypicalNetMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 128), # 新增的隐藏层
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, embedding_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class TrafficDataset(Dataset):
    """一个简单的PyTorch数据集包装器"""
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class EpisodicBatchSampler(Sampler):
    """
    Episodic Batch Sampler.
    为每个 episode (batch) 产生 N-way K-shot Q-query 的样本索引。
    """
    def __init__(self, labels, n_episodes, n_way, n_samples):
        super().__init__(labels)
        self.labels = labels
        self.n_episodes = n_episodes
        self.n_way = n_way
        self.n_samples = n_samples # n_support + n_query

        # 按类别对样本索引进行分组
        self.class_indices = {c: np.where(self.labels == c)[0] for c in np.unique(self.labels)}

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for _ in range(self.n_episodes):
            episode_indices = []
            # 随机选择 N 个类别
            available_classes = [c for c, indices in self.class_indices.items() if len(indices) >= self.n_samples]
            if len(available_classes) < self.n_way:
                continue # 如果没有足够的类来构建episode，则跳过
            
            selected_classes = np.random.choice(available_classes, self.n_way, replace=False)
            
            for c in selected_classes:
                # 从每个选定类别中随机选择 n_samples 个样本
                class_idx = np.random.choice(self.class_indices[c], self.n_samples, replace=False)
                episode_indices.extend(class_idx)
            
            yield torch.LongTensor(episode_indices)


# train_protonet.py (接上文)

from torch.optim import Adam
from tqdm import tqdm

def prototypical_loss(embeddings, n_support, n_way, n_query):
    """计算 Prototypical Loss 和 Accuracy"""
    
    # 1. 分离 support 和 query 样本的嵌入
    embedding_dim = embeddings.size(-1)
    support_embeddings = embeddings[:n_way * n_support].view(n_way, n_support, embedding_dim)
    query_embeddings = embeddings[n_way * n_support:]

    # 2. 计算每个类的原型
    prototypes = support_embeddings.mean(dim=1)

    # 3. 计算 query 样本到每个原型的距离 (平方欧氏距离)
    # (n_query * n_way, embedding_dim) -> (n_query * n_way, 1, embedding_dim)
    # (n_way, embedding_dim) -> (1, n_way, embedding_dim)
    # 广播后相减
    distances = (query_embeddings.unsqueeze(1) - prototypes.unsqueeze(0)).pow(2).sum(dim=2)

    # 4. 计算损失
    # 将距离转换为负对数概率
    log_p_y = F.log_softmax(-distances, dim=1)
    
    # 生成 query set 的真实标签
    query_labels = torch.arange(n_way).repeat_interleave(n_query)
    
    # 使用负对数似然损失 (NLL Loss)
    loss = F.nll_loss(log_p_y, query_labels)

    # 5. 计算准确率
    y_hat = log_p_y.argmax(dim=1)
    acc = (y_hat == query_labels).float().mean()
    
    return loss, acc


from torch.optim import Adam
from tqdm import tqdm
# [新增] 导入可视化库
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def main():
    # ... (main函数前面的所有代码，包括训练和评估，都保持不变) ...
    # --- 超参数配置 ---
    N_WAY = 5
    K_SHOT = 5
    N_QUERY = 10
    N_TRAIN_EPISODES = 5000 # 使用增加后的训练量
    N_TEST_EPISODES = 200
    EMBEDDING_DIM = 64
    LEARNING_RATE = 0.001

    # --- 1. 数据准备 ---
    file_path = 'w/merged_all_attacks.csv'
    X, y, label_encoder = preprocess_data(file_path)

    if X is None:
        return 

    num_classes = len(np.unique(y))
    if num_classes < N_WAY:
        print(f"⚠️ 数据集类别数 ({num_classes}) 少于 N_WAY ({N_WAY})。")
        N_WAY = num_classes
        print(f"   已自动将 N_WAY 调整为: {N_WAY}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    train_dataset = TrafficDataset(X_train, y_train)
    test_dataset = TrafficDataset(X_test, y_test)

    # --- 2. 模型和优化器 ---
    print("\n--- 步骤 2: 初始化模型和优化器 ---")
    input_dim = X_train.shape[1]
    model = PrototypicalNetMLP(input_dim, embedding_dim=EMBEDDING_DIM)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    print(model)
    print("----------------------------------\n")

    # --- 3. 训练 ---
    print("\n--- 步骤 3: 开始训练 ---")
    # ... (训练循环代码不变) ...
    train_sampler = EpisodicBatchSampler(y_train, N_TRAIN_EPISODES, N_WAY, K_SHOT + N_QUERY)
    train_losses = []
    train_accuracies = []
    model.train()
    for episode_indices in tqdm(train_sampler, desc="Training Episodes"):
        optimizer.zero_grad()
        data, _ = train_dataset[episode_indices]
        embeddings = model(data)
        loss, acc = prototypical_loss(embeddings, n_support=K_SHOT, n_way=N_WAY, n_query=N_QUERY)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        train_accuracies.append(acc.item())
    print("训练完成。")
    print("----------------------------------\n")
    
    # --- 4. 评估 ---
    print("\n--- 步骤 4: 开始评估 ---")
    test_sampler = EpisodicBatchSampler(y_test, N_TEST_EPISODES, N_WAY, K_SHOT + N_QUERY)
    
    model.eval()
    
    # [新增] 创建一个字典来存储每个episode的预测和真实标签
    # 这样可以处理每个episode类别都不同的情况
    episode_results = {'preds': [], 'labels': [], 'class_map': []}
    
    # 获取所有可能的类别标签的数值编码
    all_possible_labels = np.arange(len(label_encoder.classes_))
    
    total_acc = 0.0

    with torch.no_grad():
        for episode_indices in tqdm(test_sampler, desc="Testing Episodes"):
            data, true_episode_labels_flat = test_dataset[episode_indices]
            
            # 确定当前episode中独特的类别及其在原始编码中的值
            # 例如: [Benign, SYNFlood, UDPScan] -> [0, 3, 7]
            true_episode_classes = np.unique(true_episode_labels_flat.numpy())
            
            embeddings = model(data)
            
            embedding_dim = embeddings.size(-1)
            support_embeddings = embeddings[:N_WAY * K_SHOT].view(N_WAY, K_SHOT, embedding_dim)
            query_embeddings = embeddings[N_WAY * K_SHOT:]
            prototypes = support_embeddings.mean(dim=1)
            distances = (query_embeddings.unsqueeze(1) - prototypes.unsqueeze(0)).pow(2).sum(dim=2)
            
            # 预测的标签是相对于当前 episode 的索引 (0, 1, 2, 3, 4)
            predictions_in_episode_idx = (-distances).argmax(dim=1)
            
            # 将 episode 内的索引映射回原始的类别编码
            # 例如，如果 episode 内预测为 1，且当前 episode 的类别是 [0, 3, 7]，那么预测 1 对应原始类别 3
            predicted_original_labels = torch.tensor([true_episode_classes[i] for i in predictions_in_episode_idx])
            
            # 获取 query set 的真实标签 (原始编码)
            true_query_original_labels = true_episode_labels_flat[N_WAY * K_SHOT:]

            episode_results['preds'].extend(predicted_original_labels.numpy())
            episode_results['labels'].extend(true_query_original_labels.numpy())

            # 计算准确率
            acc = (predicted_original_labels == true_query_original_labels).float().mean()
            total_acc += acc.item()
            
    avg_acc = total_acc / N_TEST_EPISODES
    print(f"\n评估结果: 平均准确率: {avg_acc * 100:.2f}%")
    
    # [核心修改] 在计算混淆矩阵时，提供完整的类别列表
    cm = confusion_matrix(
        episode_results['labels'], 
        episode_results['preds'], 
        labels=all_possible_labels  # <-- 关键修改！
    )
    
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, 
        display_labels=label_encoder.classes_  # <-- 这里的标签数量现在和 cm 的维度匹配了
    )
    
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, xticks_rotation='vertical', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("🔢 混淆矩阵已保存为 'confusion_matrix.png'")
    plt.show()

    # --- 5. 可视化 ---
    print("\n--- 步骤 5: 开始可视化 ---")
    # ... (训练曲线绘制部分代码不变) ...
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title("Training Loss per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies)
    plt.title("Training Accuracy per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig("training_curves.png")
    print("📈 训练曲线已保存为 'training_curves.png'")
    plt.show()

    print("🎨 正在生成 t-SNE 嵌入空间可视化...")
    n_samples_for_tsne = 1000 
    test_subset_indices = np.random.choice(len(test_dataset), n_samples_for_tsne, replace=False)
    test_subset_data, test_subset_labels = test_dataset[test_subset_indices]

    with torch.no_grad():
        test_embeddings = model(test_subset_data).numpy()
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    tsne_results = tsne.fit_transform(test_embeddings)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=test_subset_labels, cmap='viridis', alpha=0.7)
    
    # [核心修改] 将 NumPy 数组转换为 Python 列表
    legend_labels = label_encoder.inverse_transform(np.unique(test_subset_labels))
    plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels.tolist(), title="Classes")
    
    plt.title("t-SNE Visualization of Test Set Embeddings")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.savefig("embedding_space_tsne.png")
    print("🖼️ t-SNE 可视化图已保存为 'embedding_space_tsne.png'")
    plt.show()

if __name__ == '__main__':
    main()