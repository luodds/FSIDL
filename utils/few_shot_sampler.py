import torch
import numpy as np
from torch.utils.data import Subset

def get_few_shot_indices(dataset, n_shots=5):
    """
    从数据集中构建 N-shot 索引
    返回:
        train_indices: 包含每个类别 n_shots 个样本的索引列表
        test_indices: 剩余所有样本的索引列表
    """
    # 获取所有标签
    # 注意：我们的 InMemoryDataset 使用 .labels (Tensor) 存储标签
    if isinstance(dataset.labels, torch.Tensor):
        labels = dataset.labels.numpy()
    else:
        labels = np.array(dataset.labels)
        
    num_classes = len(dataset.classes)
    train_indices = []
    test_indices = []
    
    print(f"[Sampler] 正在构建 {n_shots}-shot 数据集...")
    
    for class_id in range(num_classes):
        # 找到当前类别所有样本的索引
        cls_indices = np.where(labels == class_id)[0]
        
        # 随机打乱
        np.random.shuffle(cls_indices)
        
        # 取前 n_shots 个作为训练集
        # 如果某类样本不足 n_shots，则全取
        cut = min(n_shots, len(cls_indices))
        
        train_indices.extend(cls_indices[:cut])
        test_indices.extend(cls_indices[cut:])
        
    print(f"[Sampler] 训练集样本数: {len(train_indices)} (每类约 {n_shots} 个)")
    print(f"[Sampler] 测试集样本数: {len(test_indices)}")
    
    return train_indices, test_indices