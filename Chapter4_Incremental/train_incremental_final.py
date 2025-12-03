import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import copy
from sklearn.metrics import accuracy_score, classification_report

# Windows 环境修复
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Chapter4_Incremental.models.diffusion_mlp import ConditionalDiffusionMLP

# ================= 配置 =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURE_PATH = os.path.join(os.path.dirname(__file__), 'data', '5G_NIDD_features_512dim.pt')
DIFFUSION_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'saved_models', 'diffusion_final.pth')
STATS_PATH = os.path.join(os.path.dirname(__file__), 'saved_models', 'feature_stats.pt')

# 模拟增量设置
# Task 1: 基础业务 (旧类)
OLD_CLASSES = ['Benign', 'HTTPFlood', 'ICMPFlood', 'SlowrateDoS', 'SYNFlood']
# Task 2: 新增威胁 (新类)
NEW_CLASSES = ['UDPFlood', 'SYNScan', 'TCPConnectScan', 'UDPScan']

# 训练参数
BATCH_SIZE = 256
LR = 1e-3
EPOCHS = 30

# Diffusion 参数 (必须与训练好的模型一致)
DIFFUSION_HIDDEN_DIM = 1024 
DIFFUSION_LAYERS = 6        # 注意：虽然你的脚本写着3，但我们用的是AdaLN版，代码结构已变
DIFFUSION_TIMESTEPS = 1000

# ================= 工具函数 =================

class SimpleClassifier(nn.Module):
    """简单的线性分类器"""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.fc(x)

@torch.no_grad()
def generate_old_features(model, labels_to_gen, n_samples_per_class, mean, std):
    """使用 Diffusion 生成旧类特征"""
    model.eval()
    generated_feats = []
    generated_labels = []
    
    print(f"\n[Generative Replay] Generating {n_samples_per_class} samples for each old class...")
    
    # 预计算调度参数
    betas = torch.linspace(0.0001, 0.02, DIFFUSION_TIMESTEPS).to(DEVICE)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.]).to(DEVICE), alphas_cumprod[:-1]])
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    for label in labels_to_gen:
        # 批量生成
        x = torch.randn((n_samples_per_class, 512)).to(DEVICE)
        y = torch.full((n_samples_per_class,), label, device=DEVICE).long()
        
        for t in reversed(range(DIFFUSION_TIMESTEPS)):
            t_batch = torch.full((n_samples_per_class,), t, device=DEVICE).long()
            predicted_noise = model(x, t_batch, y)
            
            beta_t = betas[t]
            sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1. - alphas_cumprod[t])
            sqrt_recip_alpha_t = torch.sqrt(1.0 / alphas[t])
            
            mean_t = sqrt_recip_alpha_t * (x - beta_t * predicted_noise / sqrt_one_minus_alpha_cumprod_t)
            
            if t > 0:
                noise = torch.randn_like(x)
                sigma_t = torch.sqrt(posterior_variance[t])
                x = mean_t + sigma_t * noise
            else:
                x = mean_t
        
        # 反归一化
        x = x * std + mean
        generated_feats.append(x.cpu())
        generated_labels.append(y.cpu())
        print(f"  -> Class {label} generated.")
        
    return torch.cat(generated_feats), torch.cat(generated_labels)

def train_classifier(name, train_loader, test_loader_all, num_classes):
    """训练分类器并评估"""
    print(f"\n=== Training {name} ===")
    model = SimpleClassifier(512, num_classes).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
    
    # 评估
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in test_loader_all:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            
    acc = accuracy_score(all_targets, all_preds)
    print(f"Result for {name}: Overall Accuracy = {acc:.4f}")
    return acc, all_targets, all_preds

# ================= 主程序 =================
def main():
    # 1. 准备数据
    data_dict = torch.load(FEATURE_PATH)
    features = data_dict['features']
    labels = data_dict['labels']
    class_to_idx = data_dict['class_to_idx']
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    old_indices = [class_to_idx[c] for c in OLD_CLASSES]
    new_indices = [class_to_idx[c] for c in NEW_CLASSES]
    
    print(f"Old Classes (Task 1): {OLD_CLASSES}")
    print(f"New Classes (Task 2): {NEW_CLASSES}")
    
    # 拆分数据集
    mask_old = torch.tensor([l.item() in old_indices for l in labels])
    mask_new = torch.tensor([l.item() in new_indices for l in labels])
    
    X_old, Y_old = features[mask_old], labels[mask_old]
    X_new, Y_new = features[mask_new], labels[mask_new]
    
    # 划分 Train/Test (80/20)
    def split_data(X, Y):
        perm = torch.randperm(len(X))
        split = int(len(X) * 0.8)
        return X[perm[:split]], Y[perm[:split]], X[perm[split:]], Y[perm[split:]]

    X_old_train, Y_old_train, X_old_test, Y_old_test = split_data(X_old, Y_old)
    X_new_train, Y_new_train, X_new_test, Y_new_test = split_data(X_new, Y_new)
    
    # 全量测试集 (用于最终评估)
    X_test_all = torch.cat([X_old_test, X_new_test])
    Y_test_all = torch.cat([Y_old_test, Y_new_test])
    test_loader_all = DataLoader(TensorDataset(X_test_all, Y_test_all), batch_size=BATCH_SIZE)

    # ==========================================
    # 场景 A: 灾难性遗忘 (Catastrophic Forgetting)
    # 只用新数据训练
    # ==========================================
    loader_finetune = DataLoader(TensorDataset(X_new_train, Y_new_train), batch_size=BATCH_SIZE, shuffle=True)
    acc_forget, _, _ = train_classifier("Baseline (Fine-Tune)", loader_finetune, test_loader_all, len(class_to_idx))
    
    # ==========================================
    # 场景 B: 生成式重放 (Generative Replay)
    # 新数据 + Diffusion 生成的旧数据
    # ==========================================
    
    # B1. 加载 Diffusion 模型
    stats = torch.load(STATS_PATH)
    mean, std = stats['mean'].to(DEVICE), stats['std'].to(DEVICE)
    
    # 注意：这里如果你的 vis_generated_features 用的 3层，这里也要改为 3
    # 但是根据刚才的讨论，应该是 3层
    diffusion = ConditionalDiffusionMLP(
        input_dim=512, num_classes=len(class_to_idx), 
        hidden_dim=DIFFUSION_HIDDEN_DIM, num_layers=3 # 修正为 3，如果你 vis 脚本用的是 3
    ).to(DEVICE)
    diffusion.load_state_dict(torch.load(DIFFUSION_MODEL_PATH, map_location=DEVICE))
    
    # B2. 生成旧数据
    # 为了平衡，生成的数量大约等于新数据的数量 / 旧类个数
    n_gen_per_class = len(X_new_train) // len(old_indices) 
    n_gen_per_class = min(n_gen_per_class, 1000) # 限制上限，省时间
    
    X_gen, Y_gen = generate_old_features(diffusion, old_indices, n_gen_per_class, mean, std)
    
    # B3. 混合数据
    X_replay = torch.cat([X_new_train, X_gen])
    Y_replay = torch.cat([Y_new_train, Y_gen])
    
    loader_replay = DataLoader(TensorDataset(X_replay, Y_replay), batch_size=BATCH_SIZE, shuffle=True)
    acc_replay, targets, preds = train_classifier("Ours (Generative Replay)", loader_replay, test_loader_all, len(class_to_idx))
    
    # ================= 结果对比 =================
    print("\n" + "="*40)
    print("FINAL RESULTS")
    print("="*40)
    print(f"1. Baseline (Forget): {acc_forget:.2%} (旧类大概率全错)")
    print(f"2. Our Method (Replay): {acc_replay:.2%} (旧类应该能保持)")
    print("="*40)
    
    # 打印详细分类报告
    print("\nDetailed Report for Our Method:")
    print(classification_report(targets, preds, target_names=list(class_to_idx.keys())))

if __name__ == '__main__':
    main()