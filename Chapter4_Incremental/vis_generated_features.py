import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

# Windows 环境修复
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 路径设置
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Chapter4_Incremental.models.diffusion_mlp import ConditionalDiffusionMLP

# ================= 配置 (必须与训练时一致) =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'saved_models', 'diffusion_final.pth')
STATS_PATH = os.path.join(os.path.dirname(__file__), 'saved_models', 'feature_stats.pt')
REAL_DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', '5G_NIDD_features_512dim.pt')
SAVE_IMG_PATH = os.path.join(os.path.dirname(__file__), 'generated_vs_real_v2.png')

# 采样参数
TARGET_CLASS_NAME = 'UDPFlood' # 目标攻击类别
NUM_SAMPLES = 500              # 采样数量
TIMESTEPS = 1000
HIDDEN_DIM = 1024              # <--- 已修正：与训练脚本保持一致
NUM_LAYERS = 6                 # <--- 已修正：与训练脚本保持一致

def get_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps).to(DEVICE)

@torch.no_grad()
def sample(model, n_samples, class_label, input_dim=512):
    """Diffusion 逆向采样过程"""
    model.eval()
    
    # 1. 从纯高斯噪声开始
    x = torch.randn((n_samples, input_dim)).to(DEVICE)
    labels = torch.full((n_samples,), class_label, device=DEVICE).long()
    
    # 准备调度参数
    betas = get_beta_schedule(TIMESTEPS)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.]).to(DEVICE), alphas_cumprod[:-1]])
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    
    # 2. 逐步去噪
    for t in tqdm(reversed(range(TIMESTEPS)), desc="Sampling", total=TIMESTEPS):
        t_batch = torch.full((n_samples,), t, device=DEVICE).long()
        
        predicted_noise = model(x, t_batch, labels)
        
        beta_t = betas[t]
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1. - alphas_cumprod[t])
        sqrt_recip_alpha_t = torch.sqrt(1.0 / alphas[t])
        
        mean = sqrt_recip_alpha_t * (x - beta_t * predicted_noise / sqrt_one_minus_alpha_cumprod_t)
        
        if t > 0:
            noise = torch.randn_like(x)
            sigma_t = torch.sqrt(posterior_variance[t])
            x = mean + sigma_t * noise
        else:
            x = mean
            
    return x

def main():
    print(f"Running on {DEVICE}")
    
    # 1. 加载统计量
    if not os.path.exists(STATS_PATH):
        print("Error: Stats file not found.")
        return
    stats = torch.load(STATS_PATH)
    mean = stats['mean'].to(DEVICE)
    std = stats['std'].to(DEVICE)
    
    # 2. 加载真实数据
    print("Loading Real Data...")
    data_dict = torch.load(REAL_DATA_PATH)
    all_features = data_dict['features']
    all_labels = data_dict['labels']
    class_to_idx = data_dict['class_to_idx']
    
    target_idx = class_to_idx[TARGET_CLASS_NAME]
    real_indices = (all_labels == target_idx).nonzero(as_tuple=True)[0]
    
    # 随机抽取真实样本
    if len(real_indices) > NUM_SAMPLES:
        indices = torch.randperm(len(real_indices))[:NUM_SAMPLES]
        real_features = all_features[real_indices[indices]].to(DEVICE)
    else:
        real_features = all_features[real_indices].to(DEVICE)

    # 3. 加载模型
    print(f"Loading Model (Dim={HIDDEN_DIM}, Layers={NUM_LAYERS})...")
    model = ConditionalDiffusionMLP(
        input_dim=512, 
        num_classes=len(class_to_idx),
        hidden_dim=HIDDEN_DIM, 
        num_layers=NUM_LAYERS
    ).to(DEVICE)
    
    # 加载权重
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Checkpoint loaded successfully.")
    except RuntimeError as e:
        print(f"\n[FATAL ERROR] Model mismatch: {e}")
        print("Please check if HIDDEN_DIM in this script matches the training script.")
        return

    # 4. 生成
    print(f"Generating {NUM_SAMPLES} Fake {TARGET_CLASS_NAME} samples...")
    fake_features_norm = sample(model, NUM_SAMPLES, target_idx)
    fake_features = fake_features_norm * std + mean # 反归一化
    
    # 5. t-SNE 可视化
    print("Computing t-SNE (this may take a moment)...")
    real_np = real_features.cpu().numpy()
    fake_np = fake_features.cpu().numpy()
    
    combined = np.vstack([real_np, fake_np])
    labels = np.concatenate([np.zeros(len(real_np)), np.ones(len(fake_np))])
    
    # 使用更高的 perplexity 以展示全局结构
    tsne = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto', random_state=42)
    embedded = tsne.fit_transform(combined)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(embedded[labels==0, 0], embedded[labels==0, 1], c='blue', alpha=0.3, label='Real Data', s=20)
    plt.scatter(embedded[labels==1, 0], embedded[labels==1, 1], c='red', alpha=0.3, label='Generated (AdaLN)', s=20)
    
    plt.title(f"Real vs Generated Features ({TARGET_CLASS_NAME})\nArchitecture: AdaLN-Diffusion")
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    plt.savefig(SAVE_IMG_PATH)
    print(f"Visualization saved to: {SAVE_IMG_PATH}")

if __name__ == '__main__':
    main()