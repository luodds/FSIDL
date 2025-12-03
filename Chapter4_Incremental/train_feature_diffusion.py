import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# 路径配置
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Chapter4_Incremental.models.diffusion_mlp import ConditionalDiffusionMLP

# ================= 配置 =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURE_PATH = os.path.join(os.path.dirname(__file__), 'data', '5G_NIDD_features_512dim.pt')
SAVE_DIR = os.path.join(os.path.dirname(__file__), 'saved_models')
os.makedirs(SAVE_DIR, exist_ok=True)

# 超参数
EPOCHS = 200            # 训练轮数
BATCH_SIZE = 4096       # 特征向量很小，显存足够大可以开大
LR = 1e-4
TIMESTEPS = 1000        # 扩散步数
HIDDEN_DIM = 1024        # 模型隐藏层宽度
NUM_LAYERS = 6          # 深度

def get_beta_schedule(timesteps):
    """线性的 Beta Schedule"""
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps).to(DEVICE)

def main():
    print(f"Running on {DEVICE}")
    
    # 1. 加载特征数据
    if not os.path.exists(FEATURE_PATH):
        print("Error: Feature file not found. Run extract_features.py first.")
        return
        
    data_dict = torch.load(FEATURE_PATH)
    features = data_dict['features'].float() # (N, 512)
    labels = data_dict['labels'].long()      # (N,)
    num_classes = len(data_dict['class_to_idx'])
    
    print(f"Loaded Features: {features.shape}")
    
    # 2. 数据标准化 (Standardization) -> Crucial for Diffusion!
    # 计算全局均值和方差
    mean = features.mean(dim=0)
    std = features.std(dim=0) + 1e-6 # 防止除零
    
    # 保存统计量，用于后续生成时反归一化
    stats = {'mean': mean, 'std': std}
    torch.save(stats, os.path.join(SAVE_DIR, 'feature_stats.pt'))
    
    # 归一化数据
    features_norm = (features - mean) / std
    
    dataset = TensorDataset(features_norm, labels)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    # 3. 初始化模型与优化器
    model = ConditionalDiffusionMLP(
        input_dim=512, 
        num_classes=num_classes, 
        hidden_dim=HIDDEN_DIM, 
        num_layers=NUM_LAYERS
    ).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    
    # 4. 扩散过程参数
    betas = get_beta_schedule(TIMESTEPS)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    # 5. 训练循环
    loss_history = []
    
    print("Start Training Diffusion Model...")
    model.train()
    
    for epoch in range(EPOCHS):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        epoch_loss = 0
        
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(DEVICE) # (B, 512)
            batch_y = batch_y.to(DEVICE) # (B,)
            batch_size = batch_x.size(0)
            
            # A. 随机采样时间步 t
            t = torch.randint(0, TIMESTEPS, (batch_size,), device=DEVICE).long()
            
            # B. 生成高斯噪声 epsilon
            noise = torch.randn_like(batch_x)
            
            # C. 加噪 (Forward Process)
            # x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * noise
            sqrt_alpha_bar_t = torch.sqrt(alphas_cumprod[t])[:, None]
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alphas_cumprod[t])[:, None]
            
            x_t = sqrt_alpha_bar_t * batch_x + sqrt_one_minus_alpha_bar_t * noise
            
            # D. 模型预测噪声 (Reverse Process)
            predicted_noise = model(x_t, t, batch_y)
            
            # E. 计算损失
            loss = loss_fn(predicted_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'Loss': loss.item()})
            
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        
        # 每10轮保存一次
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(SAVE_DIR, f'diffusion_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), save_path)
            
    # 6. 保存最终模型与 Loss 曲线
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'diffusion_final.pth'))
    
    plt.plot(loss_history)
    plt.title('Diffusion Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.savefig(os.path.join(SAVE_DIR, 'training_loss.png'))
    print("Training Complete.")

if __name__ == '__main__':
    main()