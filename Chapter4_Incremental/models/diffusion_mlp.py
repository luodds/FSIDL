import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class AdaLNBlock(nn.Module):
    """
    Adaptive Layer Normalization Block (FiLM机制)
    这是让 Label 强力控制生成的关键组件
    """
    def __init__(self, dim):
        super().__init__()
        # 标准的 MLP 部分
        self.norm1 = nn.LayerNorm(dim)
        self.act = nn.SiLU()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        
        # 条件投影：从 cond 映射出 (scale, shift)
        # 输出维度是 2 * dim，一半给 scale，一半给 shift
        self.ada_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 2)
        )

    def forward(self, x, cond):
        # x: (B, dim)
        # cond: (B, dim)
        
        residual = x
        
        # 1. 计算 Scale 和 Shift
        style = self.ada_proj(cond) 
        scale, shift = style.chunk(2, dim=-1) # 分割成两份
        
        # 2. AdaLN 核心公式: x = norm(x) * (1 + scale) + shift
        x = self.norm1(x)
        x = x * (1 + scale) + shift 
        
        # 3. MLP 处理
        x = self.act(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        
        return x + residual

class ConditionalDiffusionMLP(nn.Module):
    def __init__(self, input_dim=512, num_classes=10, hidden_dim=1024, num_layers=6):
        super().__init__()
        
        # 1. Embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.class_emb = nn.Embedding(num_classes, hidden_dim)

        # 2. 主干网络
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # 增加深度到 6层，增强拟合能力
        self.blocks = nn.ModuleList([
            AdaLNBlock(hidden_dim) for _ in range(num_layers)
        ])

        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, t, labels):
        # Embed
        t_emb = self.time_mlp(t)
        c_emb = self.class_emb(labels)
        
        # 融合条件：将 Time 和 Class 相加作为全局 Condition
        cond = t_emb + c_emb 
        
        h = self.input_proj(x)
        
        for block in self.blocks:
            h = block(h, cond) # 传入 cond 进行 AdaLN
            
        return self.output_proj(h)