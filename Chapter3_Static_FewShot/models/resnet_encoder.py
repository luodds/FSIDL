import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class SupConResNet(nn.Module):
    """
    用于监督对比学习 (SupCon) 的 ResNet 模型
    结构: Encoder (ResNet18) + Projection Head (MLP)
    """
    def __init__(self, name='resnet18', head='mlp', feat_dim=128):
        super(SupConResNet, self).__init__()
        
        # 1. 加载标准 ResNet18 骨干网络
        # weights=None 表示不使用 ImageNet 预训练权重，因为我们的流量图和自然图像差异巨大，
        # 且我们有 50万张图，完全可以从头训练 (Train from scratch)。
        model_fun = resnet18
        self.encoder = model_fun(weights=None)
        
        # 2. [关键修改] 修改第一层卷积以适应灰度图 (1通道)
        # 原版是: nn.Conv2d(3, 64, kernel_size=7, ...)
        # 修改为: nn.Conv2d(1, 64, kernel_size=7, ...)
        self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 3. 获取 Encoder 的输出维度 (ResNet18 为 512)
        dim_in = self.encoder.fc.in_features
        
        # 4. 移除原始的全连接分类层 (FC)，用 Identity 占位
        # 这样调用 self.encoder(x) 时输出的就是 512维 特征向量
        self.encoder.fc = nn.Identity()
        
        # 5. 定义 Projection Head (投影头)
        # 结构: Linear -> ReLU -> Linear
        # 作用: 将特征映射到低维空间计算 Contrastive Loss
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError('head not supported: {}'.format(head))

    def forward(self, x):
        """
        前向传播
        返回: 经过 L2 归一化的投影特征 (bsz, feat_dim)
        """
        # 提取特征 (Batch, 512)
        feat = self.encoder(x)
        
        # 投影到低维空间 (Batch, 128)
        feat = self.head(feat)
        
        # [重要] 对比学习通常要在单位球面上计算距离，所以要做 Normalize
        feat = F.normalize(feat, dim=1)
        
        return feat

    def extract_features(self, x):
        """
        仅用于推理/第四章使用：提取不含 Projection Head 的原始特征
        返回: (Batch, 512)
        """
        return self.encoder(x)

# --- 测试代码 ---
if __name__ == '__main__':
    # 模拟输入数据: Batch=4, Channel=1, H=28, W=28
    dummy_input = torch.randn(4, 1, 28, 28)
    
    # 实例化模型
    model = SupConResNet(name='resnet18', head='mlp', feat_dim=128)
    
    # 1. 测试训练模式输出
    output = model(dummy_input)
    print(f"[测试] 训练模式输出形状 (Projection): {output.shape}") # 预期: [4, 128]
    
    # 2. 测试特征提取模式输出
    features = model.extract_features(dummy_input)
    print(f"[测试] 特征提取输出形状 (Encoder): {features.shape}") # 预期: [4, 512]
    
    # 3. 检查网络第一层通道数
    first_layer = model.encoder.conv1
    print(f"[测试] 第一层卷积输入通道数: {first_layer.in_channels}") # 预期: 1
    
    print("✅ 模型定义测试通过！")