import torch
import torch.nn as nn
from .resnet_encoder import SupConResNet

class VisualPromptResNet(nn.Module):
    def __init__(self, pretrained_path, num_classes=9, prompt_size=28):
        super(VisualPromptResNet, self).__init__()
        
        # 1. 加载预训练好的 SupCon Encoder
        print(f"[Model] Loading pretrained weights from: {pretrained_path}")
        checkpoint = torch.load(pretrained_path)
        
        # 初始化骨干网
        self.backbone = SupConResNet(name='resnet18', head='mlp')
        
        # 加载权重 (注意：我们只需要 encoder 部分，不需要 projection head)
        state_dict = checkpoint['model_state_dict']
        # 过滤掉 'head' 开头的权重，只保留 encoder
        encoder_dict = {k: v for k, v in state_dict.items() if 'encoder' in k}
        self.backbone.load_state_dict(state_dict, strict=False)
        
        # 2. [关键] 冻结骨干网络参数 (Frozen Backbone)
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        print("[Model] Backbone frozen. Only Prompt and Classifier head will be updated.")

        # 3. 定义 Visual Prompt (可学习的参数)
        # 方法：Input-level Prompting (在输入图像上叠加一个可学习的 Pattern)
        # 形状与输入图像一致: [1, 1, 28, 28]
        self.prompt = nn.Parameter(torch.zeros(1, 1, prompt_size, prompt_size))
        # 使用 Xavier 初始化 Prompt，使其具有一定的初始扰动
        nn.init.xavier_uniform_(self.prompt)

        # 4. 定义分类头 (Classifier Head)
        # ResNet18 的输出特征维度是 512
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # x shape: [batch, 1, 28, 28]
        
        # 1. 将 Prompt 叠加到输入图像上
        # 广播机制: prompt 会自动扩展到 batch 大小
        x_prompted = x + self.prompt
        
        # 2. 通过冻结的 Encoder 提取特征
        features = self.backbone.extract_features(x_prompted)
        
        # 3. 分类
        logits = self.fc(features)
        
        return logits