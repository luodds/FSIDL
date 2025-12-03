import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# =================配置区域=================
# Windows 环境修复
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 路径配置 (根据你的截图目录结构调整)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)  # 添加根目录以便导入 Chapter 3 模块

# 数据与模型路径
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', '5G-NIDD', 'processed_images')
# 注意：使用 epoch_50 的预训练权重，它的泛化特征最好
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, 'Chapter3_Static_FewShot', 'saved_models', 'supcon_epoch_50.pth')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, '5G_NIDD_features_512dim.pt')

# 硬件配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256  # 推理阶段Batch Size可以大一点
NUM_WORKERS = 4   # Windows下如果报错，改为0

# =================导入模型=================
try:
    from Chapter3_Static_FewShot.models.resnet_encoder import SupConResNet
except ImportError as e:
    print(f"Error: 无法导入 Chapter 3 的模型定义。请检查 sys.path 设置。\nDetails: {e}")
    sys.exit(1)

def main():
    print(f"Running on: {DEVICE}")
    print(f"Data Source: {DATA_DIR}")
    
    # 1. 准备数据预处理 (必须与预训练时保持一致：灰度 -> Tensor -> Normalize)
    # 不需要 TwoCropTransform，只需要标准的验证集变换
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # 使用 ImageFolder 加载全量数据
    dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )
    
    print(f"Found {len(dataset)} images across {len(dataset.classes)} classes.")
    print(f"Classes: {dataset.class_to_idx}")

    # 2. 加载模型
    model = SupConResNet(name='resnet18')
    
    # === 修复后的加载逻辑 ===
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint: {CHECKPOINT_PATH}")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        
        # 1. 尝试定位 state_dict
        if 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
            print("Found 'model_state_dict' in checkpoint.")
        elif 'model' in ckpt:
            state_dict = ckpt['model']
            print("Found 'model' in checkpoint.")
        else:
            state_dict = ckpt
            print("Assuming checkpoint is a raw state_dict.")
            
        # 2. 处理 'module.' 前缀 (多卡训练遗留)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v
            
        # 3. 加载权重 (使用 strict=False 忽略 head 层的差异，但必须确保 encoder 加载成功)
        msg = model.load_state_dict(new_state_dict, strict=False)
        print(f"Load Result: {msg}")
        
        # 4. [关键] 验证是否真的加载成功
        # 检查 conv1 的权重是否与随机初始化不同，或者直接检查 missing_keys 是否包含核心层
        if any(k.startswith('encoder.conv1') for k in msg.missing_keys):
            print("\n[CRITICAL WARNING] 核心编码器权重未加载！请检查 Checkpoint 结构！")
            print(f"Missing keys example: {msg.missing_keys[:5]}")
            sys.exit(1) # 强制退出，防止生成无效数据
        else:
            print("\n[SUCCESS] Encoder weights loaded successfully.")
            
    else:
        raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}")

    model = model.to(DEVICE)
    model.eval()

    # 3. 特征提取循环
    all_features = []
    all_labels = []

    print("Start feature extraction...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting"):
            images = images.to(DEVICE)
            
            # === 关键步骤 ===
            # 我们需要 Encoder 的输出 (512维)，而不是 Projection Head 的输出
            # 假设 SupConResNet 结构是 self.encoder (ResNet) -> self.head (MLP)
            # ResNet18 的 forward 通常输出 logits，这里我们需要取倒数第二层
            # 这里的调用方式取决于你的 resnet_encoder.py 实现。
            # 通常 SupConResNet 的 .encoder 属性就是一个标准的 ResNet
            
            if hasattr(model, 'encoder'):
                # 显式调用 encoder 部分
                features = model.encoder(images)
            else:
                # 如果没有显式 encoder 属性，尝试直接 forward (需确认 forward 是否包含 head)
                # 备用方案：通常 ResNet 除去 fc 层后直接输出就是 features
                features = model(images) 
            
            # 确保是平铺的向量 (Batch, 512)
            features = torch.flatten(features, 1)
            
            all_features.append(features.cpu())
            all_labels.append(labels)

    # 4. 拼接与保存
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    print(f"Extraction complete.")
    print(f"Feature shape: {all_features.shape}") # 应该是 [Total_Images, 512]
    print(f"Label shape: {all_labels.shape}")

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 保存字典
    save_dict = {
        'features': all_features,
        'labels': all_labels,
        'class_to_idx': dataset.class_to_idx,
        'idx_to_class': {v: k for k, v in dataset.class_to_idx.items()}
    }

    torch.save(save_dict, OUTPUT_FILE)
    print(f"Saved dataset to: {OUTPUT_FILE}")
    print("Next Step: Use this file to train the Diffusion Model.")

if __name__ == '__main__':
    main()