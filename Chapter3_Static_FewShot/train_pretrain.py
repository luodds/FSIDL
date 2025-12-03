import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import torch
import torch.optim as optim
import time
from tqdm import tqdm

# 将项目根目录加入 Python 路径，确保能 import 到 utils 和 losses
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Chapter3_Static_FewShot.data_loader import get_dataloaders
from Chapter3_Static_FewShot.models.resnet_encoder import SupConResNet
from Chapter3_Static_FewShot.losses.supcon_loss import SupConLoss

def train():
    # --- 超参数设置 ---
    DATA_PATH = './data/5G-NIDD/processed_images'
    SAVE_DIR = './Chapter3_Static_FewShot/saved_models'
    
    BATCH_SIZE = 1024      # 显存够大可以开到 256 或 512
    EPOCHS = 50          # 预训练通常需要较多轮次，建议 50-100
    LEARNING_RATE = 1e-3 # 初始学习率
    TEMP = 0.07          # SupCon 温度系数
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 确保保存目录存在
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"[Info] 使用设备: {DEVICE}")
    
    # 1. 加载数据
    # train_loader 返回的 images 是一个 list: [view1, view2]
    train_loader, _, n_cls = get_dataloaders(DATA_PATH, batch_size=BATCH_SIZE, num_workers=4) # Windows设为0

    # 2. 初始化模型
    model = SupConResNet(name='resnet18', head='mlp', feat_dim=128)
    model = model.to(DEVICE)

    # 3. 定义 Loss 和 优化器
    criterion = SupConLoss(temperature=TEMP).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 学习率衰减策略 (可选)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    # --- 训练循环 ---
    print(f"[Info] 开始预训练 SupCon Encoder... 总轮数: {EPOCHS}")
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        
        # 使用 tqdm 显示进度条
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        
        for images, labels in progress_bar:
            # images 是 [view1, view2]
            # 这里的 labels 是真实的类别标签，SupCon 利用它来拉近同类
            
            # 堆叠视图: [bsz, 1, 28, 28] -> [bsz, 2, 1, 28, 28] -> reshape 传入模型?
            # 为了效率，通常把 view1 和 view2 拼成一个大的 batch 传入模型，然后再拆开
            
            images = torch.cat([images[0], images[1]], dim=0) # [2*bsz, 1, 28, 28]
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # 前向传播
            features = model(images) # 输出 [2*bsz, 128]
            
            # 重塑特征以适应 SupConLoss 格式: [bsz, n_views, dim]
            bsz = labels.shape[0]
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.stack([f1, f2], dim=1) # [bsz, 2, 128]
            
            # 计算 Loss
            loss = criterion(features, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 更新进度条后缀
            progress_bar.set_postfix(loss=loss.item())

        # 更新学习率
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")

        # --- 定期保存模型 ---
        if epoch % 10 == 0 or epoch == EPOCHS:
            save_path = os.path.join(SAVE_DIR, f'supcon_epoch_{epoch}.pth')
            # 保存模型状态（包含 encoder 和 head）
            # 后续使用时主要加载 model.encoder 的权重
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path)
            print(f"[Save] 模型已保存至: {save_path}")

    print("[Done] 预训练完成！")

if __name__ == '__main__':
    train()