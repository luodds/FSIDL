import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np

# 路径 Hack
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Chapter3_Static_FewShot.data_loader import InMemoryDataset, get_supcon_transforms
from Chapter3_Static_FewShot.models.prompt_learner import VisualPromptResNet
from utils.few_shot_sampler import get_few_shot_indices
# 引入绘图工具
from utils.visualizer import plot_training_curves, plot_confusion_matrix, plot_tsne

class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, index):
        x, y = self.subset[index]
        if isinstance(x, torch.Tensor):
            from PIL import Image
            x = Image.fromarray(x.numpy(), mode='L')
        if self.transform:
            x = self.transform(x)
        return x, y
    def __len__(self):
        return len(self.subset)

def train_few_shot():
    # --- 配置 ---
    N_SHOTS = 5
    EPOCHS = 30
    BATCH_SIZE = 64
    LR = 0.01
    DEVICE = 'cuda'
    SAVE_DIR = './Chapter3_Static_FewShot/results_visual' # 图片保存路径
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    DATA_PATH = './data/5G-NIDD/processed_images'
    PRETRAINED_PATH = './Chapter3_Static_FewShot/saved_models/supcon_epoch_50.pth'
    cache_file = './data/5G-NIDD/5G_NIDD_tensor_cache.pt'

    # 1. 加载数据
    full_dataset = InMemoryDataset(DATA_PATH, cache_path=cache_file)
    class_names = full_dataset.classes # 获取类别名列表

    # 2. 划分
    train_idx, test_idx = get_few_shot_indices(full_dataset, n_shots=N_SHOTS)
    
    train_set = DatasetWrapper(Subset(full_dataset, train_idx), transform=get_supcon_transforms(mode='train'))
    test_set = DatasetWrapper(Subset(full_dataset, test_idx), transform=get_supcon_transforms(mode='test'))
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=2048, shuffle=False, num_workers=0)

    # 3. 模型
    model = VisualPromptResNet(pretrained_path=PRETRAINED_PATH, num_classes=len(class_names))
    model = model.to(DEVICE)
    
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
                          lr=LR, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # --- 记录数据用于绘图 ---
    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}

    # # --- 训练循环 ---
    # best_acc = 0.0
    # print(f"[Info] 开始训练... 结果将保存至 {SAVE_DIR}")

    # for epoch in range(1, EPOCHS + 1):
    #     model.train()
    #     total_loss = 0; correct = 0; total = 0
        
    #     for images, labels in train_loader:
    #         images, labels = images.to(DEVICE), labels.to(DEVICE)
    #         optimizer.zero_grad()
    #         outputs = model(images)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
            
    #         total_loss += loss.item()
    #         _, predicted = outputs.max(1)
    #         total += labels.size(0)
    #         correct += predicted.eq(labels).sum().item()
            
    #     train_acc = 100. * correct / total
        
    #     # 验证
    #     model.eval()
    #     test_correct = 0; test_total = 0
    #     with torch.no_grad():
    #         for images, labels in test_loader:
    #             images, labels = images.to(DEVICE), labels.to(DEVICE)
    #             outputs = model(images)
    #             _, predicted = outputs.max(1)
    #             test_total += labels.size(0)
    #             test_correct += predicted.eq(labels).sum().item()
        
    #     test_acc = 100. * test_correct / test_total
        
    #     # 记录历史
    #     history['train_loss'].append(total_loss)
    #     history['train_acc'].append(train_acc)
    #     history['test_acc'].append(test_acc)

    #     print(f"Epoch {epoch}: Train Loss={total_loss:.4f}, Test Acc={test_acc:.2f}%")
        
    #     if test_acc > best_acc:
    #         best_acc = test_acc
    #         torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_fewshot_model.pth'))

    # print(f"[Done] 训练结束。最高准确率: {best_acc:.2f}%")

    # ==========================================
    #              可视化阶段
    # ==========================================
    print("\n[Visual] 开始生成可视化图表...")

    # 1. 绘制训练曲线 (保持不变)
    plot_training_curves(history['train_loss'], history['train_acc'], history['test_acc'], SAVE_DIR)

    # 2. 均衡采样逻辑 (专门修复 t-SNE 只有 Benign 的问题)
    print("[Visual] 正在进行【均衡采样】以生成高质量 t-SNE...")
    
    # 重新加载最佳模型
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'best_fewshot_model.pth')))
    model.eval()
    
    # --- 均衡采样配置 ---
    SAMPLES_PER_CLASS = 200  # 每个类别只取 200 个样本画图，保证图像清晰且计算快
    num_classes = len(class_names)
    class_counts = {i: 0 for i in range(num_classes)} # 计数器: {0: 0, 1: 0, ...}
    
    tsne_features = []
    tsne_labels = []
    
    # 同时也收集全量预测用于混淆矩阵
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        # 遍历测试集
        for images, labels in tqdm(test_loader, desc="Balanced Sampling"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # 1. 获取分类结果 (用于混淆矩阵 - 全量)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 2. 获取特征 (用于 t-SNE - 均衡采样)
            # 检查当前 batch 里有没有我们需要填补名额的类别
            # 如果所有类别都凑够了 SAMPLES_PER_CLASS，就跳过特征提取，节省时间
            if all(c >= SAMPLES_PER_CLASS for c in class_counts.values()):
                continue

            # 提取特征
            x_prompted = images + model.prompt
            feats = model.backbone.extract_features(x_prompted)
            
            # 逐个样本判断是否需要保留
            feats_np = feats.cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            for i in range(len(labels_np)):
                label = int(labels_np[i])
                # 如果该类别的名额还没满，就收录
                if class_counts[label] < SAMPLES_PER_CLASS:
                    tsne_features.append(feats_np[i])
                    tsne_labels.append(labels_np[i])
                    class_counts[label] += 1

    print(f"[Visual] 采样统计: {class_counts}")

    # 3. 绘制混淆矩阵 (全量)
    plot_confusion_matrix(all_labels, all_preds, class_names, SAVE_DIR)

    # 4. 绘制 t-SNE (均衡样本)
    if len(tsne_features) > 0:
        tsne_features = np.array(tsne_features)
        tsne_labels = np.array(tsne_labels)
        print(f"[Visual] t-SNE 输入数据形状: {tsne_features.shape}")
        
        plot_tsne(tsne_features, tsne_labels, class_names, SAVE_DIR, 
                  title=f"t-SNE Visualization ({SAMPLES_PER_CLASS} samples/class)")
    else:
        print("[Error] 未收集到 t-SNE 数据，请检查测试集。")
    
    print(f"\n🎉 所有结果已保存至: {os.path.abspath(SAVE_DIR)}")

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    train_few_shot()