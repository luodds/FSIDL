import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import numpy as np
import os

# 设置风格
plt.style.use('seaborn-v0_8-whitegrid')

def plot_training_curves(train_losses, train_accs, test_accs, save_dir):
    """绘制 Loss 和 Accuracy 曲线"""
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # 1. Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 2. Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'g--', label='Train Acc', linewidth=2)
    plt.plot(epochs, test_accs, 'r-', label='Test Acc', linewidth=2)
    plt.title('Accuracy Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=300)
    print(f"[Visual] 训练曲线已保存: {save_path}")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, save_dir):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (Few-Shot)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=300)
    print(f"[Visual] 混淆矩阵已保存: {save_path}")
    plt.close()

def plot_tsne(features, labels, class_names, save_dir, title="t-SNE Feature Visualization"):
    """
    绘制 t-SNE 特征分布图
    features: (N, D) numpy array
    labels: (N,) numpy array
    """
    print("[Visual] 正在计算 t-SNE (这可能需要几秒钟)...")
    # 降维到 2D
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(features)

    plt.figure(figsize=(10, 10))
    # 使用这种方式绘制散点图以支持图例
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels, cmap='tab10', alpha=0.6, s=10)
    
    # 创建图例
    handles, _ = scatter.legend_elements()
    plt.legend(handles, class_names, loc="best", title="Classes")
    
    plt.title(title)
    plt.axis('off') # 去掉坐标轴更美观
    
    save_path = os.path.join(save_dir, 'tsne_visualization.png')
    plt.savefig(save_path, dpi=300)
    print(f"[Visual] t-SNE 图已保存: {save_path}")
    plt.close()