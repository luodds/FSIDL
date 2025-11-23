import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
from PIL import Image
from tqdm import tqdm
import concurrent.futures

# --- 解决 OMP 冲突 ---
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# --- 1. 全局类定义 ---

class TwoCropTransform:
    def __init__(self, transform):
        self.transform = transform
    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

class TransformSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, index):
        # subset[index] 返回的是 (img_tensor, label)
        x, y = self.subset[index]
        
        # 这里 x 已经是 Tensor(uint8) 或 PIL，取决于 Dataset 的实现
        # 为了兼容 torchvision transforms，我们需要确保它是 PIL 格式
        # 因为我们底层存储改为了 Tensor，这里需要转换回 PIL 以应用 RandomResizedCrop 等
        if isinstance(x, torch.Tensor):
            x = Image.fromarray(x.numpy(), mode='L')

        if self.transform:
            x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.subset)

def load_image_file(args):
    """辅助函数：读取单张图片，返回 numpy array 以节省内存并方便 Tensor 化"""
    path, label = args
    try:
        with open(path, 'rb') as f:
            # 统一转为灰度图 'L'，节省3倍内存
            img = Image.open(f).convert('L')
            # 转为 numpy 数组 (28, 28)
            return np.array(img, dtype=np.uint8), label
    except:
        return None

class InMemoryDataset(Dataset):
    """
    [Tensor 优化版] 
    将所有图片存储为一个巨大的 ByteTensor。
    优势：PyTorch Tensor 支持多进程共享内存，Windows 下 num_workers > 0 不会卡死！
    """
    def __init__(self, root_dir, cache_path='./dataset_tensor_cache.pt'):
        self.data = None # 形状 [N, 28, 28] (uint8)
        self.labels = []
        self.classes = []
        
        # 1. 尝试加载缓存
        if os.path.exists(cache_path):
            print(f"[极速加载] 发现 Tensor 缓存 {cache_path}，正在加载...")
            try:
                # weights_only=False 是为了兼容旧逻辑，但在安全环境通常没问题
                saved_data = torch.load(cache_path, weights_only=False) 
                self.data = saved_data['data']
                self.labels = saved_data['labels']
                self.classes = saved_data['classes']
                print(f"[完成] 已加载 {len(self.data)} 张图片。")
                return
            except Exception as e:
                print(f"[警告] 缓存加载失败: {e}")

        # 2. 从硬盘读取
        print(f"[初始化] 正在扫描文件列表: {root_dir}")
        temp_dataset = datasets.ImageFolder(root_dir)
        self.classes = temp_dataset.classes
        samples = temp_dataset.samples 
        
        print(f"[并行读取] 开启多线程读取并转换为 Tensor...")
        
        # 使用 list 暂存 numpy 数组
        img_list = []
        label_list = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            results = list(tqdm(
                executor.map(load_image_file, samples), 
                total=len(samples),
                desc="Reading & Converting",
                unit="img"
            ))

        for item in results:
            if item is not None:
                img_list.append(item[0])
                label_list.append(item[1])
        
        # 3. 核心步骤：转换为 PyTorch Tensor
        print("[内存优化] 正在将数据打包为共享 Tensor...")
        # stack 后形状: [N, 28, 28]
        self.data = torch.tensor(np.array(img_list), dtype=torch.uint8)
        self.labels = torch.tensor(label_list, dtype=torch.long)
        
        print(f"[缓存] 保存 Tensor 缓存到 {cache_path}...")
        torch.save({
            'data': self.data,
            'labels': self.labels,
            'classes': self.classes
        }, cache_path)
        print("[完成] 数据集构建完毕！")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 返回 Tensor (28, 28) 和 label
        # 注意：这里不转 float，等到 TransformSubset 里转 PIL 再处理，效率最高
        return self.data[idx], self.labels[idx]

# --- 2. 辅助函数 ---

def get_supcon_transforms(size=28, mode='train'):
    if mode == 'train':
        return transforms.Compose([
            # 输入已经是 'L' (灰度) 的 PIL
            # transforms.Grayscale(num_output_channels=1), # 已经是灰度了，这步可以省
            transforms.RandomResizedCrop(size=size, scale=(0.5, 1.0)),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.4, contrast=0.4)
            ], p=0.8),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

# --- 3. 主加载函数 ---

def get_dataloaders(data_dir, batch_size=1024, val_split=0.2, num_workers=4):
    # 缓存文件名改一下，避免和之前的冲突
    cache_file = os.path.join(os.path.dirname(data_dir), '5G_NIDD_tensor_cache.pt')
    
    train_transform = TwoCropTransform(get_supcon_transforms(mode='train'))
    val_transform = get_supcon_transforms(mode='test')

    # 加载数据集
    full_dataset = InMemoryDataset(root_dir=data_dir, cache_path=cache_file)
    num_classes = len(full_dataset.classes)
    
    # 划分
    dataset_size = len(full_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
    train_subset, val_subset = random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )

    train_dataset = TransformSubset(train_subset, transform=train_transform)
    val_dataset = TransformSubset(val_subset, transform=val_transform)

    # 关键修改：现在可以安全开启 num_workers 了！
    # 建议设置为 CPU 核心数，例如 4 或 8
    print(f"[Info] DataLoader num_workers set to: {num_workers}")
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=(num_workers > 0)
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0)
    )

    return train_loader, val_loader, num_classes

if __name__ == '__main__':
    # 测试块
    DATA_PATH = './data/5G-NIDD/processed_images'
    if os.path.exists(DATA_PATH):
        get_dataloaders(DATA_PATH, batch_size=1024, num_workers=4)