import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
from sklearn.cluster import KMeans
import sys
sys.path.append(r'/home/liuziyi/cub')

import os.path as osp
from networks.res12 import Res12
from dataloader.mini_imagenet import MiniImageNet
from trainer_single.utils import (
    pprint, set_gpu,
    get_command_line_parser,
    postprocess_args,
)
# Step 1: 提取所有n个类别、所有样本的特征
def extract_features(backbone, dataloader):
    features = []
    for images, _ in dataloader:
        with torch.no_grad():
            images = images.to(device)
            feature = backbone(images)
            features.append(feature.cpu().numpy())
    features = np.concatenate(features, axis=0)
    return features

# Step 2: 求每个类别样本特征的平均值，得到n个class prototypes
def compute_prototypes(features, all_labels):
    num_unique_labels = len(np.unique(all_labels))
    prototypes = np.zeros((num_unique_labels, features.shape[1]))

    for i in range(num_unique_labels):
        indices = np.where(all_labels == i)[0]
        class_features = features[indices]
        class_average = np.mean(class_features, axis=0)
        prototypes[i] = class_average

    return prototypes


# Step 3: 对这n个prototypes做K-means聚类，K=2，从而将n个类别划分成两部分
def kmeans_clustering(prototypes):
    kmeans = KMeans(n_clusters=5, random_state=0)
    cluster_labels = kmeans.fit_predict(prototypes)
    return cluster_labels

# Step 4: 根据类中心的划分结果，将所有类别中的数据划分成两部分
def split_classes(cluster_labels, labels, n_classes):
    subset_a = []
    subset_b = []
    subset_c = []
    for i in range(n_classes):
        if cluster_labels[i] == 0:
            subset_a.extend(np.where(labels == i)[0])
        else:
            subset_b.extend(np.where(labels == i)[0])
        return subset_a, subset_b

parser = get_command_line_parser()
args = postprocess_args(parser.parse_args())
dataset = MiniImageNet('train',args)

dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 调用 Res12 函数并传递所需的参数
backbone = Res12(keep_prob=1.0, avg_pool=True, drop_rate=0.1, dropblock_size=5).to(device)

# Step 1: 提取所有n个类别、所有样本的特征
features = extract_features(backbone, dataloader)
print(features.shape[1])
print("features:",features)
print(features.shape[1])
# 获取标签
# 创建一个空列表来存储所有标签
all_labels = []

# 遍历数据集
for images, labels in dataloader:
    # 将批次的标签添加到列表中
    all_labels.extend(labels.tolist())

# 将列表转换为NumPy数组
all_labels = np.array(all_labels)

# 打印获取到的标签数组
print("all_labels:"
      "", all_labels)

# 获取类别数量
# 使用numpy.unique来获取唯一值和出现次数
unique_labels, label_counts = np.unique(all_labels, return_counts=True)
print("unique_labels:",unique_labels)
# 打印唯一值和出现次数
for label, count in zip(unique_labels, label_counts):
    print(f"Label {label}: Count {count}")

# 获取标签种类数
num_unique_labels = len(unique_labels)
print(f"Number of unique labels: {num_unique_labels}")
# n_classes = len(dataset.classes)



# Step 2: 求每个类别样本特征的平均值，得到n个class prototypes
prototypes = compute_prototypes(features,  all_labels)
print(prototypes)

# Step 3: 对这n个prototypes做K-means聚类，K=3，从而将n个类别划分
cluster_labels = kmeans_clustering(prototypes)
print(cluster_labels)
# Step 4: 根据类中心的划分结果，将所有类别中的数据划分
#subset_a, subset_b = split_classes(cluster_labels, labels, num_unique_labels)

#print('Subset A:', subset_a)
#print('Subset B:', subset_b)
# print('Subset C:', subset_c)