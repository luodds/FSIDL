import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import sys
sys.path.append('/home/liuziyi/cub')

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
    kmeans = KMeans(n_clusters=2, random_state=0)
    cluster_labels = kmeans.fit_predict(prototypes)
    return cluster_labels

parser = get_command_line_parser()
args = postprocess_args(parser.parse_args())
dataset = MiniImageNet('train',args)

dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 调用 Res12 函数并传递所需的参数
backbone = Res12(keep_prob=1.0, avg_pool=True, drop_rate=0.1, dropblock_size=5).to(device)

# Step 1: 提取所有n个类别、所有样本的特征
features = extract_features(backbone, dataloader)

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
# 读取CSV文件，假设第二列是标签列
csv_path = "/home/liuziyi/cub/data/miniimagenet/split/train_test.csv"
data_frame = pd.read_csv(csv_path)
# 获取第二列的标签值
labels_column = data_frame.iloc[:, 1]  # 假设第二列的索引是1

# 获取每种类型的唯一标签值
unique_labels = labels_column.unique()

print("Unique labels:")
print(unique_labels)

# 原始数组
array1 = unique_labels
array2 = cluster_labels

# 将数组按照聚类结果分成两堆
cluster_0_indices = np.where(array2 == 0)[0]
cluster_1_indices = np.where(array2 == 1)[0]
#cluster_2_indices = np.where(array2 == 2)[0]
#cluster_3_indices = np.where(array2 == 3)[0]
# 根据索引从原始数组中获取对应的值
cluster_0_values = array1[cluster_0_indices]
cluster_1_values = array1[cluster_1_indices]
#cluster_2_values = array1[cluster_2_indices]
#cluster_3_values = array1[cluster_3_indices]
print("Cluster 0 values:", cluster_0_values)
print("Cluster 1 values:", cluster_1_values)
#print("Cluster 2 values:", cluster_2_values)
#print("Cluster 3 values:", cluster_3_values)
# print('Subset A:', subset_a)
# print('Subset B:', subset_b)
# print('Subset C:', subset_c)