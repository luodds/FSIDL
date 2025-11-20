import os
import os
import pandas as pd

# 读取merged.csv数据文件
all_data = pd.read_csv('G:/项目/新信令/项目文档/中期节点/数据集/中期测试数据集/指标2数据集/XL_test.csv')
# 获取所有的标签名称
labels = all_data["Label"].unique()

# 创建对应的文件夹（如果不存在）
directory = 'G:/项目/新信令/项目文档/中期节点/数据集/中期测试数据集/指标2数据集/data_pre'
if not os.path.exists(directory):
    os.makedirs(directory)

# 对每个标签进行操作
for label in labels:
    # 获取当前标签下的数据
    current_label_data = all_data[all_data["Label"] == label]

    # 将当前标签下的数据保存到对应的CSV文件，文件名设置为标签名
    output_file = os.path.join(directory, f"{label}.csv")
    current_label_data.to_csv(output_file, index=False)