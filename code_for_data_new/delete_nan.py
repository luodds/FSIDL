import os
import pandas as pd
import numpy as np

# 待处理CSV文件所在的文件夹路径
directory = 'G:/项目/新信令/项目文档/中期节点/数据集/中期测试数据集/指标2数据集/data_pre'

# 创建一个新的文件夹来保存处理后的数据
output_directory = 'G:/项目/新信令/项目文档/中期节点/数据集/中期测试数据集/指标2数据集/data_have_peocessed'

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

for filename in os.listdir(directory):
    if filename.endswith('.csv'):  # 仅处理CSV文件
        # 获取文件完整路径
        file_path = os.path.join(directory, filename)
        # 读取CSV数据
        data = pd.read_csv(file_path)

        # 在这里添加你需要的数据处理代码
        # 例如：删除无用列
        columns_to_drop = ['Label']
        data = data.drop(columns=columns_to_drop)
        # 寻找并删除包含异常值和空值的行
        print("删除空值前：",data.shape)
        data = data.dropna()  # 删除包含空值的行
        print("删除空值后：", data.shape)

        # 将处理后的数据保存到新的文件夹
        output_file_path = os.path.join(output_directory, filename)
        data.to_csv(output_file_path, index=False)
