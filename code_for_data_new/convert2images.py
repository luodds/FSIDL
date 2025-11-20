import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_normalize(data):
    normalized = sigmoid(data - np.mean(data))  # 减去均值
    normalized /= np.max(np.abs(normalized))  # 归一化
    return normalized

def data_process(input_folder, output_folder):
    # 获取输入文件夹中的所有文件
    file_list = os.listdir(input_folder)
    for idx, file_name in enumerate(file_list):
        # 构建输入文件的完整路径
        file_path = os.path.join(input_folder, file_name)

        # 读取CSV文件
        df = pd.read_csv(file_path, encoding='ISO-8859-1')

        # # 排除不需要的列
        # excluded = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Protocol', 'Timestamp', 'Label']
        # df = df.drop(columns=excluded)

        # 删除包含无穷值的行
        df = df[np.isfinite(df.T).all()]
        print(df.shape)
        df = df.iloc[:260]
        print(df.shape)

        # 将Pandas DataFrame对象转换为Numpy数组
        x = df.to_numpy()
        x = np.pad(x, ((0, 0), (0, 81 - x.shape[1])), mode='constant')

        # 对数组进行处理
        x = sigmoid(x)
        x = sigmoid_normalize(x)

        # 将数组映射到(0, 255)之间
        x_mapped = np.interp(x, (x.min(), x.max()), (0, 255)).astype(np.uint8)

        a=f"{idx+203:02}"
        # 将每一行转换为一个新的二维数组并绘制图像
        for i in range(x_mapped.shape[0]):
            row = x_mapped[i, :]
            row_2d = row.reshape((9, 9))  # 将一维数组转换为二维数组
            plt.imshow(row_2d, cmap='gray')
            # plt.title('Row {:05d}'.format(i + 1))  # 使用'{:05d}'进行5位数字格式补零

            # 构建输出图像的文件名，并保存在输出文件夹中
            output_filename = os.path.join(output_folder, a+'{:05d}.png'.format(i + 133))
            plt.savefig(output_filename)
            plt.close()  # 关闭图像绘制，防止图像重叠显示


input = 'G:/项目/新信令/项目文档/中期节点/数据集/中期测试数据集/指标2数据集/data'
output = "G:/项目/新信令/项目文档/中期节点/数据集/中期测试数据集/指标2数据集/data_images/203"
data_process(input, output)





