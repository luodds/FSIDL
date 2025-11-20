import pandas as pd
import os

directory = 'G:/毕业/论文与代码/元学习梯度下降/MAML-CNN-FSIDS-IoT-main/dataset/IDS2018/data_have_processed'    # 替换为你的文件夹路径
output_directory = 'G:/毕业/论文与代码/元学习梯度下降/MAML-CNN-FSIDS-IoT-main/dataset/IDS2018/data_100' # 替换为你想放置输出文件的文件夹路径


category_label = 33  # 文件种类标签的初始值设为1

for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        df = pd.read_csv(os.path.join(directory, filename))
        df_head = df.head(100)

        # 对于文件中的每一行，添加标签值
        for i in range(len(df_head)):
            seq_label = str(i + 1).zfill(5)  # 序列标签值，用零填充到5位
            df_head.loc[i, 'dataLabel'] = str(category_label).zfill(2) + seq_label  # 设定"Label"列的值

        df_head.to_csv(os.path.join(output_directory, filename), index=False)  # 保存数据到新的CSV文件

        category_label += 1  # 处理完一个文件后，种类标签值加1



directory = 'G:/毕业/论文与代码/元学习梯度下降/MAML-CNN-FSIDS-IoT-main/dataset/IDS2018/data_100'  # 替换为你的文件夹路径
output_filename = 'G:/毕业/论文与代码/元学习梯度下降/MAML-CNN-FSIDS-IoT-main/dataset/IDS2018/data_100/merge_ids2018.csv'  # 替换为你想给合并后的文件起的名字

df_list = []  # 用于存储每个csv文件的DataFrame

for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        df = pd.read_csv(os.path.join(directory, filename))
        df_list.append(df)

merged_df = pd.concat(df_list, ignore_index=True)

merged_df.to_csv(output_filename, index=False)
