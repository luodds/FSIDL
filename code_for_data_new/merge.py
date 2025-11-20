import os
import pandas as pd

# 设置文件夹路径
directory = "G:/项目/新信令/小样本模型数据/3threat_1normalv2.csv"

# 找到文件夹下的所有CSV文件
csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

# 初始化一个空的DataFrame用于存放数据
all_data = pd.DataFrame()

# 遍历CSV文件并将每个文件的数据添加到all_data中
for file in csv_files:
    print(file)
    path = os.path.join(directory, file)
    print(path)
    current_data = pd.read_csv(path,encoding='ISO-8859-1')
    all_data = pd.concat([all_data, current_data])

# 打印输出合并后的数据大小
print(f"Merged data size: {all_data.shape}")

# 将合并后的数据保存到新的CSV文件，这里暂定保存路径为同一文件夹下的 "merged.csv"
output_file = os.path.join(directory, "merged.csv")
all_data.to_csv(output_file, index=False)

