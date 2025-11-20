import pandas as pd
import os
import glob


def merge_csv_in_directory(input_directory, output_file):
    """
    查找指定目录下的所有CSV文件，将它们合并，并保存到一个新的CSV文件中。

    这个函数假设所有的CSV文件都有相同的列标题（headers）。

    参数:
    input_directory (str): 包含多个CSV文件的目录路径。
    output_file (str): 合并后输出的CSV文件名。
    """
    print(f"🚀 开始合并CSV文件...")
    print(f"源目录: '{input_directory}'")
    print(f"目标文件: '{output_file}'")

    # 1. 构造一个路径模式来匹配所有CSV文件
    # os.path.join 会根据您的操作系统（Windows/Mac/Linux）正确地拼接路径
    csv_pattern = os.path.join(input_directory, '*.csv')

    # 2. 使用 glob 找到所有匹配模式的文件
    csv_files = glob.glob(csv_pattern)

    if not csv_files:
        print(f"❌ 在目录 '{input_directory}' 中没有找到任何CSV文件。请检查路径。")
        return

    print(f"📂 找到了 {len(csv_files)} 个CSV文件准备合并。")

    # 3. 循环读取每个CSV文件，并将它们的数据框（DataFrame）添加到一个列表中
    list_of_dataframes = []
    for filename in csv_files:
        print(f"  -> 正在读取: {os.path.basename(filename)}...")
        try:
            df = pd.read_csv(filename, low_memory=False)
            list_of_dataframes.append(df)
        except Exception as e:
            print(f"  -> ⚠️ 读取文件 {filename} 时出错: {e}。已跳过此文件。")

    if not list_of_dataframes:
        print("❌ 未能成功读取任何文件。合并中止。")
        return

    # 4. 使用 pandas.concat 将列表中的所有数据框合并成一个
    print("\n🔗 正在合并所有数据...")
    merged_df = pd.concat(list_of_dataframes, ignore_index=True)

    print(f"✅ 合并完成！")
    print(f"最终数据集有 {merged_df.shape[0]} 行和 {merged_df.shape[1]} 列。")

    # 5. 将合并后的数据框保存到输出文件
    # index=False 参数可以防止 pandas 将数据框的索引写入CSV的第一列
    try:
        print(f"💾 正在保存到 '{output_file}'...")
        merged_df.to_csv(output_file, index=False)
        print(f"🎉 成功！合并后的文件已保存为 '{output_file}'。")
    except Exception as e:
        print(f"❌ 保存文件时出错: {e}")

# ==============================================================================
# ---                            主程序入口                            ---
# ==============================================================================
if __name__ == '__main__':
    # --- 请在这里配置您的路径 ---

    # 1. 设置包含您所有CSV文件的目录路径
    #    例如: 'data/my_attacks' 或 'C:\\Users\\YourName\\Downloads\\CSVs'
    #    使用 '.' 代表当前脚本所在的目录
    INPUT_DIRECTORY = 'data/5G-NIDD/BS1_each_attack_csv'

    # 2. 设置您希望保存的合并后文件的名称
    OUTPUT_FILE = 'w/merged_all_attacks.csv'

    # --------------------------------

    merge_csv_in_directory(INPUT_DIRECTORY, OUTPUT_FILE)