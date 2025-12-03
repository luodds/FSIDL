import pandas as pd
import os

# 指向那个有问题的 CSV
CSV_PATH = "data/5G-NIDD/labels/BS2_each_attack_csv/SYNFlood2.csv"

def inspect():
    if not os.path.exists(CSV_PATH):
        print(f"❌ 文件不存在: {CSV_PATH}")
        # 尝试修正路径（防止路径写错）
        return

    print(f"正在分析: {os.path.basename(CSV_PATH)}")
    
    try:
        # 读取全部列
        df = pd.read_csv(CSV_PATH, low_memory=False)
        
        print(f"\n1. === 基本信息 ===")
        print(f"行数: {len(df)}")
        print(f"列名列表: {list(df.columns)}")
        
        print(f"\n2. === 关键列数据预览 (前5行) ===")
        # 尝试找几个关键列，看看名字是不是变了，或者数据格式是不是奇怪
        cols_to_check = ['SrcAddr', 'Sport', 'DstAddr', 'Dport', 'Proto', 'Label', 'Attack Type']
        existing_cols = [c for c in cols_to_check if c in df.columns]
        
        if existing_cols:
            print(df[existing_cols].head(5))
        else:
            print("⚠️ 没找到预期的关键列名！请检查上方的列名列表。")

        print(f"\n3. === 协议列 (Proto) 唯一值分布 ===")
        if 'Proto' in df.columns:
            print(df['Proto'].unique())
        else:
            print("列 'Proto' 不存在。")

        print(f"\n4. === 端口列 (Sport) 数据类型检查 ===")
        if 'Sport' in df.columns:
            print(f"Sport 类型: {df['Sport'].dtype}")
            print(f"Sport 前5个原始值: {df['Sport'].head(5).tolist()}")

    except Exception as e:
        print(f"❌ 读取出错: {e}")

if __name__ == "__main__":
    inspect()