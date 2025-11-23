import os
import pandas as pd
from scapy.all import rdpcap, IP, TCP, UDP

# === 配置路径 ===
# 请确保你的目录结构和我设定的一致
DATASET_ROOT = "data/5G-NIDD"
PCAP_DIR = os.path.join(DATASET_ROOT, "raw_pcap")
LABEL_DIR = os.path.join(DATASET_ROOT, "labels")

def print_separator(title):
    print(f"\n{'='*10} {title} {'='*10}")

def explore_csv(file_path):
    print(f"正在读取 CSV 文件: {os.path.basename(file_path)} ...")
    try:
        # 读取前几行，防止文件太大爆内存
        df = pd.read_csv(file_path, nrows=1000) 
        
        print(f"[-] 列名 (Columns): {list(df.columns)}")
        print(f"[-] 前 3 行数据预览:")
        print(df.head(3))
        
        # 尝试寻找可能的标签列
        possible_label_cols = [col for col in df.columns if 'label' in col.lower() or 'attack' in col.lower() or 'type' in col.lower()]
        if possible_label_cols:
            print(f"[-] 猜测标签列可能为: {possible_label_cols}")
            # 如果找到了，打印一下这一列的值
            for col in possible_label_cols:
                print(f"    -> 列 '{col}' 的前5个值: {df[col].unique()[:5]}")
        else:
            print("[-] ⚠️ 未找到明显的标签列，请手动检查列名。")

        # 尝试寻找关联 PCAP 的关键信息（IP, 时间戳）
        print("[-] 关键字段检查:")
        for key in ['ip', 'addr', 'time', 'date', 'seq']:
            matches = [c for c in df.columns if key in c.lower()]
            print(f"    -> 包含 '{key}' 的列: {matches}")
            
    except Exception as e:
        print(f"[!] 读取 CSV 失败: {e}")

def explore_pcap(file_path):
    print(f"正在读取 PCAP 文件: {os.path.basename(file_path)} ...")
    try:
        # 只读取前 5 个包
        packets = rdpcap(file_path, count=5)
        print(f"[-] 成功读取 {len(packets)} 个数据包。")
        
        if len(packets) > 0:
            first_pkt = packets[0]
            print(f"[-] 第 1 个包的摘要: {first_pkt.summary()}")
            print(f"[-] 第 1 个包的层级结构:")
            first_pkt.show()
            
            # 检查是否有 IP 层
            if IP in first_pkt:
                print(f"[-] ✅ 检测到 IP 层: Src={first_pkt[IP].src}, Dst={first_pkt[IP].dst}")
            else:
                print(f"[-] ⚠️ 未检测到标准 IP 层，可能是二层帧或有封装头。")
    except Exception as e:
        print(f"[!] 读取 PCAP 失败: {e}")

def main():
    print_separator("目录结构检查")
    
    # 1. 遍历 PCAP 文件夹
    pcap_files = []
    for root, dirs, files in os.walk(PCAP_DIR):
        for f in files:
            if f.endswith('.pcap') or f.endswith('.pcapng'):
                pcap_files.append(os.path.join(root, f))
    
    print(f"在 {PCAP_DIR} 下发现了 {len(pcap_files)} 个 PCAP 文件。")
    
    # 2. 遍历 Labels 文件夹
    csv_files = []
    for root, dirs, files in os.walk(LABEL_DIR):
        for f in files:
            if f.endswith('.csv'):
                csv_files.append(os.path.join(root, f))
    
    print(f"在 {LABEL_DIR} 下发现了 {len(csv_files)} 个 CSV 文件。")

    # 3. 采样分析 (只分析第一个找到的文件，节省时间)
    print_separator("采样分析 - CSV")
    if csv_files:
        explore_csv(csv_files[0])
    else:
        print("没有 CSV 文件，跳过。")

    print_separator("采样分析 - PCAP")
    if pcap_files:
        explore_pcap(pcap_files[0])
    else:
        print("没有 PCAP 文件，跳过。")

if __name__ == "__main__":
    main()