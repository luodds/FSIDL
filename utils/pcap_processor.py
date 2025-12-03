import os
import numpy as np
import pandas as pd
from scapy.all import PcapReader, IP, TCP, UDP
from PIL import Image
from tqdm import tqdm
import glob
import math

# ================= 配置区域 =================
DATASET_ROOT = "data/5G-NIDD"
RAW_PCAP_DIR = os.path.join(DATASET_ROOT, "raw_pcap")
LABEL_DIR = os.path.join(DATASET_ROOT, "labels")
OUTPUT_DIR = os.path.join(DATASET_ROOT, "processed_images")

IMG_SIZE = (28, 28)
TOTAL_BYTES = 28 * 28
MAX_PACKETS_PER_FLOW = 16
BYTES_PER_PACKET = math.ceil(TOTAL_BYTES / MAX_PACKETS_PER_FLOW) 

PROTO_MAP = {
    'tcp': 6, 'udp': 17, 'icmp': 1, 'ipv6-icmp': 58,
    '6': 6, '17': 17, '1': 1
}
# ===========================================

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_flow_key(src_ip, dst_ip, src_port, dst_port, proto):
    return f"{src_ip}_{dst_ip}_{int(src_port)}_{int(dst_port)}_{int(proto)}"

def clean_and_load_df(csv_path):
    try:
        df = pd.read_csv(csv_path, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        
        possible_labels = ['Attack Type', 'Label', 'Class', 'AttackType']
        label_col = None
        for col in possible_labels:
            if col in df.columns:
                label_col = col
                break
        
        if not label_col:
            print(f"[!] 错误: 在 {os.path.basename(csv_path)} 中未找到标签列。")
            return None

        col_map = {
            'SrcAddr': ['SrcAddr', 'SrcIP', 'sIP'],
            'DstAddr': ['DstAddr', 'DstIP', 'dIP'],
            'Sport': ['Sport', 'SrcPort', 'sPort'],
            'Dport': ['Dport', 'DstPort', 'dPort'],
            'Proto': ['Proto', 'Protocol']
        }
        
        final_cols = {}
        for key, candidates in col_map.items():
            found = False
            for cand in candidates:
                if cand in df.columns:
                    final_cols[key] = cand
                    found = True
                    break
            if not found:
                print(f"[!] 错误: 缺少关键列 {key}")
                return None

        df = df[[final_cols['SrcAddr'], final_cols['DstAddr'], 
                 final_cols['Sport'], final_cols['Dport'], 
                 final_cols['Proto'], label_col]].copy()
        
        df.columns = ['SrcAddr', 'DstAddr', 'Sport', 'Dport', 'Proto', 'Label']
        
        df['Sport'] = df['Sport'].fillna(0)
        df['Dport'] = df['Dport'].fillna(0)
        
        def map_proto(p):
            p_str = str(p).lower().strip()
            return PROTO_MAP.get(p_str, 0)
            
        df['Proto'] = df['Proto'].apply(map_proto)
        df['Sport'] = pd.to_numeric(df['Sport'], errors='coerce').fillna(0).astype(int)
        df['Dport'] = pd.to_numeric(df['Dport'], errors='coerce').fillna(0).astype(int)

        return df
        
    except Exception as e:
        print(f"[!] 读取 CSV 异常: {e}")
        return None

def load_labels_from_csv(csv_path):
    print(f"[-] 正在加载标签文件: {os.path.basename(csv_path)} ...")
    df = clean_and_load_df(csv_path)
    if df is None or len(df) == 0: return {}

    label_map = {}
    count = 0
    for _, row in df.iterrows():
        if row['Proto'] == 0: continue
        key = get_flow_key(row['SrcAddr'], row['DstAddr'], row['Sport'], row['Dport'], row['Proto'])
        label_map[key] = row['Label']
        count += 1

    print(f"[-] 标签加载完成，有效流记录: {count}")
    return label_map

def bytes_to_image(byte_data, save_path):
    if len(byte_data) < TOTAL_BYTES:
        padding = [0] * (TOTAL_BYTES - len(byte_data))
        byte_data.extend(padding)
    else:
        byte_data = byte_data[:TOTAL_BYTES]
        
    arr = np.array(byte_data, dtype=np.uint8).reshape(IMG_SIZE)
    img = Image.fromarray(arr, mode='L')
    img.save(save_path)

def save_flow_image(flow_buffer_bytes, label, flow_key, img_index):
    """辅助函数：保存图片"""
    safe_label = "".join([c if c.isalnum() else "_" for c in str(label)])
    save_dir = os.path.join(OUTPUT_DIR, safe_label)
    ensure_dir(save_dir)
    
    # 文件名增加 index，支持同一个流生成多张图片
    save_path = os.path.join(save_dir, f"{flow_key}_{img_index}.png")
    bytes_to_image(flow_buffer_bytes, save_path)

def process_pcap(pcap_path, label_map):
    filename = os.path.basename(pcap_path)
    print(f"[*] 开始处理 PCAP: {filename}")
    
    # 格式: { flow_key: [bytes] }
    flow_buffer = {}
    # 格式: { flow_key: int_count } 记录每个流已经生成了多少张图
    flow_img_counts = {}
    
    total_img_count = 0
    
    try:
        with PcapReader(pcap_path) as pcap_reader:
            for pkt in tqdm(pcap_reader, desc="解析数据包", unit="pkt"):
                if not pkt.haslayer(IP): continue
                
                src_ip = pkt[IP].src
                dst_ip = pkt[IP].dst
                proto = pkt[IP].proto
                
                src_port = 0
                dst_port = 0
                if pkt.haslayer(TCP):
                    src_port = pkt[TCP].sport
                    dst_port = pkt[TCP].dport
                elif pkt.haslayer(UDP):
                    src_port = pkt[UDP].sport
                    dst_port = pkt[UDP].dport
                
                flow_key = get_flow_key(src_ip, dst_ip, src_port, dst_port, proto)
                label = label_map.get(flow_key)
                
                if label is None:
                    reverse_key = get_flow_key(dst_ip, src_ip, dst_port, src_port, proto)
                    label = label_map.get(reverse_key)
                    if label is not None:
                        flow_key = reverse_key
                
                if label is None: continue 
                
                if flow_key not in flow_buffer:
                    flow_buffer[flow_key] = []
                    flow_img_counts[flow_key] = 0
                
                # 提取 Payload
                raw_bytes = list(bytes(pkt[IP].payload))
                if len(raw_bytes) > BYTES_PER_PACKET:
                    raw_bytes = raw_bytes[:BYTES_PER_PACKET]
                else:
                    raw_bytes.extend([0] * (BYTES_PER_PACKET - len(raw_bytes)))
                
                flow_buffer[flow_key].extend(raw_bytes)
                
                # 【改动1：切片模式】
                # 只要攒够 784 字节，就立刻保存一张图，并清空 buffer 继续攒
                if len(flow_buffer[flow_key]) >= TOTAL_BYTES:
                    # 取前784字节保存
                    data_to_save = flow_buffer[flow_key][:TOTAL_BYTES]
                    save_flow_image(data_to_save, label, flow_key, flow_img_counts[flow_key])
                    
                    # 更新计数器
                    flow_img_counts[flow_key] += 1
                    total_img_count += 1
                    
                    # 保留剩余的数据 (如果有的话)，继续攒下一张
                    flow_buffer[flow_key] = flow_buffer[flow_key][TOTAL_BYTES:]

        # 【改动2：尾部清理】
        # 循环结束后，检查 buffer 里是否有残留数据
        # 如果有数据（哪怕只有几十字节），也补零保存为一张图
        print("[-] 正在处理剩余的短流/尾部数据...")
        for key, data in flow_buffer.items():
            if len(data) > 0:
                # 只有当这个流之前没生成过图片（由于太短），或者我们要保留长流的尾巴时
                # 这里策略：所有残余都保存，确保不丢数据
                save_flow_image(data, label_map[key], key, flow_img_counts[key])
                total_img_count += 1

    except Exception as e:
        print(f"[!] PCAP 处理中断: {e}")
        import traceback
        traceback.print_exc()
        
    print(f"[*] {filename} 处理完成。累计生成: {total_img_count} 张图片。")

def find_matching_csv(pcap_name, csv_files):
    pcap_lower = pcap_name.lower()
    for csv_path in csv_files:
        csv_base = os.path.basename(csv_path)
        # 简单匹配: 去掉后缀，去掉 '1' (针对 Goldeneye1.csv)
        csv_stem = csv_base.split('.')[0].replace('1', '').lower()
        if csv_stem in pcap_lower:
            return csv_path
    return None

def main():
    ensure_dir(OUTPUT_DIR)
    pcap_files = glob.glob(os.path.join(RAW_PCAP_DIR, "**/*.pcap*"), recursive=True)
    csv_files = glob.glob(os.path.join(LABEL_DIR, "**/*.csv"), recursive=True)
    
    print(f"发现 {len(pcap_files)} 个 PCAP 文件，{len(csv_files)} 个 CSV 文件。")
    
    for pcap_path in pcap_files:
        pcap_name = os.path.basename(pcap_path)
        csv_path = find_matching_csv(pcap_name, csv_files)
        
        if csv_path:
            print(f"\n>>> 正在配对: {pcap_name} <---> {os.path.basename(csv_path)}")
            label_map = load_labels_from_csv(csv_path)
            if not label_map:
                print("[!] 标签映射为空，跳过。")
                continue
            process_pcap(pcap_path, label_map)
        else:
            print(f"\n[!] 未找到匹配的 CSV: {pcap_name}")

if __name__ == "__main__":
    main()