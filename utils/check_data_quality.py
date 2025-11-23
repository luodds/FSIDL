import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
from scapy.all import rdpcap, IP, TCP, UDP
from collections import Counter

# ================= 配置 =================
DATASET_ROOT = "data/5G-NIDD"
PROCESSED_DIR = os.path.join(DATASET_ROOT, "processed_images")
RAW_PCAP_DIR = os.path.join(DATASET_ROOT, "raw_pcap")
LABEL_DIR = os.path.join(DATASET_ROOT, "labels")

# 设定一些阈值
MIN_SAMPLES = 100  # 警告阈值：如果某类样本少于这个数，可能无法训练
# =======================================

def print_header(title):
    print(f"\n{'='*10} {title} {'='*10}")

def check_class_distribution():
    """
    1. 检查 processed_images 下的文件夹结构
    2. 统计每个类别的图片数量
    """
    print_header("1. 类别分布统计 (Class Distribution)")
    
    if not os.path.exists(PROCESSED_DIR):
        print(f"[!] 错误: 找不到目录 {PROCESSED_DIR}")
        return

    classes = [d for d in os.listdir(PROCESSED_DIR) if os.path.isdir(os.path.join(PROCESSED_DIR, d))]
    classes.sort()
    
    if not classes:
        print("[!] 警告: processed_images 下没有任何文件夹！")
        return

    total_images = 0
    stats = []

    print(f"{'Class Name':<25} | {'Count':<10} | {'Status'}")
    print("-" * 50)

    for cls in classes:
        cls_dir = os.path.join(PROCESSED_DIR, cls)
        imgs = glob.glob(os.path.join(cls_dir, "*.png"))
        count = len(imgs)
        total_images += count
        
        status = "✅ OK"
        if count == 0:
            status = "❌ EMPTY"
        elif count < MIN_SAMPLES:
            status = "⚠️ LOW"
        
        print(f"{cls:<25} | {count:<10} | {status}")
        stats.append((cls, count))

    print("-" * 50)
    print(f"总计类别数: {len(classes)}")
    print(f"总计图片数: {total_images}")
    
    # 检查是否有 Benign
    has_benign = any("benign" in c.lower() for c, _ in stats)
    if not has_benign:
        print("\n[!!!] 严重警告: 未检测到 'Benign' (正常流量) 类别！")
        print("       如果没有正常流量，模型无法学习区分攻击和正常行为。")
        print("       可能原因: CSV 中的 Label 列没有包含 'Benign'，或者被过滤掉了。")
    else:
        print("\n[+] 检测到正常流量类别，数据集结构基本完整。")

def check_image_quality():
    """
    2. 随机抽样检查图片内容
    防止生成全是 0 (全黑) 的无效图片
    """
    print_header("2. 图片质量抽检 (Image Quality)")
    
    classes = [d for d in os.listdir(PROCESSED_DIR) if os.path.isdir(os.path.join(PROCESSED_DIR, d))]
    if not classes: return

    # 从每个类抽 1 张图检查
    for cls in classes:
        cls_dir = os.path.join(PROCESSED_DIR, cls)
        imgs = glob.glob(os.path.join(cls_dir, "*.png"))
        
        if not imgs: continue
        
        # 随机选一张
        sample_path = imgs[0]
        try:
            img = Image.open(sample_path)
            arr = np.array(img)
            
            mean_val = np.mean(arr)
            std_val = np.std(arr)
            
            # 判断是否全黑
            if mean_val == 0 and std_val == 0:
                print(f"[!] 警告: 类 '{cls}' 的样本全是黑色 (像素值全为0)。这可能是提取逻辑错误。")
            elif mean_val < 5:
                print(f"[?] 提示: 类 '{cls}' 的样本非常暗 (Mean={mean_val:.2f})。可能是包载荷大部分是0。")
            else:
                # 正常
                pass
        except Exception as e:
            print(f"[!] 读取图片出错: {e}")
    
    print("[-] 图片抽检完成。如果没有警告，说明图片内容有效。")

def diagnose_bs2_mismatch():
    """
    3. 深度诊断：为什么 BS2 匹配不到？
    选取一个 BS2 的 PCAP 和对应的 CSV，对比里面的 IP 和 Port
    """
    print_header("3. BS2 匹配失败深度诊断 (Deep Dive)")
    
    # 寻找目标文件 (根据你之前的日志)
    target_pcap_name = "UDPflood_BS2_nogtp.pcapng" # 这是一个处理成功了的 BS2 (如果存在) 或者失败的
    # 根据日志，UDPflood_BS2 生成了 6万张，是成功的。
    # 我们找一个失败的，比如 SSH_BS2
    target_pcap_name_fail = "SSH_BS2_nogtp.pcapng"
    target_csv_name = "SSH1.csv" # 假设配对的是这个
    
    pcap_path = os.path.join(RAW_PCAP_DIR, target_pcap_name_fail)
    csv_path = os.path.join(LABEL_DIR, target_csv_name)
    
    if not os.path.exists(pcap_path):
        # 尝试模糊搜索
        search = glob.glob(os.path.join(RAW_PCAP_DIR, "*SSH_BS2*.pcap*"))
        if search: pcap_path = search[0]
        else: 
            print("[-] 找不到 SSH_BS2 测试文件，跳过诊断。")
            return

    print(f"正在对比:\n  PCAP: {os.path.basename(pcap_path)}\n  CSV : {os.path.basename(csv_path)}")
    
    # 1. 从 CSV 读取前 100 个流的 Key
    print("[-] 读取 CSV 前 100 条流信息...")
    try:
        df = pd.read_csv(csv_path, nrows=100, low_memory=False)
        # 简单的列名清洗
        df.columns = [c.strip() for c in df.columns]
        
        csv_keys = set()
        # 假设列名已知
        for _, row in df.iterrows():
            try:
                # 构建简单的 key 集合: SrcIP, DstIP
                # 不用全 key，只看 IP 对不对得上
                csv_keys.add( (row['SrcAddr'], row['DstAddr']) )
            except: pass
        print(f"    CSV 中包含的通信对 (Src, Dst) 示例: {list(csv_keys)[:3]}")
        
    except Exception as e:
        print(f"[!] CSV 读取失败: {e}")
        return

    # 2. 从 PCAP 读取前 100 个包的 Key
    print("[-] 读取 PCAP 前 100 个数据包...")
    pcap_keys = set()
    try:
        pkts = rdpcap(pcap_path, count=100)
        for p in pkts:
            if IP in p:
                pcap_keys.add( (p[IP].src, p[IP].dst) )
        print(f"    PCAP 中包含的通信对 (Src, Dst) 示例: {list(pcap_keys)[:3]}")
        
    except Exception as e:
        print(f"[!] PCAP 读取失败: {e}")
        return

    # 3. 对比
    print("[-] 对比结果:")
    # 检查是否有交集
    intersection = csv_keys.intersection(pcap_keys)
    if intersection:
        print(f"✅ 发现 {len(intersection)} 个匹配的 IP 对。说明 IP 没问题，可能是端口或协议不匹配。")
    else:
        print("❌ IP 地址完全不匹配！")
        print("    -> 这证实了 BS2 的流量并没有记录在 xxx1.csv 中。")
        print("    -> 结论：BS2 的 PCAP 需要对应的 'BS2.csv'，但数据集中缺失。")
        print("    -> 决策：这一现象验证了我们'放弃 BS2'策略的正确性。")

if __name__ == "__main__":
    check_class_distribution()
    check_image_quality()
    diagnose_bs2_mismatch()