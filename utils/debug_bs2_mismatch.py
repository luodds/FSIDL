import pandas as pd
from scapy.all import PcapReader, IP, TCP, UDP
import os

# === 这里硬编码一个生成了 0 张图片的典型案例 ===
# 请确保文件名和你硬盘上的一致
CSV_PATH = "data/5G-NIDD/labels/BS2_each_attack_csv/SYNFlood2.csv"   # 假设这是对应的 CSV
PCAP_PATH = "data/5G-NIDD/raw_pcap/BS2_GTP_removed/SYNflood_BS2_nogtp.pcapng" # 假设这是对应的 PCAP

# 如果你的文件路径不一样，请在这里修改！
# 比如你的 CSV 都在 labels/ 下，没有子文件夹，请去掉子文件夹
# 根据你之前的 log，CSV 似乎在 labels/ 下
# CSV_PATH = "data/5G-NIDD/labels/SYNFlood1.csv"
# PCAP_PATH = "data/5G-NIDD/raw_pcap/SYNflood_BS2_nogtp.pcapng" 

def get_key(src, dst, sport, dport, proto):
    return f"{src}_{dst}_{int(sport)}_{int(dport)}_{int(proto)}"

def debug():
    print(f"正在诊断:\nPCAP: {PCAP_PATH}\nCSV : {CSV_PATH}\n")

    # 1. 读取 CSV 的前 20 个 Key
    print("[-] 正在读取 CSV 生成指纹...")
    df = pd.read_csv(CSV_PATH)
    # 清洗列名
    df.columns = [c.strip() for c in df.columns]
    
    csv_keys = set()
    print("[-] CSV 中的前 10 个流指纹 (五元组):")
    for i, row in df.head(20).iterrows():
        try:
            # 处理协议
            proto = str(row['Proto']).lower().strip()
            if proto == 'tcp' or proto == '6': p_num = 6
            elif proto == 'udp' or proto == '17': p_num = 17
            else: continue # 忽略其他
            
            # 处理端口
            sport = int(float(row['Sport'])) if pd.notna(row['Sport']) else 0
            dport = int(float(row['Dport'])) if pd.notna(row['Dport']) else 0
            
            key = get_key(row['SrcAddr'], row['DstAddr'], sport, dport, p_num)
            csv_keys.add(key)
            
            if i < 10:
                print(f"    CSV: {key}  (Label: {row.get('Label', 'N/A')})")
        except Exception as e:
            pass

    print(f"[-] CSV 加载了 {len(csv_keys)} 个测试 Key。\n")

    # 2. 读取 PCAP 的前 20 个 Key
    print("[-] 正在读取 PCAP 生成指纹...")
    pcap_keys = set()
    match_count = 0
    reverse_match_count = 0
    
    try:
        with PcapReader(PCAP_PATH) as pcap_reader:
            for i, pkt in enumerate(pcap_reader):
                if i >= 20: break # 只看前20个包
                
                if not pkt.haslayer(IP): continue
                
                src = pkt[IP].src
                dst = pkt[IP].dst
                proto = pkt[IP].proto
                sport, dport = 0, 0
                
                if pkt.haslayer(TCP):
                    sport = pkt[TCP].sport
                    dport = pkt[TCP].dport
                elif pkt.haslayer(UDP):
                    sport = pkt[UDP].sport
                    dport = pkt[UDP].dport
                
                key = get_key(src, dst, sport, dport, proto)
                rev_key = get_key(dst, src, dport, sport, proto)
                
                print(f"    PCAP Pkt {i}: {key}")
                
                if key in csv_keys:
                    print(f"        -> ✅ 直接匹配成功!")
                    match_count += 1
                elif rev_key in csv_keys:
                    print(f"        -> 🔄 反向匹配成功!")
                    reverse_match_count += 1
                else:
                    print(f"        -> ❌ 匹配失败")

    except FileNotFoundError:
        print("找不到文件！请检查代码里的 CSV_PATH 和 PCAP_PATH 是否正确！")

if __name__ == "__main__":
    debug()