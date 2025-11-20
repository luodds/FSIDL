import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_network_data(file_path, target_column='Attack Typ'):
    """
    对网络流量数据集进行探索性数据分析 (EDA)。

    参数:
    file_path (str): 数据集文件的路径 (例如 'your_dataset.csv').
    target_column (str): 用于标识攻击类型的目标列名。
    """
    print(f"🚀 开始分析数据集: {file_path}")

    # --- 1. 加载数据 ---
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"❌ 错误: 文件未找到 at '{file_path}'. 请检查路径是否正确。")
        return
    except Exception as e:
        print(f"❌ 加载文件时发生错误: {e}")
        return

    # 创建一个目录来保存图表
    if not os.path.exists('analysis_plots'):
        os.makedirs('analysis_plots')
    print("📊 图表将保存在 'analysis_plots/' 目录下。")

    # --- 2. 数据基本信息 ---
    print("\n" + "="*50)
    print("📋 1. 数据基本信息")
    print("="*50)
    print(f"数据集维度 (行, 列): {df.shape}")
    print("\n前5行数据:")
    print(df.head())
    print("\n数据类型和非空值计数:")
    # 使用 .info() 来检查数据类型和缺失值
    df.info(verbose=True, show_counts=True)


    # --- 3. 目标变量分析 (攻击类型分布) ---
    print("\n" + "="*50)
    print(f"📈 2. 攻击类型分布 ('{target_column}')")
    print("="*50)
    if target_column not in df.columns:
        print(f"❌ 警告: 目标列 '{target_column}' 不在数据集中。请检查列名。")
        # 尝试使用'Label'作为备用
        if 'Label' in df.columns:
            target_column = 'Label'
            print(f"ℹ️ 切换到备用目标列: '{target_column}'")
        else:
            print("❌ 无法找到任何标签列，跳过与标签相关的分析。")
            return

    attack_counts = df[target_column].value_counts()
    print("各类别的样本数量:")
    print(attack_counts)

    plt.figure(figsize=(12, 7))
    sns.barplot(x=attack_counts.index, y=attack_counts.values, palette='viridis')
    plt.title(f'Distribution of Attack Types ({target_column})', fontsize=16)
    plt.xlabel('Attack Type', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('analysis_plots/1_attack_type_distribution.png')
    plt.show()

    # --- 4. 数值型特征描述性统计 ---
    print("\n" + "="*50)
    print("🔢 3. 数值型特征描述性统计")
    print("="*50)
    # 选择部分关键的数值特征进行展示
    numerical_features = df.select_dtypes(include=np.number).columns.tolist()
    print(df[numerical_features].describe().transpose())


    # --- 5. 关键特征分布可视化 ---
    print("\n" + "="*50)
    print("📊 4. 关键特征分布可视化")
    print("="*50)

    # a. 协议分布
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Proto', order=df['Proto'].value_counts().index, palette='rocket')
    plt.title('Protocol Distribution', fontsize=16)
    plt.xlabel('Protocol', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.savefig('analysis_plots/2_protocol_distribution.png')
    plt.show()

    # b. 流持续时间 (Dur) 分布 (使用对数尺度，因为它可能高度倾斜)
    plt.figure(figsize=(10, 6))
    # 添加一个很小的值以避免log(0)
    sns.histplot(np.log1p(df['Dur']), kde=True, bins=50)
    plt.title('Distribution of Flow Duration (log scale)', fontsize=16)
    plt.xlabel('Log(1 + Duration)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.savefig('analysis_plots/3_duration_distribution.png')
    plt.show()


    # --- 6. 特征与攻击类型的关系 ---
    print("\n" + "="*50)
    print("🔗 5. 特征与攻击类型的关系")
    print("="*50)

    # a. 不同攻击类型下的协议使用情况
    plt.figure(figsize=(14, 8))
    sns.countplot(data=df, x=target_column, hue='Proto', palette='magma')
    plt.title('Protocol Usage by Attack Type', fontsize=16)
    plt.xlabel('Attack Type', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title='Protocol')
    plt.tight_layout()
    plt.savefig('analysis_plots/4_protocol_vs_attack.png')
    plt.show()

    # b. 不同攻击类型下的流持续时间 (Dur) 对比
    plt.figure(figsize=(14, 8))
    # 同样使用对数尺度
    df['log_dur'] = np.log1p(df['Dur'])
    sns.boxplot(data=df, x=target_column, y='log_dur', palette='coolwarm')
    plt.title('Flow Duration (log scale) vs. Attack Type', fontsize=16)
    plt.xlabel('Attack Type', fontsize=12)
    plt.ylabel('Log(1 + Duration)', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('analysis_plots/5_duration_vs_attack.png')
    plt.show()

    # c. 不同攻击类型下总字节数 (TotBytes) 对比
    plt.figure(figsize=(14, 8))
    df['log_totbytes'] = np.log1p(df['TotBytes'])
    sns.boxplot(data=df, x=target_column, y='log_totbytes', palette='plasma')
    plt.title('Total Bytes (log scale) vs. Attack Type', fontsize=16)
    plt.xlabel('Attack Type', fontsize=12)
    plt.ylabel('Log(1 + Total Bytes)', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('analysis_plots/6_bytes_vs_attack.png')
    plt.show()

    # --- 7. 相关性分析 ---
    print("\n" + "="*50)
    print("🎨 6. 数值特征相关性热力图")
    print("="*50)
    # 选择一部分特征进行相关性分析，避免图像过于拥挤
    corr_features = ['Dur', 'TotPkts', 'TotBytes', 'SrcPkts', 'SrcBytes', 'Rate', 'sMeanPkt', 'TcpRtt']
    # 确保这些特征在数据集中存在
    corr_features = [f for f in corr_features if f in df.columns]
    
    if len(corr_features) > 1:
        correlation_matrix = df[corr_features].corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
        plt.title('Correlation Matrix of Key Numerical Features', fontsize=16)
        plt.tight_layout()
        plt.savefig('analysis_plots/7_correlation_heatmap.png')
        plt.show()
    else:
        print("ℹ️ 可用于相关性分析的特征不足。")

    print("\n✅ 分析完成！")


if __name__ == '__main__':
    # ===============================================================
    # 请将 'your_dataset.csv' 替换为您的数据集的实际文件路径
    # ===============================================================
    file_path = 'w/Goldeneye1.csv'
    
    # ===============================================================
    # 请将 'Attack Typ' 替换为您的数据集中表示攻击类型的列名
    # 如果没有，可以使用 'Label' 等
    # ===============================================================
    target_column = 'Attack Typ'

    analyze_network_data(file_path, target_column)