import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import f_classif, mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_data_for_analysis(file_path):
    """
    加载并预处理数据，专门用于相关性分析。
    (复用并优化我们之前版本的功能)
    """
    print("--- 步骤 1: 开始数据预处理 (包含特征选择) ---")
    
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except Exception as e:
        print(f"❌ 加载文件时发生错误: {e}")
        return None, None, None, None

    print(f"原始数据集维度: {df.shape}")

    # 定义目标列
    target_column = 'Attack Type'
    if target_column not in df.columns:
        print(f"❌ 错误: 目标列 '{target_column}' 不在CSV文件中。")
        return None, None, None, None
        
    # 定义要移除的列 (基于我们之前的分析)
    columns_to_drop = [
        'SrcId', 'SrcAddr', 'DstAddr', 'SrcMac', 'DstMac', 'SrcOui', 'DstOui',
        'sIpId', 'dIpId', 'RunTime', 'Label', 'Attack Tool', 'attack_cat',
        'sCo', 'dCo', 'sMpls', 'dMpls', 'Cause', 'NStrok', 'sNStrok', 'dNStrok',
        'PCRatio', 'StartTime', 'LastTime'
    ]
    
    existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
    df.drop(columns=existing_cols_to_drop, inplace=True)

    # 处理目标标签
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=[target_column], inplace=True)
    if df.shape[0] == 0:
        print(f"❌ 错误: 数据集变为空。")
        return None, None, None, None
        
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[target_column])
    df.drop(columns=[target_column], inplace=True)
    
    original_features = df.columns.tolist()
    print(f"保留 {len(original_features)} 个特征用于分析。")

    # 处理特征数据类型和缺失值
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 注意：对于相关性分析，填充缺失值可能会轻微影响结果，但必须处理。
    # 这里我们先用0填充，因为互信息对缩放不敏感，且能处理0值。
    df.fillna(0, inplace=True)
    
    X = df
    
    print(f"\n预处理完成。")
    print("-------------------------------------------\n")
    return X, y, label_encoder, original_features

def main():
    """主函数，用于执行特征相关性分析"""
    file_path = 'w/merged_all_attacks.csv'
    
    # 1. 数据准备
    # 注意：我们使用原始数值的X，而不是标准化的，这样结果更直观。
    X, y, label_encoder, feature_names = preprocess_data_for_analysis(file_path)

    if X is None:
        return

    print("--- 步骤 2: 计算特征与目标的相关性分数 ---")
    
    # --- 方法一: ANOVA F-test ---
    # f_classif会返回F值和p值，我们主要关心F值用于排序
    f_scores, _ = f_classif(X, y)
    
    # --- 方法二: 互信息 (Mutual Information) ---
    # 互信息计算可能需要一些时间
    print("正在计算互信息分数，这可能需要几分钟...")
    mi_scores = mutual_info_classif(X, y, random_state=42)
    print("计算完成。")
    
    # --- 3. 整理并展示结果 ---
    print("\n--- 步骤 3: 整理并展示分析结果 ---")
    
    # 创建一个DataFrame来存储所有结果
    results_df = pd.DataFrame({
        'Feature': feature_names,
        'F-Score': f_scores,
        'Mutual_Information': mi_scores
    })
    
    # 按F-Score排序并显示前20名
    f_score_ranking = results_df.sort_values('F-Score', ascending=False)
    print("\n========== 基于 ANOVA F-Score 的特征排名 (前20) ==========")
    print(f_score_ranking[['Feature', 'F-Score']].head(20).to_string(index=False))
    
    # 按互信息排序并显示前20名
    mi_ranking = results_df.sort_values('Mutual_Information', ascending=False)
    print("\n========== 基于互信息 (Mutual Information) 的特征排名 (前20) ==========")
    print(mi_ranking[['Feature', 'Mutual_Information']].head(20).to_string(index=False))
    
    # --- 4. 可视化结果 ---
    print("\n--- 步骤 4: 生成可视化图表 ---")
    
    # 可视化 F-Score
    plt.figure(figsize=(12, 10))
    top_f_scores = f_score_ranking.head(25)
    sns.barplot(x='F-Score', y='Feature', data=top_f_scores, palette='viridis')
    plt.title('Top 25 Features Ranked by ANOVA F-Score', fontsize=16)
    plt.xlabel('F-Score (越高越相关)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig("feature_correlation_f_score.png")
    print("📊 F-Score排名图已保存为 'feature_correlation_f_score.png'")
    plt.show()
    
    # 可视化 互信息
    plt.figure(figsize=(12, 10))
    top_mi_scores = mi_ranking.head(25)
    sns.barplot(x='Mutual_Information', y='Feature', data=top_mi_scores, palette='plasma')
    plt.title('Top 25 Features Ranked by Mutual Information', fontsize=16)
    plt.xlabel('Mutual Information (越高越相关)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig("feature_correlation_mutual_info.png")
    print("📊 互信息排名图已保存为 'feature_correlation_mutual_info.png'")
    plt.show()

if __name__ == '__main__':
    main()