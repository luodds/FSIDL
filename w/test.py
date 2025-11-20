import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import time

def preprocess_data(file_path):
    """
    加载、清洗并预处理网络流量数据集。
    (增加了基于分析的显式特征选择)
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
        
    # --- 1. 定义要移除的列 ---
    columns_to_drop = [
        # 标识符和高基数特征
        'SrcId', 'SrcAddr', 'DstAddr', 'SrcMac', 'DstMac', 'SrcOui', 'DstOui',
        'sIpId', 'dIpId', 'IdleTime', 'AckDat', 'DstTCPBase', 'SrcTCPBase', 
        'TcpRtt', 'dHops', 'sHops', 'dTtl', 'Seq', 'Rank', 'Offset', 
        'dMeanPktSz', 'Sport', 'Dport', 'Dur', 'Max', 'Mean', 
        'Sum', 'Min', 'DstWin', 'sTtl', 'sVid', 'SynAck', 'DstLoss', 
        'sMeanPktSz',
        
        # 冗余或直接相关的特征
        'RunTime', 'Label', 'Attack Tool', 'attack_cat',
        
        # 几乎没有变化或大部分为空/零值的特征 (基于样本观察)
        'sCo', 'dCo', 'sMpls', 'dMpls', 'Cause', 'NStrok', 'sNStrok', 'dNStrok',
        'PCRatio', # 这个特征通常在很多场景下是0
        
        # 原始时间戳
        'StartTime', 'LastTime'
    ]
    
    # 过滤掉数据集中不存在的列名，避免出错
    existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
    
    print(f"\n计划移除 {len(existing_cols_to_drop)} 个特征...")
    df.drop(columns=existing_cols_to_drop, inplace=True)
    print("特征移除完成。")

    # --- 2. 处理目标标签 ---
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=[target_column], inplace=True)
    if df.shape[0] == 0:
        print(f"❌ 错误: 数据集变为空。")
        return None, None, None, None
        
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[target_column])
    
    # 从特征集中移除目标列
    df.drop(columns=[target_column], inplace=True)
    
    # 保存真实的特征名用于后续分析
    original_features = df.columns.tolist()
    print(f"保留 {len(original_features)} 个特征用于训练。")

    # --- 3. 处理特征数据类型和缺失值 ---
    # 将 object 类型的列强制转换为数值，无法转换的变为 NaN
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 用中位数填充所有剩余的NaN，这通常比用0更稳健
    df.fillna(df.median(), inplace=True)
    print(f"所有特征已转换为数值类型并填充缺失值。")
    
    # --- 4. 特征标准化 ---
    scaler = StandardScaler()
    X = scaler.fit_transform(df)
    
    print(f"\n预处理完成。最终特征矩阵维度: {X.shape}")
    print("-------------------------------------------\n")
    return X, y, label_encoder, original_features

def main():
    """主函数，用于训练和评估基线模型"""
    file_path = 'w/merged_all_attacks_process.csv'
    
    # 1. 数据准备
    X, y, label_encoder, feature_names = preprocess_data(file_path)

    if X is None:
        return

    # 2. 划分数据集 (使用标准方法)
    print("--- 步骤 2: 划分训练集和测试集 ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"训练集维度: {X_train.shape}")
    print(f"测试集维度: {X_test.shape}")
    print("-------------------------------------\n")

    # 3. 初始化并训练 LightGBM 模型
    print("--- 步骤 3: 训练 LightGBM 模型 ---")
    
    lgb_classifier = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=len(label_encoder.classes_),
        n_estimators=1000,  # 初始设置较多树，通过早停找到最佳数量
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        colsample_bytree=0.8, # 特征采样
        subsample=0.8       # 数据采样
    )

    start_time = time.time()
    
    # 使用早停法 (Early Stopping) 来防止过拟合
    lgb_classifier.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='multi_logloss',
        callbacks=[lgb.early_stopping(100, verbose=True)]
    )
    
    end_time = time.time()
    print(f"模型训练完成。耗时: {end_time - start_time:.2f} 秒")
    print("----------------------------------\n")

    # 4. 在测试集上评估模型
    print("--- 步骤 4: 评估模型性能 ---")
    y_pred = lgb_classifier.predict(X_test)
    
    # 准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ 整体准确率: {accuracy * 100:.2f}%\n")
    
    # 分类报告 (包含精确率、召回率、F1分数)
    print("📊 分类报告:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # 5. 可视化结果
    print("--- 步骤 5: 可视化结果 ---")
    
    # 绘制混淆矩阵 (这部分代码不变)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix - LightGBM Baseline')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("confusion_matrix_lgbm.png")
    print("🔢 混淆矩阵已保存为 'confusion_matrix_lgbm.png'")
    plt.show()

    # ==================================================================
    # |              >>>>>> 以下是修改后的代码 <<<<<<                  |
    # ==================================================================
    
    # 绘制带真实名称的特征重要性图
    print("⭐ 正在生成带真实名称的特征重要性图...")

    # 1. 创建一个包含特征名和重要性分数的DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': lgb_classifier.feature_importances_
    }).sort_values('importance', ascending=False)

    # 2. 筛选出最重要的前30个特征用于绘图
    top_features_df = importance_df.head(30)

    # 3. 使用Seaborn绘制条形图，更加美观
    plt.figure(figsize=(12, 10))
    sns.barplot(x='importance', y='feature', data=top_features_df, palette='viridis')
    
    # 添加数值标签
    for index, value in enumerate(top_features_df['importance']):
        plt.text(value, index, f' {value}', va='center')
        
    plt.title('Top 30 Feature Importance - LightGBM', fontsize=16)
    plt.xlabel('Feature Importance', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.tight_layout()
    plt.savefig("feature_importance_lgbm_named.png")
    print("⭐ 带有真实名称的特征重要性图已保存为 'feature_importance_lgbm_named.png'")
    plt.show()


if __name__ == '__main__':
    main()
    