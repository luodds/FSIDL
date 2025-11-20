"""
preprocess_5gnidd.py

统一预处理 5G-NIDD 数据集（如 BTS1_BTS2_fields_preserved.csv 或 Encoded.csv），
输出：
  - data/processed/5gnidd_all.joblib: X_train, X_val, X_test, y_train, y_val, y_test
  - data/processed/num_imputer.joblib, cat_imputer.joblib, ohe.joblib, scaler.joblib
  - data/processed/5gnidd_meta.json: 标签映射、特征列信息

后续所有方法（本地、集中、联邦、小样本等）都只需要读取这些处理好的数据。
"""

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from scipy import sparse


# ========= 配置区：请根据你的实际情况修改 =========

# 原始 CSV 路径：你刚才运行用的是这个路径
RAW_CSV_PATH = "data/5G-NIDD_ALL/BTS1_BTS2_fields_preserved.csv"
# 如果想用 Encoded.csv，可以改成：
# RAW_CSV_PATH = "data/5G-NIDD_ALL/Encoded.csv"

# 标签列候选名（脚本会从中自动寻找实际存在的一列作为标签）
LABEL_COL_CANDIDATES = [
    "Attack Type",  # 你目前数据中的标签列
    "label",
    "Label",
    "attack",
    "Attack",
    "class",
    "Class",
]

# 输出目录
OUTPUT_DIR = "data/processed"

# train/val/test 比例
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# 是否使用稀疏矩阵存储（One-Hot 之后维度可能很大，建议 True）
USE_SPARSE = True


# ========= 工具函数 =========

def find_label_column(df: pd.DataFrame, candidates):
    """在候选名里查找真实存在的标签列"""
    for name in candidates:
        if name in df.columns:
            print(f"[信息] 使用列 '{name}' 作为标签列。")
            return name
    raise ValueError(
        f"在数据集中没有找到指定的标签列候选名。\n"
        f"候选: {candidates}\n"
        f"实际前若干列: {list(df.columns)[:20]} ..."
    )


def drop_useless_columns(df: pd.DataFrame, label_col: str):
    """去掉全空列、常数列（但保留标签列）"""
    print("[信息] 正在删除全部为空的列...")
    df = df.dropna(axis=1, how="all")

    print("[信息] 正在删除常数列（所有行取值相同）...")
    nunique = df.nunique(dropna=False)
    const_cols = nunique[(nunique <= 1) & (nunique.index != label_col)].index.tolist()
    if const_cols:
        preview = const_cols[:10]
        print(f"[信息] 共发现 {len(const_cols)} 个常数列，将被删除。示例: {preview} ...")
        df = df.drop(columns=const_cols)
    else:
        print("[信息] 未发现常数列。")

    return df


def split_features_labels(df: pd.DataFrame, label_col: str):
    """拆分特征和标签"""
    y_str = df[label_col].astype(str)
    df_features = df.drop(columns=[label_col])
    return df_features, y_str


def separate_num_cat(df_features: pd.DataFrame):
    """区分数值列和类别列（同时尝试将能转为数值的 object 列自动转型）"""
    num_cols = df_features.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df_features.select_dtypes(include=["object", "bool"]).columns.tolist()

    # 尝试将某些 object 列转为数字
    extra_num_cols = []
    for col in cat_cols:
        try:
            df_features[col].astype(float)
            extra_num_cols.append(col)
        except Exception:
            pass

    if extra_num_cols:
        print(f"[信息] 检测到 {len(extra_num_cols)} 个原本是 object 的列可以转换为数值，将其作为数值特征处理。示例: {extra_num_cols[:10]} ...")
        for col in extra_num_cols:
            df_features[col] = pd.to_numeric(df_features[col], errors="coerce")

    # 重新识别数据类型
    num_cols = df_features.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df_features.select_dtypes(include=["object", "bool"]).columns.tolist()

    print(f"[信息] 最终识别的数值特征列数: {len(num_cols)}")
    print(f"[信息] 最终识别的类别特征列数: {len(cat_cols)}")

    return df_features, num_cols, cat_cols


def preprocess_features(df_features, num_cols, cat_cols):
    """
    数值列：SimpleImputer(median) + StandardScaler
    类别列：SimpleImputer(most_frequent) + OneHotEncoder
    返回：X（稀疏或稠密）、预处理器对象
    """
    # 数值列处理
    num_imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    if num_cols:
        print("[信息] 正在对数值特征列进行缺失值填充（中位数）...")
        X_num = num_imputer.fit_transform(df_features[num_cols])
        print("[信息] 正在对数值特征列进行标准化（StandardScaler）...")
        X_num = scaler.fit_transform(X_num)
    else:
        print("[警告] 未检测到数值特征列，将不使用数值特征。")
        X_num = None

    # 类别列处理
    if cat_cols:
        print("[信息] 正在对类别特征列进行缺失值填充（众数）...")
        cat_imputer = SimpleImputer(strategy="most_frequent")
        df_cat = pd.DataFrame(
            cat_imputer.fit_transform(df_features[cat_cols]),
            columns=cat_cols
        )

        print("[信息] 正在对类别特征列进行 One-Hot 编码...")
        # 适配 sklearn >= 1.2 的新参数 sparse_output
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        except TypeError:
            # 兼容旧版本
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)

        X_cat = ohe.fit_transform(df_cat)
    else:
        print("[信息] 未检测到类别特征列，将不使用类别特征。")
        cat_imputer = None
        ohe = None
        X_cat = None

    # 组合数值和类别特征
    if X_num is not None and X_cat is not None:
        if USE_SPARSE:
            print("[信息] 正在将数值（稠密）和类别（稀疏）特征拼接为稀疏矩阵...")
            X = sparse.hstack([X_num, X_cat]).tocsr()
        else:
            print("[信息] 正在将数值和类别特征拼接为稠密矩阵（注意内存占用可能较大）...")
            X = np.hstack([X_num, X_cat.toarray()])
    elif X_num is not None:
        print("[信息] 仅使用数值特征。")
        X = X_num
    elif X_cat is not None:
        print("[信息] 仅使用类别特征。")
        X = X_cat if USE_SPARSE else X_cat.toarray()
    else:
        raise ValueError("清洗后没有剩余特征，无法继续训练，请检查数据。")

    preprocessors = {
        "num_imputer": num_imputer,
        "cat_imputer": cat_imputer,
        "ohe": ohe,
        "scaler": scaler,
    }

    return X, preprocessors


def stratified_train_val_test_split(X, y, train_ratio, val_ratio, test_ratio, random_state=42):
    """
    按照类别分层划分 train/val/test
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio 必须等于 1。")

    print("[信息] 正在按比例划分训练集 / 验证集 / 测试集 ...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(1 - train_ratio),
        random_state=random_state,
        stratify=y
    )

    val_fraction_of_temp = val_ratio / (val_ratio + test_ratio)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_fraction_of_temp),
        random_state=random_state,
        stratify=y_temp
    )

    print(f"[信息] 训练集大小: {len(y_train)}, 验证集大小: {len(y_val)}, 测试集大小: {len(y_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test,
                        preprocessors, label2id, num_cols, cat_cols,
                        output_dir):
    """
    保存数据、预处理器和元信息
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[信息] 正在将处理后的数据保存到目录: {output_dir} ...")
    data_obj = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }
    joblib.dump(data_obj, output_dir / "5gnidd_all.joblib")

    # 保存预处理器
    joblib.dump(preprocessors["num_imputer"], output_dir / "num_imputer.joblib")
    joblib.dump(preprocessors["scaler"], output_dir / "scaler.joblib")
    if preprocessors["cat_imputer"] is not None:
        joblib.dump(preprocessors["cat_imputer"], output_dir / "cat_imputer.joblib")
    if preprocessors["ohe"] is not None:
        joblib.dump(preprocessors["ohe"], output_dir / "ohe.joblib")

    # 保存元数据（标签映射 + 特征列信息）
    meta = {
        "label2id": label2id,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "use_sparse": USE_SPARSE,
    }
    with open(output_dir / "5gnidd_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[信息] 处理后的数据和元信息已全部保存完成。")


# ========= 主流程 =========

def main():
    print(f"[信息] 正在读取原始 CSV 文件: {RAW_CSV_PATH}")
    if not os.path.exists(RAW_CSV_PATH):
        raise FileNotFoundError(f"未找到 RAW_CSV_PATH 指定的文件: {RAW_CSV_PATH}")

    # low_memory=False 可以减少 dtype 警告，并一次性读入
    df = pd.read_csv(RAW_CSV_PATH, low_memory=False)
    print(f"[信息] 原始数据维度: {df.shape} (行, 列)")

    # 找标签列
    label_col = find_label_column(df, LABEL_COL_CANDIDATES)

    # 丢掉全空 & 常数列
    df = drop_useless_columns(df, label_col=label_col)

    # 特征/标签拆分
    df_features, y_str = split_features_labels(df, label_col=label_col)

    # 标签映射成整型 ID
    label_names = sorted(y_str.unique())
    label2id = {name: i for i, name in enumerate(label_names)}
    y = y_str.map(label2id).values
    print(f"[信息] 标签类别数: {len(label_names)}")
    print(f"[信息] 标签映射示例: {list(label2id.items())[:5]} ...")

    # 区分数值列和类别列，并做预处理
    df_features, num_cols, cat_cols = separate_num_cat(df_features)
    X, preprocessors = preprocess_features(df_features, num_cols, cat_cols)

    # train/val/test 划分
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_train_val_test_split(
        X, y, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
    )

    # 保存结果
    save_processed_data(
        X_train, X_val, X_test, y_train, y_val, y_test,
        preprocessors, label2id, num_cols, cat_cols,
        OUTPUT_DIR
    )

    print("[信息] 全部预处理流程执行完毕。")


if __name__ == "__main__":
    main()
 