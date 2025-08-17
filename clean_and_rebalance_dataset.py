import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- 配置参数 ---
# 初始要移除的异常样本数量
n_outliers = 10

# 数据集划分比例
VAL_SIZE = 0.2

# 随机种子，确保每次划分结果一致
RANDOM_STATE = 42

# --- 文件路径定义 ---
BASE_DIR = '/home/irving/workspace/CEMP_data/'
INPUT_DIR = os.path.join(BASE_DIR, 'split_data_denoized')
OUTPUT_DIR = os.path.join(BASE_DIR, 'split_data_cleaned_and_rebalanced')


def load_and_merge_set(set_name):
    """加载并合并指定数据集（train/val/test）的光谱和标签"""
    print(f"- 正在加载 {set_name} 数据集...")
    spectra_path = os.path.join(INPUT_DIR, set_name, 'spectra.csv')
    labels_path = os.path.join(INPUT_DIR, set_name, 'labels.csv')
    
    if not os.path.exists(spectra_path) or not os.path.exists(labels_path):
        print(f"  警告：在 {set_name} 目录中找不到 spectra.csv 或 labels.csv，将跳过。")
        return None

    spectra_df = pd.read_csv(spectra_path)
    labels_df = pd.read_csv(labels_path)
    
    if 'obsid' in labels_df.columns and 'obsid' in spectra_df.columns:
        return pd.merge(spectra_df, labels_df, on='obsid')
    else:
        return pd.concat([spectra_df, labels_df], axis=1)

def save_distribution_plots(labels_df, output_dir, set_name):
    """为关键标签生成并保存分布图。"""
    print(f"- 正在为 {set_name} 集生成分布图...")
    features_to_plot = ['Teff', 'logg', 'FeH', 'CFe']
    
    sns.set_theme(style="whitegrid")

    for feature in features_to_plot:
        if feature not in labels_df.columns:
            print(f"  - 警告: 在标签中未找到 '{feature}' 列，跳过绘图。")
            continue
        
        plt.figure(figsize=(10, 6))
        sns.histplot(data=labels_df, x=feature, kde=True, bins=30)
        plt.title(f'{feature} Distribution in {set_name.capitalize()} Set', fontsize=16)
        plt.xlabel(feature, fontsize=12)
        plt.ylabel('Count', fontsize=12)
        
        plot_path = os.path.join(output_dir, f'{feature}_distribution.png')
        plt.savefig(plot_path)
        plt.close() # 释放内存
    print(f"- {set_name} 集的分布图已保存。")

def main():
    """主执行函数"""
    # --- 1. 数据整合 ---
    print("步骤 1: 开始整合所有数据集...")
    all_sets = ['train', 'val', 'test']
    df_list = [load_and_merge_set(s) for s in all_sets]
    
    df_list = [df for df in df_list if df is not None]
    if not df_list:
        print("错误：未能加载任何数据，请检查输入目录结构和文件。")
        return

    full_df = pd.concat(df_list, ignore_index=True)
    print(f"数据整合完成。总样本数: {len(full_df)}\n")

    # --- 2. 异常点剔除 (带条件) ---
    print("步骤 2: 使用隔离森林进行异常检测...")
    features_for_detection = ['Teff', 'logg', 'FeH', 'CFe']
    X_detection = full_df[features_for_detection].copy()
    X_detection.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_detection.fillna(X_detection.median(), inplace=True)

    contamination = n_outliers / len(X_detection) if len(X_detection) > 0 else 0
    iso_forest = IsolationForest(n_estimators=100, contamination=contamination, random_state=RANDOM_STATE)
    outlier_predictions = iso_forest.fit_predict(X_detection)
    full_df['is_outlier'] = outlier_predictions

    num_before_cleaning = len(full_df)
    cleaned_df = full_df[(full_df['is_outlier'] == 1) | (full_df['FeH'] < -3)].copy()
    num_after_cleaning = len(cleaned_df)
    print(f"异常检测完成。移除了 {num_before_cleaning - num_after_cleaning} 个样本。")
    print(f"清洗后总样本数: {num_after_cleaning}\n")

    # --- 3. 分层重新划分 ---
    print("步骤 3: 根据FeH值进行分层，重新划分数据集...")
    feh_bins = [-np.inf, -3, -2, -1, np.inf]
    feh_labels = ['<-3', '-3_to_-2', '-2_to_-1', '>=-1']
    cleaned_df['feh_bin'] = pd.cut(cleaned_df['FeH'], bins=feh_bins, labels=feh_labels, right=False)

    print("各FeH分层中的样本数量:")
    print(cleaned_df['feh_bin'].value_counts().sort_index())

    label_cols = ['obsid', 'Teff', 'logg', 'FeH', 'CFe', 'feh_bin']
    label_cols_exist = [col for col in label_cols if col in cleaned_df.columns]
    labels = cleaned_df[label_cols_exist]
    spectra = cleaned_df.drop(columns=label_cols_exist)

    train_spectra, val_spectra, train_labels, val_labels = train_test_split(
        spectra, labels, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=labels['feh_bin']
    )
    print(f"\n数据集已按 {1-VAL_SIZE:.0%}:{VAL_SIZE:.0%} 的比例重新划分为训练集和验证集。")
    print(f"新训练集样本数: {len(train_spectra)}")
    print(f"新验证集样本数: {len(val_spectra)}\n")

    # --- 4. 保存结果和分布图 ---
    print("步骤 4: 保存新的数据集和分布图...")
    for set_name, s_df, l_df in [('train', train_spectra, train_labels), ('val', val_spectra, val_labels)]:
        output_set_dir = os.path.join(OUTPUT_DIR, set_name)
        os.makedirs(output_set_dir, exist_ok=True)
        
        # 保存CSV文件
        s_df.to_csv(os.path.join(output_set_dir, 'spectra.csv'), index=False)
        l_df.drop(columns=['feh_bin']).to_csv(os.path.join(output_set_dir, 'labels.csv'), index=False)
        print(f"- 已成功保存 {set_name} 数据到: {output_set_dir}")
        
        # 生成并保存分布图
        save_distribution_plots(l_df, output_set_dir, set_name)

    print("\n处理完成！")

if __name__ == '__main__':
    main()