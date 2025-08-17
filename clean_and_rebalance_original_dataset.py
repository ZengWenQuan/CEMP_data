import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- 配置参数 ---
n_outliers = 10
VAL_SIZE = 0.2
RANDOM_STATE = 42

# --- 文件路径定义 ---
BASE_DIR = '/home/irving/workspace/CEMP_data/'
INPUT_DIR = os.path.join(BASE_DIR, 'split_data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'split_data_original_cleaned_and_rebalanced')


def load_data_from_sets(set_names):
    """从所有数据集中加载并合并所有需要的文件。"""
    all_labels, all_normalized, all_continuum = [], [], []
    for set_name in set_names:
        print(f"- 正在加载 {set_name} 数据集...")
        labels_path = os.path.join(INPUT_DIR, set_name, 'labels.csv')
        normalized_path = os.path.join(INPUT_DIR, set_name, 'normalized.csv')
        continuum_path = os.path.join(INPUT_DIR, set_name, 'continuum.csv')
        
        if not all(os.path.exists(p) for p in [labels_path, normalized_path, continuum_path]):
            print(f"  警告：在 {set_name} 目录中文件不完整，将跳过。" )
            continue
        
        all_labels.append(pd.read_csv(labels_path))
        all_normalized.append(pd.read_csv(normalized_path))
        all_continuum.append(pd.read_csv(continuum_path))
        
    if not all_labels:
        return None, None, None

    return pd.concat(all_labels, ignore_index=True), \
           pd.concat(all_normalized, ignore_index=True), \
           pd.concat(all_continuum, ignore_index=True)

def create_dual_channel_features(normalized_df, continuum_df):
    """将两个光谱数据帧交错合并成双通道特征。"""
    assert normalized_df.shape[1] == continuum_df.shape[1], "光谱和连续谱的点数必须相同"
    num_samples, num_points = normalized_df.shape
    dual_channel_data = np.empty((num_samples, num_points * 2))
    dual_channel_data[:, 0::2] = normalized_df.values
    dual_channel_data[:, 1::2] = continuum_df.values
    new_columns = [f'point_{i}_{ch}' for i in range(num_points) for ch in ['norm', 'cont']]
    return pd.DataFrame(dual_channel_data, columns=new_columns)

def save_distribution_plots(labels_df, output_dir, set_name):
    print(f"- 正在为 {set_name} 集生成分布图...")
    features_to_plot = ['Teff', 'logg', 'FeH', 'CFe']
    sns.set_theme(style="whitegrid")
    for feature in features_to_plot:
        if feature not in labels_df.columns: continue
        plt.figure(figsize=(10, 6))
        sns.histplot(data=labels_df, x=feature, kde=True, bins=30)
        plt.title(f'{feature} Distribution in {set_name.capitalize()} Set', fontsize=16)
        plt.savefig(os.path.join(output_dir, f'{feature}_distribution.png'))
        plt.close()
    print(f"- {set_name} 集的分布图已保存。" )

def main():
    # 步骤 1: 加载数据
    print("步骤 1: 加载所有数据集的全部文件...")
    labels_df, normalized_df, continuum_df = load_data_from_sets(['train', 'val', 'test'])
    if labels_df is None: return
    print(f"数据加载完成。总样本数: {len(labels_df)}")
    print(f"  - DEBUG: 初始 a. normalized_df shape: {normalized_df.shape}")
    print(f"  - DEBUG: 初始 b. continuum_df shape: {continuum_df.shape}\n")

    # 修正：确保归一化和连续谱数据帧具有相同的形状
    if 'obsid' in normalized_df.columns and 'obsid' in continuum_df.columns:
        print("  - DEBUG: 发现obsid列，将在合并前移除")
        normalized_df = normalized_df.drop(columns=['obsid'])
        continuum_df = continuum_df.drop(columns=['obsid'])

    if normalized_df.shape[1] != continuum_df.shape[1]:
        print(f"警告：归一化光谱 ({normalized_df.shape[1]}) 和连续谱 ({continuum_df.shape[1]}) 的列数不匹配。" )
        min_cols = min(normalized_df.shape[1], continuum_df.shape[1])
        print(f"为保持一致性，将两个数据帧都裁剪为前 {min_cols} 列。" )
        
        # 获取要保留的列名
        norm_cols = normalized_df.columns[:min_cols]
        cont_cols = continuum_df.columns[:min_cols]
        
        normalized_df = normalized_df[norm_cols]
        continuum_df = continuum_df[cont_cols]
        
        print(f"  - DEBUG: 裁剪后 a. normalized_df shape: {normalized_df.shape}")
        print(f"  - DEBUG: 裁剪后 b. continuum_df shape: {continuum_df.shape}\n")

    # 保存原始光谱数据的列名，以便后续使用
    original_flux_columns = continuum_df.columns.tolist()

    # 步骤 2: 连续谱归一化
    print("步骤 2: 对连续谱进行Min-Max归一化...")
    print(f"  - DEBUG: 用于训练scaler的continuum_df shape: {continuum_df.shape}")
    scaler = MinMaxScaler()
    continuum_scaled_values = scaler.fit_transform(continuum_df)
    continuum_scaled_df = pd.DataFrame(continuum_scaled_values, columns=continuum_df.columns)
    print(f"  - DEBUG: scaler.n_features_in_: {scaler.n_features_in_}")
    print(f"  - DEBUG: continuum_scaled_df shape: {continuum_scaled_df.shape}")
    print("连续谱归一化完成。\n")

    # 步骤 3: 创建双通道特征
    print("步骤 3: 创建双通道光谱特征...")
    print(f"  - DEBUG: 用于创建双通道的 a. normalized_df shape: {normalized_df.shape}")
    print(f"  - DEBUG: 用于创建双通道的 b. continuum_scaled_df shape: {continuum_scaled_df.shape}")
    dual_channel_spectra_df = create_dual_channel_features(normalized_df, continuum_scaled_df)
    spectral_colnames = dual_channel_spectra_df.columns.tolist() # 保存光谱列名用于后续精确提取
    print(f"  - DEBUG: dual_channel_spectra_df shape: {dual_channel_spectra_df.shape}")
    full_df = pd.concat([labels_df, dual_channel_spectra_df], axis=1)
    print("双通道特征创建完成。\n")

    # 步骤 4: 异常点剔除
    print("步骤 4: 使用隔离森林进行异常检测...")
    features_for_detection = ['Teff', 'logg', 'FeH', 'CFe']
    X_detection = full_df[features_for_detection].copy()
    X_detection.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_detection.fillna(X_detection.median(), inplace=True)
    contamination = n_outliers / len(X_detection) if len(X_detection) > 0 else 0
    iso_forest = IsolationForest(n_estimators=100, contamination=contamination, random_state=RANDOM_STATE)
    full_df['is_outlier'] = iso_forest.fit_predict(X_detection)
    num_before = len(full_df)
    cleaned_df = full_df[(full_df['is_outlier'] == 1) | (full_df['FeH'] < -3)].copy()
    print(f"异常检测完成。移除了 {num_before - len(cleaned_df)} 个样本。清洗后总样本数: {len(cleaned_df)}\n")

    # 步骤 5: 分层重新划分
    print("步骤 5: 根据FeH值进行分层，重新划分数据集...")
    feh_bins = [-np.inf, -3, -2, -1, np.inf]
    cleaned_df['feh_bin'] = pd.cut(cleaned_df['FeH'], bins=feh_bins, labels=['<-3', '-3_to_-2', '-2_to_-1', '>=-1'], right=False)
    label_cols = ['obsid', 'Teff', 'logg', 'FeH', 'CFe', 'feh_bin']
    label_cols_exist = [col for col in label_cols if col in cleaned_df.columns]
    final_labels = cleaned_df[label_cols_exist]
    # 使用之前保存的光谱列名来精确提取光谱数据，避免drop的不可靠性
    final_spectra = cleaned_df[spectral_colnames]
    print(f"  - DEBUG: final_spectra shape: {final_spectra.shape}")
    train_spectra, val_spectra, train_labels, val_labels = train_test_split(
        final_spectra, final_labels, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=final_labels['feh_bin']
    )
    print("数据集划分完成。\n")

    # 步骤 6: 拆分双通道数据并分别保存
    print("步骤 6: 保存新的数据集和分布图...")
    for set_name, s_df, l_df in [('train', train_spectra, train_labels), ('val', val_spectra, val_labels)]:
        output_set_dir = os.path.join(OUTPUT_DIR, set_name)
        os.makedirs(output_set_dir, exist_ok=True)
        
        print(f"- 正在处理和保存 {set_name} 数据集...")
        
        # 1. 从双通道数据中分离归一化光谱和（已缩放的）连续谱
        normalized_split_df = s_df[s_df.columns[0::2]]
        continuum_scaled_split_df = s_df[s_df.columns[1::2]]
        
        # 2. 对连续谱部分进行反归一化
        print(f"  - 正在对 {set_name} 集的连续谱进行反归一化...")
        continuum_original_values = scaler.inverse_transform(continuum_scaled_split_df)
        
        # 3. 将光谱数据恢复为带有原始列名的DataFrame
        normalized_split_df = pd.DataFrame(normalized_split_df.values, columns=original_flux_columns)
        continuum_original_df = pd.DataFrame(continuum_original_values, columns=original_flux_columns)
        
        # 4. 将obsid重新加回光谱文件
        if 'obsid' in l_df.columns:
            # 确保索引对齐
            l_df_reset = l_df.reset_index(drop=True)
            normalized_split_df = pd.concat([l_df_reset['obsid'], normalized_split_df.reset_index(drop=True)], axis=1)
            continuum_original_df = pd.concat([l_df_reset['obsid'], continuum_original_df.reset_index(drop=True)], axis=1)

        # 5. 保存三个独立的文件：labels, normalized, continuum
        labels_to_save = l_df.drop(columns=['feh_bin'])
        
        labels_to_save.to_csv(os.path.join(output_set_dir, 'labels.csv'), index=False)
        normalized_split_df.to_csv(os.path.join(output_set_dir, 'normalized.csv'), index=False)
        continuum_original_df.to_csv(os.path.join(output_set_dir, 'continuum.csv'), index=False)
        
        print(f"- 已成功保存 {set_name} 的三个文件到: {output_set_dir}")
        
        # 6. 保存分布图
        save_distribution_plots(l_df, output_set_dir, set_name)

    print("\n处理完成！")

if __name__ == '__main__':
    main()