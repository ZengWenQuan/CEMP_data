"""
步骤 5: 光谱重采样与格式化

将处理后的光谱数据重采样到标准波长网格，并格式化为科学分析常用的表格形式。
"""
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def process_step(spectra_data, wavelength_config):
    """
    对光谱数据进行重采样，并返回两个格式化的DataFrame。

    Args:
        spectra_data (list): 包含待处理光谱字典的列表。
        wavelength_config (list): 定义波长网格的列表。

    Returns:
        tuple: 包含两个Pandas DataFrame (df_normalized, df_continuum)。
               每个DataFrame的索引是obsid，列是标准化的波长。
    """
    print(f"开始对 {len(spectra_data)} 个光谱进行重采样和格式化...")

    # 1. 生成最终的波长网格
    final_wavelength_grid = np.array([])
    for start, end, step in wavelength_config:
        segment_grid = np.arange(start, end, step)
        final_wavelength_grid = np.concatenate([final_wavelength_grid, segment_grid])
    
    # 将波长值格式化为字符串，作为DataFrame的列名
    column_names = [f"{wl:.2f}" for wl in final_wavelength_grid]
    print(f"  -> 已生成包含 {len(column_names)} 个数据点的最终波长网格。")

    # 准备用于存储所有样本数据的列表
    all_normalized_flux = []
    all_continuum_flux = []
    all_obsids = []

    for i, spec in enumerate(spectra_data):
        if (i + 1) % 10 == 0:
            print(f"  正在处理光谱 {i + 1}/{len(spectra_data)}...")
        
        original_wavelength = spec.get('wavelength_rest')
        flux_normalized = spec.get('flux_normalized')
        continuum = spec.get('continuum')

        if original_wavelength is None or flux_normalized is None or continuum is None:
            continue

        try:
            interp_func_normalized = interp1d(original_wavelength, flux_normalized, kind='linear', bounds_error=False, fill_value=np.nan)
            interp_func_continuum = interp1d(original_wavelength, continuum, kind='linear', bounds_error=False, fill_value=np.nan)

            resampled_flux_normalized = interp_func_normalized(final_wavelength_grid)
            resampled_continuum = interp_func_continuum(final_wavelength_grid)

            # --- 前向填充开头的NaN值 ---
            first_valid_idx_norm = np.isfinite(resampled_flux_normalized).argmax()
            if first_valid_idx_norm > 0:
                resampled_flux_normalized[:first_valid_idx_norm] = resampled_flux_normalized[first_valid_idx_norm]

            first_valid_idx_cont = np.isfinite(resampled_continuum).argmax()
            if first_valid_idx_cont > 0:
                resampled_continuum[:first_valid_idx_cont] = resampled_continuum[first_valid_idx_cont]
            # --- 填充结束 ---

            all_normalized_flux.append(resampled_flux_normalized)
            all_continuum_flux.append(resampled_continuum)
            all_obsids.append(spec['obsid'])

        except Exception as e:
            print(f"错误: 在处理 OBSID {spec.get('obsid')} 时发生插值错误: {e}")

    # 3. 创建最终的DataFrame
    df_normalized = pd.DataFrame(all_normalized_flux, index=all_obsids, columns=column_names)
    df_continuum = pd.DataFrame(all_continuum_flux, index=all_obsids, columns=column_names)

    print("光谱重采样和格式化完成！")
    return df_normalized, df_continuum