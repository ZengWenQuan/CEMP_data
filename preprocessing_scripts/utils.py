
"""
共享工具函数库

包含文件操作、绘图等被多个步骤共享的函数。
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def ensure_dir(directory):
    """确保目录存在，如果不存在则创建。"""
    os.makedirs(directory, exist_ok=True)

def _add_labels_to_plot(ax, obsid, labels_df):
    """在图表的右上角添加光谱标签。"""
    if labels_df is None or obsid not in labels_df.index:
        return
    
    # Ensure obsid is of the correct type if there are mismatches (e.g., int vs str)
    try:
        labels = labels_df.loc[obsid]
    except KeyError:
        try:
            labels = labels_df.loc[int(obsid)]
        except (KeyError, ValueError):
            print(f"警告: 在标签数据中未找到 OBSID: {obsid}，无法添加标签。")
            return

    label_text = (
        f"Teff = {labels.get('Teff', 'N/A'):.0f} K\n"
        f"log(g) = {labels.get('logg', 'N/A'):.2f}\n"
        f"[Fe/H] = {labels.get('FeH', 'N/A'):.2f}\n"
        f"[C/Fe] = {labels.get('CFe', 'N/A'):.2f}"
    )
    
    ax.text(0.98, 0.98, label_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7))

def save_spectra_to_csv(spectra_data, output_path, keys_to_save):
    """将处理过程中的光谱数据保存为CSV文件。"""
    rows_to_save = []
    for spec in spectra_data:
        wl_key = 'wavelength_final' if 'wavelength_final' in spec else ('wavelength_rest' if 'wavelength_rest' in spec else 'wavelength')
        if wl_key not in spec:
            continue

        rows_to_save.append([spec['obsid'], 'WAVELENGTH'] + spec[wl_key].tolist())
        
        for key in keys_to_save:
            if key in spec and spec[key] is not None:
                rows_to_save.append([spec['obsid'], key.upper()] + spec[key].tolist())
    
    pd.DataFrame(rows_to_save).to_csv(output_path, header=False, index=False)
    print(f"  -> 数据已保存到: {output_path}")

def save_dataframe_to_csv(dataframe, output_path):
    """将格式化后的DataFrame保存到CSV文件，索引为obsid。"""
    dataframe.to_csv(output_path, index_label='obsid')
    print(f"  -> 最终数据产品已保存到: {output_path}")

def visualize_single_step(step_name, spectra_data, obsids_to_visualize, figure_dir, labels_df=None):
    """
    为简单的处理步骤（如提取、红移校正）生成并保存指定OBSID的可视化图表。
    """
    print(f"为步骤 '{step_name}' 生成 {len(obsids_to_visualize)} 个指定的可视化样本...")
    spectra_map = {spec['obsid']: spec for spec in spectra_data}

    for obsid in obsids_to_visualize:
        spec = spectra_map.get(obsid)
        if spec is None:
            print(f"警告: 在数据中未找到指定的 OBSID: {obsid}，跳过可视化。")
            continue

        fig_path = os.path.join(figure_dir, f'{step_name}_obsid_{obsid}.pdf')
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_title(f'{step_name} - OBSID: {obsid}')

        if step_name == 'Step1_Extraction':
            ax.plot(spec['wavelength'], spec['flux'], label='Original Flux', linewidth=0.8)
            ax.set_xlabel('Observed Wavelength (Å)'); ax.set_ylabel('Flux')

        elif step_name == 'Step2_RedshiftCorrection':
            ax.plot(spec['wavelength'], spec['flux'], label=f'Original (Observed)', alpha=0.7, linewidth=0.8)
            ax.plot(spec['wavelength_rest'], spec['flux'], label=f'Corrected (Rest, z={spec.get("z", 0):.4f})', linestyle='--', alpha=0.9, linewidth=1.0)
            ax.set_xlabel('Wavelength (Å)'); ax.set_ylabel('Flux')

        _add_labels_to_plot(ax, obsid, labels_df)
        ax.legend()
        plt.savefig(fig_path, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"  -> 单步可视化图表已保存: {fig_path}")

def visualize_comparison_step(step_name, before_dataset, after_dataset, obsids_to_visualize, before_key, after_key, before_label, after_label, figure_dir, labels_df=None):
    """
    为复杂的处理步骤（如去噪）生成对比图（重叠图与子图）。
    """
    print(f"为步骤 '{step_name}' 生成 {len(obsids_to_visualize)} 个指定的对比可视化样本...")
    before_map = {spec['obsid']: spec for spec in before_dataset}
    after_map = {spec['obsid']: spec for spec in after_dataset}

    for obsid in obsids_to_visualize:
        spec_before = before_map.get(obsid)
        spec_after = after_map.get(obsid)
        if spec_before is None or spec_after is None:
            print(f"警告: 在某个数据集中未找到 OBSID: {obsid}，跳过对比可视化。")
            continue

        wl_key = 'wavelength_rest' if 'wavelength_rest' in spec_after else 'wavelength'

        # Overlap plot
        fig_path_overlap = os.path.join(figure_dir, f'{step_name}_obsid_{obsid}_overlap.pdf')
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.set_title(f'{step_name} (Overlap) - OBSID: {obsid}')
        ax.plot(spec_before[wl_key], spec_before[before_key], label=before_label, alpha=0.7, linewidth=0.8)
        ax.plot(spec_after[wl_key], spec_after[after_key], label=after_label, alpha=0.9, linewidth=0.8)
        ax.set_xlabel('Wavelength (Å)'); ax.set_ylabel('Flux')
        _add_labels_to_plot(ax, obsid, labels_df)
        ax.legend(); ax.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(fig_path_overlap, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"  -> 重叠对比图已保存: {fig_path_overlap}")

        # Subplots
        fig_path_subplots = os.path.join(figure_dir, f'{step_name}_obsid_{obsid}_subplots.pdf')
        fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        fig.suptitle(f'{step_name} (Subplots) - OBSID: {obsid}', fontsize=16)
        
        axes[0].plot(spec_before[wl_key], spec_before[before_key], label=before_label)
        axes[0].set_ylabel('Flux'); axes[0].legend(); axes[0].grid(True, linestyle='--', alpha=0.6)
        _add_labels_to_plot(axes[0], obsid, labels_df)

        axes[1].plot(spec_after[wl_key], spec_after[after_key], label=after_label, color='C1')
        axes[1].set_ylabel('Flux'); axes[1].legend(); axes[1].grid(True, linestyle='--', alpha=0.6)

        axes[1].set_xlabel('Wavelength (Å)')
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.savefig(fig_path_subplots, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"  -> 子图对比图已保存: {fig_path_subplots}")

def visualize_normalization_step(step_name, spectra_data, obsids_to_visualize, figure_dir, labels_df=None):
    """
    为归一化步骤生成专门的可视化图表，包含三个子图进行完整对比。
    """
    print(f"为步骤 '{step_name}' 生成 {len(obsids_to_visualize)} 个指定的归一化可视化样本...")
    spectra_map = {spec['obsid']: spec for spec in spectra_data}

    for obsid in obsids_to_visualize:
        spec = spectra_map.get(obsid)
        if spec is None:
            print(f"警告: 在数据中未找到指定的 OBSID: {obsid}，跳过归一化可视化。")
            continue

        required_keys = ['wavelength_rest', 'flux', 'flux_denoised', 'continuum', 'flux_normalized']
        if not all(key in spec for key in required_keys):
            print(f"警告: OBSID {obsid} 的光谱数据不完整，跳过可视化。")
            continue

        wl_key = 'wavelength_rest'
        fig_path = os.path.join(figure_dir, f'{step_name}_obsid_{obsid}_full_comparison.pdf')
        fig, axes = plt.subplots(3, 1, figsize=(15, 15), sharex=True)
        fig.suptitle(f'Full Normalization Process - OBSID: {obsid}', fontsize=16)

        # Plot 1: Original Spectrum
        axes[0].plot(spec[wl_key], spec['flux'], label='Original Flux (Redshift Corrected)', linewidth=0.8)
        axes[0].set_title('1. Original Spectrum')
        axes[0].set_ylabel('Flux')
        axes[0].legend()
        axes[0].grid(True, linestyle='--', alpha=0.6)
        _add_labels_to_plot(axes[0], obsid, labels_df)

        # Plot 2: Denoising & Continuum
        axes[1].plot(spec[wl_key], spec['flux_denoised'], label='Denoised Flux', alpha=0.9, linewidth=0.8)
        axes[1].plot(spec[wl_key], spec['continuum'], label='Fitted Continuum', linestyle='--', color='red', linewidth=1.2)
        axes[1].set_title('2. Denoising & Continuum Fitting')
        axes[1].set_ylabel('Flux')
        axes[1].legend()
        axes[1].grid(True, linestyle='--', alpha=0.6)
        
        # Plot 3: Normalized Spectrum
        axes[2].plot(spec[wl_key], spec['flux_normalized'], label='Final Normalized Flux', color='green', linewidth=0.8)
        axes[2].axhline(1.0, linestyle='--', color='gray', alpha=0.7)
        axes[2].set_title('3. Final Normalized Spectrum')
        axes[2].set_ylabel('Normalized Flux')
        axes[2].legend()
        axes[2].grid(True, linestyle='--', alpha=0.6)
        axes[2].set_ylim(bottom=min(0.0, np.min(spec['flux_normalized']) - 0.1), top=max(1.2, np.percentile(spec['flux_normalized'], 99.5) * 1.1))

        axes[2].set_xlabel('Wavelength (Å)')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(fig_path, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"  -> 归一化全流程对比图已保存: {fig_path}")

def visualize_continuum_fit(spectra_data, obsids_to_visualize, figure_dir, labels_df=None):
    """
    专门用于可视化连续谱拟合效果的函数。
    """
    print(f"为连续谱拟合生成 {len(obsids_to_visualize)} 个指定的可视化样本...")
    spectra_map = {spec['obsid']: spec for spec in spectra_data}

    for obsid in obsids_to_visualize:
        spec = spectra_map.get(obsid)
        if spec is None or 'flux_denoised' not in spec or 'continuum' not in spec:
            print(f"警告: OBSID {obsid} 的数据不完整，跳过连续谱可视化。")
            continue

        fig_path = os.path.join(figure_dir, f'Step4_ContinuumFit_obsid_{obsid}.pdf')
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.set_title(f'Continuum Fitting - OBSID: {obsid}')
        ax.plot(spec['wavelength_rest'], spec['flux_denoised'], label='Denoised Flux', alpha=0.7, linewidth=0.8)
        ax.plot(spec['wavelength_rest'], spec['continuum'], label='Fitted Continuum', color='red', linestyle='--', linewidth=1.5)
        ax.set_xlabel('Wavelength (Å)'); ax.set_ylabel('Flux')
        _add_labels_to_plot(ax, obsid, labels_df)
        ax.legend(); ax.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(fig_path, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"  -> 连续谱拟合图已保存: {fig_path}")

def visualize_final_spectra(dataframe, obsids_to_visualize, figure_dir, title_prefix, labels_df=None):
    """从最终的DataFrame中抽取样本进行可视化。"""
    print(f"为最终数据产品 '{title_prefix}' 生成 {len(obsids_to_visualize)} 个可视化样本...")
    
    # 将列名从字符串转换回浮点数用于绘图
    wavelengths = [float(col) for col in dataframe.columns]

    for obsid in obsids_to_visualize:
        if obsid not in dataframe.index:
            print(f"警告: 在DataFrame中未找到指定的 OBSID: {obsid}，跳过可视化。")
            continue

        flux = dataframe.loc[obsid].values
        fig_path = os.path.join(figure_dir, f'Final_Spectrum_{title_prefix}_obsid_{obsid}.pdf')
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.set_title(f'{title_prefix} - OBSID: {obsid}')
        ax.plot(wavelengths, flux, linewidth=0.8)
        ax.set_xlabel('Wavelength (Å)'); ax.set_ylabel('Flux')
        _add_labels_to_plot(ax, obsid, labels_df)
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(fig_path, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"  -> 最终产品可视化图表已保存: {fig_path}")
