
"""
步骤 1: 从FITS文件提取光谱
"""
import os
from astropy.io import fits

def process_step(fit_dir):
    fits_files = [f for f in os.listdir(fit_dir) if f.endswith('.fits')]
    all_spectra = []
    total_files = len(fits_files)
    print(f"找到 {total_files} 个FITS文件，开始提取...")
    for i, fits_file in enumerate(sorted(fits_files)):
        if (i + 1) % 10 == 0:
            print(f"  正在处理文件 {i + 1}/{total_files}...")
        filepath = os.path.join(fit_dir, fits_file)
        try:
            with fits.open(filepath) as hdul:
                all_spectra.append({
                    'obsid': hdul[0].header['OBSID'],
                    'wavelength': hdul[1].data['WAVELENGTH'][0],
                    'flux': hdul[1].data['FLUX'][0]
                })
        except Exception as e:
            print(f"警告：处理文件 {filepath} 时出错: {e}")
    print("提取完成！")
    return all_spectra
