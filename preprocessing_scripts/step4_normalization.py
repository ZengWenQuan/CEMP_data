"""
步骤 4: 连续谱归一化

提供多种先进的连续谱拟合与归一化策略。
"""
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter, medfilt
from scipy.ndimage import maximum_filter

# 尝试导入PyTorch相关库
try:
    import torch
    from pytorch_wavelets import DWT1D, IDWT1D
    PYTORCH_WAVELETS_AVAILABLE = True
except ImportError:
    PYTORCH_WAVELETS_AVAILABLE = False

# --- 内部归一化策略函数 ---

def _normalize_spline_iterative(spec, lower_sigma=1.5, upper_sigma=3.0, max_iter=10, spline_k=3, spline_s_factor=1.0):
    """策略1: 迭代样条拟合与非对称拒绝 (Iterative Spline Fitting)
    
    一种非常强大和灵活的拟合方法，是很多专业软件包的核心思想。
    - 优点: 结合了样条的局部适应性和非对称拒绝的准确性，效果精确且稳健。
    - 缺点: 参数稍多，但物理解释性强。
    - 适用: 绝大多数情况下的首选方法。
    """
    flux = spec.get('flux_denoised').astype(np.float64)
    wavelength = spec.get('wavelength_rest').astype(np.float64)
    if flux is None or wavelength is None:
        return flux, np.ones_like(flux) if flux is not None else None

    mask = np.ones_like(flux, dtype=bool)
    continuum = np.ones_like(flux)

    for _ in range(max_iter):
        if np.sum(mask) <= spline_k:
            break
        
        spline = UnivariateSpline(wavelength[mask], flux[mask], k=spline_k, s=len(wavelength[mask]) * spline_s_factor)
        continuum = spline(wavelength)
        
        residuals = flux - continuum
        sigma = np.std(residuals[mask])
        
        new_mask = (residuals > -lower_sigma * sigma) & (residuals < upper_sigma * sigma)
        if np.array_equal(mask, new_mask):
            break
        mask = new_mask

    continuum[continuum <= 1e-9] = 1e-9
    
    normalized_flux = flux / continuum
    normalized_flux = np.clip(normalized_flux, -0.1, 2.0)
    
    return normalized_flux, continuum

def _normalize_wavelet_pytorch(spec, wavelet='db8', level=5, device='cpu'):
    """策略2: 小波变换归一化 (PyTorch GPU/CPU版本)
    
    利用信号处理技术分离连续谱（低频）和谱线（高频）。
    - 优点: 物理意义清晰，非迭代，速度快，可GPU加速。
    - 缺点: 参数选择（小波基、分解层数）需要实验确定。
    - 适用: 连续谱形状复杂、信噪比较高的光谱，尤其适合大批量处理。
    """
    flux = spec.get('flux_denoised').astype(np.float64)
    if flux is None:
        return flux, np.ones_like(flux) if flux is not None else None

    flux_tensor = torch.from_numpy(flux.copy()).float().to(device)
    flux_tensor = flux_tensor.unsqueeze(0).unsqueeze(0)

    dwt = DWT1D(wave=wavelet, J=level, mode='symmetric').to(device)
    idwt = IDWT1D(wave=wavelet, mode='symmetric').to(device)

    yl, yh = dwt(flux_tensor)
    
    yh_zeros = [torch.zeros_like(h) for h in yh]
    
    continuum_tensor = idwt((yl, yh_zeros))
    
    continuum = continuum_tensor.squeeze().cpu().numpy()

    if len(continuum) > len(flux):
        continuum = continuum[:len(flux)]
    elif len(continuum) < len(flux):
        continuum = np.pad(continuum, (0, len(flux) - len(continuum)), 'edge')

    continuum[continuum <= 1e-9] = 1e-9
    
    normalized_flux = flux / continuum
    normalized_flux = np.clip(normalized_flux, -0.1, 2.0)
    
    return normalized_flux, continuum

def _normalize_moving_percentile(spec, window_size=101, percentile=95):
    """策略3: 移动百分位滤波器 (Moving Percentile Filter)
    
    一个经典、高效的信号处理方法，用于寻找上包络线。
    - 优点: 对噪声和吸收线不敏感，参数少，速度快。
    - 缺点: 可能产生轻微“阶梯”效应，可通过二次平滑解决。
    - 适用: 绝大多数情况，特别是谱线密集或信噪比低的光谱。
    """
    flux = spec.get('flux_denoised').astype(np.float64)
    if flux is None:
        return flux, np.ones_like(flux) if flux is not None else None

    import pandas as pd
    # 使用pandas的rolling().quantile()实现高效的移动百分位计算
    # center=True确保窗口中心对齐当前点，min_periods=1处理边缘情况
    continuum = pd.Series(flux).rolling(window=window_size, center=True, min_periods=1).quantile(percentile / 100.0).values

    continuum[continuum <= 1e-9] = 1e-9
    
    normalized_flux = flux / continuum
    normalized_flux = np.clip(normalized_flux, -0.1, 2.0)
    
    return normalized_flux, continuum

def _normalize_conv_envelope(spec, median_window=51, max_window=51, smooth_window=51):
    """策略4: 卷积包络滤波 (Convolutional Envelope Filtering)

    一种高效、非迭代的包络线检测方法。
    其核心思想借鉴了经典信号处理中的包络检测技术。
    - 优点: 完全非迭代，速度极快，参数直观。
    - 缺点: 效果依赖于窗口大小的合理选择。
    - 适用: 需要高速处理大量光谱的场景。
    """
    flux = spec.get('flux_denoised')
    if flux is None:
        return flux, np.ones_like(flux) if flux is not None else None

    # 1. 初步平滑
    continuum = medfilt(flux.astype(np.float64), kernel_size=median_window)
    # 2. 寻找上包络
    continuum = maximum_filter(continuum, size=max_window)
    # 3. 最终平滑
    if smooth_window > 0 and len(continuum) > smooth_window:
        continuum = savgol_filter(continuum, smooth_window, 2)

    continuum[continuum <= 1e-9] = 1e-9
    
    normalized_flux = flux / continuum
    normalized_flux = np.clip(normalized_flux, -0.1, 2.0)
    
    return normalized_flux, continuum


# --- 主处理函数 ---

def process_step(spectra_data, method='spline_iterative', **kwargs):
    """对光谱进行连续谱归一化。

    新增功能: 支持对拟合后的连续谱进行二次平滑处理。
    可通过传入以下参数控制:
    - continuum_smoothing_window (int): 平滑窗口大小，奇数。如果为None或0，则不进行平滑。
    - continuum_smoothing_method (str): 'savgol' (默认) 或 'median'。
    - continuum_smoothing_polyorder (int): Savitzky-Golay滤波的多项式阶数 (仅当方法为'savgol'时有效)。
    """
    normalized_spectra = []
    total_spectra = len(spectra_data)
    print(f"开始对 {total_spectra} 个光谱进行归一化 (策略: {method})...")

    # 从kwargs中提取平滑参数，设置默认值
    smoothing_window = kwargs.pop('continuum_smoothing_window', 75)
    smoothing_method = kwargs.pop('continuum_smoothing_method', 'savgol')
    smoothing_polyorder = kwargs.pop('continuum_smoothing_polyorder', 2)

    if smoothing_window is not None and smoothing_window > 0:
        print(f"  -> 将对所有连续谱应用平滑处理 (方法: {smoothing_method}, 窗口: {smoothing_window})")

    if method == 'wavelet':
        if not PYTORCH_WAVELETS_AVAILABLE:
            print("错误: 'wavelet' 策略需要 'torch' 和 'pytorch-wavelets' 库。请运行 'pip install torch pytorch-wavelets'。")
            return spectra_data
        
        if torch.cuda.is_available() and kwargs.get('device') != 'cpu':
            device = torch.device(kwargs.get('device', 'cuda'))
            print(f"  -> 检测到CUDA，将使用 {device} 进行小波变换。")
        else:
            device = torch.device('cpu')
            print("  -> 未检测到CUDA或已指定CPU，将使用 CPU 进行小波变换。")
        kwargs['device'] = device

    for i, spec in enumerate(spectra_data):
        if (i + 1) % 10 == 0:
            print(f"  正在处理光谱 {i + 1}/{total_spectra}...")
        
        normalized_spec = spec.copy()
        flux = spec.get('flux_denoised')
        continuum = np.ones_like(flux) if flux is not None else None

        try:
            # 1. 使用选定策略计算初始连续谱
            if method == 'spline_iterative':
                flux_normalized, continuum = _normalize_spline_iterative(spec, **kwargs)
            elif method == 'wavelet':
                flux_normalized, continuum = _normalize_wavelet_pytorch(spec, **kwargs)
            elif method == 'moving_percentile':
                flux_normalized, continuum = _normalize_moving_percentile(spec, **kwargs)
            elif method == 'conv_envelope':
                flux_normalized, continuum = _normalize_conv_envelope(spec, **kwargs)
            else:
                print(f"警告：未知的归一化方法 '{method}'。将不进行处理。")
                flux_normalized, continuum = flux, continuum

            # 2. 对计算出的连续谱进行平滑 (如果需要)
            if continuum is not None and smoothing_window is not None and smoothing_window > 0:
                if len(continuum) > smoothing_window:
                    original_continuum = continuum.copy() # 保存原始谱用于比较或调试
                    if smoothing_method == 'savgol':
                        # 确保窗口大小是奇数
                        if smoothing_window % 2 == 0:
                            smoothing_window += 1
                        continuum = savgol_filter(continuum, smoothing_window, smoothing_polyorder)
                    elif smoothing_method == 'median':
                        # 确保窗口大小是奇数
                        if smoothing_window % 2 == 0:
                            smoothing_window += 1
                        continuum = medfilt(continuum.astype(np.float64), kernel_size=smoothing_window)
                    else:
                        print(f"警告: 未知的平滑方法 '{smoothing_method}'。将跳过平滑处理。")
                
                # 3. 使用平滑后的连续谱重新归一化
                continuum[continuum <= 1e-9] = 1e-9
                flux_normalized = flux / continuum
                flux_normalized = np.clip(flux_normalized, -0.1, 2.0)

            normalized_spec['flux_normalized'] = flux_normalized
            normalized_spec['continuum'] = continuum

        except Exception as e:
            print(f"错误: 在处理 OBSID {spec.get('obsid')} 时发生错误 (策略: {method}): {e}")
            normalized_spec['flux_normalized'] = flux
            normalized_spec['continuum'] = np.ones_like(flux) if flux is not None else None
            
        normalized_spectra.append(normalized_spec)
        
    print("归一化完成！")
    return normalized_spectra
