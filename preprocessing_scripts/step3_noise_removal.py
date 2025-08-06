"""
步骤 3: 光谱去噪

提供多种去噪策略，包括经典方法和更先进的信号处理技术。
"""
import numpy as np
from scipy.signal import savgol_filter, medfilt

# 尝试导入torch和pytorch_wavelets，如果失败则优雅地处理
try:
    import torch
    from pytorch_wavelets import DWT1D, IDWT1D
    PYTORCH_WAVELETS_AVAILABLE = True
except ImportError:
    PYTORCH_WAVELETS_AVAILABLE = False

# --- 内部去噪函数定义 ---

def _denoise_polynomial(spec, degree, threshold):
    """策略1: 多项式拟合 (Sigma-Clipping)
    
    通过拟合连续谱来识别并替换离群点。
    - 优点: 能有效去除远离连续谱的宇宙线等噪声点。
    - 缺点: 可能会错误地“填平”真实的吸收线特征，改变谱线轮廓。
    - 适用: 主要用于去除离群点，而不是平滑。
    """
    flux = spec.get('flux')
    wavelength = spec.get('wavelength_rest')
    if flux is None or wavelength is None or len(wavelength) <= degree:
        return flux.copy() if flux is not None else np.array([])

    try:
        coeffs = np.polyfit(wavelength, flux, degree)
        poly_fit = np.poly1d(coeffs)(wavelength)
        residuals = flux - poly_fit
        is_outlier = np.abs(residuals) > (np.std(residuals) * threshold)
        denoised_flux = flux.copy()
        denoised_flux[is_outlier] = poly_fit[is_outlier]
        return denoised_flux
    except (np.linalg.LinAlgError, ValueError):
        return flux.copy()

def _denoise_moving_average(spec, window_size):
    """策略2: 滑动平均
    
    一种简单的平滑技术。
    - 优点: 实现简单，能快速平滑高频噪声。
    - 缺点: 严重降低光谱分辨率，使谱线特征变得模糊和扁平。
    - 适用: 对分辨率要求不高的快速预览。
    """
    flux = spec.get('flux')
    if flux is None or window_size < 3:
        return flux.copy() if flux is not None else np.array([])
    # 确保窗口大小为奇数
    window_size = window_size if window_size % 2 != 0 else window_size + 1
    return np.convolve(flux, np.ones(window_size)/window_size, mode='same')

def _denoise_savgol(spec, window_length, polyorder):
    """策略3: Savitzky-Golay 滤波器
    
    滑动平均的智能升级版，天文学中最常用的方法之一。
    - 优点: 在平滑噪声的同时，能极好地保持谱线的形状、宽度和高度。
    - 缺点: 对单个尖锐的脉冲噪声不如中值滤波有效。
    - 适用: 绝大多数需要平衡噪声去除和特征保持的场景。
    """
    flux = spec.get('flux')
    if flux is None or len(flux) <= window_length:
        return flux.copy() if flux is not None else np.array([])
    # 确保窗口长度为奇数且大于多项式阶数
    window_length = window_length if window_length % 2 != 0 else window_length + 1
    if window_length <= polyorder:
        polyorder = window_length - 1
    return savgol_filter(flux, window_length, polyorder)

def _denoise_median(spec, kernel_size):
    """策略4: 中值滤波器
    
    用窗口内数据的中位数替换中心点。
    - 优点: 去除宇宙线等“椒盐”脉冲噪声的利器，且能很好地保持谱线边缘。
    - 缺点: 窗口较大时可能引入不自然的“阶梯”状结构。
    - 适用: 当主要噪声来源是尖锐的脉冲噪声时。
    """
    flux = spec.get('flux')
    if flux is None:
        return np.array([])
    # 确保核大小为奇数
    kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
    # 将数据类型转换为float64以支持medfilt
    flux_float64 = flux.astype(np.float64)
    return medfilt(flux_float64, kernel_size)

def _denoise_wavelet_pytorch(spec, wavelet, level, device):
    """策略5: 小波变换去噪 (PyTorch GPU/CPU版本)
    
    一种先进的多尺度分析技术，可利用GPU加速。
    - 优点: 理论上能在不同尺度上分离信号和噪声，特征保持能力极佳。
    - 缺点: 参数选择（小波基、分解层数）复杂，且依赖PyTorch。
    - 适用: 对信号保真度有极高要求，且数据集较大时可通过GPU加速。
    """
    flux = spec.get('flux')
    if flux is None:
        return np.array([])

    try:
        # 1. 数据准备：将numpy数组转换为torch张量
        flux_tensor = torch.from_numpy(flux.copy()).float().to(device)
        # DWT1D需要 (N, C, L) 格式的输入, N=批大小, C=通道数, L=信号长度
        flux_tensor = flux_tensor.unsqueeze(0).unsqueeze(0)

        # 2. 初始化DWT和IDWT变换
        dwt = DWT1D(wave=wavelet, J=level, mode='symmetric').to(device)
        idwt = IDWT1D(wave=wavelet, mode='symmetric').to(device)

        # 3. 执行DWT
        yl, yh = dwt(flux_tensor)

        # 4. 计算阈值 (Universal Threshold)
        # 从最精细的细节系数中估计噪声标准差
        detail_coeffs = yh[-1].squeeze()
        sigma = torch.median(torch.abs(detail_coeffs - torch.median(detail_coeffs))) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(flux)))

        # 5. 对所有层级的细节系数进行软阈值处理
        yh_thresh = [torch.nn.functional.softshrink(h, threshold) for h in yh]

        # 6. 执行IDWT重构信号
        denoised_flux_tensor = idwt((yl, yh_thresh))

        # 7. 将结果转回numpy数组
        return denoised_flux_tensor.squeeze().cpu().numpy()

    except Exception as e:
        print(f"警告: PyTorch小波变换去噪失败: {e}。返回原始光谱。")
        return flux.copy()

def _denoise_weighted_moving_average(spec, weights=(0.25, 0.5, 0.25)):
    """策略6: 自定义加权滑动平均
    
    使用一个[a, b, a]大小的滤波器进行平滑滤波。
    - 优点: 相比普通滑动平均，可以给予中心点更高的权重，更好地保留特征。
    - 缺点: 窗口大小固定为3，灵活性较低。
    - 适用: 需要轻度平滑，且希望保留中心点信息的场景。
    """
    flux = spec.get('flux')
    if flux is None:
        return np.array([])
    
    if len(weights) != 3:
        print("警告: 加权滑动平均的权重必须是3个元素。返回原始光谱。")
        return flux.copy()

    # 权重归一化
    filter_weights = np.array(weights) / np.sum(weights)
    return np.convolve(flux, filter_weights, mode='same')


# --- 主处理函数 ---

def process_step(spectra_data, strategy='savgol', **kwargs):
    """对光谱流量进行去噪。"""
    denoised_spectra = []
    total_spectra = len(spectra_data)
    print(f"开始对 {total_spectra} 个光谱进行去噪 (策略: {strategy})...")

    if strategy == 'none':
        print("  -> 'none' 策略被选中，将跳过实际去噪操作。")

    # 如果使用小波变换，检查依赖库和GPU可用性
    if strategy == 'wavelet':
        if not PYTORCH_WAVELETS_AVAILABLE:
            print("错误: 'wavelet' 策略需要 'torch' 和 'pytorch-wavelets' 库。请运行 'pip install torch pytorch-wavelets'。")
            print("将不进行去噪处理。")
            # 直接返回未修改的数据
            for spec in spectra_data:
                denoised_spec = spec.copy()
                denoised_spec['flux_denoised'] = spec.get('flux').copy()
                denoised_spectra.append(denoised_spec)
            return denoised_spectra
        
        if torch.cuda.is_available():
            device = torch.device(kwargs.get('device', 'cuda'))
            print(f"  -> 检测到CUDA，将使用 {device} 进行小波变换。")
        else:
            device = torch.device('cpu')
            print("  -> 未检测到CUDA，将使用 CPU 进行小波变换。")
        # 将device添加到kwargs中，以便传递给处理函数
        kwargs['device'] = device

    for i, spec in enumerate(spectra_data):
        if (i + 1) % 10 == 0:
            print(f"  正在处理光谱 {i + 1}/{total_spectra}...")
        
        denoised_spec = spec.copy()
        original_flux = spec.get('flux')
        
        if strategy == 'polynomial':
            denoised_spec['flux_denoised'] = _denoise_polynomial(spec, **kwargs)
        
        elif strategy == 'moving_average':
            denoised_spec['flux_denoised'] = _denoise_moving_average(spec, **kwargs)

        elif strategy == 'savgol':
            savgol_args = {
                'window_length': kwargs.get('window_length', 11),
                'polyorder': kwargs.get('polyorder', 3)
            }
            denoised_spec['flux_denoised'] = _denoise_savgol(spec, **savgol_args)

        elif strategy == 'median':
            median_args = {
                'kernel_size': kwargs.get('kernel_size', 5)
            }
            denoised_spec['flux_denoised'] = _denoise_median(spec, **median_args)

        elif strategy == 'wavelet':
            wavelet_args = {
                'wavelet': kwargs.get('wavelet', 'db4'),
                'level': kwargs.get('level', 4),
                'device': kwargs['device'] # 使用上面检测到的device
            }
            denoised_spec['flux_denoised'] = _denoise_wavelet_pytorch(spec, **wavelet_args)

        elif strategy == 'weighted_moving_average':
            wma_args = {
                'weights': kwargs.get('weights', (0.25, 0.5, 0.25))
            }
            denoised_spec['flux_denoised'] = _denoise_weighted_moving_average(spec, **wma_args)

        elif strategy == 'none':
            denoised_spec['flux_denoised'] = original_flux.copy() if original_flux is not None else np.array([])

        else:
            print(f"警告：未知的去噪策略 '{strategy}'。将不进行去噪。")
            denoised_spec['flux_denoised'] = original_flux.copy() if original_flux is not None else np.array([])

        denoised_spectra.append(denoised_spec)
        
    print("光谱去噪完成！")
    return denoised_spectra
