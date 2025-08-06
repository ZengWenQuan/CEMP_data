# CEMP 天文光谱数据处理流水线 (V2.5 学术版)

本项目是一个功能强大、高度模块化的天文光谱数据处理流水线。它将每一个核心处理步骤封装为独立的Python模块，并通过一个主控Jupyter Notebook (`pipeline.ipynb`) 进行调用，实现了清晰的工作流程和代码可维护性。

这个V2.5版本引入了多策略支持、算法的数学原理说明、参考文献、以及最终的重采样与格式化步骤，形成了一个完整的、从原始数据到科学可用产品的处理流程。

---

## 如何运行

1. **安装依赖库**: 本流水线现在依赖于包括 `pytorch`在内的多个科学计算库。请通过以下命令安装所有必需的依赖：

   ```bash
   pip install astropy pandas numpy matplotlib scipy torch pytorch-wavelets
   ```
2. **启动Jupyter**: 在项目根目录下运行 `jupyter notebook` 或 `jupyter-lab`。
3. **打开并运行Notebook**: 打开 `pipeline.ipynb`。
4. **配置并执行**: 流水线的配置已被模块化。您不再需要在文件顶部进行所有设置，而是在**每个步骤对应的代码单元格中**进行配置。请按顺序执行所有单元格。

---

## 项目结构

- **`pipeline.ipynb`**: **主入口**。
- **`preprocessing_scripts/`**: 包含所有核心逻辑的Python包。
- **`files/`**: 存放所有数据产品（.csv）。
- **`figures/`**: 存放所有步骤自动生成的可视化检验图。
- **`unzipped_fits_100/`**: 存放原始FITS数据。

---

## 核心功能与算法原理

### 第1-2步: 数据提取与红移校正

这两个初始步骤负责从原始的FITS文件中提取光谱数据（波长和通量），并根据给定的红移值 `z`，将观测波长 `λ_obs`校正到静止系波长 `λ_rest`。

> $$
> \lambda_\text{rest} = \frac{\lambda_\text{obs}}{1+z}
> $$

### 第3步: 光谱去噪

您可以在 `pipeline.ipynb`的第3步配置单元格中，通过修改 `DENOISE_STRATEGY`变量来选择以下策略之一：

- `savgol` (默认): **Savitzky-Golay 滤波器**。
- `median`: **中值滤波器**。
- `wavelet`: **小波变换 (PyTorch版)**。
- `polynomial`: 基于多项式拟合的离群点剔除。
- `moving_average`: 简单的滑动平均平滑。

### 第4步: 连续谱归一化

您可以在 `pipeline.ipynb`的第4步配置单元格中，通过修改 `NORM_METHOD`变量来选择以下策略之一：

- `spline_iterative` : **迭代样条拟合与非对称拒绝**。
- `wavelet`: **小波变换 (PyTorch版)**。
- `quantile`: **滚动分位数上包络线**。
- `conv_envelope`(默认): **卷积包络滤波**。

### 第5步: 光谱重采样与格式化

此步骤将所有处理后的光谱插值到一个**统一的、标准化的波长网格**上，并将其格式化为以 `obsid`为索引、波长为列的表格（DataFrame），这对于后续的科学分析至关重要。

- **思想**: 本步骤基于**线性插值（Linear Interpolation）** 的思想。对于每一个需要计算通量的新波长点 $\lambda_\text{new}$，我们在原始光谱中找到其左右相邻的两个波长点 $\lambda_1$ 和 $\lambda_2$，以及它们对应的通量 $F_1$ 和 $F_2$。新的通量值 $F_\text{new}$ 就是通过连接 $(\lambda_1, F_1)$ 和 $(\lambda_2, F_2)$ 的直线计算得出的：

  > $$
  > F_\text{new} = F_1 + (F_2 - F_1) \cdot \frac{\lambda_\text{new} - \lambda_1}{\lambda_2 - \lambda_1}
  > $$
  >
- **配置**: 您可以在 `pipeline.ipynb`的第5步配置单元格中，通过修改 `WAVELENGTH_CONFIG`列表来灵活地定义最终的波长网格，可以包含多个不连续的区间，并为每个区间指定不同的采样步长。

---

## 参考文献与致谢

本流水线中实现的多种算法均基于信号处理和天文学界的经典思想与开创性工作。我们在此对这些工作的作者表示感谢。

1. **Savitzky-Golay 滤波器**:

   - Savitzky, A.; Golay, M. J. E. (1964). "Smoothing and Differentiation of Data by Simplified Least Squares Procedures". *Analytical Chemistry*, 36(8), 1627-1639.

   > 这是提出此著名滤波器的原始论文，它奠定了使用局部多项式拟合进行数据平滑的基础。
   >
2. **小波变换去噪 (Wavelet Denoising)**:

   - Donoho, D. L.; Johnstone, I. M. (1994). "Ideal spatial adaptation by wavelet shrinkage". *Biometrika*, 81(3), 425-455.
   - Donoho, D. L. (1995). "De-noising by soft-thresholding". *IEEE Transactions on Information Theory*, 41(3), 613-627.

   > 这两篇是小波阈值去噪领域的奠基之作，系统地建立了通过小波变换进行信号去噪的理论框架。
   >
3. **卷积包络滤波 (`conv_envelope`)**:

   - 本算法的核心思想借鉴了经典的数字信号处理中的包络检测技术，并与传统天文数据处理软件包 **IRAF (Image Reduction and Analysis Facility)** 中 `continuum` 任务的某些非迭代滤波模式在逻辑上相似。

---

## 可视化

流水线会在开始时随机选择3个 `obsid`，并为这3个目标生成贯穿始终的、可对比的可视化图表。

- **对比图**: 对于去噪等步骤，会自动生成将处理前/后结果**重叠**和**并排**显示的PDF图表。
- **归一化全景图**: 在第4步，会生成一个包含**三个子图**的PDF，完整展示从**原始光谱 -> 连续谱拟合 -> 最终归一化光谱**的全过程。
- **最终产品可视化**: 在第5步，会调用一个专门的函数，从最终生成的DataFrame中抽取样本进行绘图，以检验最终数据产品的质量。

---

*这份文档将随着流水线的修改而自动更新。*
