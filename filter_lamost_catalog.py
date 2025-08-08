
import pandas as pd

def filter_catalog(input_path, output_path):
    """
    根据设定的物理参数和误差标准，筛选LAMOST高质量光谱。

    Args:
        input_path (str): 输入的交叉星表CSV文件路径。
        output_path (str): 筛选后输出的CSV文件路径。
    """
    print(f"正在读取星表文件: {input_path}")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"错误: 输入文件未找到，请检查路径: {input_path}")
        return

    initial_count = len(df)
    print(f"读取完成，共包含 {initial_count} 条光谱。")
    print("\n开始应用筛选条件...")

    # --- 定义筛选条件 ---
    # 1. 信噪比条件
    snr_condition = df['snrg'] > 20
    
    # 2. 视向速度误差条件
    rv_err_condition = df['rv_err'] < 20
    
    # 3. 恒星大气参数误差条件
    teff_err_condition = df['teff_err'] < 200
    logg_err_condition = df['logg_err'] < 0.5
    feh_err_condition = df['feh_err'] < 0.3

    # --- 组合所有条件 ---
    combined_conditions = (
        snr_condition &
        rv_err_condition &
        teff_err_condition &
        logg_err_condition &
        feh_err_condition
    )

    filtered_df = df[combined_conditions].copy()
    final_count = len(filtered_df)

    print("\n筛选条件应用完毕。")
    print(f"  - 初始数量: {initial_count}")
    print(f"  - 筛选后剩余数量: {final_count}")
    print(f"  - 剔除数量: {initial_count - final_count}")
    
    if final_count > 0:
        print(f"\n正在将筛选结果保存到: {output_path}")
        filtered_df.to_csv(output_path, index=False)
        print("保存成功！")
    else:
        print("\n没有光谱满足所有筛选条件，不生成输出文件。")

if __name__ == '__main__':
    # --- 文件路径配置 ---
    # 请将此脚本移动到与星表相同的目录下，或者修改下面的路径为绝对路径
    INPUT_CATALOG = 'last_lamostdr11_appogedr17.csv'
    OUTPUT_CATALOG = 'last_lamostdr11_appogedr17_filtered.csv'
    
    filter_catalog(INPUT_CATALOG, OUTPUT_CATALOG)
