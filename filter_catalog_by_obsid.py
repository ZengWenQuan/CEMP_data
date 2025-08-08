
import pandas as pd
import os

def filter_main_catalog_by_obsid_list(main_catalog_path, obsid_source_path, output_path):
    """
    使用一个文件中的obsid列表作为筛选依据，过滤一个主星表文件。

    Args:
        main_catalog_path (str): 需要被筛选的主星表CSV文件路径。
        obsid_source_path (str): 提供obsid白名单的CSV文件路径。
        output_path (str): 筛选后输出的CSV文件路径。
    """
    # --- 1. 读取obsid白名单 ---
    print(f"正在从源文件读取obsid白名单: {obsid_source_path}")
    try:
        df_source = pd.read_csv(obsid_source_path)
    except FileNotFoundError:
        print(f"错误: obsid源文件未找到 -> {obsid_source_path}")
        return

    if 'obsid' not in df_source.columns:
        print(f"错误: 在文件 {obsid_source_path} 中未找到 'obsid' 列。")
        return

    # 使用集合(set)来存储白名单，查询效率最高
    obsid_whitelist = set(df_source['obsid'].unique())
    print(f"白名单准备就绪，共包含 {len(obsid_whitelist)} 个唯一的obsid。")

    # --- 2. 读取并筛选主星表 ---
    print(f"\n正在读取主星表: {main_catalog_path}")
    try:
        df_main = pd.read_csv(main_catalog_path)
    except FileNotFoundError:
        print(f"错误: 主星表文件未找到 -> {main_catalog_path}")
        return

    if 'obsid' not in df_main.columns:
        print(f"错误: 在主星表 {main_catalog_path} 中未找到 'obsid' 列。")
        return

    initial_count = len(df_main)
    print(f"主星表读取完成，共包含 {initial_count} 行。")

    print("\n开始根据obsid白名单进行筛选...")
    # 使用 .isin() 方法进行高效筛选
    filtered_df = df_main[df_main['obsid'].isin(obsid_whitelist)].copy()
    final_count = len(filtered_df)

    print("筛选完成。")
    print(f"  - 匹配到的行数: {final_count}")
    print(f"  - 剔除的行数: {initial_count - final_count}")

    # --- 3. 保存结果 ---
    if final_count > 0:
        print(f"\n正在将筛选结果保存到: {output_path}")
        filtered_df.to_csv(output_path, index=False)
        print("保存成功！")
    else:
        print("\n没有在主星表中找到任何匹配的obsid，不生成输出文件。")

if __name__ == '__main__':
    # --- 路径配置 ---
    # 获取当前脚本所在的目录
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # obsid白名单的来源文件
    OBSID_SOURCE_FILE = os.path.join(base_dir, 'files', 'final_spectra_normalized_median_wavelet_4800.csv')

    # 需要被筛选的主星表
    MAIN_CATALOG_FILE = os.path.join(base_dir, 'removed_with_rv.csv')

    # 筛选结果的输出文件
    OUTPUT_FILE = os.path.join(base_dir, 'removed_with_rv_filtered.csv')

    filter_main_catalog_by_obsid_list(MAIN_CATALOG_FILE, OBSID_SOURCE_FILE, OUTPUT_FILE)
