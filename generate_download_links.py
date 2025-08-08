
import pandas as pd

def generate_download_links(csv_path, output_txt_path):
    """
    从CSV文件中读取obsid，并生成对应的LAMOST FITS文件下载链接。

    Args:
        csv_path (str): 包含'obsid'列的输入CSV文件路径。
        output_txt_path (str): 用于保存下载链接的输出文本文件路径。
    """
    # --- URL模板配置 ---
    # 将 {obsid} 作为占位符，方便后续替换
    URL_TEMPLATE = "https://www.lamost.org/dr11/v1.1/spectrum/fits/{obsid}?token=F5fd29a16a5"

    print(f"正在读取星表文件: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"错误: 输入文件未找到，请检查路径: {csv_path}")
        return

    # 检查'obsid'列是否存在
    if 'obsid' not in df.columns:
        print(f"错误: 在文件 {csv_path} 中未找到 'obsid' 列。")
        return

    # 获取所有唯一的obsid，避免重复下载
    obsids = df['obsid'].unique()
    print(f"找到 {len(obsids)} 个唯一的obsid。")

    # --- 生成链接 ---
    links = [URL_TEMPLATE.format(obsid=obsid) for obsid in obsids]

    # --- 保存到文件 ---
    print(f"正在将 {len(links)} 个下载链接保存到: {output_txt_path}")
    try:
        with open(output_txt_path, 'w') as f:
            for link in links:
                f.write(link + "\n")
        print("文件保存成功！")
        print(f"您现在可以使用下载工具（如 wget -i {output_txt_path}）来批量下载这些文件。")
    except IOError as e:
        print(f"错误: 无法写入文件 {output_txt_path}。原因: {e}")

if __name__ == '__main__':
    # --- 文件路径配置 ---
    # 请将此脚本和筛选后的CSV文件放在同一目录下，或修改为绝对路径
    INPUT_CSV = 'supplement_samples.csv'
    OUTPUT_TXT = 'download_links_suplement_samples.txt'
    
    # 假设脚本和数据不在同一目录，如何构建路径
    # base_dir = '/home/irving/workspace/cemp_fiter'
    # input_path = os.path.join(base_dir, INPUT_CSV)
    # output_path = os.path.join(base_dir, OUTPUT_TXT)

    generate_download_links(INPUT_CSV, OUTPUT_TXT)
