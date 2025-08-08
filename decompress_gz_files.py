
import os
import gzip
import shutil

def decompress_all_gz(source_dir, dest_dir):
    """
    解压一个目录下的所有 .gz 文件到另一个目录。

    Args:
        source_dir (str): 包含 .gz 文件的源目录路径。
        dest_dir (str): 用于存放解压后文件的目标目录路径。
    """
    # 1. 检查源目录是否存在
    if not os.path.isdir(source_dir):
        print(f"错误: 源目录不存在 -> {source_dir}")
        return

    # 2. 确保目标目录存在，如果不存在则创建
    try:
        os.makedirs(dest_dir, exist_ok=True)
        print(f"目标目录已准备好: {dest_dir}")
    except OSError as e:
        print(f"错误: 无法创建目标目录 {dest_dir}。原因: {e}")
        return

    print(f"\n开始扫描源目录: {source_dir}")
    
    # 3. 遍历源目录中的所有文件
    decompressed_count = 0
    for filename in os.listdir(source_dir):
        if filename.endswith(".gz"):
            source_path = os.path.join(source_dir, filename)
            # 构建解压后的文件路径，去掉.gz后缀
            dest_filename = filename[:-3]
            dest_path = os.path.join(dest_dir, dest_filename)
            
            print(f"  -> 正在解压: {filename} ...", end='')
            try:
                with gzip.open(source_path, 'rb') as f_in:
                    with open(dest_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                print(" 完成")
                decompressed_count += 1
            except Exception as e:
                print(f"\n错误: 解压文件 {filename} 失败。原因: {e}")

    print(f"\n处理完毕！共成功解压 {decompressed_count} 个文件。")

if __name__ == '__main__':
    # --- 路径配置 ---
    # 请根据您的实际情况修改这两个文件夹路径
    
    # 假设您使用 wget -P ./fits_downloads/ 命令下载了文件
    SOURCE_FOLDER = './fits_downloads' 
    
    # 解压后的文件将存放在这里
    DESTINATION_FOLDER = './unzipped_fits'

    decompress_all_gz(SOURCE_FOLDER, DESTINATION_FOLDER)
