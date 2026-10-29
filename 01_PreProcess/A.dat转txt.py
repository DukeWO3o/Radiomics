import os
import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_ROOT))

from pipeline_paths import IMAGE_ROOT

def rename_dat_to_txt(parent_dir):
    print(f"开始在目录中处理: {os.path.abspath(parent_dir)}")
    print("-" * 60)


    for root, dirs, files in os.walk(parent_dir):
        print(f"处理目录: {root}")
        dat_files = []


        for file in files:
            if file.lower().endswith('.dat'):
                dat_files.append(os.path.join(root, file))

        if not dat_files:
            print("  未找到dat文件")
            continue


        for dat_file in dat_files:

            txt_file = os.path.splitext(dat_file)[0] + '.txt'

            try:

                os.rename(dat_file, txt_file)
                print(f"  已重命名: {os.path.basename(dat_file)} -> {os.path.basename(txt_file)}")
            except Exception as e:
                print(f"  重命名失败: {os.path.basename(dat_file)} ({e})")

        print("-" * 60)

    print("所有.dat文件已重命名为.txt文件！")

if __name__ == "__main__":

    if len(sys.argv) > 1:
        target_directory = sys.argv[1]
    else:
        target_directory = IMAGE_ROOT / "source" / "00_DCM"

    if not os.path.exists(target_directory):
        print(f"错误: 目录不存在 - {target_directory}")
        sys.exit(1)

    print("=" * 60)
    print(f"开始处理目录: {target_directory}")
    print("此脚本将把所有.dat文件重命名为.txt文件")
    print("=" * 60)

    confirm = input("确认继续? (y/n): ").strip().lower()
    if confirm != 'y':
        print("操作已取消")
        sys.exit(0)

    rename_dat_to_txt(target_directory)
