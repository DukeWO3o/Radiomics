import os
import sys

def rename_dat_to_txt(parent_dir):
    """
    将指定目录及其子目录中的所有.dat文件重命名为.txt文件
    """
    print(f"开始在目录中处理: {os.path.abspath(parent_dir)}")
    print("-" * 60)
    
    # 遍历所有子目录
    for root, dirs, files in os.walk(parent_dir):
        print(f"处理目录: {root}")
        dat_files = []
        
        # 查找当前目录中的所有dat文件
        for file in files:
            if file.lower().endswith('.dat'):
                dat_files.append(os.path.join(root, file))
        
        if not dat_files:
            print("  未找到dat文件")
            continue
        
        # 重命名每个dat文件
        for dat_file in dat_files:
            # 构建新的txt文件路径
            txt_file = os.path.splitext(dat_file)[0] + '.txt'
            
            try:
                # 执行重命名
                os.rename(dat_file, txt_file)
                print(f"  已重命名: {os.path.basename(dat_file)} -> {os.path.basename(txt_file)}")
            except Exception as e:
                print(f"  重命名失败: {os.path.basename(dat_file)} ({e})")
        
        print("-" * 60)
    
    print("所有.dat文件已重命名为.txt文件！")

if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) > 1:
        target_directory = sys.argv[1]
    else:
        # 如果没有提供参数，使用当前目录
        target_directory = os.getcwd()
    
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