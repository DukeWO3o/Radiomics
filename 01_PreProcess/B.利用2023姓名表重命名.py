import os
import pandas as pd
import shutil

# 设置路径
root_dir = 'F:\\2023\\12DCM'   # 请替换为你的总文件夹路径
excel_path = 'F:\\2023\\2023姓名.xlsx'   # 如果不在同目录，请写绝对路径

# 读取姓名列
df = pd.read_excel(excel_path)
name_list = df['姓名'].astype(str).tolist()

# 遍历每个文件夹
for folder_name in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder_name)
    if not os.path.isdir(folder_path):
        continue

    # 查找txt文件
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    if not txt_files:
        continue
    txt_path = os.path.join(folder_path, txt_files[0])

    # 读取内容
    try:
        with open(txt_path, 'r', encoding='utf-16-le') as f:
            content = f.read()
    except Exception as e:
        print(f"⚠️ 无法读取 {txt_path}，错误: {e}")
        continue

    # 匹配姓名
    matched_name = None
    for name in name_list:
        if name in content:
            matched_name = name
            break

    # 重命名文件夹，自动编号
    if matched_name:
        count = 1
        while True:
            new_folder_name = f"{count}_{matched_name}"
            new_folder_path = os.path.join(root_dir, new_folder_name)
            if not os.path.exists(new_folder_path):
                break
            count += 1

        shutil.move(folder_path, new_folder_path)
        print(f"✅ {folder_name} ➜ {new_folder_name}")
