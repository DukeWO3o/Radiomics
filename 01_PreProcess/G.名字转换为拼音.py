import os
from pypinyin import lazy_pinyin
from collections import defaultdict

# 文件夹路径
folder_path = 'F:/2023/DCM-RMPP/03_Binarized/backup'

# 用于记录拼音名出现次数
name_count = defaultdict(int)

# 遍历文件夹
for filename in os.listdir(folder_path):
    old_path = os.path.join(folder_path, filename)

    if not os.path.isfile(old_path) or '.' not in filename:
        continue

    name, ext = os.path.splitext(filename)
    pinyin_base = ''.join(lazy_pinyin(name)).lower()

    # 增加编号（如 zhangsan.png → zhangsan_1.png）
    new_name = pinyin_base + ext
    while os.path.exists(os.path.join(folder_path, new_name)):
        name_count[pinyin_base] += 1
        new_name = f"{pinyin_base}_{name_count[pinyin_base]}{ext}"

    new_path = os.path.join(folder_path, new_name)
    os.rename(old_path, new_path)
    print(f"Renamed: {filename} → {new_name}")
