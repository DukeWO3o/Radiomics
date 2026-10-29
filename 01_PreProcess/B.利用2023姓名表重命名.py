import os
import pandas as pd
import shutil
import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_ROOT))

from pipeline_paths import CLINICAL_ROOT, IMAGE_ROOT

root_dir = IMAGE_ROOT / "source"
excel_path = CLINICAL_ROOT / "names.xlsx"


df = pd.read_excel(excel_path)
name_list = df['姓名'].astype(str).tolist()


for folder_name in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder_name)
    if not os.path.isdir(folder_path):
        continue


    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    if not txt_files:
        continue
    txt_path = os.path.join(folder_path, txt_files[0])


    try:
        with open(txt_path, 'r', encoding='utf-16-le') as f:
            content = f.read()
    except Exception as e:
        print(f"⚠️ 无法读取 {txt_path}，错误: {e}")
        continue


    matched_name = None
    for name in name_list:
        if name in content:
            matched_name = name
            break


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
