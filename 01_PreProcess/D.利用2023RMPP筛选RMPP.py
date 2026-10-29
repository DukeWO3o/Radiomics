import os
import shutil
import pandas as pd
import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_ROOT))

from pipeline_paths import CLINICAL_ROOT, IMAGE_ROOT

dcm_folder = IMAGE_ROOT / "source" / "00_DCM"
target_folder = IMAGE_ROOT / "RMPP" / "00_DCM"
name_file = CLINICAL_ROOT / "rmpp_names.xlsx"


if name_file.suffix == '.xlsx':
    df = pd.read_excel(name_file)
elif name_file.suffix == '.csv':
    df = pd.read_csv(name_file)
else:
    raise ValueError("不支持的文件格式，请提供 .xlsx 或 .csv 文件")


name_list = df['姓名'].astype(str).tolist()


for filename in os.listdir(dcm_folder):
    if filename.lower().endswith('.dcm'):
        for name in name_list:
            if name in filename:
                src_path = os.path.join(dcm_folder, filename)
                dst_path = os.path.join(target_folder, filename)
                shutil.move(src_path, dst_path)
                print(f"已移动文件: {filename}")
                break

print("文件处理完成。")
