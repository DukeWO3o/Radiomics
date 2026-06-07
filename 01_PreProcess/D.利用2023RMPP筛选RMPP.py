import os
import shutil
import pandas as pd

# 设置路径
dcm_folder = r"F:\2023\07-12DCM"          # 原始dcm文件所在文件夹
target_folder = r"F:\2023\07-12DCM-RMPP"   # 匹配后要移动到的目标文件夹
name_file = r"F:\2023\2023RMPP.xlsx"      # 存放姓名的表格文件（可为xlsx或csv）

# 读取姓名表格
if name_file.endswith('.xlsx'):
    df = pd.read_excel(name_file)
elif name_file.endswith('.csv'):
    df = pd.read_csv(name_file)
else:
    raise ValueError("不支持的文件格式，请提供 .xlsx 或 .csv 文件")

# 获取所有姓名，假设姓名列为“姓名”，可根据实际修改
name_list = df['姓名'].astype(str).tolist()

# 遍历并比对 DICOM 文件
for filename in os.listdir(dcm_folder):
    if filename.lower().endswith('.dcm'):
        for name in name_list:
            if name in filename:
                src_path = os.path.join(dcm_folder, filename)
                dst_path = os.path.join(target_folder, filename)
                shutil.move(src_path, dst_path)
                print(f"已移动文件: {filename}")
                break  # 找到一个姓名匹配后就跳出循环

print("文件处理完成。")