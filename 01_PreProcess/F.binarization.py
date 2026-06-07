from PIL import Image
import numpy as np
import os

input_folder = 'F:/2023original/DCM-RMPP/02_Mask'
output_folder = 'F:/2023original/DCM-RMPP/03_Binarized'
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    # 读取图像并转为灰度
    img = Image.open(os.path.join(input_folder, filename)).convert("L")
    img_array = np.array(img)

    # 二值化：非0设为1
    binary_array = np.where(img_array > 0, 1, 0).astype(np.uint8)

    # 保存为8位PNG（仍然是0/1灰度，但可以被SimpleITK/PyRadiomics读取）
    binary_img = Image.fromarray(binary_array)
    binary_img.save(os.path.join(output_folder, filename))
    print(f"Saved binary mask with 1/0: {filename}")
