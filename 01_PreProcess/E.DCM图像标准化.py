import os
import numpy as np
import pydicom
import cv2
from PIL import Image

input_folder = 'F:/2023original/DCM-RMPP/00_DCM'
output_folder = 'F:/2023original/DCM-RMPP/01_PNG'

# 灰度窗宽窗位处理
def apply_window(img, center, width):
    img = np.clip(img, center - width/2, center + width/2)
    img -= (center - width/2)
    img /= width
    img *= 255.0
    return img.astype(np.uint8)

# 百分位归一化
def normalize_by_percentile(img, low=1, high=99):
    img = np.clip(img, np.percentile(img, low), np.percentile(img, high))
    img -= img.min()
    img /= img.max()
    img *= 255.0
    return img.astype(np.uint8)

# 空间分辨率重采样
def resample_image(img, ds, new_spacing=[0.5, 0.5]):
    if 'PixelSpacing' not in ds:
        return img
    spacing = [float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1])]
    resize_factor = [spacing[0] / new_spacing[0], spacing[1] / new_spacing[1]]
    new_size = [int(img.shape[0] * resize_factor[0]), int(img.shape[1] * resize_factor[1])]
    return cv2.resize(img, (new_size[1], new_size[0]), interpolation=cv2.INTER_AREA)

# 保持长宽比 + 边缘填充
def resize_with_padding(img, target_size=(1024, 1024), pad_value=0):
    old_h, old_w = img.shape[:2]
    target_w, target_h = target_size

    # 计算缩放比例
    scale = min(target_w / old_w, target_h / old_h)
    new_w = int(old_w * scale)
    new_h = int(old_h * scale)

    # 缩放图像
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 创建画布并填充图像
    canvas = np.full((target_h, target_w), pad_value, dtype=img.dtype)
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    canvas[top:top+new_h, left:left+new_w] = resized

    return canvas

# 自动判断图像是否需要反转（使背景为黑）
def should_invert_image(img, threshold=128):
    h, w = img.shape
    corners = [img[0:10, 0:10], img[0:10, -10:], img[-10:, 0:10], img[-10:, -10:]]
    background_mean = np.mean([np.mean(c) for c in corners])
    return background_mean > threshold

# 图像标准化流程整合
def standardize_dcm(dcm_path, output_path, final_size=(1024, 1024)):
    ds = pydicom.dcmread(dcm_path)
    img = ds.pixel_array.astype(np.float32)

    # 灰度标准化
    if 'WindowCenter' in ds and 'WindowWidth' in ds:
        wc = ds.WindowCenter[0] if isinstance(ds.WindowCenter, pydicom.multival.MultiValue) else ds.WindowCenter
        ww = ds.WindowWidth[0] if isinstance(ds.WindowWidth, pydicom.multival.MultiValue) else ds.WindowWidth
        img = apply_window(img, float(wc), float(ww))
    else:
        img = normalize_by_percentile(img)

    # 空间重采样
    img = resample_image(img, ds, new_spacing=[0.5, 0.5])

    # 黑白统一（反转）
    if should_invert_image(img):
        img = 255 - img

    # 尺寸标准化 + 保持长宽比
    img = resize_with_padding(img, target_size=final_size, pad_value=0)

    # 保存为 PNG
    Image.fromarray(img).convert('L').save(output_path)

# 批量处理 DICOM 文件夹
def process_dcm_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".dcm"):
            dcm_path = os.path.join(input_folder, filename)
            output_name = os.path.splitext(filename)[0] + ".png"
            output_path = os.path.join(output_folder, output_name)
            try:
                standardize_dcm(dcm_path, output_path)
                print(f"Processed: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

process_dcm_folder(input_folder, output_folder)