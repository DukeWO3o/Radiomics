import os
import sys
from pathlib import Path
import numpy as np
import pydicom
import cv2
from PIL import Image

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_ROOT))

from pipeline_paths import IMAGE_ROOT


def apply_window(img, center, width):
    img = np.clip(img, center - width/2, center + width/2)
    img -= (center - width/2)
    img /= width
    img *= 255.0
    return img.astype(np.uint8)


def normalize_by_percentile(img, low=1, high=99):
    img = np.clip(img, np.percentile(img, low), np.percentile(img, high))
    img -= img.min()
    img /= img.max()
    img *= 255.0
    return img.astype(np.uint8)


def resample_image(img, ds, new_spacing=[0.5, 0.5]):
    if 'PixelSpacing' not in ds:
        return img
    spacing = [float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1])]
    resize_factor = [spacing[0] / new_spacing[0], spacing[1] / new_spacing[1]]
    new_size = [int(img.shape[0] * resize_factor[0]), int(img.shape[1] * resize_factor[1])]
    return cv2.resize(img, (new_size[1], new_size[0]), interpolation=cv2.INTER_AREA)


def resize_with_padding(img, target_size=(1024, 1024), pad_value=0):
    old_h, old_w = img.shape[:2]
    target_w, target_h = target_size


    scale = min(target_w / old_w, target_h / old_h)
    new_w = int(old_w * scale)
    new_h = int(old_h * scale)


    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


    canvas = np.full((target_h, target_w), pad_value, dtype=img.dtype)
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    canvas[top:top+new_h, left:left+new_w] = resized

    return canvas


def should_invert_image(ds, img, threshold=128):
    photometric = str(getattr(ds, 'PhotometricInterpretation', '')).strip().upper()
    lut_shape = str(getattr(ds, 'PresentationLUTShape', '')).strip().upper()
    if photometric == 'MONOCHROME1':
        return True
    if photometric == 'MONOCHROME2':
        return False
    if lut_shape == 'INVERSE':
        return True
    if lut_shape == 'IDENTITY':
        return False

    border = np.concatenate([
        img[:2].ravel(), img[-2:].ravel(), img[:, :2].ravel(), img[:, -2:].ravel()
    ])
    return float(np.median(border)) > threshold


def standardize_dcm(dcm_path, output_path, final_size=(1024, 1024)):
    ds = pydicom.dcmread(dcm_path)
    img = ds.pixel_array.astype(np.float32)


    if 'WindowCenter' in ds and 'WindowWidth' in ds:
        wc = ds.WindowCenter[0] if isinstance(ds.WindowCenter, pydicom.multival.MultiValue) else ds.WindowCenter
        ww = ds.WindowWidth[0] if isinstance(ds.WindowWidth, pydicom.multival.MultiValue) else ds.WindowWidth
        img = apply_window(img, float(wc), float(ww))
    else:
        img = normalize_by_percentile(img)


    img = resample_image(img, ds, new_spacing=[0.5, 0.5])


    if should_invert_image(ds, img):
        img = 255 - img


    img = resize_with_padding(img, target_size=final_size, pad_value=0)


    Image.fromarray(img).convert('L').save(output_path)


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

for group in ("RMPP", "nonRMPP"):
    process_dcm_folder(
        IMAGE_ROOT / group / "00_DCM",
        IMAGE_ROOT / group / "01_PNG",
    )
