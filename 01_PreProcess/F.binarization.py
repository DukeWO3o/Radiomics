from PIL import Image
import numpy as np
import os
import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_ROOT))

from pipeline_paths import IMAGE_ROOT

for group in ("RMPP", "nonRMPP"):
    input_folder = IMAGE_ROOT / group / "02_Mask"
    output_folder = IMAGE_ROOT / group / "03_Binarized"
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img = Image.open(os.path.join(input_folder, filename)).convert("L")
        img_array = np.array(img)
        binary_array = np.where(img_array > 0, 1, 0).astype(np.uint8)
        binary_img = Image.fromarray(binary_array)
        binary_img.save(os.path.join(output_folder, filename))
