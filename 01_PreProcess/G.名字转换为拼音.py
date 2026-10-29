import os
from pypinyin import lazy_pinyin
from collections import defaultdict
import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_ROOT))

from pipeline_paths import IMAGE_ROOT

for group in ("RMPP", "nonRMPP"):
    folder_path = IMAGE_ROOT / group / "03_Binarized"
    name_count = defaultdict(int)
    for filename in os.listdir(folder_path):
        old_path = os.path.join(folder_path, filename)
        if not os.path.isfile(old_path) or '.' not in filename:
            continue
        name, ext = os.path.splitext(filename)
        pinyin_base = ''.join(lazy_pinyin(name)).lower()
        new_name = pinyin_base + ext
        while os.path.exists(os.path.join(folder_path, new_name)):
            name_count[pinyin_base] += 1
            new_name = f"{pinyin_base}_{name_count[pinyin_base]}{ext}"
        os.rename(old_path, os.path.join(folder_path, new_name))
