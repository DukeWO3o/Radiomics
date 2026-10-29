import os
import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_ROOT))

from pipeline_paths import IMAGE_ROOT

root_dir = IMAGE_ROOT / "source"


for folder_name in os.listdir(root_dir):
    first_level_path = os.path.join(root_dir, folder_name)

    if os.path.isdir(first_level_path):

        for subfolder_name in os.listdir(first_level_path):
            second_level_path = os.path.join(first_level_path, subfolder_name)

            if os.path.isdir(second_level_path):

                for filename in os.listdir(second_level_path):
                    if filename.lower().endswith(".dcm"):
                        old_path = os.path.join(second_level_path, filename)
                        new_filename = f"{folder_name}_{filename}"
                        new_path = os.path.join(second_level_path, new_filename)

                        os.rename(old_path, new_path)
                        print(f"已重命名：{old_path} -> {new_path}")
