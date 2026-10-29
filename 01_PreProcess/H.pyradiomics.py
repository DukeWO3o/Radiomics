import os
from radiomics import featureextractor
import pandas as pd
from tqdm import tqdm
import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_ROOT))

from pipeline_paths import FEATURE_OUTPUT_ROOT, IMAGE_ROOT

config_file = Path(__file__).with_name("radiomics_config.yaml")
extractor = featureextractor.RadiomicsFeatureExtractor(str(config_file))

for group in ("RMPP", "nonRMPP"):
    image_dir = IMAGE_ROOT / group / "01_PNG"
    mask_dir = IMAGE_ROOT / group / "03_Binarized"
    output_csv = FEATURE_OUTPUT_ROOT / f"{group}_radiomics.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results = []

    for filename in tqdm(os.listdir(image_dir)):
        if not filename.lower().endswith('.png'):
            continue
        image_path = image_dir / filename
        mask_path = mask_dir / filename
        if not mask_path.exists():
            continue
        feature_vector = extractor.execute(str(image_path), str(mask_path))
        feature_vector['image'] = filename
        results.append(feature_vector)

    pd.DataFrame(results).to_csv(output_csv, index=False)
