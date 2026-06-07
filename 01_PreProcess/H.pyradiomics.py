import os
import radiomics
from radiomics import featureextractor
import pandas as pd
from tqdm import tqdm

# 输入路径
image_dir = r'F:/2023original/DCM-RMPP/01_PNG'
mask_dir = r'F:/2023original/DCM-RMPP/03_Binarized'
output_csv = r'F:/2023original/DCM-RMPP/04_Pyradiomics/RMPP-radiomics_features_.csv'
config_file = r'F:/2023/Pyradiomics结果/radiomics_config.yaml'

# 初始化提取器（从 YAML 文件加载）
extractor = featureextractor.RadiomicsFeatureExtractor(config_file)

# 提取特征
results = []
for filename in tqdm(os.listdir(image_dir)):
    if filename.lower().endswith('.png'):
        image_path = os.path.join(image_dir, filename)
        mask_path = os.path.join(mask_dir, filename)

        if not os.path.exists(mask_path):
            print(f"❗ 掩膜缺失：{mask_path}")
            continue

        try:
            feature_vector = extractor.execute(image_path, mask_path)
            feature_vector['image'] = filename
            results.append(feature_vector)
        except Exception as e:
            print(f"❌ 错误：{filename} - {e}")

# 保存到 CSV
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"✅ 提取完成，保存至 {output_csv}")
