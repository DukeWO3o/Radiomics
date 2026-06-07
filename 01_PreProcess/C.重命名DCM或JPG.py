import os

# 设置根目录路径
root_dir = r"F:\2023\12筛选后DCM"  # 修改为你实际的路径

# 遍历根目录下的所有一级文件夹
for folder_name in os.listdir(root_dir):
    first_level_path = os.path.join(root_dir, folder_name)

    if os.path.isdir(first_level_path):
        # 遍历一级文件夹内所有二级文件夹
        for subfolder_name in os.listdir(first_level_path):
            second_level_path = os.path.join(first_level_path, subfolder_name)

            if os.path.isdir(second_level_path):
                # 遍历并重命名jpg文件
                for filename in os.listdir(second_level_path):
                    if filename.lower().endswith(".dcm"):  # 修改为需要的后缀
                        old_path = os.path.join(second_level_path, filename)
                        new_filename = f"{folder_name}_{filename}"
                        new_path = os.path.join(second_level_path, new_filename)

                        os.rename(old_path, new_path)
                        print(f"已重命名：{old_path} -> {new_path}")
