import os
from PIL import Image

# 源目录和目标目录
source_dir = "/dataset/alldata_sz640"
target_dir = "D:\\program\\potholes-detection\\dataset\\alldata_gray"

# 如果目标目录不存在，则创建它
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# 遍历源目录中的所有文件
for filename in os.listdir(source_dir):
    # 检查文件是否为图像
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
        # 打开图像
        image_path = os.path.join(source_dir, filename)
        image = Image.open(image_path)

        # 转换为灰度
        gray_image = image.convert("L")

        # 保存到目标目录
        gray_image_path = os.path.join(target_dir, filename)
        gray_image.save(gray_image_path)
