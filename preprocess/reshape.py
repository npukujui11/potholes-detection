from PIL import Image, ImageFile
import os

# 允许加载截断的图像
ImageFile.LOAD_TRUNCATED_IMAGES = True


def resize_images(input_dir, output_dir, size=(560, 560)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            image_path = os.path.join(input_dir, filename)
            try:
                img = Image.open(image_path)
                img = img.resize(size, Image.ANTIALIAS)

                # 构建输出文件路径
                output_image_path = os.path.join(output_dir, filename)
                img.save(output_image_path)
                print(f"Resized and saved {filename} to {output_image_path}")
            except IOError:
                print(f"Cannot resize {filename}. It may be corrupted or incomplete.")


# 设置源文件夹和目标文件夹路径
source_folder = 'D:\\program\\potholes-detection\\dataset\\orialldata'
target_folder = 'D:\\program\\potholes-detection\\dataset\\sz560\\alldata_sz560'
os.makedirs(target_folder, exist_ok=True)

# 调用函数
resize_images(source_folder, target_folder)