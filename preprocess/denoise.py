import os
import cv2

def denoise_images(input_folder, output_folder):
    # 如果输出文件夹不存在，则创建它
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        # 构建完整的文件路径
        file_path = os.path.join(input_folder, filename)

        # 读取图像
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)

        # 使用非局部均值去噪方法进行去噪
        dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

        # 保存去噪后的图像到输出文件夹
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, dst)

# 使用方法
input_folder = "D:\\program\\potholes-detection\\dataset\\sz560\\alldata_sz560_filled"
output_folder = "D:\\program\\potholes-detection\\dataset\\sz560\\alldata_sz560_filled_denoise"
denoise_images(input_folder, output_folder)