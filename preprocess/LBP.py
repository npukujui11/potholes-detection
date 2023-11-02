# LBP (Local Binary Pattern)
# LBP是一个用于描述图像局部的纹理信息的特征描述符。它基于像素与其邻域像素的相对亮度来编码纹理信息。
import os
import cv2
import numpy as np
from skimage import feature
import matplotlib.pyplot as plt


def calculate_lbp_image(image_path):
    # 读取图像
    original_image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 计算LBP图像
    radius = 1
    n_points = 8 * radius
    lbp_image = feature.local_binary_pattern(original_image_gray, n_points, radius, method='uniform')
    lbp_image = np.uint8((lbp_image / lbp_image.max()) * 255)

    return lbp_image

# 输入和输出文件夹路径
input_folder = 'D:\\program\\potholes-detection\\dataset\\sz560\\alldata_sz560_filled_denoise'  # 替换为您的输入图像文件夹路径
output_folder = 'D:\\program\\potholes-detection\\dataset\\sz560\\alldata_sz560_filled_denoise_lbp'  # 替换为您希望保存LBP图像的文件夹路径
os.makedirs(output_folder, exist_ok=True)

# 遍历输入文件夹中的所有图像
for image_name in os.listdir(input_folder):
    image_path = os.path.join(input_folder, image_name)
    lbp_image = calculate_lbp_image(image_path)

    # 保存LBP图像到输出文件夹
    output_path = os.path.join(output_folder, image_name)
    cv2.imwrite(output_path, lbp_image)