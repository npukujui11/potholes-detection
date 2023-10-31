# HOG (Histogram of Oriented Gradients)
# HOG是一个用于描述图像局部的形状信息和纹理信息的特征描述符。它基于图像的梯度方向来构建直方图。
import os
import matplotlib.pyplot as plt
import cv2
from skimage import feature
import numpy as np

def calculate_hog_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    hog_features, hog_image = feature.hog(
        image,
        pixels_per_cell=(8, 8),  # Cell size in pixels
        cells_per_block=(2, 2),  # Number of cells in each block
        visualize=True,
        block_norm='L2-Hys'
    )
    return hog_image


# 输入和输出文件夹路径
input_folder = 'D:\\program\\potholes-detection\\dataset\\alldata_filled'  # 输入图像文件夹路径
output_folder = 'D:\\program\\potholes-detection\\dataset\\alldata_filled_hog'  # 保存HOG图像的文件夹路径
os.makedirs(output_folder, exist_ok=True)

# 遍历输入文件夹中的所有图像
for image_name in os.listdir(input_folder):
    image_path = os.path.join(input_folder, image_name)
    hog_image = calculate_hog_image(image_path)

    # 保存HOG图像到输出文件夹
    output_path = os.path.join(output_folder, image_name)
    plt.imsave(output_path, hog_image, cmap='gray')