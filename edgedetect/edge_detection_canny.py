import cv2
import os

def extract_edges(input_dir, output_dir):
    # 列出输入目录中的所有文件
    image_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    for image_file in image_files:
        # 读取图像
        image_path = os.path.join(input_dir, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 以灰度模式读取图像

        # 使用Canny边缘检测器提取边缘
        edges = cv2.Canny(image, 100, 180)

        # 保存边缘图像到输出目录
        output_path = os.path.join(output_dir, image_file)
        cv2.imwrite(output_path, edges)


# 使用函数
input_directory = 'D:\\program\\potholes-detection\\dataset'  # 替换为您的坑洼图像目录
output_directory = 'D:\\program\\potholes-detection\\edgedetect'  # 替换为您想要保存边缘图像的目录
extract_edges(input_directory, output_directory)