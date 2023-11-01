import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 数据集目录
data_dir = "D:\\program\\potholes-detection\\dataset\\orialldata"

# 初始化列表来存储每个通道的平均强度
red_values_pothole = []
green_values_pothole = []
blue_values_pothole = []

red_values_normal = []
green_values_normal = []
blue_values_normal = []

# 遍历数据集目录中的所有文件
for filename in os.listdir(data_dir):
    # 打开图片
    image_path = os.path.join(data_dir, filename)
    image = Image.open(image_path)
    image_data = np.array(image)

    # 计算每个通道的平均强度
    red_avg = np.mean(image_data[:, :, 0])
    green_avg = np.mean(image_data[:, :, 1])
    blue_avg = np.mean(image_data[:, :, 2])

    # 根据文件名将平均强度分配到相应的类别
    if filename.startswith("pothole"):
        red_values_pothole.append(red_avg)
        green_values_pothole.append(green_avg)
        blue_values_pothole.append(blue_avg)
    elif filename.startswith("normal"):
        red_values_normal.append(red_avg)
        green_values_normal.append(green_avg)
        blue_values_normal.append(blue_avg)

# 绘制归一化的直方图
def plot_normalized_histogram(values_pothole, values_normal, channel_name):
    plt.figure(figsize=(6, 6))

    # 定义 bins 的边缘
    bins = np.linspace(min(min(values_pothole), min(values_normal)),
                       max(max(values_pothole), max(values_normal)), 15)

    # 使用 density=True 来得到归一化的直方图
    plt.hist(values_pothole, bins=bins, alpha=0.5, label='Pothole', color='red',
             edgecolor='black', rwidth=0.4, align='mid', density=True)
    plt.hist(values_normal, bins=bins, alpha=0.5, label='Normal', color='blue',
             edgecolor='black', rwidth=0.4, align='left', density=True)

    plt.title(f'{channel_name} Channel Average Intensity Distribution (Normalized)')
    plt.xlabel('Average Intensity')
    plt.ylabel('Probability Density')
    plt.legend(loc='upper right')

    # 保存图像到当前文件夹
    plt.savefig(f"{channel_name}_distribution.png")
    plt.show()

plot_normalized_histogram(red_values_pothole, red_values_normal, 'Red')
plot_normalized_histogram(green_values_pothole, green_values_normal, 'Green')
plot_normalized_histogram(blue_values_pothole, blue_values_normal, 'Blue')

