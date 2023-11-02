import cv2
import numpy as np
import os


def hed_edge_detection(image_path, model_path, config_path):
    # 加载预训练的HED模型
    net = cv2.dnn.readNetFromCaffe(config_path, model_path)

    # 读取图像
    image = cv2.imread(image_path)
    (H, W) = image.shape[:2]

    # 将图像转换为blob并进行前向传递
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(W, H),
                                 mean=(104.00698793, 116.66876762, 122.67891434),
                                 swapRB=False, crop=False)
    net.setInput(blob)
    hed = net.forward()
    hed = cv2.resize(hed[0, 0], (W, H))
    hed = (255 * hed).astype("uint8")

    return hed

# 文件夹路径
input_folder = 'D:\\program\\potholes-detection\\dataset\\alldata_sz640_filled'  # 图像文件夹路径
output_folder = 'D:\\program\\potholes-detection\\dataset\\alldata_sz640_filled_edge'  # 输出边缘图的文件夹路径
os.makedirs(output_folder, exist_ok=True)

model_path = 'D:\\program\\potholes-detection\\edgedetect\\hed_pretrained_bsds.caffemodel'  # 替换为您下载的模型权重路径
config_path = 'D:\\program\\potholes-detection\\edgedetect\\hed_deploy.prototxt'  # 替换为您下载的配置文件路径

# 遍历文件夹中的所有图片
for image_name in os.listdir(input_folder):
    image_path = os.path.join(input_folder, image_name)
    edge_image = hed_edge_detection(image_path, model_path, config_path)
    output_path = os.path.join(output_folder, image_name)
    cv2.imwrite(output_path, edge_image)
