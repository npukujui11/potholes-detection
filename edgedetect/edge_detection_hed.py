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

# 使用函数
image_path = 'D:\\program\\potholes-detection\\dataset\\potholes1.jpg'  # 替换为您的图像路径
model_path = 'D:\\program\\potholes-detection\\edgedetect\\hed_pretrained_bsds.caffemodel'  # 替换为您下载的模型权重路径
config_path = 'D:\\program\\potholes-detection\\edgedetect\\deploy.prototxt'  # 替换为您下载的配置文件路径

edge_image = hed_edge_detection(image_path, model_path, config_path)
cv2.imwrite('edge_output.jpg', edge_image)
