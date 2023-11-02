import torch
import cv2
import os

# 加载YOLOv5模型
model_path = 'D:\\program\\potholes-detection\\model\\yolov5-master\\runs\\train\\exp\\weights\\best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
model.eval()

# 图像目录
image_dir = 'D:\\program\\potholes-detection\\dataset\\sz560\\alldata_sz560'
output_dir = 'D:\\program\\potholes-detection\\dataset\\sz560\\alldata_sz560_filled'
os.makedirs(output_dir, exist_ok=True)

# 遍历图像目录
for image_name in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_name)
    image = cv2.imread(image_path)

    # 使用YOLOv5进行目标检测
    results = model(image_path)

    # 遍历检测结果
    for *xyxy, conf, cls in results.pred[0]:
        if model.names[int(cls)] == 'car':
            x1, y1, x2, y2 = map(int, xyxy)
            # 填充检测到的"car"目标为白色
            image[y1:y2, x1:x2] = [255, 255, 255]

    # 保存处理后的图像
    output_path = os.path.join(output_dir, image_name)
    cv2.imwrite(output_path, image)
