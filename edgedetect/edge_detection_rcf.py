import cv2
import os
def rcf_edge_detection(image_path, model_path, config_path):
    # 加载预训练的RCF模型
    net = cv2.dnn.readNetFromCaffe(config_path, model_path)

    # 读取图像
    image = cv2.imread(image_path)
    (H, W) = image.shape[:2]

    # 将图像转换为blob并进行前向传递
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(W, H),
                                 mean=(104.00698793, 116.66876762, 122.67891434),
                                 swapRB=False, crop=False)
    net.setInput(blob)
    rcf = net.forward()
    rcf = cv2.resize(rcf[0, 0], (W, H))
    rcf = (255 * rcf).astype("uint8")

    return rcf

# 文件夹路径
input_folder = 'D:\\program\\potholes-detection\\dataset\\sz640\\alldata_sz640_denoise'  # 图像文件夹路径
output_folder = 'D:\\program\\potholes-detection\\dataset\\sz640\\alldata_sz640_denoise_edge'  # 输出边缘图的文件夹路径
os.makedirs(output_folder, exist_ok=True)

model_path = 'D:\\program\\potholes-detection\\edgedetect\\rcf_resnet101_bsds_iter_40000.caffemodel'  # 替换为您下载的RCF模型权重路径
config_path = 'D:\\program\\potholes-detection\\edgedetect\\rcf_resnet101.prototxt'  # 替换为您下载的配置文件路径

# 遍历文件夹中的所有图片
for image_name in os.listdir(input_folder):
    image_path = os.path.join(input_folder, image_name)
    edge_image = rcf_edge_detection(image_path, model_path, config_path)
    output_path = os.path.join(output_folder, image_name)
    cv2.imwrite(output_path, edge_image)
