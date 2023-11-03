import os
import csv
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# 加载模型
model = load_model('D:\\program\\potholes-detection\\train\\inception_resnet_v2_fused_features_denoise.h5')

def predict_image(model, image_path, feature_dirs):
    # 加载原始图像
    img = load_img(image_path, color_mode="grayscale", target_size=(640, 640))
    img_array = img_to_array(img)

    # 加载特征图
    features = [img_array]
    for feature_dir in feature_dirs:
        feature_path = os.path.join(feature_dir, os.path.basename(image_path))
        if os.path.exists(feature_path):
            feature_img = load_img(feature_path, color_mode="grayscale", target_size=(640, 640))
            feature_img_array = img_to_array(feature_img)
            features.append(feature_img_array)

    # 合并通道
    merged = np.concatenate(features, axis=-1)
    merged = np.expand_dims(merged, axis=0)
    predictions = model.predict(merged)
    return np.argmax(predictions, axis=1)[0]

# 测试数据文件夹
test_data_folder = "D:\\program\\potholes-detection\\dataset\\testdata\\testdata_sz640"
feature_dirs = ["D:\\program\\potholes-detection\\dataset\\testdata\\testdata_sz640_filled_denoise_edge",
                "D:\\program\\potholes-detection\\dataset\\testdata\\testdata_sz640_filled_denoise_hog",
                "D:\\program\\potholes-detection\\dataset\\testdata\\testdata_sz640_filled_denoise_lbp"]
# 创建CSV文件
with open('test_inceptionresnetv2_640sz_denoise_result.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["fnames", "label"])  # 写入标题

    # 对测试数据文件夹中的每个文件进行预测
    for filename in os.listdir(test_data_folder):
        if filename.lower().endswith('.jpg'):
            file_path = os.path.join(test_data_folder, filename)
            prediction = predict_image(model, file_path, feature_dirs)
            writer.writerow([filename, prediction])

print("test_inceptionresnetv2_640sz_result.csv has been generated.")