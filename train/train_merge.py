import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# 数据路径
data_dirs = {
    'original': 'D:\\program\\potholes-detection\\dataset\\alldata_filled',
    'edge': 'D:\\program\\potholes-detection\\dataset\\alldata_filled_edge',
    'hog': 'D:\\program\\potholes-detection\\dataset\\alldata_filled_hog',
    'lbp': 'D:\program\\potholes-detection\\dataset\\alldata_filled_lbp'
}

# 参数
img_width, img_height = 640, 640
batch_size = 32
epochs = 50
num_classes = 2


# 加载数据并融合特征
def load_and_merge_data(data_dirs):
    X = []
    y = []
    for image_name in os.listdir(data_dirs['original']):
        original_img = cv2.imread(os.path.join(data_dirs['original'], image_name), cv2.IMREAD_GRAYSCALE)
        edge_img = cv2.imread(os.path.join(data_dirs['edge'], image_name), cv2.IMREAD_GRAYSCALE)
        hog_img = cv2.imread(os.path.join(data_dirs['hog'], image_name), cv2.IMREAD_GRAYSCALE)
        lbp_img = cv2.imread(os.path.join(data_dirs['lbp'], image_name), cv2.IMREAD_GRAYSCALE)

        merged_img = np.dstack([original_img, edge_img, hog_img, lbp_img])
        X.append(merged_img)

        label = 1 if 'pothole' in image_name else 0
        y.append(label)

    return np.array(X), np.array(y)


X, y = load_and_merge_data(data_dirs)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, stratify=y)

# 数据增强
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(X_train)

# 构建模型
input_tensor = Input(shape=(img_width, img_height, 4))
base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_tensor=input_tensor)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 类权重
class_weights = {
    0: 1.0,  # normal
    1: (266 / 35)  # potholes
}

# 训练模型
model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
          validation_data=(X_val, y_val),
          steps_per_epoch=len(X_train) // batch_size,
          epochs=epochs,
          class_weight=class_weights)

# 保存模型
model.save('inception_resnet_v2_fused_features.h5')

# 验证模型
loss, accuracy = model.evaluate(X_val, y_val)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy*100:.2f}%")