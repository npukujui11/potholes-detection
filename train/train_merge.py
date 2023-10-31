import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Multiply, Flatten, Softmax, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import InceptionResNetV2
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.utils import shuffle

# 数据加载和预处理
def load_data(data_dir, feature_dirs):
    images = []
    labels = []
    for filename in os.listdir(data_dir):
        if filename.startswith("pothole"):
            label = [1, 0]
        elif filename.startswith("normal"):
            label = [0, 1]
        else:
            continue

        # 加载原始图像
        img_path = os.path.join(data_dir, filename)
        img = load_img(img_path, color_mode="grayscale", target_size=(640, 640))
        img = img_to_array(img)

        # 加载特征图
        features = [img]
        for feature_dir in feature_dirs:
            feature_path = os.path.join(feature_dir, filename)
            feature_img = load_img(feature_path, color_mode="grayscale", target_size=(640, 640))
            feature_img = img_to_array(feature_img)
            features.append(feature_img)

        # 合并通道
        merged = np.concatenate(features, axis=-1)
        images.append(merged)
        labels.append(label)

    return np.array(images), np.array(labels)

data_dir = "D:\\program\\potholes-detection\\dataset\\alldata"
feature_dirs = ["D:\\program\\potholes-detection\\dataset\\alldata_filled_edge",
                "D:\\program\\potholes-detection\\dataset\\alldata_filled_hog",
                "D:\\program\\potholes-detection\\dataset\\alldata_filled_lbp"]

X, y = load_data(data_dir, feature_dirs)
X, y = shuffle(X, y, random_state=42)  # 打乱数据

# 划分训练和验证数据
split = int(0.9 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# 注意力模块
def attention_module(input_tensor, channels=4):
    # 使用一个小型的卷积网络计算注意力权重
    attention_probs = Conv2D(channels, kernel_size=(1, 1), activation='softmax', padding='same',
                             name='attention_probs')(input_tensor)

    # 为边缘图（第二个通道）分配更高的初始权重
    initial_bias = tf.constant_initializer([-2., 2., -2., -2.])  # 偏向于第二个通道，即边缘图
    attention_probs_bias = Conv2D(channels, kernel_size=(1, 1), kernel_initializer='zeros',
                                  bias_initializer=initial_bias, activation='softmax', padding='same',
                                  name='attention_probs_bias')(input_tensor)

    # 将注意力权重与输入张量相乘
    attention_mul = Multiply(name='attention_multiply')([input_tensor, attention_probs_bias])

    return attention_mul

# 模型定义
def create_attention_model(input_shape):
    input_tensor = Input(shape=input_shape)
    attention_output = attention_module(input_tensor)

    base_model = InceptionResNetV2(include_top=False, weights=None, input_tensor=attention_output)
    base_features = base_model.output

    gap = GlobalAveragePooling2D()(base_features)
    fc1 = Dense(128, activation='relu')(gap)
    dropout = Dropout(0.5)(fc1)  # 添加Dropout层
    output = Dense(2, activation='softmax')(dropout)

    model = Model(inputs=input_tensor, outputs=output)
    return model

input_shape = (640, 640, 4)
model = create_attention_model(input_shape)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 使用数据增强和类权重处理数据不平衡问题
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

datagen = ImageDataGenerator(rotation_range=20,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             horizontal_flip=True)
datagen.fit(X_train)

# 计算类权重
y_integers = np.argmax(y_train, axis=1)
class_weights = compute_class_weight('balanced', classes=np.unique(y_integers), y=y_integers)
class_weights = dict(enumerate(class_weights))

# 训练模型
model.fit(datagen.flow(X_train, y_train, batch_size=16), validation_data=(X_val, y_val), epochs=40, class_weight=class_weights)

# 保存模型
model.save('inception_resnet_v2_fused_features.h5')

# 验证模型
loss, accuracy = model.evaluate(X_val, y_val)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy*100:.2f}%")