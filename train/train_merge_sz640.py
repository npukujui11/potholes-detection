import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Multiply, Flatten, Softmax, Dropout, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.utils import shuffle
from tensorflow.keras.layers import Layer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

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

data_dir = "D:\\program\\potholes-detection\\dataset\\sz640\\alldata_sz640"
feature_dirs = ["D:\\program\\potholes-detection\\dataset\\sz640\\alldata_sz640_filled_denoise_edge",
                "D:\\program\\potholes-detection\\dataset\\sz640\\alldata_sz640_filled_denoise_hog",
                "D:\\program\\potholes-detection\\dataset\\sz640\\alldata_sz640_filled_denoise_lbp"]

X, y = load_data(data_dir, feature_dirs)
X, y = shuffle(X, y, random_state=42)  # 打乱数据

# 划分训练和验证数据
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# 注意力模块
class SEBlock(Layer):
    def __init__(self, ratio=16, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        self.channel_axis = -1
        self.channels = input_shape[self.channel_axis]
        self.se_shape = (1, 1, self.channels)

        self.global_average = GlobalAveragePooling2D()
        self.reshape = Reshape(self.se_shape)
        self.dense_reduce = Dense(self.channels // self.ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)
        self.dense_expand = Dense(self.channels, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)

    def call(self, inputs, **kwargs):
        x = self.global_average(inputs)
        x = self.reshape(x)
        x = self.dense_reduce(x)
        x = self.dense_expand(x)
        return Multiply()([inputs, x])

    def get_config(self):
        config = super(SEBlock, self).get_config()
        config.update({'ratio': self.ratio})
        return config

# 模型定义
def create_attention_model(input_shape, dropout_rate, l1_reg, l2_reg):
    input_tensor = Input(shape=input_shape)
    attention_output = SEBlock()(input_tensor)

    base_model = InceptionResNetV2(include_top=False, weights=None, input_tensor=attention_output)
    base_features = base_model.output

    gap = GlobalAveragePooling2D()(base_features)
    fc1 = Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(gap)
    dropout = Dropout(dropout_rate)(fc1)
    output = Dense(2, activation='softmax')(dropout)

    model = Model(inputs=input_tensor, outputs=output)
    return model

# 设置不同的正则化参数
regularization_params = [
    {'dropout_rate': 0.3, 'l1_reg': 0.0, 'l2_reg': 0.01},
    {'dropout_rate': 0.5, 'l1_reg': 0.0, 'l2_reg': 0.01},
    {'dropout_rate': 0.3, 'l1_reg': 0.01, 'l2_reg': 0.0},
    {'dropout_rate': 0.5, 'l1_reg': 0.01, 'l2_reg': 0.0},
    {'dropout_rate': 0.5, 'l1_reg': 0.01, 'l2_reg': 0.01},
    {'dropout_rate': 0.5, 'l1_reg': 1e-3, 'l2_reg': 1e-3},
    {'dropout_rate': 0.3, 'l1_reg': 0.0, 'l2_reg': 1e-3},
    {'dropout_rate': 0.3, 'l1_reg': 1e-3, 'l2_reg': 0},
]

# 训练模型并记录历史
history_data = {}
# 使用数据增强和类权重处理数据不平衡问题
datagen = ImageDataGenerator(rotation_range=20,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             horizontal_flip=True)
datagen.fit(X_train)

# 计算类权重
y_integers = np.argmax(y_train, axis=1)
class_weights = compute_class_weight('balanced', classes=np.unique(y_integers), y=y_integers)
class_weights = dict(enumerate(class_weights))

input_shape = (640, 640, 4)

for params in regularization_params:
    print(f"Training with params: {params}")
    model = create_attention_model(input_shape, **params)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=8),
        validation_data=(X_val, y_val),
        epochs=40,
        class_weight=class_weights
    )
    history_key = f"dropout_{params['dropout_rate']}_l1_{params['l1_reg']}_l2_{params['l2_reg']}"
    history_data[history_key] = history.history
    print(f"Finished training {history_key}")

# 绘制和保存准确率曲线
for name, history in history_data.items():
    print(f"Plotting accuracy curve for {name}")
    plt.figure(figsize=(10, 4))
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title(f"Accuracy Over Epochs ({name})")
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig(f"accuracy_curve_{name}.png")
    plt.close()  # 关闭图形，避免重叠
    print(f"Saved accuracy curve as accuracy_curve_{name}.png")