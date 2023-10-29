import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from preprocess import dataprocessing

# 设置GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 数据路径
train_data_dir = 'D:\\program\\potholes-detection\\dataset'

# 参数
img_width, img_height = 299, 299
batch_size = 1024
epochs = 2000
num_classes = 2

# 获取预处理后的数据
train_generator, validation_generator, class_weights = dataprocessing.preprocess_data(data_dir=train_data_dir)

# 构建模型
base_model = InceptionResNetV2(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers:
    layer.trainable = False
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(
    train_generator,
    steps_per_epoch=np.ceil(train_generator.samples / batch_size),
    epochs=epochs)

# 保存模型
model.save('pothole_classifier_0.1.h5')

# 验证模型
loss, accuracy = model.evaluate(validation_generator, steps=np.ceil(validation_generator.samples / batch_size))
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")