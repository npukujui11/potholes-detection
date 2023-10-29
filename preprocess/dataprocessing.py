import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input

def preprocess_data(data_dir='D:\\program\\potholes-detection\\dataset', img_width=299, img_height=299, batch_size=32):
    # 数据增强和图像大小调整
    datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.1)  # 为验证集分配10%的数据

    # 数据分割
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True)

    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=True)

    # 处理数据不平衡
    class_weights = {
        0: 1.0,  # normal
        1: (266/35)  # potholes
    }

    return train_generator, validation_generator, class_weights

if __name__ == '__main__':
    train_gen, val_gen, weights = preprocess_data()
    print(f"Total training samples: {train_gen.samples}")
    print(f"Total validation samples: {val_gen.samples}")
    print(f"Class weights: {weights}")
