import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, save_img
from imblearn.over_sampling import RandomOverSampler

# 数据增强配置
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)


def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = load_img(img_path, color_mode="grayscale", target_size=(640, 640))
        img_array = img_to_array(img)
        images.append(img_array)
        if filename.startswith("pothole"):
            labels.append(0)  # pothole类别标签为0
        else:
            labels.append(1)  # normal类别标签为1
    return np.array(images), np.array(labels)

def save_augmented_images(images, labels, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i, (img, label) in enumerate(zip(images, labels)):
        prefix = 'pothole_' if label == 0 else 'normal_'
        filename = prefix + str(i) + '.jpg'
        img_path = os.path.join(save_dir, filename)

        # 确保图像是uint8类型
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)

        # 确保图像有3个维度
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)

        save_img(img_path, img)

# 加载原始图像
feature_dirs = {
    'HOG': 'D:\\program\\potholes-detection\\dataset\\sz640\\alldata_sz640_filled_hog',
    'LBP': 'D:\\program\\potholes-detection\\dataset\\sz640\\alldata_sz640_filled_lbp',
    'Edge': 'D:\\program\\potholes-detection\\dataset\\sz640\\alldata_sz640_filled_edge'
}

# 进行过采样和数据增强
for feature_name, feature_dir in feature_dirs.items():
    images, labels = load_images_from_folder(feature_dir)

    # 过采样以平衡类别
    ros = RandomOverSampler(random_state=42)
    images_res, labels_res = ros.fit_resample(images.reshape(len(images), -1), labels)
    images_res = images_res.reshape(-1, 640, 640, 1)

    # 数据增强
    augmented_images = []
    augmented_labels = []
    for x, y in zip(images_res, labels_res):
        x = np.expand_dims(x, 0)
        for _ in range(5):  # 每张图像生成5个增强图像
            iter = datagen.flow(x, batch_size=1)
            aug_img = next(iter)[0].astype('uint8')
            augmented_images.append(aug_img)
            augmented_labels.append(y)

    augmented_images = np.array(augmented_images)
    augmented_labels = np.array(augmented_labels)

    # 保存增强后的图像
    save_dir = feature_dir + '_gen'
    save_augmented_images(augmented_images, augmented_labels, save_dir)