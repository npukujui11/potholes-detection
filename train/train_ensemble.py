import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from joblib import dump

# 数据加载和预处理
def load_data(data_dir, feature_dirs):
    images = []
    labels = []
    for filename in os.listdir(data_dir):
        if filename.startswith("pothole"):
            label = 0
        elif filename.startswith("normal"):
            label = 1
        else:
            continue

        # 加载特征图
        features = []
        for feature_dir in feature_dirs:
            feature_path = os.path.join(feature_dir, filename)
            feature_img = load_img(feature_path, color_mode="grayscale", target_size=(640, 640))
            feature_img = img_to_array(feature_img).flatten()
            features.extend(feature_img)

        images.append(features)
        labels.append(label)

    return np.array(images), np.array(labels)


data_dir = "D:\\program\\potholes-detection\\dataset\\alldata"
feature_dirs = ["D:\\program\\potholes-detection\\dataset\\alldata_filled_edge",
                "D:\\program\\potholes-detection\\dataset\\alldata_filled_hog",
                "D:\\program\\potholes-detection\\dataset\\alldata_filled_lbp"]
X, y = load_data(data_dir, feature_dirs)
X, y = shuffle(X, y, random_state=42)

# 划分训练和验证数据
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# 定义分类器
knn = KNeighborsClassifier()
svm = SVC(probability=True, class_weight='balanced')
mlp = MLPClassifier(max_iter=1000)

# 使用集成学习的方法
ensemble = VotingClassifier(estimators=[('knn', knn), ('svm', svm), ('mlp', mlp)], voting='soft')
ensemble.fit(X_train, y_train)

# 输出每个分类器的训练准确度
for clf_name, clf in ensemble.named_estimators_.items():
    train_predictions = clf.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_predictions)
    print(f"{clf_name} Training Accuracy: {train_accuracy:.2f}")

# 保存集成模型
dump(ensemble, 'pothole_classifier_ensemble.joblib')

# 从文件加载模型
# loaded_ensemble = load(model_filename)

# 验证模型
y_pred = ensemble.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy:.2f}")
