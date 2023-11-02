import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
from sklearn.ensemble import VotingClassifier
import matplotlib.pyplot as plt
from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise
from skimage.filters import gaussian
import random

# 函数：应用数据增强
def augment_image(image):
    # 随机旋转
    if random.random() > 0.5:
        image = rotate(image, angle=random.uniform(-10, 10), mode='edge')
    # 随机应用噪声
    if random.random() > 0.5:
        image = random_noise(image)
    # 随机模糊
    if random.random() > 0.5:
        image = gaussian(image, sigma=random.uniform(0.1, 1.0))
    # 随机仿射变换
    if random.random() > 0.5:
        transform = AffineTransform(translation=(random.uniform(-5, 5), random.uniform(-5, 5)))
        image = warp(image, transform, mode='edge')
    return image
# 函数：加载数据和标签
def load_data_and_labels(feature_path, class_label, augment=False, num_augmented=0):
    images = []
    labels = []
    for filename in os.listdir(feature_path):
        if filename.startswith(class_label):
            img_path = os.path.join(feature_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            images.append(img.flatten())
            labels.append(1 if class_label == 'pothole' else 0)
            if augment:
                for _ in range(num_augmented):
                    img_augmented = augment_image(img)
                    images.append(img_augmented.flatten())
                    labels.append(1 if class_label == 'pothole' else 0)
    return images, labels

# 函数：训练和评估模型
def train_and_evaluate(X_train, y_train, X_test, y_test, classifier, classifier_name):
    print(f"Training {classifier_name}...")
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    jaccard = jaccard_score(y_test, y_pred)
    print(f"{classifier_name} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}, Jaccard's Index: {jaccard}")
    return accuracy, precision, recall, f1, jaccard

# 函数：可视化结果
def plot_results(results):
    labels = list(results.keys())
    accuracy = [results[label]['Accuracy'] for label in labels]
    precision = [results[label]['Precision'] for label in labels]
    recall = [results[label]['Recall'] for label in labels]
    f1 = [results[label]['F1-Score'] for label in labels]
    jaccard = [results[label]['Jaccard'] for label in labels]

    x = np.arange(len(labels))
    width = 0.15

    fig, ax = plt.subplots(figsize=(15,6))
    rects1 = ax.bar(x - width*2, accuracy, width, label='Accuracy')
    rects2 = ax.bar(x - width, precision, width, label='Precision')
    rects3 = ax.bar(x, recall, width, label='Recall')
    rects4 = ax.bar(x + width, f1, width, label='F1-Score')
    rects5 = ax.bar(x + width*2, jaccard, width, label='Jaccard')

    ax.set_ylabel('Scores')
    ax.set_title('Scores by classifier and metric')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.show()

# 主程序
feature_types = ['hog', 'lbp', 'edge']
classifiers = {
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(probability=True, random_state=42),
    'RF': RandomForestClassifier(),
    'MLP': MLPClassifier(max_iter=1000)
}
results = {}

for feature_type in feature_types:
    feature_path = f"D:\\program\\potholes-detection\\dataset\\sz640\\alldata_sz640_filled_{feature_type}"
    X_pothole, y_pothole = load_data_and_labels(feature_path, 'pothole', augment=True, num_augmented=20)
    X_normal, y_normal = load_data_and_labels(feature_path, 'normal', augment=True, num_augmented=20)
    X = X_pothole + X_normal
    y = y_pothole + y_normal

    X, y = shuffle(X, y, random_state=42)
    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    best_score = 0
    best_classifier_name = ''
    for classifier_name, classifier in classifiers.items():
        accuracy, precision, recall, f1, jaccard = train_and_evaluate(X_train, y_train, X_test, y_test, classifier, classifier_name)
        if accuracy > best_score:
            best_score = accuracy
            best_classifier_name = classifier_name
        results[f"{feature_type}_{classifier_name}"] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Jaccard': jaccard
        }
    print(f"Best classifier for {feature_type} is {best_classifier_name}")

# 集成学习
ensemble = VotingClassifier(estimators=[(name, classifiers[name]) for name in classifiers], voting='hard')
ensemble.fit(X_train, y_train)
y_pred_ensemble = ensemble.predict(X_test)
ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
ensemble_precision = precision_score(y_test, y_pred_ensemble)
ensemble_recall = recall_score(y_test, y_pred_ensemble)
ensemble_f1 = f1_score(y_test, y_pred_ensemble)
ensemble_jaccard = jaccard_score(y_test, y_pred_ensemble)
results['Ensemble'] = {
    'Accuracy': ensemble_accuracy,
    'Precision': ensemble_precision,
    'Recall': ensemble_recall,
    'F1-Score': ensemble_f1,
    'Jaccard': ensemble_jaccard
}

# 可视化结果
plot_results(results)