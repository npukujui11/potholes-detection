import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.utils import shuffle
from joblib import dump

# 定义特征图路径
feature_paths = {
    "HOG": "D:\\program\\potholes-detection\\dataset\\sz640\\alldata_sz640_filled_hog",
    "HOG_denoise": "D:\\program\\potholes-detection\\dataset\\sz640\\alldata_sz640_filled_denoise_hog",
    "LBP": "D:\\program\\potholes-detection\\dataset\\sz640\\alldata_sz640_filled_lbp",
    "LBP_denoise": "D:\\program\\potholes-detection\\dataset\\sz640\\alldata_sz640_filled_denoise_lbp",
    "Edge": "D:\\program\\potholes-detection\\dataset\\sz640\\alldata_sz640_filled_edge",
    "Edge_denoise": "D:\\program\\potholes-detection\\dataset\\sz640\\alldata_sz640_filled_denoise_edge"
}

# 定义特征图搭配方式
feature_combinations = [
    ["HOG", "LBP", "Edge"],
    ["HOG", "LBP", "Edge_denoise"],
    ["HOG", "LBP_denoise","Edge"],
    ["HOG_denoise", "LBP", "Edge"],
    ["HOG", "LBP_denoise"],
    ["HOG", "LBP"],
    ["HOG", "Edge"],
    ["HOG", "Edge_denoise"],
    ["HOG_denoise", "LBP"],
    ["HOG_denoise", "LBP_denoise"],
    ["HOG_denoise", "Edge"],
    ["HOG_denoise", "Edge_denoise"],
    ["LBP", "Edge"],
    ["LBP", "Edge_denoise"],
    ["LBP_denoise", "Edge"],
    ["LBP_denoise", "Edge_denoise"],
    ["LBP"],
    ["HOG"]
    ["Edge"],
    ["Edge_denoise"],
    ["HOG_denoise"],
    ["LBP_denoise"]
]

# 评估指标
metrics = {
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1-Score': [],
    'Jaccard': []
}

# 数据加载和预处理函数
def load_data(data_dir, feature_dirs):
    images = []
    labels = []
    print(f"Loading data from {data_dir}")
    for filename in os.listdir(data_dir):
        if filename.startswith("pothole"):
            label = 0
        elif filename.startswith("normal"):
            label = 1
        else:
            continue

        features = []
        for feature_dir in feature_dirs:
            feature_path = os.path.join(feature_dir, filename)
            feature_img = load_img(feature_path, color_mode="grayscale", target_size=(640, 640))
            feature_img = img_to_array(feature_img).flatten()
            features.extend(feature_img)

        images.append(features)
        labels.append(label)

    print(f"Loaded {len(images)} images")
    return np.array(images), np.array(labels)

# 模型训练和评估
def train_and_evaluate(feature_dirs, combination_name):
    X, y = load_data(data_dir, feature_dirs)
    X, y = shuffle(X, y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    # 定义分类器
    knn = KNeighborsClassifier()
    svm = SVC(probability=True, class_weight='balanced')
    mlp = MLPClassifier(max_iter=1000)
    ensemble = VotingClassifier(estimators=[('knn', knn), ('svm', svm), ('mlp', mlp)], voting='soft')

    print("Training ensemble model...")
    ensemble.fit(X_train, y_train)
    print("Ensemble model trained.")

    # 保存集成模型
    ensemble_filename = f'ensemble_{combination_name}.joblib'
    dump(ensemble, ensemble_filename)
    print(f"Ensemble model saved as {ensemble_filename}")

    # 评估集成模型
    y_pred = ensemble.predict(X_val)
    metrics['Accuracy'].append(accuracy_score(y_val, y_pred))
    metrics['Precision'].append(precision_score(y_val, y_pred))
    metrics['Recall'].append(recall_score(y_val, y_pred))
    metrics['F1-Score'].append(f1_score(y_val, y_pred))
    metrics['Jaccard'].append(jaccard_score(y_val, y_pred))

    print(f"Ensemble - Accuracy: {metrics['Accuracy'][-1]}")
    # ... 打印其他指标

    # 评估单个模型
    for clf_name, clf in ensemble.named_estimators_.items():
        print(f"Training {clf_name}...")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        print(f"{clf_name} - Accuracy: {accuracy_score(y_val, y_pred)}")
        # ... 打印其他指标

        # 保存单个模型
        model_filename = f'{clf_name}_{combination_name}.joblib'
        dump(clf, model_filename)
        print(f"{clf_name} model saved as {model_filename}")

# 运行评估
for i, combination in enumerate(feature_combinations):
    combination_name = '_'.join(combination)
    print(f"Evaluating feature combination {i+1}/{len(feature_combinations)}: {combination}")
    feature_dirs = [feature_paths[feature] for feature in combination]
    train_and_evaluate(feature_dirs, combination_name)
    print("------------------------------------------------")

# 绘制柱状图
def plot_metrics(metrics_dict):
    n_groups = len(metrics_dict['Accuracy'])
    fig, ax = plt.subplots()

    index = np.arange(n_groups)
    bar_width = 0.1

    opacity = 0.8
    for i, (metric_name, metric_values) in enumerate(metrics_dict.items()):
        plt.bar(index + i * bar_width, metric_values, bar_width,
                alpha=opacity,
                label=metric_name)

    plt.xlabel('Feature Combinations')
    plt.ylabel('Scores')
    plt.title('Scores by feature combination and metric')
    plt.xticks(index + bar_width, (str(i+1) for i in range(n_groups)))
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_metrics(metrics)