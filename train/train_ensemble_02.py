import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
import matplotlib.pyplot as plt

# 数据路径
HOG_DIR = 'D:\\program\\potholes-detection\\dataset\\sz640\\alldata_sz640_filled_hog'
LBP_DIR = 'D:\\program\\potholes-detection\\dataset\\sz640\\alldata_sz640_filled_lbp'
EDGE_DIR = 'D:\\program\\potholes-detection\\dataset\\sz640\\alldata_sz640_filled_edge'

# 加载数据
def load_data(feature_dir):
    X = []
    y = []
    for filename in os.listdir(feature_dir):
        filepath = os.path.join(feature_dir, filename)
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            X.append(image.flatten())
            y.append(1 if filename.startswith('pothole') else 0)
        else:
            print(f"Failed to read image {filename}")
    return np.array(X), np.array(y)

# 评估模型
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    jaccard = jaccard_score(y_test, y_pred)
    return accuracy, precision, recall, f1, jaccard

# 训练和评估分类器
def train_and_evaluate(X, y, classifier, classifier_name, feature_name):
    print(f"Training {classifier_name} on {feature_name}...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y)
    classifier.fit(X_train, y_train)
    print(f"Evaluating {classifier_name} on {feature_name}...")
    return evaluate_model(classifier, X_test, y_test)

# 主程序
def main():
    # 加载特征数据
    print("Loading data...")
    X_hog, y_hog = load_data(HOG_DIR)
    X_lbp, y_lbp = load_data(LBP_DIR)
    X_edge, y_edge = load_data(EDGE_DIR)

    # 初始化分类器
    knn = KNeighborsClassifier()
    svm = SVC(probability=True)
    mlp = MLPClassifier(max_iter=1000)

    # 训练和评估
    results = {}
    for feature_name, X, y in [('HOG', X_hog, y_hog), ('LBP', X_lbp, y_lbp), ('Edge', X_edge, y_edge)]:
        for classifier, classifier_name in [(knn, 'KNN'), (svm, 'SVM'), (mlp, 'MLP')]:
            results[(classifier_name, feature_name)] = train_and_evaluate(X, y, classifier, classifier_name, feature_name)

    # 找出每种特征的最佳分类器
    best_classifiers = {}
    for feature_name in ['HOG', 'LBP', 'Edge']:
        best_score = 0
        best_classifier_name = ''
        for classifier_name in ['KNN', 'SVM', 'MLP']:
            score = results[(classifier_name, feature_name)][0]  # 使用准确率作为评估标准
            if score > best_score:
                best_score = score
                best_classifier_name = classifier_name
        best_classifiers[feature_name] = best_classifier_name

    # 输出最佳分类器
    print("Best classifiers for each feature:")
    for feature_name, classifier_name in best_classifiers.items():
        print(f"{feature_name}: {classifier_name}")

    # 集成学习
    print("Performing ensemble learning...")
    ensemble = VotingClassifier(estimators=[
        ('knn_hog', knn if best_classifiers['HOG'] == 'KNN' else KNeighborsClassifier()),
        ('svm_lbp', svm if best_classifiers['LBP'] == 'SVM' else SVC(probability=True)),
        ('mlp_edge', mlp if best_classifiers['Edge'] == 'MLP' else MLPClassifier(max_iter=1000))
    ], voting='soft')
    X_ensemble = np.hstack((X_hog, X_lbp, X_edge))
    y_ensemble = y_hog  # Assumes all y are the same
    X_train, X_test, y_train, y_test = train_test_split(X_ensemble, y_ensemble, test_size=0.1, stratify=y_ensemble)
    ensemble.fit(X_train, y_train)
    ensemble_results = evaluate_model(ensemble, X_test, y_test)

    # 输出集成学习结果
    print("Ensemble learning results:")
    print(f"Accuracy: {ensemble_results[0]}")
    print(f"Precision: {ensemble_results[1]}")
    print(f"Recall: {ensemble_results[2]}")
    print(f"F1-Score: {ensemble_results[3]}")
    print(f"Jaccard's Index: {ensemble_results[4]}")

    # 绘制柱状图
    labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Jaccard']
    knn_scores = results[(best_classifiers['HOG'], 'HOG')]
    svm_scores = results[(best_classifiers['LBP'], 'LBP')]
    mlp_scores = results[(best_classifiers['Edge'], 'Edge')]
    ensemble_scores = ensemble_results

    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, knn_scores, width, label='KNN')
    rects2 = ax.bar(x, svm_scores, width, label='SVM')
    rects3 = ax.bar(x + width, mlp_scores, width, label='MLP')
    rects4 = ax.bar(x + 2*width, ensemble_scores, width, label='Ensemble')

    ax.set_ylabel('Scores')
    ax.set_title('Scores by classifier and metric')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()