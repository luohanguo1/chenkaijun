import numpy as np
import pandas as pd


def load_data(file_path):
    df = pd.read_csv(file_path)
    X = df.iloc[:, :-1].values
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)
    return X


class KMeansCustom:
    def __init__(self, k=2):
        self.k = k

    def fit(self, X):
        self.centroids = X[np.random.choice(len(X), self.k, replace=False)]
        for _ in range(100):
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(self.k)])
            if np.allclose(self.centroids, new_centroids): break
            self.centroids = new_centroids
        return labels


if __name__ == "__main__":
    print("=" * 30)
    print("1: 对训练集聚类 (QG_train.csv)")
    print("2: 对测试集聚类 (QG_test.csv)")
    choice = input("请输入选项 (1 或 2): ")

    file_name = 'QG_train.csv' if choice == '1' else 'QG_test.csv'

    try:
        X = load_data(file_name)
        labels = KMeansCustom(k=2).fit(X)
        counts = dict(zip(*np.unique(labels, return_counts=True)))
        print(f"\n文件 {file_name} 聚类完成！分布: {counts}")
    except Exception as e:
        print(f"出错: {e}")