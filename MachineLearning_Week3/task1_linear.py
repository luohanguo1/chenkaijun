import numpy as np
import pandas as pd


def diagnose_data(file_path):
    """加载数据并输出与截图完全一致的诊断报告"""
    df = pd.read_csv(file_path)

    # 1. 打印数据形状和前5行 (复刻截图内容)
    print("\n--- 数据诊断报告 ---")
    print(f"1. 数据形状: {df.shape}")
    print(f"2. 前5行数据:")
    print(df.head())

    # 2. 提取标签并自动映射 (处理 QG 数据集的 -1/1 问题)
    y_raw = df.iloc[:, -1].values
    unique_y = np.unique(y_raw)
    y = (y_raw == unique_y[1]).astype(int)
    print(f"\n3. 标签(y)中的不同数值: {unique_y}")
    print(f"4. 转换后的标签预览(前10个): {y[:10]}")

    # 3. 特征提取与归一化 (防止梯度爆炸)
    X = df.iloc[:, :-1].values
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X = (X - X_min) / (X_max - X_min + 1e-8)

    # 4. 添加偏置项
    X = np.c_[np.ones(X.shape[0]), X]
    return X, y


class LogisticRegressionCustom:
    def sigmoid(self, z):
        # 使用 clip 防止 exp 溢出错误
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y, lr=0.01, iters=1000):
        m, n = X.shape
        self.theta = np.zeros(n)

        print("\n开始迭代训练...")
        for i in range(iters):
            h = self.sigmoid(np.dot(X, self.theta))
            gradient = np.dot(X.T, (h - y)) / m
            self.theta -= lr * gradient

            # 每 200 次打印一次梯度均值 (复刻截图)
            if i % 200 == 0:
                grad_mean = np.abs(gradient).mean()
                print(f"迭代 {i}, 梯度绝对值均值: {grad_mean:.6f}")

    def predict(self, X):
        return (self.sigmoid(np.dot(X, self.theta)) >= 0.5).astype(int)


if __name__ == "__main__":
    print("=" * 40)
    print(" 机器学习任务：逻辑回归模型")
    print(" 1: 读取训练集 (QG_train.csv)")
    print(" 2: 读取测试集 (QG_test.csv)")
    print("=" * 40)

    choice = input("请输入选项数字 (1 或 2): ")
    file_name = 'QG_train.csv' if choice == '1' else 'QG_test.csv'

    try:
        # 加载与诊断
        X, y = diagnose_data(file_name)

        # 训练模型
        model = LogisticRegressionCustom()
        model.fit(X, y, lr=0.01, iters=1000)

        # 展示最终结果 (复刻截图)
        preds = model.predict(X)
        acc = np.mean(preds == y)

        print(f"\n--- 最终结果 ---")
        print(f"预测值预览: {preds[:10]}")
        print(f"实际值预览: {y[:10]}")
        print(f"当前准确率: {acc * 100:.2f}%")

    except FileNotFoundError:
        print(f"\n[错误] 找不到文件: {file_name}，请确认它在当前文件夹内。")
    except Exception as e:
        print(f"\n运行出错: {e}")

    input("\n任务已完成，按回车键结束程序...")