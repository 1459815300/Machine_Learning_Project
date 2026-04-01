"""
KNN 算法介绍

一、原理：
    KNN（K-Nearest Neighbors，K近邻）是一种基于实例的非参数学习方法，
    属于惰性学习（Lazy Learning）。其核心思想是：
    给定一个待预测样本，通过度量其与训练集中所有样本的距离，
    选取距离最近的 K 个邻居，根据邻居的信息进行预测。

    1.分类问题（Classification）：
        - 依据 K 个最近邻样本的类别进行“多数投票”（Majority Voting）
        - 预测类别 = 出现频率最高的类别
        - 可引入距离加权（Distance Weighting）提高精度：
              权重 ∝ 1 / 距离

    2.回归问题（Regression）：
        - 预测值为 K 个邻居的目标值的平均（Mean）
        - 或加权平均（Weighted Average）：
              预测值 = Σ(权重 × 邻居值) / Σ权重

    关键要素：
        - 距离度量（Distance Metric）：
              欧氏距离（Euclidean Distance）
              曼哈顿距离（Manhattan Distance）
              闵可夫斯基距离（Minkowski Distance）
        - K 值选择（Bias-Variance Tradeoff）：
              K 小 → 方差大（易过拟合）
              K 大 → 偏差大（易欠拟合）
        - 数据归一化（Feature Scaling）：
              必须进行（否则距离计算失真）

--------------------------------------------------

二、实验思路：

    1.分类问题：
        - 构建或加载数据集（如 Iris、手写数字）
        - 数据预处理（缺失值处理、归一化）
        - 划分训练集与测试集
        - 选择合适的 K 值
        - 使用 KNN 进行分类预测
        - 评估模型性能（Accuracy、Precision、Recall、F1-score）

    2.回归问题：
        - 构建回归数据集（如房价预测）
        - 特征工程（标准化、特征选择）
        - 数据集划分（Train/Test）
        - 训练 KNN 回归模型
        - 预测连续值
        - 评估指标：
              MSE（均方误差）
              RMSE（均方根误差）
              R²（决定系数）

--------------------------------------------------

三、分类问题代码实现思路：

    1.数据准备：
        - 导入数据集（sklearn.datasets 或 CSV）
        - 提取特征 X 和标签 y
        - 标准化处理（StandardScaler / MinMaxScaler）

    2.模型构建：
        - 使用 sklearn.neighbors.KNeighborsClassifier
        - 指定参数：
              n_neighbors=K
              metric='euclidean'
              weights='uniform' 或 'distance'

    3.训练与预测：
        - fit(X_train, y_train)
        - y_pred = predict(X_test)
        - 模型评估：
              accuracy_score
              confusion_matrix
              classification_report

--------------------------------------------------

四、回归问题代码实现思路：

    1.数据准备：
        - 加载回归数据集（如 Boston Housing / 自定义数据）
        - 特征归一化（非常关键）
        - 划分训练集与测试集

    2.模型构建：
        - 使用 sklearn.neighbors.KNeighborsRegressor
        - 设置参数：
              n_neighbors=K
              weights='distance'（常用）

    3.训练与评估：
        - 模型训练：fit()
        - 预测：predict()
        - 评估：
              mean_squared_error
              r2_score

--------------------------------------------------

五、优缺点分析（建议写在报告中）：

    优点：
        - 思想简单，易实现
        - 无需训练过程（惰性学习）
        - 对非线性问题效果好

    缺点：
        - 计算复杂度高（O(n)）
        - 对噪声敏感
        - 维度灾难（Curse of Dimensionality）
        - 对特征尺度敏感

--------------------------------------------------

六、改进方向（进阶）：

    - KD-Tree / Ball Tree 加速搜索
    - 距离加权 KNN
    - 特征降维（PCA）
    - 与深度学习结合（特征提取 + KNN）
"""
#1.导包
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
#2.准备数据集
x_train = [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]
y_train = [0,1,2,3,0,2,2,3,1,3]
x_test = [[10]]
#3.创建模型对象
estimator = KNeighborsClassifier(n_neighbors=3)
#4.训练模型
estimator.fit(x_train, y_train)
#5.模型预测
y_pred = estimator.predict(x_test)
#6.打印预测结果
print(y_pred)
#KNN回归模型，请你思考KNN的实现原理，而不是只知道调用函数
x_train1 = [[1,1,2],[1,2,4],[3,10,2],[2,3,1]]
y_train1 = [0,1,2,3]
x_test1 = [[1,4,2]]
estimator = KNeighborsRegressor(n_neighbors=3)
estimator.fit(x_train1, y_train1)
y_pred1= estimator.predict(x_test1)
print(y_pred1)
