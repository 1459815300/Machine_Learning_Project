"""
Case:
Demonstrate the Normal Equation method for linear regression to complete the Boston housing price prediction task.

Concept:
Linear regression is a type of supervised learning.
Features (X) and labels (y) are given, and the output is continuous.

Types of Linear Regression:
- Univariate linear regression: 1 feature, 1 label.
- Multivariate linear regression: multiple features, 1 label.

Mathematical Formulation:
- Univariate: y = wx + b
- Multivariate: y = w1*x1 + w2*x2 + ... + wn*xn + b = w^T x + b

Goal:
Minimize the error between predicted values and true values.

Loss Functions:
1. Least Squares
2. Mean Squared Error (MSE)
3. Root Mean Squared Error (RMSE)
4. Mean Absolute Error (MAE)

Optimization Methods:
1. Gradient Descent:
   - Full Gradient Descent (FGD)
   - Stochastic Gradient Descent (SGD)
   - Mini-batch Gradient Descent
   - Stochastic Average Gradient (SAG)

2. Normal Equation Method

Machine Learning Workflow:
1. Load data
2. Data preprocessing
3. Feature engineering (feature extraction, feature processing, etc.)
4. Model training
5. Model prediction
6. Model evaluation
"""

#import required libraries
from sklearn.preprocessing import StandardScaler        # 特征处理
from sklearn.model_selection import train_test_split    # 数据集划分
from sklearn.linear_model import LinearRegression       # 正规方程的回归模型
from sklearn.linear_model import SGDRegressor           # 梯度下降的回归模型
from sklearn.metrics import mean_squared_error,root_mean_squared_error,mean_absolute_error        # 均方误差评估
from sklearn.linear_model import Ridge, RidgeCV
import pandas as pd
import numpy as np
#1.load boston housing database
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]]) #hstack()函数作用：水平拼接数组
target = raw_df.values[1::2, 2]

# print(target.shape)
# print(target[:5])
# print(data.shape)
# print(data[:5])

# 2. Data preprocessing
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=22)
# 3. Feature engineering (feature extraction, feature processing, etc.)
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)
# 4. Model training
estimator = LinearRegression(fit_intercept=True)
estimator.fit(x_train, y_train)
print(f'bias:{estimator.intercept_}')
print(f'权重:{estimator.coef_}')
# 5. Model prediction
y_pred = estimator.predict(x_test)
print(f'预测结果{y_pred}')
# 6. Model evaluation
print(f'均方误差:{mean_squared_error(y_test, y_pred)}')
print(f'均方根误差:{root_mean_squared_error(y_test, y_pred)}')
print(f'平均绝对误差:{mean_absolute_error(y_test, y_pred)}')