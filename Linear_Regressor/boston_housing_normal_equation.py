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

# Import required libraries
from sklearn.preprocessing import StandardScaler        # Feature scaling
from sklearn.model_selection import train_test_split    # Split dataset into train and test sets
from sklearn.linear_model import LinearRegression       # Linear regression (Normal Equation)
from sklearn.linear_model import SGDRegressor           # Linear regression (Gradient Descent)
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error  # Evaluation metrics
from sklearn.linear_model import Ridge, RidgeCV         # Regularization models

import pandas as pd
import numpy as np

# 1. Load Boston Housing dataset
data_url = "http://lib.stat.cmu.edu/datasets/boston"

# Read dataset from URL
# sep="\s+" means splitting by one or more whitespace characters
# skiprows=22 skips metadata rows
# header=None indicates no column names
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)

# Combine feature data
# Even rows contain part of the features
# Odd rows contain the remaining features
# np.hstack() horizontally concatenates arrays
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])

# Extract target values (house prices)
target = raw_df.values[1::2, 2]

# 2. Data preprocessing
# Split dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    data, target, test_size=0.2, random_state=22
)

# 3. Feature engineering
# Standardize features (zero mean, unit variance)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 4. Model training
# Initialize linear regression model (Normal Equation)
estimator = LinearRegression(fit_intercept=True)
estimator.fit(x_train, y_train)

# Print model parameters
print(f'Bias (intercept): {estimator.intercept_}')
print(f'Weights (coefficients): {estimator.coef_}')

# 5. Model prediction
y_pred = estimator.predict(x_test)
print(f'Predicted values: {y_pred}')

# 6. Model evaluation
print(f'Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred)}')
print(f'Root Mean Squared Error (RMSE): {root_mean_squared_error(y_test, y_pred)}')
print(f'Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred)}')