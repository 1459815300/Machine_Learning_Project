"""
Linear Regression (线性回归)

Overview / Purpose:
    Linear regression is a fundamental supervised learning algorithm used to model
    the relationship between a dependent variable (target) and one or more independent
    variables (features). Its goal is to find a linear function that best fits the data,
    enabling prediction and inference.

Types:
    1. Simple Linear Regression:
        - One independent variable (single feature).
    2. Multiple Linear Regression:
        - Multiple independent variables (multiple features).

Formula:
    1. Simple Linear Regression:
        y = w * x + b

    2. Multiple Linear Regression:
        y = w1*x1 + w2*x2 + ... + wn*xn + b
        or in vector form:
        y = w^T x + b

    where:
        y : predicted value (target)
        x : input feature(s)
        w : weight(s) (model parameters)
        b : bias (intercept)

    Loss Function (Mean Squared Error, MSE):
        MSE = (1/n) * Σ (y_i - ŷ_i)^2

    Optimization:
        - Gradient Descent
        - Normal Equation (closed-form solution)

Key Characteristics:
    - Assumes linear relationship between variables
    - Sensitive to outliers
    - Easy to interpret and computationally efficient

Applications:
    - Price prediction (e.g., housing)
    - Trend analysis
    - Risk modeling
    - Basic baseline model in machine learning
"""
from sklearn.linear_model import LinearRegression

# Training feature data (e.g., height in cm)
x_train=[[160],[166],[172],[174],[180]]
# Training target values (e.g., weight in kg)
y_train=[56.3,60.6,65.1,68.5,75]
# Test feature data (new sample for prediction)
x_test=[[176]]
# Create a Linear Regression model instance
estimator=LinearRegression()
# Fit the model using training data
estimator.fit(x_train,y_train)
# Output the learned coefficient (slope)
print(estimator.coef_)
# Output the learned intercept (bias)
print(estimator.intercept_)
# Use the trained model to make predictions
y_pred=estimator.predict(x_test)
# Output the predicted result
print(y_pred)