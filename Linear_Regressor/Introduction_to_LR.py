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