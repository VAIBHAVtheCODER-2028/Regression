# Model 3: Polynomial Regression & The Normal Equation.

# Two important concepts in this 
# 1) Polynomial Regression : Linear Regression usually deals with simple regression i.e.draws straight line but if our data plots are curved? For that we can create new features by squaring or cubing our original inputs (feature tranformation/augmentation).
# 2) The Noraml Equation : Although it only works for Linear Regression but Polynomial Regression is itself a linear Regression in disguise. So nstead of looping 1,000 times to slowly step down the hill using Gradient Descent, the Normal Equation is a pure mathematical formula that solves for the optimal weights in one single step:$$\theta = (X^T X)^{-1} X^T y.(No iterations, no learning rate alpha, no feature scaling required!)

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 1. Generate Non-Linear Synthetic Data (A curve: y = 3x^2 + 2x + 1)
# 100 random values from -3 to 3
np.random.seed(42)
X = 6*np.random.rand(100,1)-3
y = 3*X**2 + 2*X + 1 + np.random.randn(100,1) 

m=len(X) 

print("--- PART 1: The Normal Equation (From Scartch) ---")
# To fit a parabola (degree 2), we need to engineer a new feature : X^2 
# np.c_ stands for column stack. It takes separate arrays and glues them together side-by-side (as columns) into a single 2D matrix.
X_poly_manual=np.c_[X,X**2] #combines X and X^2 

# To calculate the bias (b) in the Normal Equation, we must add a column of 1s to X
X_b=np.c_[np.ones((m,1)),X_poly_manual] 

# The Normal Equation: theta = (X^T * X)^-1 * X^T * y
# np.linalg.inv computes the inverse of a matrix
theta=np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

print(f"Calculated Parameters (theta):")
print(f"Bias(b): {theta[0][0]:.4f}")
print(f"Weight 1 (for x): {theta[1][0]:.4f}")
print(f"Weight 2 (for x^2): {theta[2][0]:.4f}\n")


print("--- PART 2: Using Scikit-Learn (Industry Standard) ---")
# 1. PolynomialFeatures automatically generates the x^2 (and x^0 column of 1s if needed)
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly_sklearn = poly_features.fit_transform(X)

# 2. LinearRegression automatically applies the closed-form math (Normal Equation/OLS)
lin_reg = LinearRegression()
lin_reg.fit(X_poly_sklearn, y)

print(f"Scikit-Learn Parameters:")
print(f"Bias (b): {lin_reg.intercept_[0]:.4f}")
print(f"Weights (w1, w2): {lin_reg.coef_[0][0]:.4f}, {lin_reg.coef_[0][1]:.4f}")

# Make a prediction for x = 2.0
# True math: 3(2.0)^2 + 2(2.0) + 1 = 17.0
x_new = np.array([[2.0]])
x_new_poly = poly_features.transform(x_new)
prediction = lin_reg.predict(x_new_poly)

print(f"\nPrediction for x=2.0 (True value ~17.0): {prediction[0][0]:.4f}")
