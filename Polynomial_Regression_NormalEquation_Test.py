# Practice Problem: Fitting a Parabola
# Imagine you are given a dataset where the underlying relationship is quadratic (a parabola), and you want to use Scikit-Learn's automated tools to find the parameters.

# Your Task:
# Write a Python script using Scikit-Learn that:
# 1) Initializes PolynomialFeatures with degree=2 and include_bias=False, then transforms your input X.
# 2) Creates a LinearRegression model object and fits it using the transformed polynomial features and y.
# 3) Prints out the model's learned intercept (bias b) and coefficients (w_1, w_2).
# 4) Makes a prediction for a new input value of x = 4.0 and prints the result.

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

X = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0], [3.0]])
y = np.array([18.0, 9.0, 4.0, 3.0, 6.0, 13.0])

m=len(X)

print("--- Fitting Curve using Scikit-learn ---")
poly_features=PolynomialFeatures(degree=2, include_bias=False)
X_poly_sklearn=poly_features.fit_transform(X) 

lin_reg=LinearRegression()
lin_reg.fit(X_poly_sklearn, y) 

print(f"Scikit-Learn Parameters:")
print(f"Bias (b): {lin_reg.intercept_:.4f}")
print(f"Weights (w1, w2): {lin_reg.coef_[0]:.4f}, {lin_reg.coef_[1]:.4f}")

# Making the prediction
x_new=np.array([[4.0]])
x_new_poly=poly_features.transform(x_new)
prediction=lin_reg.predict(x_new_poly)

print(f"\nPrediction for x=4.0 : {prediction[0]:.4f}")
