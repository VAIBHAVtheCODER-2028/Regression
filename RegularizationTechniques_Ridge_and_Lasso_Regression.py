# Reguralization (Ridge and Lasso Regression Techniques) 

# The Regualrization Trap:
# Below is a starter script. It generates a "noisy" sine wave and intentionally overfits it using a Degree-15 polynomial and standard LinearRegression.
# Your Job:
#   1) Import Ridge and Lasso from sklearn.linear_model.
#   2) Initialize both models (set alpha=1.0 for both to start).
#   3) Fit both models using the exact same X_poly and y data.
#   4) Print out the coefficients (coef_) for all three models so you can physically see how Lasso "deletes" features compared to standard regression.

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 1. Generate a noisy curved dataset (a sine wave with random noise)
np.random.seed(42)
X=np.sort(np.random.rand(20,1)*5,axis=0) # 20 random points from 0 to 5
y=np.sin(X).ravel() + np.random.randn(20)*0.2 #Sine wave + noise 

# 2. Engineer massive polynomial features (Degree 15)
ploy_features=PolynomialFeatures(degree=15 , include_bias=False) 
X_poly=ploy_features.fit_transform(X)

# 3. Fit standard Linear Regression (This will overfit wildly)
lin_reg=LinearRegression()
lin_reg.fit(X_poly,y) 

print("--- Standard Linear Regression ---")
# look at the first 5 weights. They will be massive! 
print(f"First 5 weights: {np.round(lin_reg.coef_[:5], 2)}")

# ==========================================
# YOUR TURN: ADD RIDGE AND LASSO BELOW
# ==========================================

# TODO: Initialize and fit Ridge(alpha=1.0)
# TODO: Print Ridge's first 5 weights

# TODO: Initialize and fit Lasso(alpha=1.0)
# TODO: Print Lasso's first 5 weights

from sklearn.linear_model import LinearRegression , Ridge , Lasso 

print("\n--- Ridge Regression (L2) ---")
# 1.Initializing Ridge with an alpha penalty of 1.0 
ridge_reg=Ridge(alpha=1.0) 
# 2.Fit it using the exact same massive feature set 
ridge_reg.fit(X_poly,y) 
# 3.Print the first 5 weights
print(f"First 5 weights: {np.round(ridge_reg.coef_[:5],2)}")

print("\n--- Lasso Regression (L1) ---")
# 1.Initializing Lasso with an alpha penalty of 1.0
lasso_reg=Lasso(alpha=1.0) 
# 2.Fit it using the exact massive feature set 
lasso_reg.fit(X_poly,y)
# 3.Print the first 5 weights 
print(f"First 5 weights: {np.round(lasso_reg.coef_[:5],2)}")
