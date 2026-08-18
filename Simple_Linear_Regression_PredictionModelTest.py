# Practice Problem: Predicting House Prices
# Imagine you are building a simple linear regression model to predict the price of a house (in lakhs) based on its area (in square feet).
# Here is your small training dataset:
# Input (X): [1.0, 2.0, 3.0, 4.0] (representing house area in hundreds of sq. ft.)
# Output (y): [3.0, 5.0, 7.0, 9.0] (representing price in lakhs)

# Your Task:
# Write a Python script from scratch (without using libraries like scikit-learn for the model training) that:
# 1) Initializes your parameters w and b to 0.0.
# 2) Sets up a learning rate (alpha) of 0.01 and runs gradient descent for 1000 iterations.
# 3) Prints out the final trained parameters (w and b).
# 4) Makes a prediction for a house with an area input of 5.0.

import numpy as np 

#input and output features
X=np.array([1.0, 2.0, 3.0, 4.0])
y=np.array([3.0, 5.0, 7.0, 9.0])
m=len(X)

# initializing parameters
w=0.0 
b=0.0  

# setting up the learning rate and iterations
alpha=0.01 
iterations=1000 

# looping through the gradients
for i in range(iterations):
    dj_dw=0.0
    dj_db=0.0

    for j in range(m):
        f_wb = w * X[j] + b
        error = f_wb - y[j] 

        dj_dw += error * X[j] #
        dj_db += error #

    dj_dw/=m 
    dj_db/=m 

    w=w-alpha*dj_dw 
    b=b-alpha*dj_db 

    if i%100==0:
        cost = (1/(2*m)) * np.sum(((w * X + b)-y)**2)
        print(f"House no {i}: cost:{cost:4f} , w:{w:4f} , b:{b:4f}")

print("\n--- Training Finished ---")
print(f"Final parameters -> w: {w:.4f}, b: {b:.4f}") 

test_x=870.0 
pridicted_y=w*test_x+b 
print(f"Final House Price for {test_x} is {pridicted_y}")
