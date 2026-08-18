# Model 1:Simple Linear Regression model using concepts like univariate function , MSE , Cost Function and Gradient Descent

import numpy as np 

#generating some synthetic training data (x and y)
X=np.array([1.0 , 2.0 , 3.0 , 4.0 , 5.0]) 
y=np.array([2.0 , 4.0 , 6.0 , 8.0 , 10.0]) 
m=len(X) 

#initializing the parameters w and b as 0 
w=0.0 
b=0.0 

#Hyperparameters/learning Rate
alpha = 0.01 
iterations = 1000 

#Gradient Descent
for i in range(iterations):
    # initializing gradients
    dj_dw=0.0 
    dj_db=0.0 

    # Compute predictions and gradients over all training examples (Batch Gradient Descent)
    for j in range(m):
        f_wb=w*X[j] + b    # Hypothesis: w*x + b
        error=f_wb-y[j]    # (prediction - actual)

        dj_dw += error * X[j]  # Accumulate partial derivative wrt w
        dj_db += error         # Accumulate partial derivative wrt b

    # Average out the gradients by dividing by m
    dj_dw /= m 
    dj_db /= m 

    # Simultaneously update parameters w and b
    w = w-alpha*dj_dw 
    b = b-alpha*dj_db 

    # Print cost every 100 iterations to watch it decrease
    if i%100==0:
        cost = (1/(2*m)) * np.sum(((w * X + b) - y)**2) 
        print(f"iteration {i}: Cost: {cost:4f} , w:{w:4f} , b:{b:4f}")

print("\n--- Training Finished ---")
print(f"Final parameters -> w: {w:.4f}, b: {b:.4f}") 

# 5. Make a prediction
test_x = 6.0
predicted_y=w*test_x+b 
print(f"Prediction for x = {test_x}: {predicted_y:.4f}")
