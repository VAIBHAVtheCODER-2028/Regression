# Practice Problem: Predicting Student Exam Scores
# Imagine you are building a model to predict a student's final exam score (out of 100) based on three features:
# Hours studied per week
# Hours of sleep per night
# Attendance percentage (out of 100)

# Here is your training dataset (X_raw) and target labels (y):
# X_raw (Features):
                    # [
                    #     [10.0, 6.0, 80.0],
                    #     [20.0, 7.0, 90.0],
                    #     [5.0,  5.0, 60.0],
                    #     [15.0, 8.0, 85.0],
                    #     [25.0, 6.0, 95.0]
                    # ]

# y (Final Scores): [65.0, 85.0, 45.0, 78.0, 92.0]

# Your Task:
# Write a Python script from scratch using NumPy that:
# 1) Applies Z-score normalization (Feature Scaling) on X_raw using its mean and standard deviation.
# 2) Initializes your weights vector w (zeros based on the number of features) and bias b to 0.0.
# 3) Sets up a learning rate (alpha) of 0.1 and runs vectorized gradient descent for 1000 iterations (using np.dot and X.T without any inner loops).
# 4) Prints out the final trained weights and bias.
# 5) Makes a prediction for a new student who studies 18.0 hours, sleeps 7.5 hours, and has an attendance of 88.0% (remember to scale this new student's data using the training set's mean and std!).

import numpy as np 

# input and output features
X_input=np.array([
    [10.0, 6.0, 80.0],
    [20.0, 7.0, 90.0],
    [5.0,  5.0, 60.0],
    [15.0, 8.0, 85.0],
    [25.0, 6.0, 95.0]
])

y=np.array([65.0, 85.0, 45.0, 78.0, 92.0])
m,n=X_input.shape

# Feature Scaling and Z score Normalization
X_mean=np.mean(X_input,axis=0)
X_std=np.std(X_input,axis=0)
X=(X_input-X_mean)/X_std

# Initializing parameters
w=np.zeros(n)
b=0.0 

# Initializing Hyperparameters
alpha=0.1 
iterations=1000

for i in range(iterations):
    f_wb=np.dot(X,w)+b 

    error=f_wb-y 

    dj_dw=np.dot(X.T,error)/m
    dj_db=np.sum(error)/m 

    w=w-alpha*dj_dw
    b=b-alpha*dj_db

    if i%100==0:
        cost=(1/(2*m))*np.sum((f_wb-y)**2)
        print(f"Iteration {i}: Cost {cost:.4f}")

print("\n--- Training Finished ---")
print(f"Final weights (w): {w}")
print(f"Final bias (b): {b:.4f}")

# New input prediction
test_x=np.array([18.0,7.5,88.0])
scaled_test=(test_x-X_mean)/X_std

predicted_score=np.dot(w,scaled_test)+b
print(f"\nPredicted Score for Student [18.0, 7.5, 88.0]: {predicted_score:.2f} Out of 100")

