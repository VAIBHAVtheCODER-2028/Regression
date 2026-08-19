import numpy as np

# 1. Generating synthetic multi-feature training data (e.g., House Size in sq ft, Bedrooms, Age)
# m training examples, n features
X_raw=np.array([
    [1200 , 2 ,5],
    [1500 , 3 , 10],    
    [2000 , 4 , 2],
    [2400 , 3 , 1],
    [3000 , 5 , 8]
])

y=np.array([40.0, 55.0, 75.0, 90.0, 110.0])
m,n=X_raw.shape  #.shape returns a tuple representing the size of the array along each of its dimensions, while len(Arr) only returns the size of the very first dimension (the outer layer).

# 2. Feature Scaling (Z-score normalization from your notes)
# Formula: x' = (x - mean) / sigma
X_mean=np.mean(X_raw , axis=0)
X_std=np.std(X_raw , axis=0)
X=(X_raw - X_mean) / X_std 

# 3. Initializing parameters: vector w (size n) and scalar b
w=np.zeros(n) 
b=0.0 

# 4. Hyperparameters
alpha=0.1 
iterations=1000

# 5. Vectorized Gradient Descent Loop 
for i in range(iterations):
    # Vectorized Hypothesis: f_wb = X dot w + b (computes all m predictions at once)
    f_wb=np.dot(X,w) + b

    # Error vector of size (m,)
    error=f_wb-y 

    # Vectorized Gradients (dot product handles the summation over all m examples)
    dj_dw=np.dot(X.T,error)/m # X.T is the transpose of X, which flips its dimensions to (n X m). Rows become columns, and columns become rows.
    dj_db = np.sum(error)/m

    # We flip the dimensions (transpose X) because of the strict rules of matrix multiplication.To multiply two matrices or a matrix and a vector, their inner dimensions must match, and the transpose aligns those dimensions so the math actually works out correctly.

    # Simultaneous Update
    w = w - alpha*dj_dw
    b = b - alpha*dj_db

    # Print cost every 100 iterations
    if i % 100 == 0:
        cost=(1/(2*m))*np.sum((f_wb-y)**2)
        print(f"Iteration {i}: Cost {cost:.4f}")

print("\n--- Training Finished ---")
print(f"Final weights (w): {w}")
print(f"Final bias (b): {b:.4f}")

# 6. Making a prediction on a new house [Size, Bedrooms, Age]
# IMPORTANT: New data must be scaled using the TRAINING set's mean and std!
new_house=np.array([870, 2 , 13])
scaled_house=(new_house-X_mean)/X_std

predicted_price = np.dot(scaled_house, w) + b 
print(f"\nPredicted price for house [870 sqft, 2 beds, 13 yrs old]: {predicted_price:.2f} lakhs")
