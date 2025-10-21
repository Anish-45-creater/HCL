import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("data.csv")
X = data[['X', 'Y']].values

# For binary labels: y > x -> 1, else 0
y = np.where(data['Y'] > data['X'], 1, 0)

# Add bias term
X_bias = np.c_[np.ones(X.shape[0]), X]

# Initialize weights
weights = np.zeros(X_bias.shape[1])
lr = 0.1
epochs = 50

# Step function
def step_function(z):
    return np.where(z >= 0, 1, 0)

# Train perceptron
for epoch in range(epochs):
    for i in range(X_bias.shape[0]):
        z = np.dot(X_bias[i], weights)
        pred = step_function(z)
        weights += lr * (y[i] - pred) * X_bias[i]

print("Learned Weights:", weights)

# Plot classification
plt.figure(figsize=(7,7))
for i in range(len(y)):
    plt.scatter(X[i,0], X[i,1], color='blue' if y[i]==1 else 'red')

# Decision boundary
x_line = np.linspace(min(X[:,0]), max(X[:,0]), 100)
y_line = -(weights[1]*x_line + weights[0])/weights[2]
plt.plot(x_line, y_line, color='green', label='Decision Boundary')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Single-Layer Perceptron Classification")
plt.show()
