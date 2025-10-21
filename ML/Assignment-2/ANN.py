import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("data.csv")
X = data[['X','Y']].values
y = np.where(data['Y'] > data['X'], 1, 0).reshape(-1,1)

# Normalize features
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Activation functions
def sigmoid(x): return 1 / (1 + np.exp(-x))
def tanh(x): return np.tanh(x)
def relu(x): return np.maximum(0, x)

def derivative(a, func):
    if func=='sigmoid': return a*(1-a)
    elif func=='tanh': return 1 - a**2
    else: return np.where(a>0,1,0)

input_dim = 2
hidden_dim = 4
output_dim = 1
lr = 0.1
epochs = 50

for act_func in ['sigmoid','tanh','relu']:
    # Initialize weights
    W1 = np.random.randn(input_dim, hidden_dim)
    b1 = np.zeros((1, hidden_dim))
    W2 = np.random.randn(hidden_dim, output_dim)
    b2 = np.zeros((1, output_dim))
    losses = []

    for epoch in range(epochs):
        # Forward pass
        z1 = np.dot(X,W1)+b1
        a1 = sigmoid(z1) if act_func=='sigmoid' else tanh(z1) if act_func=='tanh' else relu(z1)
        z2 = np.dot(a1,W2)+b2
        a2 = sigmoid(z2)

        # Compute loss
        loss = np.mean((a2 - y)**2)
        losses.append(loss)

        # Backpropagation
        dz2 = a2 - y
        dW2 = np.dot(a1.T,dz2)/X.shape[0]
        db2 = np.sum(dz2,axis=0,keepdims=True)/X.shape[0]
        dz1 = np.dot(dz2,W2.T)*derivative(a1, act_func)
        dW1 = np.dot(X.T,dz1)/X.shape[0]
        db1 = np.sum(dz1,axis=0,keepdims=True)/X.shape[0]

        W1 -= lr*dW1
        b1 -= lr*db1
        W2 -= lr*dW2
        b2 -= lr*db2

    plt.plot(losses,label=act_func)

plt.title("ANN Loss Curves for Different Activation Functions")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
