import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Synthetic data: size (sq ft) and price (in thousands)
np.random.seed(42)
sizes = np.random.rand(100, 1) * 2000 + 500  # 500-2500 sq ft
prices = 100 + 0.1 * sizes + np.random.rand(100, 1) * 50  # Linear relation with noise

# Simple Linear Regression
X_train, X_test, y_train, y_test = train_test_split(sizes, prices, test_size=0.2, random_state=42)
model_simple = LinearRegression()
model_simple.fit(X_train, y_train)
y_pred_simple = model_simple.predict(X_test)

# Multiple Linear Regression (add a second feature: rooms)
rooms = np.random.randint(2, 6, (100, 1))
X_multi = np.hstack((sizes, rooms))
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(X_multi, prices, test_size=0.2, random_state=42)
model_multi = LinearRegression()
model_multi.fit(X_train_multi, y_train_multi)
y_pred_multi = model_multi.predict(X_test_multi)

# Plot results
plt.scatter(X_test, y_test, color='blue', label='Actual Prices')
plt.plot(X_test, y_pred_simple, color='red', label='Simple LR Prediction')
plt.scatter(X_test_multi[:, 0], y_pred_multi, color='green', label='Multiple LR Prediction', marker='x')
plt.title('House Price Prediction')
plt.xlabel('Size (sq ft)')
plt.ylabel('Price (thousands)')
plt.legend()
plt.savefig('house_price_prediction.png')
plt.show()

print(f"Simple LR R^2: {model_simple.score(X_test, y_test):.4f}")
print(f"Multiple LR R^2: {model_multi.score(X_test_multi, y_test_multi):.4f}")