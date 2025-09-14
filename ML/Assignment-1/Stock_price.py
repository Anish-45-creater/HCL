import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Download data (AAPL as example)
data = yf.download('AAPL', start='2020-01-01', end='2023-01-01')
close_prices = data['Close'].values.reshape(-1, 1)
dates = data.index  # Get dates for plotting

# Preprocess
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(close_prices)
train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]
test_dates = dates[train_size + 60:]  # Adjust for sequence length

def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length].flatten())  # Flatten sequence for regression
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Create sequences
X_train, y_train = create_sequences(train_data)
X_test, y_test = create_sequences(test_data)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train.ravel())  # Flatten y_train for sklearn

# Test
y_pred = model.predict(X_test)

# Inverse transform predictions and actual values
y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Print sample predictions
print("Sample predictions (first 5):", y_pred_inv[:5])

# Evaluate model
mse = mean_squared_error(y_test_inv, y_pred_inv)
print(f"Mean Squared Error on test set: {mse:.4f}")

# Visualization: Plot actual vs predicted prices
plt.figure(figsize=(10, 6))
plt.plot(test_dates, y_test_inv, label='Actual Prices', color='blue')
plt.plot(test_dates, y_pred_inv, label='Predicted Prices', color='red', linestyle='--')
plt.title('Actual vs Predicted AAPL Stock Prices (Linear Regression)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('stock_price_prediction.png')
plt.show()
plt.close()