# -----------------------------
# Full MNIST MLP with different splits & optimizers
# -----------------------------
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# Enable eager execution (to avoid .numpy() errors)
tf.config.run_functions_eagerly(True)

# -----------------------------
# 1. Load MNIST dataset
# -----------------------------
(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

# Normalize images
X_train_full = X_train_full.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# One-hot encode labels
y_train_full = to_categorical(y_train_full, 10)
y_test = to_categorical(y_test, 10)

# -----------------------------
# 2. Function to create train/validation split
# -----------------------------
def create_train_val(X, y, train_ratio=0.8):
    num_train = int(train_ratio * X.shape[0])
    X_train = X[:num_train]
    y_train = y[:num_train]
    X_val = X[num_train:]
    y_val = y[num_train:]
    return X_train, y_train, X_val, y_val

# -----------------------------
# 3. Build MLP model function
# -----------------------------
def build_mlp():
    model = Sequential([
        Flatten(input_shape=(28,28)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model

# -----------------------------
# 4. Train & evaluate function
# -----------------------------
def train_and_evaluate(X_train, y_train, X_val, y_val, optimizer_name):
    model = build_mlp()
    
    if optimizer_name == 'SGD':
        optimizer = SGD()
    elif optimizer_name == 'Adam':
        optimizer = Adam()
    elif optimizer_name == 'RMSprop':
        optimizer = RMSprop()
    else:
        raise ValueError("Unsupported optimizer")
    
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,
        batch_size=64,
        verbose=0
    )
    
    val_acc = history.history['val_accuracy'][-1]  # Get last validation accuracy
    return val_acc

# -----------------------------
# 5. Run experiments for different splits and optimizers
# -----------------------------
splits = [0.6, 0.7, 0.8]
optimizers = ['SGD', 'Adam', 'RMSprop']
results = {}

for split in splits:
    X_train, y_train, X_val, y_val = create_train_val(X_train_full, y_train_full, split)
    results[split] = {}
    
    for opt in optimizers:
        val_acc = train_and_evaluate(X_train, y_train, X_val, y_val, opt)
        results[split][opt] = val_acc
        print(f"Split {split:.1f}, Optimizer {opt}, Val Accuracy: {val_acc:.4f}")

# -----------------------------
# 6. Plot results
# -----------------------------
plt.figure(figsize=(8,6))
for opt in optimizers:
    val_accs = [results[s][opt] for s in splits]
    plt.plot(splits, val_accs, marker='o', label=opt)

plt.xlabel('Train Split Ratio')
plt.ylabel('Validation Accuracy')
plt.title('MLP Accuracy on MNIST with Different Optimizers')
plt.legend()
plt.grid(True)
plt.show()
