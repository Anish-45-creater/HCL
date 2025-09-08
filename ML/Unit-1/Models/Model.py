import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay

# Load data
data = fetch_california_housing()
X = data.data
Y = data.target
y = pd.qcut(Y, q=3, labels=[0, 1, 2]).astype(int)

# Split data
X_train, X_test, Y_train, Y_test, y_train, y_test = train_test_split(X, Y, y, test_size=0.3, random_state=42)


def save_scatter_plot(y_test, y_pred, model_name):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual House Prices")
    plt.ylabel("Predicted House Prices")
    plt.title(f"{model_name} Scatter Plot")
    plt.tight_layout()
    plt.savefig(f"{model_name.lower().replace(' ', '_')}_scatter.png")
    plt.close()

def save_confusion_matrix(y_test, y_pred, model_name):
    plt.figure(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues')
    plt.title(f"{model_name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
    plt.close()

print("=== Linear Regression ===")
model = LinearRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
mean_sq_error = mean_squared_error(Y_test, Y_pred)
r2_Score = r2_score(Y_test, Y_pred)
print(f"Mean Squared Error: {mean_sq_error:.2f}")
print(f"R2 Score: {r2_Score:.2f}")
save_scatter_plot(Y_test, Y_pred, "Linear Regression")

print("\n=== Lasso ===")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = Lasso(alpha=0.6)
model.fit(X_train_scaled, Y_train)
Y_pred = model.predict(X_test_scaled)
mean_sq_error = mean_squared_error(Y_test, Y_pred)
r2_Score = r2_score(Y_test, Y_pred)
print(f"Mean Squared Error: {mean_sq_error:.2f}")
print(f"R2 Score: {r2_Score:.2f}")
save_scatter_plot(Y_test, Y_pred, "Lasso")

print("\n=== Ridge ===")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = Ridge(alpha=0.5)
model.fit(X_train_scaled, Y_train)
Y_pred = model.predict(X_test_scaled)
mean_sq_error = mean_squared_error(Y_test, Y_pred)
r2_Score = r2_score(Y_test, Y_pred)
print(f"Mean Squared Error: {mean_sq_error:.2f}")
print(f"R2 Score: {r2_Score:.2f}")
save_scatter_plot(Y_test, Y_pred, "Ridge")

print("\n=== Classification Models (Binned House Prices) ===")

print("=== Logistic Regression ===")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro')
rec = recall_score(y_test, y_pred, average='macro')
print(f"Accuracy: {acc:.2f}")
print(f"Precision: {prec:.2f}")
print(f"Recall: {rec:.2f}")
save_confusion_matrix(y_test, y_pred, "Logistic Regression")

print("\n=== SVM ===")
model = SVC()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro')
rec = recall_score(y_test, y_pred, average='macro')
print(f"Accuracy: {acc:.2f}")
print(f"Precision: {prec:.2f}")
print(f"Recall: {rec:.2f}")
save_confusion_matrix(y_test, y_pred, "SVM")

print("\n=== KNN ===")
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro')
rec = recall_score(y_test, y_pred, average='macro')
print(f"Accuracy: {acc:.2f}")
print(f"Precision: {prec:.2f}")
print(f"Recall: {rec:.2f}")
save_confusion_matrix(y_test, y_pred, "KNN")

print("\n=== Decision Tree ===")
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro')
rec = recall_score(y_test, y_pred, average='macro')
print(f"Accuracy: {acc:.2f}")
print(f"Precision: {prec:.2f}")
print(f"Recall: {rec:.2f}")
save_confusion_matrix(y_test, y_pred, "Decision Tree")