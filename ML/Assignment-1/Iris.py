import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

# Feature importance plot
feature_importance = model.feature_importances_
features = iris.feature_names
plt.bar(features, feature_importance)
plt.title('Feature Importance in Iris Classification')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.savefig('iris_feature_importance.png')
plt.show()