import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, mean_squared_error, r2_score, roc_curve, precision_recall_curve, average_precision_score
import xgboost as xgb
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import warnings
warnings.filterwarnings('ignore')

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)


data = {
    'reviewText': [
        "Not much to write about here, but it does exactly what it is supposed to do.",
        "The product does exactly as it should and is quite affordable.",
        "The primary job of this device is to block the wind noise when recording into a microphone.",
        "Nice windscreen protects my MXL mic and prevents the wind noise.",
        "This pop filter is great. It looks and performs like a studio filter.",
        "So good that I bought another one. Love the high quality and low price.",
        "I have used monster cables for years, and with good reason.",
        "I now use this cable to run from the output of my pedal chain to my amplifier.",
        "Perfect for my Epiphone Sheraton II. Monster cables are well made.",
        "Monster makes the best cables and a lifetime warranty."
    ],
    'Overall': [5, 4, 3, 5, 4, 4, 5, 5, 3, 4]
}

df = pd.DataFrame(data)
print("Complete DataFrame with Full Sentences:\n")
print(df)
print("\n" + "="*60 + "\n")

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [ps.stem(word) for word in tokens if word not in stop_words and word.isalpha()]
    return ' '.join(tokens)

df['cleaned_text'] = df['reviewText'].apply(clean_text)
print("Cleaned Review Texts:\n")
print(df[['cleaned_text', 'Overall']])
print("\n" + "="*60 + "\n")


tfidf = TfidfVectorizer(max_features=100, ngram_range=(1,2))
X = tfidf.fit_transform(df['cleaned_text']).toarray()
y = df['Overall']

feature_names = tfidf.get_feature_names_out()
print(f"TF-IDF Features Shape: {X.shape}")
print(f"Sample Features: {feature_names[:20]}")
print("\n" + "="*60 + "\n")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"Training Samples: {len(X_train)}, Test Samples: {len(X_test)}")
print(f"Train Labels: {sorted(y_train)}, Test Labels: {sorted(y_test)}")
print("\n" + "="*60 + "\n")

# ===================== CLASSIFICATION MODELS =====================
class_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Naive Bayes': MultinomialNB(),
    'KNN': KNeighborsClassifier(n_neighbors=3),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'GBM': GradientBoostingClassifier(random_state=42),
    'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
}

class_results = []

print("CLASSIFICATION RESULTS\n")
for name, model in class_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    precision = report['macro avg']['precision']
    recall = report['macro avg']['recall']
    f1 = report['macro avg']['f1-score']
    
    roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr') if y_proba is not None else np.nan
    
    class_results.append({
        'Model': name,
        'Accuracy': round(acc, 3),
        'Precision': round(precision, 3),
        'Recall': round(recall, 3),
        'F1-Score': round(f1, 3),
        'ROC AUC': round(roc_auc, 3) if not np.isnan(roc_auc) else 'N/A'
    })
    
    print(f"{name}")
    print(f"   Accuracy: {acc:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")
    print(f"   Confusion Matrix:\n{cm}")
    print("-" * 50)


class_df = pd.DataFrame(class_results)
print("\nCLASSIFICATION PERFORMANCE TABLE")
print(class_df.to_string(index=False))
print("\n" + "="*60 + "\n")


reg_models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree Reg': DecisionTreeRegressor(random_state=42),
    'Random Forest Reg': RandomForestRegressor(n_estimators=100, random_state=42),
    'GBM Regressor': GradientBoostingRegressor(random_state=42),
    'XGB Regressor': xgb.XGBRegressor(random_state=42)
}

reg_results = []

print("REGRESSION RESULTS\n")
for name, model in reg_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    reg_results.append({
        'Model': name,
        'MSE': round(mse, 3),
        'R² Score': round(r2, 3)
    })
    
    print(f"{name}")
    print(f"   MSE: {mse:.3f} | R²: {r2:.3f}")
    print(f"   Predictions: {y_pred.round(2)}")
    print(f"   Actual:      {y_test.values}")
    print("-" * 50)


reg_df = pd.DataFrame(reg_results)
print("\nREGRESSION PERFORMANCE TABLE")
print(reg_df.to_string(index=False))
print("\n" + "="*60 + "\n")

print("Generating ROC and Precision-Recall Curves...")

model_ovr = OneVsRestClassifier(LogisticRegression(max_iter=1000))
model_ovr.fit(X_train, y_train)
y_score = model_ovr.predict_proba(X_test)
y_test_bin = label_binarize(y_test, classes=[3, 4, 5])
n_classes = y_test_bin.shape[1]

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = roc_auc_score(y_test_bin[:, i], y_score[:, i])
    plt.plot(fpr, tpr, label=f'Class {i+3} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (One-vs-Rest)')
plt.legend(loc="lower right")


plt.subplot(1, 2, 2)
for i in range(n_classes):
    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
    ap = average_precision_score(y_test_bin[:, i], y_score[:, i])
    plt.plot(recall, precision, label=f'Class {i+3} (AP = {ap:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.tight_layout()
plt.show()

print("\nAll tasks completed successfully!")
print("Note: With only 10 samples (3 in test), metrics may be unstable. Use larger dataset for robust evaluation.")
