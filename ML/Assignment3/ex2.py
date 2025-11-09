# a. Import the necessary packages
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np


data = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
df = pd.DataFrame({'text': data.data, 'target': data.target})
print("Dataset Shape:", df.shape)
print("\nSample Data:\n", df.head())


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


df['cleaned_text'] = df['text'].apply(clean_text)
print("\nSample Cleaned Text:\n", df['cleaned_text'].head())


bow_vectorizer = CountVectorizer(max_features=10000, stop_words='english')  
bow_matrix = bow_vectorizer.fit_transform(df['cleaned_text'])
bow_feature_names = np.array(bow_vectorizer.get_feature_names_out())

tfidf_vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['cleaned_text'])
tfidf_feature_names = np.array(tfidf_vectorizer.get_feature_names_out())


bow_sums = np.array(bow_matrix.sum(axis=0)).flatten()
bow_top_indices = np.argsort(bow_sums)[-20:][::-1]  # Top 20 indices (descending)
bow_top_words = bow_feature_names[bow_top_indices]
bow_top_scores = bow_sums[bow_top_indices]

tfidf_sums = np.array(tfidf_matrix.sum(axis=0)).flatten()
tfidf_top_indices = np.argsort(tfidf_sums)[-20:][::-1]
tfidf_top_words = tfidf_feature_names[tfidf_top_indices]
tfidf_top_scores = tfidf_sums[tfidf_top_indices]


comparison_df = pd.DataFrame({
    'BoW Word': bow_top_words,
    'BoW Frequency': bow_top_scores.astype(int),
    'TF-IDF Word': tfidf_top_words,
    'TF-IDF Score': np.round(tfidf_top_scores, 2)
})

print("\nComparison of Top 20 Words (BoW vs TF-IDF):\n")
print(comparison_df.to_string(index=False))
