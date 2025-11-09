import pandas as pd
import numpy as np
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from google.colab import drive
import os


nltk.download('vader_lexicon', quiet=True)


print("Mounting Google Drive...")
drive.mount('/content/drive')


base_path = '/content/drive/MyDrive/IMDB'


required_files = ['Test.csv'] 
for file in required_files:
    file_path = os.path.join(base_path, file)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    else:
        print(f"Found: {file}")

# Load test.csv
df = pd.read_csv(os.path.join(base_path, 'Test.csv'))
print(f"\nLoaded {len(df)} rows from test.csv")
print("Columns:", df.columns.tolist())

text_column = 'text'
if text_column not in df.columns:
    raise ValueError(f"Column '{text_column}' not found! Available: {df.columns.tolist()}")
text_series = df[text_column].copy()

# --- Step d: Remove @handles ---
def remove_handles(text):
    return re.sub(r'@\w+', ' ', str(text).strip())

df['clean_text'] = text_series.apply(remove_handles)
print("\nHandles removed from text.")

print("Running VADER sentiment analysis...")
sid = SentimentIntensityAnalyzer()

df['vader_score'] = df['clean_text'].apply(lambda x: sid.polarity_scores(x)['compound'])

def classify_sentiment(score):
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

df['sentiment'] = df['vader_score'].apply(classify_sentiment)

sentiment_df = df[['vader_score', 'sentiment']].copy()
print("Sentiment analysis completed.")


final_df = df.copy()
print(f"\nFinal DataFrame: {final_df.shape[0]} rows × {final_df.shape[1]} columns")

positive_df = final_df[final_df['sentiment'] == 'positive']
negative_df = final_df[final_df['sentiment'] == 'negative']
neutral_df  = final_df[final_df['sentiment'] == 'neutral']

print(f"\nSentiment Distribution:")
print(f"  Positive : {len(positive_df)}")
print(f"  Negative : {len(negative_df)}")
print(f"  Neutral  : {len(neutral_df)}")

print("\n--- Sample Positive Reviews ---")
print(positive_df[['clean_text', 'vader_score']].head(3).to_string(index=False))

print("\n--- Sample Negative Reviews ---")
print(negative_df[['clean_text', 'vader_score']].head(3).to_string(index=False))

print("\n--- Sample Neutral Reviews ---")
print(neutral_df[['clean_text', 'vader_score']].head(3).to_string(index=False))
