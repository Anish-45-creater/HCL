
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
from textblob import TextBlob 


with open('file.txt', 'r') as file:
    text = file.read()

print("Original Text Corpus:\n", text)
tokens = word_tokenize(text)
print("\nFirst 30 Tokens:\n", tokens[:30])

corrected_tokens = []
for token in tokens:
    corrected = str(TextBlob(token).correct())
    corrected_tokens.append(corrected)

print("\nFirst 10 Corrected Tokens:\n", corrected_tokens[:10])
corrected_text = ' '.join(corrected_tokens)
print("\nCorrected Text Corpus:\n", corrected_text)


pos_tags = pos_tag(corrected_tokens)
print("\nPOS Tags for Corrected Tokens:\n", pos_tags)


stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in corrected_tokens if word.lower() not in stop_words]
print("\nFirst 20 Tokens After Removing Stop Words:\n", filtered_tokens[:20])


stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

stemmed_tokens = [stemmer.stem(word) for word in corrected_tokens]
print("\nFirst 20 Stemmed Tokens:\n", stemmed_tokens[:20])

lemmatized_tokens = [lemmatizer.lemmatize(word) for word in corrected_tokens]
print("\nFirst 20 Lemmatized Tokens:\n", lemmatized_tokens[:20])

sentences = sent_tokenize(text)
print("\nTotal Number of Sentences in Original Text Corpus:", len(sentences))
