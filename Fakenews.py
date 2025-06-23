import pandas as pd
import numpy as np
import re
import string
import tkinter as tk
from tkinter import scrolledtext
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Download stopwords if not already
nltk.download('stopwords')

# Load datasets
true_df = pd.read_csv("C:/Users/priya/OneDrive/Desktop/College/Minor Project/True.csv")
fake_df = pd.read_csv("C:/Users/priya/OneDrive/Desktop/College/Minor Project/Fake.csv")

# Add labels
true_df["label"] = 1  # Real
fake_df["label"] = 0  # Fake

# Define stopwords & punctuation
stop_words = set(stopwords.words("english"))
punctuations = set(string.punctuation)

# Combine and clean data
df = pd.concat([true_df, fake_df], axis=0).reset_index(drop=True)
df.drop(columns=["subject", "date"], inplace=True)

# Text Cleaning Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(f"[{string.punctuation}]", '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

df["text"] = df["text"].apply(clean_text)
fake_df["text"] = fake_df["text"].apply(clean_text)
true_df["text"] = true_df["text"].apply(clean_text)

# ---------- EDA Functions ---------- #

# 1. Label Distribution Plot
def plot_label_distribution(data):
    sns.countplot(x='label', data=data)
    plt.xticks([0, 1], ['Fake News', 'Real News'])
    plt.title("Distribution of Fake and Real News")
    plt.xlabel("News Type")
    plt.ylabel("Count")
    plt.show()

# 2. Word Cloud Plot
def plot_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white',
                          stopwords=stop_words, max_words=100).generate(" ".join(text))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16)
    plt.show()

# 4. Average Number of Words per Article (Box Plot)
def plot_word_count_boxplot(df):
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='label', y='word_count', data=df)
    plt.xticks([0, 1], ['Fake News', 'Real News'])
    plt.title("Word Count Distribution per Article by Label")
    plt.xlabel("News Type")
    plt.ylabel("Number of Words")
    plt.show()

# 5. Character Length Distribution (Histogram)
def plot_char_length_distribution(df):
    df['char_length'] = df['text'].apply(len)
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='char_length', hue='label', bins=50, kde=True)
    plt.title("Character Length Distribution by Label")
    plt.xlabel("Character Length")
    plt.ylabel("Frequency")
    plt.legend(['Fake News', 'Real News'])
    plt.show()

# Call the EDA visualizations 
plot_word_count_boxplot(df)
plot_char_length_distribution(df)

# Call EDA Visualizations
plot_label_distribution(df)
plot_wordcloud(fake_df["text"], "Most Common Words in Fake News")
plot_wordcloud(true_df["text"], "Most Common Words in Real News")


# ---------- Model Training ---------- #

# Split data
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Neural Network Classifier
mlp_model = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation='relu',
                          solver='adam', learning_rate_init=0.001,
                          max_iter=50, random_state=42)

mlp_model.fit(X_train_tfidf, y_train)

# Evaluation
y_pred = mlp_model.predict(X_test_tfidf)
print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ---------- Tkinter GUI ---------- #

def check_news():
    user_input = text_area.get("1.0", tk.END).strip()
    if user_input:
        input_tfidf = vectorizer.transform([clean_text(user_input)])
        prediction = mlp_model.predict(input_tfidf)[0]
        result = "Real News" if prediction == 1 else "Fake News"
        result_label.config(text=f"Prediction: {result}")

# GUI Setup
root = tk.Tk()
root.title("Fake News Detector")

tk.Label(root, text="Enter the news article:").pack()
text_area = scrolledtext.ScrolledText(root, height=10, width=60)
text_area.pack()

tk.Button(root, text="Check News", command=check_news).pack()
result_label = tk.Label(root, text="", font=("Arial", 12, "bold"))
result_label.pack()

root.mainloop()
