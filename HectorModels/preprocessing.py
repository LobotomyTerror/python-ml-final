"""
preprocessing.py

Preprocessing utilities for Symptom2Disease.csv:
- Normalization
- Tokenization
- Optional stemming/lemmatization
- Padding
- Train/test CSV export
- Tokenizer export

CSV file expected columns: 'label' and 'text'

Author: Hector Ramirez-Zubiria
"""

import os
import pandas as pd
import re
import string
import pickle
from typing import List

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk

# Download required NLTK data
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

def normalize_text(text: str) -> str:
    """Lowercase, remove punctuation, and extra spaces."""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_and_clean(text: str, lemmatize: bool = False, stem: bool = False) -> List[str]:
    """Tokenize the text and optionally lemmatize or stem the tokens."""
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [t for t in tokens if t not in stop_words]
    
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    elif stem:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]
    
    return tokens

def preprocess_symptoms(file_path: str,
                        max_len: int = 20,
                        lemmatize: bool = False,
                        stem: bool = False) -> None:
    """Preprocess text data and save train/test CSV files and tokenizer."""
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.lower()

    if "text" not in df.columns or "label" not in df.columns:
        raise KeyError(f"Required columns 'text' or 'label' not found. Found columns: {df.columns.tolist()}")

    processed_texts = []
    for text in df["text"]:
        norm = normalize_text(text)
        tokens = tokenize_and_clean(norm, lemmatize=lemmatize, stem=stem)
        processed_texts.append(" ".join(tokens))
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(processed_texts)
    sequences = tokenizer.texts_to_sequences(processed_texts)
    padded = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")

    padded_df = pd.DataFrame(padded)
    padded_df["label"] = df["label"]

    train_df, test_df = train_test_split(padded_df, test_size=0.2, random_state=42)

    os.makedirs("Data", exist_ok=True)
    train_df.to_csv("Data/training.csv", index=False)
    test_df.to_csv("Data/testing.csv", index=False)

    # ✅ Save tokenizer to rnn_model/tokenizer/here
    tokenizer_dir = "rnn_model/tokenizer"
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.pkl")

    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)

    print("✅ Preprocessing complete.")
    print("📁 Training data: Data/training.csv")
    print("📁 Testing data: Data/testing.csv")
    print(f"📦 Tokenizer saved to: {tokenizer_path}")

if __name__ == "__main__":
    data_path = "Data/Symptom2Disease.csv"
    preprocess_symptoms(data_path, lemmatize=True)
