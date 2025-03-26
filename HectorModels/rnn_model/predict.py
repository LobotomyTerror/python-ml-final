"""
predict.py

Loads the trained LSTM model, tokenizer, and label encoder,
then predicts the disease from a user's symptom description.

Author: Hector Ramirez-Zubiria
"""

import re
import string
import pickle
import numpy as np
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download NLTK resources if not already installed
import nltk
nltk.download("punkt")
nltk.download("stopwords")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# === 1. Load tokenizer, label encoder, and model ===
with open("tokenizer/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

model = load_model("Bi_LSTM64_94_drop2_batch16.keras")
MAX_LEN = 20  # must match what was used during training

# === 2. Preprocess input text ===
stop_words = set(stopwords.words("english"))

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

def prepare_input(text: str, max_len: int = MAX_LEN) -> np.ndarray:
    clean = preprocess_text(text)
    seq = tokenizer.texts_to_sequences([clean])
    padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    return padded

# === 3. Predict disease from input ===
def predict_disease(symptom_description: str) -> str:
    X = prepare_input(symptom_description)
    probs = model.predict(X)
    pred_index = np.argmax(probs)
    disease = label_encoder.inverse_transform([pred_index])[0]
    return disease

# === 4. Command-line input ===
if __name__ == "__main__":

    print("🩺 Symptom-to-Disease Classifier")
    user_input = input("Describe your symptoms: ")
    prediction = predict_disease(user_input)
    print(f"\n🧠 Predicted Disease: {prediction}")
