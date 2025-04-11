"""
rnn_model.py

Loads preprocessed symptom data, trains an RNN for disease classification,
evaluates performance, and saves the model and label encoder.

Author: Hector Ramirez-Zubiria
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Dropout, LSTM, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import pickle

with open("tokenizer/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

vocab_size = len(tokenizer.word_index) + 1


def load_data(train_path: str, test_path: str):
    """Load training and testing CSVs."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop("label", axis=1).values
    y_train = train_df["label"].values

    X_test = test_df.drop("label", axis=1).values
    y_test = test_df["label"].values

    return X_train, y_train, X_test, y_test

def encode_labels(y_train_raw, y_test_raw):
    """Convert string labels to integers and one-hot vectors."""
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train_raw)
    y_test_encoded = label_encoder.transform(y_test_raw)

    y_train_onehot = to_categorical(y_train_encoded)
    y_test_onehot = to_categorical(y_test_encoded)

    return y_train_onehot, y_test_onehot, label_encoder

def build_rnn(vocab_size: int, input_length: int, output_dim: int):
    """Builds a simple RNN classification model."""
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=128),
        Bidirectional(LSTM(128)),
        Dropout(0.2),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(output_dim, activation="softmax")
    ])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

if __name__ == "__main__":
    # File paths
    train_path = "Data/training.csv"
    test_path = "Data/testing.csv"

    # Load and encode data
    X_train, y_train_raw, X_test, y_test_raw = load_data(train_path, test_path)
    y_train, y_test, label_encoder = encode_labels(y_train_raw, y_test_raw)

    input_length = X_train.shape[1]
    output_dim = y_train.shape[1]

    # Build and train model
    model = build_rnn(vocab_size, input_length, output_dim)
    model.summary()

    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=13,
        batch_size=16,
        callbacks=[early_stop]
    )

    # Evaluate on test set
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"✅ Test Accuracy: {accuracy:.2%}")

    # Save model and label encoder
    model.save("HectorSymptomRNN.keras")  # Recommended
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    print("✅ Model and label encoder saved.")
