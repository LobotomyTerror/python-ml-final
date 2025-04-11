"""
Training script for RNN-based disease classification from symptoms.
- Loads pre-split training and testing data from CSV
- Builds and trains an LSTM-based neural network with dropout
- Saves the trained model
- Evaluates the model on training data to test learning ability
- Suppresses unnecessary TensorFlow logs
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings (INFO & WARNING)

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model

# Constants
TRAIN_PATH = "Data/train.csv"
TEST_PATH = "Data/test.csv"
VOCAB_SIZE = 1283  # 1282 + 1 for OOV token
SEQUENCE_LENGTH = 27
NUM_CLASSES = 24
EMBEDDING_DIM = 64

# Load pre-split training and test data
df_train = pd.read_csv(TRAIN_PATH)
df_test = pd.read_csv(TEST_PATH)

# Extract X and y from train/test
X_train = df_train.drop(columns=['label']).values
y_train = df_train['label'].values
X_val = df_test.drop(columns=['label']).values
y_val = df_test['label'].values

# Build the model with dropout for regularization
model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=SEQUENCE_LENGTH),
    LSTM(128, return_sequences=False),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Train the model
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stop]
)

# Save model
model.save("Data/symptom_disease_rnn_model.keras")

# Evaluate on validation
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Evaluate on training to test learning
train_loss, train_accuracy = model.evaluate(X_train, y_train)
print(f"Training Accuracy (overfit check): {train_accuracy:.4f}")
