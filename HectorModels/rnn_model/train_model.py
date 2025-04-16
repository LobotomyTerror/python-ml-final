"""
Training script for RNN-based disease classification from symptoms.
- Loads pre-split training and testing data from CSV
- Builds and trains a GRU-based neural network with dropout
- Saves the best performing model during training
- Evaluates the model on training and validation data
- Suppresses unnecessary TensorFlow logs
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO & WARNING logs

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, GRU, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.utils import plot_model
from tensorflow.keras.regularizers import l2
import datetime

# Constants
TRAIN_PATH = "Data/train.csv"
TEST_PATH = "Data/test.csv"
VOCAB_SIZE = 1283  # 1282 + 1 for OOV token
SEQUENCE_LENGTH = 27
NUM_CLASSES = 24
EMBEDDING_DIM = 64
MODEL_PATH = "Data/best_model.keras"

# Load pre-split training and test data
df_train = pd.read_csv(TRAIN_PATH)
df_test = pd.read_csv(TEST_PATH)

# Extract X and y from train/test
X_train = df_train.drop(columns=['label']).values
y_train = df_train['label'].values
X_val = df_test.drop(columns=['label']).values
y_val = df_test['label'].values

# Build the model
model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=SEQUENCE_LENGTH),
    LSTM(128, return_sequences=False, kernel_regularizer=l2(0.001)),
    Dropout(0.25),
    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.25),
    Dense(NUM_CLASSES, activation='softmax')
])


model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Callbacks
early_stop = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True)
log_dir = f"logs/fit/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=[early_stop, checkpoint, tensorboard],
    shuffle=True,
    verbose=1
)

# Load the best model from training
best_model = load_model(MODEL_PATH)

# Evaluate on validation
val_loss, val_accuracy = best_model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Evaluate on training to test learning
train_loss, train_accuracy = best_model.evaluate(X_train, y_train)
print(f"Training Accuracy (overfit check): {train_accuracy:.4f}")
