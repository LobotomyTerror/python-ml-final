import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import set_random_seed

# === Config ===
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
set_random_seed(623)

DATA_PATH = "Data/Processed_Symptom2DiseaseGroup.csv"
MODEL_SAVE_PATH = "Data/best_model.keras"
ENCODER_SAVE_PATH = "Data/label_encoder.pkl"

VOCAB_SIZE = 1283
NUM_CLASSES = 24
EMBEDDING_DIM = 64
NUM_FOLDS = 10

# === Load dataset ===
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["label"]).values
y = df["label"].values

# === Encode labels ===
le = LabelEncoder()
y = le.fit_transform(y)

# Save label encoder to file
with open(ENCODER_SAVE_PATH, "wb") as f:
    pickle.dump(le, f)
print(f"✅ LabelEncoder saved to: {ENCODER_SAVE_PATH}")

# === Build model ===
def create_model():
    model = Sequential([
        Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM),
        Bidirectional(LSTM(128, return_sequences=True, dropout=0.25, recurrent_dropout=0.25, kernel_regularizer=l2(0.0008))),
        Dropout(0.4),
        Bidirectional(LSTM(64, return_sequences=False, dropout=0.25, recurrent_dropout=0.25, kernel_regularizer=l2(0.0008))),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=l2(0.0008)),
        Dropout(0.4),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# === Cross-validation training ===
kfold = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
train_accuracies, fold_accuracies = [], []
best_val_acc = 0.0

for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y), 1):
    print(f"\n🔁 Fold {fold}/{NUM_FOLDS}")
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = create_model()
    early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=64,
        callbacks=[early_stop],
        verbose=0
    )

    # Evaluate
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    print(f"✅ Fold {fold} Validation Accuracy: {val_acc:.4f}")
    print(f"📈 Fold {fold} Training Accuracy: {train_acc:.4f}")
    fold_accuracies.append(val_acc)
    train_accuracies.append(train_acc)

    # Save best model
    if val_acc > best_val_acc:
        print("💾 Saving new best model!")
        best_val_acc = val_acc
        model.save(MODEL_SAVE_PATH)

# === Final report ===
print("\n📊 Cross-Validation Results:")
for i in range(NUM_FOLDS):
    print(f"Fold {i+1}: Train Acc = {train_accuracies[i]:.4f} | Val Acc = {fold_accuracies[i]:.4f} | Gap = {train_accuracies[i] - fold_accuracies[i]:.4f}")
print(f"\n✅ Average Validation Accuracy: {np.mean(fold_accuracies):.4f}")
print(f"✅ Average Training Accuracy:   {np.mean(train_accuracies):.4f}")
print(f"✅ Standard Deviation:          {np.std(fold_accuracies):.4f}")
print(f"✅ Best model saved to:         {MODEL_SAVE_PATH}")
