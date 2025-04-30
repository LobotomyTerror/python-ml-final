"""
Evaluate trained RNN model for disease classification.
Displays accuracy, classification report, and confusion matrix heatmap.

Author: Hector Ramirez-Zubiria
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# === Paths ===
MODEL_PATH = "Data/best_model.keras"
ENCODER_PATH = "Data/label_encoder.pkl"
TEST_CSV_PATH = "Data/test.csv"
CLEAN_LABELS_PATH = "Data/Cleaned_Symptom2DiseaseGroup.csv"

# === Load trained model ===
model = load_model(MODEL_PATH)
print(f"✅ Loaded model from: {MODEL_PATH}")

# === Load label encoder ===
with open(ENCODER_PATH, "rb") as f:
    label_encoder: LabelEncoder = pickle.load(f)
print(f"✅ Loaded LabelEncoder from: {ENCODER_PATH}")

# === Load test data ===
df = pd.read_csv(TEST_CSV_PATH)
X_test = df.drop(columns=["label"]).values
y_true_raw = df["label"].values

# === Encode true labels ===
try:
    y_true_encoded = label_encoder.transform(y_true_raw)
except ValueError as e:
    print(f"❌ Label encoding error: {e}")
    print("🔍 Labels in encoder:", label_encoder.classes_)
    print("🔍 Labels in test data:", np.unique(y_true_raw))
    exit(1)

# === Evaluate on test set ===
loss, accuracy = model.evaluate(X_test, y_true_encoded, verbose=0)
print(f"\n✅ Test Accuracy: {accuracy:.2%}")
print(f"📉 Test Loss: {loss:.4f}")

# === Predictions ===
y_pred_probs = model.predict(X_test, verbose=0)
y_pred_encoded = np.argmax(y_pred_probs, axis=1)

# === Confusion Matrix ===
cm = confusion_matrix(y_true_encoded, y_pred_encoded)

# Load class names from the cleaned dataset, assuming it has all 24 label names
label_df = pd.read_csv(CLEAN_LABELS_PATH)

# Create a map from numeric ID to class name using the mode label for each class index
# This assumes 'label' column in Cleaned CSV is in the same row order as model training
id_to_label_map = (
    label_df
    .groupby("label")
    .first()
    .reset_index()
    .sort_index()  # assumes label indices were assigned in sorted order
)

# Convert to display names
string_labels = [label.replace("_", " ").title() for label in id_to_label_map["label"].values]

# === Classification Report ===
print("\n📋 Classification Report:")
print(classification_report(y_true_encoded, y_pred_encoded, target_names=string_labels))

# === Plot Heatmap ===
plt.figure(figsize=(14, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="magma", 
            xticklabels=string_labels, yticklabels=string_labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix Heatmap")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
