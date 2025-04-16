# Model testing and accuracy evaluation script
import pickle
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score

# Load the trained model
model = load_model("Data/symptom_disease_rnn_model.keras")

# Load the tokenizer
with open("Data/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load the label encoder
with open("Data/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Reverse lookup: index → word
index_to_word = {v: k for k, v in tokenizer.word_index.items()}

def decode_sequence(sequence):
    """Convert list of token IDs to words, skipping padding (0)."""
    return [index_to_word.get(int(idx), "<PAD>") for idx in sequence if int(idx) != 0]

# Load processed test data
df = pd.read_csv("Data/test.csv")  # Make sure this file matches tokenized format

# Separate features and labels
X_test = df.drop(columns=['label']).values
y_true = df['label'].values

# Predict all labels
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=-1)

# Decode predicted class IDs to labels
y_pred_labels = label_encoder.inverse_transform(y_pred)
y_true_labels = label_encoder.inverse_transform(y_true)

# Accuracy evaluation
accuracy = accuracy_score(y_true, y_pred)
print(f"\n Model Accuracy on Test Set: {accuracy:.4f}")

# Show a few sample predictions with decoded inputs
print("\n Sample Predictions:")
for i in range(min(5, len(X_test))):
    decoded_words = decode_sequence(X_test[i])
    print(f"\n Input: {' '.join(decoded_words)}")
    print(f" True Label: {y_true_labels[i]}")
    print(f" Predicted: {y_pred_labels[i]}")
