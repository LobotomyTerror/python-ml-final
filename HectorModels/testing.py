#temp for testing
import pickle

with open("tokenizer/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
index_to_word = {v: k for k, v in tokenizer.word_index.items()}
def decode_sequence(sequence):
    return [index_to_word.get(idx, "<PAD>") for idx in sequence if idx != 0]
import pandas as pd

df = pd.read_csv("Data/testing.csv")  # or training.csv

# Example: get words from the first row
tokens = df.iloc[0][:-1].values.astype(int)  # exclude label
words = decode_sequence(tokens)

print("🧠 Decoded Text:", " ".join(words))
print("🩺 Label:", df.iloc[0]["label"])
