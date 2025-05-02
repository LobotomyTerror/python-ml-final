"""
Lightweight inference using full model + original tokenizer
-----------------------------------------------------------
Deps: torch, nltk, contractions, pickle, regex
"""

from __future__ import annotations
import pickle, re, string, contractions, torch, nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

# ── ensure tiny NLTK footprint ───────────────────────────────────────────────
for res in ("punkt", "stopwords"):
    try:
        nltk.data.find(f"tokenizers/{res}" if res == "punkt" else f"corpora/{res}")
    except LookupError:
        nltk.download(res, quiet=True)

_STOP  = set(stopwords.words("english"))
_PUNC  = str.maketrans("", "", string.punctuation)
PAD_IDX, MAX_LEN = 0, 27            # fixed from training

ID2NAME = [                         # 24 classes
    "Acne","Arthritis","Bronchial Asthma","Cervical spondylosis","Chicken pox",
    "Common Cold","Dengue","Dimorphic Hemorrhoids","Fungal infection",
    "Hypertension","Impetigo","Jaundice","Malaria","Migraine","Pneumonia",
    "Psoriasis","Typhoid","Varicose Veins","allergy","diabetes",
    "drug reaction","gastroesophageal reflux disease",
    "peptic ulcer disease","urinary tract infection"
]

# ── load tokenizer & build encoder ------------------------------------------
with open("tokenizer_pytorch.pkl", "rb") as f:
    tok = pickle.load(f)

# works for both real Keras Tokenizer and plain dict
if hasattr(tok, "word_index"):
    word_index = tok.word_index                # Keras tokenizer
elif isinstance(tok, dict):
    # if it's already {word:int, ...} just use it directly
    word_index = tok
else:
    raise TypeError("tokenizer_pytorch.pkl format not recognised")


def encode(tokens: list[str]) -> list[int]:
    return [word_index.get(t, 0) for t in tokens]

def pad(seq: list[int]) -> list[int]:
    return seq[:MAX_LEN] + [PAD_IDX] * (MAX_LEN - len(seq))

def preprocess(text: str) -> list[int]:
    text = contractions.fix(text).lower().translate(_PUNC)
    tokens = [w.rstrip('s') for w in word_tokenize(text) if w not in _STOP]
    return pad(encode(tokens))

# ── load model class definition --------------------------------------------
PAD_IDX      = 0
EMBED_DIM    = 64
LSTM1_HIDDEN = 128
LSTM2_HIDDEN = 64
FC_OUT_DIM   = 64
NUM_CLASSES  = 24       # len(ID2NAME)

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_sz: int = 1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_sz, EMBED_DIM, padding_idx=PAD_IDX)
        self.lstm1 = nn.LSTM(EMBED_DIM, LSTM1_HIDDEN, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(LSTM1_HIDDEN * 2, LSTM2_HIDDEN, batch_first=True, bidirectional=True)
        self.fc  = nn.Linear(LSTM2_HIDDEN * 2, FC_OUT_DIM)
        self.out = nn.Linear(FC_OUT_DIM, NUM_CLASSES)

    def forward(self, x, lengths=None):
        if lengths is None:
            lengths = (x != PAD_IDX).sum(1).clamp(min=1)
        x = self.embedding(x)
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = self.lstm1(x)
        x, (h_n, _) = self.lstm2(x)
        h_cat = torch.cat((h_n[-2], h_n[-1]), dim=1)
        x = self.fc(h_cat)
        return self.out(x)


# ── load full model object ---------------------------------------------------
model = torch.load("rnn_lstm_full.pt", map_location="cpu", weights_only=False)
model.eval()

def predict(prompt: str) -> tuple[int, str]:
    seq     = preprocess(prompt)                     # list[int]
    length  = max(1, len([t for t in seq if t != PAD_IDX]))  # ensure ≥1
    x       = torch.tensor([seq], dtype=torch.long)
    with torch.no_grad():
        idx = model(x, torch.tensor([length])).argmax(1).item()
    return idx, ID2NAME[idx]


# ── CLI ----------------------------------------------------------------------
if __name__ == "__main__":
    prompt = ("For the past week I’ve had a painful, itchy rash on my skin. "
              "The area is covered with pus‑filled pimples and blackheads, "
              "feels swollen, and sometimes burns when I touch it.")
    i, name = predict(prompt)
    print(f"Predicted class: {i} → {name}")
