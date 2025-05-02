#!/usr/bin/env python3
"""
Bidirectional-LSTM classifier for Symptom-to-Disease dataset (PyTorch version).

Fixes compared with the original draft:
1. best_val_acc / best_state / best_epoch are reset **inside** each CV fold.
2. The model uses the concatenated forward & backward hidden states
   returned by nn.LSTM instead of `x[:, -1, :]`, so it does not
   accidentally read the padding token.
3. Optional packed-sequence handling shown for completeness.
"""

from __future__ import annotations
import os
import pickle
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ───────────────────────────── Config ──────────────────────────────
SEED                = 623
DATA_PATH           = Path("Data/Processed_Symptom2DiseaseGroup.csv")
MODEL_SAVE_PATH     = Path("Data/best_model_pytorch.pt")
ENCODER_SAVE_PATH   = Path("Data/label_encoder_pytorch.pkl")

VOCAB_SIZE          = 1283
EMBEDDING_DIM       = 64
NUM_CLASSES         = 24

NUM_FOLDS           = 10
BATCH_SIZE          = 64
EPOCHS              = 30           # leave high; early-stop will cut it short
PATIENCE            = 5
DEVICE              = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)

# ──────────────────────── Dataset preparation ──────────────────────
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["label"]).values          # (samples, seq_len)
y = df["label"].values

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)
ENCODER_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(ENCODER_SAVE_PATH, "wb") as f:
    pickle.dump(le, f)
print(f"✅ LabelEncoder saved to: {ENCODER_SAVE_PATH}")

class SymptomDataset(Dataset):
    """Simple wrapper so we can feed numpy arrays into DataLoader."""
    def __init__(self, X_arr: np.ndarray, y_arr: np.ndarray) -> None:
        self.X = torch.LongTensor(X_arr)
        self.y = torch.LongTensor(y_arr)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

# ─────────────────────────── Model ──────────────────────────
class BiLSTMClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # padding_idx=0 ensures its vector stays zeros and, optionally,
        # pack_padded_sequence works as expected
        self.embedding  = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM, padding_idx=0)

        self.lstm1      = nn.LSTM(
            EMBEDDING_DIM, 128, batch_first=True,
            dropout=0.25, bidirectional=True
        )
        self.dropout1   = nn.Dropout(0.4)

        self.lstm2      = nn.LSTM(
            256, 64, batch_first=True,
            dropout=0.25, bidirectional=True
        )

        self.fc         = nn.Linear(128, 64)
        self.dropout2   = nn.Dropout(0.4)
        self.out        = nn.Linear(64, NUM_CLASSES)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args
        ----
        x        : (batch, seq_len)
        lengths  : (batch,) real lengths (optional).
                   If provided, we pack the sequence to ignore pad tokens.
        """
        x = self.embedding(x)

        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        _, (h_n1, _) = self.lstm1(x)          # h_n1: (num_layers*2, B, 128)
        h = torch.cat((h_n1[-2], h_n1[-1]), dim=1)   # (B, 256)
        h = self.dropout1(h)

        h = h.unsqueeze(1)                    # add fake time dim for lstm2
        _, (h_n2, _) = self.lstm2(h)
        h2 = torch.cat((h_n2[-2], h_n2[-1]), dim=1)  # (B, 128)

        z = torch.relu(self.fc(h2))
        z = self.dropout2(z)
        return self.out(z)                    # (B, NUM_CLASSES)

# ───────────────────── Training / evaluation helpers ─────────────────────
def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_correct = 0

    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        lengths = (xb != 0).sum(1)           # counts real tokens
        optimizer.zero_grad()
        logits = model(xb, lengths)
        loss   = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        running_loss    += loss.item() * len(xb)
        running_correct += (logits.argmax(1) == yb).sum().item()

    n = len(loader.dataset)
    return running_loss / n, running_correct / n


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> tuple[float, float]:
    model.eval()
    total_loss, total_correct = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        lengths = (xb != 0).sum(1)
        logits  = model(xb, lengths)
        loss    = criterion(logits, yb)
        total_loss    += loss.item() * len(xb)
        total_correct += (logits.argmax(1) == yb).sum().item()
    n = len(loader.dataset)
    return total_loss / n, total_correct / n

# ───────────────────────────── Cross-validation ───────────────────────────
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

train_accs, val_accs = [], []
best_global_state    = None
best_global_val      = 0.0

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n📂 Fold {fold}/{NUM_FOLDS}")

    train_ds = SymptomDataset(X[train_idx], y[train_idx])
    val_ds   = SymptomDataset(X[val_idx],   y[val_idx])

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    model     = BiLSTMClassifier().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), weight_decay=0.0008)
    criterion = nn.CrossEntropyLoss()

    # fold-local trackers
    best_fold_val   = 0.0
    best_fold_epoch = 0
    best_fold_state = deepcopy(model.state_dict())

    # ─── Training loop ───
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_dl, criterion, optimizer)
        vl_loss, vl_acc = evaluate(model, val_dl, criterion)

        print(f"Epoch {epoch:2d}: Val Acc = {vl_acc:.4f}, Train Acc = {tr_acc:.4f}")

        if vl_acc > best_fold_val:
            best_fold_val   = vl_acc
            best_fold_epoch = epoch
            best_fold_state = deepcopy(model.state_dict())

        elif epoch - best_fold_epoch >= PATIENCE:
            print(f"🛑 Early stopping at epoch {epoch} (no improvement in {PATIENCE} epochs).")
            break

    # restore best weights for this fold
    model.load_state_dict(best_fold_state)

    # final evaluation with best weights
    _, tr_acc_best = evaluate(model, train_dl, criterion)
    _, vl_acc_best = evaluate(model, val_dl, criterion)

    print(f"✅ Fold {fold} Validation Accuracy: {vl_acc_best:.4f}")
    print(f"📈 Fold {fold} Training Accuracy:   {tr_acc_best:.4f}")

    train_accs.append(tr_acc_best)
    val_accs.append(vl_acc_best)

    # keep global best
    if vl_acc_best > best_global_val:
        best_global_val   = vl_acc_best
        best_global_state = deepcopy(model.state_dict())

# ─────────────────────── Save the best overall model ─────────────────────
MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
torch.save(best_global_state, MODEL_SAVE_PATH)
print(f"\n💾 Best model (Val Acc = {best_global_val:.4f}) saved to {MODEL_SAVE_PATH}")

# ───────────────────────── Final CV report ───────────────────────────────
print("\n📊 Cross-Validation Results:")
for i, (tr, vl) in enumerate(zip(train_accs, val_accs), 1):
    print(f"Fold {i:2d}: Train = {tr:.4f} | Val = {vl:.4f} | Gap = {tr - vl:+.4f}")

print(f"\n🔎 Avg  Train Acc: {np.mean(train_accs):.4f}")
print(f"🔎 Avg  Val   Acc: {np.mean(val_accs):.4f}")
print(f"σ (Val Acc):      {np.std(val_accs):.4f}")
