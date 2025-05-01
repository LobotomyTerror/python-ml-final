import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from copy import deepcopy

# === Config ===
SEED = 623
torch.manual_seed(SEED)
np.random.seed(SEED)

DATA_PATH = "Data/Processed_Symptom2DiseaseGroup.csv"
MODEL_SAVE_PATH = "Data/best_model_pytorch.pt"
ENCODER_SAVE_PATH = "Data/label_encoder_pytorch.pkl"

VOCAB_SIZE = 1283
NUM_CLASSES = 24
EMBEDDING_DIM = 64
NUM_FOLDS = 10
BATCH_SIZE = 64
EPOCHS = 20
PATIENCE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load and preprocess dataset ===
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["label"]).values
y = df["label"].values

le = LabelEncoder()
y = le.fit_transform(y)

with open(ENCODER_SAVE_PATH, "wb") as f:
    pickle.dump(le, f)
print(f"✅ LabelEncoder saved to: {ENCODER_SAVE_PATH}")

# === Dataset class ===
class SymptomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.LongTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# === Model definition ===
class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.lstm1 = nn.LSTM(EMBEDDING_DIM, 128, batch_first=True,
                             dropout=0.25, bidirectional=True)
        self.dropout1 = nn.Dropout(0.4)
        self.lstm2 = nn.LSTM(256, 64, batch_first=True,
                             dropout=0.25, bidirectional=True)
        self.dropout2 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(0.4)
        self.out = nn.Linear(64, NUM_CLASSES)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]  # last timestep
        x = torch.relu(self.fc1(x))
        x = self.dropout3(x)
        return self.out(x)

# === Training/validation functions ===
def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss, total_correct = 0, 0
    for xb, yb in dataloader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)
        total_correct += (preds.argmax(1) == yb).sum().item()
    return total_loss / len(dataloader.dataset), total_correct / len(dataloader.dataset)

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss, total_correct = 0, 0
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb)
            loss = criterion(preds, yb)
            total_loss += loss.item() * len(xb)
            total_correct += (preds.argmax(1) == yb).sum().item()
    return total_loss / len(dataloader.dataset), total_correct / len(dataloader.dataset)

# === Cross-validation training ===
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
train_accuracies, val_accuracies = [], []
best_val_acc = 0.0
best_model_state = None

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n🔁 Fold {fold}/{NUM_FOLDS}")
    model = LSTMClassifier().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), weight_decay=0.0008)
    criterion = nn.CrossEntropyLoss()

    train_ds = SymptomDataset(X[train_idx], y[train_idx])
    val_ds = SymptomDataset(X[val_idx], y[val_idx])
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)

    best_fold_acc, best_epoch = 0, 0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_dl, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_dl, criterion)
        if val_acc > best_fold_acc:
            best_fold_acc = val_acc
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
        elif epoch - best_epoch >= PATIENCE:
            print(f"🛑 Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    train_loss, train_acc = evaluate(model, train_dl, criterion)
    val_loss, val_acc = evaluate(model, val_dl, criterion)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    print(f"✅ Fold {fold} Validation Accuracy: {val_acc:.4f}")
    print(f"📈 Fold {fold} Training Accuracy: {train_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = deepcopy(model.state_dict())
        print("💾 Saving new best model!")

# === Save best model ===
torch.save(best_model_state, MODEL_SAVE_PATH)
print(f"✅ Best model saved to: {MODEL_SAVE_PATH}")

# === Final report ===
print("\n📊 Cross-Validation Results:")
for i in range(NUM_FOLDS):
    gap = train_accuracies[i] - val_accuracies[i]
    print(f"Fold {i+1}: Train Acc = {train_accuracies[i]:.4f} | Val Acc = {val_accuracies[i]:.4f} | Gap = {gap:.4f}")
print(f"\n✅ Average Validation Accuracy: {np.mean(val_accuracies):.4f}")
print(f"✅ Average Training Accuracy:   {np.mean(train_accuracies):.4f}")
print(f"✅ Standard Deviation:          {np.std(val_accuracies):.4f}")
