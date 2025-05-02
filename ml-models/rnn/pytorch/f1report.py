"""
f1report.py

best_model_pytorch.pt              # state-dict you saved with torch.save(model.state_dict(), …)
tokenizer.pkl                      # for inputs later on
Processed_Symptom2Disease_Updated.csv   # cols: label, label_name (string for human readability)
test.csv                           # all cols: label( weights of symptoms int)
"""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

# ── CONFIG ──────────────────────────────────────────────────────────
MODEL_PATH  = Path("best_model_pytorch.pt")
MAP_CSV     = Path("Processed_Symptom2Disease_Updated.csv")
TEST_CSV    = Path("test.csv")
TOKEN_PATH  = Path("tokenizer.pkl")        # optional - not used in this version

VOCAB_SIZE      = 1283
EMBEDDING_DIM   = 64
NUM_CLASSES     = 24

BATCH_SIZE   = 64
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FIG_NAME     = "f1_per_class_pytorch.png"
FIGSIZE      = (18, 6)
ROTATION     = 30


# ── Model – identical to training-time version ─────────────────────
class BiLSTMClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
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

    def forward(self, x: torch.Tensor, lengths=None) -> torch.Tensor:
        x = self.embedding(x)

        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        _, (h1, _) = self.lstm1(x)                      
        h = torch.cat((h1[-2], h1[-1]), dim=1)          
        h = self.dropout1(h)

        h = h.unsqueeze(1)                              
        _, (h2, _) = self.lstm2(h)                      
        h2 = torch.cat((h2[-2], h2[-1]), dim=1)         

        z = torch.relu(self.fc(h2))
        z = self.dropout2(z)
        return self.out(z)                              


def maybe_load_tokenizer(path: Path):
    try:
        with path.open("rb") as fh:
            return pickle.load(fh)
    except FileNotFoundError:
        return None


def pad_sequences(token_lists, pad_value=0):
    max_len = max(len(seq) for seq in token_lists)
    out = np.full((len(token_lists), max_len), pad_value, dtype=np.int64)
    for i, seq in enumerate(token_lists):
        out[i, : len(seq)] = seq
    return out


def main() -> None:
    # label-id → name map
    label_names = (
        pd.read_csv(MAP_CSV, usecols=["label", "label_name"])
        .drop_duplicates()
        .sort_values("label")["label_name"]
        .tolist()
    )
    assert len(label_names) == NUM_CLASSES, "NUM_CLASSES constant out of sync"

    # build & load model
    model = BiLSTMClassifier().to(DEVICE)
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    print(" state-dict loaded and model ready\n")

    # test data
    df_test = pd.read_csv(TEST_CSV)
    y_true  = df_test["label"].to_numpy()

    X_pad = df_test.drop(columns=["label"]).to_numpy(dtype=np.int64)
    lengths = (X_pad != 0).sum(axis=1)          # needed for pack_padded_sequence

    ds = torch.utils.data.TensorDataset(
        torch.tensor(X_pad, dtype=torch.long),
        torch.tensor(lengths, dtype=torch.long)
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    # inference
    y_pred = []
    with torch.no_grad():
        for xb, ln in dl:
            logits = model(xb.to(DEVICE), ln.to(DEVICE))
            y_pred.extend(logits.argmax(dim=1).cpu().numpy())
    y_pred = np.array(y_pred)

    # per-class F1
    f1_scores = f1_score(
        y_true, y_pred, average=None, labels=np.arange(NUM_CLASSES)
    )

    # 6️⃣ plot
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bars = ax.bar(np.arange(NUM_CLASSES), f1_scores * 100)
    ax.set_title("Per-Class F1 Scores (Bi-LSTM PyTorch)", weight="bold", pad=12)
    ax.set_xlabel("Disease Class")
    ax.set_ylabel("F1 Score (%)")
    ax.set_xticks(np.arange(NUM_CLASSES), labels=label_names,
                  rotation=ROTATION, ha="right")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)

    for bar, score in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 2,
                f"{score*100:.1f}",
                ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_NAME, dpi=150)
    print(f"📈  Figure saved to {FIG_NAME}")
    plt.show()


if __name__ == "__main__":
    main()
