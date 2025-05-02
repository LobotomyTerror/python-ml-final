#!/usr/bin/env python3
"""
compare_tokenizers.py
Checks that tokenizer.pkl (Keras) and tokenizer_pytorch.pkl (PyTorch‑style)
produce the same integer sequence for a given sentence.
"""

import pickle

SENTENCE = (
    "really bad rash skin lately full pusfilled pimple blackhead "
    "skin also scurring lot"
)

KERAS_PKL   = "tokenizer.pkl"
PYTORCH_PKL = "tokenizer_pytorch.pkl"


# ── 1. load Keras tokenizer ───────────────────────────────────────
with open(KERAS_PKL, "rb") as f:
    keras_tok = pickle.load(f)

keras_seq = keras_tok.texts_to_sequences([SENTENCE])[0]


# ── 2. load PyTorch tokenizer (dict, tuple, or class) ─────────────
with open(PYTORCH_PKL, "rb") as f:
    pt_obj = pickle.load(f)

# The file might store:
#   • dict  with keys "word2idx", "oov_idx", "pad_idx"
#   • tuple (word2idx, pad_idx, oov_idx)
#   • custom class  having .word2idx/.oov_idx attributes

if isinstance(pt_obj, dict) and "word2idx" in pt_obj:
    word2idx = pt_obj["word2idx"]
    oov_idx  = pt_obj.get("oov_idx", 1)
elif isinstance(pt_obj, tuple):
    word2idx, _, oov_idx = (pt_obj + (1, 1, 1))[:3]  # safe slicing
else:  # custom class
    word2idx = getattr(pt_obj, "word2idx", None)
    oov_idx  = getattr(pt_obj, "oov_idx", 1)

if word2idx is None:
    raise ValueError("❌ Could not extract word2idx from tokenizer_pytorch.pkl")


def encode_pytorch(text: str) -> list[int]:
    return [word2idx.get(tok, oov_idx) for tok in text.split()]


pytorch_seq = encode_pytorch(SENTENCE)


# ── 3. compare & report ───────────────────────────────────────────
print("\nSentence:")
print(" ", SENTENCE)
print("\nKeras   :", keras_seq)
print("PyTorch :", pytorch_seq)

if keras_seq == pytorch_seq:
    print("\n✅ Tokenizers are identical for this sentence.")
else:
    print("\n❌ Sequences differ!")
    max_len = max(len(keras_seq), len(pytorch_seq))
    for i in range(max_len):
        k = keras_seq[i]  if i < len(keras_seq)  else None
        p = pytorch_seq[i] if i < len(pytorch_seq) else None
        if k != p:
            print(f"   • position {i:>2}: Keras={k}  |  PyTorch={p}")
