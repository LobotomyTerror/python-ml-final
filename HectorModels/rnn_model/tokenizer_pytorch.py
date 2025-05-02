import pickle, json

# 1. load Keras tokenizer
with open("Data/tokenizer.pkl", "rb") as f:
    tok = pickle.load(f)

# 2. export its vocab
word2idx = tok.word_index
pad_idx  = 0
oov_idx  = word2idx.get("<OOV>", 1)

# 3. save a richer object
torch_tok = {
    "word2idx": word2idx,
    "pad_idx": pad_idx,
    "oov_idx": oov_idx,
}

with open("tokenizer_pytorch.pkl", "wb") as f:
    pickle.dump(torch_tok, f)

print("✅  tokenizer_pytorch.pkl written")
