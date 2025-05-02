"""
Preprocessing pipeline for RNN-based symptom-to-disease classification.
Steps:
1. Load data from Data/Symptom2DiseaseGroup.csv
2. Clean and preprocess text (lowercase, expand contractions, remove punctuation, remove stopwords, lemmatize)
3. Tokenize and pad sequences for RNN compatibility
4. Encode labels
5. Export both cleaned and tokenized data to CSV
6. Save tokenizer and label encoder for reuse
"""

import pickle
import pandas as pd
import numpy as np
import string
import contractions
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Ensure required NLTK data is downloaded
nltk_resources = {
    "punkt": "tokenizers/punkt",
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
    "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
    "omw-1.4": "corpora/omw-1.4",
    "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng"
}
for resource, path in nltk_resources.items():
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource)

# Load dataset
DATA_PATH = "Data/Symptom2DiseaseGroup.csv"
df = pd.read_csv(DATA_PATH)
df.columns = ['label', 'text']  # Ensure columns are named correctly

# Text preprocessing functions
def expand_contractions(text):
    return contractions.fix(text)

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text, preserve_line=True)
    return " ".join([word for word in words if word.lower() not in stop_words])

def pos_tagging(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text, preserve_line=True)
    tagged_words = pos_tag(words, lang='eng')
    return " ".join([lemmatizer.lemmatize(word, pos_tagging(tag)) for word, tag in tagged_words])

def preprocess_text(text):
    text = text.lower()
    text = expand_contractions(text)
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text

# Apply preprocessing
df['clean_text'] = df['text'].astype(str).apply(preprocess_text)

# Save cleaned text before tokenization
df.to_csv("Data/Cleaned_Symptom2DiseaseGroup.csv", index=False)

# Tokenization and padding for RNN
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(df['clean_text'])
sequences = tokenizer.texts_to_sequences(df['clean_text'])
padded_sequences = pad_sequences(sequences, padding='post')

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(df['label'])

# Export-ready variables
X = padded_sequences
y = encoded_labels
word_index = tokenizer.word_index
num_classes = len(label_encoder.classes_)

# Save tokenized/padded data to CSV
processed_df = pd.DataFrame(X)
processed_df.insert(0, 'label', y)
processed_df.to_csv("Data/Processed_Symptom2DiseaseGroup.csv", index=False)

# Save tokenizer and label encoder for later reuse
with open("Data/tokenizer.pkl", "wb") as f_tok:
    pickle.dump(tokenizer, f_tok)

with open("Data/label_encoder.pkl", "wb") as f_enc:
    pickle.dump(label_encoder, f_enc)

# Summary output
print(f"✅ Data preprocessing complete.")
print(f"📚 Vocabulary size: {len(word_index)}")
print(f"📏 Sequence shape: {X.shape}")
print(f"🔢 Number of classes: {num_classes}")
print(f"💾 Tokenizer and label encoder saved in Data/")
