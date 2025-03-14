import pandas as pd
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.preprocessing import LabelEncoder

# Download necessary NLTK data files (only needed once)
nltk.download('punkt')
nltk.download('wordnet')

def preprocess_text(text):
    """
    Preprocess text by lowercasing, removing non-alphabetic characters,
    tokenizing, and lemmatizing.
    """
    text = text.lower()
    # Remove non-alphabetic characters (optional)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

def split_dataset(df):
    """
    Split the dataset such that in every block of 50 rows,
    the first 45 rows go to training and the next 5 go to testing.
    """
    df = df.reset_index(drop=True)
    train_df = df[df.index % 50 < 45]
    test_df = df[df.index % 50 >= 45]
    return train_df, test_df

def create_model(vocab_size, embedding_dim, input_length, num_classes):
    """
    Build a simple RNN model using an Embedding layer, an LSTM, and a Dense output.
    """
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length),
        LSTM(64),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def main():
    # Load the dataset (ensure your CSV file is named 'data.csv' or adjust the path)
    data_file = 'Data/Symptom2Disease.csv'
    df = pd.read_csv(data_file)
    
    # Preprocess the text data
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Split the dataset into training and testing sets
    train_df, test_df = split_dataset(df)
    
    # Optionally, save the split datasets to CSV files for reference
    train_df.to_csv('training.csv', index=False)
    test_df.to_csv('testing.csv', index=False)
    
    # Prepare the tokenizer using the training data only
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_df['processed_text'])
    
    # Convert texts to sequences for both training and testing
    X_train = tokenizer.texts_to_sequences(train_df['processed_text'])
    X_test = tokenizer.texts_to_sequences(test_df['processed_text'])
    
    # Determine maximum sequence length and pad sequences
    max_length = max(max(len(seq) for seq in X_train), max(len(seq) for seq in X_test))
    X_train = pad_sequences(X_train, maxlen=max_length, padding='post')
    X_test = pad_sequences(X_test, maxlen=max_length, padding='post')
    
    # Encode string labels into numerical form
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df['label'])
    y_test = label_encoder.transform(test_df['label'])
    
    # Define model parameters
    vocab_size = len(tokenizer.word_index) + 1  # +1 for the reserved 0 index
    embedding_dim = 100  # You can adjust the embedding dimension
    num_classes = len(label_encoder.classes_)
    
    # Build the RNN model
    model = create_model(vocab_size, embedding_dim, max_length, num_classes)
    
    # Train the model
    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))
    
    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

if __name__ == '__main__':
    main()
