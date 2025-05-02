import os
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import torch
import pickle
from pathlib import Path


# MODEL_PATH = Path("/Users/danielfishbein/Documents/python-ml-final/ml-models/rnn/rnn_lstm_full.pt")
# TOKENIZER_PATH = Path("/Users/danielfishbein/Documents/python-ml-final/ml-models/rnn/tokenizer_pytorch.pkl")
# ENCODER_PATH = Path("/Users/danielfishbein/Documents/python-ml-final/ml-models/rnn/label_encoder_pytorch.pkl")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalpha() and word not in stop_words]
    return ' '.join(words)


def unpickle_model_make_prediciton(symptom: str) -> str:
    current_dir = os.getcwd()

    with open(f'{current_dir}/utils/dans_nlp_pipeline.pkl', 'rb') as f:
        tfidf_vectorizer, rfc, knn = pickle.load(f)

    processed_symptom = preprocess_text(symptom)
    symptom_tfidf = tfidf_vectorizer.transform([processed_symptom])

    rfc_prediction = rfc.predict(symptom_tfidf)[0]
    knn_prediction = knn.predict(symptom_tfidf)[0]

    rfc_probabilities = rfc.predict_proba(symptom_tfidf)[0]

    predicted_class_index_rfc = rfc.classes_.tolist().index(rfc_prediction)
    prediction_confidence_rfc = rfc_probabilities[predicted_class_index_rfc]

    knn_probabilities = knn.predict_proba(symptom_tfidf)[0]

    predicted_class_index_knn = knn.classes_.tolist().index(knn_prediction)
    prediction_confidence_knn = knn_probabilities[predicted_class_index_knn]

    return {
        "randomForestPrediction": rfc_prediction,
        "randomForestConfidence": prediction_confidence_rfc,
        "knnPrediction": knn_prediction,
        "knnConfidence": prediction_confidence_knn
    }


def unpickle_jack_model_prediction(symptom: str) -> str:
    current_dir = os.getcwd()

    with open(f'{current_dir}/utils/jack_nlp_pipeline.pkl', 'rb') as f:
        vectorizer, myModel = pickle.load(f)
    
    processed_symptom = preprocess_text(symptom)
    symptom_tfifd = vectorizer.transform([processed_symptom])

    predicted_disease = myModel.predict(symptom_tfifd)[0]

    model_probability = myModel.predict_proba(symptom_tfifd)[0]

    prediction_class_index = myModel.classes_.tolist().index(predicted_disease)
    prediction_confidence = model_probability[prediction_class_index]

    # print(prediction_confidence)

    return {
        "modelClass": predicted_disease,
        "modelConfidence": prediction_confidence
    }


# class BiLSTMClassifier(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.embedding  = torch.nn.Embedding(1283, 64, padding_idx=0)
#         self.lstm1      = torch.nn.LSTM(64, 128, batch_first=True, bidirectional=True)
#         self.dropout1   = torch.nn.Dropout(0.4)
#         self.lstm2      = torch.nn.LSTM(256, 64, batch_first=True, bidirectional=True)
#         self.fc         = torch.nn.Linear(128, 64)
#         self.dropout2   = torch.nn.Dropout(0.4)
#         self.out        = torch.nn.Linear(64, 24)

#     def forward(self, x, lengths=None):
#         x = self.embedding(x)
#         if lengths is not None:
#             x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
#         _, (h_n1, _) = self.lstm1(x)
#         h = torch.cat((h_n1[-2], h_n1[-1]), dim=1)
#         h = self.dropout1(h)
#         h = h.unsqueeze(1)
#         _, (h_n2, _) = self.lstm2(h)
#         h2 = torch.cat((h_n2[-2], h_n2[-1]), dim=1)
#         z = torch.relu(self.fc(h2))
#         z = self.dropout2(z)
#         return self.out(z)


# model = BiLSTMClassifier().to(device)
# model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
# model.eval()


# with open(TOKENIZER_PATH, "rb") as f:
#     tokenizer = pickle.load(f)


# with open(ENCODER_PATH, "rb") as f:
#     label_encoder = pickle.load(f)


# def preprocess_input(text: str, max_len=10):
#     tokens = text.lower().split()
#     token_ids = [tokenizer.get(token, 0) for token in tokens]
#     token_ids = token_ids[:max_len] + [0] * (max_len - len(token_ids))

#     if all(t == 0 for t in token_ids):
#         raise ValueError("All tokens were unknown to the tokenizer.")

#     tensor = torch.LongTensor([token_ids])
#     lengths = (tensor != 0).sum(1)
#     return tensor.to(device), lengths.to(device)


# def unpickle_rnn_model_prediction(symptom: str) -> dict:
#     try:
#         input_tensor, lengths = preprocess_input(symptom)
#         with torch.no_grad():
#             logits = model(input_tensor, lengths)
#             probs = torch.softmax(logits, dim=1)
#             pred_idx = torch.argmax(probs, dim=1).item()
#             confidence = probs[0][pred_idx].item()
#             return {
#                 "modelClass": label_encoder.inverse_transform([pred_idx])[0],
#                 "modelConfidence": confidence
#             }
#     except ValueError as e:
#         return {"error": str(e)}
