import os
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalpha() and word not in stop_words]
    return ' '.join(words)


def unpickle_model_make_prediciton(symptom: str) -> str:
    current_dir = os.getcwd()

    with open(f'{current_dir}/utils/nlp_pipeline.pkl', 'rb') as f:
        tfidf_vectorizer, model = pickle.load(f)

    processed_symptom = preprocess_text(symptom)
    symptom_tfidf_2 = tfidf_vectorizer.transform([processed_symptom])
    predicted_disease = model.predict(symptom_tfidf_2)[0]
    return predicted_disease

