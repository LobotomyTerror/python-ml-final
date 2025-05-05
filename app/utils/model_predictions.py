import os
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle


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
