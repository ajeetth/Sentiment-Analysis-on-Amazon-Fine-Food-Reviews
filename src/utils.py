import re
import contractions
import spacy
import pickle
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

nlp = spacy.load("en_core_web_sm")

def clean_reviews(raw_text):
    """
    Cleans raw review text by removing HTML tags, converting to lowercase, 
    expanding contractions, removing URLs, keeping only alphabetic characters, 
    and lemmatizing tokens. Returns the cleaned and lemmatized text.
    Args:
        raw_text (str): The raw text of the review.
    Returns:
        str: The cleaned and lemmatized text.
    """
    text = BeautifulSoup(raw_text, 'html.parser').get_text()
    text = text.lower()
    text = contractions.fix(text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    doc = nlp(text)
    clean_tokens = [token.lemma_ for token in doc]
    return ' '.join(clean_tokens) 

def load_models():
    """
    Loads the SentenceTransformer and pre-trained LinearSVC models.
    Returns:
        tuple: (SentenceTransformer model, LinearSVC model)
    """
    embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    with open("Linear_svc_model.pkl", "rb") as file:
        sentiment_model = pickle.load(file)

    return embedding_model, sentiment_model

def predict_sentiment(cleaned_text, embedder, sentiment_model):
    """
    Predicts the sentiment of the cleaned review text using the SentenceTransformer 
    and LinearSVC models.
    Args:
        cleaned_text (str): The cleaned and lemmatized review text.
        embedder: SentenceTransformer model.
        sentiment_model: LinearSVC model.
    Returns:
        str: The predicted sentiment label ('Positive' or 'Negative').
    """
    embedding = embedder.encode([cleaned_text])
    prediction = sentiment_model.predict(embedding)
    sentiment =  "Positive" if prediction[0] == 1 else "Negative"
    return sentiment

# Ajith Devadiga