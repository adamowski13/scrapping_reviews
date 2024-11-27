import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
from transformers import logging as transformers_logging
import torch
from tqdm import tqdm

# Configuration du logging
transformers_logging.set_verbosity_error()

# Vérifie si un GPU est disponible sinon CPU
DEVICE = 0 if torch.cuda.is_available() else -1

# Pipeline BERT pour l'analyse de sentiment
bert_analyzer = pipeline(
    task='text-classification',
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    tokenizer="nlptown/bert-base-multilingual-uncased-sentiment",
    device=DEVICE
)

# Mappage des labels de BERT
label2emotion = {
    '1 star': "très négatif",
    '2 stars': "négatif",
    '3 stars': "neutre",
    '4 stars': "positif",
    '5 stars': "très positif"
}

# Fonction d'analyse de sentiment avec TextBlob
def analyze_sentiment_textblob(text):
    try:
        analysis = TextBlob(text)
        return analysis.sentiment.polarity
    except Exception:
        return None

# Fonction d'analyse de sentiment avec VADER
def analyze_sentiment_vader(text):
    try:
        analyzer = SentimentIntensityAnalyzer()
        sentiment_score = analyzer.polarity_scores(text)
        return sentiment_score['compound']
    except Exception:
        return None

# Fonction d'analyse de sentiment avec BERT
def analyze_sentiment_bert(text):
    try:
        result = bert_analyzer(text, truncation=True, max_length=512, return_all_scores=True)
        max_result = max(result[0], key=lambda x: x['score'])
        label = label2emotion[max_result['label']]
        prob = round(max_result['score'] * 100, 2)
        return label, prob
    except Exception:
        return None, None

# Extraction des mots-clés avec TF-IDF
def extract_keywords(df, n_keywords=10):
    try:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=n_keywords)
        tfidf_matrix = vectorizer.fit_transform(df['Content'].astype(str))
        feature_names = vectorizer.get_feature_names_out()
        return feature_names
    except Exception as e:
        print(f"Erreur d'extraction des mots-clés : {e}")
        return []

# Fonction principale d'analyse des sentiments
def analyze_sentiments(data_frame):
    tqdm.pandas(desc="Analyse des sentiments")
    
    # Ajoute les colonnes calculées
    data_frame['Sentiment_TextBlob'] = data_frame['Content'].progress_apply(analyze_sentiment_textblob)
    data_frame['Sentiment_VADER'] = data_frame['Content'].progress_apply(analyze_sentiment_vader)
    bert_results = data_frame['Content'].progress_apply(analyze_sentiment_bert)
    
    data_frame['Sentiment_BERT_Label'] = bert_results.apply(lambda x: x[0])
    data_frame['Sentiment_BERT_Prob'] = bert_results.apply(lambda x: x[1])
    
    return data_frame
