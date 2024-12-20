import os
# Désactiver réduit les performances mais permet d'obtenir des résultats plus précis
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
from transformers import logging as transformers_logging
import torch
from tqdm import tqdm
import tensorflow as tf

# Configuration du logging
transformers_logging.set_verbosity_error()
tf.get_logger().setLevel('ERROR')

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
    
def categorization_label(col):
    if col<=(-0.6) and col>=(-1):
        return "trés négatif"
    if col<(-0.2) and col>(-0.6):
        return "négatif"
    if col<=(0.2) and col>=(-0.2):
        return "neutre"
    if col<(0.6) and col>(0.2):
        return "positif"
    if col<=(1) and col>=(0.6):
        return "très positif"
    else:
        return "Null"

# Extraction des mots-clés avec TF-IDF
def extract_keywords(df, n_keywords=10):
    try:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=n_keywords)
        tfidf_matrix = vectorizer.fit_transform(df['Content'].astype(str))
        feature_names = vectorizer.get_feature_names_out()
        docterm = pd.DataFrame(tfidf_matrix.todense(), columns=feature_names)
        return docterm
    except Exception as e:
        print(f"Erreur d'extraction des mots-clés : {e}")
        return []

# Fonction principale d'analyse des sentiments
def analyze_sentiments(data_frame):
    
    tqdm.pandas(desc="Analyse des sentiments TextBlob")
    # Ajoute les colonnes calculées
    data_frame['Sentiment_TextBlob'] = data_frame['Content'].progress_apply(analyze_sentiment_textblob)
    data_frame['Sentiment_TextBlob_label']=data_frame['Sentiment_TextBlob'].apply(categorization_label)

    tqdm.pandas(desc="Analyse des sentiments VADER")
    data_frame['Sentiment_VADER'] = data_frame['Content'].progress_apply(analyze_sentiment_vader)
    data_frame['Sentiment_VADER_label']=data_frame['Sentiment_VADER'].apply(categorization_label)

    tqdm.pandas(desc="Analyse des sentiments BERT")
    bert_results = data_frame['Content'].progress_apply(analyze_sentiment_bert)
    # ajoutt des données dans le data frame
    data_frame['Sentiment_BERT_Label'] = bert_results.apply(lambda x: x[0])
    data_frame['Sentiment_BERT_Prob'] = bert_results.apply(lambda x: x[1])
    
    return data_frame


    
