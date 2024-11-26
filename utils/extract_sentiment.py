import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
import logging

# Configurer le logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Fonction d'analyse de sentiment avec TextBlob
def _analyze_sentiment_textblob(text):  # Fonction privée (usage interne)
    try:
        # Utilisation de TextBlob pour obtenir la polarité (positif, négatif, neutre)
        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity
        return polarity
    except Exception as e:
        logging.error(f"Erreur dans l'analyse avec TextBlob: {e}")
        return None

# Fonction d'analyse de sentiment avec VADER
def _analyze_sentiment_vader(text):  # Fonction privée (usage interne)
    analyzer = SentimentIntensityAnalyzer()
    try:
        # Analyser le texte et renvoyer le score de sentiment
        sentiment_score = analyzer.polarity_scores(text)
        score = sentiment_score['compound']  # Renvoie un score global de sentiment
        return score  # Score global
    except Exception as e:
        logging.error(f"Erreur dans l'analyse avec VADER: {e}")
        return None

# Appliquer TF-IDF sur le contenu pour extraire les mots-clés
def extract_keywords(df, n_keywords=10):
    try:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=50)
        tfidf_matrix = vectorizer.fit_transform(df['Content'])

        # Récupérer les mots-clés en fonction de leur score TF-IDF
        feature_names = vectorizer.get_feature_names_out()
        docterm = pd.DataFrame(tfidf_matrix.todense(), columns=feature_names)

        sorted_indices = tfidf_matrix.sum(axis=0).argsort()[0, ::-1]  # Tri des indices des mots-clés
        keywords = [feature_names[i] for i in sorted_indices[:n_keywords]]
        
        return pd.DataFrame(docterm)  # keywords, docterm
    except Exception as e:
        logging.error(f"Erreur dans l'extraction des mots-clés: {e}")
        return None

# Fonction d'analyse de sentiment avec BERT
def _analyze_sentiment_bert(text):  # Fonction privée (usage interne)
    try:
        # Initialiser le pipeline de sentiment avec le modèle BERT
        analyzer = pipeline(
            task='text-classification',
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            tokenizer="nlptown/bert-base-multilingual-uncased-sentiment"
        )
        
        label2emotion = {
            '1 star': "très négatif",
            '2 stars': "négatif",
            '3 stars': "neutre",
            '4 stars': "positif",
            '5 stars': "très positif"
        }
        
        result = analyzer(text, return_all_scores=True)
        max_result = max(result[0], key=lambda x: x['score'])
        label = label2emotion[max_result['label']]  # Map the label to emotion
        prob = round(max_result['score'] * 100, 2)  # Convert probability to percentage
        
        return label, prob
    except Exception as e:
        logging.error(f"Erreur dans l'analyse avec BERT: {e}")
        return None, None

# Analyse des sentiments sur l'ensemble du dataframe
def analyse_all_sentiments(data_frame):
    try:
        total_lines = len(data_frame)
        lines_processed = 0

        # Utilisation des fonctions privées pour l'analyse des sentiments
        data_frame['Sentiment_TextBlob'] = data_frame['Content'].apply(lambda text: _analyze_sentiment_textblob(text))
        data_frame['Sentiment_VADER'] = data_frame['Content'].apply(lambda text: _analyze_sentiment_vader(text))

        # Utilisation de la fonction BERT pour l'analyse de sentiment avec .apply()
        data_frame[['sentiment_label', 'sentiment_prob']] = data_frame['Content'].apply(
            lambda x: pd.Series(_analyze_sentiment_bert(x))
        )

        # Incrémenter le nombre de lignes traitées et afficher la progression
        lines_processed = len(data_frame)
        logging.info(f"Nombre de lignes traitées : {lines_processed}/{total_lines}")
        
        return data_frame
    except Exception as e:
        logging.error(f"Erreur dans l'analyse des sentiments pour le dataframe: {e}")
        return None
