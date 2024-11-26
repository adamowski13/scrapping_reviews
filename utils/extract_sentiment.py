import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
from transformers import logging as transformers_logging
import logging

# Configurer le logging
transformers_logging.set_verbosity_error()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Fonction d'analyse de sentiment avec TextBlob
def _analyze_sentiment_textblob(text):
    try:
        analysis = TextBlob(text)
        return analysis.sentiment.polarity
    except Exception as e:
        logger.error(f"Erreur dans l'analyse avec TextBlob: {e}")
        return None

# Fonction d'analyse de sentiment avec VADER
def _analyze_sentiment_vader(text):
    analyzer = SentimentIntensityAnalyzer()
    try:
        sentiment_score = analyzer.polarity_scores(text)
        return sentiment_score['compound']
    except Exception as e:
        logger.error(f"Erreur dans l'analyse avec VADER: {e}")
        return None

# Appliquer TF-IDF sur le contenu pour extraire les mots-clés
def extract_keywords(df, n_keywords=10):
    logger.info("Début de l'extraction des mots-clés")
    try:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=50)
        tfidf_matrix = vectorizer.fit_transform(df['Content'])

        feature_names = vectorizer.get_feature_names_out()
        docterm = pd.DataFrame(tfidf_matrix.todense(), columns=feature_names)

        sorted_indices = tfidf_matrix.sum(axis=0).argsort()[0, ::-1]
        keywords = [feature_names[i] for i in sorted_indices[:n_keywords]]

        logger.info("Extraction des mots-clés terminée avec succès")
        return pd.DataFrame(docterm)  # keywords, docterm
    except Exception as e:
        logger.error(f"Erreur dans l'extraction des mots-clés: {e}")
        return None

# Fonction d'analyse de sentiment avec BERT
def _analyze_sentiment_bert(text):
    try:
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
        label = label2emotion[max_result['label']]
        prob = round(max_result['score'] * 100, 2)
        
        return label, prob
    except Exception as e:
        logger.error(f"Erreur dans l'analyse avec BERT: {e}")
        return None, None

# Analyse des sentiments sur l'ensemble du dataframe
def analyse_all_sentiments(data_frame):
    logger.info("Début de l'analyse des sentiments sur le DataFrame")
    try:
        # Ajouter les colonnes d'analyse des sentiments
        data_frame['Sentiment_TextBlob'] = data_frame['Content'].apply(lambda text: _analyze_sentiment_textblob(text))
        data_frame['Sentiment_VADER'] = data_frame['Content'].apply(lambda text: _analyze_sentiment_vader(text))
        # data_frame[['sentiment_label', 'sentiment_prob']] = data_frame['Content'].apply(
        #    lambda x: pd.Series(_analyze_sentiment_bert(x))
        #)
        logger.info("Analyse des sentiments terminée avec succès")
        return data_frame
    except Exception as e:
        logger.error(f"Erreur dans l'analyse des sentiments : {e}")
        return None
