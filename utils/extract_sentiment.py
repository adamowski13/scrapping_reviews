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
        
    try:
        result = analyzer(text, return_all_scores=True)
        max_result = max(result[0], key=lambda x: x['score'])
        label = label2emotion[max_result['label']]  # Mappage vers l'émotion
        prob = round(max_result['score'] * 100, 2)  # Probabilité en pourcentage
        return label, prob
    except Exception as e:
        logger.error(f"Erreur dans l'analyse avec BERT: {e}")
        return None, None
    

# Analyse des sentiments sur l'ensemble du dataframe
def analyse_all_sentiments(data_frame):
    # Initialiser des listes pour stocker les résultats d'analyse des sentiments
    sentiment_textblob = []
    sentiment_vader = []
    sentiment_label = []
    sentiment_prob = []
    
    # Boucle pour traiter chaque ligne du DataFrame
    total_reviews = len(data_frame)
    
    logger.info("Début du traitement des avis")

    for i, row in data_frame.iterrows():
        try:
            text = str(row['Content'])  # Récupérer le texte de la revue

            # Traiter le texte avec TextBlob
            textblob_sentiment = _analyze_sentiment_textblob(text)
            sentiment_textblob.append(textblob_sentiment)

            # Traiter le texte avec VADER
            vader_sentiment = _analyze_sentiment_vader(text)
            sentiment_vader.append(vader_sentiment)

            # Traiter le texte avec BERT
            sentiment_result = _analyze_sentiment_bert(text)
            sentiment_label.append(sentiment_result[0])
            sentiment_prob.append(sentiment_result[1])

            # Log de l'avancement
            logger.info(f"Traitement de l'avis {i+1}/{total_reviews}")

        except Exception as e:
            logger.error(f"Erreur lors du traitement de la revue {i+1}: {e}")
            # Continuer à traiter les autres avis même si un avis échoue
            continue

    # Ajouter les résultats au DataFrame
    data_frame['Sentiment_TextBlob'] = sentiment_textblob
    data_frame['Sentiment_VADER'] = sentiment_vader
    data_frame['sentiment_label'] = sentiment_label
    data_frame['sentiment_prob'] = sentiment_prob

    logger.info("Traitement des avis terminé")
    return data_frame


