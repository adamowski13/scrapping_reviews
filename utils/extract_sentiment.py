from utils.clean_data import clean_and_structure_data
import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer

# df = pd.read_csv("../Tesla_Trustpilot_Reviews.csv", delimiter=',')
df = clean_and_structure_data("Tesla_Trustpilot_Reviews.csv", "output")

# Fonction d'analyse de sentiment avec TextBlob
def analyze_sentiment_textblob(text):
    try:
        # Utilisation de TextBlob pour obtenir la polarité (positif, négatif, neutre)
        analysis = TextBlob(text)
        return analysis.sentiment.polarity
    except Exception as e:
        print(f"Erreur dans l'analyse avec TextBlob: {e}")
        return None

# Fonction d'analyse de sentiment avec VADER
def analyze_sentiment_vader(text):
    analyzer = SentimentIntensityAnalyzer()
    try:
        # Analyser le texte et renvoyer le score de sentiment
        sentiment_score = analyzer.polarity_scores(text)
        return sentiment_score['compound']  # Renvoie un score global de sentiment
    except Exception as e:
        print(f"Erreur dans l'analyse avec VADER: {e}")
        return None

# Appliquer TF-IDF sur le contenu pour extraire les mots-clés
def extract_keywords(df, n_keywords=10):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=50)
    tfidf_matrix = vectorizer.fit_transform(df['Content'])

    # Récupérer les mots-clés en fonction de leur score TF-IDF
    feature_names = vectorizer.get_feature_names_out()

    docterm = pd.DataFrame(tfidf_matrix.todense(), columns=feature_names)

    sorted_indices = tfidf_matrix.sum(axis=0).argsort()[0, ::-1]  # Tri des indices des mots-clés
    # Récupérer les meilleurs mots-clés
    keywords = [feature_names[i] for i in sorted_indices[:n_keywords]]
    return pd.DataFrame(docterm) # keywords, docterm

def analyse_all_sentiments(data_frame):

    data_frame['Sentiment_TextBlob'] = data_frame['Content'].apply(analyze_sentiment_textblob)
    data_frame['Sentiment_VADER'] = data_frame['Content'].apply(analyze_sentiment_vader)

    return data_frame
