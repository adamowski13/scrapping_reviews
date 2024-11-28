import pandas as pd
import pytest

@pytest.fixture
def load_sentiments_data():
    """Charger le fichier sentiments_analyze.csv pour les tests."""
    return pd.read_csv("data/sentiments_analyze.csv")

@pytest.fixture
def load_reviews_data():
    """Charger le fichier Tesla_Trustpilot_Reviews.csv pour les tests."""
    return pd.read_csv("data/Tesla_Trustpilot_Reviews.csv")

# Tests pour sentiments_analyze.csv
def test_sentiment_labels_validity(load_sentiments_data):
    valid_labels = ["positif", "négatif", "très positif", "très négatif", "neutre"]
    assert all(load_sentiments_data["Sentiment_BERT_Label"].isin(valid_labels)), "Labels invalides détectés."

def test_sentiment_prob_range(load_sentiments_data):
    assert load_sentiments_data["Sentiment_BERT_Prob"].between(0, 100).all(), "Probabilités hors plage (0-100)."

def test_sentiment_scores_vs_labels(load_sentiments_data):
    data = load_sentiments_data
    assert (data.loc[data["Sentiment_TextBlob"] > 0, "Sentiment_TextBlob_label"] == "positif").all(), \
        "Incohérence entre score TextBlob et label."

# Tests pour Tesla_Trustpilot_Reviews.csv
def test_reviews_columns_presence(load_reviews_data):
    expected_columns = ["Username", "Title", "Content", "Rating", "Date"]
    assert all(col in load_reviews_data.columns for col in expected_columns), "Colonnes manquantes dans Reviews."

def test_reviews_rating_range(load_reviews_data):
    assert load_reviews_data["Rating"].between(1, 5).all(), "Notes hors plage (1-5)."

def test_reviews_date_format(load_reviews_data):
    try:
        pd.to_datetime(load_reviews_data["Date"])
    except Exception as e:
        pytest.fail(f"Erreur de format des dates : {e}")
