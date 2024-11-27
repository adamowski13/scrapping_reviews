import pandas as pd
import re
from datetime import datetime
import logging

# Configuration du logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Charger le fichier CSV
def load_csv(file_path):
    """
    Charge un fichier CSV en utilisant pandas.
    :param file_path: Chemin du fichier CSV
    :return: DataFrame contenant les données ou None en cas d'erreur
    """
    try:
        return pd.read_csv(file_path, delimiter=',')
    except Exception as e:
        logger.error(f"Erreur lors du chargement du fichier : {e}")
        return None

# Nettoyer le texte en supprimant les caractères spéciaux
def clean_text(text):
    """
    Nettoie le texte en supprimant les caractères spéciaux et les espaces multiples.
    :param text: Texte à nettoyer
    :return: Texte nettoyé
    """
    if pd.isnull(text):
        return ""
    text = re.sub(r"[^a-zA-Z0-9À-ÿ' ]", " ", text)  # Supprimer les caractères spéciaux
    text = re.sub(r"\s+", " ", text).strip()        # Supprimer les espaces multiples
    if len(text) > 512:
        text = text[:510]  # Truncate to fit model input size
    if len(text) < 2:
        text += "N/A"  # Handle very short or empty reviews
    return text

# Formater la date au format YYYY-MM-DD
def format_date(date_str):
    """
    Formate la date à partir du format ISO 8601 (YYYY-MM-DDTHH:MM:SS) en format YYYY-MM-DD.
    :param date_str: Date sous forme de chaîne à formater
    :return: Date formatée en chaîne ou None en cas d'échec
    """
    try:
        return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%fZ").strftime("%Y-%m-%d")
    except Exception:
        return None

# Nettoyer et structurer les données
def clean_and_structure_data(file_path):
    """
    Charge un fichier CSV, nettoie et structure les données.
    :param file_path: Chemin du fichier CSV à charger
    :return: DataFrame nettoyé ou None en cas d'échec
    """
    logger.info("Début du nettoyage des données")
    
    df = load_csv(file_path)
    if df is None:
        logger.error("Échec du nettoyage : impossible de charger le fichier CSV")
        return None

    required_columns = ["Username", "Title", "Content", "Rating", "Date"]
    for col in required_columns:
        if col not in df.columns:
            logger.error(f"Échec du nettoyage : colonne manquante '{col}' dans les données")
            return None

    try:
        # Nettoyage des colonnes
        df["Username"] = df["Username"].apply(clean_text)
        df["Title"] = df["Title"].apply(clean_text)
        df["Content"] = df["Content"].apply(clean_text)
        df["Date"] = df["Date"].apply(format_date)
        df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
        df = df.dropna(subset=["Rating", "Date"])  # Supprimer les lignes avec notes ou dates manquantes
        
        logger.info("Nettoyage des données terminé avec succès")
        return df
    except Exception as e:
        logger.error(f"Échec du nettoyage des données : {e}")
        return None
