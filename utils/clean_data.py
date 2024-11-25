import pandas as pd
import re
from datetime import datetime

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
        print(f"Erreur lors du chargement du fichier : {e}")
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
def clean_and_structure_data(file_path, output_path):
    """
    Charge un fichier CSV, nettoie et structure les données, puis sauvegarde le fichier nettoyé.
    
    :param file_path: Chemin du fichier CSV à charger
    :param output_path: Chemin où sauvegarder le fichier nettoyé
    """
    df = load_csv(file_path)
    if df is None:
        return

    required_columns = ["Username", "Title", "Content", "Rating", "Date"]
    for col in required_columns:
        if col not in df.columns:
            print(f"Colonne manquante : {col}")
            return

    df["Username"] = df["Username"].apply(clean_text)
    df["Title"] = df["Title"].apply(clean_text)
    df["Content"] = df["Content"].apply(clean_text)
    df["Date"] = df["Date"].apply(format_date)
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")  # Convertir en numérique
    df = df.dropna(subset=["Rating", "Date"])  # Supprimer les lignes avec notes ou dates manquantes

    # Sauvegarder les données nettoyées dans un nouveau fichier CSV
    try:
        df.to_csv(output_path, index=False)
        print(f"Fichier nettoyé et structuré sauvegardé sous : {output_path}\n")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde : {e}")
