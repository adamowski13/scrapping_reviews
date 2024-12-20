import os
import pandas as pd
import logging
from utils.clean_data import clean_and_structure_data
from utils.extract_sentiment import extract_keywords, analyze_sentiments
from utils.scrape_data import scrape_trustpilot_reviews
from datetime import datetime

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fonction pour sauvegarder les DataFrames
def save_dataframe(df, file_name):
    try:
        file_path = os.path.join('.\\data', file_name)
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
        logger.info(f"Fichier sauvegardé : {file_path}")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde du fichier {file_name}: {e}")

# Fonction principale de traitement
def process_data():
    try:
      
        # Scraper les données
        scrape_data = scrape_trustpilot_reviews()
        if scrape_data.empty:
            logger.warning("Aucune donnée récupérée lors du scraping.")
            return
        save_dataframe(scrape_data, f'scrape_data.csv')
 
        # Nettoyage des données
        clean_data = clean_and_structure_data(file_path='.\\data\\scrape_data.csv')
        if clean_data.empty:
            logger.warning("Aucune donnée nettoyée.")
            return
        
        df_model= clean_data.iloc[:-50]
        df_test=clean_data.iloc[-50:]
        save_dataframe(df_model, f'clean_data.csv')
        df_test.to_csv(".\\data\\echantillon_validation_test.csv", index=False, encoding='utf-8-sig')


        # Extraction des mots-clés
        keywords_data = extract_keywords(df=clean_data)
        save_dataframe(keywords_data, f'keywords_data.csv')

        # Analyse des sentiments
        sentiments_analyze = analyze_sentiments(data_frame=clean_data)
        save_dataframe(sentiments_analyze, f'sentiments_analyze.csv')

        logger.info('Processus terminé avec succès.')
 
    except Exception as e:
        logger.error(f"Erreur dans le processus de traitement des données : {e}")

if __name__ == "__main__":
    process_data()
