import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import logging

# Configurer les logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def scrape_trustpilot_reviews(num_pages=50):
    base_url = "https://www.trustpilot.com/review/www.teslamotors.com?page="
    
    # Initialisation des listes pour stocker les données extraites
    review_titles = []
    review_contents = []
    ratings = []
    dates = []
    usernames = []

    # Boucle pour scraper plusieurs pages
    for page in range(1, num_pages + 1):  
        url = base_url + str(page)
        logger.info(f"Scraping page {page}...")
        
        # Faire une requête HTTP pour récupérer le contenu de la page
        try:
            response = requests.get(url)
            response.raise_for_status()  # Vérifier si la requête a échoué (status code 4xx ou 5xx)
            soup = BeautifulSoup(response.content, 'html.parser')
        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur lors de la récupération de la page {page}: {e}")
            continue
        
        # Extraire les avis de la page
        reviews = soup.find_all('article', class_='paper_paper__1PY90')
        
        if not reviews:
            logger.warning(f"Aucun avis trouvé sur la page {page}.")
        
        for review in reviews:
            title = review.find('h2', class_='typography_heading-s__f7029').text.strip() if review.find('h2', class_='typography_heading-s__f7029') else None
            content = review.find('p', class_='typography_body-l__KUYFJ').text.strip() if review.find('p', class_='typography_body-l__KUYFJ') else None
            rating = review.find('div', {'data-service-review-rating': True})['data-service-review-rating'] if review.find('div', {'data-service-review-rating': True}) else None
            date = review.find('time')['datetime'] if review.find('time') else None
            username = review.find('span', class_='typography_heading-xxs__QKBS8').text.strip() if review.find('span', class_='typography_heading-xxs__QKBS8') else None
            
            review_titles.append(title)
            review_contents.append(content)
            ratings.append(rating)
            dates.append(date)
            usernames.append(username)
        
        # Respecter une pause entre les requêtes pour éviter d'être bloqué par le site
        time.sleep(2)

    # Créer un DataFrame à partir des données collectées
    trustpilot_data = pd.DataFrame({
        'Username': usernames,
        'Title': review_titles,
        'Content': review_contents,
        'Rating': ratings,
        'Date': dates
    })
    
    logger.info(f"Scraping terminé. {len(trustpilot_data)} avis extraits.")
    
    # Retourner le DataFrame
    return trustpilot_data


