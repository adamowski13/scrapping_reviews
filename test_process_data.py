import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath("process_data.py")))

# Charger les données pour les tests
sentiments_analyze_path = "data/sentiments_analyze.csv"
tesla_reviews_path = "data/Tesla_Trustpilot_Reviews.csv"
sentiments_analyze_data = pd.read_csv(sentiments_analyze_path)
tesla_reviews_data = pd.read_csv(tesla_reviews_path)

# Mock des fonctions dans process_data.py
from process_data import save_dataframe

# Fonction pour tester save_dataframe
def test_save_dataframe():
    # Mock des données
    mock_df = pd.DataFrame({
        "Username": ["User1", "User2"],
        "Rating": [5, 4]
    })

    # Mock du chemin et du logger
    with patch("process_data.os.path.join", return_value="mock_path.csv") as mock_path,\
         patch("process_data.logger") as mock_logger:
        
        # Appeler la fonction
        save_dataframe(mock_df, "mock_file.csv")
        
        # Vérifier que le chemin est construit correctement
        mock_path.assert_called_once_with('.\\data', "mock_file.csv")
        
        # Vérifier que le logger info est appelé
        mock_logger.info.assert_called_once_with("Fichier sauvegardé : mock_path.csv")

test_save_dataframe()
