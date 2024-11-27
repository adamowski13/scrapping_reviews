# Utiliser une image de base Python
FROM python:3.9-slim

# Définir le répertoire de travail dans le container
WORKDIR /app

# Copier les fichiers du projet dans le container
COPY . /app

# Installer les dépendances requises
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port utilisé par Streamlit (par défaut 8501)
EXPOSE 8501

# Commande pour démarrer l'application Streamlit
CMD ["streamlit", "run", "main.py"]
