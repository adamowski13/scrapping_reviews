import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
import plotly.io as pio

pio.json.config.default_engine = "json"


# Charger les données
file_path = "./data/sentiments_analyze.csv"  # Mettez à jour ce chemin si nécessaire
data = pd.read_csv(file_path)

# Préparer les données
data['Date'] = pd.to_datetime(data['Date'])
data['Month_Year'] = data['Date'].dt.to_period('M')

# Configuration de la page
st.set_page_config(
    page_title="Dashboard des Avis Tesla",
    page_icon="🚗",
    layout="wide"
)

# Titre et description
st.markdown("<h1 style='text-align: center;'>🚗 Dashboard des Avis Tesla</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Analyse des avis, notes et sentiments des utilisateurs</h3>", unsafe_allow_html=True)

# Filtres dans la barre latérale
st.sidebar.title("Filtres")

start_date, end_date = st.sidebar.date_input(
    "Filtrer par plage de dates",
    [data['Date'].min().date(), data['Date'].max().date()]
)

selected_sentiment = st.sidebar.multiselect(
    "Filtrer par sentiment",
    options=data['Sentiment_BERT_Label'].unique(),
    default=data['Sentiment_BERT_Label'].unique()
)

selected_rating = st.sidebar.slider("Filtrer par notes", 1, 5, (1, 5))

# Appliquer les filtres
filtered_data = data[
    (data['Date'] >= pd.Timestamp(start_date)) &
    (data['Date'] <= pd.Timestamp(end_date)) &
    (data['Sentiment_BERT_Label'].isin(selected_sentiment)) &
    (data['Rating'].between(*selected_rating))
]

# Section des KPI
st.markdown("## 🧮 Indicateurs Clés de Performance")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Total des Avis", value=len(filtered_data))
with col2:
    avg_rating = round(filtered_data['Rating'].mean(), 2) if not filtered_data.empty else "N/A"
    st.metric(label="Note Moyenne", value=avg_rating)
with col3:
    common_sentiment = filtered_data['Sentiment_BERT_Label'].mode()[0] if not filtered_data.empty else "N/A"
    st.metric(label="Sentiment Dominant", value=common_sentiment)

# Visualisation 1 : Répartition des notes
st.markdown("### 📊 Répartition des Notes")
fig = px.histogram(
    filtered_data,
    x="Rating",
    nbins=5,
    title="Répartition des Notes",
    labels={"Rating": "Notes", "count": "Nombre d'Avis"},
    color_discrete_sequence=["#636EFA"]
)
st.plotly_chart(fig, use_container_width=True)
if not filtered_data.empty:
    highest_rating = filtered_data['Rating'].value_counts().idxmax()
    st.write(f"Les utilisateurs donnent principalement une note de {highest_rating}, ce qui montre une tendance générale sur la perception du service ou produit Tesla.")
else:
    st.write("Aucune donnée disponible pour cette plage de filtres.")

# Visualisation 2 : Distribution des sentiments
st.markdown("### 🥧 Distribution des Sentiments")
fig = px.pie(
    filtered_data,
    names="Sentiment_BERT_Label",
    title="Distribution des Sentiments",
    color_discrete_sequence=px.colors.qualitative.Set3
)
st.plotly_chart(fig, use_container_width=True)
if not filtered_data.empty:
    dominant_sentiment = filtered_data['Sentiment_BERT_Label'].mode()[0]
    sentiment_proportion = round(filtered_data['Sentiment_BERT_Label'].value_counts(normalize=True).max() * 100, 2)
    st.write(f"Le sentiment dominant est **{dominant_sentiment}**, représentant environ {sentiment_proportion}% des avis.")
else:
    st.write("Aucune donnée disponible pour cette plage de filtres.")

# Visualisation 3 : Comparaison des modèles de sentiment
st.markdown("### 📊 Comparaison des Modèles de Sentiment")
model_comparison = filtered_data[['Sentiment_TextBlob', 'Sentiment_VADER', 'Sentiment_BERT_Prob']].describe()
st.write("**Résumé statistique des modèles :**")
st.dataframe(model_comparison)

# Visualisation des sentiments empilés pour chaque modèle
model_data = pd.DataFrame({
    "Modèle": ["BERT", "TextBlob", "VADER"],
    "Positif": [
        (filtered_data['Sentiment_TextBlob'] > 0).sum(),
        (filtered_data['Sentiment_VADER'] > 0).sum(),
        (filtered_data['Sentiment_BERT_Label'] == 'positif').sum()
    ],
    "Neutre": [
        (filtered_data['Sentiment_TextBlob'] == 0).sum(),
        (filtered_data['Sentiment_VADER'] == 0).sum(),
        0  # BERT ne fournit pas de label neutre explicitement
    ],
    "Négatif": [
        (filtered_data['Sentiment_TextBlob'] < 0).sum(),
        (filtered_data['Sentiment_VADER'] < 0).sum(),
        (filtered_data['Sentiment_BERT_Label'] == 'négatif').sum()
    ]
})

fig = px.bar(
    model_data.melt(id_vars="Modèle", var_name="Sentiment", value_name="Nombre"),
    x="Modèle",
    y="Nombre",
    color="Sentiment",
    barmode="stack",
    title="Répartition des Sentiments par Modèle",
    labels={"Nombre": "Nombre d'Avis"}
)
st.plotly_chart(fig, use_container_width=True)
st.write("""
**Analyse :** Cette visualisation montre comment les différents modèles évaluent les sentiments des avis. 
- **BERT** semble plus précis pour détecter des sentiments très positifs ou très négatifs grâce à son approche fine-tunée.
- **TextBlob** et **VADER** offrent une perspective plus simple, mais parfois moins nuancée.
""")

# Visualisation 4 : Nuage de mots
st.markdown("### ☁️ Nuage de Mots des Avis")
if not filtered_data['Content'].dropna().empty:
    wordcloud = WordCloud(
        width=800, height=400, background_color="white", colormap="viridis"
    ).generate(" ".join(filtered_data['Content'].dropna()))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)
    st.write("Le nuage de mots met en évidence les termes les plus fréquemment utilisés dans les avis, permettant d'identifier les points récurrents comme les forces ou les problèmes potentiels.")
else:
    st.write("Aucun contenu disponible pour générer un nuage de mots.")

# Table des données filtrées
st.markdown("### 📋 Table des Données Filtrées")
st.dataframe(filtered_data)

# Footer
st.markdown("<footer style='text-align: center; margin-top: 50px;'>🚀 Créé avec Streamlit | Propulsé par HTML, CSS, et Plotly</footer>", unsafe_allow_html=True)
