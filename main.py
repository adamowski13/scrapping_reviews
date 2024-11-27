import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px

# Charger les données
file_path = "./data/sentiments_analyze.csv"  # Mettez à jour ce chemin si nécessair
keywords_file_path = "./data/keywords_data.csv"  # Mettez à jour ce chemin si nécessaire

data = pd.read_csv(file_path)
keywords_data = pd.read_csv(keywords_file_path)

# Préparer les données des sentiments
data['Date'] = pd.to_datetime(data['Date'])
data['Month_Year'] = data['Date'].dt.to_period('M')

# Configuration de la page
st.set_page_config(
    page_title="Dashboard des Avis Tesla",
    page_icon="🚗",
    layout="wide"
)

# Titre principal
st.markdown("<h1 style='text-align: center;'>🚗 Dashboard des Avis Tesla</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Explorez les avis, sentiments et mots-clés des utilisateurs</h3>", unsafe_allow_html=True)

# Section des KPI
st.markdown("## 🧮 Indicateurs Clés de Performance")
kpi_col1, kpi_col2, kpi_col3 = st.columns(3)

with kpi_col1:
    total_reviews = len(data)
    st.metric(label="Total des Avis", value=total_reviews)

with kpi_col2:
    avg_rating = round(data['Rating'].mean(), 2)
    st.metric(label="Note Moyenne", value=avg_rating)

with kpi_col3:
    common_sentiment = data['Sentiment_BERT_Label'].mode()[0]
    st.metric(label="Sentiment Dominant", value=common_sentiment)

# Filtres
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

# Section : Visualisations principales
st.markdown("## 📊 Visualisations des Avis et Sentiments")

# Visualisation 1 : Répartition des notes
st.markdown("### Répartition des Notes")
fig = px.histogram(
    filtered_data,
    x="Rating",
    nbins=5,
    title="Répartition des Notes",
    labels={"Rating": "Notes", "count": "Nombre d'Avis"},
    color_discrete_sequence=["#636EFA"]
)
st.plotly_chart(fig, use_container_width=True)
st.write("""
**Observation :** La répartition des notes montre que la majorité des utilisateurs attribuent une note très negative indiquant un avis général sur la marque au plus bas. Les notes au dessus de 1 sont rares, ce qui reflète une mauvaise image globale de la marque Tesla dans l'opinion des utilisateurs de Trustpilot.
""")

# Visualisation 2 : Distribution des sentiments
st.markdown("### Distribution des Sentiments")
fig = px.pie(
    filtered_data,
    names="Sentiment_BERT_Label",
    title="Distribution des Sentiments",
    color_discrete_sequence=px.colors.qualitative.Set3
)
st.plotly_chart(fig, use_container_width=True)
st.write("""
**Observation :** Les sentiments négatifs dominent les avis, tandis que les sentiments positifs restent marginaux. Cela confirme que Tesla ne satisfait généralement que très rarement ses clients.
""")

# Visualisation 3 : Évolution des notes dans le temps
st.markdown("### Évolution des Notes dans le Temps")
ratings_over_time = filtered_data.groupby('Month_Year')['Rating'].mean()
fig = px.line(
    x=ratings_over_time.index.astype(str),
    y=ratings_over_time.values,
    title="Évolution des Notes Moyennes dans le Temps",
    labels={"x": "Mois-Année", "y": "Note Moyenne"},
    markers=True
)
st.plotly_chart(fig, use_container_width=True)
st.write("""
**Observation :** Les notes moyennes montrent une stabilité globale au fil du temps, et ce sur plusieurs années a la suite, avec néanmoins quelques pics a la fin des années 2010. Ces fluctuations ponctuelles peuvent refléter des événements spécifiques comme des lancements de produits ou des problèmes de service.
""")

# Comparaison des modèles de sentiment
st.markdown("## 🧠 Comparaison des Modèles de Sentiment")

model_comparison = pd.DataFrame({
    "Modèle": ["BERT", "TextBlob", "VADER"],
    "Positif": [
        (filtered_data['Sentiment_BERT_Label'] == 'positif').sum(),
        (filtered_data['Sentiment_TextBlob'] > 0).sum(),
        (filtered_data['Sentiment_VADER'] > 0).sum()
    ],
    "Négatif": [
        (filtered_data['Sentiment_BERT_Label'] == 'négatif').sum(),
        (filtered_data['Sentiment_TextBlob'] < 0).sum(),
        (filtered_data['Sentiment_VADER'] < 0).sum()
    ],
    "Très positif": [
        (filtered_data['Sentiment_BERT_Label'] == 'très positif').sum(),
        0,  # TextBlob et VADER n'ont pas de catégorie "très positif"
        0
    ],
    "Très négatif": [
        (filtered_data['Sentiment_BERT_Label'] == 'très négatif').sum(),
        0,  # TextBlob et VADER n'ont pas de catégorie "très négatif"
        0
    ]
})

model_comparison_melted = model_comparison.melt(
    id_vars="Modèle", var_name="Sentiment", value_name="Nombre"
)

fig = px.bar(
    model_comparison_melted,
    x="Modèle",
    y="Nombre",
    color="Sentiment",
    barmode="stack",
    title="Comparaison des Modèles de Sentiment",
    labels={"Nombre": "Nombre d'Avis", "Modèle": "Modèles", "Sentiment": "Sentiments"}
)
st.plotly_chart(fig, use_container_width=True)
st.write("""
**Observation :** Le modèle BERT détecte des sentiments extrêmes allant de très positif à très négatif, ce qui le rend plus granulaire. TextBlob et VADER sont plus limités. On peut remarquer la précision de bert, dont les estimations de sentiments correspondent plus au moins a quelques pourcent près au ratio commentaire/notes.
         En effet, les avis très négatifs dominent, comme remarqué suite au scrapping avec les notes des utilisateurs qui tournent autour de 1 pour la majorité.
         Quand à vader et textblob, les résultats ne sont pas vraiment cohérents en comparaison, avec le taux d'avis positif qui est meme majoritaire sur TextBlob. On ne pas donc pas les considérer comme fiable dans notre cas.
""")

# Section : Visualisations des mots-clés
st.markdown("## ☁️ Visualisations des Mots-Clés")

# Nuage de mots
st.markdown("### Nuage de Mots")
keywords_mean = keywords_data.mean().sort_values(ascending=False)
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color="white",
    colormap="coolwarm"
).generate_from_frequencies(keywords_mean)
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(wordcloud, interpolation="bilinear")
ax.axis("off")
st.pyplot(fig)
st.write("""
**Observation :** Les mots les plus fréquents, tels que "car" et "service", sont probablement liés à l'expérience directe des utilisateurs avec Tesla.
""")

# Top 10 mots-clés
st.markdown("### Top 10 des Mots-Clés")
top_keywords = keywords_mean.head(10)
fig = px.bar(
    x=top_keywords.index,
    y=top_keywords.values,
    labels={"x": "Mots-Clés", "y": "Valeurs Moyennes"},
    title="Top 10 des Mots-Clés les Plus Fréquents"
)
st.plotly_chart(fig, use_container_width=True)
st.write("""
**Observation :** Les mots-clés principaux mettent en avant les thèmes récurrents des avis, comme les performances de la voiture, le service client ou l'application.
""")