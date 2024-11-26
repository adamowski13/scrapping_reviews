import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

# Load the dataset
file_path = "./Tesla_Trustpilot_Reviews_Analyzed.csv"  # Update this path as needed
data = pd.read_csv(file_path)

# Convert Date to a datetime object
data['Date'] = pd.to_datetime(data['Date'])
data['Month_Year'] = data['Date'].dt.to_period('M')

# Custom Page Configuration
st.set_page_config(
    page_title="Tesla Reviews Dashboard",
    page_icon="🚗",
    layout="wide"
)

# Sidebar for Theme & Filters
st.sidebar.title("Dashboard Settings")
theme_mode = st.sidebar.radio("Select Theme", options=["Light", "Dark"], index=0)
selected_sentiment = st.sidebar.multiselect(
    "Filter by Sentiment",
    options=data['sentiment_label'].unique(),
    default=data['sentiment_label'].unique()
)

# Apply filters
filtered_data = data[data['sentiment_label'].isin(selected_sentiment)]

# KPIs Section
st.title("🚗 Tesla Trustpilot Reviews Dashboard")
st.markdown("---")

col1, col2, col3 = st.columns(3)

# KPI 1: Total Reviews
with col1:
    st.metric("📊 Total Reviews", len(filtered_data))

# KPI 2: Average Rating
with col2:
    avg_rating = filtered_data['Rating'].mean()
    st.metric("⭐ Average Rating", round(avg_rating, 2))

# KPI 3: Most Common Sentiment
with col3:
    most_common_sentiment = filtered_data['sentiment_label'].mode()[0]
    st.metric("💬 Most Common Sentiment", most_common_sentiment)

# Visualizations
st.markdown("---")

# 1. Ratings Distribution
st.subheader("Ratings Distribution")
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(
    x=filtered_data['Rating'].value_counts().index,
    y=filtered_data['Rating'].value_counts().values,
    palette="coolwarm",
    ax=ax
)
ax.set_title("Ratings Distribution")
ax.set_xlabel("Rating (1-5)")
ax.set_ylabel("Count")
st.pyplot(fig)

# 2. Sentiment Distribution
st.subheader("Sentiment Distribution")
fig, ax = plt.subplots(figsize=(8, 5))
filtered_data['sentiment_label'].value_counts().plot.pie(
    autopct='%1.1f%%',
    startangle=140,
    colors=sns.color_palette("Set2"),
    ax=ax
)
ax.set_ylabel("")
ax.set_title("Sentiment Distribution")
st.pyplot(fig)

# 3. Word Cloud for Review Content
st.subheader("Review Word Cloud & Key Terms")
wordcloud = WordCloud(
    width=800, height=400, background_color="white", colormap="viridis"
).generate(" ".join(filtered_data['Content'].dropna()))
col1, col2 = st.columns([2, 1])

# Display Word Cloud
with col1:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

# Display Top Terms
with col2:
    from collections import Counter
    top_terms = Counter(" ".join(filtered_data['Content']).split()).most_common(10)
    st.write("### Top Terms")
    for term, count in top_terms:
        st.write(f"- **{term.capitalize()}**: {count}")

# 4. Ratings Trend Over Time
st.subheader("Ratings Over Time")
ratings_over_time = filtered_data.groupby('Month_Year')['Rating'].mean()
fig, ax = plt.subplots(figsize=(10, 5))
ratings_over_time.plot(ax=ax, marker="o", color="blue")
ax.set_title("Average Ratings Over Time")
ax.set_xlabel("Month-Year")
ax.set_ylabel("Average Rating")
ax.grid(alpha=0.5)
st.pyplot(fig)

# 5. Data Table
st.subheader("Review Data Table")
st.dataframe(filtered_data)

# Footer
st.markdown("---")

