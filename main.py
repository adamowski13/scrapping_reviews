import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import plotly.express as px

# Load the dataset
file_path = "./data/Tesla_Trustpilot_Reviews_Analyzed.csv"  # Update this path as needed
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

# Custom HTML Styling for Dashboard
st.markdown(
    """
    <style>
    body {
        font-family: 'Arial', sans-serif;
    }
    .main-container {
        padding: 0px 10px;
    }
    .title {
        color: #333;
        font-size: 45px;
        font-weight: bold;
        text-align: center;
    }
    .subtitle {
        color: #666;
        font-size: 20px;
        text-align: center;
        margin-bottom: 40px;
    }
    .kpi-container {
        display: flex;
        justify-content: space-between;
        margin-bottom: 40px;
    }
    .kpi-card {
        background: #f3f4f6;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        width: 30%;
        box-shadow: 2px 4px 10px rgba(0,0,0,0.1);
    }
    .kpi-card h3 {
        margin: 0;
        font-size: 18px;
        color: #555;
    }
    .kpi-card p {
        margin: 5px 0 0;
        font-size: 30px;
        font-weight: bold;
        color: #333;
    }
    .footer {
        text-align: center;
        margin-top: 50px;
        color: #999;
        font-size: 15px;
    }
    </style>
    <div class="title">🚗 Tesla Reviews Dashboard</div>
    <div class="subtitle">Explore Customer Feedback and Sentiment Analysis</div>
    """,
    unsafe_allow_html=True,
)

# Sidebar for Filters
st.sidebar.title("Filters and Settings")

# Add Date Filter
start_date, end_date = st.sidebar.date_input(
    "Filter by Date Range",
    [data['Date'].min().date(), data['Date'].max().date()]
)

# Add Sentiment Filter
selected_sentiment = st.sidebar.multiselect(
    "Filter by Sentiment",
    options=data['sentiment_label'].unique(),
    default=data['sentiment_label'].unique()
)

# Add Rating Filter
selected_rating = st.sidebar.slider("Filter by Ratings", 1, 5, (1, 5))

# Apply filters
data['Date'] = data['Date'].dt.tz_localize(None)
filtered_data = data[
    (data['Date'].between(pd.Timestamp(start_date), pd.Timestamp(end_date))) &
    (data['sentiment_label'].isin(selected_sentiment)) &
    (data['Rating'].between(*selected_rating))
]

# KPIs Section
st.markdown(
    """
    <div class="kpi-container">
        <div class="kpi-card">
            <h3>Total Reviews</h3>
            <p>{}</p>
        </div>
        <div class="kpi-card">
            <h3>Average Rating</h3>
            <p>{}</p>
        </div>
        <div class="kpi-card">
            <h3>Most Common Sentiment</h3>
            <p>{}</p>
        </div>
    </div>
    """.format(
        len(filtered_data),
        round(filtered_data['Rating'].mean(), 2) if not filtered_data.empty else "N/A",
        filtered_data['sentiment_label'].mode()[0] if not filtered_data.empty else "N/A",
    ),
    unsafe_allow_html=True,
)

# Visualizations Section
st.markdown("## Visualizations")

# Use Plotly for Interactivity
col1, col2 = st.columns(2)

# 1. Ratings Distribution
with col1:
    st.markdown("### Ratings Distribution")
    fig = px.histogram(
        filtered_data,
        x="Rating",
        nbins=5,
        title="Distribution of Ratings",
        color_discrete_sequence=["#636EFA"],
    )
    fig.update_layout(
        xaxis_title="Ratings",
        yaxis_title="Count",
        title_x=0.5,
        template="plotly_white",
    )
    st.plotly_chart(fig)

# 2. Sentiment Distribution
with col2:
    st.markdown("### Sentiment Distribution")
    fig = px.pie(
        filtered_data,
        names="sentiment_label",
        title="Sentiment Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3,
    )
    fig.update_layout(title_x=0.5, template="plotly_white")
    st.plotly_chart(fig)

# 3. Word Cloud for Review Content
st.markdown("### Word Cloud of Reviews")
wordcloud = WordCloud(
    width=800, height=400, background_color="white", colormap="viridis"
).generate(" ".join(filtered_data['Content'].dropna()))
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(wordcloud, interpolation="bilinear")
ax.axis("off")
st.pyplot(fig)

# 4. Ratings Over Time
st.markdown("### Ratings Over Time")
ratings_over_time = filtered_data.groupby('Month_Year')['Rating'].mean()
fig = px.line(
    x=ratings_over_time.index.astype(str),
    y=ratings_over_time.values,
    title="Average Ratings Over Time",
    labels={"x": "Month-Year", "y": "Average Rating"},
    markers=True,
)
fig.update_layout(template="plotly_white", title_x=0.5)
st.plotly_chart(fig)

# 5. Data Table
st.markdown("### Filtered Data Table")
st.dataframe(filtered_data)

# Footer Section
st.markdown(
    """
    <div class="footer">🚀 Built with Streamlit | Powered by HTML, CSS, and Plotly</div>
    """,
    unsafe_allow_html=True,
)
from collections import Counter
import plotly.graph_objects as go

# Word Cloud Interactivity Section
st.markdown("### Interactive Word Cloud")

# Generate word frequencies
all_text = " ".join(filtered_data['Content'].dropna())
word_freq = Counter(all_text.split())
top_words = word_freq.most_common(100)

# Create data for the Plotly Word Cloud
words, frequencies = zip(*top_words)
sizes = [freq * 2 for freq in frequencies]  # Scale sizes for better visualization

# Create interactive Word Cloud using Plotly
fig = go.Figure()

for word, freq, size in zip(words, frequencies, sizes):
    fig.add_trace(
        go.Scatter(
            x=[freq],  # Frequency as x-axis
            y=[size],  # Scaled size for display
            text=[f"Word: {word}<br>Frequency: {freq}"],  # Tooltip
            mode="markers+text",
            textfont=dict(size=size, color="black"),
            marker=dict(
                size=size,
                color=freq,
                colorscale="Viridis",
                showscale=False,
            ),
        )
    )

# Update layout for better Word Cloud display
fig.update_layout(
    showlegend=False,
    xaxis=dict(showgrid=False, zeroline=False, visible=False),
    yaxis=dict(showgrid=False, zeroline=False, visible=False),
    title="Interactive Word Cloud",
    title_x=0.5,
    margin=dict(l=0, r=0, t=40, b=0),
    template="plotly_white",
)

# Display Word Cloud
st.plotly_chart(fig, use_container_width=True)

# User Interaction: Select a word
st.markdown("### Word Frequency Insights")
selected_word = st.text_input("Enter a word from the Word Cloud to see its frequency:", "")

if selected_word:
    word_count = word_freq.get(selected_word.lower(), 0)
    st.write(f"The word **'{selected_word}'** appears **{word_count}** times in the reviews.")
else:
    st.write("Enter a word above to see its frequency.")
