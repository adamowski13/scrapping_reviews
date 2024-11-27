# Scraping Elon and Sentiment Analysis Project

## Project Overview
This project involves web scraping and sentiment analysis focused on content related to Elon Musk. The project consists of two main components:

1. **Scraping Elon**:
   - A notebook dedicated to scraping content about Elon Musk from various sources.
   - Processes the scraped data to extract meaningful information for analysis.

2. **Sentiment Analysis**:
   - A notebook performing sentiment analysis on the scraped data using various machine learning and natural language processing techniques.

## Features
- Web scraping using Python libraries like BeautifulSoup and Requests.
- Sentiment analysis using **TextBlob**, **VADER Sentiment Analysis**, and **transformers**.
- Visualization of results with **matplotlib**, **seaborn**, and **wordcloud**.
- Interactive user interface built with **Streamlit** for displaying analysis results.

## Installation
To get started, clone the repository and install the required dependencies:

```bash
# Clone the repository
git clone < https://github.com/adamowski13/scrapping_reviews>
cd <scrapping_reviews>

# Install dependencies
pip install -r requirements.txt
```

## Dependencies
The project uses the following Python libraries:

- pandas==2.0.3
- requests==2.31.0
- beautifulsoup4==4.12.2
- textblob==0.17.1
- vaderSentiment==3.3.2
- scikit-learn==1.3.0
- transformers==4.35.0
- streamlit==1.40.0
- matplotlib==3.7.1
- wordcloud==1.9.2
- seaborn==0.12.2
- numpy==1.25.1

Refer to `requirements.txt` for the full list of dependencies.

## Usage

### Scraping Notebook
The notebook `scraping elon.ipynb` includes the logic to scrape and preprocess data related to Elon Musk. To run:

1. Open the notebook in your preferred IDE (e.g., Jupyter Notebook).
2. Run all cells to scrape and store the data for further analysis.

### Sentiment Analysis Notebook
The notebook `treútation anlysis finam version.ipynb` contains sentiment analysis logic. Steps:

1. Open the notebook in your IDE.
2. Provide the scraped data as input.
3. Run all cells to perform sentiment analysis and visualize the results.

### Streamlit App
An optional Streamlit app can be created to visualize the analysis results interactively:

```bash
streamlit run <streamlit_script.py>
```

## Outputs
- Scraped data files ready for analysis.
- Sentiment analysis results, including:
  - Sentiment polarity and subjectivity.
  - Visualizations (e.g., bar charts, word clouds).

## Contributing
Feel free to fork this repository, make changes, and submit pull requests. Suggestions and improvements are welcome!

## License
This project is licensed under the MIT License. See the LICENSE file for details.

