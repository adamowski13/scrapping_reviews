# Scraping Elon and Sentiment Analysis Project

## Project Overview
This project involves web scraping and sentiment analysis focused on content related to Elon Musk. The project consists of two main components:

1. **Scraping**:
   - A script dedicated to scraping content about Elon Musk from various sources.
   - Processes the scraped data to extract meaningful information for analysis.

2. **Sentiment Analysis**:
   - A notebook performing sentiment analysis on the scraped data using various machine learning and natural language processing techniques.

## Features
- Web scraping using Python libraries like BeautifulSoup and Requests.
- Sentiment analysis using **TextBlob**, **VADER Sentiment Analysis**, and **transformers**.
- Visualization of results with **matplotlib**, **seaborn**, and **wordcloud**.
- Interactive user interface built with **Streamlit** for displaying analysis results.

## Installation
The deployment of our application was made through streamlit servers, you can't make it any easier :)
### streamlit server
to access the dashboard, just access this url:
https://scrappingreviews-zhuspmw6yvmkxae9f8wfeq.streamlit.app/

### Docker Image
In case you find issues with the url (as a free server it could be down for exemple), you can also pull our docker Image locally and it will run on your port 8501 (you can also choose another one to run).

docker pull juliendira/tesla-reviews-dashboard

docker run -it juliendira/tesla-reviews-dashboard
In case you want to change the deployement port, use this run command instead
docker run -it -p [your_port]:[container_port] juliendira/tesla-reviews-dashboard


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

### Scraping 
The script `scrape_data.py` includes the logic to scrape and preprocess data related to Tesla
it does use the website Trustpilot (url: "https://www.trustpilot.com/review/www.teslamotors.com")
we dynamically get all the data from the website (user, date,content,rating,title) and create a csv file that holds the data of all the pages

### clean data 
The script `clean_data.py` includes the logic clean the data, by deteleing the special caracters using re.sub function, truncate the user comment if too long for the bert model, etc...

### Sentiment Analysis
The script `extract_sentiment.py` contains sentiment analysis logic

## parent file
extract_sentiment.py is the main function of treatment (excluding stramlit), it calls all the modules we created and described earlier and run them in the right order so we get the whole chain running


1. using textblob model based on ntlk
2. using vader model
3. using Bert based on neuralinks for the analysis and using context

It also add columns with each model score (between -1 and 1) for the comment and it's interpretation (positive,negative,etc...)

### Streamlit App
A Streamlit app can be created to visualize the analysis results interactively through the running of the docker image locally. it'll just need an empty port that can be modified through the command line


## Outputs
- Scraped data files ready for analysis.
- Sentiment analysis results, including:
  - Sentiment polarity and subjectivity.
  - Visualizations (e.g., bar charts, word clouds).

## Contributing
Feel free to fork this repository, make changes, and submit pull requests. Suggestions and improvements are welcome!

## License
This project is licensed under the MIT License. See the LICENSE file for details.

