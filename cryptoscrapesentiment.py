''' 1. Install and Import Baseline Dependencies '''
#Hugging Face - Pegasus is a language processing deep learning framework
from transformers import PegasusTokenizer, PegasusForConditionalGeneration #TFPegasusForConditionalGeneration
#BeautifulSoup - web scraper
from bs4 import BeautifulSoup
#Allow us to make a request and gather data from the web
import requests
import re
from transformers import pipeline
import csv

## I believe your computer must have a CUDA compatible GPU to run this program correctly

''' 2. Setup Model '''
# holds string of name of model on Hugging Face
model_name = "human-centered-summarization/financial-summarization-pegasus"
# encode/decode news articles into input format our model can work with
tokenizer = PegasusTokenizer.from_pretrained(model_name)
# loads in model
# If you want to use the Tensorflow model just replace with TFPegasusForConditionalGeneration
model = PegasusForConditionalGeneration.from_pretrained(model_name)

''' 3. Setup Pipeline '''
# stock tickers to keep an eye out for
monitored_tickers = ['ETH', 'BTC']

# 4.1. Search for Stock News using Google and Yahoo Finance
print('Searching for stock news for', monitored_tickers)
def search_for_stock_news_links(ticker):
    # url format found from looking at google and how the search url looks for using yahoo finance through google news
    search_url = 'https://www.google.com/search?q=yahoo+finance+{}&tbm=nws'.format(ticker)
    r = requests.get(search_url)
    # build a Beautiful Soup scraper object 'Soup' that parses html input
    soup = BeautifulSoup(r.text, 'html.parser')
    # find a list of all the urls of the <a> tags for the articles found
    atags = soup.find_all('a')
    # grabbing the string inside the atags
    hrefs = [link['href'] for link in atags]
    return hrefs

# creates a dictionary of raw_urls and ticker keys. Some Urls are unwanted and strange
raw_urls = {ticker:search_for_stock_news_links(ticker) for ticker in monitored_tickers}

# 4.2. Strip out unwanted URLs
print('Cleaning URLs.')
# strip out google urls we don't want. Add any words to exclude
exclude_list = ['maps', 'policies', 'preferences', 'accounts', 'support']

# make sure url has https component and no excluded words
def strip_unwanted_urls(urls, exclude_list):
    val = []
    for url in urls:
        if 'https://' in url and not any(exc in url for exc in exclude_list):
            # conditional lookup
            res = re.findall(r'(https?://\S+)', url)[0].split('&')[0]
            val.append(res)
    return list(set(val))

# all of the cleaned_urls
cleaned_urls = {ticker:strip_unwanted_urls(raw_urls[ticker] , exclude_list) for ticker in monitored_tickers} 

# 4.3. Search and Scrape Cleaned URLs
print('Scraping news links.')
def scrape_and_process(URLs):
    ARTICLES = []
    for url in URLs:
        # make request to get information from URL webpage (everything from webpage is held in 'r')
        r = requests.get(url)
        # build a Beautiful Soup scraper object 'Soup' that parses html input
        soup = BeautifulSoup(r.text, 'html.parser')
        # gather paragraphs from webpage (<p> tags)
        results = soup.find_all('p')
        # extract explicit text from all of the paragraphs into an array by looping through paragraphs 
        text = [res.text for res in results]
        #limit of how many unique tokens we can pass to Pegasus
        words = ' '.join(text).split(' ')[:300]
        # rejoin words back together into an article
        ARTICLE = ' '.join(words)
        ARTICLES.append(ARTICLE)
    return ARTICLES
articles = {ticker:scrape_and_process(cleaned_urls[ticker]) for ticker in monitored_tickers} 

# 4.4. Summarise all Articles
print('Summarizing articles.')
def summarize(articles):
    summaries = []
    for article in articles:
        # encoding into pytorch tensor input ids to pass into pegasus model
        # Tokenize our text
        # If you want to run the code in Tensorflow, please remember to return the particular tensors as simply as using return_tensors = 'tf'
        input_ids = tokenizer.encode(article, return_tensors="pt")
        # set the max length of summaries here, num beams search algo
        output = model.generate(input_ids, max_length=55, num_beams=5, early_stopping=True)
        # decode tensor ids into text
        summary = tokenizer.decode(output[0], skip_special_tokens=True)
        # append to summaries array
        summaries.append(summary)
    return summaries

summaries = {ticker:summarize(articles[ticker]) for ticker in monitored_tickers}

''' 5. Adding Sentiment Analysis '''
print('Calculating sentiment.')
sentiment = pipeline("sentiment-analysis")
scores = {ticker:sentiment(summaries[ticker]) for ticker in monitored_tickers}

# # 6. Exporting Results
print('Exporting results')
def create_output_array(summaries, scores, urls):
    output = []
    for ticker in monitored_tickers:
        for counter in range(len(summaries[ticker])):
            output_this = [
                            ticker, 
                            summaries[ticker][counter], 
                            scores[ticker][counter]['label'], 
                            scores[ticker][counter]['score'], 
                            urls[ticker][counter]
                          ]
            output.append(output_this)
    return output
final_output = create_output_array(summaries, scores, cleaned_urls)
final_output.insert(0, ['Ticker','Summary', 'Sentiment', 'Sentiment Score', 'URL'])

with open('summaries.csv', mode='w', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerows(final_output)