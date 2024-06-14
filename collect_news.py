import logging
import sys
import newspaper
from newspaper import Article
from gnews import GNews
from tqdm import tqdm
import json
from datetime import datetime, timedelta

class NewsBase:
    def __init__(self):
        self.keyword = ''
        self.newslist = []

    def get_keyword(self):
        return self.keyword

    def get_newslist(self):
        return self.newslist

    def set_keyword(self, keyword):
        self.keyword = keyword

    def set_newslist(self, newslist):
        self.newslist = newslist

    # Updated method to include start_date and end_date parameters
    def find_news(self, keyword, start_date, end_date, language='English', category='news'):
        if language == 'English':
            if category == 'news':
                return self.find_news_english_news(keyword, start_date, end_date)
            else:
                # Potential for expansion with additional conditions
                return self.find_news_english_news(keyword, start_date, end_date)


    # Updated to accept start_date and end_date
    def find_news_english_news(self, keyword, start_date, end_date):
        google_news = GNews()
        google_news.max_results = 1000
        # google_news.country = 'United States'
        google_news.language = 'english'
        google_news.start_date = start_date
        google_news.end_date = end_date
        relevant_news = google_news.get_news(keyword)
        return relevant_news

    def get_news_url(self, index):
        try:
            Gurl = self.newslist[index]['url']
            return Gurl
        except Exception as e:
            print(f"Error: {e}")
            return None

    def get_news_title(self, index, language='en'):
        try:
            title = self.newslist[index]['title']
            # print(title)
            return title
        except Exception as e:
            print(f"Error: {e}")
            return None

    def get_news_time(self, index, language='en'):
        try:
            time = self.newslist[index]['published date']
            # print(time)
            return time
        except Exception as e:
            print(f"Error: {e}")
            return None

    def get_news_content(self, url, language='en'):
        # Top import failed since it's not installed
        if 'newspaper' not in (sys.modules.keys() & globals()):
            print("\nget_full_article() requires the `newspaper` library.")
            print(
                "You can install it by running `python3 -m pip install newspaper3k` in your shell.\n")
            return None
        try:
            article = newspaper.Article(url="%s" % url, language=language)
            article.download()
            article.parse()
        except Exception as error:
            logging.error(error)
            # We can try other substitute methods here
            return None
        return article.text


def get_news_from_internet(keyword, date_range, language='English', category='news'):
    start_date, end_date = date_range  # Unpack the date range tuple
    newsbase = NewsBase()
    newsbase.set_keyword(keyword)
    # Updated to include start_date and end_date in the call
    newsbase.set_newslist(newsbase.find_news(keyword, start_date, end_date, language, category))
    news_list_len = len(newsbase.get_newslist())
    news_list = []
    
    for i in range(news_list_len):
        url = newsbase.get_news_url(i)
        content = newsbase.get_news_content(url)
        if not content:
            continue
        time = newsbase.get_news_time(i)
        title = newsbase.get_news_title(i)
        news_list.append({
            'id': i,
            'url': url,
            'time': time,
            'title': title,
            'content': content
        })
    return news_list


if __name__ == "__main__":
    start_date = datetime(2023, 8, 1)
    end_date = datetime(2023, 9, 1)
    keyword = 'Ethereum'

    date_pairs = []
    current_date = start_date
    while current_date < end_date:
        next_day = current_date + timedelta(days=1)
        date_pairs.append(((current_date.year, current_date.month, current_date.day), (next_day.year, next_day.month, next_day.day)))
        current_date = next_day

    for pair in tqdm(date_pairs):
        news = get_news_from_internet(keyword,(pair[0], pair[1]))
        print(f'{pair[0]}:', len(news))
        filename = f'results/{pair[0][0]}-{pair[0][1]}-{pair[0][2]}.json'
        with open(filename, 'w') as f:
            json.dump(news, f, indent=4)