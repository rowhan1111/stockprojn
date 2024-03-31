import requests
import yahoo_fin.stock_info as si
import time
import pandas as pd
import os

class NewsData:

    def __init__(self, start_date):
        self.tickers_list = si.tickers_nasdaq()
        self.all_dict = {i: {} for i in self.tickers_list}
        tickers = '%2C'.join(self.tickers_list)
        start_time = start_date + "T00%3A00%3A00Z"
        # set start and ending time
        self.url = (f'https://data.alpaca.markets/v1beta1/news?&sort=asc&symbols={tickers}'
                    f'&limit=50&start={start_time}')
        api_key = os.environ['api_key']
        api_secret = os.environ['api_secret']
        self.headers = {
            'Apca-Api-Key-Id': api_key,
            'Apca-Api-Secret-Key': api_secret
        }
        # set header for url

    def add_to_all_dict(self, news):
        # create dictionary to store headlines and sources for all tickers in nasdaq
        for single_news in news:
            for sing_ticker in list(set(single_news['symbols']).intersection(set(self.tickers_list))):
                date = single_news['created_at'][:10]
                if date not in self.all_dict[sing_ticker].keys():
                    self.all_dict[sing_ticker][date] = {'headline': [single_news['headline']],
                                                        'source': [single_news['source']]}
                else:
                    self.all_dict[sing_ticker][date]['headline'].append(single_news['headline'])
                    self.all_dict[sing_ticker][date]['source'].append(single_news['source'])

    def make_dict_for_tickers(self):
        # make csv files corresponding to
        response = requests.get(self.url, headers=self.headers).json()
        self.add_to_all_dict(response['news'])
        while response['next_page_token']:
            url2 = self.url + f"&page_token={response['next_page_token']}"
            response = requests.get(url2, headers=self.headers).json()
            self.add_to_all_dict(response['news'])
        return_dict = {}
        for ticker in self.all_dict.keys():
            return_dict[ticker] = pd.DataFrame.from_dict(self.all_dict[ticker],
                                                         orient='index', columns=['headline', 'source'])
        return return_dict

