import numpy as np
import pandas as pd
import datetime
import os
from math import isnan, floor
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import linregress
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import random
import threading
import concurrent.futures
from tensorflow.keras.preprocessing.text import Tokenizer


# class for preprocessing data to be put into the model
class Preprocessor:
    def __init__(self, file_path, columns_to_use, essentials, columns_to_drop, div_split, scale_path='scaled/',
                 ticker_list=None):
        self.file_path = file_path
        self.scaled_path = scale_path
        self.columns_to_use = columns_to_use
        self.essential_no_vol = essentials
        self.essentials = essentials + ["Volume"]
        self.columns_to_drop = columns_to_drop
        self.d_for = "%Y-%m-%d %H:%M:%S"
        self.d_for2 = "%Y-%m-%d"
        self.all_dict = {}
        self.info_pd = pd.read_csv(file_path + "info.csv", header=0, index_col=0, na_filter=False)
        temp_info = pd.read_csv(file_path + "sub/info.csv", header=0, index_col=0, na_filter=False)
        self.ticker_list = self.info_pd.index.tolist()
        self.part_ticker_list = self.info_pd.index.tolist()[:2] if not ticker_list else ticker_list
        self.output_method = self.OutputMethods()
        self.div_split = div_split
        self.columns_list = set()  # used to store columns in general?
        self.year_list = []  # used to store year columns
        self.quarter_list = []  # used to store quarter columns
        self.headline_columns = []
        self.daily_scaler = StandardScaler()
        self.vol_scaler = StandardScaler()
        self.quarter_scaler = StandardScaler()
        self.year_scaler = StandardScaler()
        self.tokenizer = Tokenizer()
        self.vocab_size = 0
        self.one_hot = OneHotEncoder(handle_unknown='ignore')
        self.info_list = ["industry", "sector"]
        self.info_pd = pd.DataFrame(self.one_hot.fit_transform(self.info_pd[self.info_list]).toarray(),
                                    index=self.info_pd.index)
        # one hot encode the information about the company
        self.info_columns = self.info_pd.columns

    # process specific ticker based on the range given
    def process(self, ticker_to_process, func_rang, predict_date, file_path):
        # get all the information needed about the ticker from stored data
        pd_to_process = pd.read_csv(file_path + ticker_to_process + ".csv", header=0, index_col=0)
        quarterly_data = (pd.read_csv(f"{file_path}quarterly_{ticker_to_process}.csv", header=0, index_col=0)
                          .fillna(0))
        yearly_datas = (pd.read_csv(f"{file_path}yearly_{ticker_to_process}.csv", header=0, index_col=0)
                        .fillna(0))
        ticker_info = self.info_pd.loc[ticker_to_process]
        length_of_pd = len(pd_to_process.index)
        # check if the length of the inputs won't exceed our data
        if length_of_pd < func_rang+predict_date+1:
            print(f"Possible length is {length_of_pd}")
            return None
        return_dict = {}
        # process to return a dictionary containing input & output for each date
        for date in pd_to_process.index.sort_values()[func_rang:length_of_pd-predict_date]:
            # skip the days when the market was closed
            if pd_to_process.loc[date, self.essentials].dropna().empty:
                continue
            future_date = datetime.datetime.strptime(date, self.d_for)+datetime.timedelta(days=predict_date)
            future_score = pd_to_process.loc[datetime.datetime.strftime(future_date, self.d_for), self.essentials]
            # get future data to use for output
            while isnan(future_score["Open"]):
                future_date = future_date + datetime.timedelta(days=1)
                if datetime.datetime.strftime(future_date, self.d_for) in pd_to_process.index:
                    future_score = (
                        pd_to_process.loc)[datetime.datetime.strftime(future_date, self.d_for), self.essentials]
                else:
                    return return_dict
            # store the values in the dictionary
            future_val = pd_to_process.loc[
                [datetime.datetime.strftime(i, self.d_for)
                 for i in pd.date_range(end=future_date, start=datetime.datetime.strptime(date, self.d_for), freq="D")],
                self.essentials]
            future_no_vol = future_val[self.essential_no_vol]
            unscaled_vol = pd.DataFrame(self.vol_scaler.inverse_transform(pd.DataFrame(future_val["Volume"])),
                                        columns=["Volume"], index=future_val.index)

            unscaled = future_no_vol * self.daily_scaler.scale_[0] + self.daily_scaler.mean_[0]
            concated = unscaled.join(unscaled_vol)
            unscaled_vals = self.output_method.best_fit_line(concated).dropna()
            unscaled_vals["Volume"] = unscaled_vals["Volume"]/self.vol_scaler.scale_
            '''
            swap_column_fut = self.output_method.best_fit_line(future_val).dropna()
            swap_column_fut.columns = [i + "_fut" for i in swap_column_fut.columns]
            return_dict[date] = {"future_values": swap_column_fut}
            '''
            unscaled_vals.columns = [i + "_fut" for i in unscaled_vals.columns]
            return_dict[date] = {"future_values": unscaled_vals}

            # self.output_method.seasonal_decomposition(future_val)
            # get past data
            end_date = datetime.datetime.strptime(date, self.d_for)
            temp_pd_stor = pd_to_process.loc[[datetime.datetime.strftime(i, self.d_for) for i in pd.date_range(
                end=end_date - datetime.timedelta(days=1), periods=func_rang, freq="D")], self.columns_to_use]
            return_dict[date]["input_dates"] = temp_pd_stor[self.essentials].dropna()
            # historical datas
            if not self.headline_columns:
                self.headline_columns = [i for i in pd_to_process.columns if 'headline' in i and i != 'headline']

            return_dict[date]["input_news"] = pd_to_process.loc[[datetime.datetime.strftime(i, self.d_for)
                                                                 for i in pd.date_range(
                end=end_date - datetime.timedelta(days=1), periods=func_rang, freq="D")], self.headline_columns]
            # pd.DataFrame(temp_pd_stor['headline'], columns=["headline"])
            get_pos_columns_q = [col for col in quarterly_data.columns if datetime.datetime.strptime(col, self.d_for2) <
                                 datetime.datetime.strptime(date, self.d_for)]
            trans_quarter = quarterly_data[get_pos_columns_q].transpose()
            return_dict[date]['input_quarters'] = trans_quarter.T.groupby(level=0).first().T  # quarterly datas
            get_pos_columns_y = [col for col in yearly_datas.columns if datetime.datetime.strptime(col[:10],
                                                                                                   self.d_for2)
                                 < datetime.datetime.strptime(date, self.d_for)]
            yearly_data = yearly_datas[get_pos_columns_y].transpose()
            if not self.year_list:
                self.year_list = list(set(yearly_data.columns).difference(set(trans_quarter.columns)))
            return_dict[date]['yearly'] = yearly_data[self.year_list].T.groupby(level=0).first().T  # yearly datas
            return_dict[date]['info'] = ticker_info  # information about the ticker
            temp_columns_list = []
            for key, df in return_dict[date].items():
                if type(df) is pd.Series:
                    temp_columns_list += list(df.index)
                    continue
                temp_columns_list += list(df.columns)
            # remove all the columns that only exist in some files for quarterly datas
            if not self.columns_list:
                self.columns_list = temp_columns_list
            self.columns_list = list(set(self.columns_list).intersection(set(temp_columns_list)))
        return return_dict

    # process all the tickers
    def process_all(self, past, future, file_path):
        for ticker in tqdm(self.ticker_list, desc="process all"):
            result = self.process(ticker, past, future, file_path)
            if result:
                self.all_dict[ticker] = result
        return self.all_dict

    def get_columns(self, file_path, columns_list=None):
        tickers = self.info_pd.index.tolist()
        for ticker_to_process in tqdm(tickers, desc="get columns"):
            quarterly_data = (pd.read_csv(f"{file_path}quarterly_{ticker_to_process}.csv", header=0, index_col=0)
                              .fillna(0)).transpose()
            yearly_datas = (pd.read_csv(f"{file_path}yearly_{ticker_to_process}.csv", header=0, index_col=0)
                            .fillna(0)).transpose()
            if not self.year_list:
                self.year_list = set(yearly_datas.columns).difference(set(quarterly_data.columns))
            self.year_list = set(self.year_list).intersection(set(yearly_datas.columns).difference(
                set(quarterly_data.columns)))
            if not self.quarter_list:
                self.quarter_list = set(quarterly_data.columns)
            self.quarter_list = set(self.quarter_list).intersection(set(quarterly_data.columns))
        if columns_list:
            self.year_list, self.quarter_list = (list(self.year_list.intersection(set(columns_list))),
                                                 list(self.quarter_list.intersection(set(columns_list))))
        else:
            self.year_list, self.quarter_list = list(self.year_list), list(self.quarter_list)
        return self.year_list, self.quarter_list

    def prep_input_threading(self, past, future, file_path):
        '''
        input_n_out_dict = self.process_all(past, future, file_path)
        del self.all_dict, input_n_out_dict
        '''
        self.get_columns(file_path)
        thread_num = 2
        len_ticker = len(self.part_ticker_list)
        chunk_size = floor(len_ticker/thread_num)
        params_list = [(past, future, file_path, self.part_ticker_list[i: i + chunk_size], i)
                       for i in range(0, len_ticker, chunk_size)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=thread_num) as executor:
            futures = [executor.submit(self.prep_input, *params) for params in params_list]
        concurrent.futures.wait(futures)
        thread_results = [future.result() for future in concurrent.futures.as_completed(futures)]
        return pd.concat(thread_results, ignore_index=True)

    # function to store inputs
    def thread_storing(self, past, future, file_path, store_path):
        self.get_columns(file_path)
        thread_num = 2
        len_ticker = len(self.ticker_list)
        chunk_size = floor(len_ticker / thread_num)
        params_list = [(past, future, file_path, self.ticker_list[i: i + chunk_size], store_path, i)
                       for i in range(0, len_ticker, chunk_size)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=thread_num) as executor:
            futures = [executor.submit(self.store_by_ticker, *params) for params in params_list]
        concurrent.futures.wait(futures)

    def store_by_ticker(self, past, future, file_path, ticker_list, store_path, pos):
        for ticker in tqdm(ticker_list, desc=f"storing tickers{pos}"):
            (self.prep_input(past, future, file_path, [ticker], pos).
             to_pickle(f"{store_path}input_n_output{past}&{future}&{ticker}.pkl"))
        return None

    # combine all pds for all tickers
    def prep_input(self, past, future, file_path, ticker_list, pos):
        # iterate through all tickers and its dictionary
        # for ticker, dates in tqdm(input_n_out_dict.items(), desc="preparing inputs", position=0):
        index = 0
        all_dict = {i: [] for i in self.columns_list}
        dates_list = []
        for ticker in tqdm(ticker_list, desc=f"preparing inputs{pos}", leave=False):
            dates = self.process(ticker, past, future, file_path)
            self.columns_list = list(set(self.columns_list).difference(set(self.headline_columns)))
            self.columns_list.append('headline')
            if index == 0:
                all_dict = {i: [] for i in self.columns_list}
                index += 1
            all_dict = {i: all_dict[i] for i in list(set(all_dict.keys()).intersection(self.columns_list))}
            if not dates:
                continue
            # iterate through the dictionary in the given date
            for date, date_dict in tqdm(dates.items(), desc=f"Processing dates{pos}", leave=False):
                dates_list.append(date)
                # iterate through the dictionary in the given date and store them in all_dict
                for index_key, value in tqdm(date_dict.items(), desc=f"processing datedicts{pos}", leave=False):
                    if index_key == "info":
                        for key in value.index.sort_values():
                            if key not in all_dict.keys():
                                all_dict[key] = [value.loc[key]]
                            else:
                                all_dict[key].append(value.loc[key])
                        continue
                    elif index_key == "input_news":
                        temp_list = []
                        for index in value.index.sort_values():
                            temp_list.append(value.loc[index])
                        if 'headline' not in all_dict.keys():
                            all_dict['headline'] = [temp_list]
                        else:
                            all_dict['headline'].append(temp_list)
                        continue
                    for column_name in value.columns:
                        if column_name not in self.columns_list:
                            continue
                        all_dict[column_name].append(value[column_name].sort_index(axis=0).values)
        all_dict['dates'] = dates_list
        return pd.DataFrame(all_dict)

    def to_embeddings(self, headlines):
        self.tokenizer.fit_on_texts(headlines)
        self.vocab_size = len(self.tokenizer.word_index) + 1
        embedding = self.tokenizer.texts_to_sequences(headlines)
        df = pd.DataFrame(embedding, index=headlines.index)
        df.columns = [f"headline{i}" for i in df.columns]
        self.headline_columns = list(df.columns)
        return df

    # scale daily datas
    def scale_daily(self, store=True):
        scaled_path = self.scaled_path
        orig_pds = {ticker: pd.read_csv(self.file_path + ticker + ".csv", header=0, index_col=0)
                    for ticker in self.ticker_list}
        headlines = pd.concat(orig_pds.values(), keys=orig_pds.keys())['headline']
        embeddings = self.to_embeddings(headlines)
        to_scale = pd.concat(orig_pds.values(), keys=orig_pds.keys())
        to_scale["Volume"] = self.vol_scaler.fit_transform(to_scale[["Volume"]])
        to_scale = to_scale.join(embeddings)
        # combine all open, close, high, low in one column and get scaling factors
        all_comb = pd.concat([to_scale[i].dropna() for i in self.essential_no_vol], keys=["index", "values"])
        self.daily_scaler.fit(pd.DataFrame(all_comb, columns=['testing']))
        to_scale[self.essential_no_vol] = ((to_scale[self.essential_no_vol] - self.daily_scaler.mean_[0])
                                           / self.daily_scaler.scale_[0])
        if store:
            for ticker in self.ticker_list:
                df = to_scale.loc[ticker]
                df.to_csv(f"{scaled_path}{ticker}.csv")

    # scale quarterly datas
    def scale_quarterly(self, store=True):
        scaled_path = self.scaled_path
        orig_pds = {ticker: pd.read_csv(self.file_path + "quarterly_" + ticker + ".csv", header=0,
                                        index_col=0).transpose() for ticker in self.ticker_list}
        to_scale = pd.concat(orig_pds.values(), keys=orig_pds.keys())
        columns_list = set()
        # get the intersecting columns for all quarterly datas
        for ticker, df in orig_pds.items():
            if not columns_list:
                columns_list = set(df.columns)
                continue
            columns_list = columns_list.intersection(set(df.columns))
        columns_list = list(columns_list)
        self.quarter_list = columns_list

        to_scale[columns_list] = self.quarter_scaler.fit_transform(to_scale[columns_list])
        if store:
            for ticker in self.ticker_list:
                try:
                    df = to_scale.loc[ticker]
                    df.transpose().to_csv(f"{scaled_path}quarterly_{ticker}.csv")
                except:
                    df = pd.DataFrame(columns=columns_list).transpose()
                    df.to_csv(f"{scaled_path}quarterly_{ticker}.csv")

    # scale yearly
    def scale_yearly(self, store=True):
        scaled_path = self.scaled_path
        orig_pds = {ticker: pd.read_csv(self.file_path + "yearly_" + ticker + ".csv", header=0,
                                        index_col=0).transpose() for ticker in self.ticker_list}
        # get intersecting columns of all yearly datas
        columns_list = set()
        for ticker, df in orig_pds.items():
            if not columns_list:
                columns_list = set(df.columns)
                continue
            if len(df.columns) != 0:
                columns_list = columns_list.intersection(set(df.columns))
            else:
                orig_pds[ticker] = pd.DataFrame(columns=list(columns_list))
        columns_list = list(columns_list)

        # concatenate all dataframes for all tickers into one dataframe
        to_scale = pd.concat([df[columns_list] for key, df in orig_pds.items()], keys=orig_pds.keys())
        to_scale[columns_list] = self.year_scaler.fit_transform(to_scale[columns_list])
        if store:
            for ticker in self.ticker_list:
                try:
                    df = to_scale.loc[ticker]
                    df.transpose().to_csv(f"{scaled_path}yearly_{ticker}.csv")
                except:
                    df = pd.DataFrame(columns=columns_list).transpose()
                    df.to_csv(f"{scaled_path}yearly_{ticker}.csv")

    # scale all datas
    def scale_all(self):
        pd.read_csv(self.file_path + "info.csv", header=0, index_col=0).to_csv(self.scaled_path + "info.csv")
        self.scale_daily()
        self.scale_quarterly()
        self.scale_yearly()
        return self.quarter_scaler, self.year_scaler, self.vol_scaler, self.daily_scaler, self.tokenizer

    # use to get a graph from pd dataframe
    def create_graph(self, pd_dataframe: pd.DataFrame):
        pd_dataframe = pd_dataframe[self.essentials].dropna()
        plt.figure()
        pd_dataframe.plot()
        plt.show()
        print(pd_dataframe)

    # various methods to process the output
    class OutputMethods:

        # leave the output as the stock price
        def last_day(self, future_values: pd.DataFrame):
            return future_values.tail(1)

        # create the best fit line of the future_values to identify trend?
        def best_fit_line(self, future_values: pd.DataFrame):
            def calculate_slope(column):
                x = np.array(range(1, len(column) + 1))
                y = np.array(column)

                slope, _ = np.polyfit(x, y, 1)

                return slope
            temp = pd.DataFrame(future_values.dropna().apply(calculate_slope)).transpose()
            return temp

        def seasonal_decomposition(self, future_values: pd.DataFrame):
            result = seasonal_decompose(future_values.dropna()['Close'], model='additive', period=1)
            result.plot()
            plt.show()

