import os
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import Input, layers
from preprocess import Preprocessor
import numpy as np
import pickle
from tqdm import tqdm
import datetime
import torch


class ModelMaker:
    def __init__(self, load_scale=True, load_data=True, past=100, fut=30):
        columns_used = ["headline", "Open", "High", "Low", "Close", "Volume"]
        self.essential = ["Open", "High", "Low", "Close"]
        self.yearly_columns = []
        self.essential_out = [i + "_fut" for i in self.essential]
        self.act_out = [i + "_actfut" for i in self.essential]
        div_splits = ["Dividends", "Stock Splits"]
        columns_dropped = []
        self.file_path, self.scaled_path = 'temp2/', 'scaled2/'
        self.processor = Preprocessor(self.file_path, columns_used, self.essential, columns_dropped, div_splits,
                                      self.scaled_path)
        self.past_length, self.future_length = past, fut
        # get scales' parameters
        # true loads the existing scaler while false creates new scaler and scales from given file
        self.quarter_scaler, self.year_scaler, self.vol_scaler, self.daily_scaler, self.tokenizer = self.scaler(load_scale)
        self.vocab_size = len(self.tokenizer.word_index) + 1
        device_name = tf.test.gpu_device_name()
        self.gpu_available = False if not device_name else True
        '''
        self.processor.thread_storing(self.past_length, self.future_length, self.scaled_path,
                                      store_path='inputs_n_outputs/tickers/')
        '''
        # used to create input n output data, true loads existing data
        self.input_n_output = self.data_load(load_data)
        self.dates = self.input_n_output['dates']
        self.list_for_inputs = (
            self.process_to_input(self.input_n_output))
        self.info, self.past, self.headlines, self.quarter, self.yearly, self.output, self.future = self.list_for_inputs
        self.last_days = np.array([past[-1] for past in self.past])
        '''
        test = self.model.predict([self.past, self.quarter, self.yearly, self.info, self.headlines])
        out = self.output
        '''

    # scale datas and put them into scaled folder + store scaler into scaler folder
    def scaler(self, load_file=False):
        # self.quarter_scaler, self.year_scaler, self.vol_scaler, self.daily_scaler, self.tokenizer
        if not load_file:
            to_return = self.processor.scale_all()
            index = 1
            for scaler in to_return:
                pickle.dump(scaler, file=open(f'scaler/{index}.pkl', 'wb'))
                index += 1
        else:
            scaler_path = 'scaler/'
            files = os.listdir(path=scaler_path)
            to_return = []
            for file in files:
                to_return.append(pickle.load(open(scaler_path + file, 'rb')))
            (self.processor.quarter_scaler, self.processor.year_scaler, self.processor.vol_scaler,
             self.processor.daily_scaler, self.processor.tokenizer) = to_return
        return to_return

    # load data + create data if load_data is false
    def data_load(self, load_data=True, store_as_pickle=True):
        file_path = 'inputs_n_outputs/'
        if not load_data:
            to_return = self.processor.prep_input_threading(self.past_length, self.future_length, self.scaled_path)
            if store_as_pickle:
                to_return.to_pickle(f"{file_path}input_n_output{self.past_length}&{self.future_length}.pkl")
            return to_return
        elif f'input_n_output{self.past_length}&{self.future_length}.pkl' not in os.listdir(file_path):
            print("No valid file, set load_data to false to create file or find another file")
            quit()
        return pd.read_pickle(f'{file_path}input_n_output{self.past_length}&{self.future_length}.pkl')

    # function to create the model to use for predicting stock trends
    def create_model_tensorflow(self) -> Model:
        # process past
        input_past = Input(shape=(self.past.shape[1], int(len(self.essential))))
        rnn_layer_past = layers.SimpleRNN(4, input_shape=(self.past.shape[1], int(len(self.essential))))
        out_past = rnn_layer_past(input_past)

        # process quarterly data
        input_quarter = Input(shape=(self.quarter.shape[1], int(len(self.quarter_columns))))
        rnn_layer_quarter = layers.SimpleRNN(4, input_shape=(self.quarter.shape[1], int(len(self.quarter_columns))))
        out_quarter = rnn_layer_quarter(input_quarter)

        # process yearly data
        input_yearly = Input(shape=(self.yearly.shape[1], int(len(self.year_columns))))
        rnn_layer_year = layers.SimpleRNN(4, input_shape=(self.yearly.shape[1], int(len(self.year_columns))))
        out_yearly = rnn_layer_year(input_yearly)

        # process stock info
        input_info = Input(shape=(self.info.shape[1], ))
        out_info = layers.Dense(units=5, activation='tanh')(input_info)

        # process headlines

        # normally working version, requires flattening
        input_headlines = Input(shape=(self.headlines.shape[1], ))

        # masked = layers.Masking()(input_headlines)
        embedding_layer = layers.Embedding(input_dim=self.vocab_size, output_dim=2, input_length=self.headlines.shape[1], mask_zero=True)(input_headlines)
        lstm_layer = layers.LSTM(2, input_shape=(None, 3))(embedding_layer)
        out_headlines = layers.Dense(1, activation='tanh')(lstm_layer)

        '''
        # weirdly working version - lstm stacking?
        input_headlines = Input(shape=(self.headlines.shape[1], self.headlines.shape[2]))
        headline_lstm = []  # list to store all the lstm's
        embedding_layer = layers.Embedding(input_dim=self.vocab_size, output_dim=2,
                                           input_length=self.headlines.shape[2], mask_zero=True)(input_headlines)
        split_layers = layers.Lambda(lambda x: tf.unstack(x, axis=1))(embedding_layer)  # input_headlines  # split input headlines
        # create lstm model with embedding for each split layer
        for i in range(len(split_layers)):
            input_headline = layers.Input(shape=(None, embedding_layer.shape.dims[3]))
            # embedding_layer = layers.Embedding(input_dim=self.vocab_size, output_dim=2, input_length=self.headlines.shape[2], mask_zero=True)(input_headline)
            # lstm_layer = layers.LSTM(5, input_shape=(None, 3), return_sequences=True)(embedding_layer)
            lstm_layer = layers.SimpleRNN(2, input_shape=(None, 3), return_sequences=True)(input_headline)
            headline_lstm.append(Model(inputs=input_headline, outputs=lstm_layer))
        # combine all layers with concatenate
        headlines_comb = [lstm(split_layer) for lstm, split_layer in zip(headline_lstm, split_layers)]
        con_headlines = layers.Concatenate(axis=1)(headlines_comb)
        # merged_layer_reshaped = layers.Reshape((-1, self.past_length))(con_headlines)
        # process the news headlines of different dates using lstm
        rnn_layer_headline = layers.LSTM(units=4, input_shape=(None, ))(con_headlines)
        out_headlines = layers.Dense(1, activation='tanh')(rnn_layer_headline)
        '''
        # combine all and return output + define model
        con = layers.concatenate([out_past, out_quarter, out_yearly, out_info, out_headlines])
        final_output = layers.Dense(units=4, activation='tanh')(con)
        model = Model(inputs=[input_past, input_quarter, input_yearly, input_info, input_headlines],
                      outputs=final_output)
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def train_model(self, train_model, model_name, epochs=10):
        print("Model Training commenced")
        if train_model:
            model = self.create_model_tensorflow()
            print("Model Created")
            if self.gpu_available:
                with tf.device('/gpu:0'):
                    print("gpu being used")
                    logs = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs, histogram_freq=1, profile_batch=(500, 520))
                    tf.profiler.experimental.server.start(6000)
                    model.fit([self.past, self.quarter, self.yearly, self.info, self.headlines], self.percent_diff, epochs=epochs, batch_size=128, validation_split=0.2, callbacks=[tensorboard_callback])
                    # model.fit([self.past, self.quarter, self.yearly, self.info, self.headlines], self.output, epochs=epochs, batch_size=128, validation_split=0.2, callbacks=[tensorboard_callback])
            else:
                model.fit([self.past, self.quarter, self.yearly, self.info, self.headlines], self.output,
                               epochs=epochs, batch_size=512, validation_split=0.2)
            model.save(f'models/{model_name}.keras')
            #pickle.dump(model, file=open(f'models/{model_name}.pkl', 'wb'))
            return model
        else:
            return tf.keras.models.load_model(f'models/{model_name}.keras')
            # return pickle.load(file=open(f'models/{model_name}.pkl', 'rb'))

    def process_to_input(self, input_n_output, num_news_days=10):
        if not self.processor.quarter_list and not self.processor.year_list:
            self.quarter_columns, self.year_columns = self.processor.get_columns(self.file_path,
                                                                                 self.input_n_output.columns.tolist())
        else:
            self.quarter_columns, self.year_columns = self.processor.quarter_list, self.processor.year_list
        # format datas to be put into AI model
        # preparing headlines to be put into the model
        temp_list = []
        for i in input_n_output["headline"]:
            temp_list.append(np.array(i[-num_news_days:]).flatten())  # temp_list.append(np.array(i))

        headlines = np.stack(temp_list)  # format headline into a workable np ndarray
        del temp_list
        headlines[np.isnan(headlines)] = 0  # change all nan into 0
        info = input_n_output[self.processor.info_columns]  # one hot encoded information about the company
        past = change_to_input(input_n_output[self.essential])
        input_n_output.drop(self.essential, axis=1, inplace=True)
        quarter = change_to_input(input_n_output[self.quarter_columns])
        input_n_output.drop(self.quarter_columns, axis=1, inplace=True)
        yearly = change_to_input(input_n_output[self.year_columns])
        input_n_output.drop(self.year_columns, axis=1, inplace=True)
        output = change_to_input(input_n_output[self.essential_out])
        input_n_output.drop(self.essential_out, axis=1, inplace=True)
        output = np.squeeze(output)
        future = change_to_input(input_n_output[self.act_out])
        input_n_output.drop(self.act_out, axis=1, inplace=True)
        future = np.squeeze(future)
        # del self.input_n_output
        return info, past, headlines, quarter, yearly, output, future

# function to combine pd dataframes' cells' lists in a way similar to transposing a matrix
def simplify_pd_with_lists(df: pd.DataFrame):
    return np.array(pd.DataFrame(df.to_dict()).values)


def change_to_input(df: pd.DataFrame):
    return_list = []
    for index in tqdm(df.index, desc="changing to input"):
        return_list.append(np.array(simplify_pd_with_lists(df.loc[index])))
    max_rows = max(arr.shape[0] for arr in return_list)
    return_list = [np.pad(array=arr, pad_width=((max_rows-arr.shape[0], 0), (0, 0))) for arr in return_list]
    return_list = np.stack(return_list, axis=0)
    return return_list


def retrieve_max_len(df):
    return df.shape[1]
    # return int(df.apply(len).max())


class TorchModel(torch.nn.Module):
    def __init__(self):
        pass
        


if __name__ == "__main__":
    tester = ModelMaker(load_data=False)
