import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import tensorflow as tf
import pandas as pd
from preprocess import Preprocessor
import numpy as np
import pickle
from tqdm import tqdm
import datetime
import torch


class DataHolder:
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
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.gpu_available = False if device_name == 'cpu' else True
        '''
        self.processor.thread_storing(self.past_length, self.future_length, self.scaled_path,
                                      store_path='inputs_n_outputs/tickers/')
        '''
        # used to create input n output data, true loads existing data
        self.input_n_output = self.data_load(load_data)
        self.dates = self.input_n_output['dates']
        self.dates = np.reshape(self.dates, (np.shape(self.dates)[0], 1))
        self.tickers = self.input_n_output['tickers']
        self.tickers = np.reshape(self.tickers, (np.shape(self.tickers)[0], 1))
        self.list_for_inputs = (
            self.process_to_input(self.input_n_output))
        self.info, self.past, self.headlines, self.quarter, self.yearly, self.output, self.future = self.list_for_inputs
        self.last_days = self.daily_scaler.inverse_transform(np.array([past[-1] for past in self.past]))
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

class ModelMakerPyTorch(nn.Module):
    def __init__(self, vocab_size, past_length, quarter_length, yearly_length, info_length, headlines_length):
        super(ModelMakerPyTorch, self).__init__()

        # Define layers for past data
        self.rnn_past = nn.RNN(input_size=4, hidden_size=4, batch_first=True)

        # Define layers for quarterly data
        self.rnn_quarter = nn.RNN(input_size=5, hidden_size=4, batch_first=True)

        # Define layers for yearly data
        self.rnn_yearly = nn.RNN(input_size=5, hidden_size=4, batch_first=True)

        # Define layers for stock info
        self.dense_info = nn.Linear(info_length, 5)

        # Define layers for headlines
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=2, padding_idx=0)
        self.lstm_headlines = nn.LSTM(input_size=2, hidden_size=2, batch_first=True)
        self.dense_headlines = nn.Linear(2, 1)

        # Define final dense layer
        self.final_dense = nn.Linear(4 + 4 + 4 + 5 + 1, 4)

    def forward(self, past, quarter, yearly, info, headlines):
        # Process past data
        out_past, _ = self.rnn_past(past)
        out_past = out_past[:, -1, :]

        # Process quarterly data
        out_quarter, _ = self.rnn_quarter(quarter)
        out_quarter = out_quarter[:, -1, :]

        # Process yearly data
        out_yearly, _ = self.rnn_yearly(yearly)
        out_yearly = out_yearly[:, -1, :]

        # Process stock info
        out_info = torch.tanh(self.dense_info(info))

        # Process headlines
        embedded_headlines = self.embedding(headlines)
        lstm_out, _ = self.lstm_headlines(embedded_headlines)
        out_headlines = torch.tanh(self.dense_headlines(lstm_out[:, -1, :]))

        # Combine all outputs
        combined = torch.cat([out_past, out_quarter, out_yearly, out_info, out_headlines], dim=1)
        final_output = torch.tanh(self.final_dense(combined))

        return final_output


def train_model_pytorch(model, train_data, train_labels, epochs=10, batch_size=32, learning_rate=0.001, device='cpu'):
    # Move model to the appropriate device
    model.to(device)

    # Create DataLoader
    train_dataset = TensorDataset(*train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Set model to training mode
    model.train()

    # Training loop
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (past, quarter, yearly, info, headlines, labels) in enumerate(train_loader):
            # Move data to the appropriate device
            past, quarter, yearly, info, headlines, labels = past.to(device), quarter.to(device), yearly.to(device), info.to(device), headlines.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(past, quarter, yearly, info, headlines)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 10 == 9:  # Print every 10 batches
                print(f'Epoch [{epoch + 1}/{epochs}], Batch [{i + 1}/{len(train_loader)}], Loss: {running_loss / 10:.4f}')
                running_loss = 0.0

    print('Training Finished')

# Example usage:
# Assuming train_data and train_labels are pre-defined tensors with the correct shapes
# train_data should be a tuple of tensors: (past, quarter, yearly, info, headlines)
# train_labels should be a tensor of shape (batch_size, 4)

# Example data (replace with actual data)
batch_size = 32
train_past = torch.randn(batch_size * 100, 100, 4)  # Adjust past_length accordingly
train_quarter = torch.randn(batch_size * 100, 30, 5)  # Adjust quarter_length accordingly
train_yearly = torch.randn(batch_size * 100, 30, 5)  # Adjust yearly_length accordingly
train_info = torch.randn(batch_size * 100, 10)  # Adjust info_length accordingly
train_headlines = torch.randint(0, 1000, (batch_size * 100, 10))  # Adjust vocab_size and headlines_length accordingly
train_labels = torch.randn(batch_size * 100, 4)

train_data = (train_past, train_quarter, train_yearly, train_info, train_headlines)

# Create the model
model = ModelMakerPyTorch(vocab_size=1000, past_length=100, quarter_length=30, yearly_length=30, info_length=10, headlines_length=10)

# Train the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_model_pytorch(model, train_data, train_labels, epochs=10, batch_size=batch_size, learning_rate=0.001, device=device)
