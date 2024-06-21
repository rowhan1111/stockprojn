import pandas as pd
import datetime
import os
import pickle as pk
from stock_price_model import ModelMaker
import tensorflow as tf


# the agents that will make the trades and trained based on their trades
class Trader:
    def __init__(self):
        pass

# class to keep all the stock prices and give prices to the trader
class Market:
    def __init__(self):
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
        '''
        ModelMaker containing input_n_output with dates in format of "%Y-%m-%d %H-%M-%S" and corresponding ticker
        creates model as well
        '''
        self.ModelMaker = ModelMaker(load_data=True)

        self.model = self.ModelMaker.train_model(train_model=False, model_name="model_temp3", epochs=3)
        self.output = self.ModelMaker.output

        self.pred_output = self.model.predict([self.ModelMaker.past, self.ModelMaker.quarter, self.ModelMaker.yearly, self.ModelMaker.info, self.ModelMaker.headlines])
        self.prices = self.ModelMaker.last_days
        breakpoint()

class Manager:
    def __init__(self):
        pass

x = Market()