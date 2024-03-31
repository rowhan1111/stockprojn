import pandas as pd
import datetime
import os
import pickle as pk
from model import ModelMaker


class Trader:
    def __init__(self):
        pass


class Manager:
    def __init__(self):
        self.ModelMaker = ModelMaker()

        # create model/ retrieve model
        self.model = self.ModelMaker.train_model(train_model=True, model_name='model_up_to_10_news', epochs=30)

        self.output = self.ModelMaker.output

        self.output = self.model.predict([self.ModelMaker.past, self.ModelMaker.quarter, self.ModelMaker.yearly, 
                                          self.ModelMaker.info, self.ModelMaker.headlines])

