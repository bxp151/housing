#%% Imports

import pandas as pd
import exploretransform as et
import numpy as np
import plotnine as pn
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
import requests
import os

#%% Seting Configuration

pd.set_option('display.expand_frame_repr', False)

HOME_DIR = '/Users/bxp151/ml/housing'
DATA_DIR = '/data'
DATA_URL = 'http://lib.stat.cmu.edu/datasets/boston_corrected.txt'

#%% Acquire Dataset


def acquire(data_url = DATA_URL, data_dir = HOME_DIR + DATA_DIR):
    os.makedirs(data_dir, exist_ok=True)
    r = requests.get(data_url)
    with open(data_dir + "/housing.txt", 'wb') as f:
        f.write(r.content)

acquire()

#%% Load Dataset

def load_data(file = HOME_DIR + DATA_DIR + "/housing.txt"):
    df = pd.read_table(file, skiprows= 9, encoding="cp1252")
    return df

df = load_data()

#%% Initial Exploration

len(df), len(df.columns)
# (506, 21)

et.peek(df)
# lowercase, drop: OBS, TOWN, TOWN#, MEDV, 




