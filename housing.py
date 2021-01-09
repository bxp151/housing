#%% Imports

import pandas as pd
import exploretransform as et
import numpy as np
import plotnine as pn
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
import requests
import os
import matplotlib.pyplot as plt

#%% Seting Configuration

pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', 500)

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

# Feature Names, first 5 obs., types, # Levels for factor
et.peek(df)

df["TOWN"].value_counts()
df["ZN"].value_counts()
df["CHAS"].value_counts()
df["RAD"].value_counts()
df["INDUS"].value_counts()
df["NOX"].value_counts()
df["TAX"].value_counts()
df["PTRATIO"].value_counts()

# def  factor: CHAS, RAD
# lowercase
# drop: OBS, TOWN, TOWN#, MEDV

et.explore(df)
# no n/a, no inf, CHAS 93% zero

desc = df.describe()
df.hist()
skew = et.skewstats(df)
skew["skewness"]= skew["skewness"].abs()
skew.sort_values(by=["skewness"], ascending=False)

#            dtype  skewness           magnitude
# CRIM     float64  5.207652              2-high
# CHAS       int64  3.395799              2-high
# B        float64  2.881798              2-high
# ZN       float64  2.219063              2-high
# CMEDV    float64  1.107616              2-high
# MEDV     float64  1.104811              2-high
# DIS      float64  1.008779              2-high
# RAD        int64  1.001833              2-high

# will likley use center, scaling for certain techniques

(
 pn.ggplot(df) +
 pn.aes(x = 'CMEDV') +
 pn.geom_histogram(bins = 30) +
 pn.labs(x = 'Corrected median price of homes 000s',
         title = 'cmedv normal scale')
)
# CMEDV is higly skewed, try GLMs for linear models



df.plot(kind="scatter", x = "LON", y = "LAT", alpha = 0.4, 
        c="CMEDV", cmap=plt.get_cmap("jet"), colorbar=True )

# there doesn't seem to be any immediate pattern from CMEDV based on the map.  


#%% Select Techniques:

# BASELINES:  
# Naive (mean) | Random Forest

# SECONDARY:
# Linear(GLM) + LASSO
# Linear(GLM) + Elastic
# Regression Trees
# Bagging Trees
# RF
# AdaBoost
# Gradient Boost
# Neural Networks
# XGBoost

#%% Initial Cleanup

def clean(x):
    # drop: OBS, TOWN, TOWN#, MEDV
    x = x.drop(["OBS.", "TOWN", "TOWN#", "MEDV"], axis = 1)
    # lowercase cols
    x.columns = map(str.lower, x.columns)
    # factor: CHAS, RAD
    return x

 
clean(df)



