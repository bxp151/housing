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

data = load_data()

#%% Initial Exploration

len(data), len(data.columns)
# (506, 21)

# Feature Names, first 5 obs., types, # Levels for factor
et.peek(data)

data["TOWN"].value_counts()
data["ZN"].value_counts()
data["CHAS"].value_counts()
data["RAD"].value_counts()
data["INDUS"].value_counts()
data["NOX"].value_counts()
data["TAX"].value_counts()
data["PTRATIO"].value_counts()

# def  factor: CHAS, RAD
# lowercase
# drop: OBS, TOWN, TOWN#, MEDV, TRACT

et.explore(data)
# no n/a, no inf, CHAS 93% zero

desc = data.describe()
data.hist()
skew = et.skewstats(data)
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
 pn.ggplot(data) +
 pn.aes(x = 'CMEDV') +
 pn.geom_histogram(bins = 30) +
 pn.labs(x = 'Corrected median price of homes 000s',
         title = 'cmedv normal scale')
)
# CMEDV is higly skewed, try GLMs for linear models



data.plot(kind="scatter", x = "LON", y = "LAT", alpha = 0.4, 
        c="CMEDV", cmap=plt.get_cmap("jet"), colorbar=True )

# seems that houses west of the city between certain longitues are more
# expensive generally  


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

def clean(df):
    # drop: OBS, TOWN, TOWN#, MEDV
    df = df.drop(["OBS.", "TOWN", "TRACT", "TOWN#", "MEDV"], axis = 1)
    # lowercase cols
    df.columns = map(str.lower, df.columns)
    # factor: CHAS, RAD
    df[["chas","rad"]] = df[["chas","rad"]].astype('category')
    return df



df = clean(data)

et.peek(df)

#%% Split data

def strat_split(df, target, quantiles = [0, 0.25,0.5,0.75,1]):
    # Python doesn't have a built in mechanism for stratification in regression
    # Specifiy the quantiles and target to create stratified sample
    
    # calculate the quantiles for the target
    q = df[target].quantile(quantiles)
    
    # create label names for the new category
    l = list(range(1,len(quantiles),1))
    
    # create the new target category
    df["target_cat"] = pd.cut(df[target], bins = q, right=True,
                             labels = l, include_lowest = True) 
    traintest = StratifiedShuffleSplit(n_splits = 1, test_size=0.2, random_state=42)

    for train_idx, test_idx in traintest.split(df, df["target_cat"]):
        train= df.loc[train_idx]
        test = df.loc[test_idx]

    train, test = train.drop(['target_cat'], axis = 1), test.drop(['target_cat'], axis = 1)  

    trainX, trainY = train.drop("cmedv", axis = 1), train["cmedv"]
    testX, testY = test.drop("cmedv", axis = 1), test["cmedv"]    
    
    return trainX, trainY,testX, testY 

# use more quantiles in the upper range due to the rareness of those observations
trainX, trainY,testX, testY  = strat_split(df, target = "cmedv", 
                          quantiles = [0, 0.1, 0.2, 0.3, 0.4, 
                                       0.5, 0.6, 0.7, 0.8, 0.85, 
                                       0.9, 0.95, 1])


# verify cmedv looks similar between train/test
(
 pn.ggplot(pd.DataFrame(trainY)) +
 pn.aes(x="cmedv") +
 pn.geom_histogram(bins = 30)
 )

(
 pn.ggplot(pd.DataFrame(testY)) +
 pn.aes(x="cmedv") +
 pn.geom_histogram(bins = 30)
 )


#%% Naive Baseline - calculate RMSE using mean(cmedv) as the prediction

naive = pd.DataFrame({'obs': trainY,
                      'pred': np.mean(trainY)
                      })

naive["residsq"] = (naive["obs"] - naive["pred"])**2

naiveRMSE = np.sqrt(sum(naive["residsq"])/len(naive))

#%% Black box baseline - Use random forest with defaults

# https://scikit-learn.org/stable/modules/ensemble.html#forest





