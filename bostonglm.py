import pandas as pd
import exploretransform as et
import numpy as np
import plotnine as pn
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix

# set options to display df columns
pd.set_option('display.expand_frame_repr', False)

source = "http://lib.stat.cmu.edu/datasets/boston_corrected.txt"
rd = pd.read_table(source, skiprows= 9, encoding="cp1252")

df = rd.copy()
df.columns = map(str.lower, df.columns)

# dimensions
len(df), len(df.columns)
et.peek(df) 

# drop non-numeric colums & MEDV (old target)
df = df.drop(['obs.', 'town', 'town#', 'tract', 'medv', 'chas'], axis = 1)
et.explore(df)

# plot cmedv 
(
 pn.ggplot(df) +
 pn.aes(x = 'cmedv') +
 pn.geom_histogram(bins = 30) +
 pn.labs(x = 'Corrected median price of homes 000s',
         title = 'cmedv normal scale')
)

# plot log(cmedv) 
(
 pn.ggplot(df) +
 pn.aes(x = np.log(df['cmedv'])) +
 pn.geom_histogram(bins = 30) +
 pn.labs(x = 'Corrected median price of homes log scale',
         title = 'cmedv log scale')
)


# Python doesn't have a built in mechanism for stratification in regression
# Use cmedv quantiles to create a new category cmedv_cat and split on it
q = df["cmedv"].quantile([0, 0.25,0.5,0.75,1])
df["cmedv_cat"] = pd.cut(df["cmedv"], bins = q, right=True,
                         labels = [1,2,3,4], include_lowest = True) 
traintest = StratifiedShuffleSplit(n_splits = 1, test_size=0.2, random_state=42)

for train_idx, test_idx in traintest.split(df, df["cmedv_cat"]):
    train= df.loc[train_idx]
    test = df.loc[test_idx]

train, test = train.drop(['cmedv_cat'], axis = 1), test.drop(['cmedv_cat'], axis = 1)      


# explore
train.hist(bins = 50)
scatter_matrix(train) # rm, lstat are obvious
et.ascores(train.drop("cmedv", axis = 1), train["cmedv"]) # confirming rm, lstat


# Fit gamma identity and gamma log using LASSO
# Prep - center, scale, NZV



