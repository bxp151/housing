
#%% Imports
import requests
import os

import pandas as pd
import exploretransform as et
import numpy as np
import plotnine as pn
import plotly.express as px
import matplotlib.pyplot as plt
from plotly.offline import plot 

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import GammaRegressor, LinearRegression
from sklearn.neural_network import MLPRegressor
import xgboost as xgbr

from sklearn.model_selection import StratifiedShuffleSplit, cross_validate
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance

#%% Seting Configuration
HOME_DIR = '/Users/bxp151/ml/housing'
DATA_DIR = '/data'
IMG_DIR = '/images'

DATA_URL = 'http://lib.stat.cmu.edu/datasets/boston_corrected.txt'


if not os.path.exists("images"):
    os.mkdir(HOME_DIR + IMG_DIR)
    
if not os.path.exists("data"):
    os.mkdir(HOME_DIR + DATA_DIR)
    
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', 500)

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

# will likley use center, scaling for linear modeling

# plot the target 
(
 pn.ggplot(data) +
 pn.aes(x = 'CMEDV') +
 pn.geom_histogram(bins = 30) +
 pn.labs(x = 'Corrected median price of homes 000s',
         title = 'cmedv normal scale')
)

# CMEDV is higly skewed, try gaussian and gamma linear models

# plot the locations of houses
data.plot(kind="scatter", x = "LON", y = "LAT", alpha = 0.4, 
        c="CMEDV", cmap=plt.get_cmap("jet"), colorbar=True )

# seems that houses west of the city between certain longitues are more
# expensive generally  



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
    '''
    Python doesn't have a built in mechanism for stratification in regression
    Specifiy the quantiles and target to create stratified samples
    '''
    
    # calculate the quantiles for the target
    q = df[target].quantile(quantiles)
    
    # create label names for the new category
    l = list(range(1,len(quantiles),1))
    
    # create the new target category
    df["target_cat"] = pd.cut(df[target], bins = q, right=True,
                             labels = l, include_lowest = True) 
    traintest = StratifiedShuffleSplit(n_splits = 1, test_size=0.2, random_state=42)

    for train_idx, test_idx in traintest.split(df, df["target_cat"]):
        train= df.iloc[train_idx]
        test = df.iloc[test_idx]

    train, test = train.drop(['target_cat'], axis = 1), test.drop(['target_cat'], axis = 1)  

    trainX, trainY = train.drop("cmedv", axis = 1), train["cmedv"]
    testX, testY = test.drop("cmedv", axis = 1), test["cmedv"]    
    
    return trainX, trainY,testX, testY 

trainX, trainY,testX, testY  = strat_split(df, target = "cmedv", 
                          quantiles = [0, 0.25, 0.5, 0.75, 1])


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


#%% Naive Baseline

def naive():
    
    naive = pd.DataFrame({'obs': trainY,
                          'pred': np.mean(trainY)
                          })
    
    naive["residsq"] = (naive["obs"] - naive["pred"])**2
    
    rmse = np.sqrt(sum(naive["residsq"])/len(naive))
    
    return np.round(rmse, 2)


#%% Black box baseline pipeline and helper functions

num_feat = trainX.select_dtypes('number').columns
cat_feat = trainX.select_dtypes('category').columns

# numeric
num = Pipeline([
    ( "select", et.ColumnSelect(num_feat))
    ])

# categorical
cat = Pipeline([
    ( "select", et.ColumnSelect(cat_feat)),
    ( 'one_hot',    OneHotEncoder(handle_unknown='ignore') )
    ])

# combined
base_pipe = FeatureUnion([ 
    ('numeric', num), 
    ('categorical', cat) 
    ])
    
    

def base_rmse(estimator):
    '''
    Returns RMSE from cross_validate on specified estimator
    '''
    rf_cv = cross_validate(estimator,
                       X = base_pipe.fit_transform(trainX),
                       y = trainY, 
                       scoring='neg_root_mean_squared_error', 
                       cv = 5)

    rmse = np.mean(-rf_cv['test_score'])
    
    return np.round(rmse, 2)


#%% Baseline models - non-parametric

# Naive - predicting the mean
base01_naive = naive()

# Bagging trees - use all features at every split
base02_bag = base_rmse(RandomForestRegressor(random_state=42, max_features='auto'))

# Random forest - square root of # of features at each split
base03_rf = base_rmse(RandomForestRegressor(random_state=42, max_features = 'sqrt'))

# Adaboost
base04_ada = base_rmse(AdaBoostRegressor(random_state=42))

# Gradient Boost
base05_gb = base_rmse(GradientBoostingRegressor(random_state=42))

# XGBoost
#dtrain = xgb.DMatrix(base_pipe(trainX).fit_transform(trainX), label = trainY)
base06_xgb = base_rmse(xgbr.XGBRegressor(random_state=42))

# nueral network - increased max_iter until convergance
base07_nn = base_rmse(MLPRegressor(random_state=42, max_iter=900))


#%% Baseline Linear pipeline and helper functions

# numeric
num_lin = Pipeline([
    ( "select", et.ColumnSelect(num_feat)),
    ( "scale", StandardScaler())
    ])

# categorical
cat_lin = Pipeline([
    ( "select", et.ColumnSelect(cat_feat)),
    ( 'one_hot',    OneHotEncoder(handle_unknown='ignore') )
    ])

# combined
base_pipe_lin = FeatureUnion([ 
    ('numeric', num_lin), 
    ('categorical', cat_lin) 
    ])



def lin_rmse(estimator):
    '''
    Returns RMSE from cross_validate on specified estimator
    '''
    rf_cv = cross_validate(estimator,
                       X = base_pipe_lin.fit_transform(trainX),
                       y = trainY, 
                       scoring='neg_root_mean_squared_error', 
                       cv = 5)

    rmse = np.mean(-rf_cv['test_score'])
    
    return np.round(rmse, 2)

#%% Baseline Linear (gaussian, gamma)

# Gamma distribution, ridge
base08_gr = lin_rmse(GammaRegressor(alpha=0))

# GLinear Regression
base09_lr = lin_rmse(LinearRegression())


#%% Tune and evaluate Gradient Boost


# tuning parameters
params = {'learning_rate': [0.15,0.1,0.05,0.01],
          'n_estimators':  [50,100,200,400,800,1600]}

# Inner fold for hyperparameter search
inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Outer fold determines generalization error
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)


inner = GridSearchCV(estimator=GradientBoostingRegressor(random_state = 42), 
                  param_grid=params,
                  cv=inner_cv,
                  scoring='neg_root_mean_squared_error')


outer = cross_val_score(estimator=inner, 
                        X=base_pipe.fit_transform(trainX), 
                        y=trainY, 
                        cv=outer_cv)

outer_rmse = np.mean(-outer) # 2.8 

#%% Fit final model and predict on test set

inner.fit(X = base_pipe.transform(trainX), y = trainY)

inner.best_params_

best_model = inner.best_estimator_  

y_hat = best_model.predict(X = base_pipe.transform(testX))

final_rmse = mean_squared_error(testY, y_hat, squared = False)

#%% Calculate feature importance using feature shuffling

# get column names
num_cols = np.array(trainX.columns.drop(['chas','rad']))
cat_cols = cat_lin.steps[1][1].get_feature_names(['chas','rad'])
all_cols = np.append(num_cols, cat_cols)


perm_imp = permutation_importance(best_model, 
                                  X = base_pipe.transform(testX).toarray(),
                                  y = testY,
                                  n_repeats = 100)


feat_imp = pd.DataFrame({"features": all_cols,
                         "importance": perm_imp.importances_mean,
                         "std" : perm_imp.importances_std}).sort_values \
                        ("importance",ascending=False)


#%% Function to cacluate probability that the model's results occured by chance

def target_shuffling(estimator, trainX, trainY, n_iters, scorefunc, 
                     random_state=0, verbose = False):
    '''
    Model agnostic tehcnique invented by John Elder of Elder Research.  The
    results show the probability that the model's results occured by chance
    (p-value)
    
    For n_iters:
        
        1. Shuffle the target 
        2. Fit unshuffled input to shuffled target using estimator 
        3. Make predictions using unshuffled inputs
        4. Score predictions against shuffled target using scoring function
        5. Store and return predictions 
    
    The distribtuion of scores can be used to plot a histogram in order to 
    determine p-value
        
    Parameters
    ----------
    estimator: object
        A final model estimator that will be evaluated

    trainX : {array-like, sparse matrix} of shape (n_samples, n_features)
        The training input samples.
  
    trainY : array-like of shape (n_samples,)
        The training target
            
    n_iters : int
        The number of times to shuffle, refit and score model.
        
    scorefun : function
        The scoring function. For example mean_squared_error() 
        
    random_state : int, default = None
        Controls the randomness of the target shuffling. Set for repeatable
        results  
    
    verbose : boolean, default = False
        The scoring function. For example sklearn.metrics.mean_squared_error() 
        
    Returns
    -------
    scores : array of shape (n_iters)
        These are scores calculated for each shuffle

    '''
 
    for i in range(n_iters):    
        
        # 1. Shuffle the training target 
        np.random.default_rng(seed = random_state).shuffle(trainY)
        random_state += 1

        # 2. Fit unshuffled input to shuffled target using estimator
        estimator.fit(trainX, trainY)
        
        # calculate feature importance using permutation
        
        # 3. Make predictions using unshuffled inputs
        y_hat = estimator.predict(trainX)
        
        # 4. Score predictions against shuffled target using scoring function
        score = scorefunc(trainY, y_hat, squared=False)
        
        # 5. Store and return predictions 
        if i == 0:
            allscores = np.array(score)
        else:
            allscores = np.append(allscores, score)
        
        if verbose:
            print("Shuffle: " + str(i+1) + "\t\tScore: " + str(score))
    
    return allscores



#%% Execute target shuffling

# inner.best_params_['random_state'] = 42 

# gb_estimator = GradientBoostingRegressor(**inner.best_params_)


# gb_scores = target_shuffling(estimator = gb_estimator,
#                  trainX = base_pipe.transform(trainX), 
#                  trainY = np.array(trainY), 
#                  n_iters=100000, 
#                  random_state=0,
#                  verbose=True,
#                  scorefunc=mean_squared_error)

# np.save("scores.npy", gb_scores)
gb_scores = np.load("scores.npy")

#%% Plot feature importance results

top5_feat = feat_imp.iloc[0:5,0:2].sort_values("importance")

fig1 = px.bar(top5_feat, 
              x="importance",
              y="features", 
              title = "Top 5 Features",
              orientation="h",
              labels=dict(features=""))

plot(fig1)
# fig1.write_image(HOME_DIR + IMG_DIR + "/fig1.png")

#%% Plot target shuffling result

fig2 = px.histogram(pd.DataFrame(gb_scores, columns=["RMSE"]), x = "RMSE",
                    labels=dict(RMSE="RMSE (000s USD)"))

fig2.add_vline(x=final_rmse, line_dash = "dash")
fig2.update_layout(xaxis_range=[0,6], yaxis_range=[0,1200])

fig2.add_annotation(x=final_rmse, y=1200,
            text="Best Model",
            showarrow=True,
            yshift=5,
            font=dict(
                size=16
                ))

fig2.add_annotation(x=5.2, y=1200,
            text="Shuffled Models",
            showarrow=True,
            yshift=5,
            font=dict(
                size=16
                ))

plot(fig2)
# fig2.write_image(HOME_DIR + IMG_DIR + "/fig2.png")
