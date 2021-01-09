# Predicting median housing prices in Boston

</br>

## Objective
The objective is to predict median home prices in Boston based on data published in 1978.  

The data was provided by U.S. Census service and obtained from the [StatLib Archive](http://lib.stat.cmu.edu/datasets/boston).  It has been used extensively throughout the literature to benchmark algorithms. The data was originally published by Harrison, D. and Rubinfeld, D.L. `Hedonic prices and the demand for clean air', J. Environ. Economics & Management, vol.5, 81-102, 1978. 


## Performance Measure
In order to measure the effectiveness of the model in predicting price, we used root mean square error (RMSE).  RMSE is calculated by: 

1. calculating the sum the squared differences between the predicted (model) and observed (test set) values 
2. dividing #1 by the numer of observations
3. taking the square root of #2

## Constraints
Talk about how some prices are capped

## Key Findings

KF1 - The final model achieved an RMSE of $2,033.18 on the training data and  $1,995.13 on the test data.  The features used in the final model were: 
	* listings - number of listings
	* market_location 
	* latestodometer 
	* iscertified 
	* isleather
	* trimids
	* age: number of days between startdateofprice and Jaunuary 2016
              
KF2 - The most important feature to predict price by far was trimids. Trimids is so important because it captures the year, make, model, engine and drive type and thus would be directly related to MSRP. 

## Approach

The overall approach to building the pricing model is as follows:

1. Initial data exploration
2. Select techniques
3. Split data into Train/Test
4. Build and analyze baseline models
5. Feature engineering
6. Build and analyze secondary models
7. Final predictions using test set
</br>

## Data Description

The original data are 506 observations on 14 variables. cmedv is the target variable

Variable | Description
---- | ------------- 
crim |	per capita crime rate by town
zn |	proportion of residential land zoned for lots over 25,000 sq.ft
indus	| proportion of non-retail business acres per town
chas |	Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
nox	| nitric oxides concentration (parts per 10 million)
rm	| average number of rooms per dwelling
age	| proportion of owner-occupied units built prior to 1940
dis	| weighted distances to five Boston employment centres
rad	| index of accessibility to radial highways
tax	| full-value property-tax rate per USD 10,000
ptratio	| pupil-teacher ratio by town
b	1000(B - 0.63)^2 | where B is the proportion of blacks by town
lstat	| percentage of lower status of the population
medv	| median value of owner-occupied homes in USD 1000's
cmedv	| corrected median value of owner-occupied homes in USD 1000's
town	| name of town
tract	| census tract
lon	| longitude of census tract
lat	| latitude of census tract

##Usage Instructions

Clone the repo by doing ----
