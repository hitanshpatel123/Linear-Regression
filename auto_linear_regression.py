import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.stats import boxcox

import matplotlib.pyplot as plt

auto = pd.read_csv('automobiles.csv')

print(auto.head())
print(auto.describe())
print(auto.shape)
print(auto.isna().sum())

X= auto.drop('normalized_losses', axis =1)
y= auto['normalized_losses']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state= 727, test_size= 0.2)

X= X_train[['length', 'engine_size']]

lm= LinearRegression()
lm.fit(X,y_train)

intercept = lm.intercept_
slope = lm.coef_
coefficients = lm.coef_


X_train_with_dummies = pd.get_dummies(X_train['fuel_type'])
X_diesel = X_train_with_dummies[['diesel']]

model = LinearRegression()
model.fit(X_diesel, y_train)
intercept = model.intercept_
coefficient = model.coef_

############## residuals ###########
ml2 = LinearRegression()

X_train_subset = X_train[['length', 'width']]
ml2.fit(X_train_subset, y_train)

predictions = ml2.predict(X_train_subset)
residuals = y_train - predictions
residual_mean = residuals.mean()


fig, ax = plt.subplots()
ax = plt.scatter(predictions, residuals)
plt.show()

training_mse = mean_squared_error(y_train, predictions)
training_rmse =  root_mean_squared_error(y_train, predictions)
R2 = r2_score(y_train, predictions)

################# predictors and evaluation #################
auto_corr = auto.select_dtypes(['float','number']).corr()

test_pred = ml2.predict(X_test_subset)
test_mse = mean_squared_error(y_test, test_pred)


# length squared
X_train['length_squared'] = (X_train['length'])**2
X_test['length_squared'] = (X_test['length'])**2
X_train_subset = X_train[['length','length_squared']]
X_test_subset = X_test[['length','length_squared']]

model = LinearRegression()

train_predictions = model.fit(X_train_subset, y_train).predict(X_train_subset)
test_predictions = model.predict(X_test_subset)

train_mse = mean_squared_error(y_train, train_predictions)
test_mse = mean_squared_error(y_test, test_predictions)

# outcome transform
X_train_subset = X_train[['length']]
X_test_subset = X_test[['length']]

y_train_log = np.log2(y_train)
y_test_log = np.log2(y_test)

model = LinearRegression()
train_predictions = model.fit(X_train_subset, y_train_log).predict(X_train_subset)
test_predictions = model.predict(X_test_subset)

train_mse = mean_squared_error(y_train_log, train_predictions)
test_mse = mean_squared_error(y_test_log, test_predictions)


# box cox transformations
auto['normalized_losses'].hist()
plt.show()

auto['log_norm_losses'] = boxcox(auto['normalized_losses'], lmbda=0)
auto['log_norm_losses'].hist()
plt.show()

auto['boxcox_norm_losses'] = boxcox(auto['normalized_losses'])[0]
auto['boxcox_norm_losses'].hist()
plt.show()