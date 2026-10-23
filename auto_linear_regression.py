import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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
