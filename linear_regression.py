import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

auto = pd.read_csv('automobiles.csv')
print(auto.head())
print(auto.describe())
print(auto.shape)
print(auto.isna().sum())

X = auto.drop('normalized_losses',axis = 1)
y = auto['normalized_losses']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=726)

X_train_subset = X_train[['length', 'engine_size']]
X_train_subset['intercept'] = 1

XX_inv = np.linalg.inv(np.matmul(np.transpose(X_train_subset),X_train_subset))
XY = np.matmul(np.transpose(X_train_subset),y_train)
print(XY.shape)

beta_hat = np.matmul(XX_inv,XY)
