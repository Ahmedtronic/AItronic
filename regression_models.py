import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


##################### Simple Linear Regression #####################

from sklearn.linear_model import LinearRegression
model = LinearRegression(normalize=True)
model.fit(X_train, y_train)
values_to_predict = X_test
result = model.predict([ values_to_predict ])


##################### Multi-Linear Regression #####################

from sklearn.linear_model import LinearRegression
model = LinearRegression(normalize=True)
model.fit(X_train, y_train)
values_to_predict = X_test
result = model.predict([ values_to_predict ])


##################### Polynomial Regression #####################

from sklearn.preprocessing import PolynomialFeatures
poly_dog = 4
polyFeatures = PolynomialFeatures(degree = poly_dog)
X_poly = polyFeatures.fit_transform(X)

from sklearn.linear_model import LinearRegression
model = LinearRegression(normalize=True)
model.fit(X_poly, y_train)
values_to_predict = X_test
result = model.predict([ values_to_predict ])


##################### Support Vector Machines #####################

from sklearn.svm import SVR
model = SVR(kernel = "rbf", degree = 3, C = 1.0, epsilon = 0.1)
model.fit(X_train, y_train)
values_to_predict = X_test
result = model.predict([ values_to_predict ])


##################### Decision Tree #####################

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(criterion = "mse")
model.fit(X_train, y_train)
values_to_predict = X_test
result = model.predict([ values_to_predict ])


##################### Random Forest #####################

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators = 10, criterion="mse")
model.fit(X_train, y_train)
values_to_predict = X_test
result = model.predict([ values_to_predict ])

