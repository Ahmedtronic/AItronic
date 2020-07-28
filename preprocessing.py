import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


filename = ""
dataset = pd.read_csv(filename)

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 1:].values


##################### Label Encoder #####################

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
col = [] # Columns to be encoded
labelEnc = LabelEncoder()
X[:, col] = labelEnc.fit_transform(X[:, col])

oneHotEnc = OneHotEncoder(categorical_features=[col])
X = oneHotEnc.fit_transform(X).toarray()


##################### Polynomial Features #####################

from sklearn.preprocessing import PolynomialFeatures
poly_dog = 4
polyFeatures = PolynomialFeatures(degree = poly_dog)
X_poly = polyFeatures.fit_transform(X)


##################### Split train-test #####################

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


##################### Standard Scaler #####################

from sklearn.preprocessing import StandardScaler
sc_x, sc_y = StandardScaler()
# Scale X
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)
# Scale y
y_train = sc_y.fit_transform(y_train)
y_test = sc_y.transform(y_test)


