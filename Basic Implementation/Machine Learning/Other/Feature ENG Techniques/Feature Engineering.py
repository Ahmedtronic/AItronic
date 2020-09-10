import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


filename = ""
dataset = pd.read_csv(filename)

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 1:].values

##################### SPLIT #####################
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)






##################### SCALE SCALING #####################
# Standard Scaler
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#  Min Max Scaler
from sklearn.preprocessing import MinMaxScaler
MM_scaler = MinMaxScaler()
X_train = MM_scaler.fit_transform(X_train)
X_test = MM_scaler .transform(X_test)


# Power Transformer
from sklearn.preprocessing import PowerTransformer
pow_trans = PowerTransformer()
X_train = pow_trans.fit_transform(X_train)
X_test = pow_trans .transform(X_test)





##################### Categorical one hot dummy label encoding #####################

dummy_encoding = pd.get_dummies(my_dataframe, columns=[''], prefix='')

one_hot_encoding = pd.get_dummies(my_dataframe, columns=[''], prefix='', drop_first=True)





##################### Binary Binarizing data columns ##################

my_dataframe["newColumn"] = 0
threshold = 0
# Replace all the values where myColumn is > threshold
my_dataframe.newColumn[my_dataframe[my_dataframe["myColumn"] > threshold].index] = 1







#################### Binning categorizing dividing splitting data to equal groups ############
my_dataframe['equal_binned'] = pd.cut(my_dataframe['myColumn'], bins=5)






#################### Drop fill missing nan values ################
# Remove entire row or column
no_missing_values_rows = my_dataframe.dropna(how="any", axis=0) # 0 removes rows | 1 removes columns

# remove row/col in specific variable
no_missing_values = my_dataframe.dropna(subset=["column"])

# fill missing category values
my_dataframe['column'].fillna("Not Given", inplace=True)

# fill missing numeric values
my_dataframe['column'].fillna(round(my_dataframe['folumn'].mean()), inplace=True)

