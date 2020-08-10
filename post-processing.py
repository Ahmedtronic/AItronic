import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


##################### Variables Elimination #####################

from statsmodels.tools.tools import add_constant
import statsmodels.api as sm
cols = [] # Columns to be eliminated
X = add_constant(X)
X_optimal = X[:, cols]
model_details = sm.OLS(endog=y, exog=X_optimal).fit()
model_details.summary()


##################### Metrics #####################

from sklearn.metrics import r2_score
cm = r2_score(y_test, y_pred)


##################### Grid Search #####################

from sklearn.model_selection import GridSearchCV
# Get parameters ready
params  = { 'kernel': ['rbf', ], 
           'degree': [3, 4, 5], 
           'C': [1, 10, 100, ], 
           'epsilon': [0.001, 0.01, 0.1, ], 
           }
# Search
model = "Your model"
gscv = GridSearchCV(estimator = model, param_grid = params, scoring  =None, verbose = 0)
gscv.fit(X, y)


##################### Visualizing data (naive) #####################

plt.scatter(X_test, y_test, color="red")
plt.plot(X_test, model.predict(X_test), color="blue")
plt.show()