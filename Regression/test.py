# Code source: Jaques Grobler
# License: BSD 3 clause
# Code from sklearn old version (1.5.2) site: https://scikit-learn.org/1.5/auto_examples/linear_model/plot_ols.html

import matplotlib.pyplot as plt
import numpy as np
import time

from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from regression import regression

# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# Use only one feature
diabetes_X = diabetes_X[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

a = time.time()
model_comparation = LinearRegression().fit(diabetes_X_train, diabetes_y_train)

b = time.time()
f, ang, lin = regression(diabetes_X_train, diabetes_y_train, epochs=300, learning_rate=0.01)

c = time.time()

# The coefficients
print("Coefficients: \n", model_comparation.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, model_comparation.predict(diabetes_X_test)))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(diabetes_y_test, model_comparation.predict(diabetes_X_test)))
print(f"TIME SKLEARN: {b-a}")

# Coeficients from my model
print("Coefficients Mine: \n", (ang, lin))
# MSE from my model
print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, f(diabetes_X_test)))
# The coefficient of determination from my model
print("Coefficient of determination: %.2f" % r2_score(diabetes_y_test, f(diabetes_X_test)))
print(f"TIME MODEL: {c-b}")

# Plot outputs
fig, axs = plt.subplots(1, 2, figsize=(20, 6))
axs[0].scatter(diabetes_X_test, diabetes_y_test, color='red')
axs[0].set_title("LINEAR REGRESSOR SKLEARN MODEL")
axs[0].plot(diabetes_X_test, model_comparation.predict(diabetes_X_test))

axs[1].scatter(diabetes_X_test, diabetes_y_test, color='red')
axs[1].set_title("LINEAR REGRESSOR MY MODEL")
axs[1].plot(diabetes_X_test, f(diabetes_X_test))

plt.show()