# Code source: Jaques Grobler
# License: BSD 3 clause

import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
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

f, ang, lin = regression(diabetes_X_train, diabetes_y_train, epochs=1000, learning_rate=0.001)

# The coefficients
print("Coefficients: \n", (ang, lin))
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, f(diabetes_X_test)))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(diabetes_y_test, f(diabetes_X_test)))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
plt.plot(diabetes_X_test, f(diabetes_X_test), color="blue", linewidth=3)

plt.show()