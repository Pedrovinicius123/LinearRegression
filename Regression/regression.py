import numpy as np
import matplotlib.pyplot as plt

import random, time
from alive_progress import alive_bar
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression

def mae_loss(X, y):
    return np.mean(np.abs(X-y))

def calculate_deviance(y):
    deviance = []

    for i in range(1, y.size):
        deviance.append(abs(y[i] - y[i-1]))

    return np.mean(deviance)

def calculate_minimal_point(y, f):
    minimum = float('inf')

    for y_target in y:
        if y_target < minimum:
            minimum = y_target

    return minimum

def regression(X, y, epochs=20, learning_rate=0.001):
    lin = 0
    ang = random.random()

    def f(X):
        return ang * X + lin
    
    lin = calculate_minimal_point(y, f)    
    anterior_loss = None
    fit_ang = True
    count = 0

    with alive_bar(epochs) as bar:
        for i in range(epochs):
            loss = mae_loss(f(X), y)
            if anterior_loss is not None and anterior_loss < loss:
                fit_ang = False
                count += 1
                if count == 2:
                    break

            anterior_loss = loss
            if fit_ang:
                ang += calculate_deviance(y) * loss * learning_rate
            
            lin += learning_rate * loss
            bar()

    return f, ang, lin

if __name__ == '__main__':
    inp = np.linspace(0, 10, num=300) + 0.0001

    data = 3*inp + 10
    data += np.random.uniform(-15, 15, size=300)
    

    X_train, X_test, y_train, y_test = train_test_split(inp, data, test_size=0.3, shuffle=True)

    model_fitted = LinearRegression().fit(X_train.reshape(-1,1), y_train.reshape(-1,1))
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].scatter(X_train, y_train)
    axs[1].scatter(X_train, y_train)

    axs[0].scatter(X_test, y_test, color='red')
    axs[1].scatter(X_test, y_test, color='red')
    
    axs[0].set_title("MY MODEL")
    axs[1].set_title('SKLEARN MODEL')

    f, ang, lin = regression(np.array(X_train), np.array(y_train), epochs=300, learning_rate=0.001)
    axs[0].plot(X_test, f(X_test), color='green')
    axs[1].plot(X_test, model_fitted.predict(X_test.reshape(-1,1)), color='green')

    print("MY MODEL: ", r2_score(f(X_test), y_test))
    print("SKLEARN MODEL: ", r2_score(model_fitted.predict(X_test.reshape(-1,1)), y_test))

    plt.show()
