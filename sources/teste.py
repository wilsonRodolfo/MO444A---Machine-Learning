import math
import time
import pandas as pd
import quandl
from numpy import *
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from statistics import mean
from matplotlib import style


def step_gradient(b_current, m_current, train, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(train))
    for i in range(1, len(train)):
        x = train[i, 0]
        y = train[i, 1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]


def gradient_descent_runner(train, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(train), learning_rate)
    return [b, m]


train = pd.read_csv('train.csv')

train = train[['shares', 'num_imgs']]
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(train)
df_normalized = pd.DataFrame(np_scaled)
print(train.head())
print(df_normalized.head())

learning_rate = 0.01
initial_b = 0  # initial y-intercept guess
initial_m = 0  # initial slope guess
num_iterations = 1000
print "Starting gradient descent at b = {0}, m = {1}".format(initial_b, initial_m)
print "Running..."
start_time = time.time()
[b, m] = gradient_descent_runner(df_normalized, initial_b, initial_m, learning_rate, num_iterations)
print("--- %s seconds ---" % (time.time() - start_time))
print "After {0} iterations b = {1}, m = {2}".format(num_iterations, b, m)

"""
def best_fit_slope_and_intercept(xs, ys):
    m = (((mean(xs) * mean(ys)) - mean(xs * ys)) / ((mean(xs)**2) - (mean(xs**2))))

    b = mean(ys) - m*mean(xs)
    return m, b

n_xs_train, n_ys_train = normalization_features(xs_train, ys_train)

m, b = best_fit_slope_and_intercept(n_xs_train, n_ys_train)

regression_line = [(m*x)+b for x in xs_test]

plt.scatter(xs_train, ys_train)
plt.plot(xs_test, regression_line)
plt.show()

print(m, b)

"""






