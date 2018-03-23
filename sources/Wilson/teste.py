import math
import pandas as pd
import quandl
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from statistics import mean
from matplotlib import style

style.use('fivethirtyeight')

df = pd.read_csv('train.csv')
df = df[['shares', 'num_imgs']]


def best_fit_slope_and_intercept(xs, ys):
    m = ( ((mean(xs) * mean(ys)) - mean(xs * ys)) /
          ((mean(xs)**2) - (mean(xs**2))))

    b = mean(ys) - m*mean(xs)
    return m, b


m, b = best_fit_slope_and_intercept(df['shares'], df['num_imgs'])


regression_line = [(m*x)+b for x in df['shares']]

predict_x = 600000
predict_y = (m*predict_x)+b

plt.scatter(df['shares'], df['num_imgs'])
plt.scatter(predict_x, predict_y, color='g')
plt.plot(df['shares'], regression_line)
plt.show()

print(m, b)






