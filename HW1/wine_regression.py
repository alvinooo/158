import numpy as np
import urllib
import scipy.optimize
import csv

def parseData(fname):
    data = []
    labels = []
    features = urllib.urlopen(fname)
    data_reader = csv.reader(features, delimiter=';')
    next(data_reader)
    for row in data_reader:
        data.append(map(lambda n: float(n), row[:-1]))
        labels.append(float(row[-1]))
    return data, labels

def feature(datum):
    x = np.array(datum)
    x = np.insert(x, 0, 1, axis=1)
    return x

data, labels = parseData('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv')
x = feature(data)
y = np.array(labels)

# 4
theta, residuals, rank, s = np.linalg.lstsq(x, y.T)
print "Theta", theta
print "MSE", residuals / len(data)

# 5
x_train, x_test = x[:len(x) / 2], x[len(x) / 2:]
y_train, y_test = y[:len(y) / 2], y[len(y) / 2:]
theta, residuals, rank, s = np.linalg.lstsq(x_train, y_train.T)
print "Training MSE", residuals / len(x_train)
error = np.dot(x_train, theta) - y_test
print "Test MSE", np.dot(error, error) / len(y_test)