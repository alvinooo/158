import numpy as np
import urllib
from sklearn import svm
import csv

def parseData(fname):
    data = []
    labels = []
    features = urllib.urlopen(fname)
    data_reader = csv.reader(features, delimiter=';')
    next(data_reader)
    for row in data_reader:
        data.append(map(lambda n: float(n), row[:-1]))
        labels.append('positive' if int(row[-1]) > 5 else 'negative')
    return data, labels

def feature(datum):
    x = np.array(datum)
    x = np.insert(x, 0, 1, axis=1)
    return x

def accuracy(acc, pair):
    return acc + 1 if pair[0] == pair[1] else acc

data, labels = parseData('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv')
x = feature(data)
y = np.array(labels)
x_train, x_test = x[:len(x) / 2], x[len(x) / 2:]
y_train, y_test = y[:len(y) / 2], y[len(y) / 2:]

wine_svm = svm.SVC()
wine_svm.fit(x_train, y_train)
predict_train = wine_svm.predict(x_train)
predict_test = wine_svm.predict(x_test)
correct_train = reduce(accuracy, zip(predict_train, y_train), 0)
correct_test = reduce(accuracy, zip(predict_test, y_test), 0)
print "train_accuracy", float(correct_train) / len(y_train)
print "test_accuracy", float(correct_test) / len(y_test)

x_valid, x_test = x[len(x) / 2: len(x) * 3 / 4], x[len(x) * 3 / 4:]
y_valid, y_test = y[len(y) / 2: len(y) * 3 / 4], y[len(y) * 3 / 4:]

# 7
# Iterate through different regularization levels and print the results
for reg_level in [10 ** power for power in xrange(-3, 4)]:
    wine_svm = svm.SVC(C=reg_level)
    wine_svm.fit(x_train, y_train)
    predict_train = wine_svm.predict(x_train)
    predict_valid = wine_svm.predict(x_valid)
    predict_test = wine_svm.predict(x_valid)
    correct_train = reduce(accuracy, zip(predict_train, y_train), 0)
    correct_valid = reduce(accuracy, zip(predict_valid, y_valid), 0)
    correct_test = reduce(accuracy, zip(predict_test, y_test), 0)
    print "Regularization:\t\t", reg_level
    print "Training error:\t\t", 1 - correct_train * 1.0 / len(y_train)
    print "Validation error:\t", 1 -correct_valid * 1.0 / len(y_valid)
    print "Testing error:\t\t", 1 - correct_test * 1.0 / len(y_test)
    print