import numpy as np
import urllib
import scipy.optimize
import random
from math import pi, tan

import matplotlib.pyplot as plt

# def parseData(fname):
# 	for l in urllib.urlopen(fname):
# 		yield eval(l)

def parseData(fname):
	with open(fname) as f:
		for _ in xrange(500):
			yield eval(next(f))

def average(data):
	years = [review['review/timeStruct']['year'] for review in data]
	ratings = [review['review/overall'] for review in data]
	counts = [years.count(year) for year in list(set(years))]
	calc_avg = {}
	for year, rating in zip(years, ratings):
		if year not in calc_avg:
			calc_avg[year] = []
		calc_avg[year].append(rating)
	avg = [sum(calc_avg[year]) / len(calc_avg[year]) for year in calc_avg]
	return list(set(years)), counts, avg

# data = list(parseData("http://jmcauley.ucsd.edu/cse190/data/beer/beer_50000.json"))
data = list(parseData("beer_50000.json")) # TEST

# years, counts, avg = average(data)

# print years
# print counts
# print "average", avg

# # 2
# x = np.matrix([[1, review['review/timeStruct']['year']] for review in data])
# y = np.matrix([review['review/overall'] for review in data])
# theta, residuals, rank, s = np.linalg.lstsq(x, y.T)
# print "theta"
# print theta
# print residuals
# print "MSE", residuals / len(data)

# theta
# [[ -3.91707489e+01]
#  [  2.14379786e-02]]
# [[ 24502.19099291]]
# MSE [[ 0.49004382]]

def feature(review):
    matrix = []
    for degree in xrange(8):
        matrix.append(review['review/timeStruct']['year'] ** degree)
    return matrix

# 3
x = np.matrix([[1, review['review/timeStruct']['year'], review['review/timeStruct']['year'] ** 2, review['review/timeStruct']['year'] ** 3, review['review/timeStruct']['year'] ** 4] for review in data])
y = np.matrix([review['review/overall'] for review in data])
theta, residuals, rank, s = np.linalg.lstsq(x, y.T)
print "theta"
print theta
print "SSE"
print residuals
print "MSE", residuals / len(data)
diff=np.dot(np.array(x),np.array(theta))-np.array(y)
print np.dot(diff.T,diff)

# plt.plot(avg)
# plt.savefig("avg")
# plt.clf()