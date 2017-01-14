import numpy as np
import urllib
import scipy.optimize
import random

def parseData(fname):
	for l in urllib.urlopen(fname):
		yield eval(l)

# def parseData(fname):
# 	with open(fname) as f:
# 		for _ in xrange(50000):
# 			yield eval(next(f))

data = list(parseData("http://jmcauley.ucsd.edu/cse190/data/beer/beer_50000.json"))
# data = list(parseData("beer_50000.json"))

# 1
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

years, counts, avg = average(data)

print "years", years
print "counts", counts
print "average", avg

# 2
x = np.matrix([[1, review['review/timeStruct']['year']] for review in data])
y = np.matrix([review['review/overall'] for review in data])
theta, residuals, rank, s = np.linalg.lstsq(x, y.T)
print "MSE", residuals / len(data)

def feature(review, count):
    matrix = [1]
    for degree in xrange(count):
        matrix.append(review['review/timeStruct']['year'] ** (degree + 1))
    return matrix

# 3
x = np.matrix([[1] + feature(review, 10) for review in data])
y = np.matrix([review['review/overall'] for review in data])
theta, residuals, rank, s = np.linalg.lstsq(x, y.T)
diff = np.dot(x, theta) - y.T
print "MSE", np.dot(diff.T,diff) / len(data)