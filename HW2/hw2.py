import numpy
import urllib
import scipy.optimize
import random
from math import exp
from math import log
from sklearn.decomposition import PCA

random.seed(0)

print "Reading data..."
dataFile = open("HW2/winequality-white.csv")
header = dataFile.readline()
fields = ["constant"] + header.strip().replace('"','').split(';')
featureNames = fields[:-1]
labelName = fields[-1]
lines = [[1.0] + [float(x) for x in l.split(';')] for l in dataFile]

X = [l[:-1] for l in lines]
y = [l[-1] > 5 for l in lines]
print "done"

def inner(x,y):
  return sum([x[i]*y[i] for i in range(len(x))])

def sigmoid(x):
  return 1.0 / (1 + exp(-x))

##################################################
# Logistic regression by gradient ascent         #
##################################################

# NEGATIVE Log-likelihood
def f(theta, X, y, lam):
  loglikelihood = 0
  for i in range(len(X)):
    logit = inner(X[i], theta)
    loglikelihood -= log(1 + exp(-logit))
    if not y[i]:
      loglikelihood -= logit
  for k in range(len(theta)):
    loglikelihood -= lam * theta[k]*theta[k]
  # for debugging
  # print "ll =", loglikelihood
  return -loglikelihood

# NEGATIVE Derivative of log-likelihood
def fprime(theta, X, y, lam):
  dl = [0]*len(theta)
  for i in range(len(X)):
    logit = inner(X[i], theta)
    for k in range(len(theta)):
      dl[k] += X[i][k] * (1 - sigmoid(logit))
      if not y[i]:
        dl[k] -= X[i][k]
  for k in range(len(theta)):
    dl[k] -= lam*2*theta[k]
  return numpy.array([-x for x in dl])

X_train = X[:int(len(X)/3)]
y_train = y[:int(len(y)/3)]
X_validate = X[int(len(X)/3):int(2*len(X)/3)]
y_validate = y[int(len(y)/3):int(2*len(y)/3)]
X_test = X[int(2*len(X)/3):]
y_test = y[int(2*len(X)/3):]


##################################################
# Train                                          #
##################################################

def train(lam):
  theta,_,_ = scipy.optimize.fmin_l_bfgs_b(f, [0]*len(X[0]), fprime, pgtol = 10, args = (X_train, y_train, lam))
  return theta

##################################################
# Predict                                        #
##################################################

# 1
def performance(theta):
  scores_train = [inner(theta,x) for x in X_train]
  scores_validate = [inner(theta,x) for x in X_validate]
  scores_test = [inner(theta,x) for x in X_test]

  predictions_train = [s > 0 for s in scores_train]
  predictions_validate = [s > 0 for s in scores_validate]
  predictions_test = [s > 0 for s in scores_test]

  correct_train = [(a==b) for (a,b) in zip(predictions_train,y_train)]
  correct_validate = [(a==b) for (a,b) in zip(predictions_validate,y_validate)]
  correct_test = [(a==b) for (a,b) in zip(predictions_test,y_test)]
  
  acc_train = sum(correct_train) * 1.0 / len(correct_train)
  acc_validate = sum(correct_validate) * 1.0 / len(correct_validate)
  acc_test = sum(correct_test) * 1.0 / len(correct_test)
  return acc_train, acc_validate, acc_test

##################################################
# Validation pipeline                            #
##################################################

for lam in [0, 0.01, 1.0, 100.0]:
  theta = train(lam)
  acc_train, acc_validate, acc_test = performance(theta)
  print("lambda = " + str(lam) + ";\ttrain=" + str(acc_train) + "; validate=" + str(acc_validate) + "; test=" + str(acc_test))

# 2
shuffled = lines[:]
random.shuffle(shuffled)

X = [l[:-1] for l in shuffled]
y = [l[-1] > 5 for l in shuffled]

X_train = X[:int(len(X)/3)]
y_train = y[:int(len(y)/3)]
X_validate = X[int(len(X)/3):int(2*len(X)/3)]
y_validate = y[int(len(y)/3):int(2*len(y)/3)]
X_test = X[int(2*len(X)/3):]
y_test = y[int(2*len(X)/3):]

for lam in [0, 0.01, 1.0, 100.0]:
  theta = train(lam)
  acc_train, acc_validate, acc_test = performance(theta)
  print("lambda = " + str(lam) + ";\ttrain=" + str(acc_train) + "; validate=" + str(acc_validate) + "; test=" + str(acc_test))

X = [l[:-1] for l in lines]
y = [l[-1] > 5 for l in lines]

X_train = X[:int(len(X)/3)]
y_train = y[:int(len(y)/3)]
X_validate = X[int(len(X)/3):int(2*len(X)/3)]
y_validate = y[int(len(y)/3):int(2*len(y)/3)]
X_test = X[int(2*len(X)/3):]
y_test = y[int(2*len(X)/3):]

# 3
lam = 0.01
theta = train(lam)

# 3
def ber(theta):
    scores_test = [inner(theta,x) for x in X_test]
    predictions_test = [s > 0 for s in scores_test]
    
    tp, tn, fp, fn = 0.0, 0.0, 0.0, 0.0
    for prediction, label in zip(predictions_test, y_test):
        if prediction and label:
            tp += 1
        elif not prediction and not label:
            tn += 1
        elif prediction and not label:
            fp += 1
        else:
            fn += 1
    fpr, fnr = fp / (tp + fn), fn / (tn + fp)
    return tp, tn, fp, fn, 1 - (fpr + fnr) / 2

true_pos, true_neg, false_pos, false_neg, balanced_error_rate = ber(theta)
print "True positives", true_pos
print "True negatives", true_neg
print "False positives", false_pos
print "False negatives", false_neg
print "Balanced error rate", balanced_error_rate

# 4
def rank(theta, n):
    scores_test = [inner(theta,x) for x in X_test]
    ranking = sorted(zip(scores_test, y_test), reverse=True)
    relevant_retrieved = len([y for _, y in ranking[:n] if y])
    relevant = len([y for _, y in ranking if y])
    return float(relevant_retrieved) / n, float(relevant_retrieved) / relevant

for predictions in [10, 500, 1000]:
    precision, recall = rank(theta, predictions)
    print "Query size =", predictions, "\tPrecision =", precision, "\tRecall =", recall

# 5
mean = numpy.mean(X_train, axis=0)
compression = numpy.tile(mean, (len(X_train), 1))
diff = compression - numpy.array(X_train)
print sum([numpy.dot(d, d) for d in diff])

# 6
pca = PCA()
pca.fit(X_train)
print pca.components_

# 7
print numpy.dot(X_train[0], pca.components_.T)

# 8
pca = PCA(n_components=4)
pca.fit(X_train)
compressed = numpy.dot(X_train, pca.components_.T)
uncompressed = numpy.dot(compressed, pca.components_)
diff = numpy.array(X_train) - uncompressed
print sum([numpy.dot(d, d) for d in diff])