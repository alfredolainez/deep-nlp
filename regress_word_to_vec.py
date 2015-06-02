from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from regression import BaseBowRegressor
from evaluation import rmslog_error
from sklearn.ensemble import GradientBoostingRegressor

import nltk
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import LinearSVC, SVR

reviews_texts, useful_votes, funny_votes, cool_votes, review_stars = BaseBowRegressor.get_reviews_data(range(1, 30))
model = Doc2Vec.load("docvecs_70")

N = 50000
M = 100
X = np.zeros((N, M))
y = funny_votes[:N]
for i in range(N):
    if 'REVIEW_' + str(i) not in model:
        print str(i) + "not in model?"
        X[i,:] = np.zeros(M)
    else:
        X[i,:] = model['REVIEW_' + str(i)]

N_test = 10000
X_test = np.zeros((N_test, M))
y_test = funny_votes[N:N+N_test]
for i in range(N, N + N_test):
    if 'REVIEW_' + str(i) not in model:
        print str(i) + "not in model?"
        X_test[i-N,:] = np.zeros(M)
    else:
        X_test[i-N,:] = model['REVIEW_' + str(i)]

## SGD REGRESSOR
# sgd = SGDRegressor(loss="huber", alpha=0.001, penalty="l1", n_iter=20).fit(X, y)
# predicted = sgd.predict(X_test)
# print mean_absolute_error(predicted, y_test)
# print rmslog_error(predicted, y_test)
# print rmslog_error(np.zeros(len(predicted)), y_test)

## SUPPORT VECTOR REGRESSOR
# svm = SVR(kernel='rbf').fit(X, y)
# print "Trained!"
# predicted = svm.predict(X_test)
# print mean_absolute_error(predicted, y_test)
# print rmslog_error(predicted, y_test)
# print rmslog_error(np.zeros(len(predicted)), y_test)

## GRADIENT BOOSTING REGRESSOR
est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0, loss='ls').fit(X, y)
print "Trained!"
predicted = est.predict(X_test)
print mean_absolute_error(predicted, y_test)
print mean_absolute_error(np.zeros(len(predicted)), y_test)
print rmslog_error(predicted, y_test)
print rmslog_error(np.zeros(len(predicted)), y_test)