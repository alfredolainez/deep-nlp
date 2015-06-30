import pickle

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier

TRAIN_FILE = "TrainSet_147444"
DEV_FILE = "DevSet_147444"
TEST_FILE = "TestSet_147444"

with open(TRAIN_FILE) as file:
    [train_reviews, train_labels] = pickle.load(file)
with open(DEV_FILE) as file:
    [dev_reviews, dev_labels] = pickle.load(file)
with open(TEST_FILE) as file:
    [test_reviews, test_labels] = pickle.load(file)

ngram_range = (1, 1)
count_vect = CountVectorizer(ngram_range=ngram_range, stop_words="english", max_features=5000)
X_train_counts = count_vect.fit_transform(train_reviews)
tfidf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts)
X = tfidf_transformer.transform(X_train_counts)

print "Training"
#classifier = LinearSVC().fit(X, train_labels)
#classifier = SVC().fit(X, train_labels)
classifier = RandomForestClassifier().fit(X.toarray(), train_labels)
train_accuracy = classifier.score(X.toarray(), train_labels)
print "Trained with %f of accuracy" % train_accuracy

X_dev_counts = count_vect.transform(dev_reviews)
X_dev_tfidf = tfidf_transformer.transform(X_dev_counts)
dev_accuracy = classifier.score(X_dev_tfidf.toarray(), dev_labels)
print "Dev accuracy: " + str(dev_accuracy)