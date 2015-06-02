import data_handling

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import LinearSVC, SVR

from evaluation import rmslog_error

class BaseBowRegressor(object):

    # Items can be trained for different outputs
    FUNNY_VOTES = 0
    COOL_VOTES = 1
    USEFUL_VOTES = 2
    STARS = 3

    def __init__(self, ngram_range=(1,1)):
        self.ngram_range = ngram_range

        # Labels are given in groups since we can train the system using different
        # outputs. Labels, test labels and regressors are indexed according to the
        # categories given above
        self.reviews = []
        self.labels = [[], [], [], []]

        self.test_reviews = []
        self.test_labels = [[], [], [], []]

        self.regs = [None, None, None, None]

    @staticmethod
    def get_reviews_data(partitions_to_use):
        """
        Gets data from reviews and returns it so that the class can use it to load
        training or test data
        """

        data = data_handling.load_partitions(partitions_to_use)
        review_texts = []
        useful_votes = []
        funny_votes = []
        cool_votes = []
        review_stars = []

        for review in data:
            review_texts.append(review['text'])
            useful_votes.append(review['votes']['useful'])
            cool_votes.append(review['votes']['cool'])
            funny_votes.append(review['votes']['funny'])
            review_stars.append(review['stars'])

        return review_texts, useful_votes, funny_votes, cool_votes, review_stars

    def load_training_data(self, partitions_to_use):
        self.reviews, useful, funny, cool, stars =\
            self.get_reviews_data(partitions_to_use)

        self.labels[self.USEFUL_VOTES] = useful
        self.labels[self.FUNNY_VOTES] = funny
        self.labels[self.COOL_VOTES] = cool
        self.labels[self.STARS] = stars

    def load_test_data(self, partitions_to_use):
        """
        Loads test data into the object
        """
        self.test_reviews, test_useful, test_funny, test_cool, test_stars =\
            self.get_reviews_data(partitions_to_use)

        self.test_labels[self.USEFUL_VOTES] = test_useful
        self.test_labels[self.FUNNY_VOTES] = test_funny
        self.test_labels[self.COOL_VOTES] = test_cool
        self.test_labels[self.STARS] = test_stars

    def train(self):
        raise NotImplementedError()

    def __test(self, reviews, labels):
        raise NotImplementedError()

    def get_bag_of_ngrams(self, texts, ngram_range=None):
        """ Sets vectorizer feature and returns data from object in feature form X """
        if ngram_range is None:
            ngram_range = self.ngram_range
        self.count_vect = CountVectorizer(ngram_range=ngram_range, stop_words="english")
        X_train_counts = self.count_vect.fit_transform(texts)
        self.tfidf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts)
        X_train_tfidf = self.tfidf_transformer.transform(X_train_counts)

        return X_train_tfidf

DEFAULT_LABEL = BaseBowRegressor.FUNNY_VOTES

class SGD(BaseBowRegressor):
    """
    Stochastic Gradient Descent with Tfidf
    """
    def train(self, train_on=DEFAULT_LABEL, limit_data=None):
        if not hasattr(self, 'reviews'):
            print "No data loaded"
            return

        if limit_data is None:
            limit_data = len(self.reviews)

        X = self.get_bag_of_ngrams(self.reviews[:limit_data])
        self.regs[train_on] = SGDRegressor(loss="huber", alpha=0.0001, penalty="l1", n_iter=20).fit(X, self.labels[train_on][:limit_data])

    def __test(self, reviews, labels, test_on=DEFAULT_LABEL):
        X_training_counts = self.count_vect.transform(reviews)
        X_training_tfidf = self.tfidf_transformer.transform(X_training_counts)

        predicted = self.regs[test_on].predict(X_training_tfidf)

        return rmslog_error(predicted, labels), rmslog_error(np.zeros(len(predicted)), labels)

    def get_training_error(self, train_on=DEFAULT_LABEL):
        return self.__test(self.reviews, self.labels[train_on])

    def get_generalized_error(self, test_on=DEFAULT_LABEL):
        return self.__test(self.test_reviews, self.test_labels[test_on])

    def __get_scores(self, reviews, labels, train_on=DEFAULT_LABEL):
        X_training_counts = self.count_vect.transform(reviews)
        X_training_tfidf = self.tfidf_transformer.transform(X_training_counts)

        return self.regs[train_on].score(X_training_tfidf, labels)

    def get_training_R2(self, train_on=DEFAULT_LABEL):
        return self.__get_scores(self.reviews, self.labels[train_on], train_on)

    def get_test_R2(self, test_on=DEFAULT_LABEL):
        return self.__get_scores(self.test_reviews, self.test_labels[test_on], test_on)


class SupportVectorRegressor(BaseBowRegressor):
    """
    Stochastic Gradient Descent with Tfidf
    """
    def train(self, train_on=DEFAULT_LABEL, limit_data=None):
        if not hasattr(self, 'reviews'):
            print "No data loaded"
            return

        if limit_data is None:
            limit_data = len(self.reviews)

        X = self.get_bag_of_ngrams(self.reviews[:limit_data])
        self.regs[train_on] = SVR(kernel='rbf').fit(X, self.labels[train_on][:limit_data])

    def __test(self, reviews, labels, test_on=DEFAULT_LABEL):
        X_training_counts = self.count_vect.transform(reviews)
        X_training_tfidf = self.tfidf_transformer.transform(X_training_counts)

        predicted = self.regs[test_on].predict(X_training_tfidf)

        return rmslog_error(predicted, labels), rmslog_error(np.zeros(len(predicted)), labels)

    def get_training_error(self, train_on=DEFAULT_LABEL):
        return self.__test(self.reviews, self.labels[train_on])

    def get_generalized_error(self, test_on=DEFAULT_LABEL):
        return self.__test(self.test_reviews, self.test_labels[test_on])

    def __get_scores(self, reviews, labels, train_on=DEFAULT_LABEL):
        X_training_counts = self.count_vect.transform(reviews)
        X_training_tfidf = self.tfidf_transformer.transform(X_training_counts)

        return self.regs[train_on].score(X_training_tfidf, labels)

    def get_training_R2(self, train_on=DEFAULT_LABEL):
        return self.__get_scores(self.reviews, self.labels[train_on], train_on)

    def get_test_R2(self, test_on=DEFAULT_LABEL):
        return self.__get_scores(self.test_reviews, self.test_labels[test_on], test_on)

if __name__ == "__main__":
    # Examples
    #sgd = SGD()
    sgd = SupportVectorRegressor()
    sgd.load_training_data(range(1, 2))
    sgd.load_test_data(range(3,4))
    sgd.train()
    print sgd.get_training_error()
    print sgd.get_generalized_error()
    print sgd.get_training_R2()
    print sgd.get_test_R2()
