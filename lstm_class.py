from gensim.models.word2vec import Word2Vec
from regression import BaseBowRegressor
import language
import gc

from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU

from evaluation import rmslog_error, rmslog_loss

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# TODO : Add more reviews with funny votes

WORDVECTOR_LENGTH = 100
UNKNOWN_VECTOR = np.zeros(WORDVECTOR_LENGTH) # OJO CON ESTO!

def tokens_to_word_vectors(reviews_tokens, w2v_model):
    X = []
    for review in reviews_tokens:
        wvs = []
        for token in review:
            if token in w2v_model:
                wvs.append(w2v_model[token])
            else:
                wvs.append(UNKNOWN_VECTOR)
        X.append(wvs)

    return X

def pad_sequence_word_vectors(sequences, maxlen=None):
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    x = np.zeros((nb_samples, maxlen, WORDVECTOR_LENGTH)).astype('float32')
    for idx, s in enumerate(sequences):
        x[idx, :lengths[idx]] = s[:maxlen]
    return x

#### Version using numpy, some problems with memory usage
#
# def give_balanced_classes(reviews, funny_votes):
#
#     reviews = np.array(reviews)
#     funny_votes = np.array(funny_votes)
#
#     funny_indices = np.where(funny_votes > 1)[0]
#     not_funny_indices = np.where(funny_votes == 0)[0]
#
#     funny_reviews = reviews[funny_indices]
#     not_funny_reviews = reviews[not_funny_indices]
#     # same length
#     not_funny_reviews = not_funny_reviews[:len(funny_reviews)]
#
#     reviews = np.concatenate((funny_reviews, not_funny_reviews))
#     labels = np.array([1]*len(funny_reviews) + [0]*len(not_funny_reviews))
#
#     # Randomly permute both reviews and labels
#     perm = np.random.permutation(len(reviews))
#     reviews = reviews[perm]
#     labels = labels[perm]
#
#     print "Returning %d funny reviews and %d boring reviews" % (len(funny_reviews), len(not_funny_reviews))
#
#     return (reviews, labels)

# Number of votes to consider a review funny enough
VOTES_THRESHOLD = 2

def give_balanced_classes(reviews, funny_votes):

    funny_reviews = []
    not_funny_reviews_indices = []

    # Find all the funny reviews we can
    final_reviews = []
    final_labels = []
    for i, review in enumerate(reviews):
        if funny_votes[i] >= VOTES_THRESHOLD:
            final_reviews.append(review)
            final_labels.append(1)
        elif funny_votes[i] == 0:
            not_funny_reviews_indices.append(i)

    # We want balanced classes so take same number
    np.random.shuffle(not_funny_reviews_indices)
    num_funny_reviews = len(final_reviews)
    for i in range(num_funny_reviews):
        final_reviews.append(reviews[not_funny_reviews_indices[i]])
        final_labels.append(0)

    # Shuffle final reviews and labels
    combined_lists = zip(final_reviews, final_labels)
    np.random.shuffle(combined_lists)
    final_reviews[:], final_labels[:] = zip(*combined_lists)

    print "Returning %d funny reviews and a total of %d reviews" % (num_funny_reviews, len(final_reviews))

    return (final_reviews, final_labels)


WORD2VEC_MODEL = "w2v_70_parts_100_vector_4_window"
PARTITIONS_TRAINING = range(1, 30) #15
PARTITIONS_TESTING = range(50, 53) #22

w2vmodel = Word2Vec.load(WORD2VEC_MODEL)

reviews_train, _, funny_votes_train, _, _ = BaseBowRegressor.get_reviews_data(PARTITIONS_TRAINING)
reviews_train, labels_train = give_balanced_classes(reviews_train, funny_votes_train)

print "Tokenizing"
NUM_ELEMENTS_TRAIN = None
NUM_ELEMENTS_TEST = None
reviews_tokens_train = [language.tokenize_document((i, unicode(txt))) for (i, txt) in enumerate(reviews_train[:NUM_ELEMENTS_TRAIN])]

X_train = tokens_to_word_vectors(reviews_tokens_train, w2vmodel)

reviews_tokens_train = None
reviews_train = None
gc.collect()

X_train = np.array(X_train)

labels_train = np.array(labels_train[:NUM_ELEMENTS_TRAIN])

# Load test material
print "LOADING TEST DATA"
reviews_test, _, funny_votes_test, _, _ = BaseBowRegressor.get_reviews_data(PARTITIONS_TESTING)
reviews_test, labels_test = give_balanced_classes(reviews_test, funny_votes_test)

reviews_tokens_test = [language.tokenize_document((i, unicode(txt))) for (i, txt) in enumerate(reviews_test[:NUM_ELEMENTS_TEST])]

reviews_test = None
gc.collect()

maxlen = 100 # cut texts after this number of words
batch_size = 16

print("Pad sequences (samples x time)")
X_train = pad_sequence_word_vectors(X_train, maxlen=maxlen)
print('X_train shape:', X_train.shape)


print('Build model...')
model = Sequential()
#model.add(Embedding(max_features, 256))
model.add(GRU(WORDVECTOR_LENGTH, 128)) # or lstm
model.add(Dropout(0.2))
model.add(Dense(128, 1))
model.add(Activation('sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")

print("Train...")
model.fit(X_train, labels_train, batch_size=batch_size, nb_epoch=15, validation_split=0.1, show_accuracy=True)


X_test = tokens_to_word_vectors(reviews_tokens_test, w2vmodel)
X_test = np.array(X_test)
labels_test = np.array(labels_test[:NUM_ELEMENTS_TEST])
print "Padding test sequences"
X_test = pad_sequence_word_vectors(X_test, maxlen=maxlen)
print('X_test shape:', X_test.shape)

score = model.evaluate(X_test, labels_test, batch_size=batch_size)
print('Test score:', score)

classes = model.predict_classes(X_test, batch_size=batch_size)
acc = np_utils.accuracy(classes, labels_test)
print('Test accuracy:', acc)

#model.save_weights("GRU_128_DROPOUT_0.1_RELU_50_epochs_moredata")