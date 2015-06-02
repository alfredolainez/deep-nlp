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

WORDVECTOR_LENGTH = 30
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


WORD2VEC_MODEL = "w2v_70_parts_30_vector_4_window"
PARTITIONS_TRAINING = range(1, 15)
PARTITIONS_TESTING = range(20,22)

model = Word2Vec.load(WORD2VEC_MODEL)

reviews_train, _, funny_votes_train, _, _ = BaseBowRegressor.get_reviews_data(PARTITIONS_TRAINING)

count = 0
for votes in funny_votes_train:
    if votes > 0:
        count += 1

print "Total non-zero votes: %d of %d" % (count, len(funny_votes_train))


print "Tokenizing"
NUM_ELEMENTS_TRAIN = None
NUM_ELEMENTS_TEST = None
reviews_tokens_train = [language.tokenize_document(txt) for txt in enumerate(reviews_train[:NUM_ELEMENTS_TRAIN])]

X_train = tokens_to_word_vectors(reviews_tokens_train, model)

reviews_tokens_train = None
reviews_train = None
gc.collect()

X_train = np.array(X_train)

y_train = np.array(funny_votes_train[:NUM_ELEMENTS_TRAIN]).astype('float32')

maxlen = 100 # cut texts after this number of words
batch_size = 32

print("Pad sequences (samples x time)")
X_train = pad_sequence_word_vectors(X_train, maxlen=maxlen)
print('X_train shape:', X_train.shape)


print('Build model...')
model = Sequential()
#model.add(Embedding(max_features, 256))
model.add(GRU(WORDVECTOR_LENGTH, 128)) # or lstm
model.add(Dropout(0.5))
model.add(Dense(128, 1))
model.add(Activation('relu'))

# try using different optimizers and different optimizer configs
rms = RMSprop()
model.compile(loss=rmslog_loss, optimizer=rms)

print("Train...")
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=15, validation_split=0.1, show_accuracy=True)

# Load test material
print "LOADING TEST DATA"
reviews_test, _, funny_votes_test, _, _ = BaseBowRegressor.get_reviews_data(PARTITIONS_TESTING)
reviews_tokens_test = [language.tokenize_document(txt) for txt in enumerate(reviews_test[:NUM_ELEMENTS_TEST])]
X_test = tokens_to_word_vectors(reviews_tokens_test, model)
X_test = np.array(X_test)
y_test = np.array(funny_votes_test[:NUM_ELEMENTS_TEST]).astype('float32')
print "Padding test sequences"
X_test = pad_sequence_word_vectors(X_test, maxlen=maxlen)
print('X_test shape:', X_test.shape)

score = model.evaluate(X_test, y_test, batch_size=batch_size)
print('Test score:', score)

predicted = model.predict(X_test, batch_size=batch_size)
print mean_absolute_error(predicted, y_test)
print mean_absolute_error(np.zeros(len(predicted)), y_test)
print "RMSLOG error test: " + str(rmslog_error(predicted, y_test))
print "RMSLOG on zeros: " + str(rmslog_error(np.zeros(len(predicted)), y_test))

model.save_weights("GRU_128_DROPOUT_0.1_RELU_50_epochs_moredata")