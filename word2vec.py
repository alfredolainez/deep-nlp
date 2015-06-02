from gensim.models.word2vec import Word2Vec
from language import detect_language
from multiprocessing import Pool
# import nltk
import numpy as np
from spacy.en import English
from regression import BaseBowRegressor
from functools import partial
from nltk import word_tokenize


# better tokenizer
nlp = English()

NUM_PARTITIONS = 30

FILTER_ENGLISH = False # -- set to true for real code, its just super fuckin slow.

reviews_texts, useful_votes, funny_votes, cool_votes, review_stars = BaseBowRegressor.get_reviews_data(range(1, NUM_PARTITIONS))

def tokenize_document(docpair, use_nltk=True):
    print 'working on doc {}'.format(docpair[0])
    if not use_nltk:
        if FILTER_ENGLISH:
            return [x.lower_.encode('ascii',errors='ignore') for x in nlp(docpair[1]) if detect_language(x) == 'english']
        return [x.lower_.encode('ascii',errors='ignore') for x in nlp(docpair[1])]
    else:
        if FILTER_ENGLISH:
            return [x.encode('ascii',errors='ignore').lower() for x in word_tokenize(docpair[1]) if detect_language(x) == 'english']
        return [x.encode('ascii',errors='ignore').lower() for x in word_tokenize(docpair[1])]


def parallel_run(f, parms):
    '''
    performs in-core map reduce of the function `f`
    over the parameter space spanned by parms.

    `f` MUST take only one argument. 
    '''
    pool = Pool()
    ret = pool.map(f, parms)
    pool.close(); pool.join()
    return ret

# -- run shit in parallel...
# sentences = parallel_run(tokenize_document, enumerate(reviews_texts))
sentences = [tokenize_document(txt) for txt in enumerate(reviews_texts)]

# build a default w2v model...
w2v = Word2Vec(sentences=sentences, size=100, alpha=0.025, window=4, min_count=2, sample=1e-5, workers=4, negative=10)


def tokens_to_mean_vec(tokens, w2v):
    '''
    Takes a list of tokens and a Word2Vec models
    and finds the mean word vector of that list.
    '''
    vec = []
    for w in tokens:
        try:
            vec.append(w2v[w])
        except KeyError:
            continue
    if len(vec) == 0:
        # -- shit! a sentence with no recognized tokens
        vec.append(np.zeros(w2v[w2v.vocab.keys()[1]].shape))
    return np.array(vec).mean(axis=0)


# make a mean vector for every datapoint
data = []
for i, txt in enumerate(sentences):
    print '{} of {}'.format(i, len(sentences))
    data.append(tokens_to_mean_vec(txt, w2v))

X = np.array(data)

X = X.astype('float32')

Y = np.array(funny_votes).astype('float32')
Y = np.log(Y + 1)

from keras.models import Sequential
from keras.layers.core import MaxoutDense, Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop, Adagrad

model_basic = Sequential()
model_basic.add(MaxoutDense(100, 100, 20))
model_basic.add(Activation('relu'))
model_basic.add(Dropout(0.2))

model_basic.add(MaxoutDense(100, 20, 10))
model_basic.add(Activation('relu'))
model_basic.add(Dropout(0.2))

model_basic.add(MaxoutDense(20, 1))
model_basic.add(Activation('relu'))
# model_basic.add(Dropout(0.1))

# model_basic.add(Dense(10, 1))
# model_basic.add(Activation('relu'))

ada = Adagrad()
# rms = RMSprop()
model_basic.compile(loss='mse', optimizer=ada)




model_basic.fit(X[:30000], Y[:30000], batch_size=10, nb_epoch=15)








def tokens_to_vecs(tokens, w2v):
    vec = []
    for w in tokens:
        try:
            vec.append(w2v[w].astype('float32'))
        except KeyError:
            continue
    return np.array(vec)


lengths = [len(s) for s in sentences]
MAX_SEQ_LEN = int(np.percentile(lengths, 75))

BATCH = 400

X = np.zeros((BATCH, MAX_SEQ_LEN, 100)).astype('float32')
Y_sub = np.zeros((BATCH, 1)).astype('float32')

for i in range(BATCH):
    ix = random.sample(range(len(sentences)), 1)[0]
    seq = tokens_to_vecs(sentences[ix], w2v)[:MAX_SEQ_LEN]
    X[i, :len(seq), :] = seq
    Y_sub[i] = Y_trans[ix]


def generate_batch(sentences, targets, w2v, batchsize=400, max_len=None):
    '''
    sentences is a list of lists of tokens, targets is a list / array of target values
    '''
    if max_len is None:
        lengths = [len(s) for s in sentences]
        max_len = int(np.percentile(lengths, 75))

    X = np.zeros((batchsize, max_len, w2v.layer1_size)).astype('float32')
    Y = np.zeros((batchsize, 1)).astype('float32')

    for i in range(batchsize):
        ix = random.sample(range(len(sentences)), 1)[0]
        seq = tokens_to_vecs(sentences[ix], w2v)[:max_len]
        X[i, :len(seq), :] = seq
        Y[i] = targets[ix]
    return X, Y

def generate_batches(n_batches, sentences, targets, w2v, batchsize=400, max_len=None):
    '''
    sentences is a list of lists of tokens, targets is a list / array of target values
    '''
    if max_len is None:
        lengths = [len(s) for s in sentences]
        max_len = int(np.percentile(lengths, 75))

    X = np.zeros((batchsize, max_len, w2v.layer1_size)).astype('float32')
    Y = np.zeros((batchsize, 1)).astype('float32')
    for _ in xrange(n_batches):
        for i in range(batchsize):
            ix = random.sample(range(len(sentences)), 1)[0]
            seq = tokens_to_vecs(sentences[ix], w2v)[:max_len]
            X[i, :len(seq), :] = seq
            Y[i] = targets[ix]
        yield X, Y


X_train, Y_train = generate_batch(sentences, Y_trans, w2v, 500, 190)



for X_train, Y_train in generate_batches(10, sentences, Y_trans, w2v, 500, 190):
    print 'New batch!'
    _ = model.fit(X_train, Y_train, batch_size=10, nb_epoch=1, show_accuracy = True)



data = []
for i, txt in enumerate(sentences):
    print '{} of {}'.format(i, len(sentences))
    X[i, :lengths[i], :] = tokens_to_vecs(txt, w2v)
    data.append()

data = np.array(data)

nb_samples = len(data)







X = np.empty((len(data), 100))

for i, x in enumerate(data):
    X[i,:] = x


ix = np.where(np.isnan(X).sum(axis=1) == 0)[0]

X = X[ix].astype('float32')
Y_trans = np.array(funny_votes).astype('float32')
Y_trans = np.log(Y_trans + 1)

Y_trans = 1.0 * (Y_trans > 0)


Y_trans = np.array(useful_votes)[ix].astype('float32')
Y_trans = np.log(Y_trans + 1)




from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.regularizers import l2, l1
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils




model2 = Sequential()
model2.add(Dense(100, 75))
model2.add(Activation('relu'))
model2.add(Dropout(0.1))

model2.add(Dense(75, 45, W_regularizer = l2(.0001)))
model2.add(Activation('relu'))
model2.add(Dropout(0.05))

model2.add(Dense(45, 20, W_regularizer = l2(.0001)))
model2.add(Activation('relu'))
model2.add(Dropout(0.05))

model2.add(Dense(20, 10, W_regularizer = l2(.0001)))
model2.add(Activation('relu'))
model2.add(Dropout(0.05))

model2.add(Dense(10, 1, W_regularizer = l2(.0001)))
model2.add(Activation('relu'))

rms = RMSprop()
model2.compile(loss='mse', optimizer=rms)


model2.fit(X, Y_trans, batch_size=15, nb_epoch=15)



from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Merge, MaxoutDense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, SimpleDeepRNN



model = Sequential()

model.add(LSTM(100, 100))
model.add(Dropout(0.2))

model.add(Dense(100, 50))
model.add(Activation('relu'))

model.add(Dropout(0.2))

model.add(Dense(50, 1))
model.add(Activation('sigmoid'))

ada = Adagrad()
model.compile(loss='binary_crossentropy', optimizer=ada)


model.fit(X, Y_trans, batch_size=2, nb_epoch=1)

# -- 0.4413

from regression import BaseBowRegressor
from multiprocessing import Pool
reviews_texts, useful_votes, funny_votes, cool_votes, review_stars = BaseBowRegressor.get_reviews_data(range(1,20))
sentences = []

from spacy.en import English
nlp = English()

def tokenize_document(docpair):
    print 'working on doc {}'.format(docpair[0])
    return [x.lower_.encode('ascii',errors='ignore') for x in nlp(docpair[1])]

def parallel_run(f, parms):
    '''
    performs in-core map reduce of the function `f`
    over the parameter space spanned by parms.

    `f` MUST take only one argument. 
    '''
    pool = Pool()
    ret = pool.map(f, parms)
    pool.close(); pool.join()
    return ret


sentences = parallel_run(tokenize_document, enumerate(reviews_texts))






print "Tokenizing sentences..."
for i, review in enumerate(reviews_texts):
    print '{} of {}'.format(i, len(reviews_texts))
    sentences.append([x.lower_.encode('ascii',errors='ignore') for x in nlp(review)])


from keras.preprocessing.text import Tokenizer

tk = Tokenizer()

tk.fit_on_texts((t.encode('ascii',errors='ignore') for t in reviews_texts))

tk.fit_on_texts((t.encode('ascii',errors='ignore') for t in sentences))


seq_data = [_ for _ in tk.texts_to_sequences_generator((t.encode('ascii',errors='ignore') for t in reviews_texts))]




cPickle.dump({'funny' : funny_votes, 
'useful' : useful_votes, 
'stars' : review_stars, 
'partition_range' : 'range(1, 20)', 
'sequenced_data' : seq_data, 
'meta' : 'Yelp data over the partitions 1 thru 19. sequenced_data is an embedding from the Keras Tokenizer'}, 
open('data-dump-1-19.pkl', 'wb'), cPickle.HIGHEST_PROTOCOL)



X = sequence.pad_sequences(seq_data, maxlen=250)



model = Sequential()
model.add(Embedding(271835, 100))
# model.add(GRU(100, 128))
model.add(SimpleDeepRNN(100, 100))
# model.add(Dropout(0.5))
model.add(Dense(100, 64))
model.add(Dropout(0.1))
model.add(Dense(64, 1))
model.add(Activation('relu'))

# try using different optimizers and different optimizer configs

ada = Adagrad()

model.compile(loss='mse', optimizer=ada)



model.fit(X[:10000], Y[:10000], batch_size=120, nb_epoch=5)


model.fit(X[:400000], Y[:400000], batch_size=3000, nb_epoch=5)

for it, (seq, label) in enumerate(zip(seq_data, Y_trans)):
    if it % 10 == 0:
        print 'Iteration: {}'.format(it)
    model.train(np.array([seq]), [label])



io.save('./yelp-datafile-1-30.h5', {'funny' : np.array(funny_votes), 
              'useful' : np.array(useful_votes), 
              'stars' : np.array(review_stars), 
              'sequenced_data' : seq_data, 
              'padded_data' : X,
              'meta' : 'Yelp data over the partitions 1 thru 29. sequenced_data is an embedding from the Keras Tokenizer'})







