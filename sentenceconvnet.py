from collections import defaultdict, OrderedDict
import cPickle
import re
import sys
import warnings

# from keras.optimizers import Adagrad, Adam, RMSprop, SGD
import numpy as np
import theano
import theano.tensor as T

from baselayers import LeNetConvPoolLayer, MLPDropout




warnings.filterwarnings("ignore")   



def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)
def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)
def Tanh(x):
    y = T.tanh(x)
    return(y)
def Iden(x):
    y = x
    return(y)



class SentenceCNN(object):
    """docstring for SentenceCNN"""
    def __init__(self, U, sentence_dim=200, wv_dim = 100, ngram_filters=[3, 4, 5], dropout=[0.5], hidden=[100, 1], activations=[ReLU, ], batch_size=50):
        super(SentenceCNN, self).__init__()

        self.rng = np.random.RandomState(3435)

        # -- U is a matrix with U.shape = (size_vocab, wv_dim)
        self.U = U

        # -- the maximum sentence length...
        self.sentence_dim = sentence_dim
        self.wv_dim = wv_dim

        # -- list of filter sizes we want to consider...
        self.ngram_filters = ngram_filters
        self.dropout = dropout
        self.hidden = hidden
        self.batch_size = batch_size

        self.feature_maps = hidden[0]

        filter_shapes = []
        pool_sizes = []

        # -- get the conv parameters from each ngram...
        for ngram in ngram_filters:
            # -- we want to look at (ngram, wv_dim) sized patches...
            filter_shapes.append((self.feature_maps, 1, ngram, wv_dim))
            pool_sizes.append((sentence_dim - ngram + 1, img_w - wv_dim + 1))

        # -- define the model architecture

        # -- this is the index of the dataset...
        self.index = T.lscalar()

        # -- this is the matrix of indices for words in a sentence...
        self.x = T.matrix('x')   

        # -- this is the vector of target values...
        self.y = T.ivector('y')

        # -- initialize our wordvectors!
        self.Words = theano.shared(value = self.U, name = "Words")


        self.zero_vec_tensor = T.vector()
        self.zero_vec = np.zeros(img_w)
        self.set_zero = theano.function([zero_vec_tensor], updates=[(Words, T.set_subtensor(Words[0,:], zero_vec_tensor))])
    
        # -- make the actual image from the word vectors!
        self.sentence_image = Words[T.cast(x.flatten(),dtype="int32")].reshape((x.shape[0], 1, x.shape[1], Words.shape[1]))                                  
        
        # -- make our model split
        self.conv_layers = []
        self.conv_output_buffer = []

        CONVOLUTION_NONLINEARITY = 'relu'

        STATIC_WV = False

        for i in xrange(len(ngram_filters)):
            # -- get the filter sizes for this particular ngram filter.
            filter_shape = filter_shapes[i]
            pool_size = pool_sizes[i]

            conv_layer = LeNetConvPoolLayer(rng=self.rng, 
                                            input=sentence_image,
                                            image_shape=(batch_size, 1, sentence_dim, wv_dim),
                                            filter_shape=filter_shape, 
                                            poolsize=pool_size, 
                                            non_linear=CONVOLUTION_NONLINEARITY)

            conv_out = conv_layer.output.flatten(2)

            # -- concatenate this stuff
            self.conv_layers.append(conv_layer)
            self.conv_output_buffer.append(conv_out)


        # -- convert the parallel outputs into a tensor!
        self.conv_outputs = T.concatenate(conv_output_buffer, 1)

        # -- we need to flatten them output!
        self.hidden[0] = self.feature_maps * len(ngram_filters)    

        self.fully_connected = MLPDropout(self.rng, input=self.conv_outputs, layer_sizes=self.hidden, activations=activations, dropout_rates=dropout_rate, classifier=False)
        
        # -- define parameters of the model and update functions using adadelta
        self.params = self.fully_connected.params     
        for conv_layer in self.conv_layers:
            self.params += conv_layer.params
        if not STATIC_WV:
            #if word vectors are allowed to change, add them as model parameters
            self.params += [self.Words]

        lr_decay = 0.95
        sqr_norm_lim = 9
        
        # now, need to hack away at th MLP class...    
        self.cost = self.fully_connected.cost(y) 
        self.dropout_cost = self.fully_connected.dropout_cost(y)           
        self.grad_updates = sgd_updates_adadelta(params, dropout_cost, lr_decay, 1e-6, sqr_norm_lim) 

    def fit(self, X, y, validation):
        '''
        `validation` is a *mandatory* tuple of 
        length 2, with elements (X_val, y_val)
        '''

        X_val, y_val = validation

        np.random.seed(3435)

        batch_size = self.batch_size

        if X.shape[0] % batch_size > 0:
            extra_data_num = batch_size - X.shape[0] % batch_size

            # -- get random ix from first dimension
            shuffle_ix = np.random.permutation(X.shape[0])    
            extra_X = X[ix[:extra_data_num]]
            extra_y = y[ix[:extra_data_num]]

            X_training = np.append(X,extra_X,axis=0)
            y_training = np.append(y,extra_y,axis=0)
        else:
            X_training = X
            y_training = y

        # -- shuffle the training data
        shuffle_ix = np.random.permutation(X.shape[0]) 

        X_training = X_training[shuffle_ix]
        y_training = y_training[shuffle_ix]

        # -- find the number of batches we can train on...
        n_batches = X_training.shape[0] / batch_size
        n_train_batches = int(np.round(n_batches*0.9))
        
        #divide train set into train/val sets 
        val_set_x = X_val 
        val_set_y = y_val  

        # -- get our training and dev sets...
        train_set_x, train_set_y = shared_dataset(
                X_training[:n_train_batches * batch_size, :], 
                y_training[:n_train_batches * batch_size, :]
            )

        dev_set_x, dev_set_y = shared_dataset(
                X_training[n_train_batches*batch_size:, :], 
                y_training[n_train_batches*batch_size:, :]
            )

        # train_set_x, train_set_y = shared_dataset((train_set[:,:img_h],train_set[:,-1]))
        # dev_set_x, dev_set_y = shared_dataset((dev_set[:,:img_h],dev_set[:,-1]))

        n_dev_batches = n_batches - n_train_batches

        #compile theano functions to get train/val/test errors

        # -- gets the error on the dev set...
        validate_model = theano.function([self.index], fully_connected.cost(y),
                givens = {
                    x: dev_set_x[self.index * batch_size: (self.index + 1) * batch_size],
                    y: dev_set_y[self.index * batch_size: (self.index + 1) * batch_size]
                }
            )

        # -- gets the error on the training set...
        test_model = theano.function([self.index], fully_connected.cost(y),
                givens = {
                    x: train_set_x[self.index * batch_size: (self.index + 1) * batch_size],
                    y: train_set_y[self.index * batch_size: (self.index + 1) * batch_size]
                }
            )       
        
        # -- actually trains the model!        
        train_model = theano.function([self.index], self.cost, updates=grad_updates,
              givens={
                x: train_set_x[self.index*batch_size:(self.index+1)*batch_size],
                y: train_set_y[self.index*batch_size:(self.index+1)*batch_size]}) 



        test_pred_layers = []
        test_size = val_set_x.shape[0]
        test_layer0_input = self.Words[T.cast(x.flatten(),dtype="int32")].reshape((test_size,1,self.sentence_dim,self.Words.shape[1]))
        
        for conv_layer in self.conv_layers:
            test_layer0_output = conv_layer.predict(test_layer0_input, test_size)
            test_pred_layers.append(test_layer0_output.flatten(2))
        test_layer1_input = T.concatenate(test_pred_layers, 1)
        test_y_pred = self.fully_connected.predict(test_layer1_input)
        test_error = self.fully_connected.cost(y)

        # -- function to test model.
        test_model_all = theano.function([x,y], test_error)   
        
        #start training over mini-batches
        print '... training'
        epoch = 0
        best_val_perf = 0
        val_perf = 0
        test_perf = 0       
        cost_epoch = 0   
        shuffle_batch = True 
        while (epoch < n_epochs):        
            epoch = epoch + 1
            if shuffle_batch:
                for minibatch_index in np.random.permutation(range(n_train_batches)):
                    cost_epoch = train_model(minibatch_index)
                    self.set_zero(zero_vec)
            else:
                for minibatch_index in xrange(n_train_batches):
                    cost_epoch = train_model(minibatch_index)  
                    self.set_zero(zero_vec)
            train_losses = [test_model(i) for i in xrange(n_train_batches)]
            train_perf = np.mean(train_losses)
            val_losses = [validate_model(i) for i in xrange(n_dev_batches)]
            val_perf = np.mean(val_losses)                        
            print('epoch %i, train perf %f %%, val perf %f' % (epoch, train_perf, val_perf))
            if val_perf <= best_val_perf:
                best_val_perf = val_perf
                test_loss = test_model_all(val_set_x,val_set_y)        
                test_perf = test_loss         
        return test_perf




def sgd_updates_adadelta(params,cost,rho=0.95,epsilon=1e-6,norm_lim=9,word_vec_name='Words'):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) and (param.name!='Words'):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param      
    return updates 
        


def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, theano.config.floatX)


def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)        