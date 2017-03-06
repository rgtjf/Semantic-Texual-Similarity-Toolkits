import theano
import numpy as np
from theano import tensor as T
from theano import config
import lasagne
from lasagne_average_layer import lasagne_average_layer

class WordModel(object):

    def __init__(self, We_initial, params):

        print "WordModel"

        initial_We = theano.shared(np.asarray(We_initial, dtype=config.floatX))
        We = theano.shared(np.asarray(We_initial, dtype=config.floatX))

        g1batchindices = T.imatrix()
        g2batchindices = T.imatrix()
        g1mask = T.matrix()
        g2mask = T.matrix()
        scores = T.matrix()

        l_in = lasagne.layers.InputLayer((None, None))
        l_mask = lasagne.layers.InputLayer(shape=(None, None))
        l_emb = lasagne.layers.EmbeddingLayer(l_in, input_size=We.get_value().shape[0], output_size=We.get_value().shape[1], W=We)
        l_out = lasagne_average_layer([l_emb, l_mask])

        embg1 = lasagne.layers.get_output(l_out, {l_in: g1batchindices, l_mask: g1mask})
        embg2 = lasagne.layers.get_output(l_out, {l_in: g2batchindices, l_mask: g2mask})

        g1_dot_g2 = embg1 * embg2
        g1_abs_g2 = abs(embg1 - embg2)

        lin_dot = lasagne.layers.InputLayer((None, We.get_value().shape[1]))
        lin_abs = lasagne.layers.InputLayer((None, We.get_value().shape[1]))
        l_sum = lasagne.layers.ConcatLayer([lin_dot, lin_abs])
        l_sigmoid = lasagne.layers.DenseLayer(l_sum, params.memsize, nonlinearity=lasagne.nonlinearities.sigmoid)

        l_softmax = lasagne.layers.DenseLayer(l_sigmoid, params.nout, nonlinearity=T.nnet.softmax)
        X = lasagne.layers.get_output(l_softmax, {lin_dot: g1_dot_g2, lin_abs: g1_abs_g2})
        Y = T.log(X)

        cost = scores * (T.log(scores) - Y)
        cost = cost.sum(axis=1) / (float(params.nout))

        prediction = 0.
        i = params.minval
        while i <= params.maxval:
            prediction = prediction + i * X[:, i - 1]
            i += 1


        self.network_params = lasagne.layers.get_all_params(l_out, trainable=True) + \
                              lasagne.layers.get_all_params(l_softmax, trainable=True)
        self.network_params.pop(0)

        self.all_params = lasagne.layers.get_all_params(l_out, trainable=True) + \
                          lasagne.layers.get_all_params(l_softmax, trainable=True)

        reg = self.getRegTerm(params, We, initial_We)
        self.trainable = self.getTrainableParams(params)
        cost = T.mean(cost) + reg

        self.feedforward_function = theano.function([g1batchindices, g1mask], embg1)
        self.scoring_function = theano.function([g1batchindices, g2batchindices, g1mask, g2mask], prediction)
        self.cost_function = theano.function([scores, g1batchindices, g2batchindices, g1mask, g2mask], cost)

        grads = theano.gradient.grad(cost, self.trainable)
        if params.clip:
            grads = [lasagne.updates.norm_constraint(grad, params.clip, range(grad.ndim)) for grad in grads]
        updates = params.learner(grads, self.trainable, params.eta)
        self.train_function = theano.function([scores, g1batchindices, g2batchindices,
                                               g1mask, g2mask], cost, updates=updates)


    def getRegTerm(self, params, We, initial_We):

        l2 = 0.0
        # l2 += 0.5 * params.LC * sum([lasagne.regularization.l2(x) for x in self.network_params])

        # l2 += 0.5 * params.LW * lasagne.regularization.l2(We - initial_We)
        return l2

    def getTrainableParams(self, params):
        update_word = True
        if update_word is True:
            return self.all_params
        else:
            return self.network_params


class WordL1Model(object):

    def __init__(self, We_initial, params):

        print "WordL1Model"

        initial_We = theano.shared(np.asarray(We_initial, dtype=config.floatX))
        We = theano.shared(np.asarray(We_initial, dtype=config.floatX))

        g1batchindices = T.imatrix()
        g2batchindices = T.imatrix()
        g1mask = T.matrix()
        g2mask = T.matrix()
        scores = T.matrix()

        l_in = lasagne.layers.InputLayer((None, None))
        l_mask = lasagne.layers.InputLayer(shape=(None, None))
        l_emb = lasagne.layers.EmbeddingLayer(l_in, input_size=We.get_value().shape[0], output_size=We.get_value().shape[1], W=We)
        l_out = lasagne_average_layer([l_emb, l_mask])

        embg1 = lasagne.layers.get_output(l_out, {l_in: g1batchindices, l_mask: g1mask})
        embg2 = lasagne.layers.get_output(l_out, {l_in: g2batchindices, l_mask: g2mask})

        def L2_norm(vec):
            return vec / np.sqrt((vec ** 2).sum() + 1e-4)

        embg1 = L2_norm(embg1)
        embg2 = L2_norm(embg2)

        gold = 0.
        i = params.minval
        while i <= params.maxval:
            gold = gold + i * scores[:, i - 1]
            i += 1

        dif = (embg1 - embg2).norm(L=1, axis=1)
        sim = T.exp(-dif)
        sim = T.clip(sim, 1e-7, 1-1e-7)
        gold = T.clip(gold/5.0, 1e-7, 1-1e-7)


        self.network_params = lasagne.layers.get_all_params(l_out, trainable=True)
        self.network_params.pop(0)

        self.all_params = lasagne.layers.get_all_params(l_out, trainable=True)

        reg = self.getRegTerm(params, We, initial_We)
        self.trainable = self.getTrainableParams(params)
        cost = T.mean((sim - gold) ** 2) + reg


        self.feedforward_function = theano.function([g1batchindices, g1mask], embg1)
        self.scoring_function = theano.function([g1batchindices, g2batchindices, g1mask, g2mask], sim)
        self.cost_function = theano.function([scores, g1batchindices, g2batchindices, g1mask, g2mask], cost)

        grads = theano.gradient.grad(cost, self.trainable)
        if params.clip:
            grads = [lasagne.updates.norm_constraint(grad, params.clip, range(grad.ndim)) for grad in grads]
        updates = params.learner(grads, self.trainable, params.eta)
        self.train_function = theano.function([scores, g1batchindices, g2batchindices,
                                               g1mask, g2mask], cost, updates=updates)


    def getRegTerm(self, params, We, initial_We):

        l2 = 0.0
        l2 += 0.5 * params.LC * sum([lasagne.regularization.l2(x) for x in self.network_params])

        l2 += 0.5 * params.LW * lasagne.regularization.l2(We - initial_We)
        return l2

    def getTrainableParams(self, params):
        if params.updatewords is True:
            return self.all_params
        else:
            return self.network_params
