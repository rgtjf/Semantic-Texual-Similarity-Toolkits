import gzip
import lasagne
import theano
import numpy as np
from theano import config
from time import time
from evaluate import evaluate, getSeqs
import cPickle
import sys, codecs, json


def checkIfQuarter(idx, n):
    if idx == round(n / 4.) or idx == round(n / 2.) or idx == round(3 * n / 4.):
        return True
    return False


def saveParams(model, fname):
    f = file(fname, 'wb')
    cPickle.dump(model.all_params, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()


def prepare_data(list_of_seqs):
    lengths = [len(s) for s in list_of_seqs]
    n_samples = len(list_of_seqs)
    maxlen = np.max(lengths)
    x = np.zeros((n_samples, maxlen)).astype('int32')
    x_mask = np.zeros((n_samples, maxlen)).astype(theano.config.floatX)
    for idx, s in enumerate(list_of_seqs):
        x[idx, :lengths[idx]] = s
        x_mask[idx, :lengths[idx]] = 1.
    x_mask = np.asarray(x_mask, dtype=config.floatX)
    return x, x_mask


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
        minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def getDataSim(batch, nout):
    g1 = [];
    g2 = []
    for i in batch:
        g1.append(i[0].embeddings)
        g2.append(i[1].embeddings)

    g1x, g1mask = prepare_data(g1)
    g2x, g2mask = prepare_data(g2)

    scores = []
    for i in batch:
        temp = np.zeros(nout)
        score = float(i[2])
        ceil, fl = int(np.ceil(score)), int(np.floor(score))
        if ceil == fl:
            temp[fl - 1] = 1
        else:
            temp[fl - 1] = ceil - score
            temp[ceil - 1] = score - fl
        scores.append(temp)
    scores = np.matrix(scores) + 0.000001
    scores = np.asarray(scores, dtype=config.floatX)
    return (scores, g1x, g1mask, g2x, g2mask)


def getDataEntailment(batch):
    g1 = [];
    g2 = []
    for i in batch:
        g1.append(i[0].embeddings)
        g2.append(i[1].embeddings)

    g1x, g1mask = prepare_data(g1)
    g2x, g2mask = prepare_data(g2)

    scores = []
    for i in batch:
        temp = np.zeros(3)
        label = i[2].strip()
        if label == "CONTRADICTION":
            temp[0] = 1
        if label == "NEUTRAL":
            temp[1] = 1
        if label == "ENTAILMENT":
            temp[2] = 1
        scores.append(temp)
    scores = np.matrix(scores) + 0.000001
    scores = np.asarray(scores, dtype=config.floatX)
    return (scores, g1x, g1mask, g2x, g2mask)


def getDataSentiment(batch):
    g1 = []
    for i in batch:
        g1.append(i[0].embeddings)

    g1x, g1mask = prepare_data(g1)

    scores = []
    for i in batch:
        temp = np.zeros(2)
        label = i[1].strip()
        if label == "0":
            temp[0] = 1
        if label == "1":
            temp[1] = 1
        scores.append(temp)
    scores = np.matrix(scores) + 0.000001
    scores = np.asarray(scores, dtype=config.floatX)
    return (scores, g1x, g1mask)


###ddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd####
#
# def early_stop(dp):
#     global stop_round
#     global bdp
#     if dp >= bdp:
#         bdp = dp
#         stop_round = 0
#     else:
#         stop_round = stop_round + 1
#     if stop_round > 3:
#         return True
#     return False
#
#
# bdp = 0.0
# stop_round = 0
#
#
# def save_params(file_name, params, **kwargs):
#     """
#         Save params to file_name.
#         params: a list of Theano variables
#     """
#     dic = {'params': [x.get_value() for x in params]}
#     dic.update(kwargs)
#     with gzip.open(file_name, "w") as save_file:
#         cPickle.dump(obj=dic, file=save_file, protocol=-1)
#
#
# def load_params(file_name):
#     """
#         Load params from file_name.
#     """
#     with gzip.open(file_name, "rb") as save_file:
#         dic = cPickle.load(save_file)
#     return dic


def test(model, f_test, words, params):
    start_time = time()
    try:
        f = open(f_test, 'r')
        lines = f.readlines()
        preds = []
        seq1 = []
        seq2 = []
        for i in lines:
            i = i.split("\t")
            p1 = i[0]
            p2 = i[1]
            X1, X2 = getSeqs(p1, p2, words)
            seq1.append(X1)
            seq2.append(X2)
        x1, m1 = prepare_data(seq1)
        x2, m2 = prepare_data(seq2)
        scores = model.scoring_function(x1, x2, m1, m2)
        embg1 = model.feedforward_function(x1, m1)
        embg2 = model.feedforward_function(x2, m2)
        embg1 = np.squeeze(embg1)
        embg2 = np.squeeze(embg2)
        preds = np.squeeze(scores)

        ''' write to file '''
        f_emb_name = f_test.replace('eval', 'features/'+params.nntype)
        print "Write to file %s" % f_emb_name
        with codecs.open(f_emb_name, 'w', encoding='utf8') as f_emb:
            for score, emb1, emb2 in zip(scores, embg1, embg2):
                print >> f_emb, json.dumps([float(score), emb1.tolist(), emb2.tolist()])

    except KeyboardInterrupt:
        print "Test interupted"

    end_time = time()
    print "total time:", (end_time - start_time)


###uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu####

def train(model, train_data, dev, test, train, words, params):
    start_time = time()

    try:
        if params.task == "sim":
            dp, ds = evaluate(model, words, dev, params)
            tp, ts = evaluate(model, words, test, params)
            rp, rs = evaluate(model, words, train, params)
            print "evaluation: ", dp, ds, tp, ts, rp, rs

        for eidx in xrange(params.epochs):

            kf = get_minibatches_idx(len(train_data), params.batchsize, shuffle=True)
            uidx = 0
            for _, train_index in kf:

                uidx += 1
                batch = [train_data[t] for t in train_index]

                for i in batch:
                    i[0].populate_embeddings(words)
                    if not params.task == "sentiment":
                        i[1].populate_embeddings(words)

                if params.task == "ent":
                    (scores, g1x, g1mask, g2x, g2mask) = getDataEntailment(batch)
                elif params.task == "sim":
                    (scores, g1x, g1mask, g2x, g2mask) = getDataSim(batch, model.nout)
                elif params.task == "sentiment":
                    (scores, g1x, g1mask) = getDataSentiment(batch)
                else:
                    raise ValueError('Task should be ent or sim.')

                if not params.task == "sentiment":
                    cost = model.train_function(scores, g1x, g2x, g1mask, g2mask)
                else:
                    cost = model.train_function(scores, g1x, g1mask)

                if np.isnan(cost) or np.isinf(cost):
                    print 'NaN detected'

                # print 'Epoch ', (eidx+1), 'Update ', (uidx+1), 'Cost ', cost

                # undo batch to save RAM
                for i in batch:
                    i[0].representation = None
                    if not params.task == "sentiment":
                        i[1].representation = None
                    i[0].unpopulate_embeddings()
                    if not params.task == "sentiment":
                        i[1].unpopulate_embeddings()

            if params.task == "sim":
                dp, ds = evaluate(model, words, dev, params)
                tp, ts = evaluate(model, words, test, params)
                rp, rs = evaluate(model, words, train, params)
                print "evaluation: ", dp, ds, tp, ts, rp, rs
            elif params.task == "ent" or params.task == "sentiment":
                ds = evaluate(model, words, dev, params)
                ts = evaluate(model, words, test, params)
                rs = evaluate(model, words, train, params)
                print "evaluation: ", ds, ts, rs
            else:
                raise ValueError('Task should be ent or sim.')

            print 'Epoch ', (eidx + 1), 'Cost ', cost
            sys.stdout.flush()

    except KeyboardInterrupt:
        print "Training interupted"

    end_time = time()
    print "total time:", (end_time - start_time)
