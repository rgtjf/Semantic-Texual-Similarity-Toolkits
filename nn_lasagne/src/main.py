import lasagne
import numpy as np
import pyprind
import random
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import argparse
import utils
from similarity.word_model import WordModel
from similarity.word_model import WordL1Model
from similarity.proj_model import ProjModel
from similarity.proj_model import ProjL1Model
from similarity.lstm_model import LSTMModel
from similarity.lstm_model import LSTML1Model

parser = argparse.ArgumentParser()
parser.add_argument("--task", help="Either sick or sts.", type=str, default='sts')
parser.add_argument("--nntype", help="Type of neural network.")
parser.add_argument("--updatewords", help="Whether to update the word embeddings", type=bool, default=True)
args = parser.parse_args()

# get train data
task = args.task
if task == 'sts':
    train_file = '../data/stsbenchmark_tok/sts-train.csv'
    dev_file  = '../data/stsbenchmark_tok/sts-dev.csv'
    test_file = '../data/stsbenchmark_tok/sts-test.csv'

    train = utils.load_STS(train_file)
    dev = utils.load_STS(dev_file)
    test = utils.load_STS(test_file)
elif task == 'sick':
    train_file = '../data/SICK/SICK_train.txt'
    dev_file  = '../data/SICK/SICK_trial.txt'
    test_file = '../data/SICK/SICK_test_annotated.txt'

    train = utils.load_SICK(train_file)
    dev = utils.load_SICK(dev_file)
    test = utils.load_SICK(test_file)


# get emb data
data = train + dev + test
sentences = []
for x in data:
    sentences.append(x[0])
    sentences.append(x[1])

idf_weight = utils.idf_calculator(sentences)
w2i = {w:i for i,w in enumerate(idf_weight.keys())}
emb = utils.load_word_embedding(w2i, utils.my_emb_rep['paragram300'], 300)

# params
params = utils.Params()
params.memsize = 50
params.minval = 0
params.maxval = 5
params.nout = params.maxval - params.minval + 1
params.LW = 1e-03
params.LC = 1e-05
params.learner = lasagne.updates.adam
params.batchsize = 50
params.dim = 300
params.eta = 0.01
params.clip = None
# params.hid_size = 300

print params.LW


params.updatewords = args.updatewords

""" PROJ Model """
params.layersize = 300
# params.nonlinearity = lasagne.nonlinearities.linear
params.nonlinearity = lasagne.nonlinearities.tanh
# params.nonlinearity = lasagne.nonlinearities.rectify
# params.nonlinearity = lasagne.nonlinearities.sigmoid

""" LSTM Model """
params.numlayers = 1


if args.nntype == 'word':
    model = WordModel(emb, params)
elif args.nntype == 'wordl1':
    model = WordL1Model(emb, params)
elif args.nntype == 'proj':
    model = ProjModel(emb, params)
elif args.nntype == 'projl1':
    model = ProjL1Model(emb, params)
elif args.nntype == 'lstm':
    model = LSTMModel(emb, params)
elif args.nntype == 'lstml1':
    model = LSTML1Model(emb, params)


def transform(train, w2i):
    train_num = []
    for sa, sb, score in train:
        sa = [w2i[w] for w in sa]
        sb = [w2i[w] for w in sb]
        train_num.append((sa, sb, score))
    return train_num

train = transform(train, w2i)
dev = transform(dev, w2i)
test = transform(test, w2i)


def evaluate(model, dev, params):
    _, g1x, g1mask, g2x, g2mask = utils.get_prepare_data(dev, params.nout)
    golds = [score for sa, sb, score in dev]
    scores = model.scoring_function(g1x, g2x, g1mask, g2mask)
    preds = np.squeeze(scores)
    return pearsonr(preds, golds)[0], spearmanr(preds, golds)[0]

for epoch in range(300):
    process_bar = pyprind.ProgPercent(len(train))
    kf = utils.get_minibatches_idx(len(train), params.batchsize, shuffle=True)
    uidx = 0
    for _, train_index in kf:
        uidx += 1
        batch = [ train[t] for t in train_index]

        scores, g1x, g1mask, g2x, g2mask = utils.get_prepare_data(batch, params.nout)

        # print scores[:2], g1x[:2], g1mask[:2], g2x[:2], g2mask[:2]
        cost = model.train_function(scores, g1x, g2x, g1mask, g2mask)

        if np.isnan(cost) or np.isinf(cost):
            print 'NaN detected'

    print 'Epoch ', (epoch + 1), 'Update ', (uidx + 1), 'Cost ', cost

    # if ITER % 1000 == 0:
    dp, ds = evaluate(model, dev, params)
    tp, ts = evaluate(model, test, params)
    rp, rs = evaluate(model, train, params)
    print "evaluation: ", dp, ds, tp, ts, rp, rs

