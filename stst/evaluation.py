import scipy.stats as meas


def evaluation(predict, gold):
    """

    :param predict: list
    :param gold: list
    :return: mape
    """
    pearsonr = meas.pearsonr(predict, gold)[0]
    return pearsonr


def eval_file(predict_file, gold_feature_file):
    predict = open(predict_file).readlines()
    gold = open(gold_feature_file).readlines()
    predict = [float(x.strip().split()[0])for x in predict]
    gold = [float(x.strip().split()[0]) for x in gold]
    pearsonr = evaluation(predict, gold)
    return pearsonr


def eval_file_corpus(predict_file_list, gold_file_list):
    predicts, golds = [], []
    for predict_file, gold_file in zip(predict_file_list, gold_file_list):
        predict = open(predict_file).readlines()
        gold = open(gold_file).readlines()
        predicts += predict
        golds += gold
    predicts = [float(x.strip().split()[0]) for x in predicts]
    golds = [float(x.strip().split()[0]) for x in golds]
    pearsonr = evaluation(predicts, golds)
    return pearsonr



if __name__ == '__main__':
    pass
