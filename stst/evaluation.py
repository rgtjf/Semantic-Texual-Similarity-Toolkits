import scipy.stats as meas


def evaluation(predict, gold):
    """
    pearsonr of predict and gold
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


def eval_output_file(predict_file):
    predict, gold = [], []
    with open(predict_file) as f:
        for line in f:
            line = line.strip().split('\t#\t')
            predict.append(float(line[0]))
            gold.append(float(line[1].split('\t')[0]))
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
