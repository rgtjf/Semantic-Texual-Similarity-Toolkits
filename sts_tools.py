# coding: utf8
import stst

nlp = stst.StanfordNLP('http://localhost:9000')

def train_sts(model):
    train_file = './data/stsbenchmark/sts-train.csv'
    train_instances = stst.load_parse_data(train_file, nlp)
    model.train(train_instances, train_file)


def dev_sts(model):
    dev_file = './data/stsbenchmark/sts-dev.csv'
    dev_instances = stst.load_parse_data(dev_file, nlp)
    model.test(dev_instances, dev_file)
    dev_pearsonr = stst.eval_output_file(model.output_file)
    print('Dev:', dev_pearsonr)
    return dev_pearsonr

def test_sts(model):
    test_file = './data/stsbenchmark/sts-test.csv'
    test_instances = stst.load_parse_data(test_file, nlp)
    model.test(test_instances, test_file)
    test_pearsonr = stst.eval_output_file(model.output_file)
    print('Test:', test_pearsonr)
    return test_pearsonr


def hill_climbing(model, choose_list=[]):
    chooses = choose_list
    feature_list= model.feature_list
    visited = [True if x in choose_list else False for x in range(len(feature_list))]

    for idx in range(len(choose_list), len(feature_list)):
        chooseIndex = -1
        best_score = 0.0
        best_test_score = 0.0
        chooses.append(-1)
        for i in range(len(feature_list)):
            if visited[i] == False:
                chooses[idx] = i
                feature = [feature_list[s] for s in chooses]
                # print(len(feature_list))
                model.feature_list = feature
                train_sts(model)
                cur_score = dev_sts(model)
                test_score = test_sts(model)
                stst.record('./data/records.csv', cur_score, test_score, model)
                if best_score < cur_score:
                    chooseIndex = i
                    best_score = cur_score
                    best_test_score = test_score

        chooses[idx] = chooseIndex
        visited[chooseIndex] = True
        # feature = [ feature_list[s] for s in chooses]
        print('Best Score: %.2f %%,  %.2f%%,choose Feature %s' % (best_score * 100, best_test_score * 100,
                                                                  feature_list[chooseIndex].feature_name))

def feature_ablation(model):
    feature_list = model.feature_list
    train_sts(model)
    cur_score, info = test_sts(model)
    print('Cur Score: %.2f %% , All Features' % (cur_score * 100))
    for idx in range(len(feature_list)):
        feature = [feature_list[s] for s in filter(lambda x:x != idx, range(len(feature_list))) ]
        model.feature_list = feature
        train_sts(model)
        cur_score, info = test_sts(model)
        print('Cur Score: %.2f %%, except Feature %s' % (cur_score * 100, feature_list[idx].feature_name))


def feature_importance(model):
    feature_list = model.feature_list
    train_sts(model)
    dev_score = dev_sts(model)
    test_score = test_sts(model)
    print('Cur Score: Dev: %.2f %% , Test: %.2f %% ,  All Features' % (dev_score * 100, test_score * 100))
    for idx in range(len(feature_list)):
        feature = [feature_list[idx]]
        model.feature_list = feature
        train_sts(model)
        dev_score = dev_sts(model)
        test_score = test_sts(model)
        print('Cur Score: Dev: %.2f %% , Test: %.2f %% ,  Only Feature %s' % (dev_score * 100, test_score * 100, feature[0].feature_name))


if __name__ == '__main__':
    pass