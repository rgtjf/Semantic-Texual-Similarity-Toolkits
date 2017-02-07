# coding: utf8
from __future__ import print_function

import config, utils
from features import Feature


class Model(object):
    def __init__(self,
                 model_name,
                 classifier):
        self.model_name = model_name

        self.feature_name = model_name

        self.classifier = classifier
        self.feature_list = []

        self.train_feature_file = config.FEARURE_DIR + '/' + self.model_name + '.train.txt'
        self.dev_feature_file = config.FEARURE_DIR + '/' + self.model_name + '.dev.txt'

        self.output_file = None  # config.OUTPUT_DIR + '/' + self.model_name + '.output.txt'

        self.model_file = config.MODEL_DIR + '/' + self.model_name + '.txt'

    def add(self, feature):
        self.feature_list.append(feature)

    def train(self, train_instances, train_file, out_list=None):
        ''' 1. Extract Features '''
        self.make_feature_file(train_instances, train_file)

        ''' select dataset '''
        # STS2012 - train  # STS.input.MSRpar.txt	4568	5318
        # STS2012 - train  # STS.input.MSRvid.txt	3818	4568
        # STS2012 - train  # STS.input.SMTeuroparl.txt	750	1484
        # if 'ar' in self.model_name:
        #     out = range(750, 1484) + range(3818, 4568) + range(4568, 5318)
        #     dev = utils.create_read_file(self.dev_feature_file).readlines()
        #     dev = [dev[idx].strip() for idx in range(len(dev)) if idx not in out]
        #     f_dev = utils.create_write_file(self.dev_feature_file)
        #     print('\n'.join(dev), file=f_dev)
        #     f_dev.close()
        #     print('finish ar filter, train examples %d', len(dev))

        if out_list:
            dev = utils.create_read_file(self.dev_feature_file).readlines()
            dev = [dev[idx].strip() for idx in range(len(dev)) if idx not in out_list]
            f_dev = utils.create_write_file(self.dev_feature_file)
            print('\n'.join(dev), file=f_dev)
            f_dev.close()
            print('finish ar filter, train examples %d', len(dev))

        ''' 2. Train Classifier '''
        self.classifier.train_model(self.dev_feature_file, self.model_file)  # Attention! self.dev_feature_file

        self.output_file = train_file.replace('resources/data/',
                                              'outputs/'+self.model_name+'/')  # alert! self.output_file is not show right before test

        ''' 3. Predict Answers '''
        predict_label = self.classifier.test_model(self.dev_feature_file, self.model_file, self.output_file)

        # output_file = self.output_file.replace('output.txt', 'out.txt')
        f_out = utils.create_write_file(self.output_file)
        for label, train_instances in zip(predict_label, train_instances):
            print('%.2f\t#\t%s' % (label, train_instances.get_instance_string()), file=f_out)

        return self.classifier

    def test(self, dev_instances, dev_file):
        pass
        ''' 1. Extract Features '''
        self.make_feature_file(dev_instances, dev_file, dev=True)

        self.output_file = dev_file.replace('resources/data/',
                                            'outputs/'+self.model_name+'/')  # alert! self.output_file is not show right before test
        print(self.output_file)
        ''' 2. Predict Answers '''
        predict_label = self.classifier.test_model(self.dev_feature_file, self.model_file, self.output_file)

        # output_file = self.output_file.replace('output.txt', 'out.txt')
        f_out = utils.create_write_file(self.output_file)
        for label, dev_instance in zip(predict_label, dev_instances):
            print('%.2f\t#\t%s' % (label, dev_instance.get_instance_string()), file=f_out)

        return predict_label

    def make_feature_file(self, train_instances, train_file, dev=False):
        """
        TODO. similar to feature, write to file
        """

        print("-" * 120)
        print("\n".join([f.feature_name for f in self.feature_list]))
        print("-" * 120)

        ''' Extract Features '''
        feature_strings = []
        feature_dimensions = []
        sum_feature_dimension = 0
        for feature_class in self.feature_list:
            if isinstance(feature_class, Feature):
                feature_string, feature_dimension, n_instance = feature_class.extract_dataset_instances(train_instances,
                                                                                                        train_file)
                feature_strings.append(feature_string)
                feature_dimensions.append(feature_dimension)

                # print(feature_dimension)
                sum_feature_dimension += feature_dimension
                print(feature_class.feature_name, feature_dimension, sum_feature_dimension)

            elif isinstance(feature_class, Model):

                if dev:
                    feature_class.test(train_instances, train_file)
                    feature_string = feature_class.load_features(train_file)
                else:
                    ''' seperate to train for speed up '''
                    # feature_class.train(train_instances, train_file)
                    feature_string = feature_class.load_features(train_file)
                feature_strings.append(feature_string)
                feature_dimensions.append(1)

                sum_feature_dimension += 1
                print(feature_class.feature_name, 1, sum_feature_dimension)

        ''' Merge Features'''

        merged_feature_string_list = []
        for feature_strings in zip(*feature_strings):
            merged_feature_string = ""
            dimension = 0
            for feature_dimension, feature_string in zip(feature_dimensions, feature_strings):
                if dimension == 0:  # 第一个
                    merged_feature_string = feature_string
                else:
                    if feature_string != "":
                        # 修改当前feature的index
                        temp = ""
                        for item in feature_string.split(" "):
                            if len(item.split(":")) == 1:
                                print(item)
                            index, value = item.split(":")
                            temp += " %d:%s" % (int(index) + dimension, value)
                        merged_feature_string += temp
                dimension += feature_dimension

            merged_feature_string_list.append(merged_feature_string)

        merged_feature_dimension = sum(feature_dimensions)

        ''' Write to feature file '''
        f_feature = utils.create_write_file(self.dev_feature_file)  # Attention! self.train_feature_file
        for idx, feature_string in enumerate(merged_feature_string_list):
            train_instance = train_instances[idx]
            print(str(train_instance.get_score()), feature_string, file=f_feature)

        return merged_feature_string_list, merged_feature_dimension, len(merged_feature_string_list)

    def load_features(self, train_file):
        self.output_file = train_file.replace('resources/data/',
                                              'outputs/'+self.model_name+'/')  # alert! self.output_file is not show right before test
        print(self.output_file)
        y_pred = open(self.output_file, 'r').readlines()
        y_pred = ['1:' + x.strip().split("\t#\t")[0] for x in y_pred]
        return y_pred



    def cross_validation(self, data_instances, data_file, k_fold=5, shuffle=False):

        from evaluation import evaluation

        self.make_feature_file(data_instances, data_file)

        n_data = len(data_instances)
        n_batch = n_data // k_fold
        data_instances = list(zip(range(n_data), data_instances))

        id_map = range(n_data)
        import random
        # if shuffle is True:
        random.shuffle(id_map)

        preds = []
        golds = []
        for fold in range(k_fold):
            st = fold * n_batch
            ed = (fold + 1) * n_batch
            if ed > n_data:
                ed = n_data

            print(st, ed)

            data = utils.create_read_file(self.dev_feature_file).readlines()

            # make  train data
            train = [data[id_map[idx]].strip() for idx in range(len(data)) if idx not in range(st, ed)]
            dev_feature_file_train = self.dev_feature_file.replace('txt', 'train')
            f_train = utils.create_write_file(dev_feature_file_train)
            print('\n'.join(train), file=f_train)
            f_train.close()

            # make dev data
            dev = [data[id_map[idx]].strip() for idx in range(st, ed)]
            dev_feature_file_dev = self.dev_feature_file.replace('txt', 'cv.dev')
            f_dev = utils.create_write_file(dev_feature_file_dev)
            print('\n'.join(dev), file=f_dev)
            f_dev.close()

            ''' Train Classifier '''
            self.classifier.train_model(dev_feature_file_train, self.model_file)  # Attention! self.dev_feature_file

            ''' Predict Lables'''
            self.output_file = data_file.replace('resources/data/',
                                                 'outputs/' + self.model_name + '/')  # alert! self.output_file is not show right before test

            predict_label = self.classifier.test_model(dev_feature_file_dev, self.model_file, self.output_file)

            pred = predict_label
            gold = [data_instances[id_map[x]][1].get_score() for x in range(st, ed)]
            # eval this fold
            print("%d_fold: %.2f%%" % (fold, 100*evaluation(pred, gold)))

            preds += list(pred)
            golds += gold

        # self.classifier.train_model(self.dev_feature_file, self.model_file)  # Attention! self.dev_feature_file
        print("Summary: %.2f%%" % (100*evaluation(preds, golds)))

        ''' Write to File '''
        self.output_file = data_file.replace('resources/data/',
                                             'outputs/' + self.model_name + '/')  # alert! self.output_file is not show right before test

        f_out = utils.create_write_file(self.output_file)
        for label, train_instance in zip(preds, data_instances):
            print('%.2f\t#\t%s' % (label, train_instance[1].get_instance_string()), file=f_out)