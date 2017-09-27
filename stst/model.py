# coding: utf8
from __future__ import print_function

import os
import random

from stst import utils, config
from stst import Feature

class Model(object):
    def __init__(self,
                 model_name,
                 classifier):
        self.model_name = model_name

        self.feature_name = model_name

        self.classifier = classifier

        self.feature_list = []

        self.train_feature_file = '{}/{}.feature.train.txt'.format(
            config.MODEL_DIR, self.model_name)
        self.dev_feature_file = '{}/{}.feature.dev.txt'.format(
            config.MODEL_DIR, self.model_name)

        self.model_file = '{}/{}.pkl'.format(config.MODEL_DIR, self.model_name)

        # config.OUTPUT_DIR + '/' + self.model_name + '.output.txt'
        self.output_file = None

        self.get_output_file = lambda train_file: '{}/{}/{}'.format(
            config.OUTPUT_DIR, self.model_name, os.path.basename(train_file))


    def add(self, feature):
        self.feature_list.append(feature)

    def train(self, train_instances, train_file, out_list=None):
        """
        out_list is used to sub train_instances from all train_instances
        if only happen when all faetures have been made.
        """

        ''' 1. Extract Features '''
        self.make_feature_file(train_instances, train_file)

        if out_list:
            dev = utils.create_read_file(self.train_feature_file).readlines()
            dev = [dev[idx].strip() for idx in range(len(dev)) if idx not in out_list]
            f_dev = utils.create_write_file(self.train_feature_file)
            print('\n'.join(dev), file=f_dev)
            f_dev.close()
            print('finish filter, train examples %d', len(dev))

        ''' 2. Train Classifier '''
        self.classifier.train_model(self.train_feature_file, self.model_file)

        ''' 3. Predict Answers '''
        self.output_file = self.get_output_file(train_file)
        predict_label = self.classifier.test_model(self.train_feature_file, self.model_file, self.output_file)

        f_out = utils.create_write_file(self.output_file)
        for label, train_instances in zip(predict_label, train_instances):
            print('{:.2f}\t#\t{}'.format(label, train_instances.get_instance_string()), file=f_out)

        return self.classifier

    def test(self, dev_instances, dev_file):
        """
        """
        ''' 1. Extract Features '''
        self.make_feature_file(dev_instances, dev_file, dev=True)

        self.output_file = self.get_output_file(dev_file)
        print(self.output_file)

        ''' 2. Predict Answers '''
        predict_label = self.classifier.test_model(self.dev_feature_file, self.model_file, self.output_file)

        f_out = utils.create_write_file(self.output_file)
        for label, dev_instance in zip(predict_label, dev_instances):
            print('{:.2f}\t#\t{}'.format(label, dev_instance.get_instance_string()), file=f_out)

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
                feature_string, feature_dimension, n_instance = \
                    feature_class.extract_dataset_instances(train_instances, train_file, not dev)
                feature_strings.append(feature_string)
                feature_dimensions.append(feature_dimension)
                sum_feature_dimension += feature_dimension
                print(feature_class.feature_name, feature_dimension, sum_feature_dimension)

            elif isinstance(feature_class, Model):
                if dev:
                    feature_class.test(train_instances, train_file)
                    feature_string = feature_class.load_model_score(train_file)
                else:
                    ''' seperate to train for speed up '''
                    # feature_class.train(train_instances, train_file)
                    feature_string = feature_class.load_model_score(train_file)
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
        if dev:
            f_feature = utils.create_write_file(self.dev_feature_file)
        else:
            f_feature = utils.create_write_file(self.train_feature_file)

        for idx, feature_string in enumerate(merged_feature_string_list):
            train_instance = train_instances[idx]
            print(str(train_instance.get_score()), feature_string, file=f_feature)

        return merged_feature_string_list, merged_feature_dimension, len(merged_feature_string_list)

    def load_model_score(self, train_file):
        self.output_file = self.get_output_file(train_file)

        y_pred = utils.create_read_file(self.output_file).readlines()
        y_pred = ['1:' + x.strip().split("\t#\t")[0] for x in y_pred]
        return y_pred

    def cross_validation(self, data_instances, data_file, k_fold=5, shuffle=False):

        self.make_feature_file(data_instances, data_file)

        n_data = len(data_instances)
        n_batch = n_data // k_fold
        data_instances = list(zip(range(n_data), data_instances))

        print(n_data)

        id_map = range(n_data)
        if shuffle is True:
            random.shuffle(id_map)

        preds = [None] * n_data
        for fold in range(k_fold):
            st = fold * n_batch
            ed = (fold + 1) * n_batch
            if ed > n_data:
                ed = n_data

            data = utils.create_read_file(self.train_feature_file).readlines()
            print(len(data))
            # make train data
            train = [data[id_map[idx]].strip() for idx in range(len(data)) if idx not in range(st, ed)]
            data_feature_file_train = self.train_feature_file.replace('txt', 'train')
            f_train = utils.create_write_file(data_feature_file_train)
            print('\n'.join(train), file=f_train)
            f_train.close()

            # make dev data
            dev = [data[id_map[idx]].strip() for idx in range(st, ed)]
            data_feature_file_dev = self.train_feature_file.replace('txt', 'dev')
            f_dev = utils.create_write_file(data_feature_file_dev)
            print('\n'.join(dev), file=f_dev)
            f_dev.close()

            ''' Train Classifier '''
            # Attention! self.dev_feature_file
            self.classifier.train_model(data_feature_file_train, self.model_file)

            ''' Predict Lables'''
            self.output_file = self.get_output_file(data_file)

            predict_label = self.classifier.test_model(data_feature_file_dev, self.model_file, self.output_file)

            for idx in range(st, ed):
                idy = idx - st
                preds[id_map[idx]] = predict_label[idy]

        ''' Write to File '''
        self.output_file = self.get_output_file(data_file)

        f_out = utils.create_write_file(self.output_file)
        for label, train_instance in zip(preds, data_instances):
            print('{:.2f}\t#\t{}'.format(
                label, train_instance[1].get_instance_string()), file=f_out)
