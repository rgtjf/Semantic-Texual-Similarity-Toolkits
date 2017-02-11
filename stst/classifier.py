# coding:utf-8
from __future__ import print_function

import pickle
from numpy import shape
from sklearn import preprocessing
from sklearn.datasets import load_svmlight_file
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from stst import utils

__all__ = [
    "Strategy",
    "Classifier",
    "RandomForestRegression",
    "GradientBoostingRegression",
    "AverageEnsemble"
]


class Strategy(object):
    def train_model(self, train_file_path, model_path):
        return None

    def test_model(self, test_file_path, model_path, result_file_path):
        return None

    def load_file(self, file_path):
        data = load_svmlight_file(file_path)
        return data[0], data[1]


class Classifier(object):
    def __init__(self, strategy):
        self.strategy = strategy

    def train_model(self, train_file_path, model_path):
        return self.strategy.train_model(train_file_path, model_path)

    def test_model(self, test_file_path, model_path, result_file_path):
        return self.strategy.test_model(test_file_path, model_path, result_file_path)


class RandomForestRegression(Strategy):
    """
    RandomForest Regression

    """
    def __init__(self, n_estimators=300):
        self.trainer = "RandomForest Regression"
        print("==> Using %s Classifier" % (self.trainer))
        self.n_estimators = n_estimators

    def train_model(self, train_file_path, model_path):
        print("==> Load the data ...")
        X_train, Y_train = self.load_file(train_file_path)
        print(train_file_path, shape(X_train))

        print("==> Train the model ...")
        min_max_scaler = preprocessing.MaxAbsScaler()
        X_train_minmax = min_max_scaler.fit_transform(X_train)
        clf = RandomForestRegressor(n_estimators=self.n_estimators)
        clf.fit(X_train_minmax.toarray(), Y_train)

        print("==> Save the model ...")
        pickle.dump(clf, open(model_path, 'wb'))

        scaler_path = model_path.replace('.pkl', '.scaler.pkl')
        pickle.dump(min_max_scaler, open(scaler_path, 'wb'))
        return clf

    def test_model(self, test_file_path, model_path, result_file_path):
        print("==> Load the data ...")
        X_test, Y_test = self.load_file(test_file_path)
        print(test_file_path, shape(X_test))

        print("==> Load the model ...")
        clf = pickle.load(open(model_path, 'rb'))
        scaler_path = model_path.replace('.pkl', '.scaler.pkl')
        min_max_scaler = pickle.load(open(scaler_path, 'rb'))

        print("==> Test the model ...")
        X_test_minmax = min_max_scaler.transform(X_test)
        y_pred = clf.predict(X_test_minmax.toarray())

        print("==> Save the result ...")
        with utils.create_write_file(result_file_path) as f:
            for y in y_pred:
                print(y, file=f)
        return y_pred


class GradientBoostingRegression(Strategy):
    """
    Gradient Boosting Regression
    """
    def __init__(self, n_estimators=140):
        self.trainer = "GradientBoostingRegression"
        print("Using %s Classifier" % (self.trainer))
        self.n_estimators = n_estimators

    def train_model(self, train_file_path, model_path):
        print("==> Load the data ...")
        X_train, Y_train = self.load_file(train_file_path)
        print(train_file_path, shape(X_train))

        print("==> Train the model ...")
        min_max_scaler = preprocessing.MaxAbsScaler()
        X_train_minmax = min_max_scaler.fit_transform(X_train)

        clf = GradientBoostingRegressor(n_estimators=self.n_estimators)
        clf.fit(X_train_minmax.toarray(), Y_train)

        print("==> Save the model ...")
        pickle.dump(clf, open(model_path, 'wb'))

        scaler_path = model_path.replace('.pkl', '.scaler.pkl')
        pickle.dump(min_max_scaler, open(scaler_path, 'wb'))
        return clf

    def test_model(self, test_file_path, model_path, result_file_path):
        print("==> Load the data ...")
        X_test, Y_test = self.load_file(test_file_path)
        print(test_file_path, shape(X_test))

        print("==> Load the model ...")
        clf = pickle.load(open(model_path, 'rb'))
        scaler_path = model_path.replace('.pkl', '.scaler.pkl')
        min_max_scaler = pickle.load(open(scaler_path, 'rb'))

        print("==> Test the model ...")
        X_test_minmax = min_max_scaler.transform(X_test)
        y_pred = clf.predict(X_test_minmax.toarray())

        print("==> Save the result ...")
        with utils.create_write_file(result_file_path) as f:
            for y in y_pred:
                print(y, file=f)
        return y_pred


class AverageEnsemble(Strategy):
    def __init__(self):
        self.trainer = "Average Ensemble"
        print("Using %s Classifier" % (self.trainer))


    def train_model(self, train_file_path, model_path):
        pass

    def test_model(self, test_file_path, model_path, result_file_path):
        print("==> Load the data ...")
        X_test, Y_test = self.load_file(test_file_path)
        print(test_file_path, shape(X_test))
        X_test = X_test.toarray()
        for x in X_test[:10]:
            print(x)

        print("==> Test the model ...")
        y_pred = []
        for x in X_test:
            x = sum(x) / len(x)
            y_pred.append(x)

        print("==> Save the result ...")
        with utils.create_write_file(result_file_path) as f:
            for y in y_pred:
                print(y, file=f)
        return y_pred
