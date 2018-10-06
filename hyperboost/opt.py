import numpy as np
import xgboost as xgb
import pickle
import os
from sklearn.metrics import roc_auc_score
from hyperopt import tpe, Trials, space_eval, fmin
from sklearn.model_selection import KFold


class HyperparametersOptimizer(object):
    def __init__(self, x: np.ndarray, y: np.ndarray, path: str):
        """Object of class HyperparametersOptimizer performs XGBoost's hyperparameters optimization using the Hyperopt's
        Tree Parzen Estimators. During each iteration the 5-fold cross-validation is performed and the algorithm
        optimizes test AUC.

        :param x: X data of type numpy ndarray
        :param y: y data of type numpy ndarray
        :param path: Experiment path which is where the Trials object is saved/loaded
        """
        self.x: np.ndarray = x
        self.y: np.ndarray = y
        self.trials: Trials = None
        self.best: dict = None
        self.path: str = os.path.join(path, 'trials.p')
        self.max_evals: int = None

    def cross_validation(self, params: dict):
        """Performs a single run of 5-fold cross-validation and returns the average auc.

        :param params: Dictionary of parameters
        :return: Average auc
        """
        auc = []
        for train_index, test_index in KFold(n_splits=5, shuffle=True).split(self.x):
            train = xgb.DMatrix(self.x[train_index], self.y[train_index])
            test = xgb.DMatrix(self.x[test_index], self.y[test_index])
            model = xgb.train(params, train, 30)

            preds = model.predict(test)
            auc.append(roc_auc_score(test.get_label(), preds))
        return 1 - np.mean(auc)

    def run_experiment(self, space: dict, evals: int):
        """Function either loads the Trials object and continues previous experiments or starts form the beginning.

        :param space: Parameters space
        :param evals: Number of evals in the experiment
        :return: Function saves the trials.p pickle object in the experiment path.
        """
        try:
            self.trials = pickle.load(open(self.path, "rb"))
        except FileNotFoundError:
            self.trials = Trials()

        self.max_evals = len(self.trials.trials) + evals
        self.best = fmin(fn=self.cross_validation, space=space, algo=tpe.suggest, max_evals=self.max_evals,
                         trials=self.trials)
        pickle.dump(self.trials, open(self.path, "wb"))

    def opt_params(self, space: dict):
        """Function returns best parameters set based on previous experiments.

        :param space: parameters space
        :return: dictionary of parameter values
        """
        if self.trials is None:
            print("No experiment has been conducted!")
        else:
            return space_eval(space, self.trials.argmin)
