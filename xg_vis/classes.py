from hyperopt import hp, tpe, fmin, space_eval, Trials
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd


class HyperparametersVis(object):
    def __init__(self, trials: list,
                       space: dict):
        self.trials = trials
        self.space = space
        self.extracted_params = [x['misc']['vals'] for x in self.trials.trials]

        for param_list in self.extracted_params:
            for param in param_list.keys():
                try:
                    param_list[param] = param_list[param][0]    
                except:
                    param_list[param] = param_list[param]

        self.extracted_params = [space_eval(self.space, z) for z in self.extracted_params]

        self.eval_values = pd.Series([x['result']['loss'] for x in self.trials.trials], name = 'eval')
        self.extracted_params = pd.DataFrame.from_dict(self.extracted_params)
        self.result = pd.concat([self.eval_values, self.extracted_params], axis=1)
    def numeric_plots(self, param: str):
        ax1 = sns.kdeplot(self.result[param], label = 'KDE')
        ax1.legend(loc='upper left')
        ax1.set_title(param, size = 15)
        ax1.set_xlabel("Param values")
        ax1.set_ylabel("kde density")
        ax2 = ax1.twinx()
        ax2.scatter(self.result[param], self.result['eval'], c = 'red', alpha = .3, label = 'Evals')
        ax2.legend(loc='upper right')
        ax2.set_ylabel('loss')
        plt.show()
    def object_plots(self, param: str):
        ax1 = sns.countplot(x=self.result[param], color='blue')
        ax1.set_title(param, size=15)
        plt.show()



















