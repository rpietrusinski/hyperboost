import seaborn as sns
import pandas as pd
from hyperopt import space_eval, Trials
from matplotlib import pyplot as plt


class HyperparametersVis(object):
    def __init__(self, trials: Trials, space: dict):
        """Object of class HyperparametersVis generates plots describing performance of XGBoost optimization based on
            provided trials and space variables.

            :param trials: Trials object to use in analysis
            :param space: Hyperparameters space dictionary
            """
        self.trials: Trials = trials
        self.space: dict = space
        self.extracted_params: list = [x['misc']['vals'] for x in self.trials.trials]
        self.extracted_params_df: pd.DataFrame = None
        self.eval_values: pd.Series = None
        self.result: pd.DataFrame = None

        for param_list in self.extracted_params:
            for param in param_list.keys():
                try:
                    param_list[param] = param_list[param][0]
                except TypeError:
                    param_list[param] = param_list[param]

        self.extracted_params = [space_eval(self.space, z) for z in self.extracted_params]
        self.eval_values = pd.Series([x['result']['loss'] for x in self.trials.trials], name='eval')
        self.extracted_params_df = pd.DataFrame.from_dict(self.extracted_params)
        self.result = pd.concat([self.eval_values, self.extracted_params_df], axis=1)

    def numeric_plot(self, param: str):
        """Function generates performance plots for XGBoost's numeric parameters (eg. 'gamma, 'eta, etc.)

        :param param: name of a parameter to generate a plot
        :return: Plot
        """
        ax1 = sns.kdeplot(self.result[param], label='KDE')
        ax1.legend(loc='upper left')
        ax1.set_title(param, size=15)
        ax1.set_xlabel("Param values")
        ax1.set_ylabel("kde density")
        ax2 = ax1.twinx()
        ax2.scatter(self.result[param], self.result['eval'], c='red', alpha=.3, label='Evals')
        ax2.legend(loc='upper right')
        ax2.set_ylabel('loss')
        plt.show()

    def object_plot(self, param: str):
        """Function generates performance plots for XGBoost's object parameters (eg. 'booster, etc.)

        :param param: name of a parameter to generate a plot
        :return: Plot
        """
        ax1 = sns.countplot(x=self.result[param], color='blue')
        ax1.set_title(param, size=15)
        plt.show()

    def performance_plot(self):
        """Function generates performance plot with cross-validated AUC through iterations.

        :return: Plot
        """
        n = self.result.shape[0]
        plt.plot(range(1, n + 1), self.result['eval'])
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Performance plot', size=15)
        plt.show()

    def make_plots(self):
        """Function generates plots for all parameters under analysis in form of matplotlib's subplots.

        :return: Subplots
        """
        types = self.result.dtypes.to_dict()
        plt.figure(figsize=(15, 15))
        plt.suptitle('Parameters', size=20)
        plt.subplots_adjust(wspace=0.4, hspace=0.6)
        i = 1
        for key in types:
            plt.subplot(4, 3, i)
            if key == 'eval':
                self.performance_plot()
            elif types[key] == object:
                self.object_plot(key)
            else:
                self.numeric_plot(key)
            i += 1
