import numpy as np
from hyperopt import hp
from sklearn.datasets import load_iris

from hyperboost.vis import HyperparametersVis
from hyperboost.opt import HyperparametersOptimizer

# load data
data = load_iris()
X = data['data']
y = data['target']
y = np.array([int(y==2) for y in y])

# parameter space
space = {'eta': hp.choice('eta', np.arange(0.001, 0.3, 0.001)),
          'gamma': hp.choice('gamma', np.arange(0, 100, 1)),
          'max_depth': hp.choice('max_depth', np.arange(1, 9, 1)),
          'objective': hp.choice('objective', ['binary:logistic']),
          'eval_metric': hp.choice('eval_metric',['auc']),
          'booster': hp.choice('booster', ['gbtree','gblinear','dart'])
          }

optimizer = HyperparametersOptimizer(X, y, 'C:\\Users\\Robert\\data_science\\test')
optimizer.run_experiment(space, 10)

best = optimizer.opt_params(space)

# visualize search
vis = HyperparametersVis(optimizer.trials, space)
vis.numeric_plots('eta')
vis.numeric_plots('gamma')
vis.object_plots('booster')