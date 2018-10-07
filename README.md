# Hyperboost - XGBoost optimization with Hyperopt

Package aims to improve optimization of XGBoost hyperparameters. It uses Bayesian Optimization approach, in particular
the loss function approximation with Tree Parzen Estimators from Hyperopt.   

##### Install:
`
pip install git+https://github.com/rpietrusinski/hyperboost
`

##### Example:
```python
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
space = {'eta': hp.uniform('eta', 0.001, 0.5),
          'gamma': hp.uniform('gamma', 0, 100),
          'max_depth': hp.choice('max_depth', np.arange(1, 9, 1)),
          'objective': hp.choice('objective', ['binary:logistic']),
          'eval_metric': hp.choice('eval_metric',['auc']),
          'booster': hp.choice('booster', ['gbtree','gblinear','dart'])
          }

# run experiment          
optimizer = HyperparametersOptimizer(X, y, '/home/xgb_model')
optimizer.run_experiment(space, 10)

# get best params
best = optimizer.opt_params(space)

# visualize search
vis = HyperparametersVis(optimizer.trials, space)
vis.make_plots()
```