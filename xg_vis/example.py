import numpy as np
import xgboost as xgb
from hyperopt import hp, tpe, fmin, Trials
from sklearn.datasets import load_iris
from sklearn.metrics import roc_auc_score

from xg_vis.vis import HyperparametersVis

# load data
data = load_iris()
X = data['data']
y = data['target']
y = np.array([round(np.random.rand()) for y in y])

# run xgb
xgb_data = xgb.DMatrix(X, y)

#hyperopt space
space = {'eta': hp.choice('eta', np.arange(0.001, 0.3, 0.001)),
          'gamma': hp.choice('gamma', np.arange(0, 100, 1)),
          'max_depth': hp.choice('max_depth', np.arange(1, 9, 1)),
          'objective': hp.choice('objective', ['binary:logistic']),
          'eval_metric': hp.choice('eval_metric',['auc']),
          'booster': hp.choice('booster', ['gbtree','gblinear','dart'])
          }
trials = Trials()

#optimizer
def opt(params):
    model = xgb.train(params=params, dtrain=xgb_data, num_boost_round=20)  
    preds = model.predict(xgb_data)
    auc = roc_auc_score(xgb_data.get_label(), preds)
    return 1-auc

#optimize
best = fmin(fn=opt, space=space, algo=tpe.suggest, max_evals=50, trials=trials)

#generate plots
HP = HyperparametersVis(trials, space)
HP.object_plots('booster')
HP.numeric_plots('gamma')


