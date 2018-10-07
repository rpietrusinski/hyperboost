from distutils.core import setup

setup(
    name='hyperboost',
    version='0.1',
    description='Package aims to improve optimization of XGBoost hyperparameters. It uses Bayesian Optimization'
                'approach, in particular the loss function approximation with Tree Parzen Estimators from Hyperopt.',
    author='Robert Pietrusinski',
    author_email='pietrusinski.robert@gmail.com',
    packages=['hyperboost'],
    license='Open-source',
    long_description=open('README.md').read(),
    install_requires=['matplotlib', 'hyperopt', 'seaborn', 'pandas', 'scikit-learn', 'xgboost', 'numpy']
)
