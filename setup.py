from distutils.core import setup

setup(
    name='xg_vis',
    version='0.1',
    packages=['xg_vis', 'sklearn', 'numpy', 'xgboost', 'hyperopt', 'matplotlib', 'seaborn', 'pandas'],
    license='Open-source',
    long_description=open('README.md').read(), requires=['matplotlib', 'hyperopt', 'seaborn', 'pandas', 'scikit-learn',
                                                         'xgboost', 'numpy']
)
