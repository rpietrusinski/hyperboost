from distutils.core import setup

setup(
    name='hyperboost',
    version='0.1',
    packages=['hyperboost'],
    license='Open-source',
    long_description=open('README.md').read(),
    install_requires=['matplotlib', 'hyperopt', 'seaborn', 'pandas', 'scikit-learn', 'xgboost', 'numpy']
)
