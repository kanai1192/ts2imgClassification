from os import path

from setuptools import setup


def read(fname):
    with open(path.join(path.dirname(__file__), fname)) as f:
        return f.read()

core_requirements = [
    'numpy',
    'scipy',
    'pandas',
    'matplotlib',
    'seaborn',
    'pyts'
]

ml_requirements = ['scikit-learn', 'imbalanced-learn', 'joblib', 'pyts']
jupyter_requirements = ['jupyterlab']
dev_requirements = ['pylint', 'autopep8']
cluster_requirements = ['ipyparallel']


setup(
    name='ts2imageClassification',
    version='0.1',
    url='https://github.com/sensorlab/ts2imgClassification',
    author='SensorLab',
    author_email='sensorlab@ijs.si',
    description='Time series to image transfomration for wireless anomaly detection',
    long_description=read('README.md'),
    license='GPL3+',
    packages=['classification'],
    include_package_data=True,
    python_requires='>=3.6',
    install_requires=core_requirements,
    extras_require={
        'dev': dev_requirements,
        'ml': ml_requirements,
        'lab': jupyter_requirements,
        'cluster': cluster_requirements,
        'all': dev_requirements + ml_requirements + jupyter_requirements + cluster_requirements
    },
)