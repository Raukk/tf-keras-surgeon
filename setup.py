#!/usr/bin/env python

from setuptools import setup
from setuptools import find_packages


setup(
    name='tfkerassurgeon',
    version="0.2.01",
    url='https://github.com/Raukk/tf-keras-surgeon',
    license='MIT',
    description='A library for performing network surgery on trained tf.Keras '
                'models. Useful for deep neural network pruning.',
    long_description = 'A library for performing network surgery on trained tf.Keras models for network pruning.',
    maintainer='Raukk',
    maintainer_email='raukk@raukk.me',
    python_requires='>=3',
    extras_require={'examples': ['pandas'], },
    tests_require=['pytest'],
    packages=find_packages('src'),
    package_dir={'': 'src'}
)

