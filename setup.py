#!/usr/bin/env python

from setuptools import setup

setup(name='liar',
      version='0.0.1',
      description='scripts for calculating likelihood ratios',
      packages=['liar'],
      setup_requires=['nose'],
      test_suite='nose.collector'
     )
