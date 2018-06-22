#!/usr/bin/env python

from setuptools import setup

setup(name='aardbei',
      version='0.0.1',
      description='LR scripts voor de zaak AARDBEI',
      packages=['liar'],
      setup_requires=['nose'],
      test_suite='nose.collector'
     )
