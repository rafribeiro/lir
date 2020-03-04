#!/usr/bin/env python

import os
from setuptools import setup


package_dir = os.path.dirname(__file__)
requirements_file_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')

with open(requirements_file_path, 'r') as f:
    packages = [str(f) for f in f.readlines()]

setup(name='lir',
      version='0.0.1',
      description='scripts for calculating likelihood ratios',
      packages=[f'{package_dir}/lir'],
      setup_requires=['nose'],
      test_suite='nose.collector',
      install_requires=packages,
     )
