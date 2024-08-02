import sys 
import os 
import torch 
from setuptools import find_packages, setup


sys.path.append(os.getcwd())
setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='A short description of the project.',
    author='Sagar Sinha',
    license='MIT',
)