# hugging_bench/setup.py

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='hugging_bench',
    version='1.0',
    packages=find_packages(),
    # install_requires=[
    #     'client',  # Specify the client package as a dependency
    # ],
    install_requires=requirements,
)
