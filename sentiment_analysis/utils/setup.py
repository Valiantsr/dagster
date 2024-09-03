# setup.py

from setuptools import setup, find_packages

setup(
    name='Sentiment',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'transformers',
        'nltk',
        'tqdm',
        'pandas',
        'numpy'
    ],
    include_package_data=True,
    description='A sentiment analysis package',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://your.url.here'
)
