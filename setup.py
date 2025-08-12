from setuptools import setup, find_packages

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()


setup(
    name='python-competitive',
    version='0.2',
    packages=find_packages(),
    install_requires=[],
    description='A python module for competitive programmers',
    long_description=long_description,
    long_description_content_type="text/markdown",
)
