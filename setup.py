
from setuptools import setup, find_packages

setup(
    name='FINKER',
    version='1.0.0',
    author='Fiorenzo Stoppa',
    author_email='f.stoppa@astro.ru.nl',
    description='A universal approach to optimal frequency identification using nonparametric kernel regression',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/FiorenSt/FINKER',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'scikit-learn',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
