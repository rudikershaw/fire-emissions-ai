# -*- coding: utf-8 -*-
import io
import os
import sys
from shutil import rmtree
from pipenv.project import Project
from pipenv.utils import convert_deps_to_pip
from setuptools import find_packages, setup, Command

# Package meta-data.
NAME = 'fireemissionsai'
DESCRIPTION = 'Fire Emissions AI is designed to GFED emissions data.'
URL = 'https://github.com/rudikershaw/fire-emissions-ai'
EMAIL = 'alexander_kershaw@hotmail.co.uk'
AUTHOR = 'Rudi Kershaw'

# Retrieve required packages from Pipenv.
pfile = Project(chdir=False).parsed_pipfile
REQUIRED = convert_deps_to_pip(pfile['packages'], r=False)

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = '\n' + f.read()

# Where the magic happens:
setup(
    name=NAME,
    version="0.0.1",
    description=DESCRIPTION,
    long_description=long_description,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=find_packages(exclude=('tests',)),
    install_requires=REQUIRED,
    include_package_data=True,
    license='APACHE 2.0',
    test_suite='fireemissionsai.tests',
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ]
)
