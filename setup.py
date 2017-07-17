import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="pytrack_analysis",
    version="0.0.1",
    author="Dennis Goldschmidt",
    author_email="dennis.goldschmidt@neuro.fchampalimaud.org",
    description=("Toolbox for analyzing object centroid data with Python."),
    license="GPLv3",
    keywords=['tracking', 'data analysis', 'fly'],
    # url="http://packages.python.org/pycircstat",
    packages=['pytrack_analysis', 'tests'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: Alpha",
        "Topic :: Utilities",
        # "License :: OSI Approved :: MIT License",
    ],
    platforms=['Windows 10 Pro', 'MacOS X El Capitan'],
    setup_requires=['numpy', 'pyyaml', 'pandas', 'scipy'],
)
