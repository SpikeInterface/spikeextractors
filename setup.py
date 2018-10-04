import setuptools

pkg_name="spikeinterface"

import unittest
def my_test_suite():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    return test_suite

setuptools.setup(
    name=pkg_name,
    version="0.1.6",
    author="Cole Hurwitz, Jeremy Magland, Alessio Paolo Buccino, Matthias Hennig",
    author_email="colehurwitz@gmail.com",
    description="Python interface for extracting recorded and spike sorted extracellular data from different file types and formats",
    url="",
    packages=setuptools.find_packages(),
    package_data={},
    install_requires=[
        'numpy',
        'quantities',
        'mountainlab_pytools',
        'neo',
        'pyyaml',
        'h5py'
    ],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ),
    test_suite='setup.my_test_suite'
)
