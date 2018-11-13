import setuptools

pkg_name="spikeinterface"

setuptools.setup(
    name=pkg_name,
    version="0.1.23",
    author="Cole Hurwitz, Jeremy Magland, Alessio Paolo Buccino, Matthias Hennig",
    author_email="colehurwitz@gmail.com",
    description="Python interface for extracting recorded and spike sorted extracellular data from different file types and formats",
    url="https://github.com/colehurwitz31/spikeinterface",
    packages=setuptools.find_packages(),
    package_data={},
    install_requires=[
        'numpy',
        'quantities',
        'neo',
        'pyyaml',
        'h5py',
    ],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    )
)
