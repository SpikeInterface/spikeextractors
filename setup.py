import setuptools

pkg_name="spikeextractors"

setuptools.setup(
    name=pkg_name,
    version="0.4.4",
    author="Cole Hurwitz, Jeremy Magland, Alessio Paolo Buccino, Matthias Hennig",
    author_email="colehurwitz@gmail.com",
    description="Python module for extracting recorded and spike sorted extracellular data from different file types and formats",
    url="https://github.com/SpikeInterface/spikeextractors",
    packages=setuptools.find_packages(),
    package_data={},
    install_requires=[
        'numpy',
    ],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    )
)
